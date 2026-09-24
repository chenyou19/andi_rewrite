"""Independent, read-only replay/audit for the v3 model-grid producer.

The original producer recorded its ``e20`` source-code digest but did not
retain the producer bytes.  This module therefore does not claim to recreate
that process.  It rebuilds the frozen candidate/pair selection in a fresh
revision root, compares only semantic manifest fields, and streams canonical
tensor hashes against the already materialized selected-cache hashes.  It
never calls the builder's selected-cache writer, never copies an NPZ, and
never starts a classifier fit.

The expensive ``run_replay`` entry point is intentionally separate from the
small pure comparison/hash helpers.  Runtime may freeze and launch it after a
source audit; importing this module does not inspect MRI data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.data.datasets.lmdb import LMDBSliceDataset  # noqa: E402
from andi_rewrite.data.domain_classifier.matching import (  # noqa: E402
    SPLITS,
    build_pairs,
)
from andi_rewrite.data.domain_classifier.records import (  # noqa: E402
    MODEL_SHAPE,
    SliceRecord,
    namespaced_participant,
    z_bin_for,
)
from scripts.build_model_grid_v3 import (  # noqa: E402
    ATLAS_DEPTH,
    ATLAS_SHAPE,
    BRATS_CSV,
    BRATS_ROOT,
    COHORT_ROOTS,
    PAIR_CAP_PER_BIN,
    SUPPORT_EPS,
    SUPPORT_THRESHOLD,
    Z_BINS,
    assign_map,
    canonical_source_rows,
    load_brats_candidate_rows,
    make_brats_records,
    make_healthy_records,
    materialize_healthy_support,
    read_old_fomo_registered_paths,
    read_old_split_maps,
    records_by_split,
    registered_paths_for_row,
    sha256,
    source_lmdb_path,
    source_sidecar_path,
    source_index,
)


SEMANTIC_FIELDS = (
    "source_dataset",
    "source_split",
    "source_key",
    "participant_id",
    "case_id",
    "session_id",
    "split",
    "pair_id",
    "z",
    "z_bin",
    "label",
)
COMPARISONS = ("fomo45k", "mpi", "oasis3", "mixed")
EXPECTED_MANIFEST_TOTAL = 29864
EXPECTED_CANDIDATE_ROWS = 38967
EXPECTED_CANDIDATE_SUBJECTS = 938
EXPECTED_SOURCE_CSV_ROWS = 938
DEFAULT_SOURCE_ROOT = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_v3_fullcandidate_20260917_final"
DEFAULT_REPLAY_ROOT = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_v3_replay"


class ReplayMismatch(RuntimeError):
    """Raised when a replay cannot establish the frozen semantic contract."""


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping for optional JSON sections.

    The replay reads sidecars produced by several revisions of the builder.
    Treating a missing or malformed optional section as empty keeps the audit
    fail-closed while avoiding an ``UnboundLocalError`` at the final audit
    write path.
    """

    return value if isinstance(value, Mapping) else {}


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def _as_row(value: Mapping[str, Any] | SliceRecord) -> Mapping[str, Any]:
    return value.to_dict() if isinstance(value, SliceRecord) else value


def semantic_projection(value: Mapping[str, Any] | SliceRecord) -> tuple[Any, ...]:
    """Return the immutable fields used to compare two manifest rows."""

    row = _as_row(value)
    projected: list[Any] = []
    for field in SEMANTIC_FIELDS:
        item = row.get(field)
        if field in {"z", "z_bin", "label"}:
            projected.append(None if item is None else int(item))
        else:
            projected.append("" if item is None else str(item))
    return tuple(projected)


def compare_semantic_rows(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any] | SliceRecord],
    *,
    comparison: str,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare rows fail-closed, including duplicate identities and fields."""

    expected_projection = [semantic_projection(row) for row in expected]
    actual_projection = [semantic_projection(row) for row in actual]
    expected_counter = Counter(expected_projection)
    actual_counter = Counter(actual_projection)
    missing = expected_counter - actual_counter
    extra = actual_counter - expected_counter
    field_mismatches = Counter()
    positional_examples: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(expected_projection, actual_projection)):
        if left == right:
            continue
        for field, expected_value, actual_value in zip(SEMANTIC_FIELDS, left, right):
            if expected_value != actual_value:
                field_mismatches[field] += 1
        if len(positional_examples) < max_examples:
            positional_examples.append(
                {
                    "index": index,
                    "expected": dict(zip(SEMANTIC_FIELDS, left)),
                    "actual": dict(zip(SEMANTIC_FIELDS, right)),
                }
            )
    missing_examples = [dict(zip(SEMANTIC_FIELDS, key)) for key in list(missing)[:max_examples]]
    extra_examples = [dict(zip(SEMANTIC_FIELDS, key)) for key in list(extra)[:max_examples]]
    return {
        "comparison": comparison,
        "status": "PASS" if len(expected_projection) == len(actual_projection) and not missing and not extra else "FAIL",
        "semantic_fields": list(SEMANTIC_FIELDS),
        "expected_rows": len(expected_projection),
        "actual_rows": len(actual_projection),
        "missing_count": int(sum(missing.values())),
        "extra_count": int(sum(extra.values())),
        "field_mismatch_counts": dict(field_mismatches),
        "missing_examples": missing_examples,
        "extra_examples": extra_examples,
        "positional_examples": positional_examples,
    }


def _canonical_json_value(value: Any) -> Any:
    """Normalize JSON-like values for strict, order-independent comparison."""

    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_json_value(value.item())
    return value


def _candidate_identity(row: Mapping[str, Any]) -> tuple[str, int | None]:
    """Return the candidate identity used by the full inventory audit."""

    participant = row.get("participant_id") or row.get("source_participant_id") or row.get("case_id")
    raw_z = row.get("z")
    try:
        z = None if raw_z in (None, "") else int(raw_z)
    except (TypeError, ValueError):
        z = None
    return str(participant or ""), z


def _candidate_serialization(row: Mapping[str, Any]) -> str:
    """Serialize every candidate field, including eligibility/support data.

    Candidate rows are metadata records rather than model outputs.  Keeping
    the complete row in the projection makes a same-count/different-candidate
    replay fail closed and covers native/model eligibility flags, support
    fractions, paths, and provenance identity.  JSON object key order and the
    builder's tuple/list representation are normalized only for comparison.
    """

    return json.dumps(_canonical_json_value(dict(row)), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compare_candidate_rows(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare the full candidate inventory, not just its row count.

    The comparison is a multiset over complete serialized rows and a second
    multiset over ``(participant_id, z)`` identities.  The latter is reported
    separately so a same-sized but shifted candidate pool cannot be mistaken
    for an equivalent replay.  All row fields remain in the full projection,
    including the three support fractions and eligibility/mask metadata.
    """

    failures: list[dict[str, Any]] = []
    expected_serialized: list[str] = []
    actual_serialized: list[str] = []
    try:
        expected_serialized = [_candidate_serialization(row) for row in expected]
        actual_serialized = [_candidate_serialization(row) for row in actual]
    except (TypeError, ValueError) as exc:
        return {
            "status": "FAIL",
            "reason": "candidate_row_not_strictly_serializable",
            "detail": str(exc),
            "expected_rows": len(expected),
            "actual_rows": len(actual),
            "candidate_identity_fields": ["participant_id", "z"],
            "full_row_fields_compared": "all",
        }

    expected_counter = Counter(expected_serialized)
    actual_counter = Counter(actual_serialized)
    missing_full = expected_counter - actual_counter
    extra_full = actual_counter - expected_counter
    expected_identity = Counter(_candidate_identity(row) for row in expected)
    actual_identity = Counter(_candidate_identity(row) for row in actual)
    missing_identity = expected_identity - actual_identity
    extra_identity = actual_identity - expected_identity

    def _decode_examples(counter: Counter[str]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for serialized, count in list(counter.items())[:max_examples]:
            try:
                row = json.loads(serialized)
            except json.JSONDecodeError:
                row = {"serialized": serialized}
            examples.append({"count": int(count), "row": row})
        return examples

    for identity, count in list(missing_identity.items())[:max_examples]:
        failures.append({"kind": "missing_identity", "identity": list(identity), "count": int(count)})
    for identity, count in list(extra_identity.items())[:max_examples]:
        failures.append({"kind": "extra_identity", "identity": list(identity), "count": int(count)})
    return {
        "status": "PASS" if len(expected) == len(actual) and not missing_full and not extra_full and not missing_identity and not extra_identity else "FAIL",
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "expected_unique_participant_z": len(expected_identity),
        "actual_unique_participant_z": len(actual_identity),
        "missing_identity_count": int(sum(missing_identity.values())),
        "extra_identity_count": int(sum(extra_identity.values())),
        "missing_full_row_count": int(sum(missing_full.values())),
        "extra_full_row_count": int(sum(extra_full.values())),
        "missing_identity_examples": [
            {"participant_id": identity[0], "z": identity[1], "count": int(count)}
            for identity, count in list(missing_identity.items())[:max_examples]
        ],
        "extra_identity_examples": [
            {"participant_id": identity[0], "z": identity[1], "count": int(count)}
            for identity, count in list(extra_identity.items())[:max_examples]
        ],
        "missing_full_row_examples": _decode_examples(missing_full),
        "extra_full_row_examples": _decode_examples(extra_full),
        "candidate_identity_fields": ["participant_id", "z"],
        "full_row_fields_compared": "all",
    }


def _candidate_ledger_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    """Keep semantic subject ledger results while excluding replay nonce."""

    # ``resume_fingerprint`` includes the output-root protocol hash and is
    # expected to differ in a fresh replay.  The subject status and complete
    # per-subject audit are the immutable ledger evidence we need to compare.
    return {
        "subject": event.get("subject"),
        "status": event.get("status"),
        "audit": _canonical_json_value(event.get("audit") if isinstance(event.get("audit"), Mapping) else {}),
    }


def compare_candidate_ledger(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare every per-subject completion result in the candidate ledger."""

    expected_projection = [json.dumps(_candidate_ledger_projection(event), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) for event in expected]
    actual_projection = [json.dumps(_candidate_ledger_projection(event), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) for event in actual]
    expected_counter = Counter(expected_projection)
    actual_counter = Counter(actual_projection)
    missing = expected_counter - actual_counter
    extra = actual_counter - expected_counter
    expected_subjects = Counter(str(event.get("subject", "")) for event in expected)
    actual_subjects = Counter(str(event.get("subject", "")) for event in actual)
    missing_subjects = expected_subjects - actual_subjects
    extra_subjects = actual_subjects - expected_subjects
    return {
        "status": "PASS" if len(expected) == len(actual) and not missing and not extra and not missing_subjects and not extra_subjects else "FAIL",
        "expected_subject_events": len(expected),
        "actual_subject_events": len(actual),
        "expected_unique_subjects": len(expected_subjects),
        "actual_unique_subjects": len(actual_subjects),
        "missing_subject_count": int(sum(missing_subjects.values())),
        "extra_subject_count": int(sum(extra_subjects.values())),
        "missing_ledger_result_count": int(sum(missing.values())),
        "extra_ledger_result_count": int(sum(extra.values())),
        "missing_subject_examples": [{"subject": subject, "count": int(count)} for subject, count in list(missing_subjects.items())[:max_examples]],
        "extra_subject_examples": [{"subject": subject, "count": int(count)} for subject, count in list(extra_subjects.items())[:max_examples]],
        "full_subject_audit_compared": True,
        "resume_fingerprint_compared": False,
    }


def tensor_sha256(tensor: Any) -> str:
    """Hash contiguous float32 tensor bytes without changing values."""

    if isinstance(tensor, torch.Tensor):
        value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    else:
        value = np.asarray(tensor, dtype=np.float32)
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _raw_brats_subject(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    value = metadata.get("source_participant_id") or row.get("source_participant_id") or row.get("participant_id") or row.get("case_id")
    text = str(value or "")
    return text.split(":", 1)[1] if text.startswith("brats21:") else text


def verify_selected_brats_tensor_hashes(
    selected_rows: Sequence[Mapping[str, Any]],
    volume_reader: Callable[[str], Any],
    *,
    max_examples: int = 30,
) -> dict[str, Any]:
    """Stream canonical volume slices and compare existing cache tensor hashes.

    ``volume_reader(subject)`` must return ``(volume, model_mask)`` for one
    raw BraTS subject.  The reader is called once per subject, so all four
    comparison selections are checked over their unique ``(participant,z)``
    union in one canonical-volume pass.  Existing rows must carry
    ``provenance.model_input_sha256``; missing hashes are a hard failure.
    """

    expected: dict[tuple[str, int], set[str]] = defaultdict(set)
    missing_hash_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        if int(row.get("label", 1)) != 1:
            continue
        subject = _raw_brats_subject(row)
        z = int(row.get("z"))
        provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        digest = provenance.get("model_input_sha256")
        if not subject or digest in (None, ""):
            missing_hash_rows.append({"participant_id": row.get("participant_id"), "z": z, "reason": "missing_existing_model_input_sha256"})
            continue
        expected[(subject, z)].add(str(digest))

    mismatches: list[dict[str, Any]] = []
    conflicting_expected_hashes: list[dict[str, Any]] = [
        {
            "subject": subject,
            "z": z,
            "expected_sha256": sorted(digests),
            "reason": "conflicting_expected_model_input_sha256",
        }
        for (subject, z), digests in sorted(expected.items())
        if len(digests) != 1
    ]
    mismatches.extend(conflicting_expected_hashes)
    checked = 0
    subjects_loaded: list[str] = []
    aggregate = hashlib.sha256()
    for subject in sorted({subject for subject, _z in expected}):
        value = volume_reader(subject)
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            mismatches.append({"subject": subject, "reason": "volume_reader_must_return_volume_and_model_mask"})
            continue
        volume, model_mask = value[0], value[1]
        volume_tensor = torch.as_tensor(volume, dtype=torch.float32)
        mask_tensor = torch.as_tensor(model_mask)
        subjects_loaded.append(subject)
        if tuple(volume_tensor.shape) != (3, 128, 128, ATLAS_DEPTH):
            mismatches.append({"subject": subject, "reason": "unexpected_volume_shape", "shape": list(volume_tensor.shape)})
            continue
        if tuple(mask_tensor.shape) != (128, 128, ATLAS_DEPTH):
            mismatches.append({"subject": subject, "reason": "unexpected_model_mask_shape", "shape": list(mask_tensor.shape)})
            continue
        if not bool(torch.isfinite(volume_tensor).all()):
            mismatches.append({"subject": subject, "reason": "nonfinite_volume"})
            continue
        for key in sorted(key for key in expected if key[0] == subject):
            _subject, z = key
            if z < 0 or z >= ATLAS_DEPTH:
                mismatches.append({"subject": subject, "z": z, "reason": "z_outside_atlas_depth"})
                continue
            if bool(mask_tensor[..., z].any()):
                mismatches.append({"subject": subject, "z": z, "reason": "model_mask_nonzero"})
            digest = tensor_sha256(volume_tensor[..., z])
            expected_digests = sorted(expected[key])
            aggregate.update(f"{subject}|{z}|{digest}\n".encode("utf-8"))
            checked += 1
            # Multiple historical rows for one (participant,z) must agree on
            # exactly one cache digest.  Membership in a set would allow two
            # conflicting historical hashes to pass accidentally.
            if len(expected[key]) != 1 or digest != next(iter(expected[key])):
                mismatches.append({"subject": subject, "z": z, "expected_sha256": expected_digests, "actual_sha256": digest})
    return {
        "status": "PASS" if expected and not missing_hash_rows and checked == len(expected) and not mismatches else "FAIL",
        "selected_rows": len([row for row in selected_rows if int(row.get("label", 1)) == 1]),
        "unique_subject_z": len(expected),
        "subjects_loaded": len(subjects_loaded),
        "checked_subject_z": checked,
        "aggregate_sha256": aggregate.hexdigest(),
        "missing_hash_rows": missing_hash_rows[:max_examples],
        "missing_hash_count": len(missing_hash_rows),
        "conflicting_expected_hash_count": len(conflicting_expected_hashes),
        "mismatch_examples": mismatches[:max_examples],
        "mismatch_count": len(mismatches),
        "hash_algorithm": "sha256(float32 contiguous tensor bytes)",
        "scope": "all four comparison selected BraTS (participant,z) union; one canonical volume read per subject",
    }


def _close_reader(reader: Any) -> None:
    transaction = getattr(reader, "txn", None)
    if transaction is not None:
        transaction.abort()
        reader.txn = None
    environment = getattr(reader, "env", None)
    if environment is not None:
        environment.close()
        reader.env = None


def verify_healthy_source_tensors(
    rows_by_comparison: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reader_factory: Callable[[str, str], Any] | None = None,
    tensor_hash_ledger: Any = None,
    max_examples: int = 30,
) -> dict[str, Any]:
    """Verify every selected healthy source tensor, including Mixed concat.

    Direct cohorts are checked for canonical shape/dtype/finite values.  Mixed
    healthy rows additionally compare the underlying source LMDB tensor to the
    Mixed LMDB tensor with ``torch.equal``.  Readers are cached per
    ``(dataset, split)`` and closed before returning.
    """

    factory = reader_factory or (lambda dataset, split: LMDBSliceDataset(source_lmdb_path(dataset, split), image_size=None))
    readers: dict[tuple[str, str], Any] = {}
    failures: list[dict[str, Any]] = []
    checked = 0
    mixed_checked = 0
    ledger_index = _healthy_hash_ledger_index(tensor_hash_ledger) if tensor_hash_ledger is not None else {}
    ledger_checked = 0
    ledger_failures: list[dict[str, Any]] = []
    ledger_observed: set[tuple[str, str, str, str, str]] = set()
    source_counts: Counter[str] = Counter()
    aggregate = hashlib.sha256()
    try:
        for comparison in COMPARISONS:
            for row in rows_by_comparison.get(comparison, []):
                if int(row.get("label", 0)) != 0:
                    continue
                source_dataset = str(row.get("source_dataset") or comparison)
                source_split = str(row.get("source_split") or "")
                source_key = str(row.get("source_key") or "")
                if comparison == "mixed":
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
                    underlying_dataset = str(metadata.get("underlying_source_dataset") or "")
                    underlying_split = str(metadata.get("underlying_source_split") or source_split)
                    underlying_key = str(metadata.get("underlying_source_key") or "")
                    if not underlying_dataset or not underlying_key:
                        failures.append({"comparison": comparison, "record_id": row.get("source_key"), "reason": "missing_mixed_underlying_identity"})
                        continue
                    source_dataset = underlying_dataset
                    source_split = underlying_split
                    source_key = underlying_key
                if not source_dataset or not source_split or not source_key:
                    failures.append({"comparison": comparison, "reason": "missing_source_identity", "row": dict(row)})
                    continue
                source_reader_key = (source_dataset, source_split)
                if source_reader_key not in readers:
                    readers[source_reader_key] = factory(*source_reader_key)
                source_tensor = torch.as_tensor(readers[source_reader_key][int(source_key)], dtype=torch.float32).contiguous()
                if tuple(source_tensor.shape) != MODEL_SHAPE or not bool(torch.isfinite(source_tensor).all()):
                    failures.append({"comparison": comparison, "source_dataset": source_dataset, "source_split": source_split, "source_key": source_key, "reason": "source_tensor_shape_or_finite_failure", "shape": list(source_tensor.shape)})
                    continue
                source_digest = tensor_sha256(source_tensor)
                aggregate.update(f"{comparison}|{row.get('participant_id')}|{source_split}|{source_key}|{source_digest}\n".encode("utf-8"))
                checked += 1
                source_counts[source_dataset] += 1
                if tensor_hash_ledger is not None:
                    identity = _healthy_tensor_identity(comparison, row)
                    ledger_observed.add(identity)
                    expected_digests = ledger_index.get(identity, set())
                    if not expected_digests:
                        ledger_failures.append({"comparison": comparison, "identity": list(identity), "reason": "selected_healthy_identity_missing_from_tensor_hash_ledger"})
                    elif len(expected_digests) != 1 or source_digest != next(iter(expected_digests)):
                        ledger_failures.append({"comparison": comparison, "identity": list(identity), "expected_sha256": sorted(expected_digests), "actual_sha256": source_digest, "reason": "selected_healthy_tensor_hash_mismatch"})
                    else:
                        ledger_checked += 1
                if comparison == "mixed":
                    mixed_reader_key = ("mixed", "train" if source_split == "train" else "val")
                    if mixed_reader_key not in readers:
                        readers[mixed_reader_key] = factory(*mixed_reader_key)
                    mixed_key = str(row.get("source_key") or "")
                    mixed_tensor = torch.as_tensor(readers[mixed_reader_key][int(mixed_key)], dtype=torch.float32).contiguous()
                    mixed_checked += 1
                    if tuple(mixed_tensor.shape) != MODEL_SHAPE or not torch.equal(source_tensor, mixed_tensor):
                        failures.append({"comparison": comparison, "record_id": row.get("source_key"), "reason": "mixed_tensor_not_exact_source", "source_sha256": source_digest, "mixed_sha256": tensor_sha256(mixed_tensor), "mixed_shape": list(mixed_tensor.shape)})
    except Exception as exc:
        failures.append({"reason": "source_tensor_reader_error", "type": type(exc).__name__, "detail": str(exc)})
    finally:
        for reader in readers.values():
            _close_reader(reader)
    expected_healthy = sum(sum(1 for row in rows if int(row.get("label", 0)) == 0) for rows in rows_by_comparison.values())
    if tensor_hash_ledger is not None:
        for identity in sorted(set(ledger_index) - ledger_observed):
            ledger_failures.append({"identity": list(identity), "reason": "tensor_hash_ledger_contains_unselected_identity"})
    if tensor_hash_ledger is None:
        tensor_ledger_status = "NOT_PROVIDED"
    elif not ledger_index:
        tensor_ledger_status = "INCONCLUSIVE_NO_HISTORICAL_SELECTED_TENSOR_HASH"
    elif ledger_failures:
        tensor_ledger_status = "FAIL"
    elif ledger_checked == expected_healthy:
        tensor_ledger_status = "PASS"
    else:
        tensor_ledger_status = "FAIL"
    all_failures = failures + ledger_failures
    return {
        "status": "PASS" if checked == expected_healthy and not all_failures else "FAIL",
        "expected_healthy_rows": expected_healthy,
        "checked_healthy_rows": checked,
        "mixed_rows_checked": mixed_checked,
        "source_dataset_counts": dict(source_counts),
        "aggregate_sha256": aggregate.hexdigest(),
        "failure_count": len(all_failures),
        "failure_examples": all_failures[:max_examples],
        "tensor_hash_ledger_status": tensor_ledger_status,
        "tensor_hash_ledger_expected_entries": len(ledger_index),
        "tensor_hash_ledger_checked_entries": ledger_checked,
        "tensor_hash_ledger_unobserved_entries": len(set(ledger_index) - ledger_observed),
        "tensor_hash_ledger_failure_count": len(ledger_failures),
        "hash_algorithm": "sha256(float32 contiguous tensor bytes)",
    }


def _source_fingerprint_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the immutable ``dataset:split -> digest`` map from a sidecar."""

    sidecars = payload.get("sidecars")
    if isinstance(sidecars, Mapping):
        return sidecars
    # A compact test/replay sidecar may use the map directly.  Do not recurse
    # through arbitrary nested JSON: an absent explicit map must remain an
    # auditable limitation instead of being guessed from unrelated hashes.
    if payload and all(isinstance(key, str) and ":" in key for key in payload):
        return payload
    return {}


def verify_source_fingerprint_continuity(
    rows_by_comparison: Mapping[str, Sequence[Mapping[str, Any]]],
    source_fingerprints: Mapping[str, Any] | None,
    *,
    max_examples: int = 30,
) -> dict[str, Any]:
    """Check selected-row provenance against historical source sidecar hashes.

    A newly computed digest by itself cannot prove continuity with the source
    used by a prior selection.  This helper therefore requires both the
    historical ``source_fingerprints.sidecars`` map and each selected row's
    recorded digest/path.  Missing historical evidence is returned as an
    explicit inconclusive limitation, rather than a PASS.
    """

    historical = _source_fingerprint_map(_mapping(source_fingerprints)) if source_fingerprints else {}
    failures: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    checked_rows = 0
    checked_keys: set[str] = set()
    digest_cache: dict[str, str] = {}
    for comparison in COMPARISONS:
        for row in rows_by_comparison.get(comparison, []):
            if int(row.get("label", 0)) != 0:
                continue
            metadata = _mapping(row.get("metadata"))
            provenance = _mapping(row.get("provenance"))
            if comparison == "mixed":
                # The mixed LMDB sidecar is keyed by the mixed split while
                # the row's source_split records the underlying source split.
                dataset = "mixed"
                split = "train" if str(row.get("source_split", "")) == "train" else "val"
            else:
                dataset = str(row.get("source_dataset") or comparison)
                split = str(row.get("source_split") or "")
            map_key = f"{dataset}:{split}"
            checked_keys.add(map_key)
            expected_digest = historical.get(map_key)
            if isinstance(expected_digest, Mapping):
                expected_digest = expected_digest.get("sha256") or expected_digest.get("digest")
            expected_digest = str(expected_digest or "")
            row_digest = str(metadata.get("source_sidecar_sha256") or "")
            sidecar_path = str(provenance.get("source_sidecar") or "")
            if not expected_digest:
                limitations.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "reason": "missing_historical_source_fingerprint"})
                continue
            if not row_digest:
                failures.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "reason": "selected_row_missing_source_sidecar_sha256"})
                continue
            if row_digest != expected_digest:
                failures.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "expected_sha256": expected_digest, "row_sha256": row_digest, "reason": "selected_row_source_fingerprint_mismatch"})
                continue
            if not sidecar_path:
                limitations.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "reason": "historical_source_fingerprint_path_missing"})
                continue
            if sidecar_path not in digest_cache:
                path = Path(sidecar_path)
                if not path.is_file():
                    limitations.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "path": sidecar_path, "reason": "historical_source_fingerprint_path_unreadable"})
                    continue
                try:
                    digest_cache[sidecar_path] = sha256(path)
                except (OSError, ValueError) as exc:
                    limitations.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "path": sidecar_path, "reason": "historical_source_fingerprint_read_error", "detail": str(exc)})
                    continue
            actual_digest = digest_cache.get(sidecar_path)
            if actual_digest != expected_digest:
                failures.append({"comparison": comparison, "source_key": row.get("source_key"), "fingerprint_key": map_key, "path": sidecar_path, "expected_sha256": expected_digest, "actual_sha256": actual_digest, "reason": "source_sidecar_changed_since_historical_fingerprint"})
                continue
            checked_rows += 1
    if not historical:
        status = "INCONCLUSIVE_NO_HISTORICAL_FINGERPRINT"
    elif failures:
        status = "FAIL"
    elif limitations:
        status = "INCONCLUSIVE_HISTORICAL_FINGERPRINT_INCOMPLETE"
    elif checked_rows:
        status = "PASS"
    else:
        status = "INCONCLUSIVE_NO_HEALTHY_ROWS"
    return {
        "status": status,
        "historical_fingerprint_keys": sorted(str(key) for key in historical),
        "checked_rows": checked_rows,
        "checked_fingerprint_keys": sorted(checked_keys),
        "failure_count": len(failures),
        "failure_examples": failures[:max_examples],
        "limitation_count": len(limitations),
        "limitation_examples": limitations[:max_examples],
        "scope": "selected healthy row provenance/source sidecar continuity; this does not by itself prove tensor-byte continuity",
    }


def _healthy_hash_ledger_entries(payload: Any) -> list[Mapping[str, Any]]:
    """Read the explicit selected-healthy tensor hash ledger container."""

    if isinstance(payload, Mapping):
        for key in ("entries", "rows", "records", "selected_healthy", "healthy_rows", "tensor_hashes", "ledger_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
        return []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _healthy_tensor_identity(comparison: str, row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    metadata = _mapping(row.get("metadata"))
    if comparison == "mixed":
        source_dataset = str(metadata.get("underlying_source_dataset") or "")
        source_split = str(metadata.get("underlying_source_split") or row.get("source_split") or "")
        source_key = str(metadata.get("underlying_source_key") or "")
        local_key = str(row.get("source_key") or metadata.get("mixed_local_key") or "")
    else:
        source_dataset = str(row.get("source_dataset") or comparison)
        source_split = str(row.get("source_split") or "")
        source_key = str(row.get("source_key") or "")
        local_key = ""
    return comparison, source_dataset, source_split, source_key, local_key


def _healthy_hash_ledger_index(payload: Any) -> dict[tuple[str, str, str, str, str], set[str]]:
    """Normalize the ledger writer's canonical rows into row identities."""

    expected: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for entry in _healthy_hash_ledger_entries(payload):
        canonical_dataset = str(entry.get("canonical_source_dataset") or entry.get("source_dataset") or entry.get("underlying_source_dataset") or "")
        canonical_split = str(entry.get("canonical_source_split") or entry.get("source_split") or entry.get("underlying_source_split") or "")
        canonical_key = str(entry.get("canonical_source_key") or entry.get("source_key") or entry.get("underlying_source_key") or "")
        digest = str(entry.get("tensor_sha256") or entry.get("source_tensor_sha256") or entry.get("model_input_sha256") or entry.get("sha256") or "")
        memberships = entry.get("memberships")
        if isinstance(memberships, list) and memberships:
            for membership in memberships:
                if not isinstance(membership, Mapping):
                    continue
                comparison = str(membership.get("comparison") or membership.get("cohort") or membership.get("domain") or "")
                source_dataset = str(membership.get("underlying_source_dataset") or canonical_dataset)
                source_split = str(membership.get("underlying_source_split") or canonical_split)
                source_key = str(membership.get("underlying_source_key") or canonical_key)
                local_key = str(membership.get("mixed_local_key") or membership.get("local_source_key") or membership.get("source_key") or "") if comparison == "mixed" else ""
                expected[(comparison, source_dataset, source_split, source_key, local_key)].add(digest)
            continue
        comparison = str(entry.get("comparison") or entry.get("cohort") or entry.get("domain") or "")
        source_dataset = str(entry.get("underlying_source_dataset") or canonical_dataset or comparison)
        source_split = str(entry.get("underlying_source_split") or canonical_split)
        source_key = str(entry.get("underlying_source_key") or canonical_key)
        local_key = str(entry.get("mixed_local_key") or entry.get("local_source_key") or entry.get("source_key") or "") if comparison == "mixed" else ""
        expected[(comparison, source_dataset, source_split, source_key, local_key)].add(digest)
    return expected


def verify_healthy_tensor_hash_ledger(
    rows_by_comparison: Mapping[str, Sequence[Mapping[str, Any]]],
    ledger_payload: Any,
    *,
    max_examples: int = 30,
) -> dict[str, Any]:
    """Compare selected healthy identities and hashes with a frozen ledger.

    The ledger is a future/runtime binding artifact.  Its absence is reported
    explicitly so a fresh tensor digest cannot be mistaken for proof against a
    prior cache.  Mixed entries include both the local and underlying identity.
    """

    entries = _healthy_hash_ledger_entries(ledger_payload)
    if not entries:
        return {
            "status": "INCONCLUSIVE_NO_HISTORICAL_SELECTED_TENSOR_HASH",
            "expected_entries": 0,
            "checked_entries": 0,
            "failure_count": 0,
            "limitation": "selected-healthy tensor hash ledger is absent or has no explicit entries",
        }
    expected = _healthy_hash_ledger_index(entries)
    actual: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for comparison in COMPARISONS:
        for row in rows_by_comparison.get(comparison, []):
            if int(row.get("label", 0)) != 0:
                continue
            identity = _healthy_tensor_identity(comparison, row)
            provenance = _mapping(row.get("provenance"))
            digest = str(provenance.get("model_input_sha256") or provenance.get("source_tensor_sha256") or provenance.get("tensor_sha256") or "")
            if digest:
                actual[identity].add(digest)
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    mismatches = [key for key in set(expected) & set(actual) if expected[key] != actual[key] or len(expected[key]) != 1 or len(actual[key]) != 1]
    examples = [{"identity": list(key), "expected": sorted(expected.get(key, set())), "actual": sorted(actual.get(key, set()))} for key in list(missing | extra | set(mismatches))[:max_examples]]
    status = "PASS" if expected and not missing and not extra and not mismatches else "FAIL"
    return {
        "status": status,
        "expected_entries": len(expected),
        "actual_entries": len(actual),
        "checked_entries": len(set(expected) & set(actual)),
        "missing_entry_count": len(missing),
        "extra_entry_count": len(extra),
        "mismatch_count": len(mismatches),
        "mismatch_examples": examples,
        "scope": "selected healthy source tensor bytes, including Mixed local/underlying identity",
    }


def _load_existing_manifests(source_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for comparison in COMPARISONS:
        split_rows: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            path = source_root / "manifests" / comparison / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            split_rows[split] = _jsonl_rows(path)
        result[comparison] = split_rows
    return result


def _flatten_rows(rows_by_comparison: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    return {comparison: [dict(row) for split in SPLITS for row in rows_by_comparison.get(comparison, {}).get(split, [])] for comparison in COMPARISONS}


def _make_mixed_healthy_records(
    mixed_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_support: Mapping[str, Mapping[tuple[str, str], tuple[float, float, float]]],
    old_maps: Mapping[str, Mapping[str, str]],
    source_hashes: Mapping[tuple[str, str], str],
) -> list[SliceRecord]:
    """Recreate builder mixed healthy records without selected-cache writes."""

    base_indices = {name: source_index(rows) for name, rows in source_rows.items() if name != "mixed"}
    healthy_rows: list[dict[str, Any]] = []
    for row in mixed_rows:
        source_dataset = str(row.get("source_dataset") or row.get("underlying_source_dataset") or "")
        source_split = str(row.get("source_split") or "")
        source_key = str(row.get("source_key") or "")
        underlying = base_indices.get(source_dataset, {}).get((source_split, source_key))
        if underlying is None:
            raise ReplayMismatch(f"Mixed row cannot resolve {source_dataset}:{source_split}:{source_key}")
        item = dict(underlying)
        item["source_dataset"] = source_dataset
        item["mixed_local_key"] = str(row.get("mixed_local_key") or row.get("key") or "")
        item["_support_fraction"] = source_support[source_dataset][(source_split, str(item["source_key"]))]
        healthy_rows.append(item)

    participants = [namespaced_participant(str(row["source_dataset"]), str(row["participant_id"])) for row in healthy_rows]
    standalone_maps: dict[str, dict[str, str]] = {}
    for dataset in ("fomo45k", "mpi", "oasis3"):
        ids = sorted({namespaced_participant(dataset, str(row["participant_id"])) for row in healthy_rows if str(row["source_dataset"]) == dataset})
        standalone_maps[dataset], _info = assign_map(ids, seed=73, existing=old_maps.get(dataset, {}))
    participant_map = {pid: standalone_maps[pid.split(":", 1)[0]][pid] for pid in participants}
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in healthy_rows:
        by_participant[namespaced_participant(str(row["source_dataset"]), str(row["participant_id"]))].append(row)
    primary = {pid: sorted((str(item.get("case_id", "")), str(item.get("session_id", ""))) for item in values)[0] for pid, values in by_participant.items()}
    registered = read_old_fomo_registered_paths()
    records: list[SliceRecord] = []
    for row in healthy_rows:
        dataset = str(row["source_dataset"])
        pid = namespaced_participant(dataset, str(row["participant_id"]))
        case = str(row.get("case_id", ""))
        session = str(row.get("session_id", ""))
        if (case, session) != primary[pid]:
            continue
        fractions = tuple(float(value) for value in row["_support_fraction"])
        if any(value + 1e-12 < SUPPORT_THRESHOLD for value in fractions):
            continue
        source_split = str(row["source_split"])
        source_key = str(row["source_key"])
        mixed_split = "train" if source_split == "train" else "val"
        mixed_key = str(row["mixed_local_key"])
        z = int(row["z"])
        lmdb_path = source_lmdb_path("mixed", mixed_split)
        metadata = {
            "input_kind": "lmdb",
            "lmdb_path": str(lmdb_path),
            "normalization": "robust_iqr",
            "normalize_input": False,
            "model_grid_support_fraction": list(fractions),
            "model_grid_support_expression": f"abs(x + 1) > {SUPPORT_EPS:g}",
            "model_grid_support_threshold": SUPPORT_THRESHOLD,
            "model_grid_support_denominator": 128 * 128,
            "geometry_proxy": "atlas_model_grid_shape_only",
            "geometry_proxy_is_native_brain_mask": False,
            "source_participant_id": str(row["participant_id"]),
            "source_case_id": case,
            "source_session_id": session,
            "source_key_namespace": "mixed_local_per_source_split",
            "mixed_local_key": mixed_key,
            "underlying_source_dataset": dataset,
            "underlying_source_key": source_key,
            "underlying_source_split": source_split,
            "source_sidecar_sha256": source_hashes[("mixed", mixed_split)],
        }
        provenance = {
            "source_sidecar": str(source_sidecar_path("mixed", mixed_split)),
            "source_lmdb_path": str(lmdb_path),
            "source_split_immutable": source_split,
            "source_key_immutable": source_key,
            "mixed_local_key_immutable": mixed_key,
            "z_norm_denominator": ATLAS_DEPTH,
            "z_norm_basis": "verified_atlas_depth_155_not_max_available_lmdb_index",
            "build_protocol": "model_grid_v3_replay",
            "mixed_exact_concat_source": True,
        }
        z_norm = float(z) / float(ATLAS_DEPTH - 1)
        records.append(
            SliceRecord(
                split=participant_map[pid],
                label=0,
                domain="mixed",
                participant_id=pid,
                session_id=session,
                case_id=case,
                z=z,
                z_norm=z_norm,
                z_bin=z_bin_for(z_norm, Z_BINS),
                source_dataset="mixed",
                source_key=mixed_key,
                source_split=source_split,
                registered_paths=registered_paths_for_row(registered, source_split=source_split, source_key=source_key, participant_id=pid, case_id=case, session_id=session),
                model_shape=MODEL_SHAPE,
                geometry_shape=ATLAS_SHAPE,
                stage="final",
                foreground_fraction=fractions,
                provenance=provenance,
                metadata=metadata,
            )
        )
    return records


def _source_hashes(source_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for dataset in ("fomo45k", "mpi", "oasis3", "mixed"):
        for split in sorted({str(row.get("source_split")) for row in source_rows.get(dataset, [])}):
            hashes[(dataset, split)] = sha256(source_sidecar_path(dataset, split))
    return hashes


def _new_output_root(source_root: Path, output_root: Path) -> None:
    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("replay output must be a fresh root; source root cannot be overwritten")
    if output_resolved.exists() and any(output_resolved.iterdir()):
        raise FileExistsError(f"replay output root must be new and empty: {output_resolved}")
    output_resolved.mkdir(parents=True, exist_ok=True)


def _load_optional_healthy_hash_ledger(source_root: Path) -> tuple[Any, dict[str, Any]]:
    """Load the runtime's frozen selected-healthy hash ledger if present.

    The ledger writer currently emits a JSONL file plus a summary containing
    its absolute path.  A few explicitly named aliases are accepted for
    schema evolution; arbitrary recursive discovery is deliberately avoided
    so an unrelated hash file cannot silently become binding evidence.
    """

    candidates: list[Path] = [
        source_root / "selected_healthy_source_tensor_ledger.jsonl",
        source_root / "healthy_source_tensor_ledger.jsonl",
        source_root / "healthy_tensor_hash_ledger.jsonl",
        source_root / "selected_healthy_source_tensor_ledger" / "selected_healthy_source_tensor_ledger.jsonl",
        source_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl",
    ]
    summary_candidates = [
        source_root / "selected_healthy_source_tensor_ledger_summary.json",
        source_root / "healthy_source_tensor_ledger_summary.json",
        source_root / "healthy_tensor_hash_ledger_summary.json",
        source_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger_summary.json",
    ]
    summary_path: Path | None = next((path for path in summary_candidates if path.is_file()), None)
    summary: Mapping[str, Any] = {}
    declared_ledger_path: Path | None = None
    if summary_path is not None:
        try:
            summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            summary = {}
        ledger_value = summary.get("ledger_path")
        if ledger_value not in (None, ""):
            declared_ledger_path = Path(str(ledger_value))
            if not declared_ledger_path.is_absolute():
                declared_ledger_path = summary_path.parent / declared_ledger_path
            candidates.insert(0, declared_ledger_path)
    ledger_path = next((path for path in candidates if path.is_file()), None)
    if ledger_path is None:
        return None, {
            "status": "INCONCLUSIVE_NO_HISTORICAL_SELECTED_TENSOR_HASH",
            "path": None,
            "summary_path": str(summary_path.resolve()) if summary_path is not None else None,
            "summary_status": summary.get("status"),
            "summary_sha256": summary.get("ledger_sha256"),
        }
    try:
        # Hash the exact bytes that are parsed below.  Reopening the file after
        # hashing would allow a concurrent rewrite to pair a valid digest with
        # different JSON records, so keep the binding to one byte snapshot.
        ledger_bytes = ledger_path.read_bytes()
        actual_ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
        entries = []
        for line_number, line in enumerate(ledger_bytes.splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line.decode("utf-8"))
            if not isinstance(item, dict):
                raise ValueError(f"{ledger_path}:{line_number}: expected JSON object")
            entries.append(item)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, {
            "status": "FAIL",
            "path": str(ledger_path.resolve()),
            "summary_path": str(summary_path.resolve()) if summary_path is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary_digest = str(summary.get("ledger_sha256") or "").strip().lower()
    summary_status = str(summary.get("status") or "").strip().upper()
    declared_path_text = str(declared_ledger_path.resolve()) if declared_ledger_path is not None else None
    selected_path_text = str(ledger_path.resolve())
    path_matches = declared_path_text is None or declared_path_text == selected_path_text
    digest_matches = bool(summary_digest) and actual_ledger_sha256 == summary_digest
    if summary_path is None:
        binding_status = "INCONCLUSIVE_LEDGER_SUMMARY_MISSING"
    elif not summary_digest:
        binding_status = "INCONCLUSIVE_LEDGER_DIGEST_MISSING"
    elif not path_matches:
        binding_status = "FAIL_LEDGER_PATH_MISMATCH"
    elif not digest_matches:
        binding_status = "FAIL_LEDGER_DIGEST_MISMATCH"
    elif summary_status != "PASS":
        binding_status = "FAIL_LEDGER_SUMMARY_NOT_PASS"
    else:
        binding_status = "PASS"
    return entries, {
        "status": binding_status,
        "path": selected_path_text,
        "summary_path": str(summary_path.resolve()) if summary_path is not None else None,
        "entry_count": len(entries),
        "summary_status": summary.get("status"),
        "summary_sha256": summary.get("ledger_sha256"),
        "actual_ledger_sha256": actual_ledger_sha256,
        "ledger_sha256_match": digest_matches,
        "declared_ledger_path": declared_path_text,
        "ledger_path_match": path_matches,
        "binding_scope": "summary status PASS and summary ledger_sha256 equals the parsed ledger byte snapshot",
    }


def run_replay(
    source_root: str | Path,
    output_root: str | Path,
    *,
    brats_root: str | Path = BRATS_ROOT,
    brats_csv: str | Path = BRATS_CSV,
) -> dict[str, Any]:
    """Replay v3 selection into a new root and write only audit/rows artifacts.

    This function is intentionally expensive and must be launched explicitly
    by the runtime owner.  It never calls ``build_one_comparison`` or
    ``materialize_selected_brats_cache``.
    """

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    brats_root = Path(brats_root).resolve()
    brats_csv = Path(brats_csv).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    _new_output_root(source_root, output_root)
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    existing = _load_existing_manifests(source_root)
    existing_flat = _flatten_rows(existing)
    producer_audit_path = source_root / "producer_code_provenance_audit.json"
    producer_audit: Mapping[str, Any] = {}
    if producer_audit_path.is_file():
        producer_audit = json.loads(producer_audit_path.read_text(encoding="utf-8"))
    source_fingerprints_path = source_root / "source_fingerprints.json"
    source_fingerprints: Mapping[str, Any] = {}
    if source_fingerprints_path.is_file():
        try:
            source_fingerprints = _mapping(json.loads(source_fingerprints_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            source_fingerprints = {}
    healthy_hash_ledger, healthy_hash_ledger_info = _load_optional_healthy_hash_ledger(source_root)
    protocol = {
        "schema_version": 1,
        "replay_id": "model_grid_v3_independent_replay_v1",
        "created_at_utc": created_at,
        "source_root": str(source_root),
        "source_build_summary_sha256": sha256(source_root / "build_summary.json"),
        "semantic_fields": list(SEMANTIC_FIELDS),
        "seed": 73,
        "z_bins": Z_BINS,
        "cap_per_pair_per_bin": PAIR_CAP_PER_BIN,
        "candidate_source": "full CSV/map native-zero then model-mask-zero then model-grid support",
        "candidate_rows_expected": EXPECTED_CANDIDATE_ROWS,
        "replay_code_sha256": sha256(Path(__file__).resolve()),
        "builder_code_sha256": sha256((REPO_ROOT / "scripts" / "build_model_grid_v3.py").resolve()),
        "no_npz_copy": True,
        "no_training": True,
        "producer_snapshot_available": producer_audit.get("source_snapshot_available"),
        "producer_code_sha256": producer_audit.get("producer_code_sha256"),
        "producer_snapshot_note": producer_audit.get("source_snapshot_note", "producer provenance sidecar unavailable"),
        "source_fingerprints_path": str(source_fingerprints_path.resolve()) if source_fingerprints_path.is_file() else None,
        "source_fingerprints_sha256": sha256(source_fingerprints_path) if source_fingerprints_path.is_file() else None,
        "healthy_tensor_hash_ledger": healthy_hash_ledger_info,
    }
    _json_dump(output_root / "protocol.json", protocol)
    _json_dump(output_root / "source_reference.json", {"source_root": str(source_root), "producer_audit": str(producer_audit_path), "producer_audit_sha256": sha256(producer_audit_path) if producer_audit_path.is_file() else None})

    audit: dict[str, Any] = {
        "schema_version": 1,
        "replay_id": protocol["replay_id"],
        "status": "RUNNING",
        "created_at_utc": created_at,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "existing_manifest_records": {name: len(rows) for name, rows in existing_flat.items()},
        "existing_manifest_total": sum(len(rows) for rows in existing_flat.values()),
        "expected_manifest_total": EXPECTED_MANIFEST_TOTAL,
        "candidate_inventory": {},
        "candidate_replay": {},
        "semantic_comparison": {},
        "brats_tensor_hash_audit": {},
        "healthy_source_tensor_audit": {},
        "healthy_source_fingerprint_audit": {},
        "healthy_tensor_hash_ledger": healthy_hash_ledger_info,
        "control_binding_status": "NOT_BOUND_UNTIL_FULL_REPLAY_EQUIVALENCE",
        "failures": [],
    }

    try:
        old_maps = read_old_split_maps()
        source_rows = canonical_source_rows()
        source_hashes = _source_hashes(source_rows)
        brats_rows, brats_summary = load_brats_candidate_rows(
            old_maps["brats21"], output_root, brats_root=brats_root, csv_path=brats_csv
        )
        audit["candidate_inventory"] = {
            "status": brats_summary.get("status"),
            "full_candidate_coverage": brats_summary.get("full_candidate_coverage", {}),
            "model_grid_counts": brats_summary.get("model_grid_counts", {}),
            "candidate_rows_path": str((output_root / "brats_candidate_rows.jsonl").resolve()),
            "candidate_rows_sha256": sha256(output_root / "brats_candidate_rows.jsonl"),
        }
        if audit["existing_manifest_total"] != EXPECTED_MANIFEST_TOTAL:
            raise ReplayMismatch(
                f"existing manifest total mismatch: {audit['existing_manifest_total']} != {EXPECTED_MANIFEST_TOTAL}"
            )
        candidate_counts = brats_summary.get("model_grid_counts", {})
        actual_candidate_rows = int(candidate_counts.get("candidate_rows", -1))
        actual_candidate_subjects = int(candidate_counts.get("subjects", -1))
        source_csv_rows = int(_mapping(brats_summary.get("source")).get("csv_rows_verified", -1))
        if (actual_candidate_rows, actual_candidate_subjects, source_csv_rows) != (
            EXPECTED_CANDIDATE_ROWS,
            EXPECTED_CANDIDATE_SUBJECTS,
            EXPECTED_SOURCE_CSV_ROWS,
        ):
            raise ReplayMismatch(
                "full candidate inventory count mismatch: "
                f"rows={actual_candidate_rows}, subjects={actual_candidate_subjects}, csv={source_csv_rows}"
            )
        if brats_summary.get("status") != "PASS" or not brats_summary.get("full_candidate_coverage", {}).get("verified"):
            raise ReplayMismatch("full candidate inventory did not pass")

        historical_candidate_path = source_root / "brats_candidate_rows.jsonl"
        historical_ledger_path = source_root / "brats_candidate_completion.jsonl"
        if not historical_candidate_path.is_file() or not historical_ledger_path.is_file():
            raise ReplayMismatch("historical candidate rows and completion ledger are required for full candidate replay comparison")
        candidate_audit = compare_candidate_rows(_jsonl_rows(historical_candidate_path), brats_rows)
        ledger_audit = compare_candidate_ledger(_jsonl_rows(historical_ledger_path), _jsonl_rows(output_root / "brats_candidate_completion.jsonl"))
        audit["candidate_replay"] = {
            "status": "PASS" if candidate_audit.get("status") == "PASS" and ledger_audit.get("status") == "PASS" else "FAIL",
            "historical_candidate_rows_path": str(historical_candidate_path.resolve()),
            "replay_candidate_rows_path": str((output_root / "brats_candidate_rows.jsonl").resolve()),
            "historical_completion_ledger_path": str(historical_ledger_path.resolve()),
            "replay_completion_ledger_path": str((output_root / "brats_candidate_completion.jsonl").resolve()),
            "candidate_rows": candidate_audit,
            "subject_completion_ledger": ledger_audit,
        }
        if audit["candidate_replay"]["status"] != "PASS":
            raise ReplayMismatch("full candidate rows or per-subject completion ledger differ from the historical revision")

        source_support: dict[str, Mapping[tuple[str, str], tuple[float, float, float]]] = {}
        support_summaries: dict[str, Any] = {}
        for dataset in ("fomo45k", "mpi", "oasis3"):
            source_support[dataset], support_summaries[dataset] = materialize_healthy_support(source_rows[dataset], dataset)

        replay_records: dict[str, list[dict[str, Any]]] = {}
        for comparison in ("fomo45k", "mpi", "oasis3"):
            healthy_participants = [namespaced_participant(comparison, str(row["participant_id"])) for row in source_rows[comparison]]
            participant_map, _map_info = assign_map(healthy_participants, seed=73, existing=old_maps.get(comparison, {}))
            healthy_records, _healthy_exclusions = make_healthy_records(
                source_rows[comparison], comparison, source_support[comparison], participant_map, source_hashes,
                registered_paths=read_old_fomo_registered_paths() if comparison == "fomo45k" else {},
            )
            brats_records, _brats_exclusions = make_brats_records(brats_rows, old_maps["brats21"])
            pairing = build_pairs(healthy_records, brats_records, comparison=comparison, seed=73, bins=Z_BINS, cap_per_bin=PAIR_CAP_PER_BIN)
            grouped = records_by_split(pairing.records)
            replay_records[comparison] = [row.to_dict() for split in SPLITS for row in grouped[split]]
            for split in SPLITS:
                _write_jsonl(output_root / "rows" / comparison / f"{split}.jsonl", (row.to_dict() for row in grouped[split]))
            audit["semantic_comparison"][comparison] = compare_semantic_rows(existing[comparison]["train"] + existing[comparison]["val"] + existing[comparison]["test"], replay_records[comparison], comparison=comparison)

        mixed_healthy = _make_mixed_healthy_records(source_rows["mixed"], source_rows, source_support, old_maps, source_hashes)
        mixed_brats, _mixed_brats_exclusions = make_brats_records(brats_rows, old_maps["brats21"])
        mixed_pairing = build_pairs(mixed_healthy, mixed_brats, comparison="mixed", seed=73, bins=Z_BINS, cap_per_bin=PAIR_CAP_PER_BIN)
        mixed_grouped = records_by_split(mixed_pairing.records)
        replay_records["mixed"] = [row.to_dict() for split in SPLITS for row in mixed_grouped[split]]
        for split in SPLITS:
            _write_jsonl(output_root / "rows" / "mixed" / f"{split}.jsonl", (row.to_dict() for row in mixed_grouped[split]))
        audit["semantic_comparison"]["mixed"] = compare_semantic_rows(existing["mixed"]["train"] + existing["mixed"]["val"] + existing["mixed"]["test"], replay_records["mixed"], comparison="mixed")

        all_existing = [row for rows in existing_flat.values() for row in rows]
        selected_brats = [row for row in all_existing if int(row.get("label", 1)) == 1]
        # This callback is replaced by the real MRIDataVolume reader below;
        # keeping the subject-index construction local ensures one full-volume
        # read per subject and no selected NPZ materialization.
        from andi_rewrite.data.datasets.brats import MRIDataVolume

        canonical_dataset = MRIDataVolume(
            csv_path=brats_csv,
            dataset_path=brats_root,
            image_size=128,
            modalities=["flair", "t1", "t2"],
            segmentation_suffix="seg",
            filename_separator="_",
            return_metadata=True,
            intensity_normalization="robust_iqr",
        )
        subjects = [str(value).strip() for value in canonical_dataset.df.iloc[:, 0].tolist()]
        subject_index = {subject: index for index, subject in enumerate(subjects)}

        def volume_reader(subject: str) -> tuple[Any, Any]:
            if subject not in subject_index:
                raise ReplayMismatch(f"selected BraTS subject absent from canonical CSV: {subject}")
            volume, model_mask, _metadata = canonical_dataset[subject_index[subject]]
            return volume, model_mask

        audit["brats_tensor_hash_audit"] = verify_selected_brats_tensor_hashes(selected_brats, volume_reader)
        audit["healthy_source_fingerprint_audit"] = verify_source_fingerprint_continuity(existing_flat, source_fingerprints)
        audit["healthy_source_tensor_audit"] = verify_healthy_source_tensors(
            replay_records,
            tensor_hash_ledger=healthy_hash_ledger,
        )
        source_ledger_status = str(healthy_hash_ledger_info.get("status") or "INCONCLUSIVE")
        observed_ledger_status = str(
            audit["healthy_source_tensor_audit"].get("tensor_hash_ledger_status") or "INCONCLUSIVE"
        )
        # The source reader can prove that replayed tensors agree with the
        # loaded entries, but it cannot repair a missing/mutated summary
        # binding.  Preserve the stricter loader status in the final audit.
        final_ledger_status = (
            observed_ledger_status if source_ledger_status == "PASS" else source_ledger_status
        )
        audit["healthy_tensor_hash_ledger"] = {
            "status": final_ledger_status,
            "expected_entries": audit["healthy_source_tensor_audit"].get("tensor_hash_ledger_expected_entries", 0),
            "checked_entries": audit["healthy_source_tensor_audit"].get("tensor_hash_ledger_checked_entries", 0),
            "unobserved_entries": audit["healthy_source_tensor_audit"].get("tensor_hash_ledger_unobserved_entries", 0),
            "failure_count": audit["healthy_source_tensor_audit"].get("tensor_hash_ledger_failure_count", 0),
            "path": healthy_hash_ledger_info.get("path"),
            "summary_path": healthy_hash_ledger_info.get("summary_path"),
            "summary_status": healthy_hash_ledger_info.get("summary_status"),
            "summary_sha256": healthy_hash_ledger_info.get("summary_sha256"),
            "actual_ledger_sha256": healthy_hash_ledger_info.get("actual_ledger_sha256"),
            "ledger_sha256_match": healthy_hash_ledger_info.get("ledger_sha256_match"),
            "ledger_path_match": healthy_hash_ledger_info.get("ledger_path_match"),
        }
        failures = [
            f"semantic_{name}"
            for name, value in audit["semantic_comparison"].items()
            if value.get("status") != "PASS"
        ]
        if audit["brats_tensor_hash_audit"].get("status") != "PASS":
            failures.append("brats_tensor_hash_mismatch")
        if audit["healthy_source_tensor_audit"].get("status") != "PASS":
            failures.append("healthy_source_tensor_mismatch")
        if audit["healthy_source_fingerprint_audit"].get("status") != "PASS":
            failures.append("healthy_source_fingerprint_inconclusive_or_mismatch")
        if audit["healthy_tensor_hash_ledger"].get("status") != "PASS":
            failures.append("healthy_tensor_hash_ledger_inconclusive_or_mismatch")
        audit["failures"] = failures
        audit["status"] = "PASS" if not failures else "FAIL_CLOSED"
        if audit["status"] == "PASS":
            audit["control_binding_status"] = "BOUND_AFTER_FULL_REPLAY_EQUIVALENCE"
    except Exception as exc:
        audit["failures"].append({"reason": "replay_exception", "type": type(exc).__name__, "detail": str(exc)})
        audit["status"] = "FAIL_CLOSED"
    finally:
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        audit["elapsed_seconds"] = time.perf_counter() - started
        audit["producer_snapshot_available"] = producer_audit.get("source_snapshot_available")
        _json_dump(output_root / "replay_audit.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent read-only replay/audit for model-grid v3")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--brats-root", type=Path, default=BRATS_ROOT)
    parser.add_argument("--brats-csv", type=Path, default=BRATS_CSV)
    args = parser.parse_args(argv)
    result = run_replay(args.source_root, args.output_root, brats_root=args.brats_root, brats_csv=args.brats_csv)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPARISONS",
    "EXPECTED_CANDIDATE_ROWS",
    "EXPECTED_CANDIDATE_SUBJECTS",
    "EXPECTED_MANIFEST_TOTAL",
    "EXPECTED_SOURCE_CSV_ROWS",
    "SEMANTIC_FIELDS",
    "compare_candidate_ledger",
    "compare_candidate_rows",
    "compare_semantic_rows",
    "run_replay",
    "semantic_projection",
    "tensor_sha256",
    "verify_healthy_tensor_hash_ledger",
    "verify_healthy_source_tensors",
    "verify_source_fingerprint_continuity",
    "verify_selected_brats_tensor_hashes",
]
