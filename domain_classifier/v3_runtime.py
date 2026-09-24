"""Frozen-input runtime helpers for the v3 domain-classifier protocol.

The v3 builder deliberately stops before model fitting.  This module is the
small boundary between its frozen JSONL manifests and a fitting entrypoint.
It reads each canonical model input at most once, checks the source identity
and tensor digest against the frozen evidence, and exposes datasets backed by
the resulting in-memory tensors.  Label permutations only copy manifest rows;
they never read an image again.

The helper is intentionally independent from the active matrix orchestrator
and from the Stage-A calibration script.  A caller can therefore use
``validate_and_materialize_v3_inputs`` for an observed cell or for any
secondary input projection, then pass the returned object to
``fit_cached_v3_cell``.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .runner import (
    DomainClassifierRunner,
    TrainConfig,
    _strip_prediction_arrays,
    extract_statistical_features,
    fit_statistical_logistic,
    prediction_rows,
    read_jsonl_manifest,
    subject_prediction_rows,
)
from .metrics import (
    bootstrap_subject_auc,
    compute_dataset_metrics,
    paired_heldout_swap_test,
)


SPLITS = ("train", "val", "test")
CANONICAL_MODALITIES = ("flair", "t1", "t2")
MODEL_SHAPE = (3, 128, 128)
DEFAULT_PRIMARY_REPLICATES = 199
DEFAULT_INIT_SEED = 73
DEFAULT_LABEL_STREAM_ROOT = 0xC2A57E
DEFAULT_HEALTHY_LEDGER_SHA256 = (
    "3e53e80024fbcff52c54b8cdf0b3420d75b7177d840d78c69af18bd06ab2cd06"
)
LARGE_SOURCE_BYTES = 64 * 1024 * 1024


class V3RuntimeError(RuntimeError):
    """Base exception for fail-closed v3 runtime checks."""


class V3InputValidationError(V3RuntimeError, ValueError):
    """Raised when frozen manifest/input evidence cannot be joined safely."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype != torch.float32:
        raise V3InputValidationError(
            f"tensor digest requires float32, found {tensor.dtype}"
        )
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_identity(row: Mapping[str, Any], *, split: str) -> tuple[Any, ...]:
    """Return a duplicate-detection identity for one frozen manifest row."""

    return (
        str(split),
        _text(row.get("source_dataset")),
        _text(row.get("source_split")),
        _text(row.get("source_key")),
        _text(row.get("participant_id")),
        _text(row.get("case_id")),
        int(row.get("z", -1)) if str(row.get("z", "")).strip() else -1,
        _text(row.get("pair_id")),
        int(row.get("label", -1)) if str(row.get("label", "")).strip() else -1,
    )


def _source_component(value: Any) -> str:
    text = _text(value)
    return text.split(":", 1)[1] if ":" in text else text


def canonical_source_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Resolve the source tensor key used by the frozen cache.

    Healthy Mixed rows expose a local key in ``source_key`` but point to the
    underlying standalone source through metadata/provenance.  Using the
    underlying tuple here is what lets a Mixed row reuse the same cached
    tensor as its standalone comparison while still retaining the Mixed local
    identity for membership checks.
    """

    source = _text(row.get("source_dataset")).lower()
    metadata = _mapping(row.get("metadata"))
    provenance = _mapping(row.get("provenance"))
    # Source identity is immutable under a null label swap.  Resolve whether
    # a row is a healthy or BraTS tensor from its source fields, rather than
    # from the mutable ``label`` value; otherwise flipping a healthy row to
    # label 1 would make the cache lookup fail before the fit starts.
    healthy_sources = {"fomo", "fomo45k", "mpi", "oasis", "oasis3", "mixed", "lmdb"}
    brats_sources = {"brats", "brats21", "brats2021"}
    if source in healthy_sources:
        if source == "mixed":
            dataset = _text(
                metadata.get("underlying_source_dataset")
                or provenance.get("underlying_source_dataset")
            ).lower()
            source_split = _text(
                metadata.get("underlying_source_split")
                or provenance.get("source_split_immutable")
                or row.get("source_split")
            )
            source_key = _text(
                metadata.get("underlying_source_key")
                or provenance.get("source_key_immutable")
            )
            if not dataset or not source_split or not source_key:
                raise V3InputValidationError(
                    "mixed healthy row is missing underlying source dataset/split/key"
                )
            return ("healthy", dataset, source_split, source_key)
        source_split = _text(
            provenance.get("source_split_immutable") or row.get("source_split")
        )
        source_key = _text(
            provenance.get("source_key_immutable") or row.get("source_key")
        )
        if not source or not source_split or not source_key:
            raise V3InputValidationError(
                "healthy row is missing source dataset/split/key"
            )
        return ("healthy", source, source_split, source_key)

    if source not in brats_sources:
        raise V3InputValidationError(
            f"label-1 row must use a BraTS source_dataset, found {source!r}"
        )
    participant = _text(
        metadata.get("source_participant_id")
        or provenance.get("source_participant_id")
        or row.get("participant_id")
    )
    if not participant:
        raise V3InputValidationError("BraTS row is missing source participant identity")
    return ("brats21", _source_component(participant), "z", str(int(row.get("z", -1))))


def _parse_label(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise V3InputValidationError(f"{field} must be binary 0/1") from exc
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise V3InputValidationError(f"{field} must be binary 0/1")
    return int(numeric)


def _parse_z(row: Mapping[str, Any]) -> int:
    try:
        value = int(row.get("z"))
    except (TypeError, ValueError) as exc:
        raise V3InputValidationError("manifest row z must be an integer") from exc
    if value < 0:
        raise V3InputValidationError("manifest row z must be non-negative")
    return value


def _resolve_build_and_manifest_root(
    manifest_root: str | Path,
    comparison: str | None,
    build_root: str | Path | None,
) -> tuple[Path, Path, str]:
    root = Path(manifest_root).resolve()
    name = _text(comparison).lower()
    if root.is_file():
        root = root.parent
    if not name:
        if root.name.lower() in {"fomo45k", "mpi", "oasis3", "mixed"}:
            name = root.name.lower()
        else:
            raise V3InputValidationError("comparison is required when manifest_root is not a cohort directory")
    if name not in {"fomo45k", "mpi", "oasis3", "mixed"}:
        raise V3InputValidationError(f"unsupported comparison {name!r}")
    if (root / "manifests" / name).is_dir():
        cohort_root = root / "manifests" / name
    elif (root / name).is_dir() and (root / name / "train.jsonl").is_file():
        cohort_root = root / name
    elif (root / "train.jsonl").is_file():
        cohort_root = root
    else:
        raise V3InputValidationError(f"cannot resolve frozen manifest directory from {root}")
    if build_root is None:
        if cohort_root.parent.name == "manifests":
            resolved_build = cohort_root.parent.parent
        else:
            resolved_build = root
    else:
        resolved_build = Path(build_root).resolve()
    return resolved_build, cohort_root.resolve(), name


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3InputValidationError(f"cannot read JSON artifact {path}") from exc


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl_manifest(path)
    except (OSError, ValueError) as exc:
        raise V3InputValidationError(f"cannot read manifest {path}") from exc
    return [dict(row) for row in rows]


def _row_summary_key(row: Mapping[str, Any], *, split: str) -> str:
    identity = _record_identity(row, split=split)
    return _canonical_json(identity)


def _expected_counts(build_root: Path, comparison: str) -> dict[str, Any]:
    """Read optional frozen count audits without treating absent fields as zero."""

    result: dict[str, Any] = {}
    summary = build_root / "build_summary.json"
    if summary.is_file():
        payload = _read_json(summary)
        comparisons = _mapping(payload).get("comparisons")
        item = _mapping(comparisons).get(comparison)
        if item:
            result.update({key: item.get(key) for key in ("records", "split_counts", "domain_counts") if key in item})
    audit = build_root / "audits" / f"{comparison}.json"
    if audit.is_file():
        payload = _read_json(audit)
        if "records" in payload:
            result["records"] = payload.get("records")
        pairing = _mapping(payload.get("pairing"))
        if isinstance(pairing.get("pairs"), list):
            result["pairs"] = len(pairing["pairs"])
        for key in ("split_counts", "domain_counts"):
            if key in payload:
                result[key] = payload.get(key)
    return result


def _validate_manifest_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    comparison: str,
    expected_counts: Mapping[str, Any] | None = None,
) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    failures: list[str] = []
    seen_rows: dict[tuple[Any, ...], tuple[str, int]] = {}
    source_keys: dict[tuple[str, str, str, str], list[tuple[str, int]]] = defaultdict(list)
    pair_participants: dict[str, dict[str, int]] = defaultdict(dict)
    pair_splits: dict[str, str] = {}
    pair_rows: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for split in SPLITS:
        rows = rows_by_split.get(split)
        if rows is None:
            failures.append(f"missing_split:{split}")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                failures.append(f"row_not_object:{split}:{index}")
                continue
            participant = _text(row.get("participant_id"))
            pair = _text(row.get("pair_id"))
            if not participant:
                failures.append(f"empty_participant_id:{split}:{index}")
            if not pair:
                failures.append(f"empty_pair_id:{split}:{index}")
            try:
                label = _parse_label(row.get("label"), f"label:{split}:{index}")
            except V3InputValidationError:
                failures.append(f"invalid_label:{split}:{index}")
                continue
            source_name = _text(row.get("source_dataset")).lower()
            is_brats_source = source_name in {"brats", "brats21", "brats2021"}
            if (label == 1) != is_brats_source:
                failures.append(f"label_source_mismatch:{split}:{index}")
            if _text(row.get("split")) and _text(row.get("split")) != split:
                failures.append(f"target_split_mismatch:{split}:{index}")
            model_shape = tuple(int(v) for v in row.get("model_shape", ())) if row.get("model_shape") is not None else ()
            if model_shape != MODEL_SHAPE:
                failures.append(f"manifest_shape:{split}:{index}")
            try:
                z = _parse_z(row)
                source_identity = canonical_source_identity(row)
            except (TypeError, ValueError, V3InputValidationError):
                failures.append(f"invalid_source_identity:{split}:{index}")
                continue
            identity = _record_identity(row, split=split)
            if identity in seen_rows:
                failures.append(f"duplicate_manifest_row:{split}:{index}")
            else:
                seen_rows[identity] = (split, index)
            source_keys[source_identity].append((split, index))
            if pair:
                pair_rows[pair] += 1
                previous_split = pair_splits.setdefault(pair, split)
                if previous_split != split:
                    failures.append(f"pair_crosses_splits:{pair}")
                previous_label = pair_participants[pair].get(participant)
                if previous_label is not None and previous_label != label:
                    failures.append(f"participant_conflicting_labels:{pair}:{participant}")
                pair_participants[pair][participant] = label
            domain_counts[str(row.get("domain") or ("healthy" if label == 0 else "brats21"))] += 1
            split_counts[split] += 1
            # Force the z value into the audit path; an invalid z is not a
            # recoverable metadata omission.
            _ = z
    for pair, participants in pair_participants.items():
        if len(participants) != 2:
            failures.append(f"pair_participant_count:{pair}")
        elif sorted(participants.values()) != [0, 1]:
            failures.append(f"pair_labels_not_01:{pair}")
    if expected_counts:
        if expected_counts.get("records") is not None and int(expected_counts["records"]) != sum(split_counts.values()):
            failures.append("manifest_record_count_differs_from_frozen_audit")
        expected_split = _mapping(expected_counts.get("split_counts"))
        for split in SPLITS:
            if expected_split.get(split) is not None and int(expected_split[split]) != split_counts[split]:
                failures.append(f"manifest_split_count_differs_from_frozen_audit:{split}")
        expected_pairs = expected_counts.get("pairs")
        if expected_pairs is not None and int(expected_pairs) != len(pair_participants):
            failures.append("manifest_pair_count_differs_from_frozen_audit")
    summary = {
        "rows": int(sum(split_counts.values())),
        "participants": int(len({p for members in pair_participants.values() for p in members})),
        "pairs": int(len(pair_participants)),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "duplicate_or_identity_failures": failures,
    }
    if failures:
        raise V3InputValidationError(
            "frozen manifest identity/label/pair validation failed: " + "; ".join(failures[:12])
        )
    row_locations = {
        _row_summary_key(row, split=split): (split, index)
        for split in SPLITS
        for index, row in enumerate(rows_by_split[split])
        for _ in [0]
    }
    return row_locations, summary


def _ledger_path_pair(build_root: Path, ledger_path: Path | None, summary_path: Path | None) -> tuple[Path, Path]:
    ledger = (
        Path(ledger_path).resolve()
        if ledger_path is not None
        else build_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl"
    )
    summary = (
        Path(summary_path).resolve()
        if summary_path is not None
        else build_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger_summary.json"
    )
    return ledger.resolve(), summary.resolve()


def _ledger_identity_from_membership(membership: Mapping[str, Any]) -> tuple[Any, ...]:
    dataset = _text(
        membership.get("underlying_source_dataset") or membership.get("local_source_dataset")
    ).lower()
    source_split = _text(
        membership.get("underlying_source_split") or membership.get("local_source_split")
    )
    source_key = _text(
        membership.get("underlying_source_key") or membership.get("local_source_key")
    )
    if not dataset or not source_split or not source_key:
        raise V3InputValidationError("healthy ledger membership is missing canonical source identity")
    return ("healthy", dataset, source_split, source_key)


def _membership_key(
    membership: Mapping[str, Any],
    *,
    comparison: str,
) -> tuple[Any, ...]:
    return (
        comparison,
        _ledger_identity_from_membership(membership),
        _text(membership.get("target_split") or membership.get("manifest_split")),
        _text(membership.get("participant_id")),
        _text(membership.get("pair_id")),
        int(membership.get("z", -1)),
        _text(membership.get("local_source_dataset")),
        _text(membership.get("local_source_split")),
        _text(membership.get("local_source_key")),
        _text(membership.get("mixed_local_key")),
    )


@dataclass(frozen=True)
class HealthyLedger:
    path: Path
    summary_path: Path
    summary_sha256: str
    entries: dict[tuple[str, str, str, str], Mapping[str, Any]]
    memberships: dict[tuple[Any, ...], Mapping[str, Any]]
    row_count: int
    ledger_sha256: str
    manifest_fingerprints: dict[str, str]


def load_healthy_ledger(
    build_root: str | Path,
    *,
    ledger_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    expected_sha256: str | None = DEFAULT_HEALTHY_LEDGER_SHA256,
) -> HealthyLedger:
    """Load and byte-bind the prelaunch healthy tensor ledger."""

    ledger, summary = _ledger_path_pair(Path(build_root).resolve(), Path(ledger_path) if ledger_path else None, Path(summary_path) if summary_path else None)
    if not ledger.is_file() or not summary.is_file():
        raise V3InputValidationError(f"healthy tensor ledger/summary missing: {ledger}; {summary}")
    summary_payload = _read_json(summary)
    if str(summary_payload.get("status", "")).upper() != "PASS":
        raise V3InputValidationError("healthy tensor ledger summary is not PASS")
    actual_sha = sha256_file(ledger)
    declared_sha = _text(summary_payload.get("ledger_sha256"))
    if not declared_sha or declared_sha != actual_sha:
        raise V3InputValidationError(
            f"healthy ledger bytes do not match summary ledger_sha256: {actual_sha} != {declared_sha}"
        )
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise V3InputValidationError(
            f"healthy ledger is not the frozen prelaunch ledger: {actual_sha} != {expected_sha256}"
        )
    entries: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    memberships: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    count = 0
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V3InputValidationError(f"invalid healthy ledger JSON at line {line_number}") from exc
            if not isinstance(value, Mapping):
                raise V3InputValidationError(f"healthy ledger row {line_number} is not an object")
            identity = (
                "healthy",
                _text(value.get("canonical_source_dataset")).lower(),
                _text(value.get("canonical_source_split")),
                _text(value.get("canonical_source_key")),
            )
            if not all(identity[1:]):
                raise V3InputValidationError(f"healthy ledger row {line_number} lacks canonical identity")
            if identity in entries:
                raise V3InputValidationError(f"duplicate healthy ledger identity {identity!r}")
            digest = _text(value.get("tensor_sha256"))
            if not digest:
                raise V3InputValidationError(f"healthy ledger row {line_number} lacks tensor_sha256")
            shape = tuple(int(v) for v in value.get("shape", ()))
            if shape != MODEL_SHAPE or _text(value.get("dtype")) != "float32":
                raise V3InputValidationError(f"healthy ledger row {line_number} has wrong shape/dtype")
            entries[identity] = value
            raw_memberships = value.get("memberships")
            if not isinstance(raw_memberships, list) or not raw_memberships:
                raise V3InputValidationError(f"healthy ledger row {line_number} has no memberships")
            for membership in raw_memberships:
                if not isinstance(membership, Mapping):
                    raise V3InputValidationError(f"healthy ledger row {line_number} has invalid membership")
                comparison = _text(membership.get("comparison")).lower()
                if not comparison:
                    raise V3InputValidationError(f"healthy ledger row {line_number} membership has no comparison")
                key = _membership_key(membership, comparison=comparison)
                if key in memberships:
                    raise V3InputValidationError(f"duplicate healthy ledger membership {key!r}")
                memberships[key] = membership
            count += 1
    declared_count = summary_payload.get("ledger_rows")
    if declared_count is not None and int(declared_count) != count:
        raise V3InputValidationError("healthy ledger row count differs from summary")
    manifest_fingerprints: dict[str, str] = {}
    raw_manifest_fingerprints = summary_payload.get("manifest_fingerprints")
    if raw_manifest_fingerprints is None:
        raw_manifest_fingerprints = summary_payload.get("manifest_identity")
    if isinstance(raw_manifest_fingerprints, Mapping):
        for key, value in raw_manifest_fingerprints.items():
            if isinstance(value, Mapping):
                digest = _text(value.get("sha256"))
            else:
                digest = _text(value)
            if digest:
                manifest_fingerprints[str(key)] = digest
    return HealthyLedger(
        path=ledger,
        summary_path=summary,
        summary_sha256=actual_sha,
        entries=entries,
        memberships=memberships,
        row_count=count,
        ledger_sha256=actual_sha,
        manifest_fingerprints=manifest_fingerprints,
    )


def _record_membership_key(row: Mapping[str, Any], *, comparison: str, split: str) -> tuple[Any, ...]:
    metadata = _mapping(row.get("metadata"))
    provenance = _mapping(row.get("provenance"))
    identity = canonical_source_identity(row)
    source = _text(row.get("source_dataset")).lower()
    local_dataset = source
    local_split = _text(row.get("source_split"))
    local_key = _text(row.get("source_key"))
    mixed_local = ""
    if source == "mixed":
        local_split = _text(provenance.get("source_split_immutable") or local_split)
        local_key = _text(provenance.get("mixed_local_key_immutable") or metadata.get("mixed_local_key") or local_key)
        mixed_local = _text(metadata.get("mixed_local_key") or provenance.get("mixed_local_key_immutable") or row.get("source_key"))
    return (
        comparison,
        identity,
        split,
        _text(row.get("participant_id")),
        _text(row.get("pair_id")),
        _parse_z(row),
        local_dataset,
        local_split,
        local_key,
        mixed_local,
    )


def _expected_digest_for_row(
    row: Mapping[str, Any],
    *,
    identity: tuple[str, str, str, str],
    comparison: str,
    split: str,
    ledger: HealthyLedger | None,
) -> tuple[str | None, dict[str, Any]]:
    if identity[0] == "healthy":
        if ledger is None:
            raise V3InputValidationError("healthy row requires a prelaunch tensor ledger")
        entry = ledger.entries.get(identity)
        if entry is None:
            raise V3InputValidationError(f"healthy source identity missing from ledger: {identity!r}")
        expected_membership = _record_membership_key(row, comparison=comparison, split=split)
        membership = ledger.memberships.get(expected_membership)
        if membership is None:
            raise V3InputValidationError(
                f"healthy manifest membership missing from scoped ledger: {expected_membership!r}"
            )
        digest = _text(entry.get("tensor_sha256"))
        return digest, {
            "source": "prelaunch_healthy_tensor_ledger",
            "ledger_identity": list(identity),
            "membership": dict(membership),
        }
    provenance = _mapping(row.get("provenance"))
    expected = _text(provenance.get("model_input_sha256"))
    if not expected:
        raise V3InputValidationError("BraTS row lacks provenance.model_input_sha256")
    return expected, {"source": "manifest.provenance.model_input_sha256"}


def _loader_from_factory(
    loader_factory: Callable[..., Any],
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> Any:
    try:
        return loader_factory(row, manifest_path=manifest_path)
    except TypeError:
        try:
            return loader_factory(row, manifest_path)
        except TypeError:
            return loader_factory(row)


def _production_loaders(manifest_paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        from andi_rewrite.data.domain_classifier import DomainClassifierDataset
    except ImportError:
        try:
            from data.domain_classifier import DomainClassifierDataset
        except ImportError as exc:  # pragma: no cover - environment contract
            raise V3InputValidationError("cannot import canonical data.domain_classifier reader") from exc
    return {
        split: DomainClassifierDataset(path, stage="final", image_size=128, return_metadata=False)
        for split, path in manifest_paths.items()
    }


def _read_tensor(
    row: Mapping[str, Any],
    *,
    split: str,
    index: int,
    manifest_path: Path,
    production_dataset: Any | None,
    loader_factory: Callable[..., Any] | None,
) -> Tensor:
    if loader_factory is not None:
        value = _loader_from_factory(loader_factory, row, manifest_path=manifest_path)
    elif production_dataset is not None:
        record = production_dataset.records[index]
        value = production_dataset.load_slice(record, stage="final")
    else:  # pragma: no cover - guarded by caller
        raise V3InputValidationError("no canonical tensor loader is available")
    if isinstance(value, (tuple, list)) and value and not isinstance(value, Tensor):
        value = value[0]
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.dtype != torch.float32:
        raise V3InputValidationError(
            f"{split}[{index}] canonical tensor dtype must be torch.float32, found {tensor.dtype}"
        )
    if tuple(tensor.shape) != MODEL_SHAPE:
        raise V3InputValidationError(
            f"{split}[{index}] canonical tensor shape must be {MODEL_SHAPE}, found {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise V3InputValidationError(f"{split}[{index}] canonical tensor contains NaN/Inf")
    return tensor.detach().cpu().contiguous()


def _find_recorded_fingerprint(path: Path, payload: Any) -> Mapping[str, Any] | None:
    """Find a preflight fingerprint for a large source file, if present."""

    if isinstance(payload, Mapping):
        if _text(payload.get("path")) and Path(str(payload.get("path"))).resolve() == path.resolve():
            if _text(payload.get("sha256")):
                return payload
        for value in payload.values():
            found = _find_recorded_fingerprint(path, value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_recorded_fingerprint(path, value)
            if found is not None:
                return found
    return None


def _fingerprint_path(path: Path, *, recorded_payload: Any | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise V3InputValidationError(f"frozen source/config/data file is missing: {resolved}")
    stat = resolved.stat()
    recorded = _find_recorded_fingerprint(resolved, recorded_payload) if recorded_payload is not None else None
    if recorded is not None:
        if int(recorded.get("size", stat.st_size)) != int(stat.st_size) or int(recorded.get("mtime_ns", stat.st_mtime_ns)) != int(stat.st_mtime_ns):
            raise V3InputValidationError(f"preflight fingerprint metadata drifted: {resolved}")
        return {
            "path": str(resolved),
            "status": "PASS",
            "sha256": _text(recorded.get("sha256")),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "hash_source": "recorded_prelaunch_fingerprint",
            "verification_scope": "prelaunch_digest_plus_metadata; selected_tensor_bytes_checked_separately",
        }
    if int(stat.st_size) >= LARGE_SOURCE_BYTES:
        # A source fingerprint may legitimately contain only a size/mtime
        # anchor for a large LMDB.  Re-hashing such a file on every cell would
        # turn a 112-cell matrix into repeated terabyte-scale I/O.  The v3
        # input binding still checks every selected canonical tensor byte
        # against the frozen ledger; this entry records the remaining source
        # continuity check explicitly instead of claiming a whole-file hash.
        return {
            "path": str(resolved),
            "status": "PASS",
            "sha256": "",
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "hash_source": "metadata_only_no_prelaunch_fingerprint",
            "verification_scope": "metadata_continuity_plus_selected_tensor_bytes",
        }
    return {
        "path": str(resolved),
        "status": "PASS",
        "sha256": sha256_file(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "hash_source": "bytes_read_before_fit",
        "verification_scope": "full_file_bytes",
    }


def _collect_freeze_paths(
    build_root: Path,
    manifest_paths: Mapping[str, Path],
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config_path: Path | None,
) -> list[Path]:
    paths: list[Path] = list(manifest_paths.values())
    for name in (
        "protocol.json",
        "training_protocol.json",
        "training_protocol_manifest.json",
        "training_protocol_amendment_20260917.json",
        "training_protocol_amendment_manifest.json",
        "source_fingerprints.json",
        "build_summary.json",
    ):
        candidate = build_root / name
        if candidate.is_file():
            paths.append(candidate)
    if config_path is not None:
        paths.append(config_path)
    for rows in rows_by_split.values():
        for row in rows:
            metadata = _mapping(row.get("metadata"))
            provenance = _mapping(row.get("provenance"))
            for key in ("lmdb_path", "source_lmdb_path", "selected_cache_path"):
                value = metadata.get(key) or provenance.get(key)
                if value:
                    candidate = Path(str(value))
                    if candidate.is_file():
                        paths.append(candidate)
                    elif candidate.is_dir():
                        for name in ("data.mdb", "normalization.json"):
                            nested = candidate / name
                            if nested.is_file():
                                paths.append(nested)
    unique: dict[str, Path] = {str(path.resolve()).lower(): path.resolve() for path in paths}
    return [unique[key] for key in sorted(unique)]


def _source_freeze(
    paths: Sequence[Path],
    *,
    build_root: Path,
) -> dict[str, Any]:
    recorded_payload: Any | None = None
    source_fingerprints = build_root / "source_fingerprints.json"
    if source_fingerprints.is_file():
        recorded_payload = _read_json(source_fingerprints)
    return {
        "status": "PASS",
        "files": [_fingerprint_path(path, recorded_payload=recorded_payload) for path in paths],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "manifest/protocol/config/data bytes or prelaunch byte fingerprints checked before fit",
    }


def verify_source_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck a persisted source/config/data freeze before resuming a run."""

    failures: list[str] = []
    checked: list[dict[str, Any]] = []
    for item in freeze.get("files", []):
        if not isinstance(item, Mapping):
            failures.append("invalid_freeze_entry")
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            failures.append(f"missing:{path}")
            continue
        stat = path.stat()
        if int(item.get("size", -1)) != int(stat.st_size) or int(item.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            failures.append(f"metadata_drift:{path}")
            continue
        expected = _text(item.get("sha256"))
        hash_source = _text(item.get("hash_source"))
        if hash_source == "metadata_only_no_prelaunch_fingerprint" or (
            hash_source == "recorded_prelaunch_fingerprint"
            and int(item.get("size", 0)) >= LARGE_SOURCE_BYTES
        ):
            # For large immutable sources the authoritative prelaunch digest,
            # when available, is retained as an anchor.  Rechecking size and
            # mtime here avoids reading multi-gigabyte LMDB files for every
            # resumed cell; selected tensor bytes are independently rehashed
            # by validate_and_materialize_v3_inputs.
            actual = None
        else:
            actual = sha256_file(path)
            if expected != actual:
                failures.append(f"sha256_drift:{path}")
        checked.append({
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_source": hash_source,
        })
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checked": checked}
    if failures:
        raise V3InputValidationError("frozen source/config/data bytes drifted: " + "; ".join(failures[:8]))
    return result


class CachedV3Dataset(Dataset[dict[str, Any]]):
    """Dataset view over immutable canonical tensors and mutable labels."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tensors: Sequence[Tensor],
        *,
        modalities: Sequence[str] = CANONICAL_MODALITIES,
    ) -> None:
        if len(records) != len(tensors):
            raise V3InputValidationError("cached records and tensors have different lengths")
        normalized_modalities = tuple(str(value).lower() for value in modalities)
        if not normalized_modalities or any(value not in CANONICAL_MODALITIES for value in normalized_modalities):
            raise V3InputValidationError("modalities must be a non-empty subset of flair/t1/t2")
        if len(set(normalized_modalities)) != len(normalized_modalities):
            raise V3InputValidationError("modalities must not contain duplicates")
        self.records = [dict(row) for row in records]
        self.tensors = [value.detach().cpu().contiguous() for value in tensors]
        self.modalities = normalized_modalities
        indices = [CANONICAL_MODALITIES.index(value) for value in normalized_modalities]
        self._indices = tuple(indices)
        for index, tensor in enumerate(self.tensors):
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != MODEL_SHAPE:
                raise V3InputValidationError(f"cached tensor {index} is not canonical float32 {MODEL_SHAPE}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[int(index)]
        tensor = self.tensors[int(index)]
        image = tensor[list(self._indices)] if self._indices != (0, 1, 2) else tensor
        return {
            "image": image,
            "label": _parse_label(row.get("label"), "cached row label"),
            "participant_id": row.get("participant_id"),
            "pair_id": row.get("pair_id"),
            "case_id": row.get("case_id", int(index)),
            "record_index": int(index),
        }

    def projected(self, modalities: Sequence[str]) -> "CachedV3Dataset":
        return CachedV3Dataset(self.records, self.tensors, modalities=modalities)


@dataclass
class V3CachedInputs:
    comparison: str
    build_root: Path
    manifest_root: Path
    manifest_paths: dict[str, Path]
    records_by_split: dict[str, list[dict[str, Any]]]
    datasets: dict[str, CachedV3Dataset]
    tensors_by_identity: dict[tuple[str, str, str, str], Tensor]
    tensor_digests_by_identity: dict[tuple[str, str, str, str], str]
    input_binding_audit: dict[str, Any]
    source_freeze: dict[str, Any]
    ledger: HealthyLedger
    manifest_fingerprints: dict[str, str]

    def projected(self, modalities: Sequence[str]) -> dict[str, CachedV3Dataset]:
        """Return modality views backed by the same canonical tensor objects."""

        return {split: dataset.projected(modalities) for split, dataset in self.datasets.items()}

    def with_label_rows(self, records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, CachedV3Dataset]:
        """Create a label-mutated view without touching canonical tensor cache."""

        output: dict[str, CachedV3Dataset] = {}
        for split in SPLITS:
            rows = [dict(row) for row in records_by_split[split]]
            tensors: list[Tensor] = []
            for row in rows:
                identity = canonical_source_identity(row)
                tensor = self.tensors_by_identity.get(identity)
                if tensor is None:
                    raise V3InputValidationError(f"permuted row is absent from cached identity map: {identity!r}")
                tensors.append(tensor)
            output[split] = CachedV3Dataset(rows, tensors, modalities=CANONICAL_MODALITIES)
        return output


def validate_and_materialize_v3_inputs(
    manifest_root: str | Path,
    *,
    comparison: str | None = None,
    build_root: str | Path | None = None,
    ledger_path: str | Path | None = None,
    ledger_summary_path: str | Path | None = None,
    expected_ledger_sha256: str | None = DEFAULT_HEALTHY_LEDGER_SHA256,
    loader_factory: Callable[..., Any] | None = None,
    config_path: str | Path | None = None,
    expected_manifest_fingerprints: Mapping[str, str] | None = None,
) -> V3CachedInputs:
    """Validate frozen v3 rows and materialize one canonical tensor per source key.

    ``loader_factory`` is only a test/integration injection point.  Production
    callers use the canonical ``data.domain_classifier.DomainClassifierDataset``
    reader, which preserves the existing LMDB and BraTS cache code paths.
    """

    resolved_build, cohort_root, name = _resolve_build_and_manifest_root(manifest_root, comparison, build_root)
    manifest_paths = {split: cohort_root / f"{split}.jsonl" for split in SPLITS}
    missing = [str(path) for path in manifest_paths.values() if not path.is_file()]
    if missing:
        raise V3InputValidationError("missing frozen manifest(s): " + "; ".join(missing))
    rows_by_split = {split: _read_manifest(path) for split, path in manifest_paths.items()}
    _, row_summary = _validate_manifest_rows(
        rows_by_split,
        comparison=name,
        expected_counts=_expected_counts(resolved_build, name),
    )
    ledger = load_healthy_ledger(
        resolved_build,
        ledger_path=ledger_path,
        summary_path=ledger_summary_path,
        expected_sha256=expected_ledger_sha256,
    )
    manifest_anchor = dict(expected_manifest_fingerprints or ledger.manifest_fingerprints)
    manifest_anchor_rows: dict[str, Any] = {}
    for split, path in manifest_paths.items():
        anchor_key = f"{name}/{split}"
        actual_manifest_sha = sha256_file(path)
        expected_manifest_sha = _text(manifest_anchor.get(anchor_key))
        if expected_manifest_sha and actual_manifest_sha != expected_manifest_sha:
            raise V3InputValidationError(
                f"manifest bytes do not match frozen prelaunch fingerprint for {anchor_key}: "
                f"{actual_manifest_sha} != {expected_manifest_sha}"
            )
        manifest_anchor_rows[anchor_key] = {
            "path": str(path.resolve()),
            "sha256": actual_manifest_sha,
            "expected_sha256": expected_manifest_sha or None,
            "status": "PASS" if expected_manifest_sha else "INCONCLUSIVE_MISSING_PRELAUNCH_FINGERPRINT",
        }
    source_freeze_paths = _collect_freeze_paths(
        resolved_build,
        manifest_paths,
        rows_by_split,
        config_path=Path(config_path).resolve() if config_path is not None else None,
    )
    source_freeze = _source_freeze(source_freeze_paths, build_root=resolved_build)
    production_datasets = None if loader_factory is not None else _production_loaders(manifest_paths)
    tensors_by_identity: dict[tuple[str, str, str, str], Tensor] = {}
    digests_by_identity: dict[tuple[str, str, str, str], str] = {}
    expected_by_identity: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    binding_rows: list[dict[str, Any]] = []
    scoped_ledger_keys: set[tuple[Any, ...]] = set()
    for split in SPLITS:
        for index, row in enumerate(rows_by_split[split]):
            identity = canonical_source_identity(row)
            expected_digest, expected_detail = _expected_digest_for_row(
                row,
                identity=identity,
                comparison=name,
                split=split,
                ledger=ledger,
            )
            if expected_digest is None:
                raise V3InputValidationError(f"no expected tensor digest for {split}[{index}]")
            expected_by_identity[identity].add(expected_digest)
            if len(expected_by_identity[identity]) > 1:
                raise V3InputValidationError(f"conflicting expected tensor hashes for identity {identity!r}")
            if identity not in tensors_by_identity:
                tensor = _read_tensor(
                    row,
                    split=split,
                    index=index,
                    manifest_path=manifest_paths[split],
                    production_dataset=production_datasets.get(split) if production_datasets else None,
                    loader_factory=loader_factory,
                )
                actual_digest = _tensor_sha256(tensor)
                tensors_by_identity[identity] = tensor
                digests_by_identity[identity] = actual_digest
            actual_digest = digests_by_identity[identity]
            if actual_digest != expected_digest:
                raise V3InputValidationError(
                    f"tensor hash mismatch at {split}[{index}] identity={identity!r}: {actual_digest} != {expected_digest}"
                )
            if identity[0] == "healthy":
                membership_key = _record_membership_key(row, comparison=name, split=split)
                scoped_ledger_keys.add(membership_key)
            binding_rows.append({
                "split": split,
                "index": int(index),
                "identity": list(identity),
                "participant_id": _text(row.get("participant_id")),
                "pair_id": _text(row.get("pair_id")),
                "label": _parse_label(row.get("label"), "manifest label"),
                "z": _parse_z(row),
                "tensor_sha256": actual_digest,
                "expected_tensor_sha256": expected_digest,
                "expected_source": expected_detail.get("source"),
            })
    scoped_ledger_entries = {
        key for key, value in ledger.memberships.items()
        if _text(value.get("comparison")).lower() == name
    }
    missing_memberships = sorted(scoped_ledger_entries - scoped_ledger_keys, key=repr)
    extra_memberships = sorted(scoped_ledger_keys - scoped_ledger_entries, key=repr)
    if missing_memberships or extra_memberships:
        raise V3InputValidationError(
            f"scoped healthy ledger membership mismatch: missing={missing_memberships[:3]!r}, extra={extra_memberships[:3]!r}"
        )
    datasets = {
        split: CachedV3Dataset(
            rows_by_split[split],
            [tensors_by_identity[canonical_source_identity(row)] for row in rows_by_split[split]],
        )
        for split in SPLITS
    }
    input_binding_audit = {
        "status": "PASS",
        "schema_version": 1,
        "comparison": name,
        "manifest_root": str(cohort_root),
        "manifest_identity": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows_by_split[split]),
            }
            for split, path in manifest_paths.items()
        },
        "manifest_prelaunch_anchor": {
            "status": "PASS" if all(item["expected_sha256"] for item in manifest_anchor_rows.values()) else "INCONCLUSIVE_MISSING_PRELAUNCH_FINGERPRINT",
            "rows": manifest_anchor_rows,
            "source": "explicit_expected_manifest_fingerprints_or_healthy_ledger_summary",
        },
        "rows_checked": len(binding_rows),
        "unique_canonical_tensors": len(tensors_by_identity),
        "shape": list(MODEL_SHAPE),
        "dtype": "torch.float32",
        "all_three_modalities_materialized_once": True,
        "healthy_ledger": {
            "path": str(ledger.path),
            "summary_path": str(ledger.summary_path),
            "sha256": ledger.ledger_sha256,
            "rows": ledger.row_count,
            "scoped_memberships": len(scoped_ledger_keys),
        },
        "brats_provenance_hashes_checked": sum(1 for row in binding_rows if row["label"] == 1),
        "manifest_summary": row_summary,
        "rows": binding_rows,
    }
    manifest_fingerprints = {
        split: str(input_binding_audit["manifest_identity"][split]["sha256"])
        for split in SPLITS
    }
    return V3CachedInputs(
        comparison=name,
        build_root=resolved_build,
        manifest_root=cohort_root,
        manifest_paths=manifest_paths,
        records_by_split=rows_by_split,
        datasets=datasets,
        tensors_by_identity=tensors_by_identity,
        tensor_digests_by_identity=digests_by_identity,
        input_binding_audit=input_binding_audit,
        source_freeze=source_freeze,
        ledger=ledger,
        manifest_fingerprints=manifest_fingerprints,
    )


def _label_stream_seeds(index: int, *, root: int = DEFAULT_LABEL_STREAM_ROOT) -> dict[str, int]:
    parent = np.random.SeedSequence([int(root), int(index)])
    children = parent.spawn(len(SPLITS))
    values = {
        split: int(child.generate_state(1, dtype=np.uint32)[0])
        for split, child in zip(SPLITS, children)
    }
    if len(set(values.values())) != len(SPLITS):
        raise V3RuntimeError("v3 label streams collided")
    return values


def permute_cached_v3_labels(
    cached: V3CachedInputs,
    *,
    index: int,
    label_stream_root: int = DEFAULT_LABEL_STREAM_ROOT,
) -> tuple[dict[str, CachedV3Dataset], dict[str, int], dict[str, Any]]:
    """Swap each complete pair independently in train/val/test."""

    streams = _label_stream_seeds(int(index), root=int(label_stream_root))
    output_rows: dict[str, list[dict[str, Any]]] = {}
    swap_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = [dict(row) for row in cached.records_by_split[split]]
        by_pair: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            pair = _text(row.get("pair_id"))
            participant = _text(row.get("participant_id"))
            if not pair or not participant:
                raise V3InputValidationError("null permutation requires non-empty pair/participant IDs")
            label = _parse_label(row.get("label"), "null source label")
            previous = by_pair[pair].get(participant)
            if previous is not None and previous != label:
                raise V3InputValidationError(f"pair has conflicting labels: {pair!r}/{participant!r}")
            by_pair[pair][participant] = label
        mapping: dict[tuple[str, str], int] = {}
        rng = np.random.default_rng(int(streams[split]))
        swaps = 0
        for pair in sorted(by_pair):
            members = by_pair[pair]
            if len(members) != 2 or sorted(members.values()) != [0, 1]:
                raise V3InputValidationError(f"pair is not a complete 0/1 pair: {split}/{pair}")
            do_swap = bool(rng.integers(0, 2))
            swaps += int(do_swap)
            for participant, old in members.items():
                mapping[(pair, participant)] = 1 - old if do_swap else old
        for row in rows:
            key = (_text(row.get("pair_id")), _text(row.get("participant_id")))
            row["label"] = mapping[key]
        output_rows[split] = rows
        swap_counts[split] = swaps
    datasets = cached.with_label_rows(output_rows)
    return datasets, streams, {
        "index": int(index),
        "label_stream_root": int(label_stream_root),
        "swap_counts": swap_counts,
        "label_scope": "train_val_test_complete_pair_swaps",
    }


def _dataset_arrays(dataset: CachedV3Dataset, modalities: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected = dataset.projected(modalities)
    images = np.stack([projected[index]["image"].numpy() for index in range(len(projected))], axis=0)
    labels = np.asarray([_parse_label(row.get("label"), "fit label") for row in projected.records], dtype=np.int64)
    participants = np.asarray([row.get("participant_id") for row in projected.records], dtype=object)
    pairs = np.asarray([row.get("pair_id") for row in projected.records], dtype=object)
    cases = np.asarray([row.get("case_id", index) for index, row in enumerate(projected.records)], dtype=object)
    return images, labels, participants, pairs, cases


def _fit_logistic_cached(
    cached: V3CachedInputs,
    *,
    config: TrainConfig,
    seed: int,
) -> dict[str, Any]:
    train_images, train_labels, train_participants, _train_pairs, _train_cases = _dataset_arrays(cached.datasets["train"], config.modalities)
    val_images, val_labels, val_participants, val_pairs, val_cases = _dataset_arrays(cached.datasets["val"], config.modalities)
    test_images, test_labels, test_participants, test_pairs, test_cases = _dataset_arrays(cached.datasets["test"], config.modalities)
    train_features = extract_statistical_features(train_images, background_value=-1.0)
    val_features = extract_statistical_features(val_images, background_value=-1.0)
    test_features = extract_statistical_features(test_images, background_value=-1.0)
    c_value = float(getattr(config, "logistic_C", 1.0))
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        fitted = fit_statistical_logistic(
            train_features,
            train_labels,
            val_features,
            seed=int(seed),
            c=c_value,
            train_participant_ids=train_participants,
        )
    val_scores = np.asarray(fitted["scores"], dtype=np.float64)
    val_metrics = compute_dataset_metrics(
        val_labels,
        val_scores,
        participant_ids=val_participants,
        pair_ids=val_pairs,
        threshold=float(config.threshold),
        subject_method=config.subject_method,
    )
    # The train-only scaler and classifier are fit exactly once.  Re-fitting
    # on the same training rows merely to obtain test scores can look harmless
    # for deterministic LBFGS, but it creates two model objects and makes the
    # persisted coefficients ambiguous.  Transform the held-out test rows with
    # the already-fitted train scaler and use that same classifier.
    scaler = fitted.get("scaler")
    model = fitted.get("model")
    if scaler is None or model is None or not hasattr(scaler, "transform") or not hasattr(model, "predict_proba"):
        raise V3RuntimeError("statistical logistic fit did not return a reusable train-only model")
    test_scores = np.asarray(
        model.predict_proba(scaler.transform(test_features))[:, 1],
        dtype=np.float64,
    )
    test_metrics = compute_dataset_metrics(
        test_labels,
        test_scores,
        participant_ids=test_participants,
        pair_ids=test_pairs,
        threshold=float(config.threshold),
        subject_method=config.subject_method,
    )
    test_statistics: dict[str, Any] = {}
    test_subject_predictions = test_metrics.get("subject_predictions")
    if isinstance(test_subject_predictions, Mapping):
        subject_labels = test_subject_predictions.get("labels")
        subject_scores = test_subject_predictions.get("scores")
        subject_pair_ids = test_subject_predictions.get("pair_ids")
        pair_values = (
            subject_pair_ids
            if subject_pair_ids is not None
            and all(
                value is not None and str(value).strip() != ""
                for value in np.asarray(subject_pair_ids, dtype=object).tolist()
            )
            else None
        )
        if int(config.bootstrap_replicates) > 0:
            test_statistics["subject_bootstrap"] = bootstrap_subject_auc(
                subject_labels,
                subject_scores,
                pair_ids=pair_values,
                n_bootstrap=int(config.bootstrap_replicates),
                seed=int(seed),
            )
        if pair_values is not None and int(config.swap_replicates) > 0:
            test_statistics["heldout_pair_swap"] = paired_heldout_swap_test(
                subject_labels,
                subject_scores,
                pair_values,
                n_swaps=int(config.swap_replicates),
                seed=int(seed),
            )
    model = fitted["model"]
    n_iter = np.asarray(getattr(model, "n_iter_", []), dtype=np.int64).reshape(-1)
    if n_iter.size == 0 or not np.isfinite(n_iter.astype(np.float64)).all() or np.any(n_iter <= 0):
        raise V3RuntimeError("statistical logistic model did not expose finite positive n_iter_")
    max_iter = int(getattr(model, "max_iter", 0))
    convergence = {
        "n_iter": [int(value) for value in n_iter.tolist()],
        "max_iter": max_iter,
        "converged": bool(np.all(n_iter < max_iter)),
        "warnings": [f"{type(item.message).__name__}: {item.message}" for item in fit_warnings],
    }
    return {
        "seed": int(seed),
        "split_seed": int(config.split_seed),
        "config": asdict(config),
        "device": "cpu",
        "best_epoch": 0,
        "epochs_completed": 0,
        "history": [],
        "validation": _strip_prediction_arrays(val_metrics),
        "test": _strip_prediction_arrays(test_metrics),
        "test_statistics": test_statistics,
        "train_final": None,
        "validation_predictions": prediction_rows(val_metrics),
        "validation_subject_predictions": subject_prediction_rows(val_metrics),
        "test_predictions": prediction_rows(test_metrics),
        "test_subject_predictions": subject_prediction_rows(test_metrics),
        "train_final_predictions": [],
        "train_final_subject_predictions": [],
        "model": {"class": "LogisticRegression", "train_only_standardizer": True},
        "state_dict": None,
        "logistic": {
            "C": c_value,
            "train_only_standardizer": True,
            "single_train_fit_for_val_and_test": True,
            "train_feature_mean": _json_safe(fitted.get("train_feature_mean")),
            "train_feature_scale": _json_safe(fitted.get("train_feature_scale")),
            "model_coef": _json_safe(fitted["model"].coef_),
            "model_intercept": _json_safe(fitted["model"].intercept_),
            "convergence": convergence,
        },
    }


def fit_cached_v3_cell(
    cached: V3CachedInputs,
    *,
    config: TrainConfig,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit one cell using only the already materialized canonical tensors."""

    actual_seed = int(config.seed if seed is None else seed)
    if tuple(config.modalities) not in {
        ("flair",),
        ("t1",),
        ("t2",),
        CANONICAL_MODALITIES,
    }:
        raise V3RuntimeError(f"unsupported v3 modality projection: {config.modalities!r}")
    model_name = str(config.model).strip().lower().replace("-", "_")
    if model_name in {"statistical_logistic", "logistic", "statistical"}:
        return _fit_logistic_cached(cached, config=config, seed=actual_seed)
    if model_name not in {"small_cnn", "cnn", "small", "resnet", "resnet18", "resnet_18"}:
        raise V3RuntimeError(f"unsupported cached v3 classifier {config.model!r}")
    datasets = cached.projected(config.modalities)
    runner = DomainClassifierRunner(config)
    return runner.train_one_seed(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        seed=actual_seed,
    )


def set_single_thread_runtime(*, strict: bool = True) -> dict[str, Any]:
    """Apply and verify the v3 single-thread contract.

    PyTorch only permits changing its inter-op pool before parallel work has
    started.  Silently swallowing that ``RuntimeError`` would leave a run
    claiming the contract while executing with a different pool, so the
    default is fail-closed.  ``strict=False`` is available to callers that
    need an audit snapshot while preparing a process; the returned values still
    expose the actual settings and the caller can decide whether to proceed.
    """

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    errors: list[str] = []
    try:
        torch.set_num_threads(1)
    except RuntimeError as exc:
        errors.append(f"torch.set_num_threads: {exc}")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        errors.append(f"torch.set_num_interop_threads: {exc}")
    try:
        actual_threads = int(torch.get_num_threads())
        actual_interop = int(torch.get_num_interop_threads())
    except (AttributeError, RuntimeError) as exc:  # pragma: no cover - old torch only
        raise V3RuntimeError(f"cannot read PyTorch thread settings: {exc}") from exc
    snapshot = {
        "status": "PASS" if not errors and actual_threads == 1 and actual_interop == 1 else "FAIL",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": actual_threads,
        "torch_num_interop_threads": actual_interop,
        "errors": errors,
    }
    if strict and snapshot["status"] != "PASS":
        raise V3RuntimeError(
            "v3 single-thread contract could not be verified: "
            + _canonical_json(snapshot)
        )
    return snapshot


def run_fingerprint(
    *,
    comparison: str,
    config: TrainConfig,
    cached: V3CachedInputs,
    code_paths: Sequence[Path] = (),
    protocol_id: str = "domain_classifier_primary_null_v3",
) -> str:
    freeze_identity = {
        key: value for key, value in cached.source_freeze.items()
        if key != "created_at_utc"
    }
    payload = {
        "protocol_id": protocol_id,
        "comparison": comparison,
        "config": asdict(config),
        "manifest_sha256": cached.manifest_fingerprints,
        "source_freeze": freeze_identity,
        "ledger_sha256": cached.ledger.ledger_sha256,
        "code_sha256": {str(path.resolve()): sha256_file(path) for path in sorted(code_paths, key=str) if path.is_file()},
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _prediction_rows_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_v3_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "CANONICAL_MODALITIES",
    "CachedV3Dataset",
    "DEFAULT_HEALTHY_LEDGER_SHA256",
    "DEFAULT_INIT_SEED",
    "DEFAULT_LABEL_STREAM_ROOT",
    "DEFAULT_PRIMARY_REPLICATES",
    "MODEL_SHAPE",
    "SPLITS",
    "V3CachedInputs",
    "V3InputValidationError",
    "V3RuntimeError",
    "canonical_source_identity",
    "fit_cached_v3_cell",
    "load_healthy_ledger",
    "permute_cached_v3_labels",
    "run_fingerprint",
    "set_single_thread_runtime",
    "sha256_file",
    "validate_and_materialize_v3_inputs",
    "verify_source_freeze",
    "write_v3_json",
]
