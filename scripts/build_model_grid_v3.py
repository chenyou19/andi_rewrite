"""Build the frozen, metadata-audited v3 domain-classifier model-grid.

The command is deliberately a *build-only* operation.  It reads the existing
healthy LMDB tensors in their canonical ``[3,128,128]`` float32 form to apply
the preregistered support criterion, reads the existing selected BraTS cache
and segmentation metadata, and writes new manifests under
``outputs/diagnostics/domain_classifier/model_grid_v3``.  It never writes a
source dataset and never starts model training.

Logical train/validation/test assignment is participant-level.  The old
FOMO45K and BraTS21 seed-73 maps are copied exactly; new cohort IDs are
assigned by the same deterministic seed-73 allocator.  For Mixed, FOMO IDs
retain their old map entries and MPI/OASIS IDs are a deterministic extension.
The source split/key is retained separately so local keys are resolved in the
correct split before any metadata is attached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    participant_split_map,
)
from andi_rewrite.data.domain_classifier.records import (  # noqa: E402
    MODEL_SHAPE,
    SliceRecord,
    namespaced_participant,
    z_bin_for,
)


OUTPUT_ROOT = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_v3"
SURVEY_PATH = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_protocol_survey" / "survey.json"
OLD_SPLIT_MAP_PATH = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "split_maps.json"
OLD_MANIFEST_ROOT = REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "manifests" / "fomo45k"
BRATS_ROOT = Path(r"C:\ML\data\BraTS_2021")
BRATS_CSV = REPO_ROOT / "splits" / "BraTS21" / "scans_train.csv"
ATLAS_DEPTH = 155
ATLAS_SHAPE = (240, 240, ATLAS_DEPTH)
SUPPORT_THRESHOLD = 0.10
SUPPORT_EPS = 1e-6
Z_BINS = 20
PAIR_CAP_PER_BIN = 2

COHORT_ROOTS = {
    "fomo45k": REPO_ROOT / "outputs" / "datasets" / "fomo45k_sri24_robust_iqr",
    "mpi": REPO_ROOT / "outputs" / "datasets" / "mpi_sri24_robust_iqr",
    "oasis3": REPO_ROOT / "outputs" / "datasets" / "oasis3_sri24_robust_iqr",
    "mixed": REPO_ROOT / "outputs" / "datasets" / "mpi_oasis3_fomo45k_sri24_robust_iqr",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(row)
    return rows


def sidecar_hash(path: Path) -> str:
    return sha256(path)


def read_fomo_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        path = root / "manifests" / f"{split}_entries.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = dict(row)
                item["z"] = int(item["z"])
                item["source_split"] = str(item.get("split") or split)
                item["source_dataset"] = "fomo45k"
                item["source_key"] = str(item["key"])
                rows.append(item)
    return rows


def read_jsonl_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    path = root / "entries.jsonl"
    rows = jsonl_rows(path)
    for item in rows:
        item["z"] = int(item["z"])
        item["source_split"] = str(item.get("split") or "")
        item["source_dataset"] = dataset
        item["source_key"] = str(item.get("key") or "")
    return rows


def read_mixed_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split, path in (("train", root / "source_entries.jsonl"), ("val", root / "validation" / "source_entries.jsonl")):
        for item in jsonl_rows(path):
            item = dict(item)
            item["source_split"] = str(item.get("source_split") or source_split)
            item["mixed_local_key"] = str(item.get("key") or "")
            item["underlying_source_dataset"] = str(item.get("source_dataset") or "")
            item["source_key"] = str(item.get("source_key") or "")
            rows.append(item)
    return rows


def source_lmdb_path(dataset: str, source_split: str) -> Path:
    root = COHORT_ROOTS[dataset]
    if dataset == "mixed":
        return root / (Path("train") if source_split == "train" else Path("validation") / "val")
    return root / source_split


def source_sidecar_path(dataset: str, source_split: str) -> Path:
    root = COHORT_ROOTS[dataset]
    if dataset == "fomo45k":
        return root / "manifests" / f"{source_split}_entries.csv"
    if dataset == "mixed":
        return root / (Path("source_entries.jsonl") if source_split == "train" else Path("validation") / "source_entries.jsonl")
    return root / "entries.jsonl"


def canonical_source_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "fomo45k": read_fomo_rows(COHORT_ROOTS["fomo45k"]),
        "mpi": read_jsonl_rows(COHORT_ROOTS["mpi"], "mpi"),
        "oasis3": read_jsonl_rows(COHORT_ROOTS["oasis3"], "oasis3"),
        "mixed": read_mixed_rows(COHORT_ROOTS["mixed"]),
    }


def source_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_split"]), str(row["source_key"]))
        if key in result:
            raise ValueError(f"duplicate split-local source key {key}")
        result[key] = dict(row)
    return result


def read_old_split_maps() -> dict[str, dict[str, str]]:
    payload = json.loads(OLD_SPLIT_MAP_PATH.read_text(encoding="utf-8"))
    return {str(name): {str(k): str(v) for k, v in values.items()} for name, values in payload.items()}


def read_old_fomo_registered_paths() -> dict[tuple[str, ...], dict[str, str]]:
    """Read FOMO registered volumes and index them by participant/case.

    The historical manifest contains only selected z rows.  A v3 support
    selection may choose a different z from the same primary volume, so a
    split-local ``(source_split, source_key)`` lookup alone is incomplete.
    Indexing the immutable raw paths by participant, case, and session lets
    every newly selected z reuse the same verified three modality volumes.
    """

    output: dict[tuple[str, ...], dict[str, str]] = {}
    for split in SPLITS:
        path = OLD_MANIFEST_ROOT / f"{split}.jsonl"
        if not path.is_file():
            continue
        for row in jsonl_rows(path):
            if str(row.get("source_dataset")) != "fomo45k":
                continue
            key = (str(row.get("source_split") or split), str(row.get("source_key") or ""))
            paths = row.get("registered_paths", {})
            if isinstance(paths, Mapping) and paths:
                normalized = {str(name): str(value) for name, value in paths.items()}
                output[key] = normalized
                participant = str(row.get("participant_id") or "")
                case_id = str(row.get("case_id") or "")
                session_id = str(row.get("session_id") or "")
                if participant and case_id:
                    participant_keys = {participant}
                    if ":" in participant:
                        participant_keys.add(participant.split(":", 1)[1])
                    for participant_key in participant_keys:
                        case_key = ("participant_case", participant_key, case_id, session_id)
                        prior = output.get(case_key)
                        if prior is not None and prior != normalized:
                            raise ValueError(f"Conflicting registered paths for {case_key!r}")
                        output[case_key] = normalized
    return output


def registered_paths_for_row(
    registered_paths: Mapping[tuple[str, ...], Mapping[str, str]],
    *,
    source_split: str,
    source_key: str,
    participant_id: str,
    case_id: str,
    session_id: str,
) -> dict[str, str]:
    """Resolve immutable FOMO paths for any z in the selected primary case."""

    keys = [
        (source_split, source_key),
        ("participant_case", participant_id, case_id, session_id),
    ]
    if ":" in participant_id:
        keys.append(("participant_case", participant_id.split(":", 1)[1], case_id, session_id))
    for key in keys:
        value = registered_paths.get(key)
        if isinstance(value, Mapping) and value:
            return {str(name): str(path) for name, path in value.items()}
    return {}


def assign_map(participants: Sequence[str], *, seed: int, existing: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    ids = sorted({str(value) for value in participants})
    existing_map = {str(k): str(v) for k, v in (existing or {}).items() if str(k) in ids}
    unknown = [value for value in ids if value not in existing_map]
    if existing_map and unknown:
        extension = participant_split_map(unknown, seed=seed)
        output = {**existing_map, **extension}
        source = "preserved_existing_plus_seed73_extension"
    elif existing_map:
        output = dict(existing_map)
        source = "preserved_existing_seed73_map"
    else:
        output = participant_split_map(ids, seed=seed)
        source = "derived_seed73_map"
    if set(output) != set(ids) or any(value not in SPLITS for value in output.values()):
        raise ValueError("participant map does not cover exactly the requested IDs")
    return output, {
        "seed": seed,
        "source": source,
        "participant_count": len(ids),
        "split_counts": dict(Counter(output.values())),
        "preserved_count": len(existing_map),
        "extended_count": len(unknown),
    }


def support_fraction(tensor: torch.Tensor) -> tuple[float, float, float]:
    value = torch.as_tensor(tensor, dtype=torch.float32)
    if tuple(value.shape) != MODEL_SHAPE:
        raise ValueError(f"canonical LMDB tensor must be {MODEL_SHAPE}, found {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("canonical LMDB tensor contains NaN/Inf")
    support = (torch.abs(value + 1.0) > SUPPORT_EPS).to(torch.float64).mean(dim=(1, 2))
    return tuple(float(x) for x in support.tolist())  # type: ignore[return-value]


def materialize_healthy_support(rows: Sequence[Mapping[str, Any]], dataset: str) -> tuple[dict[tuple[str, str], tuple[float, float, float]], dict[str, Any]]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["source_split"])].append(row)
    support: dict[tuple[str, str], tuple[float, float, float]] = {}
    summary: dict[str, Any] = {"dataset": dataset, "split_counts": {}, "max_source_z": {}, "read_errors": []}
    for split, split_rows in sorted(by_split.items()):
        path = source_lmdb_path(dataset, split)
        if not path.is_dir():
            raise FileNotFoundError(path)
        dataset_reader = LMDBSliceDataset(path, image_size=None)
        try:
            for row in split_rows:
                key = str(row["source_key"])
                index = int(key)
                if index < 0 or index >= len(dataset_reader):
                    raise IndexError(f"{dataset}:{split}:{key} outside LMDB length {len(dataset_reader)}")
                fractions = support_fraction(dataset_reader[index])
                support[(split, key)] = fractions
        finally:
            transaction = getattr(dataset_reader, "txn", None)
            if transaction is not None:
                transaction.abort()
                dataset_reader.txn = None
            environment = getattr(dataset_reader, "env", None)
            if environment is not None:
                environment.close()
                dataset_reader.env = None
        summary["split_counts"][split] = len(split_rows)
        summary["max_source_z"][split] = max(int(row["z"]) for row in split_rows)
    summary["support_threshold"] = SUPPORT_THRESHOLD
    summary["support_expression"] = f"abs(x + 1) > {SUPPORT_EPS:g}"
    summary["denominator_pixels"] = 128 * 128
    return support, summary


def _raw_brats_id(value: str) -> str:
    text = str(value).strip()
    return text.split(":", 1)[1] if ":" in text else text


def validate_brats_csv_map(
    csv_path: Path,
    brats_map: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the complete fixed CSV/map identity before opening volumes."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"BraTS CSV is empty: {csv_path}")
    if "BraTS21ID" not in rows[0]:
        raise ValueError(f"BraTS CSV must contain BraTS21ID: {csv_path}")
    raw_ids = [str(row.get("BraTS21ID", "")).strip() for row in rows]
    empty_ids = [index for index, value in enumerate(raw_ids) if not value]
    duplicate_ids = sorted(value for value, count in Counter(raw_ids).items() if value and count > 1)
    map_raw_ids = [_raw_brats_id(value) for value in brats_map]
    csv_set = set(raw_ids)
    map_set = set(map_raw_ids)
    bad_split = sorted({str(value) for value in brats_map.values()} - set(SPLITS))
    missing_map_ids = sorted(csv_set - map_set)
    extra_map_ids = sorted(map_set - csv_set)
    if empty_ids or duplicate_ids or bad_split or missing_map_ids or extra_map_ids or len(map_raw_ids) != len(set(map_raw_ids)):
        raise ValueError(
            "BraTS CSV/map identity validation failed: "
            + json.dumps(
                {
                    "empty_row_indices": empty_ids[:20],
                    "duplicate_ids": duplicate_ids[:20],
                    "bad_split": bad_split,
                    "missing_map_ids": missing_map_ids[:20],
                    "extra_map_ids": extra_map_ids[:20],
                    "csv_rows": len(rows),
                    "map_entries": len(brats_map),
                },
                sort_keys=True,
            )
        )
    return {
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": sha256(csv_path),
        "csv_rows": len(rows),
        "csv_unique_ids": len(csv_set),
        "map_entries": len(brats_map),
        "map_unique_raw_ids": len(map_set),
        "id_set_equal": True,
        "empty_ids": 0,
        "duplicate_ids": 0,
        "bad_split": [],
        "split_counts": dict(Counter(str(value) for value in brats_map.values())),
    }


def _candidate_pool_overlap(
    candidate_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_keys = {(str(row["participant_id"]), int(row["z"])) for row in candidate_rows}
    legacy_keys = {(str(row.get("participant_id")), int(row.get("z", -1))) for row in legacy_rows}
    return {
        "candidate_keys": len(candidate_keys),
        "legacy_keys": len(legacy_keys),
        "overlap_keys": len(candidate_keys.intersection(legacy_keys)),
        "candidate_keys_outside_legacy": len(candidate_keys - legacy_keys),
        "legacy_keys_missing_from_candidate": len(legacy_keys - candidate_keys),
    }


def load_brats_candidate_rows(
    old_maps: Mapping[str, str],
    output_root: Path,
    *,
    brats_root: Path = BRATS_ROOT,
    csv_path: Path = BRATS_CSV,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a full BraTS candidate inventory before z matching.

    The historical FOMO manifest is retained only for an overlap audit.  It
    is never used to define candidates.  Every CSV subject is loaded once via
    the canonical ANDi ``MRIDataVolume`` path; native-zero z values are then
    checked for model-grid mask zero and the three-channel support criterion.
    Only metadata rows are retained here.  Selected tensors are materialized
    after pairing, which bounds disk and memory use.
    """

    csv_path = csv_path.resolve()
    brats_root = brats_root.resolve()
    map_audit = validate_brats_csv_map(csv_path, old_maps)
    from andi_rewrite.scripts.prepare_domain_classifier import load_brats_records

    try:
        native_records, native_inventory = load_brats_records(brats_root, csv_path)
    except Exception as exc:
        failure_inventory = {
            "schema_version": 4,
            "status": "FAIL",
            "inventory_revision": "model_grid_v3_fullcandidate_v1",
            "source": {"root": str(brats_root), "csv": map_audit, "canonical_loader": "prepare_domain_classifier.load_brats_records"},
            "errors": [{"reason": "native_inventory_scan_error", "type": type(exc).__name__, "detail": str(exc)}],
            "old_pool_is_not_candidate_source": True,
        }
        json_dump(output_root / "brats_candidate_inventory.json", failure_inventory)
        raise RuntimeError(f"Native BraTS candidate inventory failed; see {output_root / 'brats_candidate_inventory.json'}") from exc
    native_by_participant: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in native_records:
        native_by_participant[record.participant_id].append(record)
    expected_participants = {str(key) for key in old_maps}
    observed_participants = set(native_by_participant)
    missing_participants = sorted(expected_participants - observed_participants)
    unexpected_participants = sorted(observed_participants - expected_participants)

    # The CSV/map identity is checked above.  Missing/extra native participants
    # are preserved in the inventory and cause this build to fail closed after
    # writing its error audit.
    errors: list[dict[str, Any]] = []
    native_exclusion_subjects = {
        namespaced_participant("brats21", str(item.get("subject", "")))
        for item in native_inventory.get("exclusions", [])
        if str(item.get("subject", "")).strip()
    }
    no_native_zero_participants = sorted(set(missing_participants) - native_exclusion_subjects)
    if no_native_zero_participants:
        # A valid subject can have no native tumor-free axial slice.  Keep it
        # explicit in the inventory, but do not call it a missing participant.
        pass
    if unexpected_participants:
        errors.append({"reason": "native_inventory_unexpected_participants", "participants": unexpected_participants})
    for item in native_inventory.get("exclusions", []):
        errors.append({"reason": "native_inventory_exclusion", **dict(item)})

    try:
        from andi_rewrite.data.datasets.brats import MRIDataVolume
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise ImportError("Full BraTS model-grid inventory requires MRIDataVolume") from exc
    dataset = MRIDataVolume(
        csv_path=csv_path,
        dataset_path=brats_root,
        image_size=128,
        modalities=["flair", "t1", "t2"],
        segmentation_suffix="seg",
        filename_separator="_",
        return_metadata=True,
        intensity_normalization="robust_iqr",
    )
    source_subjects = [str(value).strip() for value in dataset.df.iloc[:, 0].tolist()]
    if len(source_subjects) != len(set(source_subjects)) or set(source_subjects) != {_raw_brats_id(key) for key in old_maps}:
        errors.append({
            "reason": "mri_dataset_subject_identity_mismatch",
            "dataset_count": len(source_subjects),
            "dataset_unique_count": len(set(source_subjects)),
            "dataset_missing": sorted({_raw_brats_id(key) for key in old_maps} - set(source_subjects))[:20],
            "dataset_extra": sorted(set(source_subjects) - {_raw_brats_id(key) for key in old_maps})[:20],
        })
    subject_index = {subject: index for index, subject in enumerate(source_subjects)}

    protocol_path = output_root / "protocol.json"
    protocol_sha = sha256(protocol_path) if protocol_path.is_file() else ""
    map_sha = hashlib.sha256(json.dumps(dict(sorted(old_maps.items())), sort_keys=True).encode("utf-8")).hexdigest()
    resume_fingerprint = hashlib.sha256(json.dumps({"csv_sha256": map_audit["csv_sha256"], "map_sha256": map_sha, "protocol_sha256": protocol_sha, "inventory_revision": "model_grid_v3_fullcandidate_v1"}, sort_keys=True).encode("utf-8")).hexdigest()
    candidate_rows_path = output_root / "brats_candidate_rows.jsonl"
    completion_ledger_path = output_root / "brats_candidate_completion.jsonl"
    completed_by_subject: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    subject_audit: list[dict[str, Any]] = []
    if candidate_rows_path.is_file() or completion_ledger_path.is_file():
        if not candidate_rows_path.is_file() or not completion_ledger_path.is_file():
            raise RuntimeError("BraTS candidate resume requires both rows and completion ledger")
        for event in jsonl_rows(completion_ledger_path):
            if str(event.get("resume_fingerprint")) != resume_fingerprint:
                raise RuntimeError("BraTS candidate resume fingerprint mismatch; use a fresh revision root")
            subject = str(event.get("subject", ""))
            if not subject or subject in completed_by_subject:
                raise RuntimeError(f"invalid/duplicate BraTS completion ledger subject: {subject!r}")
            completed_by_subject[subject] = dict(event)
            subject_audit.append(dict(event.get("audit") or {}))
            if event.get("status") == "ERROR":
                errors.append({"subject": subject, "reason": "resumed_subject_error", "detail": event.get("audit", {}).get("errors", [])})
        rows = jsonl_rows(candidate_rows_path)
        for row in rows:
            row["_support_fraction"] = tuple(float(value) for value in row.get("_support_fraction", ()))
        if len(subject_audit) > len(source_subjects):
            raise RuntimeError("BraTS completion ledger has more subjects than CSV")
        ledger_subject_set = set(completed_by_subject)
        row_subject_set = {_raw_brats_id(str(row.get("participant_id", ""))) for row in rows}
        orphan_subjects = sorted(row_subject_set - ledger_subject_set)
        if orphan_subjects:
            raise RuntimeError(f"BraTS candidate rows contain subjects without an atomic completion event: {orphan_subjects[:20]}")
        expected_completed_rows = {
            str(event["subject"]): int((event.get("audit") or {}).get("support_pass_slices", 0))
            for event in completed_by_subject.values()
        }
        actual_completed_rows = Counter(_raw_brats_id(str(row.get("participant_id", ""))) for row in rows)
        count_mismatches = [
            {"subject": subject, "expected": expected_count, "actual": int(actual_completed_rows.get(subject, 0))}
            for subject, expected_count in expected_completed_rows.items()
            if int(actual_completed_rows.get(subject, 0)) != expected_count
        ]
        if count_mismatches:
            raise RuntimeError(f"BraTS candidate row/completion counts mismatch: {count_mismatches[:10]}")
    rows_handle = candidate_rows_path.open("a" if completed_by_subject else "w", encoding="utf-8", newline="\n")
    ledger_handle = completion_ledger_path.open("a" if completed_by_subject else "w", encoding="utf-8", newline="\n")

    def finish_subject(audit: dict[str, Any], status: str) -> None:
        audit["status"] = status
        subject_audit.append(audit)
        rows_handle.flush()
        ledger_handle.write(json.dumps({"schema_version": 1, "resume_fingerprint": resume_fingerprint, "subject": audit["subject"], "status": status, "audit": audit}, sort_keys=True) + "\n")
        ledger_handle.flush()

    started = time.perf_counter()
    for subject_number, raw_subject in enumerate(source_subjects, start=1):
        participant = namespaced_participant("brats21", raw_subject)
        if raw_subject in completed_by_subject:
            if subject_number == 1 or subject_number % 50 == 0 or subject_number == len(source_subjects):
                print(json.dumps({"event": "brats_full_candidate_resume_progress", "subjects_done": subject_number, "subjects_total": len(source_subjects), "candidate_rows": len(rows), "errors": len(errors), "elapsed_seconds": time.perf_counter() - started}), flush=True)
            continue
        native_rows = sorted(native_by_participant.get(participant, []), key=lambda row: row.z)
        item_audit: dict[str, Any] = {
            "subject": raw_subject,
            "participant_id": participant,
            "logical_split": old_maps.get(participant),
            "native_zero_slices": len(native_rows),
            "model_mask_zero_slices": 0,
            "model_mask_nonzero_slices": 0,
            "support_pass_slices": 0,
            "support_fail_slices": 0,
            "errors": [],
        }
        if participant in native_exclusion_subjects:
            item_audit["status"] = "NATIVE_INVENTORY_EXCLUDED"
            finish_subject(item_audit, "ERROR")
            continue
        if not native_rows:
            item_audit["status"] = "NO_ELIGIBLE_NATIVE_ZERO"
            finish_subject(item_audit, "COMPLETE")
            continue
        if participant not in old_maps:
            item_audit["errors"].append("missing_split_map")
            errors.append({"subject": raw_subject, "reason": "missing_split_map"})
            finish_subject(item_audit, "ERROR")
            continue
        try:
            volume, model_mask, metadata = dataset[subject_index[raw_subject]]
            if tuple(volume.shape) != (3, 128, 128, ATLAS_DEPTH):
                raise ValueError(f"unexpected_volume_shape={tuple(volume.shape)}")
            if tuple(model_mask.shape) != (128, 128, ATLAS_DEPTH):
                raise ValueError(f"unexpected_model_mask_shape={tuple(model_mask.shape)}")
            if not bool(torch.isfinite(volume).all()):
                raise ValueError("nonfinite_model_volume")
            image_paths = dict(metadata.get("input_paths", {}))
            seg_path = str(metadata.get("segmentation_path", ""))
            for native_row in native_rows:
                z = int(native_row.z)
                model_voxels = int(model_mask[..., z].to(torch.int64).sum().item())
                if model_voxels != 0:
                    item_audit["model_mask_nonzero_slices"] += 1
                    continue
                item_audit["model_mask_zero_slices"] += 1
                fractions = support_fraction(volume[..., z])
                if any(value + 1e-12 < SUPPORT_THRESHOLD for value in fractions):
                    item_audit["support_fail_slices"] += 1
                    continue
                item_audit["support_pass_slices"] += 1
                candidate_row = {
                        "participant_id": participant,
                        "source_participant_id": raw_subject,
                        "session_id": raw_subject,
                        "case_id": raw_subject,
                        "z": z,
                        "z_norm": float(z) / float(ATLAS_DEPTH - 1),
                        "z_bin": z_bin_for(float(z) / float(ATLAS_DEPTH - 1), Z_BINS),
                        "source_dataset": "brats21",
                        "source_key": f"{raw_subject}:{z}",
                        "source_split": "train",
                        "logical_split": str(old_maps[participant]),
                        "image_paths": image_paths,
                        "seg_path": seg_path,
                        "native_seg_voxels": 0,
                        "model_mask_voxels": 0,
                        "_support_fraction": fractions,
                        "metadata": {
                            "dataset_path": str(brats_root),
                            "source_participant_id": raw_subject,
                            "native_segmentation_checked": True,
                            "model_mask_checked": True,
                            "full_candidate_inventory": True,
                            "candidate_inventory_revision": "model_grid_v3_fullcandidate_v1",
                            "normalization": "robust_iqr",
                            "normalize_input": False,
                            "model_grid_support_fraction": list(fractions),
                            "model_grid_support_expression": f"abs(x + 1) > {SUPPORT_EPS:g}",
                            "model_grid_support_threshold": SUPPORT_THRESHOLD,
                            "model_grid_support_denominator": 128 * 128,
                        },
                        "provenance": {
                            "brats_csv": str(csv_path),
                            "brats_csv_sha256": map_audit["csv_sha256"],
                            "dataset_root": str(brats_root),
                            "native_segmentation_voxels_verified": 0,
                            "model_grid_mask_voxels_verified": 0,
                            "z_norm_denominator": ATLAS_DEPTH,
                            "z_norm_basis": "verified_atlas_depth_155_not_max_available_index",
                            "build_protocol": "model_grid_v3_fullcandidate_v1",
                        },
                    }
                rows.append(candidate_row)
                rows_handle.write(json.dumps(candidate_row, sort_keys=True) + "\n")
            del volume, model_mask, metadata
        except Exception as exc:  # write a complete per-subject error audit
            item_audit["errors"].append(f"{type(exc).__name__}: {exc}")
            errors.append({"subject": raw_subject, "reason": "model_grid_scan_error", "detail": str(exc)})
        finish_subject(item_audit, "ERROR" if item_audit["errors"] else "COMPLETE")
        if subject_number == 1 or subject_number % 25 == 0 or subject_number == len(source_subjects):
            print(json.dumps({
                "event": "brats_full_candidate_progress",
                "subjects_done": subject_number,
                "subjects_total": len(source_subjects),
                "candidate_rows": len(rows),
                "errors": len(errors),
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    rows_handle.close()
    ledger_handle.close()
    try:
        legacy_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            path = OLD_MANIFEST_ROOT / f"{split}.jsonl"
            if path.is_file():
                legacy_rows.extend(row for row in jsonl_rows(path) if str(row.get("source_dataset")) == "brats21")
        legacy_overlap = _candidate_pool_overlap(rows, legacy_rows)
    except Exception as exc:
        legacy_overlap = {"status": "ERROR", "detail": str(exc)}
    ledger_subject_ids = [str(item.get("subject", "")) for item in subject_audit]
    row_keys = [(str(row.get("participant_id", "")), int(row.get("z", -1))) for row in rows]
    row_subject_counts = Counter(_raw_brats_id(participant) for participant, _z in row_keys)
    expected_row_counts = {str(item.get("subject")): int(item.get("support_pass_slices", 0)) for item in subject_audit}
    ledger_failures: list[dict[str, Any]] = []
    if len(ledger_subject_ids) != len(source_subjects) or len(set(ledger_subject_ids)) != len(ledger_subject_ids) or set(ledger_subject_ids) != set(source_subjects):
        ledger_failures.append({"reason": "completion_ledger_subject_set_mismatch", "ledger_count": len(ledger_subject_ids), "csv_count": len(source_subjects), "ledger_duplicates": len(ledger_subject_ids) - len(set(ledger_subject_ids)), "missing": sorted(set(source_subjects) - set(ledger_subject_ids))[:20], "extra": sorted(set(ledger_subject_ids) - set(source_subjects))[:20]})
    duplicate_row_keys = sorted(key for key, count in Counter(row_keys).items() if count > 1)
    if duplicate_row_keys:
        ledger_failures.append({"reason": "duplicate_candidate_participant_z", "count": len(duplicate_row_keys), "examples": duplicate_row_keys[:20]})
    for subject in source_subjects:
        if row_subject_counts.get(subject, 0) != expected_row_counts.get(subject, 0):
            ledger_failures.append({"reason": "candidate_row_count_mismatch", "subject": subject, "rows": row_subject_counts.get(subject, 0), "support_pass_slices": expected_row_counts.get(subject, 0)})
    if any(_raw_brats_id(participant) not in set(source_subjects) for participant, _z in row_keys):
        ledger_failures.append({"reason": "candidate_row_subject_not_in_csv"})
    candidate_artifact_audit = {
        "status": "PASS" if not ledger_failures else "FAIL",
        "csv_subject_count": len(source_subjects),
        "ledger_subject_count": len(ledger_subject_ids),
        "candidate_row_count": len(rows),
        "candidate_unique_participant_z": len(set(row_keys)),
        "failures": ledger_failures[:100],
        "failure_count": len(ledger_failures),
    }
    if ledger_failures:
        errors.extend(ledger_failures)
    inventory = {
        "schema_version": 4,
        "status": "FAIL" if errors else "PASS",
        "full_candidate_coverage": {
            "status": "FAIL" if errors else "PASS",
            "verified": not bool(errors) and candidate_artifact_audit["status"] == "PASS",
            "csv_rows": map_audit["csv_rows"],
            "processed_subjects": len(subject_audit),
            "candidate_rows": len(rows),
            "candidate_source": "full_csv_native_zero_then_model_mask_zero_then_model_grid_support",
            "old_selected_pool_used_as_source": False,
        },
        "inventory_revision": "model_grid_v3_fullcandidate_v1",
        "source": {
            "root": str(brats_root),
            "csv": map_audit,
            "csv_rows_verified": map_audit["csv_rows"],
            "mri_dataset_rows": len(source_subjects),
            "all_csv_ids_processed": len(subject_audit) == map_audit["csv_rows"],
            "source_modalities": ["flair", "t1", "t2"],
            "canonical_loader": "data.datasets.brats.MRIDataVolume",
            "intensity_normalization": "robust_iqr",
            "model_shape": list(MODEL_SHAPE),
            "atlas_shape": list(ATLAS_SHAPE),
        },
        "split_map": {
            "entries": len(old_maps),
            "split_counts": dict(Counter(str(value) for value in old_maps.values())),
            "source": str(OLD_SPLIT_MAP_PATH),
            "id_set_equal": map_audit["id_set_equal"],
        },
        "native_inventory": {
            "records": len(native_records),
            "participants": len(native_by_participant),
            "no_native_zero_participants": no_native_zero_participants,
            "no_native_zero_count": len(no_native_zero_participants),
            "exclusions": list(native_inventory.get("exclusions", [])),
        },
        "model_grid_counts": {
            "subjects": len(subject_audit),
            "native_zero_slices": sum(int(item["native_zero_slices"]) for item in subject_audit),
            "model_mask_zero_slices": sum(int(item["model_mask_zero_slices"]) for item in subject_audit),
            "model_mask_nonzero_slices": sum(int(item["model_mask_nonzero_slices"]) for item in subject_audit),
            "support_pass_slices": sum(int(item["support_pass_slices"]) for item in subject_audit),
            "support_fail_slices": sum(int(item["support_fail_slices"]) for item in subject_audit),
            "candidate_rows": len(rows),
            "candidate_participants": len({str(row["participant_id"]) for row in rows}),
            "candidate_split_counts": dict(Counter(str(row["logical_split"]) for row in rows)),
        },
        "candidate_pool_comparison": legacy_overlap,
        "candidate_artifact_audit": candidate_artifact_audit,
        "subject_audit": subject_audit,
        "errors": errors,
        "old_pool_is_not_candidate_source": True,
        "selected_tensor_cache": "materialized_after_pairing_only",
        "candidate_rows_jsonl": {
            "path": str(candidate_rows_path),
            "sha256": sha256(candidate_rows_path),
            "rows": len(rows),
            "completion_ledger": str(completion_ledger_path),
            "completion_ledger_sha256": sha256(completion_ledger_path),
            "resume_fingerprint": resume_fingerprint,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary_inventory = output_root / "brats_candidate_inventory.json.tmp"
    json_dump(temporary_inventory, inventory)
    temporary_inventory.replace(output_root / "brats_candidate_inventory.json")
    if errors:
        raise RuntimeError(f"Full BraTS candidate inventory failed; see {output_root / 'brats_candidate_inventory.json'}")
    return rows, inventory


def make_healthy_records(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    support: Mapping[tuple[str, str], tuple[float, float, float]],
    split_map: Mapping[str, str],
    source_hashes: Mapping[tuple[str, str], str],
    registered_paths: Mapping[tuple[str, ...], Mapping[str, str]] | None = None,
) -> tuple[list[SliceRecord], list[dict[str, Any]]]:
    # Primary case is fixed before pairing: lexicographically first case and
    # session for each participant, across the full source metadata table.
    by_participant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_id = str(row["participant_id"])
        pid = namespaced_participant(dataset, raw_id)
        by_participant[pid].append(row)
    primary: dict[str, tuple[str, str]] = {}
    for pid, participant_rows in by_participant.items():
        cases = sorted((str(row.get("case_id", "")), str(row.get("session_id", ""))) for row in participant_rows)
        primary[pid] = cases[0]
    records: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        raw_id = str(row["participant_id"])
        pid = namespaced_participant(dataset, raw_id)
        case_id = str(row.get("case_id", ""))
        session_id = str(row.get("session_id", ""))
        if (case_id, session_id) != primary[pid]:
            continue
        source_split = str(row["source_split"])
        source_key = str(row["source_key"])
        fractions = support[(source_split, source_key)]
        if any(value + 1e-12 < SUPPORT_THRESHOLD for value in fractions):
            exclusions.append({"participant_id": pid, "source_split": source_split, "source_key": source_key, "z": int(row["z"]), "reason": "model_grid_support_below_threshold", "support_fraction": list(fractions)})
            continue
        z = int(row["z"])
        z_norm = float(z) / float(ATLAS_DEPTH - 1)
        logical_split = str(split_map[pid])
        lmdb_path = source_lmdb_path(dataset if dataset != "mixed" else "mixed", source_split)
        source_sidecar = source_sidecar_path(dataset, source_split)
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
            "source_participant_id": raw_id,
            "source_case_id": case_id,
            "source_session_id": session_id,
            "source_key_namespace": "local_per_source_split",
            "source_sidecar_sha256": source_hashes[(dataset, source_split)],
        }
        provenance = {
            "source_sidecar": str(source_sidecar),
            "source_lmdb_path": str(lmdb_path),
            "source_split_immutable": source_split,
            "source_key_immutable": source_key,
            "z_norm_denominator": ATLAS_DEPTH,
            "z_norm_basis": "verified_atlas_depth_155_not_max_available_lmdb_index",
            "build_protocol": "model_grid_v3",
        }
        records.append(SliceRecord(
            split=logical_split,
            label=0,
            domain=dataset,
            participant_id=pid,
            session_id=session_id,
            case_id=case_id,
            z=z,
            z_norm=z_norm,
            z_bin=z_bin_for(z_norm, Z_BINS),
            source_dataset=dataset,
            source_key=source_key,
            source_split=source_split,
            registered_paths=registered_paths_for_row(
                registered_paths or {},
                source_split=source_split,
                source_key=source_key,
                participant_id=pid,
                case_id=case_id,
                session_id=session_id,
            ),
            model_shape=MODEL_SHAPE,
            geometry_shape=ATLAS_SHAPE,
            stage="final",
            foreground_fraction=fractions,
            provenance=provenance,
            metadata=metadata,
        ))
    return records, exclusions


def make_brats_records(rows: Sequence[Mapping[str, Any]], split_map: Mapping[str, str]) -> tuple[list[SliceRecord], list[dict[str, Any]]]:
    records: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        participant = str(row["participant_id"])
        metadata_old = dict(row.get("metadata", {}) or {})
        fractions = tuple(float(x) for x in row["_support_fraction"])
        if any(value + 1e-12 < SUPPORT_THRESHOLD for value in fractions):
            exclusions.append({"participant_id": participant, "z": int(row["z"]), "reason": "model_grid_support_below_threshold", "support_fraction": list(fractions)})
            continue
        native = row.get("native_seg_voxels")
        model = row.get("model_mask_voxels")
        native_value = -1 if native is None else int(native)
        model_value = -1 if model is None else int(model)
        if native_value != 0 or model_value != 0 or not bool(metadata_old.get("native_segmentation_checked")) or not bool(metadata_old.get("model_mask_checked")):
            exclusions.append({"participant_id": participant, "z": int(row["z"]), "reason": "strict_brats_lesion_zero_metadata_check"})
            continue
        z = int(row["z"])
        z_norm = float(z) / float(ATLAS_DEPTH - 1)
        old_provenance = dict(row.get("provenance") or {})
        old_metadata = dict(metadata_old)
        old_metadata.update({
            "model_grid_support_fraction": list(fractions),
            "model_grid_support_expression": f"abs(x + 1) > {SUPPORT_EPS:g}",
            "model_grid_support_threshold": SUPPORT_THRESHOLD,
            "model_grid_support_denominator": 128 * 128,
            "geometry_proxy": "atlas_model_grid_shape_only",
            "geometry_proxy_is_native_brain_mask": False,
            "native_segmentation_checked": True,
            "model_mask_checked": True,
            "source_selection": (
                "full_brat_candidate_inventory_then_pair_assignment_rebuilt_v3"
                if bool(old_metadata.get("full_candidate_inventory"))
                else "read_only_exact_tensor_source_from_historical_cache; pair_assignment_rebuilt_v3"
            ),
        })
        provenance = {
            **old_provenance,
            "historical_candidate_manifest": str(OLD_MANIFEST_ROOT) if not bool(old_metadata.get("full_candidate_inventory")) else None,
            "candidate_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
            "z_norm_denominator": ATLAS_DEPTH,
            "z_norm_basis": "verified_atlas_depth_155_not_max_available_index",
            "build_protocol": "model_grid_v3_fullcandidate_v1" if bool(old_metadata.get("full_candidate_inventory")) else "model_grid_v3",
            "native_segmentation_voxels_verified_by_prior_readonly_audit": 0,
            "model_grid_mask_voxels_verified_by_prior_readonly_audit": 0,
        }
        image_paths = dict(row.get("image_paths") or {})
        records.append(SliceRecord(
            split=str(split_map[participant]),
            label=1,
            domain="brats21",
            participant_id=participant,
            session_id=str(row.get("session_id") or row.get("case_id") or participant),
            case_id=str(row.get("case_id") or participant),
            z=z,
            z_norm=z_norm,
            z_bin=z_bin_for(z_norm, Z_BINS),
            source_dataset="brats21",
            source_key=str(row.get("source_key") or f"{row.get('case_id')}:{z}"),
            source_split=str(row.get("source_split") or "train"),
            image_paths=image_paths,
            seg_path=str(row.get("seg_path") or ""),
            model_shape=MODEL_SHAPE,
            geometry_shape=ATLAS_SHAPE,
            stage="final",
            native_seg_voxels=0,
            model_mask_voxels=0,
            foreground_fraction=fractions,
            provenance=provenance,
            metadata=old_metadata,
        ))
    return records, exclusions


def survey_fingerprint_map(survey: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, item in survey.get("cohorts", {}).items():
        result[name] = {
            "root": item.get("root"),
            "config_path": item.get("config_path"),
            "config_fingerprint": item.get("config_fingerprint"),
            "metadata": item.get("metadata"),
            "splits": item.get("splits"),
            "normalization_consistency": item.get("normalization_consistency"),
            "errors": item.get("errors"),
        }
    return result


def records_by_split(records: Iterable[SliceRecord]) -> dict[str, list[SliceRecord]]:
    output = {split: [] for split in SPLITS}
    for record in records:
        output[record.split].append(record)
    for split in SPLITS:
        output[split].sort(key=lambda row: (row.pair_id, row.label, row.z_bin, row.z, row.record_id))
    return output


def write_manifest(path: Path, records: Sequence[SliceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def check_pair_histograms(records: Sequence[SliceRecord], summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[SliceRecord]] = defaultdict(list)
    for row in records:
        by_pair[row.pair_id].append(row)
    failures: list[dict[str, Any]] = []
    for summary in summaries:
        pair_id = str(summary["pair_id"])
        rows = by_pair[pair_id]
        hist = {0: Counter(row.z_bin for row in rows if row.label == 0), 1: Counter(row.z_bin for row in rows if row.label == 1)}
        if hist[0] != hist[1] or any(value > PAIR_CAP_PER_BIN for value in hist[0].values()):
            failures.append({"pair_id": pair_id, "healthy_histogram": dict(hist[0]), "brats_histogram": dict(hist[1])})
    return {"status": "PASS" if not failures else "FAIL", "pair_count": len(summaries), "failures": failures}


def check_participant_overlap(records: Sequence[SliceRecord]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in records:
        by_split[row.split].add(row.participant_id)
    overlaps: list[dict[str, Any]] = []
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            overlap = sorted(by_split[left].intersection(by_split[right]))
            if overlap:
                overlaps.append({"left": left, "right": right, "participants": overlap})
    return {"status": "PASS" if not overlaps else "FAIL", "counts": {split: len(values) for split, values in by_split.items()}, "overlaps": overlaps}


def check_z_norm(records: Sequence[SliceRecord]) -> dict[str, Any]:
    failures = [
        {"record_id": row.record_id, "z": row.z, "z_norm": row.z_norm}
        for row in records
        if not math.isclose(row.z_norm, float(row.z) / float(ATLAS_DEPTH - 1), rel_tol=0.0, abs_tol=1e-12)
    ]
    return {"status": "PASS" if not failures else "FAIL", "denominator_depth": ATLAS_DEPTH, "failures": failures[:50], "failure_count": len(failures)}


def check_support(records: Sequence[SliceRecord]) -> dict[str, Any]:
    failures = []
    for row in records:
        values = row.metadata.get("model_grid_support_fraction")
        if not isinstance(values, list) or len(values) != 3 or any(float(value) + 1e-12 < SUPPORT_THRESHOLD for value in values):
            failures.append({"record_id": row.record_id, "values": values})
    return {"status": "PASS" if not failures else "FAIL", "threshold": SUPPORT_THRESHOLD, "epsilon": SUPPORT_EPS, "expression": f"abs(x + 1) > {SUPPORT_EPS:g}", "denominator_pixels": 128 * 128, "failures": failures[:50], "failure_count": len(failures)}


def check_brats_zero(records: Sequence[SliceRecord]) -> dict[str, Any]:
    rows = [row for row in records if row.label == 1]
    failures = [
        {"record_id": row.record_id, "native_seg_voxels": row.native_seg_voxels, "model_mask_voxels": row.model_mask_voxels}
        for row in rows
        if row.native_seg_voxels != 0 or row.model_mask_voxels != 0 or not bool(row.metadata.get("native_segmentation_checked")) or not bool(row.metadata.get("model_mask_checked")) or not row.seg_path or not Path(row.seg_path).is_file()
    ]
    return {"status": "PASS" if not failures else "FAIL", "records": len(rows), "failures": failures[:50], "failure_count": len(failures)}


def check_source_joins(records: Sequence[SliceRecord], dataset: str, source_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    # For Mixed, the output source key is the mixed local key while the
    # immutable underlying key is in provenance.  Resolve both namespaces.
    base_indices = {name: source_index(rows) for name, rows in source_rows.items() if name != "mixed"}
    direct_index = base_indices.get(dataset, {})
    mixed_indices = {
        (str(item["source_split"]), str(item["mixed_local_key"])): item
        for item in source_rows.get("mixed", [])
    }
    mixed_key_counts = Counter((str(item["source_split"]), str(item["mixed_local_key"])) for item in source_rows.get("mixed", []))
    for key, count in mixed_key_counts.items():
        if count != 1:
            failures.append({"key": key, "reason": "duplicate_mixed_split_local_key", "count": count})
    for row in records:
        if row.label == 1:
            continue
        if dataset == "mixed":
            mixed_key = (row.source_split, row.source_key)
            source_item = mixed_indices.get(mixed_key)
            if source_item is None:
                failures.append({"record_id": row.record_id, "reason": "missing_mixed_split_local_key"})
                continue
            source_dataset = str(source_item.get("underlying_source_dataset"))
            underlying_key = (str(source_item.get("source_split")), str(source_item.get("source_key")))
            if underlying_key not in base_indices.get(source_dataset, {}):
                failures.append({"record_id": row.record_id, "reason": "missing_underlying_source_key", "source_dataset": source_dataset, "key": underlying_key})
                continue
            base_item = base_indices[source_dataset][underlying_key]
            if str(base_item.get("participant_id")) != str(row.metadata.get("source_participant_id")) or str(base_item.get("case_id")) != row.case_id or int(base_item.get("z")) != int(row.z):
                failures.append({"record_id": row.record_id, "reason": "mixed_underlying_metadata_mismatch", "source_dataset": source_dataset, "key": underlying_key})
            continue
        source_item = direct_index.get((row.source_split, row.source_key))
        if source_item is None:
            failures.append({"record_id": row.record_id, "reason": "missing_split_local_key"})
            continue
        if str(source_item.get("participant_id")) != str(row.metadata.get("source_participant_id")) or str(source_item.get("case_id")) != row.case_id or int(source_item.get("z")) != int(row.z):
            failures.append({"record_id": row.record_id, "reason": "metadata_mismatch"})
    return {"status": "PASS" if not failures else "FAIL", "failures": failures[:50], "failure_count": len(failures)}


def check_map_reuse(records: Sequence[SliceRecord], comparison: str, old_maps: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Confirm that every retained participant honors an existing seed-73 map."""

    failures: list[dict[str, Any]] = []
    observed: dict[str, set[str]] = defaultdict(set)
    for row in records:
        observed[row.participant_id].add(row.split)
    existing_healthy = old_maps.get("fomo45k", {}) if comparison == "mixed" else old_maps.get(comparison, {})
    for participant, splits in observed.items():
        if len(splits) != 1:
            failures.append({"participant_id": participant, "reason": "participant_has_multiple_output_splits", "splits": sorted(splits)})
            continue
        expected = (existing_healthy.get(participant) if participant in existing_healthy else None)
        if expected is not None and expected != next(iter(splits)):
            failures.append({"participant_id": participant, "reason": "healthy_seed73_map_changed", "expected": expected, "actual": next(iter(splits))})
    for participant, splits in ((p, {row.split for row in records if row.participant_id == p}) for p in {row.participant_id for row in records if row.label == 1}):
        expected = old_maps.get("brats21", {}).get(participant)
        if expected is not None and splits != {expected}:
            failures.append({"participant_id": participant, "reason": "brats_seed73_map_changed", "expected": expected, "actual": sorted(splits)})
    return {
        "status": "PASS" if not failures else "FAIL",
        "healthy_existing_map_checked": bool(existing_healthy),
        "brats_existing_map_checked": bool(old_maps.get("brats21")),
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def check_expected_participant_map(records: Sequence[SliceRecord], expected_map: Mapping[str, str]) -> dict[str, Any]:
    """Check a manifest against a previously built standalone cohort map."""

    observed: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row.label == 0:
            observed[row.participant_id].add(row.split)
    failures: list[dict[str, Any]] = []
    for participant, splits in observed.items():
        if participant not in expected_map:
            failures.append({"participant_id": str(participant), "reason": "participant_not_in_standalone_map", "actual": sorted(splits)})
            continue
        expected_split = str(expected_map[participant])
        if splits != {expected_split}:
            failures.append({"participant_id": str(participant), "expected": expected_split, "actual": sorted(splits)})
    missing_participants = sorted(set(expected_map) - set(observed))
    return {
        "status": "PASS" if not failures else "FAIL",
        "expected_participants": len(expected_map),
        "observed_participants": len(observed),
        "unselected_expected_participants": missing_participants,
        "unselected_expected_count": len(missing_participants),
        "failures": failures[:100],
        "failure_count": len(failures),
    }


def canonical_parity_contract(comparison: str) -> dict[str, Any]:
    root = COHORT_ROOTS[comparison]
    normalization_files = []
    for split in ("train", "val"):
        path = source_lmdb_path(comparison, split)
        normalization_files.append(path / "normalization.json")
    checks = {
        "source_root_exists": root.is_dir(),
        "normalization_contracts_exist": all(path.is_file() for path in normalization_files),
        "canonical_shape": list(MODEL_SHAPE),
        "canonical_dtype": "float32",
        "normalization": "robust_iqr",
        "source_loader": "LMDBSliceDataset(image_size=None), no resize/scaling",
    }
    return {"status": "PASS" if checks["source_root_exists"] and checks["normalization_contracts_exist"] else "FAIL", "checks": checks, "survey_path": str(SURVEY_PATH), "mixed_concat_build_report": str(COHORT_ROOTS["mixed"] / "build_report.json") if comparison == "mixed" else None}


def _close_lmdb_reader(reader: LMDBSliceDataset) -> None:
    transaction = getattr(reader, "txn", None)
    if transaction is not None:
        transaction.abort()
        reader.txn = None
    environment = getattr(reader, "env", None)
    if environment is not None:
        environment.close()
        reader.env = None


def check_mixed_tensor_parity(records: Sequence[SliceRecord]) -> dict[str, Any]:
    """Byte/hash compare every selected Mixed healthy tensor to its source."""

    selected = [row for row in records if row.label == 0]
    readers: dict[tuple[str, str], LMDBSliceDataset] = {}
    mismatches: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    checked = 0
    source_counts: Counter[str] = Counter()
    try:
        for row in selected:
            source_dataset = str(row.metadata.get("underlying_source_dataset") or "")
            source_split = str(row.metadata.get("underlying_source_split") or row.source_split)
            source_key = str(row.metadata.get("underlying_source_key") or "")
            mixed_split = str(row.source_split)
            mixed_key = str(row.source_key)
            if not source_dataset or not source_key or not mixed_key:
                mismatches.append({"record_id": row.record_id, "reason": "missing_mixed_or_source_identity"})
                continue
            source_reader_key = (source_dataset, source_split)
            mixed_reader_key = ("mixed", mixed_split)
            if source_reader_key not in readers:
                readers[source_reader_key] = LMDBSliceDataset(source_lmdb_path(source_dataset, source_split), image_size=None)
            if mixed_reader_key not in readers:
                readers[mixed_reader_key] = LMDBSliceDataset(source_lmdb_path("mixed", mixed_split), image_size=None)
            source_tensor = torch.as_tensor(readers[source_reader_key][int(source_key)], dtype=torch.float32).contiguous()
            mixed_tensor = torch.as_tensor(readers[mixed_reader_key][int(mixed_key)], dtype=torch.float32).contiguous()
            source_digest = hashlib.sha256(source_tensor.numpy().tobytes()).hexdigest()
            mixed_digest = hashlib.sha256(mixed_tensor.numpy().tobytes()).hexdigest()
            aggregate.update(f"{row.record_id}|{source_digest}|{mixed_digest}\n".encode("utf-8"))
            checked += 1
            source_counts[source_dataset] += 1
            if tuple(source_tensor.shape) != MODEL_SHAPE or tuple(mixed_tensor.shape) != MODEL_SHAPE or source_tensor.dtype != mixed_tensor.dtype or not torch.equal(source_tensor, mixed_tensor):
                mismatches.append({"record_id": row.record_id, "source_dataset": source_dataset, "source_key": source_key, "mixed_key": mixed_key, "source_sha256": source_digest, "mixed_sha256": mixed_digest, "source_shape": list(source_tensor.shape), "mixed_shape": list(mixed_tensor.shape)})
    finally:
        for reader in readers.values():
            _close_lmdb_reader(reader)
    return {"status": "PASS" if checked == len(selected) and not mismatches else "FAIL", "selected_records": len(selected), "checked_records": checked, "source_dataset_counts": dict(source_counts), "aggregate_sha256": aggregate.hexdigest(), "hash_algorithm": "sha256(tensor.float32 contiguous bytes)", "mismatches": mismatches[:50], "mismatch_count": len(mismatches)}


def _composition_domain(row: SliceRecord, comparison: str) -> str:
    if row.label == 1:
        return "brats21"
    if comparison == "mixed":
        return str(row.metadata.get("underlying_source_dataset") or "mixed")
    return comparison


def composition_summary(eligible: Sequence[SliceRecord], selected: Sequence[SliceRecord], comparison: str) -> dict[str, Any]:
    """Report participant/case/z-bin composition before and after pairing."""

    def summarize(rows: Sequence[SliceRecord]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        grouped: dict[tuple[str, str], list[SliceRecord]] = defaultdict(list)
        for row in rows:
            grouped[(_composition_domain(row, comparison), row.split)].append(row)
        for (domain, split), values in sorted(grouped.items()):
            participants = sorted({row.participant_id for row in values})
            result[f"{domain}:{split}"] = {
                "records": len(values),
                "participants": len(participants),
                "participant_ids": participants,
                "cases": len({row.case_id for row in values}),
                "z_bin_counts": dict(sorted(Counter(str(row.z_bin) for row in values).items(), key=lambda item: int(item[0]))),
                "z_min": min(row.z for row in values),
                "z_max": max(row.z for row in values),
            }
        return result

    eligible_by_key = summarize(eligible)
    selected_by_key = summarize(selected)
    keys = sorted(set(eligible_by_key) | set(selected_by_key))
    rows: dict[str, Any] = {}
    for key in keys:
        e = eligible_by_key.get(key, {"records": 0, "participants": 0, "participant_ids": [], "cases": 0, "z_bin_counts": {}})
        s = selected_by_key.get(key, {"records": 0, "participants": 0, "participant_ids": [], "cases": 0, "z_bin_counts": {}})
        rows[key] = {
            "eligible_records": e["records"],
            "selected_records": s["records"],
            "eligible_participants": e["participants"],
            "selected_participants": s["participants"],
            "unpaired_participants": sorted(set(e["participant_ids"]) - set(s["participant_ids"])),
            "eligible_cases": e["cases"],
            "selected_cases": s["cases"],
            "eligible_z_bin_counts": e["z_bin_counts"],
            "selected_z_bin_counts": s["z_bin_counts"],
            "eligible_z_min": e.get("z_min"),
            "eligible_z_max": e.get("z_max"),
            "selected_z_min": s.get("z_min"),
            "selected_z_max": s.get("z_max"),
        }
    return {"by_underlying_cohort_split": rows}


def materialize_selected_brats_cache(
    records: Sequence[SliceRecord],
    *,
    comparison: str,
    output_root: Path,
    brats_root: Path = BRATS_ROOT,
    csv_path: Path = BRATS_CSV,
) -> tuple[list[SliceRecord], dict[str, Any]]:
    """Cache only paired BraTS slices using the canonical MRIDataVolume path."""

    selected = [row for row in records if row.label == 1]
    if not selected:
        return list(records), {"status": "FAIL", "reason": "no_selected_brats_records", "records": 0}
    from andi_rewrite.data.datasets.brats import MRIDataVolume

    dataset = MRIDataVolume(
        csv_path=csv_path,
        dataset_path=brats_root,
        image_size=128,
        modalities=["flair", "t1", "t2"],
        segmentation_suffix="seg",
        filename_separator="_",
        return_metadata=True,
        intensity_normalization="robust_iqr",
    )
    subjects = [str(value).strip() for value in dataset.df.iloc[:, 0].tolist()]
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    by_subject: dict[str, list[SliceRecord]] = defaultdict(list)
    for row in selected:
        raw_subject = str(row.metadata.get("source_participant_id") or row.case_id)
        if ":" in raw_subject:
            raw_subject = raw_subject.split(":", 1)[1]
        by_subject[raw_subject].append(row)
    cache_root = output_root / "brats_cache" / comparison
    cache_root.mkdir(parents=True, exist_ok=True)
    updates: dict[str, SliceRecord] = {}
    cache_files: list[dict[str, Any]] = []
    started = time.perf_counter()
    for subject_number, (subject, subject_rows) in enumerate(sorted(by_subject.items()), start=1):
        if subject not in subject_index:
            raise ValueError(f"selected BraTS subject absent from canonical CSV: {subject}")
        by_z: dict[int, SliceRecord] = {}
        for row in subject_rows:
            z = int(row.z)
            if z in by_z:
                raise ValueError(f"duplicate selected BraTS z for {subject}: {z}")
            by_z[z] = row
        volume, model_mask, metadata = dataset[subject_index[subject]]
        if tuple(volume.shape) != (3, 128, 128, ATLAS_DEPTH) or tuple(model_mask.shape) != (128, 128, ATLAS_DEPTH):
            raise ValueError(f"canonical selected cache shape mismatch for {subject}: {tuple(volume.shape)}, {tuple(model_mask.shape)}")
        if not bool(torch.isfinite(volume).all()):
            raise ValueError(f"canonical selected cache volume nonfinite for {subject}")
        z_values = sorted(by_z)
        tensors = []
        for z in z_values:
            if bool(model_mask[..., z].any()):
                raise ValueError(f"selected BraTS model mask became nonzero for {subject}:{z}")
            tensor = volume[..., z].contiguous()
            fractions = support_fraction(tensor)
            expected = tuple(float(value) for value in by_z[z].foreground_fraction)
            if expected and fractions != expected and not np.allclose(fractions, expected, atol=1e-12, rtol=0.0):
                raise ValueError(f"selected BraTS support changed for {subject}:{z}: {fractions} != {expected}")
            tensors.append(tensor)
        images = torch.stack(tensors, dim=0).numpy().astype(np.float32, copy=False)
        cache_path = cache_root / f"{subject}.npz"
        temporary = cache_root / f".{subject}.tmp.npz"
        np.savez_compressed(temporary, images=images, z=np.asarray(z_values, dtype=np.int16))
        temporary.replace(cache_path)
        file_digest = sha256(cache_path)
        cache_files.append({"subject": subject, "path": str(cache_path), "sha256": file_digest, "records": len(z_values), "z": z_values})
        for index, z in enumerate(z_values):
            row = by_z[z]
            tensor_digest = hashlib.sha256(images[index].tobytes()).hexdigest()
            new_metadata = dict(row.metadata)
            new_metadata.update({
                "selected_cache_path": str(cache_path),
                "selected_cache_index": index,
                "selected_cache_sha256": file_digest,
                "selected_cache_source": "MRIDataVolume(full_volume_robust_iqr_then_model_grid_slice)",
            })
            new_provenance = dict(row.provenance)
            new_provenance.update({
                "model_input_sha256": tensor_digest,
                "selected_cache_sha256": file_digest,
                "selected_cache_path": str(cache_path),
                "selected_cache_index": index,
            })
            updates[row.record_id] = row.with_updates(metadata=new_metadata, provenance=new_provenance)
        del volume, model_mask, metadata, tensors, images
        if subject_number == 1 or subject_number % 25 == 0 or subject_number == len(by_subject):
            print(json.dumps({"event": "brats_selected_cache_progress", "comparison": comparison, "subjects_done": subject_number, "subjects_total": len(by_subject), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    updated = [updates.get(row.record_id, row) for row in records]
    summary = {
        "status": "PASS" if len(updates) == len(selected) else "FAIL",
        "comparison": comparison,
        "records": len(selected),
        "participants": len(by_subject),
        "cache_root": str(cache_root),
        "cache_files": cache_files,
        "cache_file_count": len(cache_files),
        "updated_records": len(updates),
        "elapsed_seconds": time.perf_counter() - started,
        "loader": "data.datasets.brats.MRIDataVolume",
        "normalization": "robust_iqr",
        "model_shape": list(MODEL_SHAPE),
    }
    json_dump(output_root / "audits" / f"{comparison}_brats_cache.json", summary)
    return updated, summary


def build_one_comparison(
    comparison: str,
    healthy_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    old_maps: Mapping[str, Mapping[str, str]],
    source_hashes: Mapping[tuple[str, str], str],
    brats_rows: Sequence[Mapping[str, Any]],
    brats_map: Mapping[str, str],
    output_root: Path,
    support_cache: Mapping[tuple[str, str], tuple[float, float, float]] | None = None,
    support_summary: Mapping[str, Any] | None = None,
    brats_root: Path = BRATS_ROOT,
    brats_csv: Path = BRATS_CSV,
) -> dict[str, Any]:
    healthy_participants = [namespaced_participant(comparison, str(row["participant_id"])) if comparison != "mixed" else namespaced_participant(str(row.get("source_dataset")), str(row["participant_id"])) for row in healthy_rows]
    if comparison == "mixed":
        preserved = {namespaced_participant("fomo45k", str(row["participant_id"])): split for row in healthy_rows if str(row.get("source_dataset")) == "fomo45k" for split in [old_maps.get("fomo45k", {}).get(namespaced_participant("fomo45k", str(row["participant_id"])), "")] if split}
        participant_map, map_info = assign_map(healthy_participants, seed=73, existing=preserved)
    else:
        existing = old_maps.get(comparison, {})
        participant_map, map_info = assign_map(healthy_participants, seed=73, existing=existing)
    if support_cache is None:
        support_cache, support_summary = materialize_healthy_support(healthy_rows, comparison)
    else:
        support_summary = dict(support_summary or {})
    registered_paths = read_old_fomo_registered_paths() if comparison == "fomo45k" else {}
    healthy_records, healthy_exclusions = make_healthy_records(healthy_rows, comparison, support_cache, participant_map, source_hashes, registered_paths=registered_paths)
    brats_records, brats_exclusions = make_brats_records(brats_rows, brats_map)
    pairing = build_pairs(healthy_records, brats_records, comparison=comparison, seed=73, bins=Z_BINS, cap_per_bin=PAIR_CAP_PER_BIN)
    materialized_records, brats_cache_summary = materialize_selected_brats_cache(
        pairing.records,
        comparison=comparison,
        output_root=output_root,
        brats_root=brats_root,
        csv_path=brats_csv,
    )
    grouped = records_by_split(materialized_records)
    manifest_dir = output_root / "manifests" / comparison
    for split in SPLITS:
        write_manifest(manifest_dir / f"{split}.jsonl", grouped[split])
    all_records = list(materialized_records)
    join_audit = check_source_joins(all_records, comparison, source_rows)
    healthy_final_records = [row for row in all_records if row.label == 0]
    registered_path_rows = [row for row in healthy_final_records if row.registered_paths]
    registered_path_failures = [
        row.record_id for row in healthy_final_records
        if comparison == "fomo45k" and set(row.registered_paths) != {"flair", "t1", "t2"}
    ]
    missing_registered_files = [
        {
            "record_id": row.record_id,
            "missing": [name for name, path in row.registered_paths.items() if not Path(path).is_file()],
        }
        for row in healthy_final_records
        if comparison == "fomo45k" and set(row.registered_paths) == {"flair", "t1", "t2"}
        and any(not Path(path).is_file() for path in row.registered_paths.values())
    ]
    registered_audit = {
        "status": ("PASS" if not registered_path_failures and not missing_registered_files and len(registered_path_rows) == len(healthy_final_records) else "FAIL") if comparison == "fomo45k" else "NOT_APPLICABLE",
        "records": len(healthy_final_records),
        "records_with_registered_paths": len(registered_path_rows),
        "failure_count": len(registered_path_failures),
        "failures": registered_path_failures[:50],
        "missing_file_count": len(missing_registered_files),
        "missing_files": missing_registered_files[:50],
        "source": "historical FOMO manifest raw registered_paths indexed by participant/case/session with split/key fallback",
    }
    audit = {
        "schema_version": 3,
        "comparison": comparison,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(all_records),
        "domain_counts": dict(Counter(row.domain for row in all_records)),
        "label_counts": dict(Counter(str(row.label) for row in all_records)),
        "split_counts": {split: len(grouped[split]) for split in SPLITS},
        "subject_counts": {split: len({row.participant_id for row in grouped[split]}) for split in SPLITS},
        "participant_split_map": participant_map,
        "participant_map_info": map_info,
        "primary_case_rule": "lexicographic (case_id, session_id) per participant before pairing",
        "pairing": {"pairs": list(pairing.pairs), "exclusions": list(pairing.exclusions), "cap_per_bin": PAIR_CAP_PER_BIN, "z_bins": Z_BINS, "seed": 73},
        "eligibility_composition": composition_summary(healthy_records + brats_records, all_records, comparison),
        "exclusions": {"healthy_support": healthy_exclusions, "brats_support_or_mask": brats_exclusions},
        "support_eligibility": check_support(all_records),
        "z_norm": check_z_norm(all_records),
        "participant_overlap": check_participant_overlap(all_records),
        "pair_z_histogram": check_pair_histograms(all_records, pairing.pairs),
        "brats_native_and_model_zero": check_brats_zero(all_records),
        "source_split_local_joins": join_audit,
        "registered_input_paths": registered_audit,
        "participant_map_reuse": check_map_reuse(all_records, comparison, old_maps),
        "canonical_contract": canonical_parity_contract(comparison),
        "mixed_tensor_parity": {"status": "NOT_APPLICABLE", "selected_records": 0, "checked_records": 0},
        "brats_selected_tensor_cache": brats_cache_summary,
        "healthy_source_scan": support_summary,
        "healthy_limitations": {
            "raw_volumes_opened": False,
            "canonical_lmdb_read": True,
            "support_is_model_grid_proxy": True,
            "support_is_not_native_brain_mask": True,
        },
        "historical_control_reuse": {
            "results_reused": False,
            "old_fomo_pair_assignments_reused": False,
            "brats_candidate_tensors": "read_only_exact_tensor_source_with_source_hashes; v3 pairing rebuilt",
            "selection_equivalence_proof": False,
        },
    }
    checks = [audit[key]["status"] for key in ("support_eligibility", "z_norm", "participant_overlap", "pair_z_histogram", "brats_native_and_model_zero", "source_split_local_joins", "participant_map_reuse", "canonical_contract", "brats_selected_tensor_cache")]
    if comparison == "fomo45k":
        checks.append(registered_audit["status"])
    required_nonempty = all(audit["split_counts"].get(split, 0) > 0 for split in SPLITS) and all(audit["domain_counts"].get(domain, 0) > 0 for domain in (comparison, "brats21"))
    audit["status"] = "PASS" if all(status == "PASS" for status in checks) and required_nonempty else "FAIL"
    json_dump(output_root / "audits" / f"{comparison}.json", audit)
    return audit


def build_comparison_with_mixed(
    comparison: str,
    mixed_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_support: Mapping[str, Mapping[tuple[str, str], tuple[float, float, float]]],
    old_maps: Mapping[str, Mapping[str, str]],
    source_hashes: Mapping[tuple[str, str], str],
    brats_rows: Sequence[Mapping[str, Any]],
    brats_map: Mapping[str, str],
    output_root: Path,
    brats_root: Path = BRATS_ROOT,
    brats_csv: Path = BRATS_CSV,
) -> dict[str, Any]:
    base_indices = {name: source_index(rows) for name, rows in source_rows.items() if name != "mixed"}
    healthy_rows: list[dict[str, Any]] = []
    support: dict[tuple[str, str], tuple[float, float, float]] = {}
    registered_paths = read_old_fomo_registered_paths()
    for row in mixed_rows:
        source_dataset = str(row["source_dataset"]) if str(row.get("source_dataset")) != "mixed" else str(row.get("source_dataset"))
        # Mixed source rows themselves carry source_dataset (mpi/oasis3/fomo45k).
        source_dataset = str(row.get("source_dataset"))
        if source_dataset == "mixed":
            source_dataset = str(row.get("underlying_source_dataset"))
        source_split = str(row["source_split"])
        source_key = str(row.get("source_key") or "")
        underlying = base_indices.get(source_dataset, {}).get((source_split, source_key))
        if underlying is None:
            # The sidecar's source_dataset is authoritative; this branch
            # creates a diagnostic failure row instead of guessing a key.
            raise KeyError(f"Mixed row cannot resolve {source_dataset}:{source_split}:{source_key}")
        item = dict(underlying)
        item["source_dataset"] = source_dataset
        item["mixed_local_key"] = str(row.get("mixed_local_key") or row.get("key") or "")
        item["mixed_source_key"] = source_key
        item["mixed_source_split"] = source_split
        healthy_rows.append(item)
        item["_support_fraction"] = source_support[source_dataset][(source_split, str(item["source_key"]))]
    # Extend SliceRecord construction with the mixed LMDB/local key while
    # retaining source participant/case/split provenance.
    participants = [namespaced_participant(str(row["source_dataset"]), str(row["participant_id"])) for row in healthy_rows]
    # Mixed reuses each standalone cohort's seed-73 participant map.  Calling
    # participant_split_map once on the combined ID list would reshuffle MPI
    # and OASIS participants and silently break cross-comparison split parity.
    standalone_maps: dict[str, dict[str, str]] = {}
    standalone_map_info: dict[str, Any] = {}
    for dataset_name in ("fomo45k", "mpi", "oasis3"):
        cohort_ids = sorted({
            namespaced_participant(dataset_name, str(row["participant_id"]))
            for row in healthy_rows
            if str(row["source_dataset"]) == dataset_name
        })
        standalone_maps[dataset_name], standalone_map_info[dataset_name] = assign_map(
            cohort_ids,
            seed=73,
            existing=old_maps.get(dataset_name, {}),
        )
    expected_mixed_map = {
        participant: split
        for dataset_map in standalone_maps.values()
        for participant, split in dataset_map.items()
    }
    participant_map = {participant: expected_mixed_map[participant] for participant in participants}
    map_info = {
        "seed": 73,
        "source": "union_of_standalone_seed73_maps",
        "standalone_maps": standalone_map_info,
        "participant_count": len(participant_map),
        "split_counts": dict(Counter(participant_map.values())),
        "preserved_count": sum(info.get("preserved_count", 0) for info in standalone_map_info.values()),
        "extended_count": sum(info.get("extended_count", 0) for info in standalone_map_info.values()),
    }
    records: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in healthy_rows:
        by_participant[namespaced_participant(str(row["source_dataset"]), str(row["participant_id"]))].append(row)
    primary = {pid: sorted((str(r.get("case_id", "")), str(r.get("session_id", ""))) for r in rows)[0] for pid, rows in by_participant.items()}
    for row in healthy_rows:
        source_dataset = str(row["source_dataset"])
        pid = namespaced_participant(source_dataset, str(row["participant_id"]))
        if (str(row.get("case_id", "")), str(row.get("session_id", ""))) != primary[pid]:
            continue
        source_split = str(row["source_split"])
        source_key = str(row["source_key"])
        fractions = tuple(float(value) for value in row["_support_fraction"])
        if any(value + 1e-12 < SUPPORT_THRESHOLD for value in fractions):
            exclusions.append({"participant_id": pid, "source_split": source_split, "source_key": str(row.get("mixed_local_key")), "reason": "model_grid_support_below_threshold", "support_fraction": list(fractions)})
            continue
        z = int(row["z"])
        z_norm = float(z) / float(ATLAS_DEPTH - 1)
        mixed_split = "train" if source_split == "train" else "val"
        mixed_key = str(row["mixed_local_key"])
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
            "source_case_id": str(row.get("case_id", "")),
            "source_session_id": str(row.get("session_id", "")),
            "source_key_namespace": "mixed_local_per_source_split",
            "mixed_local_key": mixed_key,
            "underlying_source_dataset": source_dataset,
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
            "build_protocol": "model_grid_v3",
            "mixed_exact_concat_source": True,
        }
        records.append(SliceRecord(split=str(participant_map[pid]), label=0, domain="mixed", participant_id=pid, session_id=str(row.get("session_id", "")), case_id=str(row.get("case_id", "")), z=z, z_norm=z_norm, z_bin=z_bin_for(z_norm, Z_BINS), source_dataset="mixed", source_key=mixed_key, source_split=source_split, registered_paths=registered_paths_for_row(registered_paths, source_split=source_split, source_key=source_key, participant_id=namespaced_participant(str(row.get("source_dataset")), str(row.get("participant_id"))), case_id=str(row.get("case_id", "")), session_id=str(row.get("session_id", ""))), model_shape=MODEL_SHAPE, geometry_shape=ATLAS_SHAPE, stage="final", foreground_fraction=fractions, provenance=provenance, metadata=metadata))
    brats_records, brats_exclusions = make_brats_records(brats_rows, brats_map)
    pairing = build_pairs(records, brats_records, comparison="mixed", seed=73, bins=Z_BINS, cap_per_bin=PAIR_CAP_PER_BIN)
    materialized_records, brats_cache_summary = materialize_selected_brats_cache(
        pairing.records,
        comparison="mixed",
        output_root=output_root,
        brats_root=brats_root,
        csv_path=brats_csv,
    )
    grouped = records_by_split(materialized_records)
    manifest_dir = output_root / "manifests" / "mixed"
    for split in SPLITS:
        write_manifest(manifest_dir / f"{split}.jsonl", grouped[split])
    join_audit = check_source_joins(materialized_records, "mixed", source_rows)
    mixed_parity = check_mixed_tensor_parity(materialized_records)
    audit = {
        "schema_version": 3,
        "comparison": "mixed",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(materialized_records),
        "domain_counts": dict(Counter(row.domain for row in materialized_records)),
        "label_counts": dict(Counter(str(row.label) for row in materialized_records)),
        "split_counts": {split: len(grouped[split]) for split in SPLITS},
        "subject_counts": {split: len({row.participant_id for row in grouped[split]}) for split in SPLITS},
        "participant_split_map": participant_map,
        "participant_map_info": map_info,
        "primary_case_rule": "lexicographic (case_id, session_id) per source participant before pairing",
        "pairing": {"pairs": list(pairing.pairs), "exclusions": list(pairing.exclusions), "cap_per_bin": PAIR_CAP_PER_BIN, "z_bins": Z_BINS, "seed": 73},
        "eligibility_composition": composition_summary(records + brats_records, list(materialized_records), "mixed"),
        "exclusions": {"healthy_support": exclusions, "brats_support_or_mask": brats_exclusions},
        "support_eligibility": check_support(materialized_records),
        "z_norm": check_z_norm(materialized_records),
        "participant_overlap": check_participant_overlap(materialized_records),
        "pair_z_histogram": check_pair_histograms(materialized_records, pairing.pairs),
        "brats_native_and_model_zero": check_brats_zero(materialized_records),
        "source_split_local_joins": join_audit,
        "participant_map_reuse": check_map_reuse(materialized_records, "mixed", old_maps),
        "standalone_map_consistency": check_expected_participant_map(materialized_records, expected_mixed_map),
        "canonical_contract": canonical_parity_contract("mixed"),
        "mixed_tensor_parity": mixed_parity,
        "brats_selected_tensor_cache": brats_cache_summary,
        "mixed_source_join": {"status": join_audit["status"], "source_dataset_split_local_keys": True, "mixed_exact_concat_build_report": str(COHORT_ROOTS["mixed"] / "build_report.json"), "source_participant_case_split_reused": join_audit["status"] == "PASS", "new_random_split": False},
        "healthy_limitations": {"raw_volumes_opened": False, "canonical_lmdb_read": True, "support_is_model_grid_proxy": True, "support_is_not_native_brain_mask": True},
        "historical_control_reuse": {"results_reused": False, "old_fomo_pair_assignments_reused": False, "brats_candidate_tensors": "read_only_exact_tensor_source_with_source_hashes; v3 pairing rebuilt", "selection_equivalence_proof": False},
    }
    checks = [audit[key]["status"] for key in ("support_eligibility", "z_norm", "participant_overlap", "pair_z_histogram", "brats_native_and_model_zero", "source_split_local_joins", "participant_map_reuse", "standalone_map_consistency", "canonical_contract", "mixed_tensor_parity", "brats_selected_tensor_cache")]
    required_nonempty = all(audit["split_counts"].get(split, 0) > 0 for split in SPLITS) and all(audit["domain_counts"].get(domain, 0) > 0 for domain in ("mixed", "brats21"))
    audit["status"] = "PASS" if all(status == "PASS" for status in checks) and required_nonempty else "FAIL"
    json_dump(output_root / "audits" / "mixed.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v3 model-grid manifests without training")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="write protocol/source inventory only; do not read LMDB tensors")
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    old_maps = read_old_split_maps()
    protocol = {
        "schema_version": 3,
        "protocol_id": "model_grid_v3_fullcandidate_v1",
        "status": "FROZEN_BUILD_ONLY",
        "created_at_utc": started_utc,
        "training_started": False,
        "canonical_contract": {"channels": ["flair", "t1", "t2"], "channel_order": ["FLAIR", "T1", "T2"], "shape": list(MODEL_SHAPE), "dtype": "float32", "normalization": "robust_iqr", "healthy_reader": "existing canonical LMDB only; no resize or intensity transform"},
        "eligibility": {"support_expression": f"abs(x + 1) > {SUPPORT_EPS:g}", "support_threshold_each_channel": SUPPORT_THRESHOLD, "support_denominator_pixels": 128 * 128, "all_three_channels_required": True, "support_is_explicit_model_grid_proxy": True, "support_is_not_native_brain_mask": True, "brats_native_segmentation_voxels": 0, "brats_model_mask_voxels": 0},
        "z_matching": {"z_norm_denominator_depth": ATLAS_DEPTH, "denominator_basis": "atlas/reference configuration shape 240x240x155, never max available LMDB index", "bins": Z_BINS, "cap_per_pair_per_bin": PAIR_CAP_PER_BIN},
        "participant_splits": {"seed": 73, "fractions": [0.70, 0.15, 0.15], "fomo45k_and_brats21": "preserve exact historical seed-73 maps", "new_cohorts": "same deterministic seed-73 allocator", "mixed": "preserve FOMO map entries and extend to MPI/OASIS IDs; retain source participant/case/source_split/local key"},
        "historical_controls": {"results_reused": False, "old_calibration_replay": str(REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "fomo45k_calibration_smallcnn"), "selection_equivalence_proof_required_before_reuse": True, "supersedes": str(REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_v3_attempt_oldpool_20260917_151934"), "superseded_reason": "historical BraTS selected-cache pool was incorrectly used as the candidate inventory"},
        "provenance_freeze": {"root": str(output_root / "provenance_freeze"), "pre_v3_source_snapshot": str(output_root / "provenance_freeze" / "pre_v3_snapshot_manifest.json"), "v3_builder_snapshot_extension": str(output_root / "provenance_freeze" / "freeze_extension_manifest.json")},
        "future_controls_frozen": {"tiny": True, "available_registered_positive": True, "same_cohort_negative": True, "heldout_label_sanity": True, "train_shuffle_true_test_diagnostic_only": True, "full_retrained_c2st_replicates": 199},
        "geometry_contract": {"atlas_shape": list(ATLAS_SHAPE), "atlas_sha256": "1941028b68080161144150aabf5c5645968d164310b21509773a594ca37e1891", "z_axis": "registered/model-grid axial depth; denominator is full atlas depth 155"},
        "limitations": {"healthy_raw_paths": {"fomo45k": "optional metadata available through historical manifests", "mpi": "not available in this source sidecar", "oasis3": "not available in this source sidecar"}, "brats_side": "BraTS participants remain tumor patients even when selected native/model masks are zero; this is a selected-slice domain diagnostic"},
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if str(existing_protocol.get("protocol_id")) != str(protocol["protocol_id"]):
            raise ValueError(f"Existing protocol identity mismatch at {protocol_path}")
        # The protocol is a frozen input to candidate resumption.  Preserve
        # its original timestamp/content instead of changing the resume hash
        # on every invocation.
        protocol = existing_protocol
    else:
        json_dump(protocol_path, protocol)
    source_rows = canonical_source_rows()
    source_hashes: dict[tuple[str, str], str] = {}
    for dataset in ("fomo45k", "mpi", "oasis3", "mixed"):
        splits = {str(row["source_split"]) for row in source_rows[dataset]}
        for split in splits:
            path = source_sidecar_path(dataset, split)
            source_hashes[(dataset, split)] = sidecar_hash(path)
    json_dump(output_root / "source_fingerprints.json", {"survey": survey_fingerprint_map(survey), "sidecars": {f"{dataset}:{split}": value for (dataset, split), value in source_hashes.items()}, "old_split_maps": {name: {"sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest(), "count": len(values)} for name, values in old_maps.items()}, "atlas_depth": ATLAS_DEPTH, "atlas_shape": list(ATLAS_SHAPE)})
    if args.dry_run:
        json_dump(output_root / "build_timing.json", {"status": "DRY_RUN", "started_at_utc": started_utc, "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"status": "DRY_RUN", "output_root": str(output_root)}, indent=2))
        return 0

    brats_rows, brats_summary = load_brats_candidate_rows(old_maps["brats21"], output_root, brats_root=BRATS_ROOT, csv_path=BRATS_CSV)
    # Materialize each canonical source exactly once.  Mixed support is
    # inherited from its source tensor because the source build report records
    # bytewise exact concatenation; no mixed resize or transform is applied.
    base_support: dict[str, Mapping[tuple[str, str], tuple[float, float, float]]] = {}
    base_support_summaries: dict[str, Any] = {}
    for dataset in ("fomo45k", "mpi", "oasis3"):
        base_support[dataset], base_support_summaries[dataset] = materialize_healthy_support(source_rows[dataset], dataset)
        json_dump(output_root / "canonical_support" / f"{dataset}.json", base_support_summaries[dataset])
        print(json.dumps({"event": "canonical_support_complete", "dataset": dataset, "rows": len(source_rows[dataset]), "elapsed_seconds": time.perf_counter() - started}))
    audits: dict[str, Any] = {}
    for comparison in ("fomo45k", "mpi", "oasis3"):
        audits[comparison] = build_one_comparison(comparison, source_rows[comparison], source_rows, old_maps, source_hashes, brats_rows, old_maps["brats21"], output_root, base_support[comparison], base_support_summaries[comparison], BRATS_ROOT, BRATS_CSV)
        print(json.dumps({"event": "comparison_complete", "comparison": comparison, "status": audits[comparison]["status"], "records": audits[comparison]["records"], "elapsed_seconds": time.perf_counter() - started}))
    audits["mixed"] = build_comparison_with_mixed("mixed", source_rows["mixed"], source_rows, base_support, old_maps, source_hashes, brats_rows, old_maps["brats21"], output_root, BRATS_ROOT, BRATS_CSV)
    print(json.dumps({"event": "comparison_complete", "comparison": "mixed", "status": audits["mixed"]["status"], "records": audits["mixed"]["records"], "elapsed_seconds": time.perf_counter() - started}))
    summary = {
        "schema_version": 3,
        "status": "PASS" if all(item["status"] == "PASS" for item in audits.values()) else "FAIL",
        "created_at_utc": started_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "training_started": False,
        "comparisons": {name: {"status": item["status"], "records": item["records"], "domain_counts": item["domain_counts"], "split_counts": item["split_counts"], "audit": str(output_root / "audits" / f"{name}.json")} for name, item in audits.items()},
        "source_support_summaries": base_support_summaries,
        "full_candidate_coverage": {
            "status": brats_summary.get("status"),
            "verified": brats_summary.get("status") == "PASS" and not brats_summary.get("errors"),
            "csv_rows": brats_summary.get("source", {}).get("csv_rows_verified"),
            "processed_subjects": brats_summary.get("model_grid_counts", {}).get("subjects"),
            "candidate_rows": brats_summary.get("model_grid_counts", {}).get("candidate_rows"),
            "candidate_source": "full_csv_native_zero_then_model_mask_zero_then_model_grid_support",
            "old_selected_pool_used_as_source": False,
            "candidate_inventory": str(output_root / "brats_candidate_inventory.json"),
        },
        "build_command": "C:\\Users\\E-118-3\\miniconda3\\envs\\ANDi\\python.exe scripts\\build_model_grid_v3.py",
        "no_training": True,
    }
    json_dump(output_root / "build_summary.json", summary)
    json_dump(output_root / "build_timing.json", {"status": summary["status"], "started_at_utc": started_utc, "completed_at_utc": summary["completed_at_utc"], "elapsed_seconds": summary["elapsed_seconds"], "training_started": False})
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"], "output_root": str(output_root)}, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
