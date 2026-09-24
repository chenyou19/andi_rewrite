"""Prepare auditable Stage1 domain-classifier manifests.

The command inventories existing outputs, creates participant-level splits, and
matches one primary case per participant by normalized axial position.  It
does not alter source LMDBs, NIfTIs, or existing pipeline files.  BraTS final
model slices are materialized only after matching and only under the new
``outputs/diagnostics/domain_classifier/cache`` tree.

Examples
--------
    python scripts/prepare_domain_classifier.py --comparison fomo
    python scripts/prepare_domain_classifier.py --comparison all --no-cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from _bootstrap import bootstrap
except ImportError:  # pragma: no cover - module invocation
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import numpy as np

from andi_rewrite.data.domain_classifier import (
    MODALITIES,
    MODEL_SHAPE,
    SliceRecord,
    apply_participant_splits,
    audit_records,
    audit_source_provenance,
    build_pairs,
    file_fingerprint,
    namespaced_participant,
    write_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "diagnostics" / "domain_classifier"
DEFAULT_BRATS_ROOT = Path(r"C:\ML\data\BraTS_2021")
DEFAULT_FOMO_ROOT = ROOT / "outputs" / "datasets" / "fomo45k_sri24_robust_iqr"
DEFAULT_MPI_ROOT = ROOT / "outputs" / "datasets" / "mpi_sri24_robust_iqr"
DEFAULT_OASIS_ROOT = ROOT / "outputs" / "datasets" / "oasis3_sri24_robust_iqr"
DEFAULT_MIXED_ROOT = ROOT / "outputs" / "datasets" / "mpi_oasis3_fomo45k_sri24_robust_iqr"
DEFAULT_BRATS_CSV = ROOT / "splits" / "BraTS21" / "scans_train.csv"
SCHEMA_VERSION = 1
FOREGROUND_THRESHOLD = 0.10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress(message: str) -> None:
    """Emit a durable stage marker for long read-only source audits."""

    print(f"[domain-classifier] {_now()} {message}", flush=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _stable_source_hash(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _status_for_case(root: Path, case_id: str) -> tuple[Path, dict[str, Any]]:
    status_path = root / "volumes" / Path(case_id) / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = _read_json(status_path)
    if status.get("status") != "PASS":
        raise ValueError(f"Source case is not PASS: {case_id}")
    return status_path, status


def _registered_paths(root: Path, case_id: str, status: Mapping[str, Any]) -> dict[str, str]:
    case_root = root / "volumes" / Path(case_id)
    outputs = status.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"status.json has no outputs mapping: {case_id}")
    result: dict[str, str] = {}
    for modality in MODALITIES:
        name = outputs.get(modality)
        if not name:
            raise ValueError(f"status.json is missing {modality}: {case_id}")
        path = case_root / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[modality] = str(path.resolve())
    return result


def _healthy_record(
    *,
    dataset_name: str,
    root: Path,
    row: Mapping[str, Any],
    status: Mapping[str, Any],
    status_path: Path,
    lmdb_path: Path,
    source_manifest: Path,
    source_manifest_hash: str,
) -> SliceRecord:
    raw_participant = str(row["participant_id"])
    case_id = str(row["case_id"])
    source_split = str(row.get("split", row.get("source_split", "train")))
    z = int(row.get("z", row.get("slice", 0)))
    shape = tuple(int(value) for value in status.get("qc", {}).get("output_shape", (240, 240, 155)))
    if len(shape) != 3:
        raise ValueError(f"Healthy source geometry must be 3-D: {case_id}")
    registered = _registered_paths(root, case_id, status)
    source_identity = {
        "status": file_fingerprint(status_path, content=True),
        "entries": {"path": str(source_manifest.resolve()), "sha256": source_manifest_hash},
        "lmdb": file_fingerprint(lmdb_path / "normalization.json", content=True),
        "source_key": str(row.get("key", "")),
    }
    return SliceRecord(
        split=source_split,
        label=0,
        domain=dataset_name,
        participant_id=namespaced_participant(dataset_name, raw_participant),
        session_id=str(row.get("session_id", case_id.rsplit("/", 1)[-1])),
        case_id=case_id,
        z=z,
        z_norm=float(z) / max(1, shape[-1] - 1),
        z_bin=min(19, int((float(z) / max(1, shape[-1] - 1)) * 20)),
        source_dataset=dataset_name,
        source_key=str(row.get("key", "")),
        source_split=source_split,
        image_paths={},
        registered_paths=registered,
        geometry_shape=shape,
        model_shape=MODEL_SHAPE,
        stage="final",
        provenance={
            "source_identity_sha256": _stable_source_hash(source_identity),
            "status_path": str(status_path.resolve()),
            "entries_manifest": str(source_manifest.resolve()),
            "lmdb_path": str(lmdb_path.resolve()),
        },
        metadata={
            "input_kind": "lmdb",
            "lmdb_path": str(lmdb_path.resolve()),
            "source_participant_id": raw_participant,
            "source_manifest_sha256": source_manifest_hash,
            "normalization": "robust_iqr",
            "normalize_input": False,
            "foreground_threshold": FOREGROUND_THRESHOLD,
        },
    )


def load_healthy_records(dataset_name: str, root: str | Path) -> tuple[list[SliceRecord], dict[str, Any]]:
    """Read existing Healthy metadata without opening every image volume."""

    root = Path(root).resolve()
    report_path = root / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = _read_json(report_path)
    if report.get("status") != "PASS":
        raise ValueError(f"Healthy build report is not PASS: {report_path}")
    if dataset_name == "fomo45k":
        session_rows = {str(row["case_id"]): row for row in report.get("sessions", [])}
        entries = []
        for source_split in ("train", "val"):
            manifest = root / "manifests" / f"{source_split}_entries.csv"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            manifest_hash = _sha256_file(manifest)
            rows = _read_csv(manifest)
            lmdb_path = root / source_split
            if not (lmdb_path / "data.mdb").is_file():
                raise FileNotFoundError(lmdb_path / "data.mdb")
            for row in rows:
                if str(row.get("split")) != source_split:
                    raise ValueError(f"Entry split mismatch in {manifest}: {row}")
                case_id = str(row["case_id"])
                session = session_rows.get(case_id)
                if session is None:
                    raise ValueError(f"Entry references missing build_report session: {case_id}")
                status = {
                    "qc": {"output_shape": [240, 240, 155]},
                    "outputs": {},
                }
                # FOMO build_report stores raw registered paths in sessions.
                # It does not have a status.json beside every case, so retain
                # these paths directly and use an equivalent status object.
                raw_paths = session.get("input_paths")
                if not isinstance(raw_paths, Mapping):
                    raise ValueError(f"FOMO session lacks input_paths: {case_id}")
                registered = {str(key).lower(): str(value) for key, value in raw_paths.items()}
                source_identity = {
                    "build_report": file_fingerprint(report_path, content=True),
                    "entries": {"path": str(manifest.resolve()), "sha256": manifest_hash},
                    "lmdb": file_fingerprint(lmdb_path / "normalization.json", content=True),
                    "source_key": str(row.get("key", "")),
                }
                z = int(row["z"])
                shape = (240, 240, 155)
                entries.append(
                    SliceRecord(
                        split=source_split,
                        label=0,
                        domain=dataset_name,
                        participant_id=namespaced_participant(dataset_name, str(row["participant_id"])),
                        session_id=str(row.get("session_id", case_id.rsplit("/", 1)[-1])),
                        case_id=case_id,
                        z=z,
                        z_norm=float(z) / (shape[-1] - 1),
                        z_bin=min(19, int(float(z) / (shape[-1] - 1) * 20)),
                        source_dataset=dataset_name,
                        source_key=str(row.get("key", "")),
                        source_split=source_split,
                        image_paths={},
                        registered_paths=registered,
                        geometry_shape=shape,
                        model_shape=MODEL_SHAPE,
                        stage="final",
                        provenance={
                            "source_identity_sha256": _stable_source_hash(source_identity),
                            "entries_manifest": str(manifest.resolve()),
                            "lmdb_path": str(lmdb_path.resolve()),
                        },
                        metadata={
                            "input_kind": "lmdb",
                            "lmdb_path": str(lmdb_path.resolve()),
                            "source_participant_id": str(row["participant_id"]),
                            "source_manifest_sha256": manifest_hash,
                            "normalization": "robust_iqr",
                            "normalize_input": False,
                            "foreground_threshold": FOREGROUND_THRESHOLD,
                        },
                    )
                )
        inventory = {
            "dataset": dataset_name,
            "root": str(root),
            "source_report": str(report_path),
            "source_report_sha256": _sha256_file(report_path),
            "records": len(entries),
            "participants": len({row.participant_id for row in entries}),
            "sessions": len({row.case_id for row in entries}),
            "source_splits": dict(Counter(row.source_split for row in entries)),
        }
        return entries, inventory

    entries_path = root / "entries.jsonl"
    rows = _read_jsonl(entries_path)
    manifest_hash = _sha256_file(entries_path)
    records: list[SliceRecord] = []
    status_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in status_cache:
            status_cache[case_id] = _status_for_case(root, case_id)
        status_path, status = status_cache[case_id]
        source_split = str(row.get("split", "train"))
        lmdb_path = root / source_split
        if not (lmdb_path / "data.mdb").is_file():
            raise FileNotFoundError(lmdb_path / "data.mdb")
        records.append(
            _healthy_record(
                dataset_name=dataset_name,
                root=root,
                row=row,
                status=status,
                status_path=status_path,
                lmdb_path=lmdb_path,
                source_manifest=entries_path,
                source_manifest_hash=manifest_hash,
            )
        )
    inventory = {
        "dataset": dataset_name,
        "root": str(root),
        "source_manifest": str(entries_path),
        "source_manifest_sha256": manifest_hash,
        "records": len(records),
        "participants": len({row.participant_id for row in records}),
        "sessions": len({row.case_id for row in records}),
        "source_splits": dict(Counter(row.source_split for row in records)),
    }
    return records, inventory


def load_mixed_records(
    root: str | Path,
    source_records: Mapping[str, Sequence[SliceRecord]],
) -> tuple[list[SliceRecord], dict[str, Any]]:
    """Join mixed output keys back to source identities via source_entries."""

    root = Path(root).resolve()
    lookup: dict[tuple[str, str, str], SliceRecord] = {}
    for dataset_name, rows in source_records.items():
        for row in rows:
            lookup[(dataset_name, row.source_split, row.source_key)] = row
    locations = [
        (root / "source_entries.jsonl", root / "train"),
        (root / "validation" / "source_entries.jsonl", root / "validation" / "val"),
        (root / "val" / "source_entries.jsonl", root / "val"),
    ]
    output: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    for entries_path, lmdb_path in locations:
        if not entries_path.is_file():
            continue
        if not (lmdb_path / "data.mdb").is_file():
            raise FileNotFoundError(lmdb_path / "data.mdb")
        manifest_hash = _sha256_file(entries_path)
        for row in _read_jsonl(entries_path):
            key = (str(row["source_dataset"]), str(row["source_split"]), str(row["source_key"]))
            base = lookup.get(key)
            if base is None:
                exclusions.append({"row": row, "reason": "missing_source_record"})
                continue
            identity = {
                "mixed_entries": {"path": str(entries_path), "sha256": manifest_hash},
                "source_record": base.provenance.get("source_identity_sha256", ""),
                "source_key": str(row["key"]),
            }
            output.append(
                base.with_updates(
                    split=base.split,
                    domain="mixed",
                    source_dataset="mixed",
                    source_key=str(row["key"]),
                    source_split=str(row["source_split"]),
                    provenance={
                        **base.provenance,
                        "source_identity_sha256": _stable_source_hash(identity),
                        "mixed_entries": str(entries_path.resolve()),
                    },
                    metadata={
                        **base.metadata,
                        "input_kind": "lmdb",
                        "lmdb_path": str(lmdb_path.resolve()),
                        "mixed_source_dataset": str(row["source_dataset"]),
                        "mixed_source_key": str(row["source_key"]),
                    },
                )
            )
    inventory = {
        "dataset": "mixed",
        "root": str(root),
        "records": len(output),
        "participants": len({row.participant_id for row in output}),
        "sessions": len({row.case_id for row in output}),
        "source_counts": dict(Counter(row.metadata.get("mixed_source_dataset", "") for row in output)),
        "exclusions": exclusions,
    }
    return output, inventory


def _brats_paths(root: Path, subject: str) -> dict[str, str]:
    subject_dir = root / subject
    return {modality: str((subject_dir / f"{subject}_{modality}.nii.gz").resolve()) for modality in MODALITIES}


def load_brats_records(root: str | Path, csv_path: str | Path) -> tuple[list[SliceRecord], dict[str, Any]]:
    """Inventory BraTS geometry and native tumor-free z availability."""

    root = Path(root).resolve()
    csv_path = Path(csv_path).resolve()
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"BraTS CSV is empty: {csv_path}")
    column = "BraTS21ID" if "BraTS21ID" in rows[0] else next(iter(rows[0]))
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise ImportError("BraTS inventory requires nibabel.") from exc

    records: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    source_hash = _sha256_file(csv_path)
    for row in rows:
        subject = str(row[column]).strip()
        if not subject:
            exclusions.append({"subject": subject, "reason": "empty_subject_id"})
            continue
        paths = _brats_paths(root, subject)
        seg_path = root / subject / f"{subject}_seg.nii.gz"
        required = [Path(paths[modality]) for modality in MODALITIES] + [seg_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            exclusions.append({"subject": subject, "reason": "missing_paths", "paths": missing})
            continue
        seg_image = nib.load(str(seg_path))
        seg = np.asarray(seg_image.dataobj, dtype=np.float32)
        if seg.ndim != 3 or not np.isfinite(seg).all():
            exclusions.append({"subject": subject, "reason": "invalid_segmentation_geometry_or_finite"})
            continue
        shape = tuple(int(value) for value in seg.shape)
        reference_affine = np.asarray(seg_image.affine, dtype=np.float64)
        if reference_affine.shape != (4, 4) or not np.isfinite(reference_affine).all():
            exclusions.append({"subject": subject, "reason": "invalid_segmentation_affine"})
            continue
        geometry_failure = None
        for modality in MODALITIES:
            image = nib.load(paths[modality])
            if len(image.shape) != 3 or tuple(int(value) for value in image.shape) != shape:
                geometry_failure = f"{modality}_shape={tuple(image.shape)} versus seg={shape}"
                break
            if not np.allclose(np.asarray(image.affine), reference_affine, atol=1e-5, rtol=0):
                geometry_failure = f"{modality}_affine_mismatch"
                break
        if geometry_failure is not None:
            exclusions.append({"subject": subject, "reason": "geometry_mismatch", "detail": geometry_failure})
            continue
        # Native segmentation is the source of availability.  The model-grid
        # mask is checked after actual MRIDataVolume loading for selected rows.
        eligible_z = [z for z in range(shape[-1]) if not bool(np.any(seg[..., z] != 0))]
        status_identity = {
            "csv": {"path": str(csv_path), "sha256": source_hash},
            "seg": file_fingerprint(seg_path, content=True),
            "images": {modality: file_fingerprint(Path(paths[modality])) for modality in MODALITIES},
        }
        for z in eligible_z:
            z_norm = float(z) / max(1, shape[-1] - 1)
            records.append(
                SliceRecord(
                    split="train",
                    label=1,
                    domain="brats21",
                    participant_id=namespaced_participant("brats21", subject),
                    session_id=subject,
                    case_id=subject,
                    z=z,
                    z_norm=z_norm,
                    z_bin=min(19, int(z_norm * 20)),
                    source_dataset="brats21",
                    source_key=f"{subject}:{z}",
                    source_split="train",
                    image_paths=paths,
                    seg_path=str(seg_path.resolve()),
                    geometry_shape=shape,
                    model_shape=MODEL_SHAPE,
                    stage="final",
                    native_seg_voxels=0,
                    provenance={
                        "source_identity_sha256": _stable_source_hash(status_identity),
                        "brats_csv": str(csv_path),
                        "seg_path": str(seg_path.resolve()),
                    },
                    metadata={
                        "dataset_path": str(root),
                        "source_participant_id": subject,
                        "native_segmentation_checked": True,
                        "foreground_threshold": FOREGROUND_THRESHOLD,
                        "normalization": "robust_iqr",
                        "normalize_input": False,
                    },
                )
            )
    inventory = {
        "dataset": "brats21",
        "root": str(root),
        "csv": str(csv_path),
        "csv_sha256": source_hash,
        "records": len(records),
        "participants": len({row.participant_id for row in records}),
        "sessions": len({row.case_id for row in records}),
        "native_tumor_free_slices": len(records),
        "exclusions": exclusions,
    }
    return records, inventory


def _primary_case_groups(records: Sequence[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for row in records:
        grouped[row.participant_id].append(row)
    result: dict[str, list[SliceRecord]] = {}
    for participant, rows in grouped.items():
        primary_case, primary_session = min((row.case_id, row.session_id) for row in rows)
        result[participant] = [
            row for row in rows if row.case_id == primary_case and row.session_id == primary_session
        ]
    return result


def apply_foreground_eligibility(
    records: Sequence[SliceRecord],
    *,
    threshold: float = FOREGROUND_THRESHOLD,
) -> tuple[list[SliceRecord], list[dict[str, Any]]]:
    """Filter primary cases using the shared raw-positive foreground rule.

    This reads one raw registered volume per participant's primary case.  It
    intentionally happens after inventory and before z matching, so empty
    superior/inferior tips cannot become a domain shortcut.  Non-primary
    sessions remain in the source inventory and split audit; only the primary
    case participates in a fixed pair.
    """

    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("foreground threshold must lie in (0,1]")
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise ImportError("Foreground eligibility requires nibabel") from exc
    eligible: list[SliceRecord] = []
    exclusions: list[dict[str, Any]] = []
    for participant, rows in sorted(_primary_case_groups(records).items()):
        first = rows[0]
        paths = first.registered_paths or first.image_paths
        try:
            images = []
            reference = None
            for modality in MODALITIES:
                path_text = paths.get(modality, "")
                if not path_text:
                    raise ValueError(f"missing_{modality}_path")
                image = nib.load(str(Path(path_text)))
                values = np.asarray(image.dataobj, dtype=np.float32)
                if values.ndim != 3 or not np.isfinite(values).all():
                    raise ValueError(f"invalid_{modality}_geometry_or_finite")
                if reference is not None and (
                    values.shape != reference.shape
                    or not np.allclose(image.affine, reference_affine, atol=1e-5, rtol=0)
                ):
                    raise ValueError(f"{modality}_geometry_mismatch")
                if reference is None:
                    reference = values
                    reference_affine = np.asarray(image.affine, dtype=np.float64)
                images.append(values)
            volume = np.stack(images, axis=0)
            fractions = (volume > 0).mean(axis=(1, 2))
            by_z = {
                int(row.z): tuple(float(value) for value in fractions[:, int(row.z)])
                for row in rows
                if 0 <= int(row.z) < volume.shape[-1]
            }
            kept = []
            for row in rows:
                fraction = by_z.get(int(row.z))
                if fraction is None or not all(value >= threshold for value in fraction):
                    continue
                kept.append(
                    row.with_updates(
                        foreground_fraction=fraction,
                        metadata={
                            **row.metadata,
                            "foreground_checked": True,
                            "foreground_threshold": float(threshold),
                        },
                    )
                )
            if not kept:
                exclusions.append(
                    {"participant_id": participant, "case_id": first.case_id, "reason": "no_foreground_eligible_z"}
                )
                continue
            eligible.extend(kept)
        except Exception as exc:
            exclusions.append(
                {
                    "participant_id": participant,
                    "case_id": first.case_id,
                    "reason": "foreground_validation_failed",
                    "detail": str(exc),
                }
            )
    return eligible, exclusions


def _write_participant_splits(
    output: Path,
    comparison: str,
    records: Sequence[SliceRecord],
) -> None:
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in records:
        by_dataset[row.domain].add(row.participant_id)
    for dataset_name, participants in sorted(by_dataset.items()):
        for split in ("train", "val", "test"):
            path = output / "splits" / comparison / f"{dataset_name}_{split}_participants.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("participant_id\n")
                for participant in sorted(
                    {row.participant_id for row in records if row.domain == dataset_name and row.split == split}
                ):
                    handle.write(participant + "\n")


def _split_audit(records: Sequence[SliceRecord]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = {split: set() for split in ("train", "val", "test")}
    for row in records:
        if row.split not in by_split:
            raise ValueError(f"unknown participant split: {row.split}")
        by_split[row.split].add(row.participant_id)
    overlaps: dict[str, list[str]] = {}
    split_names = tuple(by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = sorted(by_split[left].intersection(by_split[right]))
            if overlap:
                overlaps[f"{left}_intersect_{right}"] = overlap
    if overlaps:
        raise ValueError(f"participant split overlap: {overlaps}")
    return {
        "participant_counts": {split: len(by_split[split]) for split in split_names},
        "record_counts": {
            split: sum(row.split == split for row in records) for split in split_names
        },
        "overlaps": overlaps,
    }


def _cache_selected_brats(
    records: Sequence[SliceRecord],
    *,
    output: Path,
    comparison: str,
    brats_root: Path,
    use_cache: bool,
) -> tuple[list[SliceRecord], dict[str, Any]]:
    selected = [row for row in records if row.label == 1]
    estimated = len(selected) * int(np.prod(MODEL_SHAPE)) * 4
    free = shutil.disk_usage(output).free
    budget = int(estimated * 1.30 + 128 * 1024 * 1024)
    if free < budget:
        raise RuntimeError(
            f"Insufficient free space for selected BraTS cache: need {budget} bytes, have {free}"
        )
    if not selected:
        return list(records), {"records": 0, "estimated_bytes": 0, "cache_files": []}
    try:
        import torch
        from andi_rewrite.data.datasets.brats import MRIDataVolume
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise ImportError("BraTS selected-slice cache requires torch, torchvision, nibabel") from exc

    dataset_cache: dict[str, Any] = {}
    groups: dict[tuple[str, str, str], list[SliceRecord]] = defaultdict(list)
    for row in selected:
        dataset_path = str(row.metadata.get("dataset_path", brats_root))
        source_subject = str(row.metadata.get("source_participant_id", row.case_id))
        groups[(dataset_path, source_subject, row.split)].append(row)
    updated_by_id: dict[str, SliceRecord] = {}
    cache_files: list[str] = []
    for (dataset_path_text, subject, split), rows in sorted(groups.items()):
        if dataset_path_text not in dataset_cache:
            dataset_cache[dataset_path_text] = MRIDataVolume(
                csv_path=None,
                dataset_path=Path(dataset_path_text),
                image_size=128,
                modalities=["flair", "t1", "t2"],
                segmentation_suffix="seg",
                filename_separator="_",
                return_metadata=True,
                intensity_normalization="robust_iqr",
            )
        dataset = dataset_cache[dataset_path_text]
        subject_ids = [str(value) for value in dataset.df.iloc[:, 0].tolist()]
        if subject not in subject_ids:
            raise ValueError(f"BraTS subject {subject} is absent from {dataset_path_text}")
        index = subject_ids.index(subject)
        volume, model_mask, metadata = dataset[index]
        volume = volume.detach().cpu().float()
        model_mask = model_mask.detach().cpu().bool()
        if tuple(volume.shape[:3]) != MODEL_SHAPE or not bool(torch.isfinite(volume).all()):
            raise ValueError(f"Invalid MRIDataVolume model shape/finite values for {subject}")
        rows = sorted(rows, key=lambda row: (row.z, row.record_id))
        images: list[np.ndarray] = []
        for row in rows:
            if row.z < 0 or row.z >= volume.shape[-1]:
                raise IndexError(f"BraTS z={row.z} outside model depth for {subject}")
            if bool(model_mask[..., row.z].any()):
                raise ValueError(f"Model-grid segmentation contains lesion at {subject}, z={row.z}")
            image = volume[..., row.z].numpy().astype(np.float32, copy=True)
            if image.shape != MODEL_SHAPE or not np.isfinite(image).all():
                raise ValueError(f"Invalid selected BraTS model input at {subject}, z={row.z}")
            images.append(image)
        pair_token = hashlib.sha256(
            f"{comparison}|{split}|{subject}".encode("utf-8")
        ).hexdigest()[:16]
        cache_path = output / "cache" / comparison / split / f"{pair_token}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if use_cache:
            np.savez_compressed(cache_path, images=np.stack(images, axis=0), z=np.asarray([row.z for row in rows], dtype=np.int16))
            cache_files.append(str(cache_path.resolve()))
            cache_hash = _sha256_file(cache_path)
        else:
            cache_hash = ""
        for cache_index, row in enumerate(rows):
            metadata_updates = {
                **row.metadata,
                "model_mask_checked": True,
                "model_mask_voxels": 0,
            }
            provenance_updates = {
                **row.provenance,
                "model_input_sha256": _sha256_bytes(images[cache_index].tobytes()),
            }
            if use_cache:
                metadata_updates.update(
                    {
                        "selected_cache_path": str(cache_path.resolve()),
                        "selected_cache_index": cache_index,
                        "selected_cache_sha256": cache_hash,
                    }
                )
            updated_by_id[row.record_id] = row.with_updates(
                model_mask_voxels=0,
                metadata=metadata_updates,
                provenance=provenance_updates,
            )
    updated = [updated_by_id.get(row.record_id, row) for row in records]
    return updated, {
        "records": len(selected),
        "estimated_bytes": estimated,
        "budget_bytes": budget,
        "cache_files": cache_files,
        "cache_enabled": bool(use_cache),
    }


def prepare_comparison(
    comparison: str,
    *,
    output: Path,
    roots: Mapping[str, Path],
    brats_csv: Path,
    seed: int = 73,
    bins: int = 20,
    cap_per_bin: int = 2,
    use_cache: bool = True,
    participant_limit: int | None = None,
) -> dict[str, Any]:
    if bins != 20:
        raise ValueError("Stage1 contract fixes normalized z to 20 bins")
    _progress(f"comparison={comparison} inventory_start")
    # Inventory only the cohort needed by this comparison.  In particular a
    # FOMO Stage1 run must not touch MPI/OASIS status files merely because
    # those roots are configured as defaults; those sources can be on a
    # separate ACL and are irrelevant until their own comparison is asked
    # for.  Mixed is the sole comparison that intentionally inventories all
    # standalone cohorts before joining source_entries.jsonl.
    healthy_names = (
        [comparison]
        if comparison in {"fomo45k", "mpi", "oasis3"}
        else ["fomo45k", "mpi", "oasis3"]
    )
    source_records: dict[str, list[SliceRecord]] = {}
    source_inventory: dict[str, Any] = {}
    for dataset_name in healthy_names:
        if dataset_name not in roots:
            continue
        rows, inventory = load_healthy_records(dataset_name, roots[dataset_name])
        source_records[dataset_name] = rows
        source_inventory[dataset_name] = inventory
        _progress(
            f"comparison={comparison} source={dataset_name} inventory_done "
            f"participants={len({row.participant_id for row in rows})} records={len(rows)}"
        )
    # Splits for standalone cohorts are defined once and then reused by Mixed.
    split_records: dict[str, list[SliceRecord]] = {}
    split_maps: dict[str, dict[str, str]] = {}
    for dataset_name, rows in source_records.items():
        updated, split_map = apply_participant_splits(rows, seed=seed)
        split_records[dataset_name] = updated
        split_maps[dataset_name] = split_map
    if comparison == "mixed":
        healthy, mixed_inventory = load_mixed_records(roots["mixed"], split_records)
        source_inventory["mixed"] = mixed_inventory
    elif comparison in split_records:
        healthy = split_records[comparison]
    else:
        raise ValueError(f"No healthy source records available for comparison {comparison}")
    brats, brats_inventory = load_brats_records(roots["brats"], brats_csv)
    _progress(
        f"comparison={comparison} source=brats21 inventory_done "
        f"participants={len({row.participant_id for row in brats})} records={len(brats)}"
    )
    brats, brats_split_map = apply_participant_splits(brats, seed=seed)
    split_maps["brats21"] = brats_split_map
    source_split_audit = {
        dataset_name: _split_audit(rows) for dataset_name, rows in split_records.items()
    }
    source_split_audit["brats21"] = _split_audit(brats)
    if participant_limit is not None:
        # Explicit smoke-only option; formal runs never cap subjects silently.
        selected = set(sorted({row.participant_id for row in healthy})[:participant_limit])
        healthy = [row for row in healthy if row.participant_id in selected]
        brats_selected = set(sorted({row.participant_id for row in brats})[:participant_limit])
        brats = [row for row in brats if row.participant_id in brats_selected]
    split_rows_for_csv = list(healthy) + list(brats)
    healthy, healthy_foreground_exclusions = apply_foreground_eligibility(
        healthy, threshold=FOREGROUND_THRESHOLD
    )
    brats, brats_foreground_exclusions = apply_foreground_eligibility(
        brats, threshold=FOREGROUND_THRESHOLD
    )
    _progress(
        f"comparison={comparison} foreground_done healthy_records={len(healthy)} "
        f"brats_records={len(brats)} healthy_exclusions={len(healthy_foreground_exclusions)} "
        f"brats_exclusions={len(brats_foreground_exclusions)}"
    )
    pairing = build_pairs(
        healthy,
        brats,
        comparison=comparison,
        seed=seed,
        bins=bins,
        cap_per_bin=cap_per_bin,
    )
    _progress(
        f"comparison={comparison} matching_done pairs={len(pairing.pairs)} "
        f"paired_records={len(pairing.records)} exclusions={len(pairing.exclusions)}"
    )
    paired_records, cache_inventory = _cache_selected_brats(
        pairing.records,
        output=output,
        comparison=comparison,
        brats_root=roots["brats"],
        use_cache=use_cache,
    )
    _progress(
        f"comparison={comparison} cache_done selected_records={cache_inventory['records']} "
        f"cache_files={len(cache_inventory['cache_files'])}"
    )
    audit = audit_records(
        paired_records,
        bins=bins,
        cap_per_bin=cap_per_bin,
        require_pairs=True,
        require_tumor_free=True,
    )
    audit["source_provenance"] = audit_source_provenance(paired_records)
    audit["pairing_exclusions"] = list(pairing.exclusions)
    audit["source_split_audit"] = source_split_audit
    audit["foreground_exclusions"] = {
        "healthy": healthy_foreground_exclusions,
        "brats21": brats_foreground_exclusions,
    }
    audit["unpaired_healthy_participants"] = sorted(
        set(row.participant_id for row in healthy).difference(
            row.participant_id for row in paired_records if row.label == 0
        )
    )
    audit["unpaired_brats_participants"] = sorted(
        set(row.participant_id for row in brats).difference(
            row.participant_id for row in paired_records if row.label == 1
        )
    )
    audit["native_tumor_free_records"] = sum(row.label == 1 for row in paired_records)
    audit["model_mask_false_records"] = sum(
        row.label == 1 and row.model_mask_voxels == 0 for row in paired_records
    )
    audit["foreground_eligibility"] = {
        "threshold": FOREGROUND_THRESHOLD,
        "healthy_metadata_checked": True,
        "selected_model_inputs_checked": True,
        "basis": "raw_positive_fraction_per_modality on each participant primary registered volume",
    }
    _progress(
        f"comparison={comparison} audit_done status={audit['status']} "
        f"pair_count={audit['pair_count']} records={audit['records']}"
    )
    comparison_dir = output / "manifests" / comparison
    for split in ("train", "val", "test"):
        write_records(comparison_dir / f"{split}.jsonl", [row for row in paired_records if row.split == split])
    _write_participant_splits(output, comparison, split_rows_for_csv)
    (output / "split_maps.json").write_text(
        json.dumps(split_maps, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": _now(),
        "comparison": comparison,
        "seed": seed,
        "bins": bins,
        "cap_per_bin": cap_per_bin,
        "modalities": list(MODALITIES),
        "model_shape": list(MODEL_SHAPE),
        "normalization": "robust_iqr",
        "normalize_input": False,
        "healthy_domain": comparison,
        "source_inventory": source_inventory,
        "brats_inventory": brats_inventory,
        "pairing": {"pairs": list(pairing.pairs), "exclusions": list(pairing.exclusions)},
        "cache": cache_inventory,
        "audit": audit,
        "manifest_paths": {
            split: str((comparison_dir / f"{split}.jsonl").resolve()) for split in ("train", "val", "test")
        },
        "smoke_only": participant_limit is not None,
        "participant_limit": participant_limit,
    }
    (output / "comparisons").mkdir(parents=True, exist_ok=True)
    (output / "comparisons" / f"{comparison}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _progress(f"comparison={comparison} manifests_written status=PASS")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=["fomo", "mpi", "oasis3", "mixed", "all"], default="fomo")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fomo-root", type=Path, default=DEFAULT_FOMO_ROOT)
    parser.add_argument("--mpi-root", type=Path, default=DEFAULT_MPI_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--mixed-root", type=Path, default=DEFAULT_MIXED_ROOT)
    parser.add_argument("--brats-root", type=Path, default=DEFAULT_BRATS_ROOT)
    parser.add_argument("--brats-csv", type=Path, default=DEFAULT_BRATS_CSV)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--cap-per-bin", type=int, choices=(1, 2), default=2)
    parser.add_argument("--participant-limit", type=int, default=None, help="explicit smoke-only participant cap")
    parser.add_argument("--no-cache", action="store_true", help="validate selected BraTS slices without writing cache")
    args = parser.parse_args(argv)
    if args.seed != 73:
        parser.error("Stage1 contract fixes seed=73")
    if args.participant_limit is not None and args.participant_limit < 1:
        parser.error("--participant-limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = {
        "fomo45k": args.fomo_root.resolve(),
        "mpi": args.mpi_root.resolve(),
        "oasis3": args.oasis_root.resolve(),
        "mixed": args.mixed_root.resolve(),
        "brats": args.brats_root.resolve(),
    }
    requested = {
        "fomo": "fomo45k",
        "mpi": "mpi",
        "oasis3": "oasis3",
        "mixed": "mixed",
    }
    comparisons = ["fomo45k", "mpi", "oasis3", "mixed"] if args.comparison == "all" else [requested[args.comparison]]
    all_results = []
    for comparison in comparisons:
        all_results.append(
            prepare_comparison(
                comparison,
                output=output,
                roots=roots,
                brats_csv=args.brats_csv.resolve(),
                seed=args.seed,
                bins=20,
                cap_per_bin=args.cap_per_bin,
                use_cache=not args.no_cache,
                participant_limit=args.participant_limit,
            )
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "seed": args.seed,
        "bins": 20,
        "cap_per_bin": args.cap_per_bin,
        "modalities": list(MODALITIES),
        "model_shape": list(MODEL_SHAPE),
        "normalization": "robust_iqr",
        "normalize_input": False,
        "foreground_threshold": FOREGROUND_THRESHOLD,
        "comparisons": comparisons,
        "roots": {key: str(value) for key, value in roots.items()},
        "brats_csv": file_fingerprint(args.brats_csv.resolve(), content=True),
        "cache_enabled": not args.no_cache,
        "participant_limit": args.participant_limit,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": _now(),
        "comparisons": {result["comparison"]: result["audit"] for result in all_results},
        "source_inventories": {result["comparison"]: result["source_inventory"] for result in all_results},
        "brats_inventories": {result["comparison"]: result["brats_inventory"] for result in all_results},
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output), "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
