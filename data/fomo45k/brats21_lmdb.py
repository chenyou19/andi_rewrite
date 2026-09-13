"""Build an auditable LMDB from the registered FOMO45K SRI24 publication.

The model-facing contract intentionally matches native BraTS21 inference:

* channels are ``FLAIR, T1, T2``;
* each modality is divided by its positive-foreground 99th percentile;
* values above one are not clipped;
* axial slices are resized to the configured 2-D model size.

The source NIfTI publication is read-only.  A complete build is written to a
sibling staging directory, audited record-by-record, and only then atomically
published at the requested output path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import lmdb
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from andi_rewrite.data.datasets.imaging import normalize_volume

from .brats21 import BRATS_SHAPE, BRATS_SPACING_MM, MODEL_CHANNEL_ORDER


SCHEMA_VERSION = 1
CHANNEL_ORDER = tuple(name.upper() for name in MODEL_CHANNEL_ORDER)
NORMALIZATION = "per_modality_positive_foreground_p99_no_clip"
DEFAULT_SEED = 73
DEFAULT_VALIDATION_FRACTION = 0.10
EXPECTED_PARTICIPANTS = 240
EXPECTED_SESSIONS = 243
EXPECTED_SLICES = 34_153


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_like_brats21(volume: torch.Tensor) -> torch.Tensor:
    """Apply the exact p99 implementation used by BraTS21 inference."""

    if volume.ndim != 4:
        raise ValueError(f"Expected [C,H,W,Z], found shape={tuple(volume.shape)}")
    normalized = normalize_volume(volume.float())
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("BraTS21 p99 normalization produced NaN/Inf.")
    return normalized


def split_participants(
    participant_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[set[str], set[str]]:
    """Return deterministic participant-level train/validation sets."""

    ordered = list(dict.fromkeys(str(value) for value in participant_ids))
    if len(ordered) < 2:
        raise ValueError("At least two unique participants are required.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    shuffled = np.asarray(ordered, dtype=object)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    validation_count = max(1, int(round(len(ordered) * float(validation_fraction))))
    validation = {str(value) for value in shuffled[:validation_count]}
    training = set(ordered).difference(validation)
    if not training or training.intersection(validation):
        raise RuntimeError("Invalid participant split.")
    return training, validation


def _resolve_source_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_source_manifests(
    source_root: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    require_expected_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    """Validate publication manifests and attach a participant-level split."""

    root = Path(source_root).resolve()
    dataset_manifest = root / "dataset_manifest.csv"
    slice_manifest = root / "slice_manifest.csv"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    if not slice_manifest.is_file():
        raise FileNotFoundError(slice_manifest)

    with dataset_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "case_id",
        "participant_id",
        "session_id",
        "processing_status",
        "flair_output",
        "t1_output",
        "t2_output",
    }
    missing = required.difference(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    pass_rows = [row for row in rows if row["processing_status"] == "PASS"]
    if not pass_rows:
        raise ValueError("Dataset manifest contains no PASS sessions.")
    case_ids = [row["case_id"] for row in pass_rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate PASS case_id values in dataset manifest.")

    participants = [row["participant_id"] for row in pass_rows]
    training, validation = split_participants(
        participants,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    prepared: list[dict[str, Any]] = []
    for row in pass_rows:
        item = dict(row)
        item["split"] = "val" if row["participant_id"] in validation else "train"
        item["paths"] = {
            modality: _resolve_source_path(root, row[f"{modality}_output"])
            for modality in MODEL_CHANNEL_ORDER
        }
        prepared.append(item)

    slices_by_case: dict[str, list[int]] = defaultdict(list)
    with slice_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        slice_rows = list(csv.DictReader(handle))
    if not slice_rows or not {"case_id", "slice"}.issubset(slice_rows[0]):
        raise ValueError("Invalid slice_manifest.csv; expected case_id,slice.")
    valid_cases = set(case_ids)
    seen_slices: set[tuple[str, int]] = set()
    for row in slice_rows:
        case_id = row["case_id"]
        if case_id not in valid_cases:
            raise ValueError(f"Slice manifest references a non-PASS case: {case_id}")
        z = int(row["slice"])
        if not 0 <= z < BRATS_SHAPE[2]:
            raise ValueError(f"Invalid axial slice for {case_id}: {z}")
        identity = (case_id, z)
        if identity in seen_slices:
            raise ValueError(f"Duplicate slice manifest entry: {case_id}, z={z}")
        seen_slices.add(identity)
        slices_by_case[case_id].append(z)
    missing_cases = sorted(valid_cases.difference(slices_by_case))
    if missing_cases:
        raise ValueError(f"PASS sessions without eligible slices: {missing_cases[:8]}")

    split_session_counts = {
        split: sum(row["split"] == split for row in prepared) for split in ("train", "val")
    }
    split_slice_counts = {
        split: sum(
            len(slices_by_case[row["case_id"]])
            for row in prepared
            if row["split"] == split
        )
        for split in ("train", "val")
    }
    report = {
        "status": "PASS",
        "validated_at": _now(),
        "source_root": str(root),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "slice_manifest": str(slice_manifest),
        "slice_manifest_sha256": _sha256(slice_manifest),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "participant_count": len(set(participants)),
        "session_count": len(prepared),
        "slice_count": len(slice_rows),
        "split_subject_counts": {"train": len(training), "val": len(validation)},
        "split_session_counts": split_session_counts,
        "split_slice_counts": split_slice_counts,
        "train_validation_overlap": sorted(training.intersection(validation)),
    }
    if require_expected_counts:
        expected = {
            "participant_count": EXPECTED_PARTICIPANTS,
            "session_count": EXPECTED_SESSIONS,
            "slice_count": EXPECTED_SLICES,
        }
        for name, value in expected.items():
            if int(report[name]) != value:
                raise ValueError(f"{name} mismatch: expected {value}, found {report[name]}.")
    return prepared, dict(slices_by_case), report


def _load_case(row: Mapping[str, Any]) -> torch.Tensor:
    arrays: list[np.ndarray] = []
    reference_affine: np.ndarray | None = None
    for modality in MODEL_CHANNEL_ORDER:
        path = Path(row["paths"][modality])
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        affine = np.asarray(image.affine, dtype=np.float64)
        if shape != BRATS_SHAPE:
            raise ValueError(f"Unexpected shape for {row['case_id']}/{modality}: {shape}")
        if not np.allclose(spacing, BRATS_SPACING_MM, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"Unexpected spacing for {row['case_id']}/{modality}: {spacing}")
        if not np.all(np.isfinite(affine)):
            raise ValueError(f"Non-finite affine for {row['case_id']}/{modality}")
        if reference_affine is None:
            reference_affine = affine
        elif not np.allclose(affine, reference_affine, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"Modality affine mismatch for {row['case_id']}/{modality}")
        array = np.asarray(image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"NaN/Inf in {row['case_id']}/{modality}")
        arrays.append(array)
    return normalize_like_brats21(torch.from_numpy(np.stack(arrays, axis=0)))


def _resize_slices(volume: torch.Tensor, image_size: int) -> torch.Tensor:
    slices = volume.permute(3, 0, 1, 2).contiguous()
    if tuple(slices.shape[-2:]) == (image_size, image_size):
        return slices
    return F.interpolate(
        slices,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _map_size(entry_count: int, image_size: int) -> int:
    sample = np.zeros((len(CHANNEL_ORDER), image_size, image_size), dtype=np.float32)
    serialized = len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
    return max(int(entry_count * serialized * 1.5) + (128 << 20), 1 << 30)


def _fingerprint(validation: Mapping[str, Any], image_size: int) -> str:
    specification = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset_manifest_sha256": validation["dataset_manifest_sha256"],
        "source_slice_manifest_sha256": validation["slice_manifest_sha256"],
        "channel_order": list(CHANNEL_ORDER),
        "normalization": NORMALIZATION,
        "normalization_implementation": "andi_rewrite.data.datasets.imaging.normalize_volume",
        "normalization_clip": False,
        "image_size": int(image_size),
        "seed": int(validation["seed"]),
        "validation_fraction": float(validation["validation_fraction"]),
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _open_write_lmdb(path: Path, map_size: int) -> lmdb.Environment:
    path.mkdir(parents=True, exist_ok=False)
    return lmdb.open(str(path), map_size=int(map_size), subdir=True, lock=True, sync=True)


def audit_output(
    output_root: str | Path,
    *,
    write_report: bool = True,
    declared_publish_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read every value and verify manifests, shapes, finiteness, and split isolation."""

    root = Path(output_root).resolve()
    build_manifest_path = root / "build_manifest.json"
    if not build_manifest_path.is_file():
        raise FileNotFoundError(build_manifest_path)
    manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channel_order") != list(CHANNEL_ORDER):
        raise ValueError("LMDB channel order is not [FLAIR,T1,T2].")
    processing = manifest.get("processing_specification", {})
    if processing.get("normalization") != NORMALIZATION or processing.get("clip") is not False:
        raise ValueError("LMDB normalization contract does not match BraTS21 inference.")
    image_size = int(processing["image_size"])
    expected_shape = (len(CHANNEL_ORDER), image_size, image_size)
    split_subjects: dict[str, set[str]] = {"train": set(), "val": set()}
    split_reports: dict[str, Any] = {}

    for split in ("train", "val"):
        manifest_path = root / "manifests" / f"{split}_entries.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows):
            if row.get("key") != f"{index:08d}" or row.get("split") != split:
                raise ValueError(f"Invalid {split} entry manifest row {index}.")
            split_subjects[split].add(row["participant_id"])
        environment = lmdb.open(
            str(root / split),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        count = 0
        minimum = float("inf")
        maximum = float("-inf")
        try:
            with environment.begin(write=False) as transaction:
                with transaction.cursor() as cursor:
                    for key_bytes, value_bytes in cursor:
                        expected_key = f"{count:08d}".encode("ascii")
                        if key_bytes != expected_key:
                            raise ValueError(
                                f"Non-contiguous LMDB key in {split}: {key_bytes!r} != {expected_key!r}"
                            )
                        value = np.asarray(pickle.loads(value_bytes))
                        if value.shape != expected_shape or value.dtype != np.float32:
                            raise ValueError(
                                f"Invalid {split}/{expected_key!r}: shape={value.shape}, dtype={value.dtype}"
                            )
                        if not np.all(np.isfinite(value)):
                            raise ValueError(f"NaN/Inf in {split}/{expected_key!r}")
                        minimum = min(minimum, float(value.min()))
                        maximum = max(maximum, float(value.max()))
                        count += 1
        finally:
            environment.close()
        if count != len(rows) or count != int(manifest["split_entry_counts"][split]):
            raise ValueError(f"{split} LMDB/manifest count mismatch.")
        split_reports[split] = {
            "entries": count,
            "minimum": minimum,
            "maximum": maximum,
            "shape": list(expected_shape),
            "dtype": "float32",
        }
    overlap = sorted(split_subjects["train"].intersection(split_subjects["val"]))
    if overlap:
        raise ValueError(f"Participant leakage between train and val: {overlap[:8]}")
    report = {
        "status": "PASS",
        "audited_at": _now(),
        "audited_root": str(root),
        "declared_publish_root": str(Path(declared_publish_root).resolve())
        if declared_publish_root is not None
        else str(root),
        "channel_order": list(CHANNEL_ORDER),
        "normalization": NORMALIZATION,
        "processing_fingerprint": manifest["processing_fingerprint"],
        "split_subject_counts": {name: len(values) for name, values in split_subjects.items()},
        "split_lmdb": split_reports,
        "total_entries": sum(value["entries"] for value in split_reports.values()),
    }
    if write_report:
        _json_dump(root / "audit_report.json", report)
    return report


def build_output(
    source_root: str | Path,
    output_root: str | Path,
    *,
    image_size: int = 128,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Build, audit, and atomically publish a new train/val LMDB root."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)

    rows, slices_by_case, validation = read_source_manifests(
        source_root,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    _json_dump(stage / "source_validation.json", validation)
    (stage / "manifests").mkdir(parents=False, exist_ok=False)
    entry_counts = validation["split_slice_counts"]
    map_sizes = {split: _map_size(int(entry_counts[split]), image_size) for split in ("train", "val")}
    environments = {
        split: _open_write_lmdb(stage / split, map_sizes[split]) for split in ("train", "val")
    }
    entry_handles: dict[str, Any] = {}
    entry_writers: dict[str, csv.DictWriter] = {}
    session_handle = None
    counts = {"train": 0, "val": 0}
    fieldnames = ["key", "participant_id", "session_id", "case_id", "z", "split"]
    try:
        for split in ("train", "val"):
            handle = (stage / "manifests" / f"{split}_entries.csv").open(
                "w", encoding="utf-8", newline=""
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            entry_handles[split] = handle
            entry_writers[split] = writer
        session_handle = (stage / "manifests" / "sessions.jsonl").open("w", encoding="utf-8")

        total_sessions = len(rows)
        for session_index, row in enumerate(rows, start=1):
            split = str(row["split"])
            z_indices = slices_by_case[row["case_id"]]
            volume = _load_case(row)
            resized = _resize_slices(volume, image_size)
            start_index = counts[split]
            with environments[split].begin(write=True) as transaction:
                for z in z_indices:
                    key = f"{counts[split]:08d}"
                    value = resized[z].detach().cpu().numpy().astype(np.float32, copy=False)
                    inserted = transaction.put(
                        key.encode("ascii"),
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                        overwrite=False,
                    )
                    if not inserted:
                        raise RuntimeError(f"Duplicate LMDB key in {split}: {key}")
                    entry_writers[split].writerow(
                        {
                            "key": key,
                            "participant_id": row["participant_id"],
                            "session_id": row["session_id"],
                            "case_id": row["case_id"],
                            "z": z,
                            "split": split,
                        }
                    )
                    counts[split] += 1
            session_handle.write(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "participant_id": row["participant_id"],
                        "session_id": row["session_id"],
                        "split": split,
                        "slice_count": len(z_indices),
                        "start_index": start_index,
                        "end_index_exclusive": counts[split],
                        "input_paths": {
                            modality.upper(): str(row["paths"][modality])
                            for modality in MODEL_CHANNEL_ORDER
                        },
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            print(
                f"Built session {session_index}/{total_sessions}: {row['case_id']} "
                f"split={split} slices={len(z_indices)}",
                flush=True,
            )
    finally:
        for handle in entry_handles.values():
            handle.close()
        if session_handle is not None:
            session_handle.close()
        for environment in environments.values():
            environment.sync()
            environment.close()

    if counts != {key: int(value) for key, value in entry_counts.items()}:
        raise RuntimeError(f"Built entry counts do not match validation: {counts} != {entry_counts}")
    processing = {
        "channel_order": list(CHANNEL_ORDER),
        "normalization": NORMALIZATION,
        "normalization_implementation": "andi_rewrite.data.datasets.imaging.normalize_volume",
        "foreground_definition": "voxel > 0 per modality",
        "percentile": 99.0,
        "clip": False,
        "source_shape": list(BRATS_SHAPE),
        "source_spacing_mm": list(BRATS_SPACING_MM),
        "slice_axis": 2,
        "image_size": int(image_size),
        "resize_mode": "bilinear",
        "resize_align_corners": False,
        "resize_antialias": True,
        "dtype": "float32",
        "mask_channel": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "source_root": str(source_root),
        "publish_root": str(output_root),
        "channel_order": list(CHANNEL_ORDER),
        "mask_channel": False,
        "processing_specification": processing,
        "processing_fingerprint": _fingerprint(validation, image_size),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "participant_count": int(validation["participant_count"]),
        "session_count": int(validation["session_count"]),
        "split_subject_counts": validation["split_subject_counts"],
        "split_session_counts": validation["split_session_counts"],
        "split_entry_counts": counts,
        "split_map_sizes": map_sizes,
    }
    _json_dump(stage / "build_manifest.json", manifest)
    audit = audit_output(stage, write_report=True, declared_publish_root=output_root)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Staging audit failed: {audit}")
    _json_dump(
        stage / "publication.json",
        {
            "status": "PASS",
            "published_at": _now(),
            "publish_root": str(output_root),
            "processing_fingerprint": manifest["processing_fingerprint"],
            "audit_status": audit["status"],
        },
    )
    os.replace(stage, output_root)
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "processing_fingerprint": manifest["processing_fingerprint"],
        "split_entry_counts": counts,
        "split_subject_counts": validation["split_subject_counts"],
        "audit_status": audit["status"],
    }


def dataset_status(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    return {
        "output_root": str(root),
        "exists": root.is_dir(),
        "files": {
            name: (root / name).is_file()
            for name in ("publication.json", "build_manifest.json", "audit_report.json")
        },
        "train_lmdb": (root / "train" / "data.mdb").is_file(),
        "val_lmdb": (root / "val" / "data.mdb").is_file(),
        "staging_directories": [
            str(path)
            for path in sorted(root.parent.glob(f".{root.name}.staging-*"))
            if path.is_dir()
        ],
    }


__all__ = [
    "CHANNEL_ORDER",
    "NORMALIZATION",
    "audit_output",
    "build_output",
    "dataset_status",
    "normalize_like_brats21",
    "read_source_manifests",
    "split_participants",
]
