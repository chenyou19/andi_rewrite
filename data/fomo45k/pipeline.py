"""Fail-closed FOMO45K validation and LPS LMDB publication.

The source NIfTI volumes are reoriented with nibabel's discrete orientation
primitives.  No interpolation or resampling is performed during RAS-to-LPS
conversion.  The only interpolation in this pipeline is the explicit 2-D
256-to-128 model resize after orientation and symmetric full-FOV padding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CHANNEL_ORDER = ("FLAIR", "T1", "T2")
FILENAME_COLUMNS = {
    "FLAIR": "FLAIR_filename",
    "T1": "T1_filename",
    "T2": "T2_filename",
}
REPO_PATH_COLUMNS = {
    "FLAIR": "FLAIR_repo_path",
    "T1": "T1_repo_path",
    "T2": "T2_repo_path",
}
EXPECTED_DATASET = "PT007_NIMH"
EXPECTED_HF_REVISION = "bf2bb12bd5cfeaed65e4a003e72d55fb8322c28a"
DEFAULT_VALIDATION_SUBJECTS = (
    "sub_11138",
    "sub_9278",
    "sub_9898",
    "sub_9337",
    "sub_3550",
    "sub_4209",
    "sub_4355",
    "sub_2765",
    "sub_6818",
    "sub_4045",
    "sub_1215",
    "sub_9173",
)


@dataclass(frozen=True)
class DatasetExpectations:
    subjects: int | None = 240
    sessions: int | None = 243
    files: int | None = 729
    train_subjects: int | None = 228
    validation_subjects: int | None = 12
    total_slices: int | None = 35_575
    train_slices: int | None = 33_798
    validation_slices: int | None = 1_777
    source_axcodes: str | None = "RAS"
    hf_revision: str | None = EXPECTED_HF_REVISION


DEFAULT_EXPECTATIONS = DatasetExpectations()


@dataclass(frozen=True)
class FOMOSessionRecord:
    csv_index: int
    dataset: str
    participant_id: str
    session_id: str
    sex: str
    age: float | None
    group: str
    flair_path: Path
    t1_path: Path
    t2_path: Path
    hf_metadata_paths: tuple[Path, Path, Path]
    hf_blob_sha256: tuple[str, str, str]
    split: str

    @property
    def identifier(self) -> str:
        return f"{self.participant_id}/{self.session_id}"

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "FLAIR": self.flair_path,
            "T1": self.t1_path,
            "T2": self.t2_path,
        }

    @property
    def metadata_paths(self) -> dict[str, Path]:
        return dict(zip(CHANNEL_ORDER, self.hf_metadata_paths))

    @property
    def blob_digests(self) -> dict[str, str]:
        return dict(zip(CHANNEL_ORDER, self.hf_blob_sha256))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, found {actual}.")


def _orientation_text(affine: np.ndarray) -> str:
    return "".join(str(value) for value in nib.aff2axcodes(affine))


def _read_hf_metadata(path: Path, expected_revision: str | None) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face download metadata is missing: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 2:
        raise ValueError(f"Malformed Hugging Face download metadata: {path}")
    revision, blob_digest = lines[:2]
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"Hugging Face revision mismatch for {path}: "
            f"expected {expected_revision}, found {revision}."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", blob_digest):
        raise ValueError(f"Invalid Hugging Face blob digest in {path}: {blob_digest!r}")
    return revision, blob_digest


def read_session_records(
    source_root: str | Path,
    metadata_tsv: str | Path,
    *,
    validation_subjects: Sequence[str] = DEFAULT_VALIDATION_SUBJECTS,
    expectations: DatasetExpectations = DEFAULT_EXPECTATIONS,
) -> list[FOMOSessionRecord]:
    """Resolve all TSV paths, split subjects, and validate HF provenance."""

    source_root = Path(source_root).resolve()
    metadata_tsv = Path(metadata_tsv).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FOMO source root does not exist: {source_root}")
    if not metadata_tsv.is_file():
        raise FileNotFoundError(f"FOMO metadata TSV does not exist: {metadata_tsv}")

    frame = pd.read_csv(metadata_tsv, sep="\t", dtype={"participant_id": str, "session_id": str})
    required = {
        "dataset",
        "participant_id",
        "session_id",
        "sex",
        "age",
        "group",
        *FILENAME_COLUMNS.values(),
        *REPO_PATH_COLUMNS.values(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"FOMO metadata TSV is missing required columns: {missing}")
    _require_equal("session count", len(frame), expectations.sessions)

    datasets = sorted(str(value) for value in frame["dataset"].dropna().unique())
    if datasets != [EXPECTED_DATASET]:
        raise ValueError(f"Expected only dataset={EXPECTED_DATASET}, found {datasets}.")
    groups = sorted(str(value) for value in frame["group"].dropna().unique())
    if groups != ["Control"]:
        raise ValueError(f"Expected only healthy Control rows, found groups={groups}.")
    if frame[["participant_id", "session_id"]].duplicated().any():
        duplicate = frame.loc[
            frame[["participant_id", "session_id"]].duplicated(keep=False),
            ["participant_id", "session_id"],
        ]
        raise ValueError(f"Duplicate participant/session rows: {duplicate.to_dict('records')}")

    subject_ids = {str(value) for value in frame["participant_id"]}
    _require_equal("subject count", len(subject_ids), expectations.subjects)
    validation_set = {str(value) for value in validation_subjects}
    missing_validation = sorted(validation_set.difference(subject_ids))
    if missing_validation:
        raise ValueError(f"Validation subjects are missing from the TSV: {missing_validation}")
    _require_equal("validation subject count", len(validation_set), expectations.validation_subjects)
    _require_equal("train subject count", len(subject_ids - validation_set), expectations.train_subjects)

    cache_root = source_root.parent / ".cache" / "huggingface" / "download"
    records: list[FOMOSessionRecord] = []
    revisions: set[str] = set()
    for csv_index, row in frame.iterrows():
        participant_id = str(row["participant_id"])
        session_id = str(row["session_id"])
        dataset = str(row["dataset"])
        paths: dict[str, Path] = {}
        metadata_paths: dict[str, Path] = {}
        blob_digests: dict[str, str] = {}
        for modality in CHANNEL_ORDER:
            filename = str(row[FILENAME_COLUMNS[modality]])
            path = source_root / participant_id / session_id / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {modality} for {participant_id}/{session_id}: {path}"
                )
            repo_path = Path(str(row[REPO_PATH_COLUMNS[modality]]).replace("/", os.sep))
            expected_repo_path = Path(dataset) / participant_id / session_id / filename
            if repo_path != expected_repo_path:
                raise ValueError(
                    f"TSV repo path mismatch for {participant_id}/{session_id}/{modality}: "
                    f"{repo_path} != {expected_repo_path}."
                )
            metadata_path = cache_root / repo_path.parent / f"{repo_path.name}.metadata"
            revision, blob_digest = _read_hf_metadata(metadata_path, expectations.hf_revision)
            revisions.add(revision)
            paths[modality] = path.resolve()
            metadata_paths[modality] = metadata_path.resolve()
            blob_digests[modality] = blob_digest

        records.append(
            FOMOSessionRecord(
                csv_index=int(csv_index),
                dataset=dataset,
                participant_id=participant_id,
                session_id=session_id,
                sex=str(row["sex"]),
                age=None if pd.isna(row["age"]) else float(row["age"]),
                group=str(row["group"]),
                flair_path=paths["FLAIR"],
                t1_path=paths["T1"],
                t2_path=paths["T2"],
                hf_metadata_paths=tuple(metadata_paths[name] for name in CHANNEL_ORDER),
                hf_blob_sha256=tuple(blob_digests[name] for name in CHANNEL_ORDER),
                split="val" if participant_id in validation_set else "train",
            )
        )

    _require_equal("NIfTI file count", len(records) * len(CHANNEL_ORDER), expectations.files)
    if expectations.hf_revision is not None and revisions != {expectations.hf_revision}:
        raise ValueError(f"Unexpected set of HF revisions: {sorted(revisions)}")
    return records


def _load_modalities(record: FOMOSessionRecord) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    for modality, path in record.paths.items():
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3-D {modality} NIfTI for {record.identifier}: {image.shape}")
        affine = np.asarray(image.affine, dtype=np.float64)
        if not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) < 1.0e-12:
            raise ValueError(f"Invalid affine for {record.identifier}/{modality}: {path}")
        array = np.asarray(image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"NaN/Inf found in {record.identifier}/{modality}: {path}")
        if reference_shape is None:
            reference_shape = tuple(int(value) for value in array.shape)
            reference_affine = affine
        elif tuple(array.shape) != reference_shape or not np.allclose(
            affine, reference_affine, rtol=0.0, atol=1.0e-5
        ):
            raise ValueError(
                f"Modality grid mismatch for {record.identifier}/{modality}: "
                f"shape={array.shape}/{reference_shape}, "
                f"affine_max_error={float(np.max(np.abs(affine - reference_affine))):.6g}."
            )
        values[modality] = array
    assert reference_affine is not None
    return values, reference_affine


def validate_dataset(
    source_root: str | Path,
    metadata_tsv: str | Path,
    *,
    validation_subjects: Sequence[str] = DEFAULT_VALIDATION_SUBJECTS,
    expectations: DatasetExpectations = DEFAULT_EXPECTATIONS,
    pad_size: int = 256,
) -> tuple[list[FOMOSessionRecord], dict[str, Any]]:
    """Perform a full voxel, geometry, split, and provenance validation pass."""

    records = read_session_records(
        source_root,
        metadata_tsv,
        validation_subjects=validation_subjects,
        expectations=expectations,
    )
    session_reports: list[dict[str, Any]] = []
    slice_counts = {"train": 0, "val": 0}
    shape_counts: dict[str, int] = {}
    orientation_counts: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        images, affine = _load_modalities(record)
        shape = tuple(int(value) for value in images["T1"].shape)
        if shape[0] > pad_size or shape[1] > pad_size:
            raise ValueError(
                f"{record.identifier} XY shape {shape[:2]} exceeds the fixed {pad_size}x{pad_size} FOV."
            )
        axcodes = _orientation_text(affine)
        if expectations.source_axcodes is not None and axcodes != expectations.source_axcodes:
            raise ValueError(
                f"Source orientation mismatch for {record.identifier}: "
                f"expected {expectations.source_axcodes}, found {axcodes}."
            )
        nonempty_z = np.flatnonzero(np.any(images["T1"] > 0, axis=(0, 1)))
        if nonempty_z.size == 0:
            raise ValueError(f"T1 contains no positive axial slices: {record.identifier}")
        count = int(nonempty_z.size)
        slice_counts[record.split] += count
        shape_text = "x".join(str(value) for value in shape)
        shape_counts[shape_text] = shape_counts.get(shape_text, 0) + 1
        orientation_counts[axcodes] = orientation_counts.get(axcodes, 0) + 1
        session_reports.append(
            {
                "csv_index": record.csv_index,
                "participant_id": record.participant_id,
                "session_id": record.session_id,
                "split": record.split,
                "sex": record.sex,
                "age": record.age,
                "shape": list(shape),
                "source_axcodes": axcodes,
                "source_affine": affine.astype(float).tolist(),
                "slice_count": count,
                "first_z": int(nonempty_z[0]),
                "last_z": int(nonempty_z[-1]),
                "positive_t1_voxels": int(np.count_nonzero(images["T1"] > 0)),
            }
        )
        if index % 10 == 0 or index == len(records):
            print(f"Validated {index}/{len(records)} FOMO sessions", flush=True)

    _require_equal("train slice count", slice_counts["train"], expectations.train_slices)
    _require_equal("validation slice count", slice_counts["val"], expectations.validation_slices)
    _require_equal("total slice count", sum(slice_counts.values()), expectations.total_slices)
    train_subjects = {record.participant_id for record in records if record.split == "train"}
    validation_set = {record.participant_id for record in records if record.split == "val"}
    if train_subjects.intersection(validation_set):
        raise ValueError("Subject leakage detected between train and validation splits.")
    validation_rows = {
        record.participant_id: record for record in records if record.participant_id in validation_set
    }
    validation_sex_counts = {
        sex: sum(record.sex == sex for record in validation_rows.values()) for sex in ("F", "M")
    }

    report = {
        "status": "PASS",
        "validated_at": _now(),
        "source_root": str(Path(source_root).resolve()),
        "metadata_tsv": str(Path(metadata_tsv).resolve()),
        "metadata_tsv_sha256": _sha256(Path(metadata_tsv).resolve()),
        "dataset": EXPECTED_DATASET,
        "hf_revision": expectations.hf_revision,
        "channel_order": list(CHANNEL_ORDER),
        "mask_policy": "none; T1 > 0 is used only for axial-slice eligibility",
        "subject_count": len(train_subjects | validation_set),
        "session_count": len(records),
        "nifti_file_count": len(records) * len(CHANNEL_ORDER),
        "hf_metadata_count": len(records) * len(CHANNEL_ORDER),
        "split_subject_counts": {
            "train": len(train_subjects),
            "val": len(validation_set),
        },
        "split_session_counts": {
            "train": sum(record.split == "train" for record in records),
            "val": sum(record.split == "val" for record in records),
        },
        "split_slice_counts": slice_counts,
        "total_slice_count": sum(slice_counts.values()),
        "validation_subjects": list(validation_subjects),
        "validation_sex_counts": validation_sex_counts,
        "missing_age_count": sum(record.age is None for record in records),
        "shape_counts": dict(sorted(shape_counts.items())),
        "source_orientation_counts": dict(sorted(orientation_counts.items())),
        "sessions": session_reports,
    }
    return records, report


def reorient_volume(
    volume: np.ndarray,
    affine: np.ndarray,
    target_axcodes: Sequence[str] = ("L", "P", "S"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reindex one 3-D volume into target voxel orientation without interpolation."""

    value = np.asarray(volume)
    if value.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, found shape={value.shape}.")
    source_ornt = nib.orientations.io_orientation(np.asarray(affine, dtype=np.float64))
    target_ornt = nib.orientations.axcodes2ornt(tuple(str(code) for code in target_axcodes))
    transform = nib.orientations.ornt_transform(source_ornt, target_ornt)
    reoriented = np.ascontiguousarray(nib.orientations.apply_orientation(value, transform))
    reoriented_affine = np.asarray(affine, dtype=np.float64) @ nib.orientations.inv_ornt_aff(
        transform, value.shape
    )
    actual = tuple(str(code) for code in nib.aff2axcodes(reoriented_affine))
    expected = tuple(str(code) for code in target_axcodes)
    if actual != expected:
        raise RuntimeError(f"Reorientation failed: expected {expected}, found {actual}.")
    return reoriented, reoriented_affine, transform


def _float32_multiset_signature(value: np.ndarray) -> dict[str, int]:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32).reshape(-1)
    return {
        "count": int(bits.size),
        "xor_u32": int(np.bitwise_xor.reduce(bits, initial=np.uint32(0))),
        "sum_u32_mod_2_64": int(np.sum(bits, dtype=np.uint64)),
        "nonzero": int(np.count_nonzero(bits)),
    }


def normalize_nonzero_p99(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clip one MRI modality to [0, 1] using its nonzero p99 scale."""

    value = np.asarray(volume, dtype=np.float32)
    foreground = value != 0
    foreground_values = value[foreground]
    if foreground_values.size == 0:
        raise ValueError("Cannot p99-normalize an all-zero modality.")
    p99 = float(np.percentile(foreground_values, 99.0))
    if not np.isfinite(p99) or p99 <= 0:
        raise ValueError(f"Invalid nonzero p99 intensity: {p99}")
    normalized = np.clip(value / p99, 0.0, 1.0).astype(np.float32, copy=False)
    normalized[~foreground] = 0.0
    if not np.all(np.isfinite(normalized)):
        raise ValueError("p99 normalization produced NaN/Inf.")
    return normalized, {
        "nonzero_voxels": int(foreground_values.size),
        "source_min": float(value.min()),
        "source_max": float(value.max()),
        "nonzero_p99": p99,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
    }


def _pad_xy(value: np.ndarray, pad_size: int) -> tuple[np.ndarray, dict[str, int]]:
    if value.ndim != 4:
        raise ValueError(f"Expected [C,X,Y,Z], found {value.shape}.")
    x, y = int(value.shape[1]), int(value.shape[2])
    if x > pad_size or y > pad_size:
        raise ValueError(f"XY shape {(x, y)} exceeds fixed pad size {pad_size}.")
    x_before = (pad_size - x) // 2
    x_after = pad_size - x - x_before
    y_before = (pad_size - y) // 2
    y_after = pad_size - y - y_before
    padded = np.pad(
        value,
        ((0, 0), (x_before, x_after), (y_before, y_after), (0, 0)),
        mode="constant",
    )
    return padded, {
        "x_before": x_before,
        "x_after": x_after,
        "y_before": y_before,
        "y_after": y_after,
    }


def process_session(
    record: FOMOSessionRecord,
    *,
    target_axcodes: Sequence[str] = ("L", "P", "S"),
    pad_size: int = 256,
    image_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reorient, normalize, pad, and resize all eligible slices in one session."""

    images, source_affine = _load_modalities(record)
    source_shape = tuple(int(value) for value in images["T1"].shape)
    source_axcodes = _orientation_text(source_affine)
    reoriented: dict[str, np.ndarray] = {}
    intensity_signatures: dict[str, Any] = {}
    transform: np.ndarray | None = None
    target_affine: np.ndarray | None = None
    for modality in CHANNEL_ORDER:
        before_signature = _float32_multiset_signature(images[modality])
        value, current_affine, current_transform = reorient_volume(
            images[modality], source_affine, target_axcodes
        )
        after_signature = _float32_multiset_signature(value)
        if before_signature != after_signature:
            raise RuntimeError(
                f"Discrete orientation changed the intensity multiset for {record.identifier}/{modality}."
            )
        if transform is None:
            transform = current_transform
            target_affine = current_affine
        elif not np.array_equal(current_transform, transform) or not np.allclose(
            current_affine, target_affine, rtol=0.0, atol=1.0e-8
        ):
            raise RuntimeError(f"Modalities received different orientation transforms: {record.identifier}")
        reoriented[modality] = value
        intensity_signatures[modality] = before_signature
    assert transform is not None and target_affine is not None
    if not np.array_equal(transform[2], np.asarray([2.0, 1.0])):
        raise ValueError(
            f"Target orientation changes the superior slice axis for {record.identifier}: {transform.tolist()}"
        )

    eligible_z = np.flatnonzero(np.any(reoriented["T1"] > 0, axis=(0, 1))).astype(np.int32)
    if eligible_z.size == 0:
        raise ValueError(f"No eligible T1 slices after reorientation: {record.identifier}")
    normalized: list[np.ndarray] = []
    normalization: dict[str, Any] = {}
    for modality in CHANNEL_ORDER:
        value, statistics = normalize_nonzero_p99(reoriented[modality])
        normalized.append(value)
        normalization[modality] = statistics
    stacked = np.stack(normalized, axis=0).astype(np.float32, copy=False)
    padded, padding = _pad_xy(stacked, int(pad_size))
    selected = np.transpose(padded[:, :, :, eligible_z], (3, 0, 1, 2)).copy()
    tensor = torch.from_numpy(selected)
    resized = F.interpolate(
        tensor,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).cpu().numpy().astype(np.float32, copy=False)
    if resized.shape != (len(eligible_z), len(CHANNEL_ORDER), image_size, image_size):
        raise RuntimeError(f"Unexpected resized shape for {record.identifier}: {resized.shape}")
    if not np.all(np.isfinite(resized)):
        raise ValueError(f"Resized slices contain NaN/Inf: {record.identifier}")
    if float(resized.min()) < -1.0e-6 or float(resized.max()) > 1.0 + 1.0e-6:
        raise ValueError(f"Resized slices escaped [0,1]: {record.identifier}")

    pad_index_transform = np.eye(4, dtype=np.float64)
    pad_index_transform[0, 3] = -float(padding["x_before"])
    pad_index_transform[1, 3] = -float(padding["y_before"])
    padded_affine = target_affine @ pad_index_transform
    scale = float(pad_size) / float(image_size)
    resize_index_transform = np.eye(4, dtype=np.float64)
    resize_index_transform[0, 0] = scale
    resize_index_transform[1, 1] = scale
    resize_index_transform[0, 3] = scale / 2.0 - 0.5
    resize_index_transform[1, 3] = scale / 2.0 - 0.5
    model_affine = padded_affine @ resize_index_transform

    metadata = {
        "csv_index": record.csv_index,
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "session_identifier": record.identifier,
        "split": record.split,
        "channel_order": list(CHANNEL_ORDER),
        "source_paths": {key: str(value) for key, value in record.paths.items()},
        "hf_metadata_paths": {
            key: str(value) for key, value in record.metadata_paths.items()
        },
        "hf_blob_sha256": record.blob_digests,
        "source_shape": list(source_shape),
        "reoriented_shape": list(reoriented["T1"].shape),
        "source_axcodes": source_axcodes,
        "target_axcodes": "".join(str(value) for value in target_axcodes),
        "source_affine": source_affine.astype(float).tolist(),
        "reoriented_affine": target_affine.astype(float).tolist(),
        "orientation_transform": transform.astype(float).tolist(),
        "orientation_operation": "axis permutation/flip only; no interpolation",
        "intensity_multiset_signatures": intensity_signatures,
        "normalization": normalization,
        "padding": {"target_size": int(pad_size), **padding},
        "padded_affine": padded_affine.astype(float).tolist(),
        "model_image_size": int(image_size),
        "model_affine_half_pixel_convention": model_affine.astype(float).tolist(),
        "slice_count": int(len(eligible_z)),
        "first_z": int(eligible_z[0]),
        "last_z": int(eligible_z[-1]),
        "mask_created": False,
        "slice_eligibility": "any(T1 > 0) over LPS x/y axes",
    }
    return eligible_z, resized, metadata


def _map_size(entry_count: int, channels: int, image_size: int) -> int:
    sample = np.zeros((channels, image_size, image_size), dtype=np.float32)
    bytes_per_entry = len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
    return max(int(entry_count * bytes_per_entry * 1.35) + 512 * 1024**2, 1024**3)


def _processing_fingerprint(
    validation_report: Mapping[str, Any],
    target_axcodes: Sequence[str],
    pad_size: int,
    image_size: int,
) -> tuple[str, dict[str, Any]]:
    specification = {
        "schema_version": 1,
        "dataset": EXPECTED_DATASET,
        "hf_revision": validation_report.get("hf_revision"),
        "metadata_tsv_sha256": validation_report["metadata_tsv_sha256"],
        "channel_order": list(CHANNEL_ORDER),
        "target_axcodes": "".join(str(value) for value in target_axcodes),
        "orientation": "nibabel apply_orientation; discrete permutation/flip; no interpolation",
        "normalization": "per-modality nonzero p99; clip [0,1]",
        "mask_policy": "none; T1 > 0 only selects axial slices",
        "pad_size": int(pad_size),
        "padding": "symmetric zero padding",
        "image_size": int(image_size),
        "resize": "torch bilinear align_corners=False antialias=True",
        "validation_subjects": validation_report["validation_subjects"],
    }
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), specification


def _open_lmdb(path: Path, map_size: int):
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FOMO LMDB preparation requires the optional lmdb package.") from exc
    path.mkdir(parents=True, exist_ok=False)
    return lmdb.open(str(path), map_size=int(map_size), subdir=True)


def build_output(
    source_root: str | Path,
    metadata_tsv: str | Path,
    output_root: str | Path,
    *,
    target_axcodes: Sequence[str] = ("L", "P", "S"),
    validation_subjects: Sequence[str] = DEFAULT_VALIDATION_SUBJECTS,
    expectations: DatasetExpectations = DEFAULT_EXPECTATIONS,
    pad_size: int = 256,
    image_size: int = 128,
    map_size_override: int | None = None,
) -> dict[str, Any]:
    """Validate, build, audit, and atomically publish a fresh FOMO LMDB root."""

    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing FOMO output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        records, validation_report = validate_dataset(
            source_root,
            metadata_tsv,
            validation_subjects=validation_subjects,
            expectations=expectations,
            pad_size=pad_size,
        )
        _json_dump(stage / "validation_report.json", validation_report)
        fingerprint, specification = _processing_fingerprint(
            validation_report, target_axcodes, pad_size, image_size
        )
        manifests = stage / "manifests"
        manifests.mkdir(parents=True, exist_ok=False)
        expected_entries = validation_report["split_slice_counts"]
        map_sizes = {
            split: int(map_size_override)
            if map_size_override is not None
            else _map_size(int(expected_entries[split]), len(CHANNEL_ORDER), image_size)
            for split in ("train", "val")
        }
        if any(value <= 0 for value in map_sizes.values()):
            raise ValueError("LMDB map size must be positive.")
        environments = {
            split: _open_lmdb(stage / split, map_sizes[split]) for split in ("train", "val")
        }
        entry_handles: dict[str, Any] = {}
        entry_writers: dict[str, csv.DictWriter] = {}
        fieldnames = ["key", "participant_id", "session_id", "session_identifier", "z", "split"]
        session_handle = None
        counts = {"train": 0, "val": 0}
        try:
            for split in ("train", "val"):
                handle = (manifests / f"{split}_entries.csv").open(
                    "w", newline="", encoding="utf-8"
                )
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                entry_handles[split] = handle
                entry_writers[split] = writer
            session_handle = (manifests / "sessions.jsonl").open("w", encoding="utf-8")
            for index, record in enumerate(records, start=1):
                z_indices, slices, metadata = process_session(
                    record,
                    target_axcodes=target_axcodes,
                    pad_size=pad_size,
                    image_size=image_size,
                )
                split = record.split
                start_index = counts[split]
                with environments[split].begin(write=True) as transaction:
                    for offset, value in enumerate(slices):
                        key = f"{start_index + offset:08d}"
                        encoded = pickle.dumps(
                            np.asarray(value, dtype=np.float32),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        if not transaction.put(key.encode("ascii"), encoded, overwrite=False):
                            raise RuntimeError(f"Duplicate LMDB key in {split}: {key}")
                for offset, z_index in enumerate(z_indices):
                    key = f"{start_index + offset:08d}"
                    entry_writers[split].writerow(
                        {
                            "key": key,
                            "participant_id": record.participant_id,
                            "session_id": record.session_id,
                            "session_identifier": record.identifier,
                            "z": int(z_index),
                            "split": split,
                        }
                    )
                counts[split] += int(len(z_indices))
                metadata["processing_fingerprint"] = fingerprint
                session_handle.write(json.dumps(metadata, allow_nan=False) + "\n")
                if index % 5 == 0 or index == len(records):
                    print(
                        f"Built {index}/{len(records)} sessions "
                        f"(train={counts['train']}, val={counts['val']})",
                        flush=True,
                    )
        finally:
            if session_handle is not None:
                session_handle.close()
            for handle in entry_handles.values():
                handle.close()
            for environment in environments.values():
                environment.sync()
                environment.close()

        if counts != {key: int(value) for key, value in expected_entries.items()}:
            raise RuntimeError(f"Built entry counts do not match validation: {counts} != {expected_entries}")
        build_manifest = {
            "status": "BUILT_STAGING",
            "built_at": _now(),
            "published_root": str(output_root),
            "staging_root": str(stage),
            "source_root": str(Path(source_root).resolve()),
            "metadata_tsv": str(Path(metadata_tsv).resolve()),
            "hf_revision": validation_report["hf_revision"],
            "processing_fingerprint": fingerprint,
            "processing_specification": specification,
            "channel_order": list(CHANNEL_ORDER),
            "mask_channel": False,
            "split_entry_counts": counts,
            "split_map_sizes": map_sizes,
            "session_count": len(records),
            "manifest_files": {
                "sessions": "manifests/sessions.jsonl",
                "train_entries": "manifests/train_entries.csv",
                "val_entries": "manifests/val_entries.csv",
            },
        }
        _json_dump(stage / "build_manifest.json", build_manifest)
        audit = audit_output(stage, write_report=True, declared_publish_root=output_root)
        if audit.get("status") != "PASS":
            raise RuntimeError(f"FOMO staging audit did not pass: {audit}")
        stage.rename(output_root)
        publication = {
            "status": "PUBLISHED",
            "published_at": _now(),
            "output_root": str(output_root),
            "processing_fingerprint": fingerprint,
            "split_entry_counts": counts,
            "audit_status": audit["status"],
        }
        _json_dump(output_root / "publication.json", publication)
        return publication
    except Exception:
        print(f"Build failed; staging data retained for diagnosis: {stage}", flush=True)
        raise


def _read_entry_manifest(path: Path, split: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        expected_key = f"{index:08d}"
        if row.get("key") != expected_key or row.get("split") != split:
            raise ValueError(
                f"Invalid {split} entry manifest row {index}: "
                f"key={row.get('key')}, split={row.get('split')}."
            )
    return rows


def _audit_lmdb(
    path: Path,
    expected_rows: list[dict[str, str]],
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FOMO LMDB audit requires the optional lmdb package.") from exc
    if not path.is_dir():
        raise FileNotFoundError(path)
    environment = lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=1
    )
    digest = hashlib.sha256()
    minimum = float("inf")
    maximum = float("-inf")
    entries = 0
    try:
        with environment.begin(write=False) as transaction:
            with transaction.cursor() as cursor:
                for index, (key_bytes, value_bytes) in enumerate(cursor):
                    expected_key = f"{index:08d}".encode("ascii")
                    if key_bytes != expected_key:
                        raise ValueError(
                            f"Non-contiguous LMDB key in {path}: {key_bytes!r} != {expected_key!r}"
                        )
                    value = np.asarray(pickle.loads(value_bytes))
                    if value.shape != expected_shape or value.dtype != np.float32:
                        raise ValueError(
                            f"Invalid LMDB value at {path}/{expected_key!r}: "
                            f"shape={value.shape}, dtype={value.dtype}."
                        )
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"NaN/Inf in LMDB value {path}/{expected_key!r}.")
                    minimum = min(minimum, float(value.min()))
                    maximum = max(maximum, float(value.max()))
                    digest.update(key_bytes)
                    digest.update(value_bytes)
                    entries += 1
                    if entries % 5000 == 0:
                        print(f"Audited {entries} entries in {path.name}", flush=True)
    finally:
        environment.close()
    if entries != len(expected_rows):
        raise ValueError(f"{path.name} entry count mismatch: {entries} != {len(expected_rows)}")
    if entries <= 0:
        raise ValueError(f"LMDB contains no entries: {path}")
    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:
        raise ValueError(f"LMDB value range is outside [0,1]: min={minimum}, max={maximum}")
    return {
        "entries": entries,
        "shape": list(expected_shape),
        "dtype": "float32",
        "minimum": minimum,
        "maximum": maximum,
        "content_sha256": digest.hexdigest(),
    }


def audit_output(
    output_root: str | Path,
    *,
    write_report: bool = True,
    declared_publish_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read every LMDB record and validate manifests/orientation/split isolation."""

    root = Path(output_root).resolve()
    build_manifest_path = root / "build_manifest.json"
    if not build_manifest_path.is_file():
        raise FileNotFoundError(build_manifest_path)
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    if build_manifest.get("channel_order") != list(CHANNEL_ORDER):
        raise ValueError("Build manifest channel order is not [FLAIR,T1,T2].")
    if build_manifest.get("mask_channel") is not False:
        raise ValueError("Build manifest does not explicitly exclude a mask channel.")

    session_path = root / "manifests" / "sessions.jsonl"
    session_rows = [
        json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(session_rows) != int(build_manifest["session_count"]):
        raise ValueError("Session manifest count does not match build manifest.")
    identifiers: set[str] = set()
    split_subjects = {"train": set(), "val": set()}
    for row in session_rows:
        identifier = str(row["session_identifier"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate session manifest entry: {identifier}")
        identifiers.add(identifier)
        if row.get("target_axcodes") != "LPS":
            raise ValueError(f"Non-LPS session in manifest: {identifier}")
        if row.get("source_axcodes") != "RAS":
            raise ValueError(f"Unexpected non-RAS source in manifest: {identifier}")
        if row.get("mask_created") is not False or row.get("channel_order") != list(CHANNEL_ORDER):
            raise ValueError(f"Mask/channel contract violation: {identifier}")
        orientation = np.asarray(row["orientation_transform"], dtype=np.float64)
        expected = np.asarray([[0, -1], [1, -1], [2, 1]], dtype=np.float64)
        if not np.array_equal(orientation, expected):
            raise ValueError(f"Unexpected RAS-to-LPS orientation transform: {identifier}/{orientation}")
        split_subjects[str(row["split"])].add(str(row["participant_id"]))
    if split_subjects["train"].intersection(split_subjects["val"]):
        raise ValueError("Subject leakage found in session manifest.")

    split_reports: dict[str, Any] = {}
    image_size = int(build_manifest["processing_specification"]["image_size"])
    expected_shape = (len(CHANNEL_ORDER), image_size, image_size)
    for split in ("train", "val"):
        entry_rows = _read_entry_manifest(root / "manifests" / f"{split}_entries.csv", split)
        split_reports[split] = _audit_lmdb(root / split, entry_rows, expected_shape)
        expected_count = int(build_manifest["split_entry_counts"][split])
        if split_reports[split]["entries"] != expected_count:
            raise ValueError(f"{split} entries do not match build manifest.")

    report = {
        "status": "PASS",
        "audited_at": _now(),
        "audited_root": str(root),
        "declared_publish_root": str(Path(declared_publish_root).resolve())
        if declared_publish_root is not None
        else str(root),
        "processing_fingerprint": build_manifest["processing_fingerprint"],
        "channel_order": list(CHANNEL_ORDER),
        "mask_channel": False,
        "source_axcodes": "RAS",
        "target_axcodes": "LPS",
        "orientation_operation": "axis permutation/flip only; no interpolation",
        "session_count": len(session_rows),
        "split_subject_counts": {key: len(value) for key, value in split_subjects.items()},
        "split_lmdb": split_reports,
        "total_entries": sum(value["entries"] for value in split_reports.values()),
    }
    if write_report:
        _json_dump(root / "audit_report.json", report)
    return report


def dataset_status(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    parent = root.parent
    staging = sorted(str(path) for path in parent.glob(f".{root.name}.staging-*"))
    payload: dict[str, Any] = {
        "output_root": str(root),
        "exists": root.exists(),
        "staging_roots": staging,
    }
    for name in ("publication.json", "build_manifest.json", "audit_report.json"):
        path = root / name
        payload[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return payload


__all__ = [
    "CHANNEL_ORDER",
    "DEFAULT_EXPECTATIONS",
    "DEFAULT_VALIDATION_SUBJECTS",
    "DatasetExpectations",
    "FOMOSessionRecord",
    "audit_output",
    "build_output",
    "dataset_status",
    "normalize_nonzero_p99",
    "process_session",
    "read_session_records",
    "reorient_volume",
    "validate_dataset",
]
