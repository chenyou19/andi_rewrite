"""BraTS subject discovery and fail-closed input geometry validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.common import _subject_file_path


CHANNEL_ORDER = ("FLAIR", "T1", "T2")
LOGICAL_MODALITIES = ("flair", "t1", "t2")


@dataclass(frozen=True)
class BraTSMPIRecord:
    csv_index: int
    subject_id: str
    flair_path: Path
    t1_path: Path
    t2_path: Path
    segmentation_path: Path

    @property
    def modality_paths(self) -> tuple[Path, Path, Path]:
        return self.flair_path, self.t1_path, self.t2_path

    @property
    def input_paths(self) -> dict[str, str]:
        return {
            "FLAIR": str(self.flair_path),
            "T1": str(self.t1_path),
            "T2": str(self.t2_path),
            "segmentation": str(self.segmentation_path),
        }


def _read_subject_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"BraTS subject CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0]:
        raise ValueError(f"BraTS subject CSV has no header: {path}")
    subject_ids = [row[0].strip() for row in rows[1:] if row]
    if not subject_ids or any(not item for item in subject_ids):
        raise ValueError(f"BraTS subject CSV contains no subjects or a blank id: {path}")
    duplicates = sorted({item for item in subject_ids if subject_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"BraTS subject CSV contains duplicate ids: {duplicates}")
    return subject_ids


def discover_records(config: dict[str, Any]) -> list[BraTSMPIRecord]:
    root = Path(config["dataset_path"]).resolve()
    csv_path = Path(config["path_to_csv"]).resolve()
    separator = str(config.get("filename_separator", "_"))
    suffixes = dict(config.get("modality_suffixes", {}))
    modality_suffixes = {
        "flair": str(suffixes.get("flair", "flair")),
        "t1": str(suffixes.get("t1", "t1")),
        "t2": str(suffixes.get("t2", "t2")),
    }
    segmentation_suffix = str(config.get("segmentation_suffix", "seg"))
    records: list[BraTSMPIRecord] = []
    for index, subject_id in enumerate(_read_subject_ids(csv_path)):
        subject_dir = root / subject_id
        paths = {
            name: _subject_file_path(subject_dir, subject_id, suffix, separator)
            for name, suffix in modality_suffixes.items()
        }
        segmentation = _subject_file_path(
            subject_dir,
            subject_id,
            segmentation_suffix,
            separator,
        )
        records.append(
            BraTSMPIRecord(
                csv_index=index,
                subject_id=subject_id,
                flair_path=paths["flair"],
                t1_path=paths["t1"],
                t2_path=paths["t2"],
                segmentation_path=segmentation,
            )
        )
    expected = config.get("expected_subjects")
    if expected is not None and len(records) != int(expected):
        raise ValueError(f"Expected {int(expected)} BraTS subjects, found {len(records)}.")
    return records


def validate_record_geometry(
    record: BraTSMPIRecord,
    *,
    check_voxels: bool,
) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BraTS-MPI validation requires nibabel.") from exc

    images = []
    for name, path in record.input_paths.items():
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Missing {name} for {record.subject_id}: {target}")
        image = nib.load(str(target))
        if len(image.shape) != 3:
            raise ValueError(
                f"Expected 3D {name} for {record.subject_id}, got {image.shape}: {target}"
            )
        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError(f"Invalid affine for {record.subject_id}/{name}: {target}")
        if abs(float(np.linalg.det(affine[:3, :3]))) <= np.finfo(float).eps:
            raise ValueError(f"Singular affine for {record.subject_id}/{name}: {target}")
        if check_voxels:
            values = image.get_fdata(dtype=np.float32, caching="unchanged")
            if not np.all(np.isfinite(values)):
                count = int(values.size - np.count_nonzero(np.isfinite(values)))
                raise ValueError(
                    f"Nonfinite voxels for {record.subject_id}/{name}: count={count}, path={target}"
                )
        images.append((name, image))

    reference_name, reference = images[0]
    for name, image in images[1:]:
        if image.shape != reference.shape or not np.allclose(
            image.affine,
            reference.affine,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                f"Geometry mismatch for {record.subject_id}: "
                f"{reference_name}={reference.shape} versus {name}={image.shape}."
            )
    canonical = nib.as_closest_canonical(reference, enforce_diag=False)
    if tuple(nib.aff2axcodes(canonical.affine)) != ("R", "A", "S"):
        raise ValueError(f"Could not canonicalize {record.subject_id} to RAS.")
    return {
        "subject_id": record.subject_id,
        "shape": [int(item) for item in reference.shape],
        "voxel_spacing_mm": [float(item) for item in reference.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(reference.affine)),
        "canonical_shape": [int(item) for item in canonical.shape],
        "canonical_orientation": list(nib.aff2axcodes(canonical.affine)),
    }


__all__ = [
    "BraTSMPIRecord",
    "CHANNEL_ORDER",
    "LOGICAL_MODALITIES",
    "discover_records",
    "validate_record_geometry",
]
