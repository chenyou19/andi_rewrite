"""Dataset adapter for the validated offline BraTS-to-MPI cache."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..brats_mpi.processing import CACHE_SCHEMA_VERSION, MODES, VALID_BRATS_LABELS


def _subject_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    values = [row[0].strip() for row in rows if row and row[0].strip()]
    if values and values[0].lower() in {"subject_id", "subject", "id", "scan"}:
        values = values[1:]
    if not values:
        raise ValueError(f"subject manifest is empty: {path}")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate subject IDs in {path}: {duplicates}")
    return values


class BraTSMPICacheDataset(Dataset):
    """Read only PASS subjects produced by ``prepare_brats_mpi.py``.

    The stored label is categorical (0/1/2/4). ``__getitem__`` derives the
    whole-tumour target as ``segmentation > 0`` so no label information is lost
    in the cache itself.
    """

    requires_batch_size_one = True

    def __init__(
        self,
        cache_root: str | Path,
        mode: str = "mni_affine",
        path_to_csv: str | Path | None = None,
        csv_path: str | Path | None = None,
        image_size: int = 128,
        expected_channels: int = 3,
        channels: int | None = None,
        return_metadata: bool = False,
        **_: Any,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unsupported BraTS MPI cache mode: {mode}")
        manifest = path_to_csv or csv_path
        if manifest is None:
            raise ValueError("path_to_csv (a generated PASS manifest) is required")
        self.cache_root = Path(cache_root)
        self.mode = mode
        self.manifest_path = Path(manifest)
        self.subjects = _subject_ids(self.manifest_path)
        self.image_size = int(image_size)
        self.expected_channels = int(channels if channels is not None else expected_channels)
        self.return_metadata = bool(return_metadata)

        if not self.cache_root.is_dir():
            raise FileNotFoundError(f"cache root does not exist: {self.cache_root}")
        for subject_id in self.subjects:
            self._validate_subject_files(subject_id)

    def __len__(self) -> int:
        return len(self.subjects)

    def _paths(self, subject_id: str) -> tuple[Path, Path]:
        root = self.cache_root / self.mode / "subjects" / subject_id
        return root / "volume.npz", root / "spatial_metadata.json"

    def _validate_subject_files(self, subject_id: str) -> None:
        volume_path, metadata_path = self._paths(subject_id)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing cache metadata for {subject_id}: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError(f"cache schema mismatch for {subject_id}")
        if metadata.get("subject_id") != subject_id or metadata.get("mode") != self.mode:
            raise ValueError(f"cache identity mismatch for {subject_id}")
        if metadata.get("qc", {}).get("status") != "PASS":
            raise ValueError(f"manifest includes excluded cache subject {subject_id}")
        if not metadata.get("fingerprint"):
            raise ValueError(f"cache fingerprint is absent for {subject_id}")
        if not volume_path.is_file():
            raise FileNotFoundError(f"missing PASS cache volume for {subject_id}: {volume_path}")

    def __getitem__(self, index: int):
        subject_id = self.subjects[index]
        volume_path, metadata_path = self._paths(subject_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(volume_path, allow_pickle=False) as cached:
            image = np.asarray(cached["image"], dtype=np.float32)
            segmentation = np.asarray(cached["segmentation"])
            brain_mask = np.asarray(cached["brain_mask"])

        expected_shape = (self.expected_channels, self.image_size, self.image_size)
        if image.ndim != 4 or image.shape[:3] != expected_shape:
            raise ValueError(
                f"invalid cached MRI shape for {subject_id}: {image.shape}; "
                f"expected ({self.expected_channels}, {self.image_size}, {self.image_size}, Z)"
            )
        if segmentation.shape != image.shape[1:] or brain_mask.shape != image.shape[1:]:
            raise ValueError(f"cached arrays do not share a grid for {subject_id}")
        if not np.isfinite(image).all():
            raise ValueError(f"cached MRI contains non-finite values for {subject_id}")
        rounded = np.rint(segmentation)
        labels = {int(value) for value in np.unique(rounded)}
        if not np.allclose(segmentation, rounded) or not labels.issubset(VALID_BRATS_LABELS):
            raise ValueError(f"cached GT is not categorical 0/1/2/4 for {subject_id}: {sorted(labels)}")

        image_tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        whole_tumour = torch.from_numpy(np.ascontiguousarray(rounded > 0))
        if not self.return_metadata:
            return image_tensor, whole_tumour

        # Keep collated metadata scalar/string-only. The exporter reads the JSON
        # for matrices and shapes, avoiding PyTorch's nested-list collation rules.
        item_metadata = {
            "subject_id": subject_id,
            "spatial_metadata_path": str(metadata_path.resolve()),
            "native_reference_path": str(metadata["native_reference_path"]),
            "reference_path": str(metadata["native_reference_path"]),
            "segmentation_path": str(metadata["segmentation_path"]),
            "preprocess_mode": self.mode,
            "cache_fingerprint": str(metadata["fingerprint"]),
        }
        return image_tensor, whole_tumour, item_metadata


# Friendly aliases for configuration files and imports.
BraTSMPI = BraTSMPICacheDataset
BraTSMPIDataset = BraTSMPICacheDataset


def build_brats_mpi_cache_dataset(config: dict[str, Any]) -> BraTSMPICacheDataset:
    """Factory adapter matching the other dataset builders."""

    return BraTSMPICacheDataset(**config)
