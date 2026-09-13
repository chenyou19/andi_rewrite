"""Dataset adapters for the registered FOMO45K BraTS21/SRI24 publication."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..fomo45k.brats21 import BRATS_SHAPE, BRATS_SPACING_MM, MODEL_CHANNEL_ORDER


def zscore_nonzero(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Per-modality z-score inside the shared brain mask, preserving zero background."""

    value = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if value.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {value.shape} != {mask.shape}")
    foreground = value[mask]
    if foreground.size == 0:
        raise ValueError("Cannot normalize with an empty brain mask.")
    if not np.all(np.isfinite(foreground)):
        raise ValueError("NaN/Inf found inside the brain mask.")
    mean = float(foreground.mean(dtype=np.float64))
    std = float(foreground.std(dtype=np.float64))
    if not np.isfinite(std) or std <= 1.0e-8:
        raise ValueError(f"Degenerate foreground standard deviation: {std}")
    output = np.zeros(value.shape, dtype=np.float32)
    output[mask] = ((foreground - mean) / std).astype(np.float32, copy=False)
    return output


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "case_id",
        "processing_status",
        "t1_output",
        "t2_output",
        "flair_output",
        "brain_mask_output",
    }
    missing = required.difference(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"BraTS-like manifest is missing columns: {sorted(missing)}")
    return rows


def _load_validated(path: Path, reference_affine: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    if tuple(int(value) for value in image.shape) != BRATS_SHAPE:
        raise ValueError(f"Unexpected BraTS grid shape for {path}: {image.shape}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not np.allclose(spacing, BRATS_SPACING_MM, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"Unexpected BraTS grid spacing for {path}: {spacing}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if reference_affine is not None and not np.allclose(
        affine, reference_affine, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(f"BraTS-like modality affine mismatch: {path}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"NaN/Inf found in {path}")
    return data, affine


class FOMO45KBraTS21VolumeDataset(Dataset):
    """Return named registered volumes and a FLAIR/T1/T2 stack."""

    requires_batch_size_one = True

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path | None = None,
        *,
        return_dict: bool = True,
        return_metadata: bool = False,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.dataset_root / "dataset_manifest.csv"
        )
        self.rows = [
            row for row in _read_manifest(self.manifest_path) if row["processing_status"] == "PASS"
        ]
        if not self.rows:
            raise ValueError(f"No complete PASS cases in {self.manifest_path}")
        self.return_dict = bool(return_dict)
        self.return_metadata = bool(return_metadata)
        self.modalities = list(MODEL_CHANNEL_ORDER)

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: Mapping[str, str], key: str) -> Path:
        path = Path(row[key])
        return path if path.is_absolute() else self.dataset_root / path

    def load_case(self, row: Mapping[str, str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        mask_data, affine = _load_validated(self._path(row, "brain_mask_output"))
        mask = mask_data > 0.5
        if not mask.any():
            raise ValueError(f"Empty brain mask for {row['case_id']}")
        tensors: dict[str, torch.Tensor] = {}
        paths: dict[str, str] = {}
        for modality in MODEL_CHANNEL_ORDER:
            key = f"{modality}_output"
            path = self._path(row, key)
            volume, _ = _load_validated(path, affine)
            normalized = zscore_nonzero(volume, mask)
            tensors[modality] = torch.from_numpy(normalized)
            paths[modality] = str(path)
        tensors["brain_mask"] = torch.from_numpy(mask.astype(np.bool_))
        tensors["image"] = torch.stack([tensors[name] for name in MODEL_CHANNEL_ORDER], dim=0)
        metadata = {
            "case_id": row["case_id"],
            "channel_order": list(MODEL_CHANNEL_ORDER),
            "affine": affine.astype(float).tolist(),
            "shape": list(BRATS_SHAPE),
            "input_paths": paths,
            "brain_mask_path": str(self._path(row, "brain_mask_output")),
        }
        return tensors, metadata

    def __getitem__(self, index: int):
        tensors, metadata = self.load_case(self.rows[index])
        value: Any = tensors if self.return_dict else tensors["image"]
        if self.return_metadata:
            return value, metadata
        return value


class FOMO45KBraTS21SliceDataset(Dataset):
    """Model-facing axial slice adapter for the current 2-D ANDi network."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path | None = None,
        slice_manifest_path: str | Path | None = None,
        *,
        image_size: int = 128,
        return_metadata: bool = False,
    ):
        self.volume_dataset = FOMO45KBraTS21VolumeDataset(
            dataset_root,
            manifest_path,
            return_dict=True,
            return_metadata=False,
        )
        self.dataset_root = self.volume_dataset.dataset_root
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.return_metadata = bool(return_metadata)
        self.modalities = list(MODEL_CHANNEL_ORDER)
        self.rows_by_case = {row["case_id"]: row for row in self.volume_dataset.rows}
        path = (
            Path(slice_manifest_path).resolve()
            if slice_manifest_path is not None
            else self.dataset_root / "slice_manifest.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.slices = list(csv.DictReader(handle))
        if not self.slices or not {"case_id", "slice"}.issubset(self.slices[0]):
            raise ValueError(f"Invalid slice manifest: {path}")
        unknown = sorted({row["case_id"] for row in self.slices}.difference(self.rows_by_case))
        if unknown:
            raise ValueError(f"Slice manifest references non-PASS cases: {unknown[:8]}")
        self._cached_case_id: str | None = None
        self._cached_tensors: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.slices)

    def _case(self, case_id: str) -> dict[str, torch.Tensor]:
        if case_id == self._cached_case_id and self._cached_tensors is not None:
            return self._cached_tensors
        tensors, _metadata = self.volume_dataset.load_case(self.rows_by_case[case_id])
        self._cached_case_id = case_id
        self._cached_tensors = tensors
        return tensors

    def __getitem__(self, index: int):
        row = self.slices[index]
        case_id = row["case_id"]
        slice_index = int(row["slice"])
        if not 0 <= slice_index < BRATS_SHAPE[2]:
            raise IndexError(f"Invalid axial slice {slice_index} for {case_id}")
        tensors = self._case(case_id)
        image = tensors["image"][:, :, :, slice_index]
        mask = tensors["brain_mask"][:, :, slice_index]
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            mask = F.interpolate(
                mask[None, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest-exact",
            )[0, 0].bool()
            image = torch.where(mask[None], image, torch.zeros_like(image))
        if not self.return_metadata:
            return image
        return image, {
            "case_id": case_id,
            "slice": slice_index,
            "channel_order": list(MODEL_CHANNEL_ORDER),
            "native_shape": list(BRATS_SHAPE),
            "model_shape": list(image.shape),
        }


def build_fomo45k_brats21_volume_dataset(config: dict[str, Any]) -> FOMO45KBraTS21VolumeDataset:
    return FOMO45KBraTS21VolumeDataset(
        dataset_root=config["dataset_path"],
        manifest_path=config.get("manifest_path"),
        return_dict=bool(config.get("return_dict", True)),
        return_metadata=bool(config.get("return_metadata", False)),
    )


def build_fomo45k_brats21_slice_dataset(config: dict[str, Any]) -> FOMO45KBraTS21SliceDataset:
    return FOMO45KBraTS21SliceDataset(
        dataset_root=config["dataset_path"],
        manifest_path=config.get("manifest_path"),
        slice_manifest_path=config.get("slice_manifest_path"),
        image_size=int(config.get("image_size", 128)),
        return_metadata=bool(config.get("return_metadata", False)),
    )


__all__ = [
    "FOMO45KBraTS21SliceDataset",
    "FOMO45KBraTS21VolumeDataset",
    "build_fomo45k_brats21_slice_dataset",
    "build_fomo45k_brats21_volume_dataset",
    "zscore_nonzero",
]
