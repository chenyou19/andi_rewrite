"""BraTS healthy-slice and BraTS-style full-volume dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from .common import _shape_text, _subject_file_path
from .imaging import histogram_normalize_volume, normalize_volume


def _subject_frame_from_directory(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    subject_ids = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())
    if not subject_ids:
        raise FileNotFoundError(f"No subject directories found under {dataset_path}")
    return pd.DataFrame({"subject_id": subject_ids})


class BraTSHealthySliceDataset(Dataset):
    """Read healthy 2D slices directly from a raw BraTS volume directory."""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        slice_column: str = "Slice",
        filename_separator: str = "_",
    ):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = Path(dataset_path)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.slice_column = slice_column
        self.filename_separator = filename_separator
        if self.slice_column not in self.df.columns:
            raise ValueError(f"Slice CSV must contain a '{self.slice_column}' column.")
        self.subject_column = self.df.columns[0]
        self._cached_subject_id: str | None = None
        self._cached_volume: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.df.shape[0]

    def _load_nifti(self, path: Path, dtype: type = float) -> np.ndarray:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("BraTSHealthySliceDataset requires the optional 'nibabel' package.") from exc

        if not path.exists():
            raise FileNotFoundError(path)
        return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)

    def _load_subject_volume(self, subject_id: str) -> torch.Tensor:
        if self._cached_subject_id == subject_id and self._cached_volume is not None:
            return self._cached_volume

        subject_dir = self.dataset_path / subject_id
        images = [
            self._load_nifti(_subject_file_path(subject_dir, subject_id, modality, self.filename_separator), dtype=float)
            for modality in self.modalities
        ]
        volume = normalize_volume(torch.from_numpy(np.stack(images, axis=0)).float())
        self._cached_subject_id = subject_id
        self._cached_volume = volume
        return volume

    def __getitem__(self, idx: int) -> torch.Tensor:
        subject_id = str(self.df.loc[idx, self.subject_column])
        slice_index = int(self.df.loc[idx, self.slice_column])
        volume = self._load_subject_volume(subject_id)
        tensor = volume[:, :, :, slice_index]
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = transforms.Resize(self.image_size, antialias=True)(tensor)
        return tensor


class MRIDataVolume(Dataset):
    """Read BraTS-style multi-modal MRI volumes for ANDi evaluation."""

    def __init__(
        self,
        csv_path: str | Path | None,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        segmentation_suffix: str = "seg",
        histogram_normalization: bool = False,
        shift_naming: bool = False,
        filename_separator: str = "_",
        return_metadata: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.df = pd.read_csv(csv_path) if csv_path else _subject_frame_from_directory(self.dataset_path)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.segmentation_suffix = segmentation_suffix
        self.histogram_normalization = bool(histogram_normalization)
        self.shift_naming = bool(shift_naming)
        self.filename_separator = filename_separator
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return self.df.shape[0]

    def _subject_file_stem(self, subject_id: str) -> str:
        if self.shift_naming:
            return subject_id.split("_")[-1]
        return subject_id

    def _load_nifti(self, path: Path, dtype: type = float) -> np.ndarray:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("MRIDataVolume requires the optional 'nibabel' package.") from exc

        if not path.exists():
            raise FileNotFoundError(path)
        return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        subject_id = str(self.df.loc[idx, self.df.columns[0]])
        file_stem = self._subject_file_stem(subject_id)
        subject_dir = self.dataset_path / subject_id

        modality_paths = {
            modality: _subject_file_path(subject_dir, file_stem, modality, self.filename_separator)
            for modality in self.modalities
        }
        images = [self._load_nifti(path, dtype=float) for path in modality_paths.values()]

        segmentation_path = _subject_file_path(
            subject_dir,
            file_stem,
            self.segmentation_suffix,
            self.filename_separator,
        )
        mask = self._load_nifti(segmentation_path, dtype=float)
        native_shape = tuple(int(value) for value in mask.shape)
        mask = torch.from_numpy((mask > 0.5).astype(np.uint8)).bool()
        mask = F.interpolate(
            mask[None, None].float(),
            size=(self.image_size, self.image_size, mask.shape[2]),
            mode="nearest-exact",
        )[0, 0].bool()

        image_array = np.stack(images, axis=0)
        if self.histogram_normalization:
            volume = histogram_normalize_volume(image_array)
        else:
            volume = normalize_volume(torch.from_numpy(image_array).float())

        resized = torch.zeros(
            volume.shape[0],
            self.image_size,
            self.image_size,
            mask.shape[2],
            dtype=volume.dtype,
        )
        resize = transforms.Resize(self.image_size, antialias=True)
        for slice_index in range(mask.shape[2]):
            resized[:, :, :, slice_index] = resize(volume[None, :, :, :, slice_index])[0]

        if not self.return_metadata:
            return resized, mask
        reference_modality = self.modalities[0]
        metadata = {
            "subject_id": subject_id,
            "has_label": True,
            "reference_modality": reference_modality,
            "reference_path": str(modality_paths[reference_modality]),
            "native_shape": _shape_text(native_shape),
            "model_shape": _shape_text(tuple(mask.shape)),
            "input_paths": {key: str(value) for key, value in modality_paths.items()},
            "segmentation_path": str(segmentation_path),
        }
        return resized, mask, metadata


def build_brats_healthy_slices_dataset(config: dict[str, Any]) -> BraTSHealthySliceDataset:
    image_size = int(config.get("image_size", 128))
    return BraTSHealthySliceDataset(
        csv_path=config["path_to_csv"],
        dataset_path=config["dataset_path"],
        image_size=image_size,
        modalities=config.get("modalities"),
        slice_column=str(config.get("slice_column", "Slice")),
        filename_separator=str(config.get("filename_separator", "_")),
    )


def build_mri_volume_dataset(config: dict[str, Any]) -> MRIDataVolume:
    image_size = int(config.get("image_size", 128))
    return MRIDataVolume(
        csv_path=config.get("path_to_csv"),
        dataset_path=config["dataset_path"],
        image_size=image_size,
        modalities=config.get("modalities"),
        segmentation_suffix=str(config.get("segmentation_suffix", "seg")),
        histogram_normalization=bool(config.get("histogram_normalization", False)),
        shift_naming=bool(config.get("shift_naming", "shifts" in str(config.get("dataset_path", "")).lower())),
        filename_separator=str(config.get("filename_separator", "_")),
        return_metadata=bool(config.get("return_metadata", False)),
    )
