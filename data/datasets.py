"""新版 ANDi framework 的 dataset builder。

LMDB 與 volume loader 保留成清楚的小型介面，讓未來新增真實資料集時
不需要修改 training loop。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .registry import DATASET_REGISTRY, register_dataset


class LMDBSliceDataset(Dataset):
    """讀取原版 ANDi healthy-slice LMDB 格式。"""

    def __init__(self, directory: str | Path, image_size: int | None = None):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("LMDBSliceDataset requires the optional 'lmdb' package.") from exc

        self._lmdb = lmdb
        self.directory = str(directory)
        self.image_size = image_size
        env = self._lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()

    def _open_lmdb(self) -> None:
        self.env = self._lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if not hasattr(self, "txn"):
            self._open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        if byteflow is None:
            raise IndexError(index)

        tensor = torch.from_numpy(pickle.loads(byteflow)).float()
        if self.image_size is not None and tensor.shape[-1] != self.image_size:
            tensor = transforms.Resize(self.image_size, antialias=True)(tensor)
        return tensor


class BraTSHealthySliceDataset(Dataset):
    """Read healthy 2D slices directly from a raw BraTS volume directory."""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        slice_column: str = "Slice",
    ):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = Path(dataset_path)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.slice_column = slice_column
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
            self._load_nifti(subject_dir / f"{subject_id}_{modality}.nii.gz", dtype=float)
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


def normalize_volume(images: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """用 foreground intensity 的第 99 百分位數正規化每個 modality。"""

    images = images.float()
    for modality in range(images.shape[0]):
        values = images[modality].reshape(-1)
        foreground = values[values > 0]
        if foreground.numel() == 0:
            continue
        images[modality] = images[modality] / torch.quantile(foreground, 0.99).clamp_min(eps)
    return images


def histogram_normalize_volume(images: np.ndarray) -> torch.Tensor:
    try:
        import skimage.exposure as exposure
    except ImportError as exc:
        raise ImportError("Histogram normalization requires the optional 'scikit-image' package.") from exc

    normalized = np.zeros_like(images, dtype=np.float32)
    for modality in range(images.shape[0]):
        image = images[modality]
        mask = image > 0
        if image.max() > 0:
            image = image / image.max()
        normalized[modality] = exposure.equalize_hist(image.astype(np.float32), mask=mask, nbins=256) * mask
    return torch.from_numpy(normalized)


class MRIDataVolume(Dataset):
    """載入完整 multi-modal MRI volume，供 ANDi evaluation 使用。"""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        segmentation_suffix: str = "seg",
        histogram_normalization: bool = False,
        shift_naming: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = Path(dataset_path)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.segmentation_suffix = segmentation_suffix
        self.histogram_normalization = bool(histogram_normalization)
        self.shift_naming = bool(shift_naming)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_id = str(self.df.loc[idx, self.df.columns[0]])
        file_stem = self._subject_file_stem(subject_id)
        subject_dir = self.dataset_path / subject_id

        images = []
        for modality in self.modalities:
            images.append(self._load_nifti(subject_dir / f"{file_stem}_{modality}.nii.gz", dtype=float))

        mask = self._load_nifti(subject_dir / f"{file_stem}_{self.segmentation_suffix}.nii.gz", dtype=float)
        # 有些資料集的 mask 經 registration/interpolation 後會變成 float；
        # 因此先 threshold，再做 nearest-neighbor resize。
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

        # 將每個 axial slice resize 成 DDPM 預期的 2D 解析度。
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

        return resized, mask


@register_dataset("lmdb")
def _build_lmdb_dataset(config: dict[str, Any]) -> Dataset:
    if "path" not in config:
        raise ValueError("data.path is required when data.type is 'lmdb'.")
    return LMDBSliceDataset(config["path"], image_size=int(config.get("image_size", 128)))


@register_dataset("brats_healthy_slices", "healthy_slices")
def _build_brats_healthy_slices(config: dict[str, Any]) -> Dataset:
    return BraTSHealthySliceDataset(
        csv_path=config["path_to_csv"],
        dataset_path=config["dataset_path"],
        image_size=int(config.get("image_size", 128)),
        modalities=config.get("modalities"),
        slice_column=str(config.get("slice_column", "Slice")),
    )


@register_dataset("volume", "mri_volume", "brats_volume")
def _build_mri_volume(config: dict[str, Any]) -> Dataset:
    return MRIDataVolume(
        csv_path=config["path_to_csv"],
        dataset_path=config["dataset_path"],
        image_size=int(config.get("image_size", 128)),
        modalities=config.get("modalities"),
        segmentation_suffix=str(config.get("segmentation_suffix", "seg")),
        histogram_normalization=bool(config.get("histogram_normalization", False)),
        shift_naming=bool(config.get("shift_naming", "shifts" in str(config.get("dataset_path", "")).lower())),
    )


def build_dataset(config: dict[str, Any]) -> Dataset:
    """從 config 建立 dataset，避免上層 trainer/evaluator 寫死 dataset type 判斷。"""

    data_type = str(config.get("type", "lmdb")).lower()
    builder = DATASET_REGISTRY.get(data_type)
    if builder is None:
        raise ValueError(f"Unknown dataset type: {data_type}")
    return builder(config)


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=bool(config.get("shuffle", False)),
        num_workers=int(config.get("workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )
