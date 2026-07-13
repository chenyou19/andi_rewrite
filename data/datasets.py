"""新版 ANDi framework 的 dataset builder。

LMDB 與 volume loader 保留成清楚的小型介面，讓未來新增真實資料集時
不需要修改 training loop。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_SHIFTS_SPLITS = ("train", "dev_in", "dev_out", "eval_in", "unsupervised")
_SHIFTS_LOCATIONS = ("ljubljana", "best", "msseg")


def _subject_frame_from_directory(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    subject_ids = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())
    if not subject_ids:
        raise FileNotFoundError(f"No subject directories found under {dataset_path}")
    return pd.DataFrame({"subject_id": subject_ids})


def _subject_file_path(subject_dir: Path, file_stem: str, suffix: str, separator: str = "_") -> Path:
    return subject_dir / f"{file_stem}{separator}{suffix}.nii.gz"


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(_NIFTI_SUFFIXES)


def _strip_nifti_suffix(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return name[:-7]
    if lowered.endswith(".nii"):
        return name[:-4]
    return name


def _subject_id_from_shifts_path(path: Path) -> str:
    return _strip_nifti_suffix(path.name).split("_", 1)[0]


def _as_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _shape_text(shape: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(int(item)) for item in shape)


@dataclass(frozen=True)
class ShiftsMSSubject:
    subject_id: str
    location: str
    split: str
    reference_path: Path
    modality_paths: dict[str, Path | None]
    modality_sources: dict[str, str]
    segmentation_path: Path | None
    brain_mask_path: Path | None


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
        csv_path: str | Path | None,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        segmentation_suffix: str = "seg",
        histogram_normalization: bool = False,
        shift_naming: bool = False,
        filename_separator: str = "_",
    ):
        self.dataset_path = Path(dataset_path)
        self.df = pd.read_csv(csv_path) if csv_path else _subject_frame_from_directory(self.dataset_path)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.segmentation_suffix = segmentation_suffix
        self.histogram_normalization = bool(histogram_normalization)
        self.shift_naming = bool(shift_naming)
        self.filename_separator = filename_separator

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
            images.append(self._load_nifti(_subject_file_path(subject_dir, file_stem, modality, self.filename_separator), dtype=float))

        mask = self._load_nifti(
            _subject_file_path(subject_dir, file_stem, self.segmentation_suffix, self.filename_separator),
            dtype=float,
        )
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


class ShiftsMSVolumeDataset(Dataset):
    """Full-volume Shifts MS / Ljubljana-style NIfTI dataset.

    The public Shifts MS archives are organized as
    ``<location>/<split>/<modality>/<subject>_<modality>_isovox.nii.gz``.
    This adapter discovers subjects directly from that layout, keeps the
    model-facing channel order stable, and returns metadata for native-space
    prediction export.
    """

    DEFAULT_MAPPING = {
        "flair": ["flair", "FLAIR"],
        "t1": ["t1", "T1"],
        "t1ce": ["t1ce", "T1Post", "t1post", "pd", "PD"],
        "t2": ["t2", "T2"],
        "segmentation": ["Gold_Standard", "gold_standard", "gt"],
        "brain_mask": ["fg_mask", "brain_mask", "mask"],
    }

    def __init__(
        self,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        modality_mapping: dict[str, Any] | None = None,
        dataset_subdir: str | None = None,
        locations: list[str] | str | None = None,
        location: str | None = None,
        preferred_locations: list[str] | None = None,
        splits: list[str] | str | None = None,
        reference_modality: str = "flair",
        require_segmentation: bool = True,
        require_modalities: bool = True,
        resample_to_reference: bool = True,
        histogram_normalization: bool = False,
        return_metadata: bool = True,
        subject_limit: int | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.root = self._resolve_shifts_root(self.dataset_path, dataset_subdir)
        self.image_size = int(image_size)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.modality_mapping = self._normalize_modality_mapping(modality_mapping or {})
        self.locations = _as_list(locations if locations is not None else location)
        self.preferred_locations = _as_list(preferred_locations, ["ljubljana", "best", "msseg"])
        self.splits = _as_list(splits)
        self.reference_modality = str(reference_modality)
        self.require_segmentation = bool(require_segmentation)
        self.require_modalities = bool(require_modalities)
        self.resample_to_reference = bool(resample_to_reference)
        self.histogram_normalization = bool(histogram_normalization)
        self.return_metadata = bool(return_metadata)
        self.subject_limit = int(subject_limit) if subject_limit not in (None, "", 0, False) else None
        self.subjects = self._discover_subjects()
        if not self.subjects:
            raise FileNotFoundError(
                f"No Shifts MS subjects found under {self.root}. "
                f"Requested locations={self.locations or 'auto'}, splits={self.splits or 'auto'}."
            )

    def __len__(self) -> int:
        return len(self.subjects)

    @staticmethod
    def _looks_like_shifts_root(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        return bool(child_names.intersection(_SHIFTS_LOCATIONS) or child_names.intersection(_SHIFTS_SPLITS))

    @classmethod
    def _resolve_shifts_root(cls, dataset_path: Path, dataset_subdir: str | None) -> Path:
        candidates: list[Path] = []
        if dataset_subdir:
            candidates.append(dataset_path / dataset_subdir)
        candidates.extend([dataset_path, dataset_path / dataset_path.name])
        if dataset_path.exists():
            candidates.extend(child for child in dataset_path.iterdir() if child.is_dir() and child.name != "__MACOSX")
        for candidate in candidates:
            if cls._looks_like_shifts_root(candidate):
                return candidate
        return dataset_path

    def _normalize_modality_mapping(self, mapping: dict[str, Any]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for key, aliases in self.DEFAULT_MAPPING.items():
            normalized[key] = _as_list(mapping.get(key), aliases)
        for key, aliases in mapping.items():
            if key not in normalized:
                normalized[str(key)] = _as_list(aliases)
        return normalized

    def _aliases_for(self, logical_name: str) -> list[str]:
        return self.modality_mapping.get(logical_name, [logical_name])

    def _location_dirs(self) -> list[tuple[str, Path]]:
        child_dirs = [child for child in self.root.iterdir() if child.is_dir() and child.name != "__MACOSX"]
        if any(child.name.lower() in _SHIFTS_SPLITS for child in child_dirs):
            return [("root", self.root)]

        available = {child.name.lower(): child for child in child_dirs}
        requested = [item.lower() for item in self.locations if item.lower() not in {"auto", "all", "*"}]
        if requested:
            missing = [name for name in requested if name not in available]
            if missing:
                raise FileNotFoundError(f"Requested Shifts MS location(s) not found under {self.root}: {missing}")
            return [(available[name].name, available[name]) for name in requested]

        ordered: list[Path] = []
        for preferred in self.preferred_locations:
            match = available.get(preferred.lower())
            if match is not None and match not in ordered:
                ordered.append(match)
        for child in sorted(child_dirs, key=lambda item: item.name.lower()):
            if child not in ordered:
                ordered.append(child)
        return [(path.name, path) for path in ordered]

    def _split_dirs(self, location_dir: Path) -> list[tuple[str, Path]]:
        available = {child.name.lower(): child for child in location_dir.iterdir() if child.is_dir()}
        requested = [item.lower() for item in self.splits if item.lower() not in {"auto", "all", "*"}]
        if requested:
            return [(available[name].name, available[name]) for name in requested if name in available]
        ordered = []
        for split in _SHIFTS_SPLITS:
            if split in available:
                ordered.append((available[split].name, available[split]))
        return ordered

    def _find_modality_folder(self, split_dir: Path, logical_name: str) -> tuple[Path | None, str | None]:
        folders = {child.name.lower(): child for child in split_dir.iterdir() if child.is_dir()}
        for alias in self._aliases_for(logical_name):
            match = folders.get(alias.lower())
            if match is not None:
                return match, match.name
        return None, None

    @staticmethod
    def _find_subject_file(folder: Path | None, subject_id: str) -> Path | None:
        if folder is None:
            return None
        candidates = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and _is_nifti(path) and (path.name.startswith(f"{subject_id}_") or path.stem == subject_id)
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _subject_sort_key(path: Path) -> tuple[int, int | str]:
        subject_id = _subject_id_from_shifts_path(path)
        if subject_id.isdigit():
            return (0, int(subject_id))
        return (1, subject_id)

    def _discover_subjects(self) -> list[ShiftsMSSubject]:
        subjects: list[ShiftsMSSubject] = []
        for location_name, location_dir in self._location_dirs():
            for split_name, split_dir in self._split_dirs(location_dir):
                reference_folder, _ = self._find_modality_folder(split_dir, self.reference_modality)
                if reference_folder is None:
                    continue
                reference_files = sorted(
                    [path for path in reference_folder.iterdir() if path.is_file() and _is_nifti(path)],
                    key=self._subject_sort_key,
                )
                for reference_path in reference_files:
                    subject_id = _subject_id_from_shifts_path(reference_path)
                    modality_paths: dict[str, Path | None] = {}
                    modality_sources: dict[str, str] = {}
                    for modality in self.modalities:
                        folder, source = self._find_modality_folder(split_dir, modality)
                        path = self._find_subject_file(folder, subject_id)
                        if path is None and self.require_modalities:
                            raise FileNotFoundError(
                                f"Missing modality '{modality}' for subject {subject_id} "
                                f"in {location_name}/{split_name}; aliases={self._aliases_for(modality)}"
                            )
                        modality_paths[modality] = path
                        modality_sources[modality] = source or "missing"

                    segmentation_folder, _ = self._find_modality_folder(split_dir, "segmentation")
                    segmentation_path = self._find_subject_file(segmentation_folder, subject_id)
                    if segmentation_path is None and self.require_segmentation:
                        raise FileNotFoundError(
                            f"Missing segmentation for subject {subject_id} in {location_name}/{split_name}; "
                            f"aliases={self._aliases_for('segmentation')}"
                        )

                    brain_mask_folder, _ = self._find_modality_folder(split_dir, "brain_mask")
                    brain_mask_path = self._find_subject_file(brain_mask_folder, subject_id)
                    subjects.append(
                        ShiftsMSSubject(
                            subject_id=subject_id,
                            location=location_name,
                            split=split_name,
                            reference_path=reference_path,
                            modality_paths=modality_paths,
                            modality_sources=modality_sources,
                            segmentation_path=segmentation_path,
                            brain_mask_path=brain_mask_path,
                        )
                    )
                    if self.subject_limit is not None and len(subjects) >= self.subject_limit:
                        return subjects
        return subjects

    def _load_nib(self, path: Path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("ShiftsMSVolumeDataset requires the optional 'nibabel' package.") from exc
        return nib.load(str(path))

    def _array_on_reference(self, image: Any, reference_image: Any, order: int) -> tuple[np.ndarray, bool]:
        same_grid = image.shape == reference_image.shape and np.allclose(image.affine, reference_image.affine, atol=1.0e-4)
        if same_grid:
            return np.asarray(image.dataobj, dtype=np.float32), False
        if not self.resample_to_reference:
            raise ValueError(
                f"Image grid {image.shape} does not match reference grid {reference_image.shape}; "
                "set data.resample_to_reference=true to resample."
            )
        try:
            from nibabel.processing import resample_from_to
        except ImportError as exc:
            raise ImportError("NIfTI resampling requires nibabel.processing.") from exc
        resampled = resample_from_to(image, (reference_image.shape, reference_image.affine), order=order)
        return np.asarray(resampled.dataobj, dtype=np.float32), True

    def _resize_image_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if tuple(volume.shape[1:3]) == (self.image_size, self.image_size):
            return volume.contiguous()
        slices = volume.permute(3, 0, 1, 2).contiguous()
        resized = F.interpolate(
            slices,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(1, 2, 3, 0).contiguous()

    def _resize_mask_volume(self, mask: torch.Tensor) -> torch.Tensor:
        if tuple(mask.shape[:2]) == (self.image_size, self.image_size):
            return mask.contiguous()
        slices = mask.permute(2, 0, 1).unsqueeze(1).float().contiguous()
        resized = F.interpolate(slices, size=(self.image_size, self.image_size), mode="nearest")
        return resized[:, 0].permute(1, 2, 0).bool().contiguous()

    def _metadata(
        self,
        subject: ShiftsMSSubject,
        reference_image: Any,
        model_shape: tuple[int, int, int],
        resampled_modalities: list[str],
    ) -> dict[str, Any]:
        return {
            "subject_id": subject.subject_id,
            "location": subject.location,
            "split": subject.split,
            "has_label": subject.segmentation_path is not None,
            "reference_modality": self.reference_modality,
            "reference_path": str(subject.reference_path),
            "native_shape": _shape_text(reference_image.shape),
            "model_shape": _shape_text(model_shape),
            "resampled_modalities": ",".join(resampled_modalities),
            "modality_mapping": {key: subject.modality_sources.get(key, "missing") for key in self.modalities},
            "input_paths": {key: str(value) if value is not None else "" for key, value in subject.modality_paths.items()},
            "segmentation_path": str(subject.segmentation_path) if subject.segmentation_path is not None else "",
            "brain_mask_path": str(subject.brain_mask_path) if subject.brain_mask_path is not None else "",
        }

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        subject = self.subjects[idx]
        reference_image = self._load_nib(subject.reference_path)
        images = []
        resampled_modalities: list[str] = []
        for modality in self.modalities:
            path = subject.modality_paths[modality]
            if path is None:
                images.append(np.zeros(reference_image.shape, dtype=np.float32))
                continue
            image = self._load_nib(path)
            array, resampled = self._array_on_reference(image, reference_image, order=1)
            if resampled:
                resampled_modalities.append(modality)
            images.append(array)

        image_array = np.stack(images, axis=0)
        if self.histogram_normalization:
            volume = histogram_normalize_volume(image_array)
        else:
            volume = normalize_volume(torch.from_numpy(image_array).float())

        if subject.segmentation_path is not None:
            segmentation_image = self._load_nib(subject.segmentation_path)
            segmentation, resampled = self._array_on_reference(segmentation_image, reference_image, order=0)
            if resampled:
                resampled_modalities.append("segmentation")
            mask = torch.from_numpy((segmentation > 0.5).astype(np.uint8)).bool()
        else:
            mask = torch.zeros(reference_image.shape, dtype=torch.bool)

        resized = self._resize_image_volume(volume)
        resized_mask = self._resize_mask_volume(mask)
        if not self.return_metadata:
            return resized, resized_mask
        metadata = self._metadata(subject, reference_image, tuple(resized_mask.shape), resampled_modalities)
        return resized, resized_mask, metadata


def build_dataset(config: dict[str, Any]) -> Dataset:
    """從 config 建立 dataset，避免上層 trainer/evaluator 寫死 dataset type 判斷。"""

    data_type = str(config.get("type", "lmdb")).lower()
    image_size = int(config.get("image_size", 128))

    if data_type == "lmdb":
        if "path" not in config:
            raise ValueError("data.path is required when data.type is 'lmdb'.")
        return LMDBSliceDataset(config["path"], image_size=image_size)
    if data_type in {"brats_healthy_slices", "healthy_slices"}:
        return BraTSHealthySliceDataset(
            csv_path=config["path_to_csv"],
            dataset_path=config["dataset_path"],
            image_size=image_size,
            modalities=config.get("modalities"),
            slice_column=str(config.get("slice_column", "Slice")),
            filename_separator=str(config.get("filename_separator", "_")),
        )
    if data_type in {"volume", "mri_volume", "brats_volume"}:
        return MRIDataVolume(
            csv_path=config.get("path_to_csv"),
            dataset_path=config["dataset_path"],
            image_size=image_size,
            modalities=config.get("modalities"),
            segmentation_suffix=str(config.get("segmentation_suffix", "seg")),
            histogram_normalization=bool(config.get("histogram_normalization", False)),
            shift_naming=bool(config.get("shift_naming", "shifts" in str(config.get("dataset_path", "")).lower())),
            filename_separator=str(config.get("filename_separator", "_")),
        )
    if data_type in {"ljubljana_ms_volume", "shifts_ms_volume", "shifts_volume"}:
        return ShiftsMSVolumeDataset(
            dataset_path=config["dataset_path"],
            image_size=image_size,
            modalities=config.get("modalities"),
            modality_mapping=config.get("modality_mapping"),
            dataset_subdir=config.get("dataset_subdir"),
            locations=config.get("locations"),
            location=config.get("location"),
            preferred_locations=config.get("preferred_locations"),
            splits=config.get("splits"),
            reference_modality=str(config.get("reference_modality", "flair")),
            require_segmentation=bool(config.get("require_segmentation", True)),
            require_modalities=bool(config.get("require_modalities", True)),
            resample_to_reference=bool(config.get("resample_to_reference", True)),
            histogram_normalization=bool(config.get("histogram_normalization", False)),
            return_metadata=bool(config.get("return_metadata", True)),
            subject_limit=config.get("subject_limit"),
        )

    raise ValueError(f"Unknown dataset type: {data_type}")


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=bool(config.get("shuffle", False)),
        num_workers=int(config.get("workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )
