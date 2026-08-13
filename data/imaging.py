"""Shared MRI primitives used by dataset adapters and preprocessing.

Only operations with identical historical behaviour live here.  Adapter-specific
NIfTI error messages, geometry checks, and interpolation policies remain owned by
their respective dataset adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision import transforms


DEFAULT_MODALITIES = ("flair", "t1", "t1ce", "t2")


def normalize_volume(images: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize each MRI modality by its foreground 99th percentile."""

    images = images.float()
    for modality in range(images.shape[0]):
        values = images[modality].reshape(-1)
        foreground = values[values > 0]
        if foreground.numel() == 0:
            continue
        percentile_99 = torch.quantile(foreground, 0.99).clamp_min(eps)
        images[modality] = images[modality] / percentile_99
    return images


def _load_nifti(path: Path, dtype: type = float) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("split_healthy requires the optional 'nibabel' package.") from exc

    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def load_subject_volume(
    dataset_path: Path,
    subject_id: str,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load one BraTS-layout MRI subject and its binarized segmentation mask."""

    subject_dir = dataset_path / subject_id
    images = []
    for modality in modalities:
        image_path = subject_dir / f"{subject_id}_{modality}.nii.gz"
        images.append(torch.from_numpy(_load_nifti(image_path, dtype=float)))

    mask_path = subject_dir / f"{subject_id}_seg.nii.gz"
    mask = _load_nifti(mask_path, dtype=int)
    mask[mask >= 1] = 1
    return normalize_volume(torch.stack(images, dim=0)), mask


def resize_slices(slices: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize N-by-C-by-H-by-W slices with the historical preprocessing policy."""

    if slices.shape[-2:] == (image_size, image_size):
        return slices
    return transforms.Resize(image_size, antialias=True)(slices)


__all__ = [
    "DEFAULT_MODALITIES",
    "_load_nifti",
    "load_subject_volume",
    "normalize_volume",
    "resize_slices",
]
