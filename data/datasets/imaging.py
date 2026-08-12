"""Shared MRI intensity normalization helpers.

Only helpers whose existing adapter behaviour is identical live here.  Geometry,
resampling, and resizing remain adapter-local because their contracts differ.
"""

from __future__ import annotations

import numpy as np
import torch


def normalize_volume(images: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize each modality by its foreground 99th percentile."""

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
