"""Shared MRI intensity normalization helpers.

Only helpers whose existing adapter behaviour is identical live here.  Geometry,
resampling, and resizing remain adapter-local because their contracts differ.
"""

from __future__ import annotations

import numpy as np
import torch

from ..imaging import normalize_volume


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
