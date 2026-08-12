"""Per-volume score-to-mask threshold selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch

from .base import SUPPORTED_THRESHOLD_METHODS
from .numerics import sanitize_scores


def _validate_threshold_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_THRESHOLD_METHODS:
        supported = ", ".join(SUPPORTED_THRESHOLD_METHODS)
        raise ValueError(
            f"Unknown threshold method: {method!r}. Supported methods: {supported}."
        )
    return normalized


def _load_yen_threshold() -> Callable[[np.ndarray], float]:
    from skimage.filters import threshold_yen

    return threshold_yen


def _load_otsu_threshold() -> Callable[[np.ndarray], float]:
    from skimage.filters import threshold_otsu

    return threshold_otsu


# Keep optional scikit-image imports lazy.  The fixed insertion order mirrors
# SUPPORTED_THRESHOLD_METHODS and leaves a small, isolated extension point for
# additional threshold algorithms.
THRESHOLD_FUNCTION_LOADERS: dict[str, Callable[[], Callable[[np.ndarray], float]]] = {
    "yen": _load_yen_threshold,
    "otsu": _load_otsu_threshold,
}


def threshold_anomaly_map(
    tensor: torch.Tensor,
    method: str = "yen",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold each batched 3-D volume and return masks plus per-volume values.

    Yen historically used a strict ``>`` comparator in this project, so Otsu
    deliberately uses the same comparator. A constant volume receives its
    constant value as the recorded threshold and an empty mask; a warning keeps
    that otherwise undefined thresholding decision visible without aborting an
    evaluation batch.
    """

    method = _validate_threshold_method(method)
    finite = sanitize_scores(tensor)
    if method == "yen":
        try:
            threshold_function = THRESHOLD_FUNCTION_LOADERS[method]()
        except ImportError:
            # Preserve the rewrite's historical Yen-only fallback. Otsu has no
            # project fallback because scikit-image is the authoritative
            # implementation used by this repository.
            dimensions = tuple(range(1, finite.ndim))
            threshold = finite.mean(dim=dimensions, keepdim=True)
            warnings.warn(
                "scikit-image is unavailable; using the legacy mean fallback for Yen thresholding.",
                RuntimeWarning,
                stacklevel=2,
            )
            return (finite > threshold).to(torch.bool), threshold.flatten()
    else:
        try:
            threshold_function = THRESHOLD_FUNCTION_LOADERS[method]()
        except ImportError as exc:
            raise ImportError(
                "Otsu thresholding requires scikit-image (skimage.filters.threshold_otsu)."
            ) from exc

    masks = []
    thresholds = []
    constant_volumes = 0
    for item in finite.detach().cpu():
        array = item.numpy()
        if array.size == 0 or np.all(array == array.flat[0]):
            value = float(array.flat[0]) if array.size else 0.0
            constant_volumes += 1
        else:
            value = float(threshold_function(array))
        thresholds.append(value)
        masks.append(item > value)
    if constant_volumes:
        warnings.warn(
            f"{method.title()} thresholding received {constant_volumes} constant volume(s); "
            "the threshold was set to the constant value and the binary mask is empty.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not masks:
        empty_thresholds = torch.empty((0,), device=finite.device, dtype=finite.dtype)
        return torch.empty_like(finite, dtype=torch.bool), empty_thresholds
    mask = torch.stack(masks).to(device=finite.device)
    return mask.to(torch.bool), torch.tensor(
        thresholds,
        device=finite.device,
        dtype=finite.dtype,
    )


def yen_threshold(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible Yen wrapper around :func:`threshold_anomaly_map`."""

    return threshold_anomaly_map(tensor, method="yen")


def otsu_threshold(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for per-volume Otsu thresholding."""

    return threshold_anomaly_map(tensor, method="otsu")
