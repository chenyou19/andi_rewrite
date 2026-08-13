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
    if normalized not in THRESHOLD_FUNCTION_LOADERS:
        supported = ", ".join(supported_threshold_methods())
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


def supported_threshold_methods() -> tuple[str, ...]:
    """Return the currently registered methods in deterministic registry order."""

    return tuple(THRESHOLD_FUNCTION_LOADERS)


def register_threshold_method(*names: str):
    """Register a lazy threshold-function loader under one or more aliases.

    The loader indirection keeps optional scientific dependencies lazy.  The
    existing mutable registry remains the authoritative runtime extension
    surface; ``SUPPORTED_THRESHOLD_METHODS`` remains the built-in compatibility
    snapshot ``("yen", "otsu")``.
    """

    normalized_names = tuple(str(name).strip().lower() for name in names)
    if not normalized_names or any(not name for name in normalized_names):
        raise ValueError("At least one non-empty threshold method name is required.")

    def decorator(
        loader: Callable[[], Callable[[np.ndarray], float]],
    ) -> Callable[[], Callable[[np.ndarray], float]]:
        for name in normalized_names:
            THRESHOLD_FUNCTION_LOADERS[name] = loader
        return loader

    return decorator


def _yen_import_fallback(
    finite: torch.Tensor,
    _error: ImportError,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = tuple(range(1, finite.ndim))
    threshold = finite.mean(dim=dimensions, keepdim=True)
    warnings.warn(
        "scikit-image is unavailable; using the legacy mean fallback for Yen thresholding.",
        RuntimeWarning,
        stacklevel=3,
    )
    return (finite > threshold).to(torch.bool), threshold.flatten()


def _otsu_import_error(
    _finite: torch.Tensor,
    error: ImportError,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise ImportError(
        "Otsu thresholding requires scikit-image (skimage.filters.threshold_otsu)."
    ) from error


_THRESHOLD_IMPORT_FAILURE_HANDLERS: dict[
    str,
    Callable[[torch.Tensor, ImportError], tuple[torch.Tensor, torch.Tensor]],
] = {
    "yen": _yen_import_fallback,
    "otsu": _otsu_import_error,
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
    try:
        threshold_function = THRESHOLD_FUNCTION_LOADERS[method]()
    except ImportError as exc:
        handler = _THRESHOLD_IMPORT_FAILURE_HANDLERS.get(method)
        if handler is None:
            raise
        return handler(finite, exc)

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
