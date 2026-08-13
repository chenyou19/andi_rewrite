"""Numerical safety and score normalization helpers."""

from __future__ import annotations

import torch

from .base import NORMALIZATION_SCOPES


def sanitize_scores(tensor: torch.Tensor) -> torch.Tensor:
    """Return finite floating-point scores without changing normal finite input.

    The original ANDi implementation assumes finite, non-constant input.  The
    rewrite keeps its existing numerical safety extension by mapping all
    non-finite values to zero before score postprocessing.
    """

    if not tensor.is_floating_point():
        tensor = tensor.float()
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_normalization_scope(scope: str) -> str:
    normalized = str(scope).strip().lower()
    if normalized not in NORMALIZATION_SCOPES:
        supported = ", ".join(sorted(NORMALIZATION_SCOPES))
        raise ValueError(f"Unknown normalization scope: {scope!r}. Supported scopes: {supported}.")
    return normalized


def normalize_minmax(
    tensor: torch.Tensor,
    eps: float = 1.0e-8,
    scope: str = "dataset",
) -> torch.Tensor:
    """Safely Min-Max normalize a dataset or each subject independently.

    ``dataset`` uses one minimum and maximum over the complete input tensor,
    matching ``norm_tensor`` in the original ANDi evaluation after all test
    subjects have been collected. ``subject`` keeps batch items independent.
    """

    finite = sanitize_scores(tensor)
    if finite.numel() == 0:
        return finite.clone()
    scope = _validate_normalization_scope(scope)
    if scope == "subject" and finite.ndim > 1:
        dimensions = tuple(range(1, finite.ndim))
        min_value = finite.amin(dim=dimensions, keepdim=True)
        max_value = finite.amax(dim=dimensions, keepdim=True)
    else:
        min_value = finite.amin()
        max_value = finite.amax()
    return (finite - min_value) / (max_value - min_value).clamp_min(float(eps))
