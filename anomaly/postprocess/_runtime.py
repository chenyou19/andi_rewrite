"""Runtime indirection for historically patchable postprocess functions.

The former monolithic module resolved these functions from its own globals at
execution time.  Keeping the providers here lets split implementation modules
retain that behavior without importing the package facade in reverse.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


MedianFilterFunction = Callable[
    [torch.Tensor, int | tuple[int, ...] | None, str],
    torch.Tensor,
]
NormalizeMinmaxFunction = Callable[[torch.Tensor, float, str], torch.Tensor]


_median_filter_tensor: MedianFilterFunction | None = None
_normalize_minmax: NormalizeMinmaxFunction | None = None


def configure_runtime_functions(
    *,
    median_filter_tensor: MedianFilterFunction,
    normalize_minmax: NormalizeMinmaxFunction,
) -> None:
    """Install facade-owned dispatchers after package initialization."""

    global _median_filter_tensor, _normalize_minmax
    _median_filter_tensor = median_filter_tensor
    _normalize_minmax = normalize_minmax


def apply_median_filter_tensor(
    tensor: torch.Tensor,
    kernel_size: int | tuple[int, ...] | None,
    mode: str = "3d",
) -> torch.Tensor:
    function = _median_filter_tensor
    if function is None:  # pragma: no cover - package initialization installs it.
        raise RuntimeError("Postprocess runtime functions have not been configured.")
    return function(tensor, kernel_size, mode)


def apply_normalize_minmax(
    tensor: torch.Tensor,
    eps: float = 1.0e-8,
    scope: str = "dataset",
) -> torch.Tensor:
    function = _normalize_minmax
    if function is None:  # pragma: no cover - package initialization installs it.
        raise RuntimeError("Postprocess runtime functions have not been configured.")
    return function(tensor, eps, scope)
