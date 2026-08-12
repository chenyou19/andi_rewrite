"""Pure morphology helpers and concrete postprocessing steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import (
    BasePostprocessor,
    register_mask_postprocessor,
    register_score_postprocessor,
)
from .numerics import normalize_minmax


def _as_kernel_size(kernel_size: int | tuple[int, ...], ndim: int) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return kernel_size
    return tuple([int(kernel_size)] * ndim)


def _kernel_enabled(kernel_size: int | tuple[int, ...] | None) -> bool:
    if kernel_size is None:
        return False
    if isinstance(kernel_size, tuple):
        return any(int(value) > 1 for value in kernel_size)
    return int(kernel_size) > 1


def median_filter_tensor(
    tensor: torch.Tensor,
    kernel_size: int | tuple[int, ...] | None,
    mode: str = "3d",
) -> torch.Tensor:
    """Apply a median filter to each batched 2-D or 3-D map."""

    if not _kernel_enabled(kernel_size):
        return tensor
    try:
        from scipy.ndimage import median_filter
        from scipy.signal import medfilt2d
    except ImportError:
        return tensor

    array = tensor.detach().cpu().numpy()
    filtered = np.empty_like(array)
    if mode.lower() == "2d":
        kernel_2d = int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
        for index in range(array.shape[0]):
            if array[index].ndim == 2:
                filtered[index] = medfilt2d(array[index], kernel_size=kernel_2d)
            else:
                for slice_index in range(array[index].shape[-1]):
                    filtered[index, :, :, slice_index] = medfilt2d(
                        array[index, :, :, slice_index],
                        kernel_size=kernel_2d,
                    )
    else:
        for index in range(array.shape[0]):
            filtered[index] = median_filter(
                array[index],
                size=_as_kernel_size(kernel_size, array[index].ndim),
            )
    return torch.from_numpy(filtered).to(device=tensor.device, dtype=tensor.dtype)


def gray_dilation_tensor(
    tensor: torch.Tensor,
    kernel_size: int | tuple[int, ...] | None = 3,
) -> torch.Tensor:
    """Apply grayscale dilation to each batched anomaly-score map."""

    if not _kernel_enabled(kernel_size):
        return tensor
    try:
        from scipy.ndimage import grey_dilation
    except ImportError:
        return tensor

    array = tensor.detach().cpu().numpy()
    dilated = np.empty_like(array)
    for index in range(array.shape[0]):
        dilated[index] = grey_dilation(
            array[index],
            size=_as_kernel_size(kernel_size, array[index].ndim),
        )
    return torch.from_numpy(dilated).to(device=tensor.device, dtype=tensor.dtype)


def binary_dilation_tensor(
    tensor: torch.Tensor,
    rank: int = 3,
    connectivity: int = 1,
    iterations: int = 1,
) -> torch.Tensor:
    """Apply binary dilation to each batched segmentation mask."""

    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure
    except ImportError:
        return tensor

    structure = generate_binary_structure(rank, connectivity)
    array = tensor.detach().cpu().numpy().astype(bool)
    dilated = np.empty_like(array)
    for index in range(array.shape[0]):
        dilated[index] = binary_dilation(
            array[index],
            structure=structure,
            iterations=int(iterations),
        )
    return torch.from_numpy(dilated).to(device=tensor.device)


def connected_components_tensor(
    tensor: torch.Tensor,
    connectivity: int = 3,
) -> tuple[torch.Tensor, list[list[dict[str, int]]]]:
    """Return connected-component labels and properties for each binary volume."""

    try:
        from skimage.measure import label, regionprops
    except ImportError:
        labels = torch.zeros_like(tensor, dtype=torch.int64)
        return labels, [[] for _ in range(tensor.shape[0])]

    array = tensor.detach().cpu().numpy().astype(bool)
    labeled_items = []
    all_components = []
    for item in array:
        labeled = label(item, connectivity=int(connectivity))
        components = [
            {
                "label": int(prop.label),
                "area": int(prop.area),
            }
            for prop in regionprops(labeled)
        ]
        labeled_items.append(labeled)
        all_components.append(components)
    labels = torch.from_numpy(np.stack(labeled_items)).to(device=tensor.device, dtype=torch.int64)
    return labels, all_components


def remove_small_components_tensor(
    tensor: torch.Tensor,
    min_size: int = 20,
    connectivity: int = 3,
) -> torch.Tensor:
    """Remove connected binary components smaller than ``min_size``."""

    if int(min_size) <= 0:
        return tensor
    labels, components = connected_components_tensor(tensor, connectivity=connectivity)
    cleaned = tensor.clone().bool()
    for batch_index, item_components in enumerate(components):
        for component in item_components:
            if component["area"] < int(min_size):
                component_mask = labels[batch_index] == component["label"]
                cleaned[batch_index] = torch.where(
                    component_mask,
                    torch.zeros_like(cleaned[batch_index]),
                    cleaned[batch_index],
                )
    return cleaned


@register_score_postprocessor("normalize", "minmax", "normalize_minmax")
class NormalizePostprocessor(BasePostprocessor):
    name = "normalize"

    def __init__(self, eps: float = 1.0e-8):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor, scope: str | None = None) -> torch.Tensor:
        return normalize_minmax(tensor, eps=self.eps, scope=scope or "dataset")

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "eps": self.eps}


@register_score_postprocessor("median_filter", "median", "mf")
class MedianFilterPostprocessor(BasePostprocessor):
    name = "median_filter"

    def __init__(self, kernel_size: int | tuple[int, ...] = 5, mode: str = "3d"):
        self.kernel_size = kernel_size
        self.mode = str(mode)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return median_filter_tensor(tensor, kernel_size=self.kernel_size, mode=self.mode)

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "kernel_size": self.kernel_size, "mode": self.mode}


@register_score_postprocessor("gray_dilation", "grey_dilation")
class GrayDilationPostprocessor(BasePostprocessor):
    name = "gray_dilation"

    def __init__(self, kernel_size: int | tuple[int, ...] = 3):
        self.kernel_size = kernel_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return gray_dilation_tensor(tensor, kernel_size=self.kernel_size)

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "kernel_size": self.kernel_size}


@register_mask_postprocessor("binary_dilation", "dilation")
class BinaryDilationPostprocessor(BasePostprocessor):
    name = "binary_dilation"

    def __init__(self, rank: int = 3, connectivity: int = 1, iterations: int = 1):
        self.rank = int(rank)
        self.connectivity = int(connectivity)
        self.iterations = int(iterations)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return binary_dilation_tensor(
            tensor,
            rank=self.rank,
            connectivity=self.connectivity,
            iterations=self.iterations,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "rank": self.rank,
            "connectivity": self.connectivity,
            "iterations": self.iterations,
        }


@register_mask_postprocessor("connected_components", "remove_small_components", "cc")
class ConnectedComponentsPostprocessor(BasePostprocessor):
    name = "connected_components"

    def __init__(self, min_size: int = 20, connectivity: int = 3):
        self.min_size = int(min_size)
        self.connectivity = int(connectivity)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return remove_small_components_tensor(
            tensor,
            min_size=self.min_size,
            connectivity=self.connectivity,
        )

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "min_size": self.min_size, "connectivity": self.connectivity}
