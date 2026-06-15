"""可組合的 anomaly map 後處理工具。

score-map step 處理連續 anomaly map，mask step 處理二值 segmentation。
兩者都透過 registry 擴充；新增濾波器時，只要新增一個 class 加上一個
decorator，不需要修改 evaluator 主流程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


def normalize_minmax(tensor: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    min_value = tensor.amin()
    max_value = tensor.amax()
    return (tensor - min_value) / (max_value - min_value).clamp_min(eps)


class BasePostprocessor(ABC):
    """score-map 與 mask 後處理 step 共用的基底介面。"""

    name = "base"

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """套用單一後處理 step。"""

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


SCORE_POSTPROCESSORS: dict[str, type[BasePostprocessor]] = {}
MASK_POSTPROCESSORS: dict[str, type[BasePostprocessor]] = {}


def register_score_postprocessor(*names: str):
    """註冊連續 anomaly score map 的後處理器。"""

    def decorator(cls: type[BasePostprocessor]) -> type[BasePostprocessor]:
        for name in names:
            SCORE_POSTPROCESSORS[name.lower()] = cls
        return cls

    return decorator


def register_mask_postprocessor(*names: str):
    """註冊二值 mask 的後處理器。"""

    def decorator(cls: type[BasePostprocessor]) -> type[BasePostprocessor]:
        for name in names:
            MASK_POSTPROCESSORS[name.lower()] = cls
        return cls

    return decorator


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
    """對 batched map 套用 median filter。

    預期形狀：
    - 3D volume map: [B, H, W, Z]
    - 2D slice map flatten/batch 後: [B, H, W]
    """

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
    """對 batched anomaly score map 套用 grayscale dilation。"""

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
    """對 batched segmentation mask 套用 binary dilation。"""

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
    """標記 batched binary volume 內的 connected components。"""

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
    """移除小於 min_size 的 connected binary component。"""

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

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return normalize_minmax(tensor, eps=self.eps)

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


def _legacy_score_pipeline(config: dict[str, Any]) -> list[dict[str, Any]]:
    """將舊版巢狀 YAML 設定轉成新版 ordered pipeline。"""

    pipeline = []
    if config.get("gray_dilation", {}).get("enabled", False):
        pipeline.append({"type": "gray_dilation", **config["gray_dilation"]})
    if config.get("median_filter", {}).get("enabled", False):
        pipeline.append({"type": "median_filter", **config["median_filter"]})
    if config.get("normalize", False):
        normalize_config = config.get("normalize")
        if isinstance(normalize_config, dict):
            pipeline.append({"type": "normalize", **normalize_config})
        else:
            pipeline.append({"type": "normalize"})
    return pipeline


def _legacy_mask_pipeline(config: dict[str, Any]) -> list[dict[str, Any]]:
    """將舊版 binary 後處理設定轉成 pipeline step。"""

    pipeline = []
    if config.get("binary_dilation", {}).get("enabled", False):
        pipeline.append({"type": "binary_dilation", **config["binary_dilation"]})
    if config.get("connected_components", {}).get("enabled", False):
        pipeline.append({"type": "connected_components", **config["connected_components"]})
    return pipeline


def build_postprocess_pipeline(
    config: dict[str, Any] | None,
    registry: dict[str, type[BasePostprocessor]],
    legacy_builder,
    step_kind: str | None = None,
) -> list[BasePostprocessor]:
    if not config:
        return []
    step_configs = config.get("pipeline")
    if step_configs is None:
        # 向後相容：pipeline 支援出現前寫的 config 仍可使用；
        # 新實驗建議改用明確的 ordered pipeline。
        step_configs = legacy_builder(config)
    if isinstance(step_configs, dict):
        step_configs = [step_configs]

    steps = []
    for step_config in step_configs:
        if not step_config or step_config.get("enabled", True) is False:
            continue
        step_type = str(step_config.get("type")).lower()
        if step_type not in registry:
            supported = ", ".join(registry.keys())
            label = f"{step_kind} postprocess" if step_kind else "postprocess"
            message = (
                f"Unknown {label} step: {step_type}.\n"
                f"Supported {label} steps: {supported}."
            )
            if step_kind == "mask" and step_type == "yen_threshold":
                message += (
                    "\nNote: yen_threshold is applied inside "
                    "VolumeEvaluator._yen_metrics() and should not be configured "
                    "as a mask postprocess step."
                )
            raise ValueError(message)
        kwargs = {
            key: value
            for key, value in step_config.items()
            if key not in {"type", "enabled"}
        }
        steps.append(registry[step_type](**kwargs))
    return steps


def apply_postprocess_pipeline(tensor: torch.Tensor, steps: list[BasePostprocessor]) -> torch.Tensor:
    """依照 config 順序執行後處理 step。"""

    output = tensor
    for step in steps:
        output = step(output)
    return output


def yen_threshold(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from skimage.filters import threshold_yen
    except ImportError:
        threshold = tensor.mean(dim=tuple(range(1, tensor.ndim)), keepdim=True)
        return (tensor > threshold).to(torch.bool), threshold.flatten()

    masks = []
    thresholds = []
    for item in tensor.detach().cpu():
        value = float(threshold_yen(item.numpy()))
        thresholds.append(value)
        masks.append(item > value)
    mask = torch.stack(masks).to(device=tensor.device)
    return mask.to(torch.bool), torch.tensor(thresholds, device=tensor.device, dtype=tensor.dtype)


def apply_score_postprocess(tensor: torch.Tensor, config: dict[str, Any] | None) -> torch.Tensor:
    """依 config 套用選擇性的 score-map 後處理。"""

    return apply_postprocess_pipeline(
        tensor,
        build_postprocess_pipeline(config, SCORE_POSTPROCESSORS, _legacy_score_pipeline, "score-map"),
    )


def apply_mask_postprocess(tensor: torch.Tensor, config: dict[str, Any] | None) -> torch.Tensor:
    """依 config 套用選擇性的 binary-mask 後處理。"""

    output = apply_postprocess_pipeline(
        tensor.bool(),
        build_postprocess_pipeline(config, MASK_POSTPROCESSORS, _legacy_mask_pipeline, "mask"),
    )
    return output.bool()
