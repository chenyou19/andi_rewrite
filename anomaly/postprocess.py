"""可組合的 anomaly map 後處理工具。

score-map step 處理連續 anomaly map，mask step 處理二值 segmentation。
兩者都透過 registry 擴充；新增濾波器時，只要新增一個 class 加上一個
decorator，不需要修改 evaluator 主流程。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


NORMALIZATION_SCOPES = {"dataset", "subject"}
SUPPORTED_THRESHOLD_METHODS = ("yen", "otsu")


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


def _validate_threshold_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_THRESHOLD_METHODS:
        supported = ", ".join(SUPPORTED_THRESHOLD_METHODS)
        raise ValueError(
            f"Unknown threshold method: {method!r}. Supported methods: {supported}."
        )
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
        warnings.warn(
            "Nested postprocess settings without an explicit 'pipeline' are deprecated; "
            "they were translated to an ordered pipeline for backward compatibility.",
            FutureWarning,
            stacklevel=2,
        )
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
            if step_kind == "mask" and step_type in {"yen_threshold", "otsu_threshold"}:
                message += (
                    f"\nNote: {step_type} is a score-to-mask thresholding stage "
                    "inside PostprocessPolicy and should not be configured as a "
                    "mask postprocess step."
                )
            raise ValueError(message)
        kwargs = {
            key: value
            for key, value in step_config.items()
            if key not in {"type", "enabled"}
        }
        steps.append(registry[step_type](**kwargs))
    return steps


def apply_postprocess_pipeline(
    tensor: torch.Tensor,
    steps: list[BasePostprocessor],
    normalization_scope: str | None = None,
) -> torch.Tensor:
    """依照 config 順序執行後處理 step。"""

    output = tensor
    for step in steps:
        if isinstance(step, NormalizePostprocessor):
            output = step(output, scope=normalization_scope)
        else:
            output = step(output)
    return output


def threshold_anomaly_map(
    tensor: torch.Tensor,
    method: str = "yen",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold each batched 3D volume and return masks plus per-volume values.

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
            from skimage.filters import threshold_yen as threshold_function
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
            from skimage.filters import threshold_otsu as threshold_function
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


def apply_score_postprocess(
    tensor: torch.Tensor,
    config: dict[str, Any] | None,
    normalization_scope: str | None = None,
) -> torch.Tensor:
    """依 config 套用選擇性的 score-map 後處理。"""

    return apply_postprocess_pipeline(
        tensor,
        build_postprocess_pipeline(config, SCORE_POSTPROCESSORS, _legacy_score_pipeline, "score-map"),
        normalization_scope=normalization_scope,
    )


def apply_mask_postprocess(tensor: torch.Tensor, config: dict[str, Any] | None) -> torch.Tensor:
    """依 config 套用選擇性的 binary-mask 後處理。"""

    output = apply_postprocess_pipeline(
        tensor.bool(),
        build_postprocess_pipeline(config, MASK_POSTPROCESSORS, _legacy_mask_pipeline, "mask"),
    )
    return output.bool()


@dataclass(frozen=True)
class PostprocessResult:
    """All score and selected-threshold products from one raw-map tensor."""

    score_raw: torch.Tensor
    score_mf: torch.Tensor
    thresholds_raw: torch.Tensor
    thresholds_mf: torch.Tensor
    binary_mask_raw: torch.Tensor
    binary_mask_mf: torch.Tensor
    binary_mask_raw_postprocessed: torch.Tensor
    binary_mask_mf_postprocessed: torch.Tensor
    threshold_method: str
    normalization_scope: str

    def as_dict(self) -> dict[str, torch.Tensor | str]:
        payload: dict[str, torch.Tensor | str] = {
            "score_raw": self.score_raw,
            "score_mf": self.score_mf,
            "thresholds_raw": self.thresholds_raw,
            "thresholds_mf": self.thresholds_mf,
            "binary_mask_raw": self.binary_mask_raw,
            "binary_mask_mf": self.binary_mask_mf,
            "binary_mask_raw_postprocessed": self.binary_mask_raw_postprocessed,
            "binary_mask_mf_postprocessed": self.binary_mask_mf_postprocessed,
            "threshold_method": self.threshold_method,
            "normalization_scope": self.normalization_scope,
        }
        if self.threshold_method == "yen":
            payload.update(
                {
                    "yen_thresholds_raw": self.thresholds_raw,
                    "yen_thresholds_mf": self.thresholds_mf,
                    "yen_mask_raw": self.binary_mask_raw,
                    "yen_mask_mf": self.binary_mask_mf,
                    "yen_mask_raw_postprocessed": self.binary_mask_raw_postprocessed,
                    "yen_mask_mf_postprocessed": self.binary_mask_mf_postprocessed,
                }
            )
        return payload

    # Compatibility aliases keep existing Yen callers byte-for-byte stable.
    # New code should use the method-neutral fields above.
    @property
    def yen_thresholds_raw(self) -> torch.Tensor:
        return self.thresholds_raw

    @property
    def yen_thresholds_mf(self) -> torch.Tensor:
        return self.thresholds_mf

    @property
    def yen_mask_raw(self) -> torch.Tensor:
        return self.binary_mask_raw

    @property
    def yen_mask_mf(self) -> torch.Tensor:
        return self.binary_mask_mf

    @property
    def yen_mask_raw_postprocessed(self) -> torch.Tensor:
        return self.binary_mask_raw_postprocessed

    @property
    def yen_mask_mf_postprocessed(self) -> torch.Tensor:
        return self.binary_mask_mf_postprocessed

    @classmethod
    def concatenate(cls, results: list["PostprocessResult"]) -> "PostprocessResult":
        if not results:
            raise ValueError("Cannot concatenate an empty list of postprocess results.")
        scopes = {result.normalization_scope for result in results}
        if len(scopes) != 1:
            raise ValueError(f"Cannot concatenate mixed normalization scopes: {sorted(scopes)}")
        methods = {result.threshold_method for result in results}
        if len(methods) != 1:
            raise ValueError(f"Cannot concatenate mixed threshold methods: {sorted(methods)}")

        def combine(name: str) -> torch.Tensor:
            return torch.cat([getattr(result, name) for result in results], dim=0)

        return cls(
            score_raw=combine("score_raw"),
            score_mf=combine("score_mf"),
            thresholds_raw=combine("thresholds_raw"),
            thresholds_mf=combine("thresholds_mf"),
            binary_mask_raw=combine("binary_mask_raw"),
            binary_mask_mf=combine("binary_mask_mf"),
            binary_mask_raw_postprocessed=combine("binary_mask_raw_postprocessed"),
            binary_mask_mf_postprocessed=combine("binary_mask_mf_postprocessed"),
            threshold_method=results[0].threshold_method,
            normalization_scope=results[0].normalization_scope,
        )


def _step_label(step: BasePostprocessor, normalization_scope: str) -> str:
    if isinstance(step, NormalizePostprocessor):
        return f"{normalization_scope}_minmax"
    if isinstance(step, MedianFilterPostprocessor):
        return f"median_filter_{step.mode.lower()}(kernel={step.kernel_size})"
    if isinstance(step, GrayDilationPostprocessor):
        return f"gray_dilation(kernel={step.kernel_size})"
    if isinstance(step, BinaryDilationPostprocessor):
        return (
            "binary_dilation("
            f"rank={step.rank}, connectivity={step.connectivity}, iterations={step.iterations}"
            ")"
        )
    if isinstance(step, ConnectedComponentsPostprocessor):
        return f"connected_components(min_size={step.min_size}, connectivity={step.connectivity})"
    return step.name


def _pipeline_labels(steps: list[BasePostprocessor], normalization_scope: str) -> list[str]:
    return [_step_label(step, normalization_scope) for step in steps]


class PostprocessPolicy(ABC):
    """Single source of truth for score, threshold, and mask postprocessing."""

    mode = "base"

    def __init__(
        self,
        *,
        normalization_scope: str,
        threshold_method: str = "yen",
        threshold_mask_config: dict[str, Any] | None = None,
        binary_mask_config: dict[str, Any] | None = None,
        yen_mask_config: dict[str, Any] | None = None,
    ):
        self.normalization_scope = _validate_normalization_scope(normalization_scope)
        self.threshold_method = _validate_threshold_method(threshold_method)
        self.threshold_mask_steps = build_postprocess_pipeline(
            threshold_mask_config,
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )
        selected_mask_config = binary_mask_config if binary_mask_config is not None else yen_mask_config
        self.binary_mask_steps = build_postprocess_pipeline(
            selected_mask_config,
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )
        # Attribute retained for callers that inspect the old Yen-specific name.
        self.yen_mask_steps = self.binary_mask_steps

    @abstractmethod
    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        """Derive both score branches and their selected-method masks."""

    def _complete(
        self,
        score_raw: torch.Tensor,
        score_mf: torch.Tensor,
        normalization_scope: str,
    ) -> PostprocessResult:
        score_raw = sanitize_scores(score_raw)
        score_mf = sanitize_scores(score_mf)
        binary_mask_raw, thresholds_raw = threshold_anomaly_map(
            score_raw,
            method=self.threshold_method,
        )
        binary_mask_mf, thresholds_mf = threshold_anomaly_map(
            score_mf,
            method=self.threshold_method,
        )
        binary_mask_raw_postprocessed = apply_postprocess_pipeline(
            binary_mask_raw,
            self.binary_mask_steps,
        ).bool()
        binary_mask_mf_postprocessed = apply_postprocess_pipeline(
            binary_mask_mf,
            self.binary_mask_steps,
        ).bool()
        return PostprocessResult(
            score_raw=score_raw,
            score_mf=score_mf,
            thresholds_raw=thresholds_raw,
            thresholds_mf=thresholds_mf,
            binary_mask_raw=binary_mask_raw,
            binary_mask_mf=binary_mask_mf,
            binary_mask_raw_postprocessed=binary_mask_raw_postprocessed,
            binary_mask_mf_postprocessed=binary_mask_mf_postprocessed,
            threshold_method=self.threshold_method,
            normalization_scope=normalization_scope,
        )

    def fixed_threshold_mask(self, score: torch.Tensor, threshold: float) -> torch.Tensor:
        mask = sanitize_scores(score) > float(threshold)
        return apply_postprocess_pipeline(mask, self.threshold_mask_steps).bool()

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Describe the exact executable order for reports and debugging."""


class OriginalANDiPostprocessPolicy(PostprocessPolicy):
    """Reference-compatible postprocessing from AlexanderFrotscher/ANDi."""

    mode = "original_andi"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        eps: float = 1.0e-8,
        *,
        threshold_method: str = "yen",
    ):
        try:
            from scipy import ndimage as _scipy_ndimage  # noqa: F401
            from skimage import filters as _skimage_filters  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "original_andi postprocessing requires scipy and scikit-image; "
                "the rewrite fallbacks are not reference-equivalent."
            ) from exc
        config = config or {}
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", True))
        self.median_kernel_size = int(median_config.get("kernel_size", 5))
        self.median_mode = str(median_config.get("mode", "3d")).lower()
        if self.median_mode != "3d":
            raise ValueError("original_andi median_filter.mode must be '3d'.")
        self.eps = float(config.get("eps", eps))

        binary_mask_config = config.get("binary_mask")
        if not isinstance(binary_mask_config, dict):
            binary_mask_config = config.get("yen", {})
        dilation_config = binary_mask_config.get("binary_dilation", {})
        dilation_enabled = bool(dilation_config.get("enabled", True))
        self.dilation_settings = {
            "enabled": dilation_enabled,
            "rank": int(dilation_config.get("rank", 3)),
            "connectivity": int(dilation_config.get("connectivity", 1)),
            "iterations": int(dilation_config.get("iterations", 1)),
        }
        binary_mask_pipeline: list[dict[str, Any]] = []
        if dilation_enabled:
            binary_mask_pipeline.append({"type": "binary_dilation", **self.dilation_settings})
        normalization_scope = _validate_normalization_scope(
            str(config.get("normalization_scope", "dataset"))
        )
        if normalization_scope != "dataset":
            raise ValueError(
                "metrics.original_andi.normalization_scope must be 'dataset' to reproduce "
                "the reference evaluation. Use prediction_output.normalization_scope for "
                "an explicitly different export scope."
            )
        super().__init__(
            normalization_scope=normalization_scope,
            threshold_method=threshold_method,
            threshold_mask_config={"pipeline": []},
            binary_mask_config={"pipeline": binary_mask_pipeline},
        )

    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        scope = _validate_normalization_scope(normalization_scope or self.normalization_scope)
        raw_finite = sanitize_scores(raw_maps)
        raw_mf = (
            median_filter_tensor(
                raw_finite,
                kernel_size=self.median_kernel_size,
                mode=self.median_mode,
            )
            if self.median_enabled
            else raw_finite.clone()
        )
        # The branches are intentionally normalized independently and only
        # after the MF branch has filtered the unnormalized raw anomaly map.
        score_raw = normalize_minmax(raw_finite, eps=self.eps, scope=scope)
        score_mf = normalize_minmax(raw_mf, eps=self.eps, scope=scope)
        return self._complete(score_raw, score_mf, scope)

    def describe(self) -> dict[str, Any]:
        scope = self.normalization_scope
        mf_pipeline = ["nan_to_num"]
        if self.median_enabled:
            mf_pipeline.append(f"median_filter_3d(kernel={self.median_kernel_size})")
        mf_pipeline.append(f"{scope}_minmax")
        binary_mask_pipeline = [
            f"subject_{self.threshold_method}_threshold",
            *_pipeline_labels(self.binary_mask_steps, scope),
        ]
        description = {
            "postprocess_mode": self.mode,
            "normalization_scope": scope,
            "raw_score_pipeline": ["nan_to_num", f"{scope}_minmax"],
            "mf_score_pipeline": mf_pipeline,
            "threshold_method": self.threshold_method,
            "threshold_strategy": "per_subject_3d_volume",
            "threshold_comparator": ">",
            "binary_mask_pipeline": binary_mask_pipeline,
            "fixed_threshold_mask_pipeline": ["score > threshold"],
            "median_filter_settings": {
                "enabled": self.median_enabled,
                "kernel_size": self.median_kernel_size,
                "mode": self.median_mode,
            },
            "dilation_settings": dict(self.dilation_settings),
            "numerical_safety": {
                "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                "eps": self.eps,
                "constant_tensor": "zeros",
                "constant_volume_threshold": "constant value; empty mask with RuntimeWarning",
            },
        }
        if self.threshold_method == "yen":
            description.update(
                {
                    "yen_threshold_strategy": description["threshold_strategy"],
                    "yen_mask_pipeline": binary_mask_pipeline,
                }
            )
        return description


def _compile_legacy_normalization_steps(
    steps: list[BasePostprocessor],
    *,
    prepend_normalize: bool,
    input_is_normalized: bool,
    ensure_final_normalize: bool,
    eps: float,
) -> list[BasePostprocessor]:
    """Compile old implicit normalization into one explicit, traceable path."""

    compiled: list[BasePostprocessor] = []
    normalized = input_is_normalized
    if prepend_normalize:
        compiled.append(NormalizePostprocessor(eps=eps))
        normalized = True
    for step in steps:
        if isinstance(step, NormalizePostprocessor):
            if not normalized:
                compiled.append(step)
            normalized = True
        else:
            compiled.append(step)
            normalized = False
    if ensure_final_normalize and not normalized:
        compiled.append(NormalizePostprocessor(eps=eps))
    return compiled


class RewritePostprocessPolicy(PostprocessPolicy):
    """Configurable rewrite pipelines, including legacy-compatible defaults."""

    mode = "rewrite"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        anomaly_config: dict[str, Any] | None = None,
        *,
        threshold_method: str = "yen",
        legacy_compatibility: bool = False,
        legacy_profile: str = "evaluator",
    ):
        config = config or {}
        anomaly_config = anomaly_config or {}
        rewrite_config = config.get("rewrite", {}) if isinstance(config.get("rewrite"), dict) else {}
        postprocess_config = config.get("postprocess")
        if not isinstance(postprocess_config, dict):
            postprocess_config = rewrite_config.get("postprocess")
        if not isinstance(postprocess_config, dict):
            postprocess_config = anomaly_config.get("postprocess", {})

        median_config = config.get("median_filter")
        if not isinstance(median_config, dict):
            median_config = anomaly_config.get("median_filter", {})
        median_enabled = bool(median_config.get("enabled", True))
        median_kernel = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        median_mode = str(median_config.get("mode", "3d"))
        self.eps = float(config.get("eps", anomaly_config.get("eps", 1.0e-8)))
        self.legacy_compatibility = bool(legacy_compatibility)
        self.legacy_profile = str(legacy_profile)

        score_config = postprocess_config.get("score")
        score_mf_config = postprocess_config.get("score_mf")
        if score_config is None:
            score_config = {} if legacy_compatibility else {"pipeline": [{"type": "normalize"}]}
        if score_mf_config is None:
            default_mf_pipeline: list[dict[str, Any]] = []
            if median_enabled:
                default_mf_pipeline.append(
                    {"type": "median_filter", "kernel_size": median_kernel, "mode": median_mode}
                )
            if not legacy_compatibility:
                default_mf_pipeline.append({"type": "normalize"})
            score_mf_config = {"pipeline": default_mf_pipeline}

        raw_steps = build_postprocess_pipeline(
            score_config,
            SCORE_POSTPROCESSORS,
            _legacy_score_pipeline,
            "score-map",
        )
        mf_steps = build_postprocess_pipeline(
            score_mf_config,
            SCORE_POSTPROCESSORS,
            _legacy_score_pipeline,
            "score-map",
        )
        if legacy_compatibility and self.legacy_profile == "evaluator":
            self.raw_score_steps = _compile_legacy_normalization_steps(
                raw_steps,
                prepend_normalize=True,
                input_is_normalized=False,
                ensure_final_normalize=True,
                eps=self.eps,
            )
            self.mf_score_steps = _compile_legacy_normalization_steps(
                mf_steps,
                prepend_normalize=False,
                input_is_normalized=True,
                ensure_final_normalize=True,
                eps=self.eps,
            )
        elif legacy_compatibility and self.legacy_profile == "detector":
            self.raw_score_steps = _compile_legacy_normalization_steps(
                raw_steps,
                prepend_normalize=True,
                input_is_normalized=False,
                ensure_final_normalize=False,
                eps=self.eps,
            )
            self.mf_score_steps = mf_steps
        else:
            self.raw_score_steps = raw_steps
            self.mf_score_steps = mf_steps

        rank = int(config.get("rank", 3))
        connectivity = int(config.get("connectivity", 1))
        detector_mask_config = postprocess_config.get("mask", {})
        threshold_mask_config = postprocess_config.get("threshold_mask", detector_mask_config)
        binary_mask_config = postprocess_config.get("binary_mask")
        if binary_mask_config is None:
            binary_mask_config = postprocess_config.get(f"{threshold_method}_mask")
        if binary_mask_config is None and threshold_method != "yen":
            # Existing configs named this shared stage ``yen_mask``. Reusing it
            # keeps Otsu isolated to the threshold algorithm itself.
            binary_mask_config = postprocess_config.get("yen_mask")
        if binary_mask_config is None:
            if self.legacy_profile == "detector" and detector_mask_config:
                binary_mask_config = detector_mask_config
            else:
                binary_mask_config = {
                    "pipeline": [
                        {
                            "type": "binary_dilation",
                            "rank": rank,
                            "connectivity": connectivity,
                            "iterations": 1,
                            "enabled": bool(config.get("yen_binary_dilation", True)),
                        }
                    ]
                }
        normalization_scope = str(
            rewrite_config.get("normalization_scope", config.get("normalization_scope", "dataset"))
        )
        super().__init__(
            normalization_scope=normalization_scope,
            threshold_method=threshold_method,
            threshold_mask_config=threshold_mask_config,
            binary_mask_config=binary_mask_config,
        )

    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        scope = _validate_normalization_scope(normalization_scope or self.normalization_scope)
        raw_finite = sanitize_scores(raw_maps)
        score_raw = apply_postprocess_pipeline(
            raw_finite,
            self.raw_score_steps,
            normalization_scope=scope,
        )
        score_mf = apply_postprocess_pipeline(
            sanitize_scores(score_raw),
            self.mf_score_steps,
            normalization_scope=scope,
        )
        return self._complete(score_raw, score_mf, scope)

    def describe(self) -> dict[str, Any]:
        scope = self.normalization_scope
        median_steps = [
            step.describe()
            for step in self.mf_score_steps
            if isinstance(step, MedianFilterPostprocessor)
        ]
        dilation_steps = [
            step.describe()
            for step in self.binary_mask_steps
            if isinstance(step, BinaryDilationPostprocessor)
        ]
        binary_mask_pipeline = [
            f"subject_{self.threshold_method}_threshold",
            *_pipeline_labels(self.binary_mask_steps, scope),
        ]
        description = {
            "postprocess_mode": self.mode,
            "normalization_scope": scope,
            "raw_score_pipeline": ["nan_to_num", *_pipeline_labels(self.raw_score_steps, scope)],
            "mf_score_pipeline": _pipeline_labels(self.mf_score_steps, scope),
            "threshold_method": self.threshold_method,
            "threshold_strategy": "per_subject_3d_volume",
            "threshold_comparator": ">",
            "binary_mask_pipeline": binary_mask_pipeline,
            "fixed_threshold_mask_pipeline": [
                "score > threshold",
                *_pipeline_labels(self.threshold_mask_steps, scope),
            ],
            "median_filter_settings": (
                {"enabled": True, **median_steps[0]} if median_steps else {"enabled": False}
            ),
            "dilation_settings": (
                {"enabled": True, **dilation_steps[0]} if dilation_steps else {"enabled": False}
            ),
            "legacy_compatibility": self.legacy_compatibility,
            "legacy_profile": self.legacy_profile if self.legacy_compatibility else None,
            "numerical_safety": {
                "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                "eps": self.eps,
                "constant_tensor": "zeros",
                "constant_volume_threshold": "constant value; empty mask with RuntimeWarning",
            },
        }
        if self.threshold_method == "yen":
            description.update(
                {
                    "yen_threshold_strategy": description["threshold_strategy"],
                    "yen_mask_pipeline": binary_mask_pipeline,
                }
            )
        return description


def build_postprocess_policy(
    config: dict[str, Any] | None,
    anomaly_config: dict[str, Any] | None = None,
    *,
    warn_legacy: bool = True,
    legacy_profile: str = "evaluator",
) -> PostprocessPolicy:
    """Build an explicit policy while retaining legacy rewrite configuration."""

    config = config or {}
    anomaly_config = anomaly_config or {}
    legacy_anomaly_threshold = str(anomaly_config.get("threshold", "")).strip().lower()
    default_threshold_method = (
        legacy_anomaly_threshold
        if legacy_anomaly_threshold in SUPPORTED_THRESHOLD_METHODS
        else "yen"
    )
    threshold_method = _validate_threshold_method(
        config.get("threshold_method", anomaly_config.get("threshold_method", default_threshold_method))
    )
    configured_mode = config.get("postprocess_mode")
    legacy_compatibility = configured_mode in (None, "")
    mode = "rewrite" if legacy_compatibility else str(configured_mode).strip().lower()
    if legacy_compatibility and warn_legacy:
        warnings.warn(
            "metrics.postprocess_mode is not set; retaining the legacy-compatible "
            "rewrite postprocessing path. Set postprocess_mode: rewrite explicitly "
            "for new configs, or postprocess_mode: original_andi for reference behavior.",
            FutureWarning,
            stacklevel=2,
        )
    if mode == "original_andi":
        original_config = config.get("original_andi", {})
        if not isinstance(original_config, dict):
            raise TypeError("metrics.original_andi must be a mapping.")
        if "eps" not in original_config:
            original_config = {
                **original_config,
                "eps": config.get("eps", anomaly_config.get("eps", 1.0e-8)),
            }
        return OriginalANDiPostprocessPolicy(
            original_config,
            threshold_method=threshold_method,
        )
    if mode == "rewrite":
        return RewritePostprocessPolicy(
            config,
            anomaly_config,
            threshold_method=threshold_method,
            legacy_compatibility=legacy_compatibility,
            legacy_profile=legacy_profile,
        )
    raise ValueError(
        f"Unknown metrics.postprocess_mode: {configured_mode!r}. "
        "Supported modes: original_andi, rewrite."
    )
