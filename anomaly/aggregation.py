"""可擴充的 anomaly map 聚合工具。

ANDi 會在兩個地方做聚合：跨 diffusion timestep，以及跨 MRI modality。
兩者共用同一個小介面，未來要新增平均方式或 pooling 策略時，只要新增
aggregator 並註冊，不需要改 detector/evaluator 的核心流程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseAggregator(ABC):
    """時間、modality 或 map 維度共用的聚合介面。"""

    name = "base"

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """沿著指定 dim 聚合 tensor。"""

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


AGGREGATORS: dict[str, type[BaseAggregator]] = {}


def register_aggregator(*names: str):
    """用一個或多個 config 名稱註冊 aggregator class。"""

    def decorator(cls: type[BaseAggregator]) -> type[BaseAggregator]:
        for name in names:
            AGGREGATORS[name.lower()] = cls
        return cls

    return decorator


@register_aggregator("mean", "arithmetic", "arithmetic_mean")
class MeanAggregator(BaseAggregator):
    name = "mean"

    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return tensor.mean(dim=dim)


@register_aggregator("geometric", "gmean", "geometric_mean")
class GeometricMeanAggregator(BaseAggregator):
    name = "geometric_mean"

    def __init__(self, eps: float = 1.0e-8):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.exp(torch.log(tensor).mean(dim=dim))

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


@register_aggregator("max", "maximum")
class MaxAggregator(BaseAggregator):
    name = "max"

    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return tensor.max(dim=dim).values


@register_aggregator("sum")
class SumAggregator(BaseAggregator):
    name = "sum"

    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return tensor.sum(dim=dim)


@register_aggregator("weighted_mean", "weighted")
class WeightedMeanAggregator(BaseAggregator):
    name = "weighted_mean"

    def __init__(self, weights: list[float]):
        if not weights:
            raise ValueError("WeightedMeanAggregator requires non-empty weights.")
        self.weights = [float(weight) for weight in weights]

    def __call__(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.tensor(self.weights, device=tensor.device, dtype=tensor.dtype)
        if tensor.shape[dim] != weights.numel():
            raise ValueError(
                f"Weighted mean expected {weights.numel()} values along dim {dim}, "
                f"got {tensor.shape[dim]}."
            )
        view_shape = [1] * tensor.ndim
        view_shape[dim] = weights.numel()
        # 將權重 reshape 成可 broadcast 的形狀，讓任意 dim 的加權平均都能共用同一段邏輯。
        weights = weights.view(*view_shape)
        weights = weights / weights.sum().clamp_min(torch.finfo(tensor.dtype).eps)
        return (tensor * weights).sum(dim=dim)

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "weights": self.weights}


def build_aggregator(config: str | dict[str, Any] | None, default: str = "mean") -> BaseAggregator:
    """從字串或 config dict 建立 aggregator。

    範例：
    - "geometric"
    - {"type": "weighted_mean", "weights": [0.2, 0.8]}
    """

    if config is None:
        config = default
    if isinstance(config, str):
        config = {"type": config}
    aggregation_type = str(config.get("type", default)).lower()
    if aggregation_type not in AGGREGATORS:
        raise ValueError(f"Unknown aggregator type: {aggregation_type}")

    kwargs = {key: value for key, value in config.items() if key != "type"}
    return AGGREGATORS[aggregation_type](**kwargs)
