"""training、sampling 與 ANDi detection 共用的 noise 介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch


class BaseNoise(ABC):
    """所有 noise sampler 都必須實作同一個 sample 方法。"""

    name: str = "base"

    @abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """依指定 shape/device/dtype 回傳 noise tensor。"""

    def describe(self) -> dict:
        return {"type": self.name}


class NoisePlan(ABC):
    """trainer 與 detector 使用的 epoch-aware noise wrapper。"""

    @abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> torch.Tensor:
        """針對特定 training epoch 或 inference call 取樣 noise。"""

    @abstractmethod
    def describe(self) -> dict:
        """回傳方便閱讀的設定摘要。"""
