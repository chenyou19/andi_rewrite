from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseNoise


class HybridNoise(BaseNoise):
    """多個可插拔 noise sampler 的加權混合。"""

    name = "hybrid"

    def __init__(
        self,
        components: list[tuple[BaseNoise, float]],
        normalize: bool = True,
    ):
        if not components:
            raise ValueError("HybridNoise requires at least one component.")
        self.components = [(sampler, float(weight)) for sampler, weight in components]
        self.normalize = bool(normalize)

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mixed = None
        for sampler, weight in self.components:
            part = sampler.sample(shape, device=device, dtype=dtype) * weight
            mixed = part if mixed is None else mixed + part

        if self.normalize:
            mixed = mixed / mixed.std().clamp_min(torch.finfo(dtype).eps)
        return mixed

    def describe(self) -> dict:
        return {
            "type": self.name,
            "normalize": self.normalize,
            "components": [
                {"weight": weight, "sampler": sampler.describe()}
                for sampler, weight in self.components
            ],
        }
