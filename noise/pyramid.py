from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F

from .base import BaseNoise


class PyramidNoise(BaseNoise):
    """multi-scale Gaussian pyramid noise，相容原版 ANDi。"""

    name = "pyramid"

    def __init__(self, discount: float = 0.8, levels: int = 10, normalize: bool = True):
        self.discount = float(discount)
        self.levels = int(levels)
        self.normalize = bool(normalize)

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(shape) != 4:
            raise ValueError("PyramidNoise expects shape [batch, channels, height, width].")

        batch, channels, height, width = map(int, shape)
        noise = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
        current_h, current_w = height, width

        for level in range(self.levels):
            ratio = random.random() * 2.0 + 2.0
            current_h = max(1, int(current_h / (ratio**level)))
            current_w = max(1, int(current_w / (ratio**level)))
            coarse = torch.randn(
                batch,
                channels,
                current_h,
                current_w,
                device=device,
                dtype=dtype,
            )
            noise = noise + F.interpolate(
                coarse,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ) * (self.discount**level)
            if current_h == 1 or current_w == 1:
                break

        if self.normalize:
            noise = noise / noise.std().clamp_min(torch.finfo(dtype).eps)
        return noise

    def describe(self) -> dict:
        return {
            "type": self.name,
            "discount": self.discount,
            "levels": self.levels,
            "normalize": self.normalize,
        }
