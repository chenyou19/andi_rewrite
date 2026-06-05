from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseNoise


class GaussianNoise(BaseNoise):
    """獨立標準 Gaussian noise。"""

    name = "gaussian"

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(tuple(shape), device=device, dtype=dtype)
