from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseNoise


class SpectrumNoise(BaseNoise):
    """以頻譜塑形的 Gaussian noise。

    實作上先產生 white noise，再調整 Fourier coefficients，最後轉回 image space。
    對外介面維持和 GaussianNoise、PyramidNoise 完全相同。
    """

    name = "spectrum"

    def __init__(
        self,
        exponent: float = 1.0,
        low_frequency_bias: bool = True,
        normalize: bool = True,
    ):
        self.exponent = float(exponent)
        self.low_frequency_bias = bool(low_frequency_bias)
        self.normalize = bool(normalize)

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(shape) != 4:
            raise ValueError("SpectrumNoise expects shape [batch, channels, height, width].")

        batch, channels, height, width = map(int, shape)
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        white = torch.randn(batch, channels, height, width, device=device, dtype=work_dtype)

        freq_y = torch.fft.fftfreq(height, device=device, dtype=work_dtype).view(height, 1)
        freq_x = torch.fft.rfftfreq(width, device=device, dtype=work_dtype).view(1, -1)
        radius = torch.sqrt(freq_y.square() + freq_x.square()).clamp_min(1.0 / max(height, width))
        if self.low_frequency_bias:
            weights = radius.pow(-self.exponent)
        else:
            weights = radius.pow(self.exponent)
        weights = weights / weights.mean().clamp_min(torch.finfo(work_dtype).eps)

        spectrum = torch.fft.rfft2(white)
        shaped = torch.fft.irfft2(spectrum * weights, s=(height, width))
        if self.normalize:
            reduce_dims = tuple(range(1, shaped.ndim))
            std = shaped.std(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(work_dtype).eps)
            shaped = (shaped - shaped.mean(dim=reduce_dims, keepdim=True)) / std
        return shaped.to(dtype=dtype)

    def describe(self) -> dict:
        return {
            "type": self.name,
            "exponent": self.exponent,
            "low_frequency_bias": self.low_frequency_bias,
            "normalize": self.normalize,
        }
