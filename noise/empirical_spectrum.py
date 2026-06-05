from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseNoise


class EmpiricalSpectrumNoise(BaseNoise):
    """Shape Gaussian noise with an empirical MRI amplitude spectrum."""

    name = "empirical_spectrum"

    def __init__(
        self,
        stats_path: str | Path,
        mode: str = "radial",
        spectrum_key: str = "mean_amplitude",
        radial_key: str = "radial_amplitude",
        per_channel: bool = True,
        strength: float = 1.0,
        normalize: bool = True,
        eps: float = 1.0e-8,
    ):
        self.stats_path = str(stats_path)
        self.mode = str(mode).lower()
        self.spectrum_key = str(spectrum_key)
        self.radial_key = str(radial_key)
        self.per_channel = bool(per_channel)
        self.strength = float(strength)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        if self.mode not in {"2d", "full2d", "radial"}:
            raise ValueError("EmpiricalSpectrumNoise mode must be one of: '2d', 'full2d', 'radial'.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("EmpiricalSpectrumNoise strength must be in [0, 1].")
        if self.eps <= 0:
            raise ValueError("EmpiricalSpectrumNoise eps must be positive.")

        path = Path(self.stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Empirical spectrum stats_path does not exist: {path}")

        with np.load(path, allow_pickle=False) as stats:
            self.channels = int(np.asarray(stats["channels"]).item()) if "channels" in stats else None
            self.height = int(np.asarray(stats["height"]).item()) if "height" in stats else None
            self.width = int(np.asarray(stats["width"]).item()) if "width" in stats else None
            if self.mode in {"2d", "full2d"}:
                if self.spectrum_key not in stats:
                    raise KeyError(f"Empirical spectrum key '{self.spectrum_key}' not found in {path}.")
                target = np.asarray(stats[self.spectrum_key], dtype=np.float32)
                if target.ndim != 3:
                    raise ValueError(
                        f"Empirical spectrum '{self.spectrum_key}' must have shape [C, H, W], got {target.shape}."
                    )
                self.channels, self.height, self.width = map(int, target.shape)
                target = np.fft.ifftshift(target, axes=(-2, -1)).copy()
            else:
                if self.radial_key not in stats:
                    raise KeyError(f"Empirical radial key '{self.radial_key}' not found in {path}.")
                profile = np.asarray(stats[self.radial_key], dtype=np.float32)
                if profile.ndim != 2:
                    raise ValueError(
                        f"Empirical radial spectrum '{self.radial_key}' must have shape [C, R], got {profile.shape}."
                    )
                if self.height is None or self.width is None:
                    raise ValueError("Radial empirical spectrum stats must include 'height' and 'width'.")
                self.channels = int(profile.shape[0])
                target = self._expand_radial_profile(profile, self.height, self.width)

        if not self.per_channel:
            target = target.mean(axis=0, keepdims=True)
        target = self._normalize_target(target)
        self.target_amp = torch.from_numpy(target.astype(np.float32, copy=False))

    def _expand_radial_profile(self, profile: np.ndarray, height: int, width: int) -> np.ndarray:
        radial_bins = int(profile.shape[1])
        fy = np.fft.fftfreq(height).astype(np.float32)
        fx = np.fft.fftfreq(width).astype(np.float32)
        radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        max_radius = float(radius.max())
        if max_radius <= 0:
            radial_position = np.zeros((height, width), dtype=np.float32)
        else:
            radial_position = radius / max_radius * float(radial_bins - 1)

        x = np.arange(radial_bins, dtype=np.float32)
        expanded = np.empty((profile.shape[0], height, width), dtype=np.float32)
        for channel in range(profile.shape[0]):
            expanded[channel] = np.interp(radial_position.ravel(), x, profile[channel]).reshape(height, width)
        return expanded

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        target = np.maximum(target.astype(np.float32, copy=False), self.eps)
        mean = target.mean(axis=(-2, -1), keepdims=True)
        return target / np.maximum(mean, self.eps)

    def _validate_shape(self, shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(shape) != 4:
            raise ValueError("EmpiricalSpectrumNoise expects shape [batch, channels, height, width].")
        batch, channels, height, width = map(int, shape)
        if (height, width) != (self.height, self.width):
            raise ValueError(
                "EmpiricalSpectrumNoise shape mismatch: "
                f"requested H/W={(height, width)}, stats H/W={(self.height, self.width)}."
            )
        if self.per_channel and channels != self.channels:
            raise ValueError(
                "EmpiricalSpectrumNoise channel mismatch: "
                f"requested C={channels}, stats C={self.channels}. "
                "Set per_channel=false to share the average spectrum across channels."
            )
        return batch, channels, height, width

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, channels, height, width = self._validate_shape(shape)
        del batch, channels, height, width

        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        white = torch.randn(tuple(shape), device=device, dtype=work_dtype)
        spectrum = torch.fft.fft2(white, dim=(-2, -1))
        white_amp = torch.abs(spectrum)
        phase = spectrum / (white_amp + self.eps)

        target = self.target_amp.to(device=device, dtype=work_dtype).unsqueeze(0)
        shaped_amp = (white_amp + self.eps).pow(1.0 - self.strength) * (target + self.eps).pow(self.strength)
        shaped = torch.fft.ifft2(phase * shaped_amp, dim=(-2, -1)).real

        if self.normalize:
            mean = shaped.mean(dim=(-2, -1), keepdim=True)
            std = shaped.std(dim=(-2, -1), keepdim=True, unbiased=False)
            shaped = (shaped - mean) / (std + self.eps)

        return shaped.to(device=device, dtype=dtype)

    def describe(self) -> dict:
        return {
            "type": self.name,
            "stats_path": self.stats_path,
            "mode": self.mode,
            "spectrum_key": self.spectrum_key,
            "radial_key": self.radial_key,
            "per_channel": self.per_channel,
            "strength": self.strength,
            "normalize": self.normalize,
            "eps": self.eps,
        }
