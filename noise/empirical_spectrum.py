from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .base import BaseNoise


_GENERATION_METHOD_ALIASES: Mapping[str, str] = {
    "fixed_magnitude": "fixed_magnitude",
    "fixed": "fixed_magnitude",
    "phase_randomized": "fixed_magnitude",
    "filtered_gaussian": "filtered_gaussian",
    "gaussian_filter": "filtered_gaussian",
    "legacy_filter": "filtered_gaussian",
}


class EmpiricalSpectrumNoise(BaseNoise):
    """Generate noise from empirical MRI amplitude or power spectra.

    ``fixed_magnitude`` is the historical implementation: every draw uses the
    empirical FFT magnitude and random phase. ``filtered_gaussian`` preserves
    both the magnitude and phase randomness of a white-Gaussian FFT and applies
    the empirical spectrum as an amplitude filter.
    """

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
        generation_method: str = "fixed_magnitude",
        spectrum_power_key: str = "mean_power",
        radial_power_key: str = "radial_power",
    ):
        self.stats_path = str(stats_path)
        self.mode = str(mode).lower()
        self.generation_method = self._canonical_generation_method(generation_method)
        self.spectrum_key = str(spectrum_key)
        self.radial_key = str(radial_key)
        self.spectrum_power_key = str(spectrum_power_key)
        self.radial_power_key = str(radial_power_key)
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

        self.channels: int | None = None
        self.height: int | None = None
        self.width: int | None = None
        self.target_amp: torch.Tensor | None = None
        self.filter_amp_rfft: torch.Tensor | None = None
        self.loaded_statistic_type: str
        self.loaded_statistic_key: str
        self.used_statistic_fallback = False

        with np.load(path, allow_pickle=False) as stats:
            self.channels = self._read_optional_scalar(stats, "channels")
            self.height = self._read_optional_scalar(stats, "height")
            self.width = self._read_optional_scalar(stats, "width")
            if self.generation_method == "fixed_magnitude":
                self._load_fixed_magnitude_target(stats, path)
            else:
                self._load_filtered_gaussian_filter(stats, path)

    @staticmethod
    def _canonical_generation_method(value: str) -> str:
        normalized = str(value).strip().lower()
        try:
            return _GENERATION_METHOD_ALIASES[normalized]
        except KeyError as exc:
            canonical = "fixed_magnitude, filtered_gaussian"
            aliases = ", ".join(_GENERATION_METHOD_ALIASES)
            raise ValueError(
                "EmpiricalSpectrumNoise generation_method must resolve to one of: "
                f"{canonical}. Accepted values: {aliases}."
            ) from exc

    @staticmethod
    def _read_optional_scalar(stats: np.lib.npyio.NpzFile, key: str) -> int | None:
        if key not in stats:
            return None
        value = np.asarray(stats[key])
        if value.size != 1:
            raise ValueError(f"Empirical spectrum metadata '{key}' must be scalar, got {value.shape}.")
        return int(value.item())

    def _load_fixed_magnitude_target(self, stats: np.lib.npyio.NpzFile, path: Path) -> None:
        if self.mode in {"2d", "full2d"}:
            target = self._load_full_statistic(stats, self.spectrum_key, path, "spectrum")
            target = np.fft.ifftshift(target, axes=(-2, -1)).copy()
            self.channels, self.height, self.width = map(int, target.shape)
            loaded_key = self.spectrum_key
        else:
            profile = self._load_radial_statistic(stats, self.radial_key, path, "spectrum")
            self._require_radial_spatial_shape()
            self.channels = int(profile.shape[0])
            target = self._expand_radial_profile(profile, self.height, self.width)
            loaded_key = self.radial_key

        # Preserve the historical ordering exactly for backward compatibility:
        # average amplitudes first, then divide by the spatial mean amplitude.
        if not self.per_channel:
            target = target.mean(axis=0, keepdims=True)
        target = self._normalize_target(target)
        self.target_amp = torch.from_numpy(target.astype(np.float32, copy=False))
        self.loaded_statistic_type = "amplitude"
        self.loaded_statistic_key = loaded_key

    def _load_filtered_gaussian_filter(self, stats: np.lib.npyio.NpzFile, path: Path) -> None:
        is_full = self.mode in {"2d", "full2d"}
        power_key = self.spectrum_power_key if is_full else self.radial_power_key
        amplitude_key = self.spectrum_key if is_full else self.radial_key

        if power_key in stats:
            source = np.asarray(stats[power_key], dtype=np.float32)
            self.loaded_statistic_type = "power"
            self.loaded_statistic_key = power_key
        elif amplitude_key in stats:
            amplitude = np.asarray(stats[amplitude_key], dtype=np.float32)
            source = np.square(amplitude, dtype=np.float32)
            self.loaded_statistic_type = "amplitude"
            self.loaded_statistic_key = amplitude_key
            self.used_statistic_fallback = True
            warnings.warn(
                f"EmpiricalSpectrumNoise filtered_gaussian could not find power key "
                f"'{power_key}' in {path}; falling back to amplitude key "
                f"'{amplitude_key}' and using power = amplitude ** 2.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise KeyError(
                f"EmpiricalSpectrumNoise filtered_gaussian requires power key '{power_key}' "
                f"or amplitude fallback key '{amplitude_key}' in {path}."
            )

        if not np.all(np.isfinite(source)):
            raise ValueError(f"Empirical spectrum '{self.loaded_statistic_key}' contains NaN or Inf.")

        if is_full:
            power = self._validate_full_array(source, self.loaded_statistic_key)
            power = np.fft.ifftshift(power, axes=(-2, -1)).copy()
            self.channels, self.height, self.width = map(int, power.shape)
            power = self._make_power_hermitian(power)
        else:
            profile = self._validate_radial_array(source, self.loaded_statistic_key)
            self._require_radial_spatial_shape()
            self.channels = int(profile.shape[0])
            power = self._expand_radial_profile(profile, self.height, self.width)

        # A shared filter is formed by averaging power, never amplitude.
        if not self.per_channel:
            power = power.mean(axis=0, keepdims=True)
        filter_amplitude = np.sqrt(np.maximum(power, self.eps)).astype(np.float32, copy=False)
        filter_amplitude = self._normalize_filter(filter_amplitude)
        # The unshifted one-sided slice matches torch.fft.rfft2 layout.
        filter_rfft = filter_amplitude[..., : self.width // 2 + 1].copy()
        self.filter_amp_rfft = torch.from_numpy(filter_rfft)

    def _load_full_statistic(
        self,
        stats: np.lib.npyio.NpzFile,
        key: str,
        path: Path,
        label: str,
    ) -> np.ndarray:
        if key not in stats:
            raise KeyError(f"Empirical {label} key '{key}' not found in {path}.")
        return self._validate_full_array(np.asarray(stats[key], dtype=np.float32), key)

    def _load_radial_statistic(
        self,
        stats: np.lib.npyio.NpzFile,
        key: str,
        path: Path,
        label: str,
    ) -> np.ndarray:
        if key not in stats:
            raise KeyError(f"Empirical radial {label} key '{key}' not found in {path}.")
        return self._validate_radial_array(np.asarray(stats[key], dtype=np.float32), key)

    @staticmethod
    def _validate_full_array(value: np.ndarray, key: str) -> np.ndarray:
        if value.ndim != 3:
            raise ValueError(f"Empirical spectrum '{key}' must have shape [C, H, W], got {value.shape}.")
        return value

    @staticmethod
    def _validate_radial_array(value: np.ndarray, key: str) -> np.ndarray:
        if value.ndim != 2:
            raise ValueError(f"Empirical radial spectrum '{key}' must have shape [C, R], got {value.shape}.")
        return value

    def _require_radial_spatial_shape(self) -> None:
        if self.height is None or self.width is None:
            raise ValueError("Radial empirical spectrum stats must include 'height' and 'width'.")

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

    @staticmethod
    def _make_power_hermitian(power: np.ndarray) -> np.ndarray:
        """Symmetrize unshifted power at k and -k before one-sided cropping."""

        height, width = power.shape[-2:]
        negative_y = (-np.arange(height)) % height
        negative_x = (-np.arange(width)) % width
        conjugate_partner = power[:, negative_y, :][:, :, negative_x]
        return ((power + conjugate_partner) * 0.5).astype(np.float32, copy=False)

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        target = np.maximum(target.astype(np.float32, copy=False), self.eps)
        mean = target.mean(axis=(-2, -1), keepdims=True)
        return target / np.maximum(mean, self.eps)

    def _normalize_filter(self, filter_amplitude: np.ndarray) -> np.ndarray:
        rms = np.sqrt(
            np.mean(np.square(filter_amplitude, dtype=np.float32), axis=(-2, -1), keepdims=True)
        )
        return filter_amplitude / np.maximum(rms, self.eps)

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

    def _normalize_spatial(self, shaped: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return shaped
        mean = shaped.mean(dim=(-2, -1), keepdim=True)
        std = shaped.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (shaped - mean) / (std + self.eps)

    def _sample_fixed_magnitude(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        work_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Keep this sequence identical to the pre-generation_method sampler.
        white = torch.randn(tuple(shape), device=device, dtype=work_dtype)
        spectrum = torch.fft.fft2(white, dim=(-2, -1))
        white_amp = torch.abs(spectrum)
        phase = spectrum / (white_amp + self.eps)

        assert self.target_amp is not None
        target = self.target_amp.to(device=device, dtype=work_dtype).unsqueeze(0)
        shaped_amp = (white_amp + self.eps).pow(1.0 - self.strength) * (target + self.eps).pow(self.strength)
        shaped = torch.fft.ifft2(phase * shaped_amp, dim=(-2, -1)).real
        return self._normalize_spatial(shaped)

    def _sample_filtered_gaussian(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        work_dtype: torch.dtype,
        height: int,
        width: int,
    ) -> torch.Tensor:
        white = torch.randn(tuple(shape), device=device, dtype=work_dtype)
        white_fft = torch.fft.rfft2(white, dim=(-2, -1))
        assert self.filter_amp_rfft is not None
        filter_amplitude = self.filter_amp_rfft.to(device=device, dtype=work_dtype)
        effective_filter = filter_amplitude.clamp_min(self.eps).pow(self.strength)
        filtered_fft = white_fft * effective_filter.unsqueeze(0)
        shaped = torch.fft.irfft2(filtered_fft, s=(height, width), dim=(-2, -1))
        return self._normalize_spatial(shaped)

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        _, _, height, width = self._validate_shape(shape)
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        if self.generation_method == "fixed_magnitude":
            shaped = self._sample_fixed_magnitude(shape, device, work_dtype)
        else:
            shaped = self._sample_filtered_gaussian(shape, device, work_dtype, height, width)
        return shaped.to(device=device, dtype=dtype)

    def describe(self) -> dict:
        return {
            "type": self.name,
            "stats_path": self.stats_path,
            "generation_method": self.generation_method,
            "mode": self.mode,
            "spectrum_key": self.spectrum_key,
            "radial_key": self.radial_key,
            "spectrum_power_key": self.spectrum_power_key,
            "radial_power_key": self.radial_power_key,
            "loaded_statistic_type": self.loaded_statistic_type,
            "loaded_statistic_key": self.loaded_statistic_key,
            "used_statistic_fallback": self.used_statistic_fallback,
            "filter_normalization": (
                "rms" if self.generation_method == "filtered_gaussian" else "not_applicable"
            ),
            "target_normalization": (
                "mean_amplitude" if self.generation_method == "fixed_magnitude" else "not_applicable"
            ),
            "per_channel": self.per_channel,
            "strength": self.strength,
            "normalize": self.normalize,
            "eps": self.eps,
        }
