"""Streaming statistics and historical helpers for noise diagnostics.

The formal method comparison in :mod:`andi_rewrite.scripts.compare_noise_statistics`
calls production :class:`EmpiricalSpectrumNoise` samplers directly. The explicit-
white generator helpers remain here for focused numerical regression tests and
historical reproduction; they are not substitutes for the production comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _validate_positive_shape(height: int, width: int) -> tuple[int, int]:
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}.")
    return height, width


def _profile_2d(profile: Any, *, name: str) -> np.ndarray:
    array = np.asarray(profile, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [R] or [C,R], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _read_scalar_int(stats: Mapping[str, Any], key: str) -> int | None:
    if key not in stats:
        return None
    value = np.asarray(stats[key])
    if value.size != 1:
        raise ValueError(f"NPZ key '{key}' must be scalar, got shape {value.shape}.")
    return int(value.reshape(-1)[0])


def _validate_metadata_shape(
    *,
    stored_channels: int | None,
    stored_height: int | None,
    stored_width: int | None,
    channels: int,
    height: int,
    width: int,
    path: Path,
) -> None:
    expected = (int(channels), int(height), int(width))
    stored = (stored_channels, stored_height, stored_width)
    for label, actual, wanted in zip(("channels", "height", "width"), stored, expected):
        if actual is not None and int(actual) != wanted:
            raise ValueError(
                f"{path} metadata mismatch for {label}: file={actual}, requested={wanted}."
            )


def expand_normalized_radial_profile(
    profile: Any,
    height: int,
    width: int,
) -> np.ndarray:
    """Expand ``[C,R]`` profiles on a normalized-radius, unshifted FFT grid.

    "Normalized" refers only to the radial coordinate: the largest corner
    radius maps to profile index ``R-1``.  No amplitude normalization is done.
    This matches the radial expansion in the rewritten empirical sampler.
    """

    height, width = _validate_positive_shape(height, width)
    # Keep these intermediates in float32 to match
    # EmpiricalSpectrumNoise._expand_radial_profile bit for bit.
    radial = _profile_2d(profile, name="radial profile").astype(np.float32)
    fy = np.fft.fftfreq(height).astype(np.float32)
    fx = np.fft.fftfreq(width).astype(np.float32)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius_max = float(radius.max())
    if radius_max <= 0.0 or radial.shape[1] == 1:
        positions = np.zeros((height, width), dtype=np.float32)
    else:
        positions = radius / radius_max * float(radial.shape[1] - 1)
    source = np.arange(radial.shape[1], dtype=np.float32)
    expanded = np.empty((radial.shape[0], height, width), dtype=np.float32)
    for channel in range(radial.shape[0]):
        expanded[channel] = np.interp(
            positions.reshape(-1), source, radial[channel]
        ).reshape(height, width)
    return expanded


def build_pixel_radius_weight_map(
    radial_power: Any,
    height: int,
    width: int,
    *,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Reproduce the legacy ``sqrt(interpolate(radial power))`` filter."""

    height, width = _validate_positive_shape(height, width)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    power = _profile_2d(radial_power, name="legacy radial power")
    power = np.maximum(power, float(eps))
    fy = np.fft.fftfreq(height).astype(np.float64) * height
    fx = np.fft.fftfreq(width).astype(np.float64) * width
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius = np.clip(radius, 0.0, float(power.shape[1] - 1))
    lower = np.floor(radius).astype(np.int64)
    upper = np.minimum(lower + 1, power.shape[1] - 1)
    fraction = radius - lower
    interpolated = (
        power[:, lower] * (1.0 - fraction[None, ...])
        + power[:, upper] * fraction[None, ...]
    )
    return np.sqrt(np.maximum(interpolated, float(eps))).astype(np.float32)


def load_new_target_amplitude(
    stats_path: str | Path,
    *,
    channels: int,
    height: int,
    width: int,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Load the exact radial target-amplitude convention of the new sampler."""

    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"Spectrum stats file does not exist: {path}")
    with np.load(path, allow_pickle=False) as stats:
        if "radial_amplitude" not in stats:
            raise KeyError(f"{path} does not contain 'radial_amplitude'.")
        profile = _profile_2d(stats["radial_amplitude"], name="radial_amplitude")
        _validate_metadata_shape(
            stored_channels=_read_scalar_int(stats, "channels"),
            stored_height=_read_scalar_int(stats, "height"),
            stored_width=_read_scalar_int(stats, "width"),
            channels=channels,
            height=height,
            width=width,
            path=path,
        )
    if profile.shape[0] != int(channels):
        raise ValueError(
            f"{path} radial_amplitude channels={profile.shape[0]}, requested={channels}."
        )
    target = expand_normalized_radial_profile(profile, height, width)
    target = np.maximum(target.astype(np.float32, copy=False), float(eps))
    channel_mean = target.mean(axis=(-2, -1), keepdims=True, dtype=np.float32)
    target = target / np.maximum(channel_mean, float(eps))
    ensure_finite(target, name="new target amplitude")
    return target.astype(np.float32, copy=False)


def _legacy_image_size(stats: Mapping[str, Any], path: Path) -> tuple[int, int]:
    if "image_size" in stats:
        values = np.asarray(stats["image_size"]).astype(np.int64).reshape(-1)
        if values.size == 1:
            return int(values[0]), int(values[0])
        if values.size >= 2:
            return int(values[0]), int(values[1])
    h = _read_scalar_int(stats, "image_height")
    w = _read_scalar_int(stats, "image_width")
    if h is not None and w is not None:
        return h, w
    raise KeyError(f"{path} lacks legacy image_size or image_height/image_width metadata.")


def load_legacy_filter(
    stats_path: str | Path,
    *,
    channels: int,
    height: int,
    width: int,
    eps: float = 1.0e-8,
    current_schema_fallback: bool = False,
) -> np.ndarray:
    """Load a legacy amplitude filter, including the requested new-NPZ adapter.

    Legacy files use pixel-radius PSD bins.  When ``current_schema_fallback``
    is true, ``radial_power`` is instead expanded with the rewritten producer's
    normalized-radius convention before taking its square root.
    """

    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy spectrum stats file does not exist: {path}")
    with np.load(path, allow_pickle=False) as stats:
        if current_schema_fallback:
            if "radial_power" not in stats:
                raise KeyError(
                    f"{path} does not contain 'radial_power' required by the current-schema fallback."
                )
            profile = _profile_2d(stats["radial_power"], name="radial_power")
            _validate_metadata_shape(
                stored_channels=_read_scalar_int(stats, "channels"),
                stored_height=_read_scalar_int(stats, "height"),
                stored_width=_read_scalar_int(stats, "width"),
                channels=channels,
                height=height,
                width=width,
                path=path,
            )
            if profile.shape[0] != int(channels):
                raise ValueError(
                    f"{path} radial_power channels={profile.shape[0]}, requested={channels}."
                )
            expanded_power = expand_normalized_radial_profile(profile, height, width)
            result = np.sqrt(np.maximum(expanded_power, float(eps))).astype(np.float32)
        else:
            key = next(
                (candidate for candidate in ("radial_psd", "radial_power_smooth", "radial_power_mean") if candidate in stats),
                None,
            )
            if key is None:
                raise KeyError(
                    f"{path} lacks legacy radial_psd/radial_power_smooth/radial_power_mean."
                )
            profile = _profile_2d(stats[key], name=key)
            stored_height, stored_width = _legacy_image_size(stats, path)
            stored_channels = _read_scalar_int(stats, "channels")
            _validate_metadata_shape(
                stored_channels=stored_channels,
                stored_height=stored_height,
                stored_width=stored_width,
                channels=channels,
                height=height,
                width=width,
                path=path,
            )
            per_channel = bool(
                np.asarray(stats["per_channel"]).reshape(-1)[0]
            ) if "per_channel" in stats else profile.shape[0] > 1
            if profile.shape[0] == 1:
                profile = np.repeat(profile, int(channels), axis=0)
            elif profile.shape[0] != int(channels):
                raise ValueError(
                    f"{path} {key} channels={profile.shape[0]}, requested={channels}."
                )
            if not per_channel and profile.shape[0] != int(channels):
                profile = np.repeat(profile[:1], int(channels), axis=0)
            result = build_pixel_radius_weight_map(
                profile, height, width, eps=eps
            )
    ensure_finite(result, name="legacy filter")
    return result.astype(np.float32, copy=False)


def _torch_map(value: Any, white: torch.Tensor, *, name: str) -> torch.Tensor:
    target = torch.as_tensor(value, device=white.device, dtype=white.dtype)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError(f"{name} must have shape [H,W] or [C,H,W], got {tuple(target.shape)}.")
    if target.shape[0] == 1 and white.shape[1] > 1:
        target = target.expand(white.shape[1], -1, -1)
    expected = (white.shape[1], white.shape[2], white.shape[3])
    if tuple(target.shape) != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(target.shape)}.")
    return target.unsqueeze(0)


def normalize_per_sample_channel(
    values: ArrayLike,
    *,
    eps: float = 1.0e-8,
    denominator: str = "clamp",
) -> ArrayLike:
    """Apply per-sample/per-channel zero mean and population-unit std."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if denominator not in {"clamp", "add"}:
        raise ValueError("denominator must be 'clamp' or 'add'.")
    if isinstance(values, torch.Tensor):
        mean = values.mean(dim=(-2, -1), keepdim=True)
        std = values.std(dim=(-2, -1), keepdim=True, unbiased=False)
        divisor = std.clamp_min(eps) if denominator == "clamp" else std + eps
        return (values - mean) / divisor
    array = np.asarray(values)
    mean = array.mean(axis=(-2, -1), keepdims=True)
    std = array.std(axis=(-2, -1), keepdims=True, ddof=0)
    divisor = np.maximum(std, eps) if denominator == "clamp" else std + eps
    return (array - mean) / divisor


def legacy_filtered_gaussian_from_white(
    white: torch.Tensor,
    amplitude_filter: Any,
    *,
    eps: float = 1.0e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """Generate legacy filtered Gaussian noise from an explicit white draw."""

    if white.ndim != 4 or not torch.is_floating_point(white):
        raise ValueError("white must be a floating BCHW tensor.")
    original_dtype = white.dtype
    work = white.float() if white.dtype in (torch.float16, torch.bfloat16) else white
    target = _torch_map(amplitude_filter, work, name="amplitude_filter")
    shaped = torch.fft.ifft2(
        torch.fft.fft2(work, dim=(-2, -1)) * target,
        dim=(-2, -1),
    ).real
    shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        shaped = normalize_per_sample_channel(
            shaped, eps=eps, denominator="clamp"
        )
    shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
    return shaped.to(dtype=original_dtype)


def fixed_magnitude_from_white(
    white: torch.Tensor,
    target_amplitude: Any,
    *,
    eps: float = 1.0e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """Generate random-phase, fixed-target-magnitude noise from ``white``."""

    if white.ndim != 4 or not torch.is_floating_point(white):
        raise ValueError("white must be a floating BCHW tensor.")
    original_dtype = white.dtype
    work = white.float() if white.dtype in (torch.float16, torch.bfloat16) else white
    target = _torch_map(target_amplitude, work, name="target_amplitude")
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    phase = spectrum / (torch.abs(spectrum) + float(eps))
    shaped = torch.fft.ifft2(phase * target, dim=(-2, -1)).real
    shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        shaped = normalize_per_sample_channel(
            shaped, eps=eps, denominator="clamp"
        )
    shaped = torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)
    return shaped.to(dtype=original_dtype)


def fft_magnitude(values: ArrayLike) -> ArrayLike:
    """Return unshifted two-dimensional FFT magnitudes."""

    if isinstance(values, torch.Tensor):
        return torch.abs(torch.fft.fft2(values, dim=(-2, -1)))
    return np.abs(np.fft.fft2(np.asarray(values), axes=(-2, -1)))


def coefficient_of_variation(
    std: ArrayLike,
    mean: ArrayLike,
    *,
    eps: float = 1.0e-8,
) -> ArrayLike:
    """Compute ``population standard deviation / mean`` (never the inverse)."""

    if isinstance(std, torch.Tensor) or isinstance(mean, torch.Tensor):
        std_tensor = torch.as_tensor(std)
        mean_tensor = torch.as_tensor(mean, device=std_tensor.device, dtype=std_tensor.dtype)
        return std_tensor / (mean_tensor + float(eps))
    return np.asarray(std) / (np.asarray(mean) + float(eps))


def valid_frequency_mask(
    mean_magnitude: ArrayLike,
    *,
    relative_threshold: float = 1.0e-6,
) -> ArrayLike:
    """Mask bins above each channel's relative maximum magnitude threshold."""

    if relative_threshold < 0:
        raise ValueError("relative_threshold must be non-negative.")
    if isinstance(mean_magnitude, torch.Tensor):
        maximum = mean_magnitude.amax(dim=(-2, -1), keepdim=True)
        return mean_magnitude > maximum * float(relative_threshold)
    array = np.asarray(mean_magnitude)
    maximum = np.max(array, axis=(-2, -1), keepdims=True)
    return array > maximum * float(relative_threshold)


def ensure_finite(values: Any, *, name: str = "values") -> Any:
    """Raise a descriptive error if an array/tensor contains NaN or Inf."""

    if isinstance(values, torch.Tensor):
        finite = bool(torch.isfinite(values).all().item())
    else:
        finite = bool(np.isfinite(np.asarray(values)).all())
    if not finite:
        raise ValueError(f"{name} contains NaN or Inf.")
    return values


def distribution_summary(values: ArrayLike) -> dict[str, float]:
    """Return the common population distribution summary used by the report."""

    array = (
        values.detach().cpu().to(torch.float64).numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values, dtype=np.float64)
    ).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    ensure_finite(array, name="summary values")
    percentiles = np.percentile(array, [1, 5, 25, 75, 90, 95, 99])
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def summarize_cv(cv: ArrayLike, *, mask: ArrayLike | None = None) -> dict[str, float]:
    """Summarize CV values and the requested near-zero percentages."""

    array = (
        cv.detach().cpu().to(torch.float64).numpy()
        if isinstance(cv, torch.Tensor)
        else np.asarray(cv, dtype=np.float64)
    )
    if mask is not None:
        mask_array = (
            mask.detach().cpu().numpy()
            if isinstance(mask, torch.Tensor)
            else np.asarray(mask)
        ).astype(bool, copy=False)
        if mask_array.shape != array.shape:
            raise ValueError(f"CV mask shape {mask_array.shape} != CV shape {array.shape}.")
        array = array[mask_array]
    else:
        array = array.reshape(-1)
    summary = distribution_summary(array)
    for threshold in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
        summary[f"percentage_cv_lt_{threshold:.0e}"] = float(
            np.mean(array < threshold) * 100.0
        )
    return summary


class RunningMeanVariance:
    """Float64 Chan/Welford accumulator over the leading sample dimension."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values: ArrayLike) -> None:
        array = (
            values.detach().cpu().to(torch.float64).numpy()
            if isinstance(values, torch.Tensor)
            else np.asarray(values, dtype=np.float64)
        )
        if array.ndim < 1 or array.shape[0] == 0:
            raise ValueError("RunningMeanVariance.update expects a non-empty sample axis.")
        ensure_finite(array, name="running-stat batch")
        batch_count = int(array.shape[0])
        batch_mean = np.mean(array, axis=0, dtype=np.float64)
        centered = array - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        if batch_mean.shape != self.mean.shape:
            raise ValueError(
                f"Running statistic shape changed from {self.mean.shape} to {batch_mean.shape}."
            )
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta * delta * (
            self.count * batch_count / total
        )
        self.count = total

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("RunningMeanVariance has no samples.")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return {
            "count": self.count,
            "mean": self.mean.copy(),
            "std": np.sqrt(variance),
        }


class RawMomentAccumulator:
    """Float64 streaming raw moments 1--4 for fixed-location marginals."""

    def __init__(self) -> None:
        self.count = 0
        self.sums: list[np.ndarray] | None = None

    def update(self, values: ArrayLike) -> None:
        array = (
            values.detach().cpu().to(torch.float64).numpy()
            if isinstance(values, torch.Tensor)
            else np.asarray(values, dtype=np.float64)
        )
        if array.ndim < 1 or array.shape[0] == 0:
            raise ValueError("RawMomentAccumulator.update expects a non-empty sample axis.")
        ensure_finite(array, name="moment batch")
        powers = []
        current = array
        for order in range(1, 5):
            if order > 1:
                current = current * array
            powers.append(np.sum(current, axis=0, dtype=np.float64))
        if self.sums is None:
            self.sums = powers
        else:
            if powers[0].shape != self.sums[0].shape:
                raise ValueError(
                    f"Moment shape changed from {self.sums[0].shape} to {powers[0].shape}."
                )
            for index in range(4):
                self.sums[index] += powers[index]
        self.count += int(array.shape[0])

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.count <= 0 or self.sums is None:
            raise ValueError("RawMomentAccumulator has no samples.")
        e1, e2, e3, e4 = (value / self.count for value in self.sums)
        variance = np.maximum(e2 - e1 * e1, 0.0)
        mu3 = e3 - 3.0 * e1 * e2 + 2.0 * e1**3
        mu4 = e4 - 4.0 * e1 * e3 + 6.0 * e1 * e1 * e2 - 3.0 * e1**4
        positive = variance > np.finfo(np.float64).eps
        skewness = np.zeros_like(variance)
        kurtosis = np.zeros_like(variance)
        skewness[positive] = mu3[positive] / np.power(variance[positive], 1.5)
        kurtosis[positive] = mu4[positive] / np.square(variance[positive])
        result: dict[str, np.ndarray | int] = {
            "count": self.count,
            "mean": e1,
            "std": np.sqrt(variance),
            "skewness": skewness,
            "kurtosis": kurtosis,
        }
        for key, value in result.items():
            if key != "count":
                ensure_finite(value, name=f"moment {key}")
        return result


def channel_correlation_matrices(
    samples: ArrayLike,
    *,
    eps: float = 1.0e-12,
) -> ArrayLike:
    """Compute one Pearson channel-correlation matrix per sample."""

    if isinstance(samples, torch.Tensor):
        if samples.ndim < 3:
            raise ValueError("samples must have shape [B,C,...].")
        flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
        centered = flat - flat.mean(dim=-1, keepdim=True)
        numerator = torch.matmul(centered, centered.transpose(1, 2))
        norm = torch.linalg.vector_norm(centered, dim=-1)
        denominator = norm.unsqueeze(2) * norm.unsqueeze(1)
        return numerator / denominator.clamp_min(eps)
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError("samples must have shape [B,C,...].")
    flat = array.reshape(array.shape[0], array.shape[1], -1)
    centered = flat - flat.mean(axis=-1, keepdims=True)
    numerator = centered @ np.swapaxes(centered, 1, 2)
    norm = np.linalg.norm(centered, axis=-1)
    denominator = norm[:, :, None] * norm[:, None, :]
    return numerator / np.maximum(denominator, eps)


def pairwise_cosine_values(samples: ArrayLike, *, eps: float = 1.0e-12) -> ArrayLike:
    """Return unique off-diagonal (upper-triangle) sample cosine values."""

    if isinstance(samples, torch.Tensor):
        if samples.ndim < 2 or samples.shape[0] < 2:
            raise ValueError("At least two samples are required for pairwise cosine.")
        flat = samples.reshape(samples.shape[0], -1)
        flat = flat / torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(eps)
        matrix = (flat @ flat.transpose(0, 1)).clamp(-1.0, 1.0)
        indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device)
        return matrix[indices[0], indices[1]]
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("At least two samples are required for pairwise cosine.")
    flat = array.reshape(array.shape[0], -1)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), eps)
    matrix = np.clip(flat @ flat.T, -1.0, 1.0)
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


@dataclass(frozen=True)
class RadialBinMap:
    """Unshifted integer-radius bin definition shared by both generators."""

    indices: np.ndarray
    counts: np.ndarray
    bins: int
    height: int
    width: int


def build_radial_bin_map(
    height: int,
    width: int,
    bins: int | None = None,
) -> RadialBinMap:
    """Build bins from unshifted ``fftfreq * image_size`` radii."""

    height, width = _validate_positive_shape(height, width)
    fy = np.fft.fftfreq(height).astype(np.float64) * height
    fx = np.fft.fftfreq(width).astype(np.float64) * width
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    if bins is None:
        indices = np.floor(radius).astype(np.int64)
        bin_count = int(indices.max()) + 1
    else:
        bin_count = int(bins)
        if bin_count <= 0:
            raise ValueError("bins must be positive.")
        maximum = float(radius.max())
        if maximum <= 0:
            indices = np.zeros((height, width), dtype=np.int64)
        else:
            indices = np.floor(radius / maximum * bin_count).astype(np.int64)
            indices = np.clip(indices, 0, bin_count - 1)
    counts = np.bincount(indices.reshape(-1), minlength=bin_count).astype(np.float64)
    return RadialBinMap(indices=indices, counts=counts, bins=bin_count, height=height, width=width)


def radial_psd_from_magnitude(
    magnitude: ArrayLike,
    radial_map: RadialBinMap,
) -> ArrayLike:
    """Average ``abs(fft2(noise))**2`` into the shared radial bins."""

    if magnitude.ndim != 4:
        raise ValueError(f"magnitude must have shape [B,C,H,W], got {tuple(magnitude.shape)}.")
    if tuple(magnitude.shape[-2:]) != (radial_map.height, radial_map.width):
        raise ValueError(
            f"magnitude H/W={tuple(magnitude.shape[-2:])}, radial map H/W="
            f"{(radial_map.height, radial_map.width)}."
        )
    if isinstance(magnitude, torch.Tensor):
        power = magnitude.square().reshape(magnitude.shape[0], magnitude.shape[1], -1)
        index = torch.as_tensor(radial_map.indices.reshape(-1), device=power.device, dtype=torch.long)
        output = torch.zeros(
            (power.shape[0], power.shape[1], radial_map.bins),
            device=power.device,
            dtype=power.dtype,
        )
        output.scatter_add_(2, index.view(1, 1, -1).expand(power.shape[0], power.shape[1], -1), power)
        counts = torch.as_tensor(radial_map.counts, device=power.device, dtype=power.dtype)
        return output / counts.clamp_min(1.0).view(1, 1, -1)
    array = np.asarray(magnitude, dtype=np.float64)
    power = np.square(array).reshape(array.shape[0], array.shape[1], -1)
    result = np.empty((array.shape[0], array.shape[1], radial_map.bins), dtype=np.float64)
    flat_indices = radial_map.indices.reshape(-1)
    counts = np.maximum(radial_map.counts, 1.0)
    for sample in range(array.shape[0]):
        for channel in range(array.shape[1]):
            result[sample, channel] = np.bincount(
                flat_indices,
                weights=power[sample, channel],
                minlength=radial_map.bins,
            ) / counts
    return result


__all__ = [
    "RadialBinMap",
    "RawMomentAccumulator",
    "RunningMeanVariance",
    "build_pixel_radius_weight_map",
    "build_radial_bin_map",
    "channel_correlation_matrices",
    "coefficient_of_variation",
    "distribution_summary",
    "ensure_finite",
    "expand_normalized_radial_profile",
    "fft_magnitude",
    "fixed_magnitude_from_white",
    "legacy_filtered_gaussian_from_white",
    "load_legacy_filter",
    "load_new_target_amplitude",
    "normalize_per_sample_channel",
    "pairwise_cosine_values",
    "radial_psd_from_magnitude",
    "summarize_cv",
    "valid_frequency_mask",
]
