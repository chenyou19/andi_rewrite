from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SUPPORTED_RADIAL_KEYS = (
    "radial_psd",
    "radial_power_smooth",
    "radial_power_mean",
    "radial_amplitude",
    "radial_power",
)


@dataclass
class SpectrumStatsItem:
    label: str
    path: Path
    radial_key: str
    radial_psd: np.ndarray
    radial_shape: tuple[int, ...]
    image_size: tuple[int, int] | None
    channels: int
    per_channel: bool
    count: int | None

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "path": str(self.path),
            "radial_key": self.radial_key,
            "radial_shape": list(self.radial_shape),
            "channels": self.channels,
            "per_channel": self.per_channel,
        }
        if self.image_size is not None:
            payload["image_size"] = list(self.image_size)
        if self.count is not None:
            payload["count"] = self.count
        return payload


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return cleaned.strip("._") or "item"


def load_lmdb_slice(dataset_path: str | Path, index: int | None = None) -> tuple[torch.Tensor, int, int]:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("compare_npz_spectra requires the optional 'lmdb' package.") from exc

    lmdb_path = Path(dataset_path)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB dataset path does not exist: {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin(write=False) as txn:
            length = int(txn.stat()["entries"])
            if length <= 0:
                raise ValueError(f"LMDB dataset is empty: {lmdb_path}")
            resolved_index = length // 2 if index is None else int(index)
            if resolved_index < 0 or resolved_index >= length:
                raise ValueError(f"Slice index {resolved_index} is out of range for dataset length {length}.")
            byteflow = txn.get(f"{resolved_index:08}".encode("ascii"))
            if byteflow is None:
                raise IndexError(f"LMDB key not found for index {resolved_index}.")
            tensor = torch.from_numpy(np.asarray(pickle.loads(byteflow))).float()
    finally:
        env.close()

    return tensor, resolved_index, length


def _extract_int(stats: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in stats:
            value = np.asarray(stats[key])
            if value.size == 0:
                continue
            return int(value.reshape(-1)[0])
    return None


def _extract_bool(stats: Any, key: str, default: bool) -> bool:
    if key not in stats:
        return default
    value = np.asarray(stats[key])
    if value.dtype.kind in {"b", "i", "u"}:
        return bool(value.reshape(-1)[0])
    if value.dtype.kind in {"U", "S", "O"}:
        text = str(value.reshape(-1)[0]).strip().lower()
        return text in {"1", "true", "yes", "y"}
    return default


def load_radial_psd(path: str | Path, label: str) -> SpectrumStatsItem:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Spectrum stats file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as stats:
        radial_key = next((key for key in SUPPORTED_RADIAL_KEYS if key in stats), None)
        if radial_key is None:
            supported = ", ".join(SUPPORTED_RADIAL_KEYS)
            raise KeyError(f"{npz_path} does not contain a supported radial spectrum key. Tried: {supported}")

        radial_psd = np.asarray(stats[radial_key], dtype=np.float32)
        if radial_psd.ndim == 1:
            radial_psd = radial_psd[None, :]
        elif radial_psd.ndim != 2:
            raise ValueError(
                f"Radial spectrum in {npz_path} must be 1D or 2D, got shape {tuple(radial_psd.shape)}."
            )

        image_height = _extract_int(stats, ("image_height", "height"))
        image_width = _extract_int(stats, ("image_width", "width"))
        image_size_scalar = _extract_int(stats, ("image_size",))
        if image_height is None and image_size_scalar is not None:
            image_height = image_size_scalar
        if image_width is None and image_size_scalar is not None:
            image_width = image_size_scalar

        channels = _extract_int(stats, ("channels",))
        if channels is None:
            channels = int(radial_psd.shape[0])
        count = _extract_int(stats, ("count", "num_slices_used"))
        per_channel = _extract_bool(stats, "per_channel", radial_psd.shape[0] > 1)

    image_size = None
    if image_height is not None and image_width is not None:
        image_size = (int(image_height), int(image_width))

    return SpectrumStatsItem(
        label=label,
        path=npz_path,
        radial_key=radial_key,
        radial_psd=radial_psd.astype(np.float32, copy=False),
        radial_shape=tuple(radial_psd.shape),
        image_size=image_size,
        channels=int(channels),
        per_channel=bool(per_channel),
        count=count,
    )


def select_radial_profile(item: SpectrumStatsItem, channel: int) -> np.ndarray:
    profile = item.radial_psd
    if profile.shape[0] == 1:
        return profile[0]
    if channel < 0 or channel >= profile.shape[0]:
        raise ValueError(
            f"Requested channel {channel} is out of range for {item.path} radial spectrum with {profile.shape[0]} channels."
        )
    return profile[channel]


def build_weight_map(radial_psd: np.ndarray, height: int, width: int) -> torch.Tensor:
    profile = np.asarray(radial_psd, dtype=np.float32)
    if profile.ndim != 1:
        raise ValueError(f"build_weight_map expects a 1D radial profile, got shape {profile.shape}.")
    if profile.size == 0:
        raise ValueError("Radial profile must not be empty.")

    fy = torch.fft.fftfreq(height, dtype=torch.float32).view(height, 1)
    fx = torch.fft.fftfreq(width, dtype=torch.float32).view(1, width)
    radius = torch.sqrt(fy.square() + fx.square())
    max_radius = float(radius.max().item())
    if max_radius <= 0.0:
        positions = torch.zeros((height, width), dtype=torch.float32)
    else:
        positions = radius / max_radius * float(profile.size - 1)

    lower = positions.floor().to(torch.long)
    upper = torch.clamp(lower + 1, max=profile.size - 1)
    fraction = positions - lower.to(torch.float32)
    values = torch.from_numpy(profile.astype(np.float32, copy=False))
    weight_map = values[lower] * (1.0 - fraction) + values[upper] * fraction
    mean = weight_map.mean().clamp_min(torch.finfo(weight_map.dtype).eps)
    return weight_map / mean


def normalize_per_channel(image: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return (image - mean) / std


def spectrum_noise_from_white(
    white: torch.Tensor,
    radial_psd: np.ndarray,
    mix_white: float,
) -> torch.Tensor:
    if white.ndim != 4:
        raise ValueError(f"Expected white noise with shape [B, C, H, W], got {tuple(white.shape)}.")
    if white.shape[1] != 1:
        raise ValueError("spectrum_noise_from_white currently expects a single selected channel [B, 1, H, W].")
    if not 0.0 <= float(mix_white) <= 1.0:
        raise ValueError("mix_white must be in [0, 1].")

    weight_map = build_weight_map(radial_psd, int(white.shape[-2]), int(white.shape[-1]))
    work = white.to(dtype=torch.float32)
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    shaped = torch.fft.ifft2(spectrum * weight_map.to(device=work.device, dtype=work.dtype), dim=(-2, -1)).real
    mixed = ((1.0 - float(mix_white)) * shaped) + (float(mix_white) * work)
    return normalize_per_channel(mixed).to(dtype=white.dtype)


def spectrum_noise_like(
    shape: tuple[int, int, int, int],
    radial_psd: np.ndarray,
    mix_white: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    white = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return spectrum_noise_from_white(white, radial_psd, mix_white)


def linear_alpha_hat(
    timestep: int,
    *,
    noise_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device | str = "cpu",
) -> float:
    if timestep < 0 or timestep >= noise_steps:
        raise ValueError(f"timestep must be in [0, {noise_steps - 1}], got {timestep}.")
    del device
    beta = torch.linspace(float(beta_start), float(beta_end), int(noise_steps), dtype=torch.float32)
    alpha_hat = torch.cumprod(1.0 - beta, dim=0)
    return float(alpha_hat[int(timestep)].item())


def apply_forward_noise(image: torch.Tensor, noise: torch.Tensor, alpha_hat: float) -> torch.Tensor:
    alpha = float(alpha_hat)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha_hat must be in (0, 1], got {alpha_hat}.")
    return (math.sqrt(alpha) * image) + (math.sqrt(1.0 - alpha) * noise)


def to_display_image(image: torch.Tensor, mode: str) -> np.ndarray:
    array = image.detach().cpu().float().squeeze().numpy()
    if array.ndim != 2:
        raise ValueError(f"to_display_image expects a 2D image after squeeze, got shape {array.shape}.")
    if mode == "mri":
        return np.clip((array + 1.0) * 0.5, 0.0, 1.0)
    if mode == "noise":
        scale = max(float(np.max(np.abs(array))), 1.0e-8)
        return np.clip((array / (2.0 * scale)) + 0.5, 0.0, 1.0)
    raise ValueError(f"Unsupported display mode: {mode}")


def log_spectrum_map(image: torch.Tensor | np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"log_spectrum_map expects a 2D image, got shape {array.shape}.")
    fft = np.fft.fftshift(np.fft.fft2(array))
    power = np.abs(fft) ** 2
    spectrum = np.log10(power + float(eps))
    spectrum -= float(spectrum.min())
    max_value = float(spectrum.max())
    if max_value > 0.0:
        spectrum /= max_value
    return spectrum.astype(np.float32, copy=False)


def radial_power_profile(image: torch.Tensor | np.ndarray, bins: int | None = None) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"radial_power_profile expects a 2D image, got shape {array.shape}.")
    height, width = array.shape
    bin_count = int(bins or max(1, min(height, width) // 2))
    fft = np.fft.fftshift(np.fft.fft2(array))
    power = np.abs(fft) ** 2

    yy, xx = np.indices((height, width), dtype=np.float32)
    radius = np.sqrt((yy - (height / 2.0)) ** 2 + (xx - (width / 2.0)) ** 2)
    max_radius = float(radius.max())
    if max_radius <= 0.0:
        index = np.zeros((height, width), dtype=np.int64)
    else:
        index = np.floor(radius / max_radius * (bin_count - 1)).astype(np.int64)
        index = np.clip(index, 0, bin_count - 1)
    counts = np.bincount(index.ravel(), minlength=bin_count).astype(np.float64)
    sums = np.bincount(index.ravel(), weights=power.ravel(), minlength=bin_count).astype(np.float64)
    return (sums / np.maximum(counts, 1.0)).astype(np.float32, copy=False)


def save_gray(path: str | Path, image: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").save(output_path)
    return output_path


def write_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path
