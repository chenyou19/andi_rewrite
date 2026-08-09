"""Create one large comparison figure for the project's noise samplers.

The figure contains four standardized noise realizations and a radial power
spectrum comparison.  The empirical-spectrum samplers are instantiated from
the same code paths used by training, so the plot is an implementation-level
comparison rather than a hand-crafted illustration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise.empirical_spectrum import EmpiricalSpectrumNoise
from noise.gaussian import GaussianNoise
from noise.pyramid import PyramidNoise
from utils.spectrum_compare import radial_power_profile


DEFAULT_SPECTRUM = ROOT / "outputs" / "spectrum" / "brats21_healthy_empirical_spectrum.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "noise_diagnostics" / "noise_comparison_large.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectrum-path",
        type=Path,
        default=DEFAULT_SPECTRUM,
        help="Empirical spectrum .npz used by the two empirical samplers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination PNG path.",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--channel", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_spectrum_metadata(path: Path, channel: int) -> tuple[int, int, int, int, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as stats:
        height = int(np.asarray(stats["height"]).item())
        width = int(np.asarray(stats["width"]).item())
        channels = int(np.asarray(stats["channels"]).item())
        radial_bins = int(np.asarray(stats["radial_bins"]).item())
        radial_power = np.asarray(stats["radial_power"], dtype=np.float64)
        if radial_power.ndim != 2 or channel < 0 or channel >= radial_power.shape[0]:
            raise ValueError(
                f"Expected radial_power with one profile per channel; got {radial_power.shape} "
                f"for requested channel {channel}."
            )
        metadata: dict[str, object] = {}
        for key in ("num_slices_used", "num_slices_skipped", "mask_mode", "window", "crop_margin"):
            if key not in stats:
                continue
            value = np.asarray(stats[key])
            metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return height, width, channels, radial_bins, radial_power[channel], metadata


def sample_noise(
    sampler: object,
    *,
    num_samples: int,
    channels: int,
    height: int,
    width: int,
) -> np.ndarray:
    with torch.inference_mode():
        sample = sampler.sample(
            (num_samples, channels, height, width),
            device="cpu",
            dtype=torch.float32,
        )
    return sample.detach().cpu().numpy().astype(np.float32, copy=False)


def standardize_channel(batch: np.ndarray, channel: int) -> np.ndarray:
    if batch.ndim != 4 or not 0 <= channel < batch.shape[1]:
        raise ValueError(f"Expected [N, C, H, W] batch and valid channel, got {batch.shape} and {channel}.")
    images = batch[:, channel].astype(np.float64, copy=True)
    images -= images.mean(axis=(-2, -1), keepdims=True)
    std = images.std(axis=(-2, -1), keepdims=True)
    return (images / np.maximum(std, 1.0e-12)).astype(np.float32, copy=False)


def normalized_radial_profiles(images: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = np.stack([radial_power_profile(image, bins=bins) for image in images], axis=0).astype(np.float64)
    profiles = np.maximum(profiles, 1.0e-12)
    profiles /= np.maximum(profiles.mean(axis=1, keepdims=True), 1.0e-12)
    return profiles.mean(axis=0), np.percentile(profiles, 10.0, axis=0), np.percentile(profiles, 90.0, axis=0)


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.maximum(np.asarray(profile, dtype=np.float64), 1.0e-12)
    return profile / np.maximum(profile.mean(), 1.0e-12)


def make_figure(
    *,
    spectrum_path: Path,
    output_path: Path,
    seed: int,
    num_samples: int,
    channel: int,
) -> tuple[Path, Path]:
    height, width, spectrum_channels, radial_bins, target_profile, spectrum_metadata = load_spectrum_metadata(
        spectrum_path, channel
    )

    samplers: dict[str, tuple[object, int, str, str]] = {
        "gaussian": (GaussianNoise(), 1, "Gaussian noise", "#4C78A8"),
        "pyramid": (
            PyramidNoise(discount=0.8, levels=10, normalize=True),
            1,
            "Pyramid noise  ·  discount=0.8, levels=10",
            "#F58518",
        ),
        "filtered_gaussian": (
            EmpiricalSpectrumNoise(
                stats_path=spectrum_path,
                mode="radial",
                radial_key="radial_amplitude",
                radial_power_key="radial_power",
                per_channel=True,
                strength=1.0,
                normalize=True,
                generation_method="filtered_gaussian",
            ),
            spectrum_channels,
            "filtered_gaussian",
            "#54A24B",
        ),
        "fixed_magnitude": (
            EmpiricalSpectrumNoise(
                stats_path=spectrum_path,
                mode="radial",
                radial_key="radial_amplitude",
                radial_power_key="radial_power",
                per_channel=True,
                strength=1.0,
                normalize=True,
                generation_method="fixed_magnitude",
            ),
            spectrum_channels,
            "fixed_magnitude",
            "#E45756",
        ),
    }

    images: dict[str, np.ndarray] = {}
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (sampler, sampler_channels, _label, _color) in samplers.items():
        batch = sample_noise(
            sampler,
            num_samples=num_samples,
            channels=sampler_channels,
            height=height,
            width=width,
        )
        standardized = standardize_channel(batch, channel if sampler_channels > 1 else 0)
        images[name] = standardized
        profiles[name] = normalized_radial_profiles(standardized, radial_bins)

    image_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    fig = plt.figure(figsize=(18, 15), dpi=220, facecolor="white")
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 1.35), hspace=0.28, wspace=0.08)

    image_axes: list[plt.Axes] = []
    image_artist = None
    placements = (
        ("gaussian", grid[0, 0:2]),
        ("pyramid", grid[0, 2:4]),
        ("filtered_gaussian", grid[1, 0:2]),
        ("fixed_magnitude", grid[1, 2:4]),
    )
    for name, placement in placements:
        _sampler, _sampler_channels, label, _color = samplers[name]
        ax = fig.add_subplot(placement)
        image_axes.append(ax)
        image_artist = ax.imshow(
            images[name][0],
            cmap="RdBu_r",
            norm=image_norm,
            interpolation="nearest",
        )
        ax.set_title(label, loc="left", fontsize=14, fontweight="bold", pad=8)
        ax.text(
            0.01,
            0.02,
            f"standardized realization  ·  channel {channel if _sampler_channels > 1 else 0}",
            transform=ax.transAxes,
            fontsize=9,
            color="#4c4c4c",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c8c8c8")
            spine.set_linewidth(0.8)

    if image_artist is not None:
        colorbar = fig.colorbar(image_artist, ax=image_axes, fraction=0.018, pad=0.015, shrink=0.82)
        colorbar.set_label("z-score", rotation=90, labelpad=9)
        colorbar.ax.tick_params(labelsize=9)

    spectrum_ax = fig.add_subplot(grid[2, :])
    frequency = np.linspace(0.0, 1.0, radial_bins)
    target = normalize_profile(target_profile)
    target_x = np.linspace(0.0, 1.0, target.size)
    all_values = [target]
    for mean, low, high in profiles.values():
        all_values.extend((mean, low, high))
    positive_values = np.concatenate([values[np.isfinite(values) & (values > 0)] for values in all_values])
    floor = max(float(np.min(positive_values)) * 0.5, 1.0e-6)

    spectrum_ax.plot(
        target_x,
        np.maximum(target, floor),
        color="#202020",
        linestyle="--",
        linewidth=2.4,
        label="empirical target · radial_power",
        zorder=5,
    )
    for name, (_sampler, _sampler_channels, label, color) in samplers.items():
        mean, low, high = profiles[name]
        spectrum_ax.fill_between(
            frequency,
            np.maximum(low, floor),
            np.maximum(high, floor),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        spectrum_ax.plot(
            frequency,
            np.maximum(mean, floor),
            color=color,
            linewidth=2.1,
            label=label.split("  ·  ")[0],
        )

    spectrum_ax.set_yscale("log")
    spectrum_ax.set_xlim(0.0, 1.0)
    spectrum_ax.set_ylim(floor, max(all_values[0].max(), *(values.max() for values in all_values[1:])) * 1.35)
    spectrum_ax.set_xlabel("Normalized radial spatial frequency  (0 = DC, 1 = corner)", fontsize=11)
    spectrum_ax.set_ylabel("Mean radial power / profile mean", fontsize=11)
    spectrum_ax.set_title("Radial power spectrum comparison", loc="left", fontsize=15, fontweight="bold", pad=10)
    spectrum_ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    spectrum_ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    spectrum_ax.legend(loc="upper right", ncol=5, frameon=False, fontsize=10)
    spectrum_ax.text(
        0.01,
        0.02,
        f"{spectrum_path.name}  ·  channel {channel}  ·  {num_samples} realizations per method  ·  shaded band = 10th–90th percentile",
        transform=spectrum_ax.transAxes,
        fontsize=9,
        color="#4c4c4c",
    )

    fig.suptitle("Noise realizations and radial-spectrum comparison", fontsize=20, fontweight="bold", y=0.985)
    source_note = f"Empirical source: {spectrum_path.relative_to(ROOT) if spectrum_path.is_relative_to(ROOT) else spectrum_path}"
    fig.text(0.01, 0.012, source_note, fontsize=9, color="#555555")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    metadata_path = output_path.with_suffix(".json")
    metadata_payload = {
        "output": str(output_path),
        "spectrum_path": str(spectrum_path),
        "seed": seed,
        "num_samples": num_samples,
        "display_channel": channel,
        "height": height,
        "width": width,
        "spectrum_channels": spectrum_channels,
        "radial_bins": radial_bins,
        "spectrum_metadata": spectrum_metadata,
        "samplers": {
            "gaussian": {"type": "gaussian"},
            "pyramid": {"type": "pyramid", "discount": 0.8, "levels": 10, "normalize": True},
            "filtered_gaussian": {"generation_method": "filtered_gaussian", "mode": "radial", "strength": 1.0},
            "fixed_magnitude": {"generation_method": "fixed_magnitude", "mode": "radial", "strength": 1.0},
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path, metadata_path


def main() -> None:
    args = parse_args()
    spectrum_path = resolve_path(args.spectrum_path)
    output_path = resolve_path(args.output)
    if not spectrum_path.exists():
        raise FileNotFoundError(f"Spectrum file does not exist: {spectrum_path}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.channel < 0:
        raise ValueError("--channel must be non-negative.")

    seed_everything(args.seed)
    output_path, metadata_path = make_figure(
        spectrum_path=spectrum_path,
        output_path=output_path,
        seed=args.seed,
        num_samples=args.num_samples,
        channel=args.channel,
    )
    print(f"saved figure: {output_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
