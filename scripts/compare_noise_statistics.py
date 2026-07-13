"""Compare two production ``EmpiricalSpectrumNoise`` generation methods.

This is a diagnostic-only CLI.  It deliberately imports the production
``EmpiricalSpectrumNoise`` implementation, but it does not modify or hook the
training/evaluation pipeline. The comparison mode is:

``actual``
    Both methods call production ``EmpiricalSpectrumNoise.sample`` with the
    same RNG state, shape, stats, strength, and normalization settings.

Large tensors are reduced batch by batch on CPU with float64 accumulators.  A
deterministic second pass produces exact common-bin Fourier histograms without
retaining all samples in memory.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from _bootstrap import bootstrap
except ImportError:  # pragma: no cover - supports ``python -m`` execution
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch

from andi_rewrite.noise.empirical_spectrum import EmpiricalSpectrumNoise
from andi_rewrite.utils.noise_statistics import (
    RawMomentAccumulator,
    RunningMeanVariance,
    build_radial_bin_map,
    channel_correlation_matrices,
    coefficient_of_variation,
    distribution_summary,
    ensure_finite,
    fft_magnitude,
    pairwise_cosine_values,
    radial_psd_from_magnitude,
    summarize_cv,
    valid_frequency_mask,
)


METHODS = ("method_a", "method_b")
MODE_ACTUAL = "actual"
VALID_MODES = (MODE_ACTUAL,)
HISTOGRAM_BINS = 160
RADIAL_EXAMPLE_COUNT = 20
NOISE_EXAMPLE_COUNT = 4
VALID_FREQUENCY_RELATIVE_THRESHOLD = 1.0e-6
_PLOT_METADATA_FOOTER: str | None = None


@dataclass(frozen=True)
class RunSettings:
    stats_path: Path
    num_samples: int
    batch_size: int
    channels: int
    height: int
    width: int
    seed: int
    device: torch.device
    output_dir: Path
    cosine_samples: int
    eps: float
    comparison_mode: str
    synthetic: bool
    no_plots: bool
    method_a: str
    method_b: str

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.channels, self.height, self.width)


@dataclass
class MethodAccumulator:
    """Streaming state for one generator within one comparison mode."""

    fft_stats: RunningMeanVariance = field(default_factory=RunningMeanVariance)
    pixel_moments: RawMomentAccumulator = field(default_factory=RawMomentAccumulator)
    spatial_corr_stats: RunningMeanVariance = field(default_factory=RunningMeanVariance)
    fft_corr_stats: RunningMeanVariance = field(default_factory=RunningMeanVariance)
    radial_stats: RunningMeanVariance = field(default_factory=RunningMeanVariance)
    spatial_off_diagonal: list[np.ndarray] = field(default_factory=list)
    fft_off_diagonal: list[np.ndarray] = field(default_factory=list)
    cosine_spatial: list[torch.Tensor] = field(default_factory=list)
    cosine_fft: list[torch.Tensor] = field(default_factory=list)
    radial_examples: list[np.ndarray] = field(default_factory=list)
    noise_examples: list[np.ndarray] = field(default_factory=list)
    fft_examples: list[np.ndarray] = field(default_factory=list)
    magnitude_min: float = math.inf
    magnitude_max: float = -math.inf
    log_magnitude_min: float = math.inf
    log_magnitude_max: float = -math.inf
    cosine_collected: int = 0
    radial_examples_collected: int = 0
    noise_examples_collected: int = 0


@dataclass
class MethodResult:
    fft_mean: np.ndarray
    fft_std: np.ndarray
    fft_cv: np.ndarray
    valid_mask: np.ndarray
    pixel_mean: np.ndarray
    pixel_std: np.ndarray
    pixel_skewness: np.ndarray
    pixel_kurtosis: np.ndarray
    spatial_corr_mean: np.ndarray
    spatial_corr_std: np.ndarray
    fft_corr_mean: np.ndarray
    fft_corr_std: np.ndarray
    spatial_off_diagonal: np.ndarray
    fft_off_diagonal: np.ndarray
    cosine_spatial: np.ndarray
    cosine_fft: np.ndarray
    radial_mean: np.ndarray
    radial_std: np.ndarray
    radial_cv: np.ndarray
    radial_examples: np.ndarray
    noise_examples: np.ndarray
    fft_examples: np.ndarray
    cv_summary: dict[str, Any]
    pixel_summary: dict[str, Any]
    correlation_summary: dict[str, Any]
    cosine_summary: dict[str, Any]
    radial_summary: dict[str, Any]
    histogram_range: dict[str, float]


@dataclass
class HistogramResult:
    magnitude_edges: np.ndarray
    log_magnitude_edges: np.ndarray
    magnitude_all: dict[str, np.ndarray]
    magnitude_valid: dict[str, np.ndarray]
    log_magnitude_all: dict[str, np.ndarray]
    log_magnitude_valid: dict[str, np.ndarray]


@dataclass
class ModeResult:
    name: str
    methods: dict[str, MethodResult]
    histograms: HistogramResult
    comparison: dict[str, Any]
    conclusion: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two generation methods through the production "
            "EmpiricalSpectrumNoise(strength=1.0) sampler."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Empirical-spectrum NPZ used by both methods. Optional only with --synthetic.",
    )
    parser.add_argument(
        "--method-a",
        default="fixed_magnitude",
        help="First production EmpiricalSpectrumNoise generation_method.",
    )
    parser.add_argument(
        "--method-b",
        default="filtered_gaussian",
        help="Second production EmpiricalSpectrumNoise generation_method.",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (for example cpu, cuda, cuda:0, or auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/noise_diagnostics/empirical_generation_methods"),
    )
    parser.add_argument("--cosine-samples", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument(
        "--comparison-mode",
        choices=(MODE_ACTUAL,),
        default=MODE_ACTUAL,
        help="Run both requested methods through the production sampler.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create self-contained synthetic stats inside --output-dir.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG creation (useful for CPU smoke tests).",
    )
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    normalized = str(name).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "height": args.height,
        "width": args.width,
        "cosine_samples": args.cosine_samples,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 for sample variation diagnostics.")
    if args.cosine_samples < 2:
        raise ValueError("--cosine-samples must be at least 2.")
    if not math.isfinite(float(args.eps)) or float(args.eps) <= 0.0:
        raise ValueError(f"--eps must be a finite positive value, got {args.eps}.")
    if not args.synthetic and args.stats_path is None:
        raise ValueError("--stats-path is required unless --synthetic is used.")
    if args.stats_path is not None and not args.synthetic and not args.stats_path.is_file():
        raise FileNotFoundError(f"Spectrum stats path does not exist: {args.stats_path}")


def comparison_modes(value: str) -> tuple[str, ...]:
    return VALID_MODES if value == "both" else (value,)


def select_radial_example_indices(num_samples: int, seed: int) -> tuple[int, ...]:
    """Choose the reproducible random sample subset used by radial example plots."""

    return tuple(
        sorted(
            np.random.default_rng(int(seed) + 17).choice(
                int(num_samples),
                size=min(RADIAL_EXAMPLE_COUNT, int(num_samples)),
                replace=False,
            ).tolist()
        )
    )


def _expand_normalized_radial(profile: np.ndarray, height: int, width: int) -> np.ndarray:
    """Mirror ``EmpiricalSpectrumNoise._expand_radial_profile`` for fixtures."""

    profile = np.asarray(profile, dtype=np.float32)
    if profile.ndim != 2 or profile.shape[-1] < 1:
        raise ValueError(f"Expected radial profile [C, R], got {profile.shape}.")
    fy = np.fft.fftfreq(height).astype(np.float32)
    fx = np.fft.fftfreq(width).astype(np.float32)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    maximum = float(radius.max())
    positions = np.zeros_like(radius) if maximum <= 0.0 else radius / maximum * (profile.shape[1] - 1)
    x = np.arange(profile.shape[1], dtype=np.float32)
    result = np.empty((profile.shape[0], height, width), dtype=np.float32)
    for channel in range(profile.shape[0]):
        result[channel] = np.interp(positions.ravel(), x, profile[channel]).reshape(height, width)
    return result


def create_synthetic_stats(
    output_dir: Path,
    channels: int,
    height: int,
    width: int,
    eps: float,
) -> Path:
    """Create a deterministic NPZ fixture for both production methods."""

    output_dir.mkdir(parents=True, exist_ok=True)
    radial_bins = max(4, int(math.ceil(math.hypot(height / 2.0, width / 2.0))) + 1)
    radial_coordinate = np.linspace(0.0, 1.0, radial_bins, dtype=np.float64)
    radial_amplitude = np.empty((channels, radial_bins), dtype=np.float32)
    for channel in range(channels):
        slope = 1.7 + 0.2 * channel
        shoulder = 5.0 + channel
        profile = 1.0 / np.power(1.0 + shoulder * radial_coordinate, slope)
        profile += 0.025 * np.exp(-0.5 * ((radial_coordinate - 0.32) / 0.08) ** 2)
        radial_amplitude[channel] = np.maximum(profile, eps).astype(np.float32)

    radial_power = np.square(radial_amplitude, dtype=np.float32)
    mean_amplitude = np.fft.fftshift(
        _expand_normalized_radial(radial_amplitude, height, width), axes=(-2, -1)
    ).copy()
    mean_power = np.square(mean_amplitude, dtype=np.float32)

    new_path = output_dir / "synthetic_empirical_spectrum_stats.npz"
    np.savez_compressed(
        new_path,
        mean_amplitude=mean_amplitude,
        mean_power=mean_power,
        radial_amplitude=radial_amplitude,
        radial_power=radial_power,
        radial_counts=np.ones(radial_bins, dtype=np.int64),
        num_slices_used=np.asarray(1, dtype=np.int64),
        num_slices_skipped=np.asarray(0, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        height=np.asarray(height, dtype=np.int64),
        width=np.asarray(width, dtype=np.int64),
        radial_bins=np.asarray(radial_bins, dtype=np.int64),
        mask_mode=np.asarray("synthetic"),
        eps=np.asarray(eps, dtype=np.float64),
        crop_margin=np.asarray(0, dtype=np.int64),
        window=np.asarray("none"),
    )

    return new_path


def build_settings(args: argparse.Namespace) -> RunSettings:
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        stats_path = create_synthetic_stats(
            output_dir=output_dir,
            channels=int(args.channels),
            height=int(args.height),
            width=int(args.width),
            eps=float(args.eps),
        )
    else:
        assert args.stats_path is not None
        stats_path = args.stats_path.resolve()

    return RunSettings(
        stats_path=stats_path,
        num_samples=int(args.num_samples),
        batch_size=min(int(args.batch_size), int(args.num_samples)),
        channels=int(args.channels),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
        device=resolve_device(args.device),
        output_dir=output_dir,
        cosine_samples=min(int(args.cosine_samples), int(args.num_samples)),
        eps=float(args.eps),
        comparison_mode=str(args.comparison_mode),
        synthetic=bool(args.synthetic),
        no_plots=bool(args.no_plots),
        method_a=EmpiricalSpectrumNoise._canonical_generation_method(str(args.method_a)),
        method_b=EmpiricalSpectrumNoise._canonical_generation_method(str(args.method_b)),
    )


def npz_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        entries: dict[str, Any] = {}
        for key in data.files:
            value = np.asarray(data[key])
            entry: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.ndim == 0:
                scalar = value.item()
                if isinstance(scalar, (str, bool, int, float, np.generic)):
                    entry["value"] = json_ready(scalar)
            entries[key] = entry
    return {"path": str(path), "keys": entries}


def json_ready(value: Any) -> Any:
    """Recursively convert numerical objects into strict finite JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def source_location(obj: Any) -> str:
    try:
        path = Path(inspect.getsourcefile(obj) or "<unknown>").resolve()
        _, start = inspect.getsourcelines(obj)
        return f"{path}:{start}"
    except (OSError, TypeError):
        return "<unavailable>"


def git_commit(project_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={project_dir.as_posix()}", "-C", str(project_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def environment_metadata(settings: RunSettings) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parents[1]
    cuda_device = None
    if settings.device.type == "cuda":
        cuda_device = torch.cuda.get_device_name(settings.device)
    return {
        "git_commit": git_commit(project_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(settings.device),
        "cuda_device_name": cuda_device,
    }


@dataclass
class GeneratorContext:
    settings: RunSettings
    method_a_generator: EmpiricalSpectrumNoise
    method_b_generator: EmpiricalSpectrumNoise
    radial_bin_map: Any
    radial_example_indices: tuple[int, ...]


def create_generator_context(settings: RunSettings) -> GeneratorContext:
    method_a_generator = EmpiricalSpectrumNoise(
        stats_path=settings.stats_path,
        mode="radial",
        generation_method=settings.method_a,
        radial_key="radial_amplitude",
        radial_power_key="radial_power",
        per_channel=True,
        strength=1.0,
        normalize=True,
        eps=settings.eps,
    )
    method_b_generator = EmpiricalSpectrumNoise(
        stats_path=settings.stats_path,
        mode="radial",
        generation_method=settings.method_b,
        radial_key="radial_amplitude",
        radial_power_key="radial_power",
        per_channel=True,
        strength=1.0,
        normalize=True,
        eps=settings.eps,
    )
    # Let the production implementation perform its own validation as well.
    method_a_generator._validate_shape((1, settings.channels, settings.height, settings.width))
    method_b_generator._validate_shape((1, settings.channels, settings.height, settings.width))

    radial_map = build_radial_bin_map(settings.height, settings.width)
    radial_example_indices = select_radial_example_indices(
        settings.num_samples, settings.seed
    )
    return GeneratorContext(
        settings=settings,
        method_a_generator=method_a_generator,
        method_b_generator=method_b_generator,
        radial_bin_map=radial_map,
        radial_example_indices=radial_example_indices,
    )


def seed_runtime(seed: int, device: torch.device) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_device_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device=device)
    return torch.get_rng_state()


def set_device_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device=device)
    else:
        torch.set_rng_state(state)


def generate_pair(
    context: GeneratorContext,
    mode: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a paired production batch without advancing RNG twice."""

    settings = context.settings
    shape = (batch_size, settings.channels, settings.height, settings.width)
    if mode == MODE_ACTUAL:
        # Restore the state before method B so both production samplers receive
        # the same white draw and the final RNG state advances exactly once.
        state = get_device_rng_state(settings.device)
        method_a = context.method_a_generator.sample(
            shape=shape,
            device=settings.device,
            dtype=torch.float32,
        )
        set_device_rng_state(settings.device, state)
        method_b = context.method_b_generator.sample(
            shape=shape,
            device=settings.device,
            dtype=torch.float32,
        )
    else:  # pragma: no cover - protected by argparse
        raise ValueError(f"Unknown comparison mode: {mode}")

    expected = shape
    if tuple(method_a.shape) != expected or tuple(method_b.shape) != expected:
        raise RuntimeError(
            f"Generator shape mismatch: expected {expected}, "
            f"method_a={tuple(method_a.shape)}, method_b={tuple(method_b.shape)}."
        )
    ensure_finite(method_a, name=f"{mode}.method_a")
    ensure_finite(method_b, name=f"{mode}.method_b")
    return method_a, method_b


def iter_paired_batches(
    context: GeneratorContext,
    mode: str,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    settings = context.settings
    seed_runtime(settings.seed, settings.device)
    produced = 0
    while produced < settings.num_samples:
        current = min(settings.batch_size, settings.num_samples - produced)
        method_a, method_b = generate_pair(context, mode, current)
        yield produced, method_a, method_b
        produced += current


def _upper_off_diagonal(matrices: torch.Tensor) -> np.ndarray:
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"Expected correlation matrices [B,C,C], got {tuple(matrices.shape)}.")
    channels = int(matrices.shape[-1])
    if channels < 2:
        return np.empty((0,), dtype=np.float64)
    indices = torch.triu_indices(channels, channels, offset=1)
    values = matrices[:, indices[0], indices[1]]
    return values.detach().cpu().to(torch.float64).numpy().reshape(-1)


def _append_examples(
    destination: list[np.ndarray],
    values: torch.Tensor,
    already_collected: int,
    limit: int,
) -> int:
    remaining = max(0, limit - already_collected)
    take = min(remaining, int(values.shape[0]))
    if take > 0:
        destination.append(values[:take].detach().cpu().to(torch.float32).numpy())
    return already_collected + take


def _running_finalize(accumulator: RunningMeanVariance) -> tuple[np.ndarray, np.ndarray]:
    result = accumulator.finalize()
    if isinstance(result, Mapping):
        mean = result["mean"]
        std = result["std"]
    elif isinstance(result, tuple) and len(result) >= 2:
        mean, std = result[:2]
    else:
        mean = getattr(result, "mean")
        std = getattr(result, "std")
    return (
        torch.as_tensor(mean).detach().cpu().to(torch.float64).numpy(),
        torch.as_tensor(std).detach().cpu().to(torch.float64).numpy(),
    )


def _moments_finalize(
    accumulator: RawMomentAccumulator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result = accumulator.finalize()
    if isinstance(result, Mapping):
        values = tuple(result[key] for key in ("mean", "std", "skewness", "kurtosis"))
    else:
        values = tuple(getattr(result, key) for key in ("mean", "std", "skewness", "kurtosis"))
    return tuple(
        torch.as_tensor(item).detach().cpu().to(torch.float64).numpy() for item in values
    )  # type: ignore[return-value]


def _concat_or_empty(chunks: list[np.ndarray], shape: tuple[int, ...] = (0,)) -> np.ndarray:
    if not chunks:
        return np.empty(shape, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _summary(values: np.ndarray | torch.Tensor) -> dict[str, float]:
    result = distribution_summary(values)
    return {str(key): float(value) for key, value in dict(result).items()}


def _safe_summary(values: np.ndarray | torch.Tensor) -> dict[str, float]:
    array = torch.as_tensor(values).detach().cpu().reshape(-1)
    if array.numel() == 0:
        return {key: 0.0 for key in ("mean", "median", "std", "min", "max", "p01", "p05", "p25", "p75", "p90", "p95", "p99")}
    return _summary(array)


def update_method_accumulator(
    state: MethodAccumulator,
    noise: torch.Tensor,
    context: GeneratorContext,
    offset: int,
) -> None:
    settings = context.settings
    magnitude = fft_magnitude(noise)
    ensure_finite(magnitude, name="fft_magnitude")

    noise_cpu = noise.detach().cpu().to(torch.float64)
    magnitude_cpu = magnitude.detach().cpu().to(torch.float64)
    state.pixel_moments.update(noise_cpu)
    state.fft_stats.update(magnitude_cpu)

    spatial_corr = channel_correlation_matrices(noise_cpu)
    fft_corr = channel_correlation_matrices(magnitude_cpu)
    state.spatial_corr_stats.update(spatial_corr)
    state.fft_corr_stats.update(fft_corr)
    state.spatial_off_diagonal.append(_upper_off_diagonal(spatial_corr))
    state.fft_off_diagonal.append(_upper_off_diagonal(fft_corr))

    radial = radial_psd_from_magnitude(magnitude_cpu, context.radial_bin_map)
    state.radial_stats.update(radial)
    selected_local = [
        sample_index - offset
        for sample_index in context.radial_example_indices
        if offset <= sample_index < offset + int(radial.shape[0])
    ]
    if selected_local:
        state.radial_examples.append(radial[selected_local].cpu().numpy())
        state.radial_examples_collected += len(selected_local)

    remaining_cosine = max(0, settings.cosine_samples - state.cosine_collected)
    take_cosine = min(remaining_cosine, int(noise_cpu.shape[0]))
    if take_cosine:
        state.cosine_spatial.append(noise_cpu[:take_cosine].to(torch.float32))
        state.cosine_fft.append(magnitude_cpu[:take_cosine].to(torch.float32))
        state.cosine_collected += take_cosine

    state.noise_examples_collected = _append_examples(
        state.noise_examples,
        noise_cpu,
        state.noise_examples_collected,
        NOISE_EXAMPLE_COUNT,
    )
    # Store unshifted magnitude. Plotting performs fftshift explicitly.
    if sum(chunk.shape[0] for chunk in state.fft_examples) < NOISE_EXAMPLE_COUNT:
        take_fft = min(
            NOISE_EXAMPLE_COUNT - sum(chunk.shape[0] for chunk in state.fft_examples),
            int(magnitude_cpu.shape[0]),
        )
        state.fft_examples.append(magnitude_cpu[:take_fft].to(torch.float32).numpy())

    state.magnitude_min = min(state.magnitude_min, float(magnitude_cpu.min().item()))
    state.magnitude_max = max(state.magnitude_max, float(magnitude_cpu.max().item()))
    log_magnitude = torch.log10(magnitude_cpu + settings.eps)
    state.log_magnitude_min = min(state.log_magnitude_min, float(log_magnitude.min().item()))
    state.log_magnitude_max = max(state.log_magnitude_max, float(log_magnitude.max().item()))


def collect_streaming_statistics(
    context: GeneratorContext,
    mode: str,
) -> dict[str, MethodAccumulator]:
    states = {method: MethodAccumulator() for method in METHODS}
    total_batches = math.ceil(context.settings.num_samples / context.settings.batch_size)
    for batch_index, (offset, method_a, method_b) in enumerate(iter_paired_batches(context, mode), start=1):
        update_method_accumulator(states["method_a"], method_a, context, offset)
        update_method_accumulator(states["method_b"], method_b, context, offset)
        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            completed = min(offset + method_a.shape[0], context.settings.num_samples)
            print(f"[{mode}] statistics {completed}/{context.settings.num_samples}", flush=True)
        del method_a, method_b
    return states


def _cv_summaries(cv: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    cv_tensor = torch.from_numpy(cv)
    mask_tensor = torch.from_numpy(mask.astype(bool, copy=False))
    all_summary = dict(summarize_cv(cv_tensor))
    valid_summary = dict(summarize_cv(cv_tensor, mask=mask_tensor))
    per_channel: list[dict[str, Any]] = []
    for channel in range(cv.shape[0]):
        per_channel.append(
            {
                "channel": channel,
                "all": dict(summarize_cv(cv_tensor[channel])),
                "valid": dict(
                    summarize_cv(cv_tensor[channel], mask=mask_tensor[channel])
                ),
                "valid_bins": int(mask[channel].sum()),
                "total_bins": int(mask[channel].size),
                "dc": float(cv[channel, 0, 0]),
            }
        )
    return {
        "all": json_ready(all_summary),
        "valid": json_ready(valid_summary),
        "dc": json_ready(_safe_summary(cv[:, 0, 0])),
        "per_channel": json_ready(per_channel),
        "valid_bins": int(mask.sum()),
        "total_bins": int(mask.size),
        "valid_definition": (
            "mean_magnitude[c,y,x] > max(mean_magnitude[c]) * 1e-6"
        ),
    }


def _field_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "pooled": json_ready(_safe_summary(values)),
        "per_channel": [
            {"channel": channel, **json_ready(_safe_summary(values[channel]))}
            for channel in range(values.shape[0])
        ],
    }


def finalize_method_accumulator(
    state: MethodAccumulator,
    settings: RunSettings,
) -> MethodResult:
    fft_mean, fft_std = _running_finalize(state.fft_stats)
    fft_cv_tensor = coefficient_of_variation(
        torch.from_numpy(fft_std), torch.from_numpy(fft_mean), eps=settings.eps
    )
    fft_cv = torch.as_tensor(fft_cv_tensor).cpu().numpy().astype(np.float64, copy=False)
    valid = torch.as_tensor(
        valid_frequency_mask(
            torch.from_numpy(fft_mean), relative_threshold=VALID_FREQUENCY_RELATIVE_THRESHOLD
        )
    ).cpu().numpy().astype(bool, copy=False)

    pixel_mean, pixel_std, pixel_skewness, pixel_kurtosis = _moments_finalize(
        state.pixel_moments
    )
    spatial_corr_mean, spatial_corr_std = _running_finalize(state.spatial_corr_stats)
    fft_corr_mean, fft_corr_std = _running_finalize(state.fft_corr_stats)
    radial_mean, radial_std = _running_finalize(state.radial_stats)
    radial_cv = torch.as_tensor(
        coefficient_of_variation(
            torch.from_numpy(radial_std), torch.from_numpy(radial_mean), eps=settings.eps
        )
    ).cpu().numpy().astype(np.float64, copy=False)

    spatial_samples = torch.cat(state.cosine_spatial, dim=0)
    fft_samples = torch.cat(state.cosine_fft, dim=0)
    cosine_spatial = torch.as_tensor(
        pairwise_cosine_values(spatial_samples)
    ).cpu().to(torch.float64).numpy()
    cosine_fft = torch.as_tensor(
        pairwise_cosine_values(fft_samples)
    ).cpu().to(torch.float64).numpy()

    spatial_off = _concat_or_empty(state.spatial_off_diagonal)
    fft_off = _concat_or_empty(state.fft_off_diagonal)
    radial_examples = _concat_or_empty(
        state.radial_examples,
        shape=(0, settings.channels, radial_mean.shape[-1]),
    )
    noise_examples = _concat_or_empty(
        state.noise_examples,
        shape=(0, settings.channels, settings.height, settings.width),
    )
    fft_examples = _concat_or_empty(
        state.fft_examples,
        shape=(0, settings.channels, settings.height, settings.width),
    )

    finite_arrays = {
        "fft_mean": fft_mean,
        "fft_std": fft_std,
        "fft_cv": fft_cv,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "pixel_skewness": pixel_skewness,
        "pixel_kurtosis": pixel_kurtosis,
        "spatial_corr_mean": spatial_corr_mean,
        "spatial_corr_std": spatial_corr_std,
        "fft_corr_mean": fft_corr_mean,
        "fft_corr_std": fft_corr_std,
        "cosine_spatial": cosine_spatial,
        "cosine_fft": cosine_fft,
        "radial_mean": radial_mean,
        "radial_std": radial_std,
        "radial_cv": radial_cv,
    }
    for name, value in finite_arrays.items():
        ensure_finite(value, name=name)

    pixel_summary = {
        "skewness": _field_summary(pixel_skewness),
        "pearson_kurtosis": _field_summary(pixel_kurtosis),
    }
    correlation_summary = {
        "spatial_off_diagonal": json_ready(_safe_summary(spatial_off)),
        "fft_magnitude_off_diagonal": json_ready(_safe_summary(fft_off)),
    }
    cosine_summary = {
        "spatial": json_ready(_safe_summary(cosine_spatial)),
        "fft_magnitude": json_ready(_safe_summary(cosine_fft)),
        "samples": int(spatial_samples.shape[0]),
        "pairs": int(cosine_spatial.size),
        "diagonal_excluded": True,
    }
    radial_summary = {
        "cv": _field_summary(radial_cv),
        "bins": int(radial_cv.shape[-1]),
    }

    return MethodResult(
        fft_mean=fft_mean,
        fft_std=fft_std,
        fft_cv=fft_cv,
        valid_mask=valid,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        pixel_skewness=pixel_skewness,
        pixel_kurtosis=pixel_kurtosis,
        spatial_corr_mean=spatial_corr_mean,
        spatial_corr_std=spatial_corr_std,
        fft_corr_mean=fft_corr_mean,
        fft_corr_std=fft_corr_std,
        spatial_off_diagonal=spatial_off,
        fft_off_diagonal=fft_off,
        cosine_spatial=cosine_spatial,
        cosine_fft=cosine_fft,
        radial_mean=radial_mean,
        radial_std=radial_std,
        radial_cv=radial_cv,
        radial_examples=radial_examples,
        noise_examples=noise_examples,
        fft_examples=fft_examples,
        cv_summary=_cv_summaries(fft_cv, valid),
        pixel_summary=json_ready(pixel_summary),
        correlation_summary=json_ready(correlation_summary),
        cosine_summary=json_ready(cosine_summary),
        radial_summary=json_ready(radial_summary),
        histogram_range={
            "magnitude_min": state.magnitude_min,
            "magnitude_max": state.magnitude_max,
            "log_magnitude_min": state.log_magnitude_min,
            "log_magnitude_max": state.log_magnitude_max,
        },
    )


def _common_histogram_edges(
    method_results: Mapping[str, MethodResult],
    minimum_key: str,
    maximum_key: str,
    bins: int = HISTOGRAM_BINS,
) -> np.ndarray:
    minimum = min(result.histogram_range[minimum_key] for result in method_results.values())
    maximum = max(result.histogram_range[maximum_key] for result in method_results.values())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"Non-finite histogram range: [{minimum}, {maximum}].")
    if maximum <= minimum:
        width = max(abs(minimum) * 1.0e-6, 1.0e-12)
        minimum -= width
        maximum += width
    else:
        # Include extrema even if a backend reproduces the deterministic
        # second-pass FFT one ULP beyond the first-pass reduction.
        minimum = float(np.nextafter(minimum, -np.inf))
        maximum = float(np.nextafter(maximum, np.inf))
    return np.linspace(minimum, maximum, bins + 1, dtype=np.float64)


def collect_histograms(
    context: GeneratorContext,
    mode: str,
    method_results: Mapping[str, MethodResult],
) -> HistogramResult:
    """Deterministic second pass with exact shared histogram bins."""

    settings = context.settings
    magnitude_edges = _common_histogram_edges(
        method_results, "magnitude_min", "magnitude_max"
    )
    log_edges = _common_histogram_edges(
        method_results, "log_magnitude_min", "log_magnitude_max"
    )
    magnitude_all = {
        method: np.zeros((settings.channels, HISTOGRAM_BINS), dtype=np.int64)
        for method in METHODS
    }
    magnitude_valid = {
        method: np.zeros((settings.channels, HISTOGRAM_BINS), dtype=np.int64)
        for method in METHODS
    }
    log_all = {
        method: np.zeros((settings.channels, HISTOGRAM_BINS), dtype=np.int64)
        for method in METHODS
    }
    log_valid = {
        method: np.zeros((settings.channels, HISTOGRAM_BINS), dtype=np.int64)
        for method in METHODS
    }

    total_batches = math.ceil(settings.num_samples / settings.batch_size)
    for batch_index, (offset, method_a, method_b) in enumerate(iter_paired_batches(context, mode), start=1):
        for method, noise in (("method_a", method_a), ("method_b", method_b)):
            magnitude = fft_magnitude(noise).detach().cpu().to(torch.float64).numpy()
            log_magnitude = np.log10(magnitude + settings.eps)
            valid = method_results[method].valid_mask
            for channel in range(settings.channels):
                channel_magnitude = magnitude[:, channel]
                channel_log = log_magnitude[:, channel]
                magnitude_all[method][channel] += np.histogram(
                    channel_magnitude.reshape(-1), bins=magnitude_edges
                )[0]
                log_all[method][channel] += np.histogram(
                    channel_log.reshape(-1), bins=log_edges
                )[0]
                valid_values = channel_magnitude[:, valid[channel]].reshape(-1)
                valid_log_values = channel_log[:, valid[channel]].reshape(-1)
                magnitude_valid[method][channel] += np.histogram(
                    valid_values, bins=magnitude_edges
                )[0]
                log_valid[method][channel] += np.histogram(
                    valid_log_values, bins=log_edges
                )[0]
        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            completed = min(offset + method_a.shape[0], settings.num_samples)
            print(f"[{mode}] histograms {completed}/{settings.num_samples}", flush=True)
        del method_a, method_b

    for method in METHODS:
        for channel in range(settings.channels):
            expected_all = settings.num_samples * settings.height * settings.width
            expected_valid = settings.num_samples * int(method_results[method].valid_mask[channel].sum())
            observed = {
                "magnitude_all": int(magnitude_all[method][channel].sum()),
                "log_magnitude_all": int(log_all[method][channel].sum()),
                "magnitude_valid": int(magnitude_valid[method][channel].sum()),
                "log_magnitude_valid": int(log_valid[method][channel].sum()),
            }
            if observed["magnitude_all"] != expected_all or observed["log_magnitude_all"] != expected_all:
                raise RuntimeError(
                    f"Histogram lost all-bin values for {mode}/{method}/channel{channel}: "
                    f"expected={expected_all}, observed={observed}."
                )
            if observed["magnitude_valid"] != expected_valid or observed["log_magnitude_valid"] != expected_valid:
                raise RuntimeError(
                    f"Histogram lost valid-bin values for {mode}/{method}/channel{channel}: "
                    f"expected={expected_valid}, observed={observed}."
                )

    return HistogramResult(
        magnitude_edges=magnitude_edges,
        log_magnitude_edges=log_edges,
        magnitude_all=magnitude_all,
        magnitude_valid=magnitude_valid,
        log_magnitude_all=log_all,
        log_magnitude_valid=log_valid,
    )


def build_comparison(methods: Mapping[str, MethodResult], eps: float) -> dict[str, Any]:
    method_a = methods["method_a"]
    method_b = methods["method_b"]
    common_valid = method_a.valid_mask & method_b.valid_mask
    if not np.any(common_valid):
        common_valid = method_a.valid_mask | method_b.valid_mask
    method_a_values = method_a.fft_cv[common_valid]
    method_b_values = method_b.fft_cv[common_valid]
    method_a_median = float(np.median(method_a_values))
    method_b_median = float(np.median(method_b_values))

    return {
        "common_valid_bins": int(common_valid.sum()),
        "method_a_fft_cv_median": method_a_median,
        "method_b_fft_cv_median": method_b_median,
        "method_a_median_cv_over_method_b": method_a_median / max(method_b_median, eps),
        "method_b_median_cv_over_method_a": method_b_median / max(method_a_median, eps),
        "percentage_bins_method_b_cv_lower": float(np.mean(method_b_values < method_a_values) * 100.0),
        "method_a_fft_magnitude_cosine_median": float(np.median(method_a.cosine_fft)),
        "method_b_fft_magnitude_cosine_median": float(np.median(method_b.cosine_fft)),
        "method_a_pixel_skewness_median": float(np.median(method_a.pixel_skewness)),
        "method_b_pixel_skewness_median": float(np.median(method_b.pixel_skewness)),
        "method_a_pixel_kurtosis_median": float(np.median(method_a.pixel_kurtosis)),
        "method_b_pixel_kurtosis_median": float(np.median(method_b.pixel_kurtosis)),
        "method_a_radial_psd_cv_median": float(np.median(method_a.radial_cv)),
        "method_b_radial_psd_cv_median": float(np.median(method_b.radial_cv)),
    }


def classify_evidence(comparison: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "method_a_fft_cv_median",
        "method_b_fft_cv_median",
        "method_a_fft_magnitude_cosine_median",
        "method_b_fft_magnitude_cosine_median",
    )
    values = [comparison.get(key) for key in required]
    conclusive = all(value is not None and math.isfinite(float(value)) for value in values)
    if not conclusive or int(comparison.get("common_valid_bins", 0)) <= 0:
        return {
            "status": "INCONCLUSIVE",
            "reason": "Required metrics were unavailable, non-finite, or had no common valid bins.",
        }

    method_a_cv = float(comparison["method_a_fft_cv_median"])
    method_b_cv = float(comparison["method_b_fft_cv_median"])
    if method_a_cv < method_b_cv:
        lower_variation = "method_a"
    elif method_b_cv < method_a_cv:
        lower_variation = "method_b"
    else:
        lower_variation = "equal"
    return {
        "status": "MEASURED",
        "lower_fft_magnitude_variation": lower_variation,
        "method_a_fft_cv": method_a_cv,
        "method_b_fft_cv": method_b_cv,
        "cv_ratio_method_a_over_method_b": method_a_cv / max(method_b_cv, np.finfo(np.float64).tiny),
    }


def run_mode(context: GeneratorContext, mode: str) -> ModeResult:
    states = collect_streaming_statistics(context, mode)
    methods = {
        method: finalize_method_accumulator(states[method], context.settings)
        for method in METHODS
    }
    # Cosine subsets are the largest retained state (~200*C*H*W per domain).
    # Their pairwise values are now reduced into ``methods``, so release the
    # source tensors before the deterministic histogram pass.
    del states
    histograms = collect_histograms(context, mode, methods)
    comparison = build_comparison(methods, context.settings.eps)
    conclusion = classify_evidence(comparison)
    return ModeResult(
        name=mode,
        methods=methods,
        histograms=histograms,
        comparison=json_ready(comparison),
        conclusion=json_ready(conclusion),
    )


def method_summary_payload(result: MethodResult) -> dict[str, Any]:
    return {
        "fft_cv": result.cv_summary,
        "pixel_marginals": result.pixel_summary,
        "channel_correlations": result.correlation_summary,
        "sample_cosine": result.cosine_summary,
        "radial_psd": result.radial_summary,
        "histogram_range": result.histogram_range,
    }


def mode_summary_payload(result: ModeResult) -> dict[str, Any]:
    return {
        "methods": {
            method: method_summary_payload(result.methods[method]) for method in METHODS
        },
        "comparison": result.comparison,
        "conclusion": result.conclusion,
    }


def save_mode_arrays(output_dir: Path, result: ModeResult) -> list[Path]:
    mode_dir = output_dir / "modes" / result.name
    mode_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for method in METHODS:
        item = result.methods[method]
        path = mode_dir / f"{method}_statistics.npz"
        np.savez_compressed(
            path,
            fft_mean=item.fft_mean,
            fft_std=item.fft_std,
            fft_cv=item.fft_cv,
            valid_frequency_mask=item.valid_mask,
            pixel_mean=item.pixel_mean,
            pixel_std=item.pixel_std,
            pixel_skewness=item.pixel_skewness,
            pixel_pearson_kurtosis=item.pixel_kurtosis,
            spatial_channel_correlation_mean=item.spatial_corr_mean,
            spatial_channel_correlation_std=item.spatial_corr_std,
            fft_channel_correlation_mean=item.fft_corr_mean,
            fft_channel_correlation_std=item.fft_corr_std,
            spatial_off_diagonal=item.spatial_off_diagonal,
            fft_off_diagonal=item.fft_off_diagonal,
            cosine_spatial=item.cosine_spatial,
            cosine_fft_magnitude=item.cosine_fft,
            radial_psd_mean=item.radial_mean,
            radial_psd_std=item.radial_std,
            radial_psd_cv=item.radial_cv,
            radial_psd_examples=item.radial_examples,
            noise_examples=item.noise_examples,
            fft_magnitude_examples=item.fft_examples,
        )
        paths.append(path)

    hist_path = mode_dir / "fft_magnitude_histograms.npz"
    np.savez_compressed(
        hist_path,
        magnitude_edges=result.histograms.magnitude_edges,
        log_magnitude_edges=result.histograms.log_magnitude_edges,
        method_a_magnitude_all=result.histograms.magnitude_all["method_a"],
        method_b_magnitude_all=result.histograms.magnitude_all["method_b"],
        method_a_magnitude_valid=result.histograms.magnitude_valid["method_a"],
        method_b_magnitude_valid=result.histograms.magnitude_valid["method_b"],
        method_a_log_magnitude_all=result.histograms.log_magnitude_all["method_a"],
        method_b_log_magnitude_all=result.histograms.log_magnitude_all["method_b"],
        method_a_log_magnitude_valid=result.histograms.log_magnitude_valid["method_a"],
        method_b_log_magnitude_valid=result.histograms.log_magnitude_valid["method_b"],
    )
    paths.append(hist_path)

    summary_path = mode_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_ready(mode_summary_payload(result)), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    paths.append(summary_path)
    return paths


def save_primary_arrays(output_dir: Path, result: ModeResult) -> list[Path]:
    method_a = result.methods["method_a"]
    method_b = result.methods["method_b"]
    outputs = {
        "fft_cv_method_a.npy": method_a.fft_cv,
        "fft_cv_method_b.npy": method_b.fft_cv,
        "fft_mean_method_a.npy": method_a.fft_mean,
        "fft_mean_method_b.npy": method_b.fft_mean,
    }
    paths: list[Path] = []
    for filename, array in outputs.items():
        path = output_dir / filename
        np.save(path, array, allow_pickle=False)
        paths.append(path)

    radial_path = output_dir / "radial_psd_statistics.npz"
    np.savez_compressed(
        radial_path,
        comparison_mode=np.asarray(result.name),
        method_a_mean=method_a.radial_mean,
        method_a_std=method_a.radial_std,
        method_a_cv=method_a.radial_cv,
        method_b_mean=method_b.radial_mean,
        method_b_std=method_b.radial_std,
        method_b_cv=method_b.radial_cv,
        method_a_sample_examples=method_a.radial_examples,
        method_b_sample_examples=method_b.radial_examples,
    )
    paths.append(radial_path)
    return paths


def _metric_rows_for_summary(
    rows: list[dict[str, Any]],
    mode: str,
    category: str,
    method: str,
    scope: str,
    summary: Mapping[str, Any],
    channel: int | str = "",
) -> None:
    for metric, value in summary.items():
        if isinstance(value, (bool, int, float, np.generic)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                rows.append(
                    {
                        "mode": mode,
                        "category": category,
                        "method": method,
                        "scope": scope,
                        "channel": channel,
                        "metric": metric,
                        "value": numeric,
                    }
                )


def build_metrics_rows(results: Mapping[str, ModeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, mode_result in results.items():
        for method in METHODS:
            item = mode_result.methods[method]
            _metric_rows_for_summary(
                rows, mode, "fft_cv", method, "all_bins", item.cv_summary["all"]
            )
            _metric_rows_for_summary(
                rows, mode, "fft_cv", method, "valid_bins", item.cv_summary["valid"]
            )
            _metric_rows_for_summary(
                rows, mode, "fft_cv", method, "dc", item.cv_summary["dc"]
            )
            for channel_payload in item.cv_summary["per_channel"]:
                channel = int(channel_payload["channel"])
                _metric_rows_for_summary(
                    rows,
                    mode,
                    "fft_cv",
                    method,
                    "valid_bins",
                    channel_payload["valid"],
                    channel,
                )
            for field_name, category in (
                ("skewness", "pixel_skewness"),
                ("pearson_kurtosis", "pixel_pearson_kurtosis"),
            ):
                payload = item.pixel_summary[field_name]
                _metric_rows_for_summary(rows, mode, category, method, "pooled", payload["pooled"])
                for channel_payload in payload["per_channel"]:
                    channel = int(channel_payload["channel"])
                    values = {key: value for key, value in channel_payload.items() if key != "channel"}
                    _metric_rows_for_summary(
                        rows, mode, category, method, "fixed_locations", values, channel
                    )
            for key, category in (
                ("spatial_off_diagonal", "channel_correlation_spatial"),
                ("fft_magnitude_off_diagonal", "channel_correlation_fft_magnitude"),
            ):
                _metric_rows_for_summary(
                    rows, mode, category, method, "off_diagonal", item.correlation_summary[key]
                )
            for key, category in (
                ("spatial", "sample_cosine_spatial"),
                ("fft_magnitude", "sample_cosine_fft_magnitude"),
            ):
                _metric_rows_for_summary(
                    rows, mode, category, method, "pairwise_no_diagonal", item.cosine_summary[key]
                )
            radial_payload = item.radial_summary["cv"]
            _metric_rows_for_summary(
                rows, mode, "radial_psd_cv", method, "pooled", radial_payload["pooled"]
            )
            for channel_payload in radial_payload["per_channel"]:
                channel = int(channel_payload["channel"])
                values = {key: value for key, value in channel_payload.items() if key != "channel"}
                _metric_rows_for_summary(
                    rows, mode, "radial_psd_cv", method, "radial_bins", values, channel
                )

        for metric, value in mode_result.comparison.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                rows.append(
                    {
                        "mode": mode,
                        "category": "comparison",
                        "method": "method_a_vs_method_b",
                        "scope": "common_valid_bins",
                        "channel": "",
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ("mode", "category", "method", "scope", "channel", "metric", "value")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(fig: Any, path: Path) -> None:
    if _PLOT_METADATA_FOOTER:
        fig.text(
            0.5,
            -0.035,
            _PLOT_METADATA_FOOTER,
            ha="center",
            va="top",
            fontsize=6,
        )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.18)
    _pyplot().close(fig)


def _finite_plot_range(*arrays: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [lower, upper])
    if high <= low:
        delta = max(abs(float(low)) * 1.0e-6, 1.0e-9)
        return float(low - delta), float(high + delta)
    return float(low), float(high)


def _density_from_counts(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    widths = np.diff(edges)
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / (total * widths)


def plot_fft_cv_maps(path: Path, result: ModeResult) -> None:
    plt = _pyplot()
    channels = result.methods["method_a"].fft_cv.shape[0]
    fig, axes = plt.subplots(2, channels, figsize=(3.2 * channels, 6.0), squeeze=False)
    valid_values = np.concatenate(
        [
            item.fft_cv[item.valid_mask]
            for item in result.methods.values()
            if np.any(item.valid_mask)
        ]
    )
    vmax = max(float(np.percentile(valid_values, 99.0)), 1.0e-6)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="0.65")
    image = None
    for row, method in enumerate(METHODS):
        item = result.methods[method]
        for channel in range(channels):
            display = np.where(item.valid_mask[channel], item.fft_cv[channel], np.nan)
            image = axes[row, channel].imshow(
                np.fft.fftshift(display),
                origin="lower",
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, channel].set_title(f"{method} · channel {channel}")
            axes[row, channel].set_xticks([])
            axes[row, channel].set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="FFT magnitude CV")
    fig.suptitle(f"FFT magnitude CV maps (fftshift; gray = invalid/DC) · {result.name}")
    _finish_figure(fig, path)


def plot_fft_cv_histogram(path: Path, result: ModeResult) -> None:
    plt = _pyplot()
    method_a = result.methods["method_a"]
    method_b = result.methods["method_b"]
    values = [method_a.fft_cv[method_a.valid_mask], method_b.fft_cv[method_b.valid_mask]]
    low, high = _finite_plot_range(*values, lower=0.0, upper=99.9)
    edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method, method_values in zip(METHODS, values):
        ax.hist(method_values, bins=edges, density=True, histtype="step", linewidth=1.8, label=method)
    ax.set_xlabel("CV = population std / (mean + eps guard)")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title(f"Valid-bin FFT magnitude CV · {result.name}")
    ax.legend()
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def _radial_average_fields(fields: np.ndarray, bin_map: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.float64)
    indices = np.asarray(getattr(bin_map, "indices", bin_map), dtype=np.int64)
    bins = int(indices.max()) + 1
    profiles = np.empty((fields.shape[0], bins), dtype=np.float64)
    for channel in range(fields.shape[0]):
        finite = np.isfinite(fields[channel])
        sums = np.bincount(
            indices.ravel(),
            weights=np.where(finite, fields[channel], 0.0).ravel(),
            minlength=bins,
        ).astype(np.float64)
        valid_counts = np.bincount(
            indices.ravel(), weights=finite.astype(np.float64).ravel(), minlength=bins
        ).astype(np.float64)
        profiles[channel] = np.divide(
            sums,
            valid_counts,
            out=np.full_like(sums, np.nan),
            where=valid_counts > 0,
        )
    return profiles


def plot_fft_cv_radial_profile(path: Path, result: ModeResult, radial_map: Any) -> None:
    plt = _pyplot()
    indices = np.asarray(getattr(radial_map, "indices", radial_map), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method in METHODS:
        item = result.methods[method]
        valid_cv = np.where(item.valid_mask, item.fft_cv, np.nan)
        profiles = _radial_average_fields(valid_cv, indices)
        usable = np.all(np.isfinite(profiles), axis=0)
        profiles = profiles[:, usable]
        mean = profiles.mean(axis=0)
        std = profiles.std(axis=0, ddof=0)
        x = np.flatnonzero(usable)
        ax.plot(x, mean, linewidth=1.8, label=method)
        ax.fill_between(x, np.maximum(mean - std, 0.0), mean + std, alpha=0.18)
    ax.set_xlabel("Radial frequency bin (integer FFT-pixel radius)")
    ax.set_ylabel("FFT magnitude CV")
    ax.set_title(f"Radial valid-bin FFT-CV profile (channel mean ± std) · {result.name}")
    ax.legend()
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def plot_field_histogram(
    path: Path,
    result: ModeResult,
    attribute: str,
    xlabel: str,
    title: str,
    reference: float | None = None,
) -> None:
    plt = _pyplot()
    arrays = [getattr(result.methods[method], attribute).reshape(-1) for method in METHODS]
    low, high = _finite_plot_range(*arrays, lower=0.1, upper=99.9)
    edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method, values in zip(METHODS, arrays):
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=1.8, label=method)
    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=1.2, label=f"Gaussian={reference:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density across fixed (channel,y,x) locations")
    ax.set_title(f"{title} · {result.name}")
    ax.legend()
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def plot_fft_magnitude_histogram(path: Path, result: ModeResult, log_values: bool) -> None:
    plt = _pyplot()
    hist = result.histograms
    edges = hist.log_magnitude_edges if log_values else hist.magnitude_edges
    all_counts = hist.log_magnitude_all if log_values else hist.magnitude_all
    valid_counts = hist.log_magnitude_valid if log_values else hist.magnitude_valid
    channels = result.methods["method_a"].fft_mean.shape[0]
    fig, axes = plt.subplots(2, channels + 1, figsize=(3.2 * (channels + 1), 6.2), squeeze=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for row, counts_by_method in enumerate(
        (all_counts, valid_counts)
    ):
        for method in METHODS:
            pooled = counts_by_method[method].sum(axis=0)
            axes[row, 0].plot(centers, _density_from_counts(pooled, edges), label=method)
            for channel in range(channels):
                axes[row, channel + 1].plot(
                    centers,
                    _density_from_counts(counts_by_method[method][channel], edges),
                    label=method,
                )
        axes[row, 0].set_title("Pooled")
        for channel in range(channels):
            axes[row, channel + 1].set_title(f"Channel {channel}")
        for axis in axes[row]:
            axis.set_yscale("log")
            axis.grid(alpha=0.15)
            axis.legend(fontsize=8)
    label = "log10(|FFT| + eps)" if log_values else "|FFT|"
    for axis in axes[-1]:
        axis.set_xlabel(label)
    axes[0, 0].set_ylabel("All bins\n(includes DC)\nDensity")
    axes[1, 0].set_ylabel("Valid bins\n(near-zero/DC masked)\nDensity")
    fig.suptitle(f"FFT magnitude distributions · {result.name}")
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.28)
    _finish_figure(fig, path)


def plot_correlation_matrices(path: Path, result: ModeResult, fft: bool) -> None:
    plt = _pyplot()
    mean_attr = "fft_corr_mean" if fft else "spatial_corr_mean"
    std_attr = "fft_corr_std" if fft else "spatial_corr_std"
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2), squeeze=False)
    images = []
    for row, method in enumerate(METHODS):
        mean = getattr(result.methods[method], mean_attr)
        std = getattr(result.methods[method], std_attr)
        images.append(axes[row, 0].imshow(mean, vmin=-1.0, vmax=1.0, cmap="coolwarm"))
        images.append(axes[row, 1].imshow(std, vmin=0.0, cmap="viridis"))
        axes[row, 0].set_title(f"{method} mean")
        axes[row, 1].set_title(f"{method} population std")
        for column, matrix in enumerate((mean, std)):
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    axes[row, column].text(
                        x,
                        y,
                        f"{matrix[y, x]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
            axes[row, column].set_xlabel("channel")
            axes[row, column].set_ylabel("channel")
    fig.colorbar(images[0], ax=axes[:, 0].ravel().tolist(), shrink=0.75, label="Correlation")
    fig.colorbar(images[-1], ax=axes[:, 1].ravel().tolist(), shrink=0.75, label="Std")
    domain = "FFT magnitude" if fft else "spatial noise"
    fig.suptitle(f"Per-sample channel correlation: {domain} · {result.name}")
    _finish_figure(fig, path)


def plot_cosine_histogram(path: Path, result: ModeResult, fft: bool) -> None:
    plt = _pyplot()
    attribute = "cosine_fft" if fft else "cosine_spatial"
    arrays = [getattr(result.methods[method], attribute) for method in METHODS]
    low, high = _finite_plot_range(*arrays, lower=0.0, upper=100.0)
    edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method, values in zip(METHODS, arrays):
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=1.8, label=method)
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Density")
    domain = "FFT magnitude" if fft else "spatial noise"
    ax.set_title(f"Sample-to-sample cosine ({domain}; diagonal excluded) · {result.name}")
    ax.legend()
    ax.grid(alpha=0.2)
    _finish_figure(fig, path)


def plot_radial_mean_std(path: Path, result: ModeResult) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 7.5), sharex=True, sharey=True, squeeze=False
    )
    for row, method in enumerate(METHODS):
        item = result.methods[method]
        axis = axes[row, 0]
        for channel in range(item.radial_mean.shape[0]):
            mean = item.radial_mean[channel]
            std = item.radial_std[channel]
            x = np.arange(mean.size)
            axis.plot(x, mean, linewidth=1.4, label=f"channel {channel}")
            axis.fill_between(x, np.maximum(mean - std, 0.0), mean + std, alpha=0.12)
        axis.set_yscale("log")
        axis.set_ylabel("Radial PSD")
        axis.set_title(f"{method}: mean ± population std across samples")
        axis.grid(alpha=0.2)
        axis.legend(ncol=min(4, item.radial_mean.shape[0]), fontsize=8)
    axes[-1, 0].set_xlabel("Radial frequency bin (integer FFT-pixel radius)")
    fig.suptitle(f"Radial PSD statistics · {result.name}")
    _finish_figure(fig, path)


def plot_radial_cv(path: Path, result: ModeResult) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    styles = {"method_a": "-", "method_b": "--"}
    for method in METHODS:
        values = result.methods[method].radial_cv
        for channel in range(values.shape[0]):
            ax.plot(
                np.arange(values.shape[-1]),
                values[channel],
                linestyle=styles[method],
                linewidth=1.2,
                label=f"{method} c{channel}",
            )
    ax.set_xlabel("Radial frequency bin (integer FFT-pixel radius)")
    ax.set_ylabel("Radial PSD CV")
    ax.set_title(f"Radial PSD sample variation · {result.name}")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    _finish_figure(fig, path)


def plot_radial_examples(path: Path, result: ModeResult) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 7.5), sharex=True, sharey=True, squeeze=False
    )
    for row, method in enumerate(METHODS):
        examples = result.methods[method].radial_examples
        axis = axes[row, 0]
        profiles = examples.mean(axis=1)
        for profile in profiles:
            axis.plot(profile, color="tab:blue", alpha=0.18, linewidth=0.8)
        if profiles.size:
            axis.plot(profiles.mean(axis=0), color="black", linewidth=2.0, label="sample mean")
        axis.set_yscale("log")
        axis.set_ylabel("Radial PSD")
        axis.set_title(
            f"{method}: {profiles.shape[0]} seeded-random samples (channel average)"
        )
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1, 0].set_xlabel("Radial frequency bin (integer FFT-pixel radius)")
    fig.suptitle(f"Individual radial PSD examples · {result.name}")
    _finish_figure(fig, path)


def plot_sample_grid(path: Path, result: ModeResult, fft: bool, eps: float) -> None:
    plt = _pyplot()
    example_count = min(
        result.methods["method_a"].noise_examples.shape[0],
        result.methods["method_b"].noise_examples.shape[0],
        NOISE_EXAMPLE_COUNT,
    )
    fig, axes = plt.subplots(2, example_count, figsize=(3.0 * example_count, 6.0), squeeze=False)
    selected: list[np.ndarray] = []
    for method in METHODS:
        item = result.methods[method]
        values = item.fft_examples if fft else item.noise_examples
        for column in range(example_count):
            channel = column % values.shape[1]
            display = values[column, channel]
            selected.append(
                np.fft.fftshift(np.log10(display + eps)) if fft else display
            )
    if fft:
        vmin, vmax = _finite_plot_range(*selected, lower=1.0, upper=99.0)
    else:
        limit = max(float(np.percentile(np.abs(np.concatenate([x.reshape(-1) for x in selected])), 99.0)), eps)
        vmin, vmax = -limit, limit
    image = None
    for row, method in enumerate(METHODS):
        item = result.methods[method]
        values = item.fft_examples if fft else item.noise_examples
        for column in range(example_count):
            channel = column % values.shape[1]
            display = values[column, channel]
            if fft:
                display = np.fft.fftshift(np.log10(display + eps))
                cmap = "magma"
            else:
                cmap = "coolwarm"
            image = axes[row, column].imshow(
                display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[row, column].set_title(f"{method} · sample {column} · c{channel}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78)
    domain = "log10 FFT magnitude (fftshift)" if fft else "spatial noise"
    fig.suptitle(f"Deterministic sample examples: {domain} · {result.name}")
    _finish_figure(fig, path)


def create_plots(
    output_dir: Path,
    result: ModeResult,
    radial_map: Any,
    eps: float,
    settings: RunSettings,
) -> list[Path]:
    global _PLOT_METADATA_FOOTER
    _PLOT_METADATA_FOOTER = (
        f"N={settings.num_samples} | seed={settings.seed} | "
        f"shape=[{settings.channels},{settings.height},{settings.width}] | mode={result.name}\n"
        f"stats={settings.stats_path} | method_a={settings.method_a} | method_b={settings.method_b}"
    )
    paths: list[Path] = []

    def emit(filename: str, function: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        path = output_dir / filename
        function(path, *args, **kwargs)
        paths.append(path)

    emit("fft_cv_maps.png", plot_fft_cv_maps, result)
    emit("fft_cv_histogram.png", plot_fft_cv_histogram, result)
    emit("fft_cv_radial_profile.png", plot_fft_cv_radial_profile, result, radial_map)
    emit(
        "pixel_skewness_histogram.png",
        plot_field_histogram,
        result,
        "pixel_skewness",
        "Skewness",
        "Fixed-location pixel skewness",
        0.0,
    )
    emit(
        "pixel_kurtosis_histogram.png",
        plot_field_histogram,
        result,
        "pixel_kurtosis",
        "Pearson kurtosis (fisher=False)",
        "Fixed-location pixel Pearson kurtosis",
        3.0,
    )
    emit("fft_magnitude_histogram.png", plot_fft_magnitude_histogram, result, False)
    emit("log_fft_magnitude_histogram.png", plot_fft_magnitude_histogram, result, True)
    emit("channel_correlation_spatial.png", plot_correlation_matrices, result, False)
    emit("channel_correlation_fft.png", plot_correlation_matrices, result, True)
    emit("sample_cosine_spatial.png", plot_cosine_histogram, result, False)
    emit("sample_cosine_fft_magnitude.png", plot_cosine_histogram, result, True)
    emit("radial_psd_mean_std.png", plot_radial_mean_std, result)
    emit("radial_psd_cv.png", plot_radial_cv, result)
    emit("radial_psd_sample_examples.png", plot_radial_examples, result)
    emit("noise_sample_grid.png", plot_sample_grid, result, False, eps)
    emit("noise_fft_sample_grid.png", plot_sample_grid, result, True, eps)
    _PLOT_METADATA_FOOTER = None
    return paths


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.6e}"
    return f"{number:.6f}"


def _stats_key_table(metadata: Mapping[str, Any] | None, label: str) -> list[str]:
    if metadata is None:
        return [f"| {label} | n/a | n/a | n/a |"]
    rows: list[str] = []
    for key, entry in metadata["keys"].items():
        shape = "scalar" if not entry["shape"] else " x ".join(str(v) for v in entry["shape"])
        value = entry.get("value", "")
        rows.append(f"| {label} | `{key}` | `{shape}` / `{entry['dtype']}` | {value} |")
    return rows


def write_report(
    path: Path,
    settings: RunSettings,
    environment: Mapping[str, Any],
    stats_metadata: Mapping[str, Any],
    results: Mapping[str, ModeResult],
    primary_mode: str,
    artifacts: Sequence[Path],
) -> None:
    primary = results[primary_mode]
    lines = [
        "# Empirical spectrum noise diagnostic",
        "",
        f"**Result status: `{primary.conclusion['status']}`**",
        "",
        "Both branches call the production `EmpiricalSpectrumNoise.sample` implementation. "
        "The measurements below determine the direction of each comparison; no expected result is hard-coded.",
        "",
        "## Run settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Git commit | `{environment.get('git_commit') or 'unavailable'}` |",
        f"| Python | `{environment.get('python')}` |",
        f"| PyTorch | `{environment.get('pytorch')}` |",
        f"| CUDA runtime | `{environment.get('cuda_runtime')}` |",
        f"| Device | `{environment.get('device')}` |",
        f"| Seed | `{settings.seed}` |",
        f"| Samples / batch size | `{settings.num_samples}` / `{settings.batch_size}` |",
        f"| Noise shape | `[{settings.num_samples}, {settings.channels}, {settings.height}, {settings.width}]` |",
        f"| Cosine subset | `{settings.cosine_samples}` samples; diagonal excluded |",
        f"| Radial example subset | seeded-random indices `{list(select_radial_example_indices(settings.num_samples, settings.seed))}` |",
        f"| Epsilon | `{settings.eps:.8g}` |",
        f"| Shared stats path | `{settings.stats_path}` |",
        f"| Method A | `{settings.method_a}` |",
        f"| Method B | `{settings.method_b}` |",
        f"| Synthetic fixture | `{settings.synthetic}` |",
        f"| Modes run | `{', '.join(results)}` |",
        f"| Primary mode | `{primary_mode}` |",
        "| FFT convention | `torch.fft.fft2`, unshifted for metrics; `fftshift` only for display |",
        "| Sample normalization | per sample, per channel, spatial zero mean and population unit std |",
        "",
        "## Production formulas",
        "",
        f"Production class: `{source_location(EmpiricalSpectrumNoise)}`; sample method: "
        f"`{source_location(EmpiricalSpectrumNoise.sample)}`.",
        "",
        "### `fixed_magnitude`",
        "",
        "```text",
        "white_fft = fft2(white)",
        "phase = white_fft / (abs(white_fft) + eps)",
        "shaped_amp = (abs(white_fft)+eps) ** (1-strength) * (target_amp+eps) ** strength",
        "shaped = ifft2(phase * shaped_amp).real",
        "```",
        "",
        "At strength 1, the empirical target replaces the draw-specific FFT magnitude while phase stays random.",
        "",
        "### `filtered_gaussian`",
        "",
        "```text",
        "white_fft = rfft2(white)",
        "filter_amp = rms_normalize(sqrt(empirical_power))",
        "effective_filter = clamp(filter_amp, min=eps) ** strength",
        "shaped = irfft2(white_fft * effective_filter)",
        "```",
        "",
        "This preserves both the random FFT magnitude and phase of the original Gaussian draw.",
        "",
        "## Spectrum-statistics keys",
        "",
        "| Source | Key | Shape / dtype | Scalar value |",
        "|---|---|---|---|",
    ]
    lines.extend(_stats_key_table(stats_metadata["shared"], "shared"))
    lines.extend(
        [
            "",
            "## Key measured results",
            "",
            "| Mode | A FFT CV | B FFT CV | A/B CV | A FFT cosine | B FFT cosine | A skew | B skew | A kurtosis | B kurtosis | A radial PSD CV | B radial PSD CV | Lower variation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for mode, result in results.items():
        values = result.comparison
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    _format_metric(values.get("method_a_fft_cv_median")),
                    _format_metric(values.get("method_b_fft_cv_median")),
                    _format_metric(values.get("method_a_median_cv_over_method_b")),
                    _format_metric(values.get("method_a_fft_magnitude_cosine_median")),
                    _format_metric(values.get("method_b_fft_magnitude_cosine_median")),
                    _format_metric(values.get("method_a_pixel_skewness_median")),
                    _format_metric(values.get("method_b_pixel_skewness_median")),
                    _format_metric(values.get("method_a_pixel_kurtosis_median")),
                    _format_metric(values.get("method_b_pixel_kurtosis_median")),
                    _format_metric(values.get("method_a_radial_psd_cv_median")),
                    _format_metric(values.get("method_b_radial_psd_cv_median")),
                    f"`{result.conclusion['lower_fft_magnitude_variation']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Method labels map to the requested production methods: "
            f"`method_a={settings.method_a}`, `method_b={settings.method_b}`.",
            "",
        ]
    )

    lines.extend(
        [
            "## Metric conventions",
            "",
            "- FFT magnitude CV is `population_std_across_samples / (mean + eps guard)`.",
            "- Valid bins satisfy `mean_mag[c,y,x] > max(mean_mag[c]) * 1e-6`.",
            "- All-bin and DC summaries remain in JSON/CSV; valid-bin summaries drive the key CV comparison.",
            "- Pixel skewness and Pearson kurtosis are computed across samples at each fixed `(channel,y,x)` location; Gaussian Pearson kurtosis is 3.",
            "- Channel correlations are computed per sample and then summarized; reported off-diagonal distributions use the strict upper triangle.",
            "- Pairwise cosine distributions exclude diagonal self-similarity.",
            "- Radial PSD is the radial mean of `abs(fft2(noise)) ** 2` in integer FFT-pixel-radius bins; its CV is across samples.",
            "- Individual radial curves use the same seeded-random sample indices for both methods.",
            "- Histogram edges are shared by method A/B and obtained from a deterministic second pass.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact in sorted(artifacts, key=lambda item: str(item)):
        try:
            shown = artifact.relative_to(settings.output_dir)
        except ValueError:
            shown = artifact
        lines.append(f"- `{shown}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary_payload(
    settings: RunSettings,
    environment: Mapping[str, Any],
    stats_metadata: Mapping[str, Any],
    results: Mapping[str, ModeResult],
    primary_mode: str,
    artifacts: Sequence[Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "primary_mode": primary_mode,
        "conclusion": results[primary_mode].conclusion,
        "settings": {
            "seed": settings.seed,
            "num_samples": settings.num_samples,
            "batch_size": settings.batch_size,
            "channels": settings.channels,
            "height": settings.height,
            "width": settings.width,
            "cosine_samples": settings.cosine_samples,
            "radial_example_indices": list(
                select_radial_example_indices(settings.num_samples, settings.seed)
            ),
            "eps": settings.eps,
            "device": str(settings.device),
            "comparison_mode": settings.comparison_mode,
            "synthetic": settings.synthetic,
            "plots_enabled": not settings.no_plots,
            "stats_path": str(settings.stats_path),
            "method_a": settings.method_a,
            "method_b": settings.method_b,
            "normalization": {
                "method_a": "production per-sample/per-channel mean; population std + eps",
                "method_b": "production per-sample/per-channel mean; population std + eps",
            },
        },
        "environment": environment,
        "stats": stats_metadata,
        "code_locations": {
            "production_empirical_spectrum": source_location(EmpiricalSpectrumNoise.sample),
        },
        "modes": {mode: mode_summary_payload(result) for mode, result in results.items()},
        "artifacts": [str(path.relative_to(settings.output_dir)) for path in artifacts],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args)
    print(f"stats_path={settings.stats_path}")
    print(f"method_a={settings.method_a}")
    print(f"method_b={settings.method_b}")
    print(f"device={settings.device}")
    print(f"output_dir={settings.output_dir}")

    context = create_generator_context(settings)
    results: dict[str, ModeResult] = {}
    for mode in comparison_modes(settings.comparison_mode):
        print(f"Running comparison mode: {mode}", flush=True)
        results[mode] = run_mode(context, mode)

    primary_mode = MODE_ACTUAL
    primary = results[primary_mode]
    artifacts: list[Path] = []
    for result in results.values():
        artifacts.extend(save_mode_arrays(settings.output_dir, result))
    artifacts.extend(save_primary_arrays(settings.output_dir, primary))

    if not settings.no_plots:
        artifacts.extend(
            create_plots(
                settings.output_dir,
                primary,
                context.radial_bin_map,
                settings.eps,
                settings,
            )
        )

    metrics_path = settings.output_dir / "metrics.csv"
    write_metrics_csv(metrics_path, build_metrics_rows(results))
    artifacts.append(metrics_path)

    summary_path = settings.output_dir / "summary.json"
    report_path = settings.output_dir / "report.md"
    artifacts.extend((summary_path, report_path))

    environment = environment_metadata(settings)
    stats_metadata = {
        "shared": npz_metadata(settings.stats_path),
    }
    summary = build_summary_payload(
        settings=settings,
        environment=environment,
        stats_metadata=stats_metadata,
        results=results,
        primary_mode=primary_mode,
        artifacts=artifacts,
    )
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        path=report_path,
        settings=settings,
        environment=environment,
        stats_metadata=stats_metadata,
        results=results,
        primary_mode=primary_mode,
        artifacts=artifacts,
    )

    print(f"primary_mode={primary_mode}")
    print(f"conclusion={primary.conclusion['status']}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
