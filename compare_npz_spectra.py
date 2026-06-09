from __future__ import annotations

import argparse
from pathlib import Path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one or more empirical MRI spectrum .npz files by radial profile and synthesized noise."
    )
    parser.add_argument(
        "--spectrum-stats-paths",
        nargs="+",
        required=True,
        help="One or more .npz files containing radial spectrum statistics.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels corresponding to --spectrum-stats-paths. Defaults to each file stem.",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/BraTS21/healthy_slices",
        help="Path to the LMDB healthy-slice dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="LMDB slice index to visualize. Defaults to the middle slice.",
    )
    parser.add_argument("--channel", type=int, default=0, help="MRI channel index to visualize.")
    parser.add_argument(
        "--output-dir",
        default="results/npz_spectrum_compare",
        help="Directory where comparison images and metadata will be written.",
    )
    parser.add_argument("--seed", type=int, default=73, help="Base random seed for reproducible noise generation.")
    parser.add_argument("--noise-steps", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--timestep", type=int, default=150, help="Diffusion timestep for forward noising.")
    parser.add_argument("--beta-start", type=float, default=1.0e-4, help="DDPM linear beta schedule start.")
    parser.add_argument("--beta-end", type=float, default=0.02, help="DDPM linear beta schedule end.")
    parser.add_argument(
        "--spectrum-mix-white",
        type=float,
        default=0.0,
        help="White-noise mixing ratio after spectrum shaping. 0.0 means fully spectrum-shaped.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    parser.add_argument(
        "--same-base-noise",
        type=str2bool,
        default=True,
        help="Reuse the same white Gaussian base noise across all .npz files for fair comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")

    import numpy as np
    import torch
    from matplotlib import pyplot as plt

    from utils.seed import set_seed
    from utils.spectrum_compare import (
        apply_forward_noise,
        linear_alpha_hat,
        load_lmdb_slice,
        load_radial_psd,
        log_spectrum_map,
        radial_power_profile,
        sanitize_label,
        save_gray,
        select_radial_profile,
        spectrum_noise_from_white,
        spectrum_noise_like,
        to_display_image,
        write_metadata,
    )

    if args.labels and len(args.labels) != len(args.spectrum_stats_paths):
        raise ValueError("--labels count must match the number of --spectrum-stats-paths.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(path).stem for path in args.spectrum_stats_paths]
    items = [load_radial_psd(path, label) for path, label in zip(args.spectrum_stats_paths, labels)]

    slice_tensor, resolved_index, dataset_length = load_lmdb_slice(args.dataset_path, args.index)
    if slice_tensor.ndim != 3:
        raise ValueError(f"Expected LMDB slice tensor with shape [C, H, W], got {tuple(slice_tensor.shape)}.")
    if args.channel < 0 or args.channel >= int(slice_tensor.shape[0]):
        raise ValueError(
            f"--channel {args.channel} is out of range for MRI slice with {int(slice_tensor.shape[0])} channels."
        )

    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    image = (slice_tensor.unsqueeze(0).float() * 2.0) - 1.0
    image_channel = image[:, args.channel : args.channel + 1]
    height, width = map(int, image_channel.shape[-2:])
    alpha_hat = linear_alpha_hat(
        int(args.timestep),
        noise_steps=int(args.noise_steps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
    )

    if bool(args.same_base_noise):
        base_generator = torch.Generator(device="cpu")
        base_generator.manual_seed(int(args.seed))
        base_white = torch.randn((1, 1, height, width), generator=base_generator, dtype=torch.float32)
    else:
        base_white = None

    original_display = to_display_image(image_channel[0, 0], "mri")
    original_spectrum_display = log_spectrum_map(image_channel[0, 0].numpy())
    original_profile = radial_power_profile(image_channel[0, 0].numpy())

    noise_results: list[dict[str, object]] = []
    for item_index, item in enumerate(items):
        radial_profile = select_radial_profile(item, int(args.channel))
        if base_white is not None:
            noise = spectrum_noise_from_white(base_white.clone(), radial_profile, float(args.spectrum_mix_white))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.seed) + item_index)
            noise = spectrum_noise_like(
                (1, 1, height, width),
                radial_profile,
                float(args.spectrum_mix_white),
                generator=generator,
            )
        noised = apply_forward_noise(image_channel, noise, alpha_hat)
        noise_display = to_display_image(noise[0, 0], "noise")
        noised_display = to_display_image(noised[0, 0], "mri")
        safe_label = sanitize_label(item.label)
        noise_path = output_dir / f"noise_{safe_label}.png"
        noised_path = output_dir / f"noised_mri_{safe_label}.png"
        save_gray(noise_path, noise_display)
        save_gray(noised_path, noised_display)
        noise_results.append(
            {
                "item": item,
                "profile": radial_profile,
                "noise_display": noise_display,
                "noised_display": noised_display,
                "noise_path": noise_path.name,
                "noised_path": noised_path.name,
            }
        )

    radial_compare_path = output_dir / "radial_spectrum_compare.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    ax.plot(np.log10(original_profile + 1.0e-12), label="Original MRI", linewidth=2.0, linestyle="--")
    for result in noise_results:
        item = result["item"]
        profile = np.asarray(result["profile"], dtype=np.float32)
        ax.plot(np.log10(profile + 1.0e-12), label=str(item.label), linewidth=2.0)
    ax.set_title(f"Radial Spectrum Comparison (channel {args.channel}, {len(items)} item(s))")
    ax.set_xlabel("Radius bin")
    ax.set_ylabel("log10(power + 1e-12)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(radial_compare_path, dpi=args.dpi)
    plt.close(fig)

    cols = max(1, len(noise_results))

    noise_grid_path = output_dir / "noise_compare_grid.png"
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), dpi=args.dpi, squeeze=False)
    for axis, result in zip(axes[0], noise_results):
        axis.imshow(np.asarray(result["noise_display"]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(str(result["item"].label))
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(noise_grid_path, dpi=args.dpi)
    plt.close(fig)

    noised_grid_path = output_dir / "noised_mri_compare_grid.png"
    fig, axes = plt.subplots(1, cols + 1, figsize=(4 * (cols + 1), 4), dpi=args.dpi, squeeze=False)
    axes[0, 0].imshow(original_display, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Original MRI")
    axes[0, 0].axis("off")
    for axis, result in zip(axes[0, 1:], noise_results):
        axis.imshow(np.asarray(result["noised_display"]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(str(result["item"].label))
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(noised_grid_path, dpi=args.dpi)
    plt.close(fig)

    dashboard_path = output_dir / "npz_spectrum_compare_dashboard.png"
    grid_cols = max(2, len(noise_results) + 1)
    fig = plt.figure(figsize=(4.2 * grid_cols, 15), dpi=args.dpi, constrained_layout=True)
    gs = fig.add_gridspec(4, grid_cols, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_display, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Original MRI")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(original_spectrum_display, cmap="magma")
    ax.set_title("Original MRI Log Spectrum")
    ax.axis("off")

    radial_ax = fig.add_subplot(gs[0, 2:] if grid_cols > 2 else gs[1, :])
    radial_ax.plot(np.log10(original_profile + 1.0e-12), label="Original MRI", linewidth=2.0, linestyle="--")
    for result in noise_results:
        radial_ax.plot(np.log10(np.asarray(result["profile"]) + 1.0e-12), label=str(result["item"].label), linewidth=2.0)
    radial_ax.set_title("Radial Spectrum Comparison")
    radial_ax.set_xlabel("Radius bin")
    radial_ax.set_ylabel("log10(power + 1e-12)")
    radial_ax.grid(True, alpha=0.3)
    radial_ax.legend(fontsize=9)

    for column, result in enumerate(noise_results):
        ax = fig.add_subplot(gs[1, column])
        ax.imshow(np.asarray(result["noise_display"]), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Noise: {result['item'].label}")
        ax.axis("off")
    for column in range(len(noise_results), grid_cols):
        ax = fig.add_subplot(gs[1, column])
        ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(original_display, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Original MRI")
    ax.axis("off")
    for column, result in enumerate(noise_results, start=1):
        ax = fig.add_subplot(gs[2, column])
        ax.imshow(np.asarray(result["noised_display"]), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(str(result["item"].label))
        ax.axis("off")
    for column in range(len(noise_results) + 1, grid_cols):
        ax = fig.add_subplot(gs[2, column])
        ax.axis("off")

    half = max(1, grid_cols // 2)
    ax = fig.add_subplot(gs[3, :half])
    ax.imshow(original_spectrum_display, cmap="magma")
    ax.set_title("Original MRI Log Spectrum")
    ax.axis("off")

    metadata_ax = fig.add_subplot(gs[3, half:])
    metadata_ax.axis("off")
    metadata_lines = [
        f"dataset_path: {args.dataset_path}",
        f"resolved_index: {resolved_index}",
        f"dataset_length: {dataset_length}",
        f"channel: {args.channel}",
        f"timestep: {args.timestep}",
        f"alpha_hat: {alpha_hat:.6f}",
        f"seed: {args.seed}",
        f"spectrum_mix_white: {args.spectrum_mix_white}",
        f"same_base_noise: {bool(args.same_base_noise)}",
    ]
    for item in items:
        metadata_lines.extend(
            [
                "",
                f"[{item.label}]",
                f"path: {item.path}",
                f"radial_key: {item.radial_key}",
                f"radial_shape: {list(item.radial_shape)}",
                f"channels: {item.channels}",
                f"per_channel: {item.per_channel}",
                f"count: {item.count}",
                f"image_size: {list(item.image_size) if item.image_size is not None else None}",
            ]
        )
    metadata_ax.text(0.0, 1.0, "\n".join(metadata_lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.savefig(dashboard_path, dpi=args.dpi)
    plt.close(fig)

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset_path": str(args.dataset_path),
        "resolved_index": int(resolved_index),
        "dataset_length": int(dataset_length),
        "channel": int(args.channel),
        "timestep": int(args.timestep),
        "alpha_hat": float(alpha_hat),
        "seed": int(args.seed),
        "spectrum_mix_white": float(args.spectrum_mix_white),
        "same_base_noise": bool(args.same_base_noise),
        "items": [item.to_metadata() for item in items],
        "outputs": {
            "dashboard": dashboard_path.name,
            "radial_compare": radial_compare_path.name,
            "noise_grid": noise_grid_path.name,
            "noised_mri_grid": noised_grid_path.name,
            "noise_images": {str(result["item"].label): str(result["noise_path"]) for result in noise_results},
            "noised_mri_images": {str(result["item"].label): str(result["noised_path"]) for result in noise_results},
        },
    }
    write_metadata(metadata_path, metadata)

    print(f"Saved comparison outputs to: {output_dir}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Radial compare: {radial_compare_path}")
    print(f"Noise grid: {noise_grid_path}")
    print(f"Noised MRI grid: {noised_grid_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
