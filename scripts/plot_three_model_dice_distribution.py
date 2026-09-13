#!/usr/bin/env python
"""Plot per-case Dice distributions for the three selected-model outputs.

The figure layout follows the user's reference image: 0.1-wide Dice bins,
count labels above each bar, and separate median/mean reference lines.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "outputs/figures/"
    "brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/"
    "case_dice_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/figures/"
    "brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/"
    "dice_distribution"
)

BIN_EDGES = np.linspace(0.0, 1.0, 11)
BIN_LABELS = [f"{left:.1f}" for left in BIN_EDGES[:-1]]

MODELS = (
    {
        "key": "mixed_epoch0199_dice",
        "label": "Mixed epoch_0199",
        "file_label": "mixed_epoch0199",
        "color": "#4f7fc7",
    },
    {
        "key": "brats21_4modal_epoch0232_dice",
        "label": "BraTS21 4-modal epoch_0232",
        "file_label": "brats21_4modal_epoch0232",
        "color": "#8c63c7",
    },
    {
        "key": "brats21_3modal_epoch0232_dice",
        "label": "BraTS21 3-modal epoch_0232",
        "file_label": "brats21_3modal_epoch0232",
        "color": "#e4863a",
    },
)

MEDIAN_COLOR = "#ef4fa3"
MEAN_COLOR = "#f0a202"
TEXT_COLOR = "#2f455e"
GRID_COLOR = "#e7edf4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected-case Dice distributions for three models."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV containing one row per case and one Dice column per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where PNG/SVG figures and summary files are written.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_values(input_csv: Path) -> dict[str, np.ndarray]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    values: dict[str, np.ndarray] = {}
    for model in MODELS:
        key = model["key"]
        missing = [row.get("case_id", "<unknown>") for row in rows if not row.get(key)]
        if missing:
            raise ValueError(f"Missing {key} for cases: {', '.join(missing[:5])}")
        values[key] = np.asarray([float(row[key]) for row in rows], dtype=float)

    return values


def distribution(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    if np.any(~np.isfinite(values)):
        raise ValueError("Dice values contain NaN or infinity")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Dice values must be between 0 and 1")
    counts, _ = np.histogram(values, bins=BIN_EDGES)
    return counts, float(np.mean(values)), float(np.median(values))


def style_axis(ax: plt.Axes, ymax: int) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(BIN_EDGES)
    ax.set_xticklabels([f"{edge:.1f}" for edge in BIN_EDGES])
    ax.set_yticks(np.arange(0, ymax + 1, 5 if ymax <= 40 else 10))
    ax.set_xlabel("Per-case Dice", color="#77869a", labelpad=8)
    ax.set_ylabel("Number of cases", color="#77869a", labelpad=8)
    ax.tick_params(axis="both", colors="#77869a", labelsize=9, length=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f7e90")
    ax.spines["bottom"].set_color("#6f7e90")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def draw_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    model: dict[str, str],
    ymax: int,
    show_ylabel: bool = True,
) -> tuple[int, float, float]:
    counts, mean, median = distribution(values)
    centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
    bars = ax.bar(
        centers,
        counts,
        width=0.098,
        color=model["color"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.97,
        align="center",
    )
    style_axis(ax, ymax)
    if not show_ylabel:
        ax.set_ylabel("")

    label_offset = max(0.18, ymax * 0.012)
    for bar, count in zip(bars, counts):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) + label_offset,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.axvline(median, color=MEDIAN_COLOR, linestyle=":", linewidth=1.8)
    ax.axvline(mean, color=MEAN_COLOR, linestyle="--", linewidth=1.8)
    ax.text(
        0.90,
        0.975,
        f"median {median:.3f}\nmean {mean:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        linespacing=1.35,
    )
    ax.set_title(
        f"{model['label']}  (n={len(values)})",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=11,
    )
    return int(len(values)), mean, median


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    values: dict[str, np.ndarray],
    stats: dict[str, tuple[int, float, float]],
) -> None:
    summary_path = output_dir / "dice_distribution_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "model_column",
                "n_cases",
                "mean_dice",
                "median_dice",
                "bin_left",
                "bin_right",
                "count",
            ]
        )
        for model in MODELS:
            n_cases, mean, median = stats[model["key"]]
            counts, _, _ = distribution(values[model["key"]])
            for index, count in enumerate(counts):
                writer.writerow(
                    [
                        model["label"],
                        model["key"],
                        n_cases,
                        f"{mean:.9f}",
                        f"{median:.9f}",
                        f"{BIN_EDGES[index]:.1f}",
                        f"{BIN_EDGES[index + 1]:.1f}",
                        int(count),
                    ]
                )


def write_readme(output_dir: Path, input_csv: Path, stats: dict[str, tuple[int, float, float]]) -> None:
    lines = [
        "# Three-model per-case Dice distributions",
        "",
        "These figures use the same 20 selected BraTS21 test cases used for the",
        "three-model slice comparison figures. Bins are 0.1 Dice wide.",
        "The dotted pink line is the median and the dashed orange line is the mean.",
        "",
        f"Source CSV: `{input_csv}`",
        "",
        "| Model | Cases | Mean | Median |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        n_cases, mean, median = stats[model["key"]]
        lines.append(f"| {model['label']} | {n_cases} | {mean:.6f} | {median:.6f} |")
    lines += [
        "",
        "Files:",
        "- `dice_distribution_three_models_selected20.png`: three-panel comparison",
        "- `dice_distribution_<model>_selected20.png`: individual model figures",
        "- `dice_distribution_summary.csv`: per-bin counts and summary statistics",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    output_dir = args.output_dir
    if not input_csv.is_absolute():
        input_csv = Path.cwd() / input_csv
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    input_csv = input_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    values = load_values(input_csv)
    all_counts = [distribution(values[model["key"]])[0] for model in MODELS]
    max_count = max(int(counts.max()) for counts in all_counts)
    ymax = max(10, int(np.ceil((max_count + 3) / 5.0) * 5))

    stats: dict[str, tuple[int, float, float]] = {}
    for model in MODELS:
        counts, mean, median = distribution(values[model["key"]])
        stats[model["key"]] = (len(values[model["key"]]), mean, median)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), sharey=True)
    for index, (ax, model) in enumerate(zip(axes, MODELS)):
        draw_distribution(
            ax,
            values[model["key"]],
            model,
            ymax,
            show_ylabel=index == 0,
        )
    fig.suptitle(
        "Per-case Dice distribution — selected 20 cases",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.02,
    )
    fig.tight_layout(w_pad=1.5)
    save_figure(
        fig,
        output_dir / "dice_distribution_three_models_selected20.png",
        args.dpi,
    )

    for model in MODELS:
        fig, ax = plt.subplots(figsize=(8.0, 5.25))
        draw_distribution(ax, values[model["key"]], model, ymax)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / f"dice_distribution_{model['file_label']}_selected20.png",
            args.dpi,
        )

    write_summary(output_dir, values, stats)
    write_readme(output_dir, input_csv, stats)

    print(f"input_csv={input_csv}")
    print(f"output_dir={output_dir}")
    for model in MODELS:
        n_cases, mean, median = stats[model["key"]]
        counts, _, _ = distribution(values[model["key"]])
        print(
            f"{model['label']}: n={n_cases}, mean={mean:.6f}, "
            f"median={median:.6f}, counts={counts.tolist()}"
        )


if __name__ == "__main__":
    main()
