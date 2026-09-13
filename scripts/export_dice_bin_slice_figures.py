"""Export every axial slice for two representative cases per Dice bin.

The case ranking is computed from the disk evaluation cache so it exactly
matches the reported MF + per-subject 3-D Yen Dice.  Figures use the native
FLAIR grid and the exported native-grid Yen/MF mask.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_yen


DEFAULT_CACHE = Path(
    "outputs/runs/mixed_sri24_robust_iqr200_continue20/"
    "evaluation/brats21_test251/cache"
)
DEFAULT_PREDICTIONS = Path(
    "outputs/runs/mixed_sri24_robust_iqr200_continue20/"
    "evaluation/brats21_test251/predictions"
)
DEFAULT_OUTPUT = Path(
    "outputs/figures/mixed_sri24_robust_iqr200_continue20_epoch0199/"
    "dice_bins_test251"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases-per-bin", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    return float(value)


def dice_from_score(score: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        threshold = safe_float(threshold_yen(score))
    prediction = np.asarray(score) > threshold
    truth = np.asarray(target).astype(bool, copy=False)
    intersection = np.logical_and(prediction, truth).sum(dtype=np.int64)
    denominator = prediction.sum(dtype=np.int64) + truth.sum(dtype=np.int64)
    return float(2 * intersection / max(int(denominator), 1)), threshold


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def normalize_flair(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    if positive.size == 0:
        low, high = 0.0, 1.0
    else:
        low = float(np.percentile(positive, 1.0))
        high = float(np.percentile(positive, 99.5))
        if high <= low:
            high = low + 1.0
    normalized = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def rotate_slice(slice_2d: np.ndarray) -> np.ndarray:
    return np.rot90(slice_2d, k=1)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    image = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    color_array = np.asarray(color, dtype=np.float32)
    image[mask_bool] = (1.0 - alpha) * image[mask_bool] + alpha * color_array
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def compose_figure(
    original: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    case_id: str,
    dice: float,
    slice_index: int,
    slice_count: int,
    panel_size: int,
) -> Image.Image:
    gap = 24
    top = 76
    bottom = 88
    canvas_width = 4 * panel_size + 3 * gap + 48
    canvas_height = top + panel_size + bottom
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 22, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    panel_font = load_font(22)
    legend_font = load_font(19)

    title = f"{case_id}   |   Dice {dice:.4f}   |   axial slice {slice_index:03d}/{slice_count - 1:03d}"
    draw.text((24, 18), title, fill=(240, 242, 246), font=title_font)

    original_u8 = rotate_slice(original)
    truth_rot = rotate_slice(truth)
    prediction_rot = rotate_slice(prediction)
    truth_panel = overlay_mask(original_u8, truth_rot, (30, 220, 255), 0.52)
    prediction_panel = overlay_mask(original_u8, prediction_rot, (255, 205, 40), 0.52)

    true_positive = truth_rot & prediction_rot
    false_positive = (~truth_rot) & prediction_rot
    false_negative = truth_rot & (~prediction_rot)
    comparison = np.repeat(original_u8[..., None], 3, axis=2).astype(np.float32)
    for mask, color, alpha in (
        (false_positive, (235, 55, 65), 0.88),
        (false_negative, (65, 125, 250), 0.88),
        (true_positive, (55, 210, 105), 0.88),
    ):
        mask_bool = np.asarray(mask).astype(bool, copy=False)
        color_array = np.asarray(color, dtype=np.float32)
        comparison[mask_bool] = (1.0 - alpha) * comparison[mask_bool] + alpha * color_array
    comparison = np.clip(comparison, 0.0, 255.0).astype(np.uint8)

    panels = [
        (original_u8, "Original FLAIR"),
        (truth_panel, "Ground truth"),
        (prediction_panel, "Predicted mask"),
        (comparison, "FP / FN / TP"),
    ]
    x = 24
    for panel, label in panels:
        panel_image = Image.fromarray(panel).convert("RGB").resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        )
        canvas.paste(panel_image, (x, top))
        draw.text((x, top + panel_size + 10), label, fill=(240, 242, 246), font=panel_font)
        x += panel_size + gap

    legend_y = top + panel_size + 48
    legend_items = [
        ((235, 55, 65), "FP false positive"),
        ((65, 125, 250), "FN false negative"),
        ((55, 210, 105), "TP true positive"),
    ]
    x = 24
    for color, label in legend_items:
        draw.rectangle((x, legend_y + 3, x + 22, legend_y + 25), fill=color)
        draw.text((x + 30, legend_y), label, fill=(220, 224, 232), font=legend_font)
        x += 255
    return canvas


def select_cases(cache_root: Path, cases_per_bin: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        score = np.load(cache_root / entry["mf"]["file"], mmap_mode="r")
        target = np.load(cache_root / entry["label_file"], mmap_mode="r")
        dice, threshold = dice_from_score(score, target)
        rows.append(
            {
                "case_id": str(entry["subject_id"]),
                "dice": dice,
                "yen_threshold": threshold,
                "reference_path": entry.get("metadata", {}).get("reference_path"),
                "segmentation_path": entry.get("metadata", {}).get("segmentation_path"),
            }
        )

    selected: list[dict[str, Any]] = []
    for bin_index in range(10):
        low = bin_index / 10.0
        high = (bin_index + 1) / 10.0
        candidates = [
            row
            for row in rows
            if low <= row["dice"] < high or (bin_index == 9 and row["dice"] <= high)
        ]
        midpoint = (low + high) / 2.0
        candidates.sort(key=lambda row: (abs(row["dice"] - midpoint), -row["dice"], row["case_id"]))
        for row in candidates[:cases_per_bin]:
            selected.append(
                {
                    **row,
                    "bin": f"{low:.1f}-{high:.1f}",
                    "bin_index": bin_index,
                    "selection_rule": "closest to bin midpoint; ties prefer higher Dice",
                }
            )
    return rows, selected


def native_dice(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_bool = np.asarray(truth).astype(bool, copy=False)
    prediction_bool = np.asarray(prediction).astype(bool, copy=False)
    intersection = np.logical_and(truth_bool, prediction_bool).sum(dtype=np.int64)
    denominator = truth_bool.sum(dtype=np.int64) + prediction_bool.sum(dtype=np.int64)
    return float(2 * intersection / max(int(denominator), 1))


def render_case(
    row: dict[str, Any],
    predictions_root: Path,
    output_root: Path,
    panel_size: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    case_id = row["case_id"]
    prediction_dir = predictions_root / case_id
    metadata = json.loads((prediction_dir / "prediction_metadata.json").read_text(encoding="utf-8"))
    reference_path = Path(metadata["reference_path"])
    segmentation_path = Path(metadata["segmentation_path"])
    prediction_path = prediction_dir / "lesion_mask_yen_mf.nii.gz"
    original = normalize_flair(np.asanyarray(nib.load(str(reference_path)).dataobj))
    truth = np.asanyarray(nib.load(str(segmentation_path)).dataobj) > 0
    prediction = np.asanyarray(nib.load(str(prediction_path)).dataobj) > 0
    if original.shape != truth.shape or truth.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: original={original.shape}, "
            f"truth={truth.shape}, prediction={prediction.shape}"
        )
    case_output = output_root / f"bin_{row['bin_index']:02d}_{row['bin'].replace('.', '')}" / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    slice_count = int(original.shape[2])
    for slice_index in range(slice_count):
        figure = compose_figure(
            original[:, :, slice_index],
            truth[:, :, slice_index],
            prediction[:, :, slice_index],
            case_id=case_id,
            dice=float(row["dice"]),
            slice_index=slice_index,
            slice_count=slice_count,
            panel_size=panel_size,
        )
        figure.save(
            case_output / f"slice_{slice_index:03d}.jpg",
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
    result = dict(row)
    result.update(
        {
            "native_dice_exported_mask": native_dice(truth, prediction),
            "slice_count": slice_count,
            "shape": list(original.shape),
            "output_directory": str(case_output.resolve()),
        }
    )
    return result


def write_readme(output_root: Path, selected: list[dict[str, Any]]) -> None:
    lines = [
        "# Dice-bin slice figures",
        "",
        "Run: mixed_sri24_robust_iqr200_continue20 / epoch_0199.pt",
        "Dataset: BraTS21 test251",
        "Selection: two cases closest to each 0.1 Dice-bin midpoint.",
        "Panels: Original FLAIR, ground truth, predicted MF/Yen mask, and FP/FN/TP comparison.",
        "Legend: FP red, FN blue, TP green.",
        "",
        "| Bin | Case | Cache Dice | Native exported Dice | Slice count |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['bin']} | {row['case_id']} | {row['dice']:.4f} | "
            f"{row.get('native_dice_exported_mask', float('nan')):.4f} | {row.get('slice_count', '')} |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.cases_per_bin <= 0:
        raise ValueError("--cases-per-bin must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    rows, selected = select_cases(args.cache_root, args.cases_per_bin)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    selection_payload = {
        "run": "mixed_sri24_robust_iqr200_continue20",
        "checkpoint": "epoch_0199.pt",
        "dataset": "BraTS21 test251",
        "metric": "MF score + per-subject 3-D Yen mask Dice from evaluation cache",
        "selection_rule": "closest to each 0.1-bin midpoint; ties prefer higher Dice",
        "total_cases": len(rows),
        "selected_cases": len(selected),
        "bins": [
            {
                "bin": f"{i / 10.0:.1f}-{(i + 1) / 10.0:.1f}",
                "count": sum(
                    1
                    for row in rows
                    if i / 10.0 <= row["dice"] < (i + 1) / 10.0
                    or (i == 9 and i / 10.0 <= row["dice"] <= 1.0)
                ),
                "cases": [row for row in selected if row["bin_index"] == i],
            }
            for i in range(10)
        ],
    }
    (output_root / "selection.json").write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"selected {len(selected)} cases from {len(rows)} total cases")
    for row in selected:
        print(f"bin={row['bin']} case={row['case_id']} dice={row['dice']:.6f}")
    if args.dry_run:
        return
    rendered: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        print(f"rendering {index}/{len(selected)} {row['case_id']} ({row['bin']})", flush=True)
        rendered.append(
            render_case(
                row,
                args.predictions_root,
                output_root,
                panel_size=args.panel_size,
                jpeg_quality=args.jpeg_quality,
            )
        )
    (output_root / "selection_rendered.json").write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(output_root, rendered)
    print(f"completed {len(rendered)} cases, {sum(row['slice_count'] for row in rendered)} slice figures")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
