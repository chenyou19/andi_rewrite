"""Infer the selected 20 BraTS21 cases with two checkpoints and render comparisons.

The mixed run already has native-grid prediction masks.  This script runs the
two requested epoch-0232 checkpoints on the same selected subjects with the
filtered-Gaussian empirical-spectrum inference configuration, computes native
3-D Yen/MF Dice per subject, and exports one large JPEG for every axial slice.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data import build_dataloader
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.scripts.eval import build_detector_from_config
from andi_rewrite.utils import load_config, set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "outputs/figures/mixed_sri24_robust_iqr200_continue20_epoch0199/dice_bins_test251/selection.json"
DEFAULT_MIXED_PREDICTIONS = ROOT / "outputs/runs/mixed_sri24_robust_iqr200_continue20/evaluation/brats21_test251/predictions"
DEFAULT_OUTPUT = ROOT / "outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199"

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "brats21_4modal_epoch0232",
        "label": "BraTS21 4-modal",
        "config": ROOT / "configs/eval_brats21_full_filtered_gaussian_epoch0232_20260609.yaml",
        "checkpoint": ROOT / "outputs/checkpoints/empirical_spectrum233_lmdb_full_gaussian_20260609/epoch_0232.pt",
        "noise": "empirical_spectrum / filtered_gaussian / radial",
    },
    {
        "key": "brats21_3modal_epoch0232",
        "label": "BraTS21 3-modal FLAIR-T1-T2",
        "config": ROOT / "configs/eval_brats21_251_brats_flair_t1_t2_empirical_spectrum233_epoch0232.yaml",
        "checkpoint": ROOT / "outputs/runs/brats_flair_t1_t2_empirical_spectrum233/epoch_0232.pt",
        "noise": "empirical_spectrum / filtered_gaussian / radial",
    },
)

MIXED_KEY = "mixed_epoch0199"
MIXED_LABEL = "Mixed epoch0199"
MASK_FILENAME = "lesion_mask_yen_mf.nii.gz"

MASK_COLORS: dict[str, tuple[int, int, int]] = {
    MIXED_KEY: (255, 205, 40),
    "brats21_4modal_epoch0232": (180, 90, 255),
    "brats21_3modal_epoch0232": (255, 125, 45),
}
COMPARISON_COLORS: tuple[tuple[int, int, int], ...] = (
    (235, 55, 65),
    (65, 125, 250),
    (55, 210, 105),
)
COMPARISON_LABELS = ("FP false positive", "FN false negative", "TP true positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--mixed-predictions-root", type=Path, default=DEFAULT_MIXED_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for a smoke run.")
    parser.add_argument(
        "--only-model",
        choices=[spec["key"] for spec in MODEL_SPECS],
        action="append",
        help="Run only the named model; may be supplied twice.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute prediction masks even when a completed case output exists.",
    )
    return parser.parse_args()


def load_selection(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for bin_entry in payload.get("bins", []):
        for case in bin_entry.get("cases", []):
            row = dict(case)
            row.setdefault("bin", bin_entry.get("bin"))
            row.setdefault("bin_index", len(selected) // 2)
            selected.append(row)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when supplied.")
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"Selection JSON contains no selected cases: {path}")
    case_ids = [str(row["case_id"]) for row in selected]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Selected cases contain duplicates: {duplicates}")
    return selected


def write_selected_csv(selected: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BraTS21ID"])
        writer.writerows([[str(row["case_id"])] for row in selected])


def case_seed(case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return 7300 + int(digest[:8], 16) % 1_000_000


def build_inference_config(
    spec: dict[str, Any],
    selected_csv: Path,
    model_output: Path,
) -> dict[str, Any]:
    config = load_config(spec["config"])
    config.pop("_config_path", None)
    config.setdefault("experiment", {})["name"] = f"{spec['key']}_selected20_filtered_gaussian"
    data = config.setdefault("data", {})
    data["path_to_csv"] = str(selected_csv.resolve())
    data["batch_size"] = 1
    data["workers"] = 0
    data["shuffle"] = False
    data["return_metadata"] = True

    model = config.setdefault("model", {})
    model["checkpoint"] = str(Path(spec["checkpoint"]).resolve())
    model["use_ema"] = True

    metrics = config.setdefault("metrics", {})
    metrics["postprocess_mode"] = "rewrite"
    metrics["threshold_method"] = "yen"
    metrics["normalization_scope"] = "dataset"
    metrics["output_csv"] = str((model_output / "ANDi.csv").resolve())
    metrics["output_mf_csv"] = str((model_output / "ANDi_mf.csv").resolve())

    evaluation = config.setdefault("evaluation", {})
    evaluation["progress"] = {"enabled": False}
    config["prediction_output"] = {
        "enabled": True,
        "directory": str((model_output / "predictions").resolve()),
        "normalization_scope": "subject",
        "binary_mask_source": "score_mf",
        "save_raw_score": False,
        "save_median_filtered_score": False,
        "save_binary_mask": True,
        "save_threshold_mask": False,
        "restore_native_grid": True,
        "save_model_grid": False,
    }
    return config


def evaluator_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config.get("data", {}),
        **config.get("metrics", {}),
        **config.get("evaluation", {}),
        "prediction_output": config.get("prediction_output", {}),
        "model": config.get("model", {}),
        "anomaly": config.get("anomaly", {}),
        "_run_config": config,
    }


def load_bool_volume(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI mask does not exist: {path}")
    return np.asanyarray(nib.load(str(path)).dataobj) > 0.5


def dice_from_masks(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_bool = np.asarray(truth).astype(bool, copy=False)
    prediction_bool = np.asarray(prediction).astype(bool, copy=False)
    if truth_bool.shape != prediction_bool.shape:
        raise ValueError(
            f"Dice shape mismatch: truth={truth_bool.shape}, prediction={prediction_bool.shape}"
        )
    intersection = np.logical_and(truth_bool, prediction_bool).sum(dtype=np.int64)
    denominator = truth_bool.sum(dtype=np.int64) + prediction_bool.sum(dtype=np.int64)
    return float(2 * intersection / max(int(denominator), 1))


def case_native_dice(prediction_path: Path, segmentation_path: Path) -> float:
    return dice_from_masks(load_bool_volume(segmentation_path), load_bool_volume(prediction_path))


def run_model(
    spec: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_csv: Path,
    output_root: Path,
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    model_output = output_root / "inference" / spec["key"]
    model_output.mkdir(parents=True, exist_ok=True)
    prediction_root = model_output / "predictions"
    config = build_inference_config(spec, selected_csv, model_output)

    print(f"Loading {spec['label']} checkpoint: {config['model']['checkpoint']}", flush=True)
    detector, accelerator = build_detector_from_config(config)
    if accelerator is not None:
        raise RuntimeError("This comparison script requires single-process inference.")
    dataloader = build_dataloader(config["data"])
    evaluator = VolumeEvaluator(
        detector=detector,
        config=evaluator_config(config),
        accelerator=accelerator,
    )
    dataloader = evaluator.prepare(dataloader)

    selected_by_id = {str(row["case_id"]): row for row in selected}
    results: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for volume_index, batch in enumerate(dataloader, start=1):
                image, _label, metadata = evaluator._split_batch(batch)
                items = evaluator._metadata_items(metadata, int(image.shape[0]))
                if len(items) != 1:
                    raise RuntimeError(
                        f"Expected batch_size=1 metadata for {spec['key']}; got {len(items)} items."
                    )
                case_id = str(items[0].get("subject_id"))
                if case_id not in selected_by_id:
                    raise RuntimeError(f"Unexpected subject from selected CSV: {case_id}")
                row = selected_by_id[case_id]
                output_mask = prediction_root / case_id / MASK_FILENAME

                if resume and output_mask.is_file() and (output_mask.parent / "prediction_metadata.json").is_file():
                    prediction_path = output_mask
                    threshold = None
                    print(f"[{spec['key']}] resume {volume_index}/{len(selected)} {case_id}", flush=True)
                else:
                    set_seed(case_seed(case_id))
                    raw_maps = evaluator._volume_scores(image, volume_index=volume_index)
                    processed = evaluator._prediction_postprocess(raw_maps, metric_processed=None)
                    evaluator._export_predictions(raw_maps, items, processed=processed)
                    prediction_path = output_mask
                    if not prediction_path.is_file():
                        raise FileNotFoundError(
                            f"Prediction export did not create {prediction_path}"
                        )
                    threshold = (
                        float(processed.thresholds_mf[0].detach().cpu().item())
                        if processed.thresholds_mf.numel()
                        else None
                    )
                    del raw_maps, processed
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                segmentation_path = Path(str(items[0]["segmentation_path"]))
                dice = case_native_dice(prediction_path, segmentation_path)
                results.append(
                    {
                        "case_id": case_id,
                        "bin": row.get("bin"),
                        "bin_index": row.get("bin_index"),
                        "dice": dice,
                        "threshold_mf": threshold,
                        "prediction_path": str(prediction_path.resolve()),
                        "segmentation_path": str(segmentation_path.resolve()),
                    }
                )
                print(
                    f"[{spec['key']}] {volume_index}/{len(selected)} {case_id} Dice={dice:.6f}",
                    flush=True,
                )
    finally:
        del dataloader, evaluator, detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda row: selected_by_id[row["case_id"]].get("bin_index", 0))
    (model_output / "case_dice.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (model_output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return results


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
    normalized = np.clip(
        (np.nan_to_num(values, nan=low) - low) / (high - low),
        0.0,
        1.0,
    )
    return (normalized * 255.0).astype(np.uint8)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def rotate_slice(slice_2d: np.ndarray) -> np.ndarray:
    return np.rot90(slice_2d, k=1)


def overlay_mask(
    base: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    image = np.repeat(np.asarray(base)[..., None], 3, axis=2).astype(np.float32)
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    color_array = np.asarray(color, dtype=np.float32)
    image[mask_bool] = (1.0 - alpha) * image[mask_bool] + alpha * color_array
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def comparison_panel(
    original_u8: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    comparison = np.repeat(original_u8[..., None], 3, axis=2).astype(np.float32)
    true_positive = np.logical_and(truth, prediction)
    false_positive = np.logical_and(~truth, prediction)
    false_negative = np.logical_and(truth, ~prediction)
    for mask, color in zip(
        (false_positive, false_negative, true_positive),
        COMPARISON_COLORS,
    ):
        mask_bool = np.asarray(mask).astype(bool, copy=False)
        color_array = np.asarray(color, dtype=np.float32)
        comparison[mask_bool] = 0.12 * comparison[mask_bool] + 0.88 * color_array
    return np.clip(comparison, 0.0, 255.0).astype(np.uint8)


def paste_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    panel: np.ndarray,
    label: str,
    x: int,
    y: int,
    panel_size: int,
    panel_font: ImageFont.ImageFont,
) -> None:
    image = Image.fromarray(panel).convert("RGB").resize(
        (panel_size, panel_size),
        Image.Resampling.NEAREST,
    )
    canvas.paste(image, (x, y))
    draw.text((x, y + panel_size + 8), label, fill=(240, 242, 246), font=panel_font)


def compose_figure(
    original: np.ndarray,
    truth: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    case_id: str,
    bin_label: str,
    dice: dict[str, float],
    slice_index: int,
    slice_count: int,
    panel_size: int,
) -> Image.Image:
    gap = 18
    margin = 24
    top = 92
    row_gap = 34
    bottom = 112
    canvas_width = margin * 2 + 4 * panel_size + 3 * gap
    canvas_height = top + 2 * panel_size + row_gap + bottom
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 22, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    subtitle_font = load_font(21)
    panel_font = load_font(18)
    legend_font = load_font(16)

    draw.text(
        (margin, 12),
        f"{case_id} | bin {bin_label} | axial slice {slice_index:03d}/{slice_count - 1:03d}",
        fill=(240, 242, 246),
        font=title_font,
    )
    draw.text(
        (margin, 44),
        "Mixed Dice "
        f"{dice[MIXED_KEY]:.4f}   |   4-modal Dice {dice['brats21_4modal_epoch0232']:.4f}"
        f"   |   3-modal Dice {dice['brats21_3modal_epoch0232']:.4f}",
        fill=(220, 224, 232),
        font=subtitle_font,
    )

    original_u8 = rotate_slice(normalize_flair(original))
    truth_rot = rotate_slice(truth)
    mask_rot = {key: rotate_slice(value) for key, value in masks.items()}
    truth_panel = overlay_mask(original_u8, truth_rot, (30, 220, 255), 0.52)
    model_panels = {
        key: overlay_mask(original_u8, mask_rot[key], MASK_COLORS[key], 0.58)
        for key in masks
    }
    compare_panels = {
        key: comparison_panel(original_u8, truth_rot, mask_rot[key])
        for key in masks
    }

    row_one_y = top
    row_two_y = top + panel_size + row_gap
    x_positions = [margin + index * (panel_size + gap) for index in range(4)]
    paste_panel(canvas, draw, original_u8, "Original FLAIR", x_positions[0], row_one_y, panel_size, panel_font)
    paste_panel(canvas, draw, truth_panel, "GT (cyan)", x_positions[1], row_one_y, panel_size, panel_font)
    paste_panel(
        canvas,
        draw,
        model_panels[MIXED_KEY],
        f"Mixed mask | Dice {dice[MIXED_KEY]:.4f}",
        x_positions[2],
        row_one_y,
        panel_size,
        panel_font,
    )
    paste_panel(
        canvas,
        draw,
        model_panels["brats21_4modal_epoch0232"],
        f"4-modal mask | Dice {dice['brats21_4modal_epoch0232']:.4f}",
        x_positions[3],
        row_one_y,
        panel_size,
        panel_font,
    )
    paste_panel(
        canvas,
        draw,
        model_panels["brats21_3modal_epoch0232"],
        f"3-modal mask | Dice {dice['brats21_3modal_epoch0232']:.4f}",
        x_positions[0],
        row_two_y,
        panel_size,
        panel_font,
    )
    paste_panel(
        canvas,
        draw,
        compare_panels[MIXED_KEY],
        "Mixed FP/FN/TP",
        x_positions[1],
        row_two_y,
        panel_size,
        panel_font,
    )
    paste_panel(
        canvas,
        draw,
        compare_panels["brats21_4modal_epoch0232"],
        "4-modal FP/FN/TP",
        x_positions[2],
        row_two_y,
        panel_size,
        panel_font,
    )
    paste_panel(
        canvas,
        draw,
        compare_panels["brats21_3modal_epoch0232"],
        "3-modal FP/FN/TP",
        x_positions[3],
        row_two_y,
        panel_size,
        panel_font,
    )

    legend_y = row_two_y + panel_size + 47
    legend_items = [
        (MASK_COLORS[MIXED_KEY], "Mixed mask"),
        (MASK_COLORS["brats21_4modal_epoch0232"], "4-modal mask"),
        (MASK_COLORS["brats21_3modal_epoch0232"], "3-modal mask"),
        *zip(COMPARISON_COLORS, COMPARISON_LABELS),
    ]
    x = margin
    for color, label in legend_items:
        draw.rectangle((x, legend_y + 2, x + 18, legend_y + 20), fill=color)
        draw.text((x + 25, legend_y), label, fill=(220, 224, 232), font=legend_font)
        x += 154 if label != "4-modal mask" else 168
    return canvas


def read_case_dice(model_output: Path) -> dict[str, float]:
    path = model_output / "case_dice.json"
    if not path.is_file():
        raise FileNotFoundError(f"Case Dice report does not exist: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_id"]): float(row["dice"]) for row in rows}


def render_figures(
    selected: list[dict[str, Any]],
    mixed_predictions_root: Path,
    output_root: Path,
    model_dice: dict[str, dict[str, float]],
    *,
    panel_size: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        case_id = str(row["case_id"])
        reference_path = Path(str(row["reference_path"]))
        segmentation_path = Path(str(row["segmentation_path"]))
        original = np.asanyarray(nib.load(str(reference_path)).dataobj)
        truth = load_bool_volume(segmentation_path)
        prediction_paths = {
            MIXED_KEY: mixed_predictions_root / case_id / MASK_FILENAME,
            "brats21_4modal_epoch0232": output_root / "inference" / "brats21_4modal_epoch0232" / "predictions" / case_id / MASK_FILENAME,
            "brats21_3modal_epoch0232": output_root / "inference" / "brats21_3modal_epoch0232" / "predictions" / case_id / MASK_FILENAME,
        }
        masks = {key: load_bool_volume(path) for key, path in prediction_paths.items()}
        expected_shape = tuple(original.shape)
        if tuple(truth.shape) != expected_shape:
            raise ValueError(f"{case_id}: GT shape {truth.shape} != original shape {expected_shape}")
        if any(tuple(mask.shape) != expected_shape for mask in masks.values()):
            raise ValueError(
                f"{case_id}: mask shapes do not match original {expected_shape}: "
                f"{ {key: tuple(mask.shape) for key, mask in masks.items()} }"
            )

        dice = {
            MIXED_KEY: dice_from_masks(truth, masks[MIXED_KEY]),
            "brats21_4modal_epoch0232": dice_from_masks(truth, masks["brats21_4modal_epoch0232"]),
            "brats21_3modal_epoch0232": dice_from_masks(truth, masks["brats21_3modal_epoch0232"]),
        }
        bin_index = int(row.get("bin_index", (index - 1) // 2))
        bin_label = str(row.get("bin", f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}"))
        case_root = figure_root / f"bin_{bin_index:02d}_{bin_label.replace('.', '')}" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        slice_count = int(original.shape[2])
        for slice_index in range(slice_count):
            figure = compose_figure(
                original[:, :, slice_index],
                truth[:, :, slice_index],
                {key: value[:, :, slice_index] for key, value in masks.items()},
                case_id=case_id,
                bin_label=bin_label,
                dice=dice,
                slice_index=slice_index,
                slice_count=slice_count,
                panel_size=panel_size,
            )
            figure.save(
                case_root / f"slice_{slice_index:03d}.jpg",
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=True,
            )

        summary_rows.append(
            {
                "case_id": case_id,
                "bin": bin_label,
                "bin_index": bin_index,
                "selection_mixed_cache_dice": float(row.get("dice", float("nan"))),
                "mixed_epoch0199_dice": dice[MIXED_KEY],
                "brats21_4modal_epoch0232_dice": dice["brats21_4modal_epoch0232"],
                "brats21_3modal_epoch0232_dice": dice["brats21_3modal_epoch0232"],
                "delta_4modal_minus_mixed": dice["brats21_4modal_epoch0232"] - dice[MIXED_KEY],
                "delta_3modal_minus_mixed": dice["brats21_3modal_epoch0232"] - dice[MIXED_KEY],
                "delta_3modal_minus_4modal": dice["brats21_3modal_epoch0232"] - dice["brats21_4modal_epoch0232"],
                "slice_count": slice_count,
                "figure_directory": str(case_root.resolve()),
            }
        )
        rendered.append(
            {
                "case_id": case_id,
                "bin": bin_label,
                "bin_index": bin_index,
                "dice": dice,
                "slice_count": slice_count,
                "figure_directory": str(case_root.resolve()),
            }
        )
        print(f"Rendered {index}/{len(selected)} {case_id}: {slice_count} slices", flush=True)

    with (output_root / "case_dice_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_root / "case_dice_comparison.json").write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return rendered


def write_readme(
    output_root: Path,
    selected: list[dict[str, Any]],
    rendered: list[dict[str, Any]],
) -> None:
    lines = [
        "# BraTS21 selected-20 three-model comparison",
        "",
        "Models:",
        "- Mixed SRI24 robust IQR, epoch_0199.pt (existing native-grid prediction)",
        "- BraTS21 4-modal empirical-spectrum epoch_0232.pt",
        "- BraTS21 3-modal FLAIR/T1/T2 empirical-spectrum epoch_0232.pt",
        "",
        "Inference noise for the two new checkpoints: empirical_spectrum / filtered_gaussian / radial, strength=1, normalize=true, per_channel=true.",
        "Each figure contains Original FLAIR, GT, three model masks, and one FP/FN/TP panel per model.",
        "FP is red, FN is blue, and TP is green. Dice values are native-grid whole-volume Dice for the case.",
        "",
        f"Selected cases: {len(selected)}; rendered figures: {sum(int(item['slice_count']) for item in rendered)}",
        "",
        "See `case_dice_comparison.csv` for the numeric comparison and `figures/` for all axial-slice JPEGs.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if args.panel_size <= 0:
        raise ValueError("--panel-size must be positive.")

    selection_path = args.selection_json.resolve()
    mixed_predictions_root = args.mixed_predictions_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = load_selection(selection_path, limit=args.limit)
    selected_csv = output_root / "selected_cases.csv"
    write_selected_csv(selected, selected_csv)
    print(f"Selected {len(selected)} cases from {selection_path}", flush=True)

    selected_keys = set(args.only_model or [spec["key"] for spec in MODEL_SPECS])
    model_dice: dict[str, dict[str, float]] = {}
    for spec in MODEL_SPECS:
        if spec["key"] not in selected_keys:
            continue
        results = run_model(
            spec,
            selected,
            selected_csv,
            output_root,
            resume=not args.no_resume,
        )
        model_dice[spec["key"]] = {str(row["case_id"]): float(row["dice"]) for row in results}

    required_model_keys = {spec["key"] for spec in MODEL_SPECS}
    missing_models = required_model_keys.difference(model_dice)
    if missing_models:
        for key in missing_models:
            model_dice[key] = read_case_dice(output_root / "inference" / key)

    if not mixed_predictions_root.is_dir():
        raise FileNotFoundError(f"Mixed prediction directory does not exist: {mixed_predictions_root}")
    rendered = render_figures(
        selected,
        mixed_predictions_root,
        output_root,
        model_dice,
        panel_size=args.panel_size,
        jpeg_quality=args.jpeg_quality,
    )
    manifest = {
        "selection_json": str(selection_path),
        "selected_cases": len(selected),
        "mixed_checkpoint": "mixed_sri24_robust_iqr200_continue20/epoch_0199.pt",
        "mixed_predictions_root": str(mixed_predictions_root),
        "new_models": [
            {
                "key": spec["key"],
                "label": spec["label"],
                "config": str(Path(spec["config"]).resolve()),
                "checkpoint": str(Path(spec["checkpoint"]).resolve()),
                "noise": spec["noise"],
            }
            for spec in MODEL_SPECS
        ],
        "comparison_colors": {
            "false_positive": list(COMPARISON_COLORS[0]),
            "false_negative": list(COMPARISON_COLORS[1]),
            "true_positive": list(COMPARISON_COLORS[2]),
        },
        "panel_size": args.panel_size,
        "jpeg_quality": args.jpeg_quality,
        "rendered_cases": rendered,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(output_root, selected, rendered)
    print(f"Completed comparison output: {output_root}", flush=True)
    print(f"Rendered {sum(int(item['slice_count']) for item in rendered)} axial figures", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
