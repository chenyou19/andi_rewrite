#!/usr/bin/env python
"""Compare whole-volume precision and recall for the selected 20 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / (
    "outputs/figures/mixed_sri24_robust_iqr200_continue20_epoch0199/"
    "dice_bins_test251/selection.json"
)
DEFAULT_COMPARISON_ROOT = ROOT / (
    "outputs/figures/"
    "brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199"
)
MASK_FILENAME = "lesion_mask_yen_mf.nii.gz"

MODELS: tuple[dict[str, str | Path], ...] = (
    {
        "key": "mixed_epoch0199",
        "label": "Mixed epoch_0199",
        "prediction_root": ROOT / (
            "outputs/runs/mixed_sri24_robust_iqr200_continue20/"
            "evaluation/brats21_test251/predictions"
        ),
    },
    {
        "key": "brats21_4modal_epoch0232",
        "label": "BraTS21 4-modal epoch_0232",
        "prediction_root": DEFAULT_COMPARISON_ROOT / (
            "inference/brats21_4modal_epoch0232/predictions"
        ),
    },
    {
        "key": "brats21_3modal_epoch0232",
        "label": "BraTS21 3-modal epoch_0232",
        "prediction_root": DEFAULT_COMPARISON_ROOT / (
            "inference/brats21_3modal_epoch0232/predictions"
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT / "case_precision_recall.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT / "precision_recall_summary.csv",
    )
    return parser.parse_args()


def flatten_selection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for bin_entry in payload.get("bins", []):
        for case in bin_entry.get("cases", []):
            row = dict(case)
            row.setdefault("bin", bin_entry.get("bin"))
            row.setdefault("bin_index", len(selected) // 2)
            selected.append(row)
    if not selected:
        raise ValueError(f"No selected cases found in {path}")
    return selected


def load_bool_volume(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask does not exist: {path}")
    return np.asanyarray(nib.load(str(path)).dataobj) > 0.5


def mask_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Mask shape mismatch: truth={truth.shape}, prediction={prediction.shape}"
        )
    truth_bool = np.asarray(truth, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    tp = int(np.logical_and(truth_bool, prediction_bool).sum(dtype=np.int64))
    fp = int(np.logical_and(~truth_bool, prediction_bool).sum(dtype=np.int64))
    fn = int(np.logical_and(truth_bool, ~prediction_bool).sum(dtype=np.int64))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    dice = float(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
    }


def model_prediction_root(model: dict[str, str | Path], comparison_root: Path) -> Path:
    key = str(model["key"])
    if key == "mixed_epoch0199":
        return Path(model["prediction_root"])
    return comparison_root / "inference" / key / "predictions"


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "bin",
        "bin_index",
        "model",
        "model_column",
        "dice",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "prediction_path",
        "segmentation_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_column",
        "n_cases",
        "mean_precision",
        "median_precision",
        "mean_recall",
        "median_recall",
        "pooled_precision",
        "pooled_recall",
        "mean_dice",
        "median_dice",
        "total_tp",
        "total_fp",
        "total_fn",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    selection_json = args.selection_json.resolve()
    comparison_root = args.comparison_root.resolve()
    output_csv = args.output_csv.resolve()
    summary_csv = args.summary_csv.resolve()
    selected = flatten_selection(selection_json)

    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_key = str(model["key"])
        model_label = str(model["label"])
        prediction_root = model_prediction_root(model, comparison_root)
        metric_rows: list[dict[str, Any]] = []
        for selected_case in selected:
            case_id = str(selected_case["case_id"])
            segmentation_path = Path(str(selected_case["segmentation_path"]))
            prediction_path = prediction_root / case_id / MASK_FILENAME
            truth = load_bool_volume(segmentation_path)
            prediction = load_bool_volume(prediction_path)
            metrics = mask_metrics(truth, prediction)
            row = {
                "case_id": case_id,
                "bin": selected_case.get("bin"),
                "bin_index": selected_case.get("bin_index"),
                "model": model_label,
                "model_column": model_key,
                **metrics,
                "prediction_path": str(prediction_path.resolve()),
                "segmentation_path": str(segmentation_path.resolve()),
            }
            metric_rows.append(row)
            case_rows.append(row)

        precisions = np.asarray([float(row["precision"]) for row in metric_rows])
        recalls = np.asarray([float(row["recall"]) for row in metric_rows])
        dices = np.asarray([float(row["dice"]) for row in metric_rows])
        total_tp = int(sum(int(row["tp"]) for row in metric_rows))
        total_fp = int(sum(int(row["fp"]) for row in metric_rows))
        total_fn = int(sum(int(row["fn"]) for row in metric_rows))
        summary_rows.append(
            {
                "model": model_label,
                "model_column": model_key,
                "n_cases": len(metric_rows),
                "mean_precision": f"{np.mean(precisions):.9f}",
                "median_precision": f"{np.median(precisions):.9f}",
                "mean_recall": f"{np.mean(recalls):.9f}",
                "median_recall": f"{np.median(recalls):.9f}",
                "pooled_precision": f"{total_tp / max(total_tp + total_fp, 1):.9f}",
                "pooled_recall": f"{total_tp / max(total_tp + total_fn, 1):.9f}",
                "mean_dice": f"{np.mean(dices):.9f}",
                "median_dice": f"{np.median(dices):.9f}",
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            }
        )

    write_case_csv(output_csv, case_rows)
    write_summary_csv(summary_csv, summary_rows)
    print(f"selection_json={selection_json}")
    print(f"case_csv={output_csv}")
    print(f"summary_csv={summary_csv}")
    for row in summary_rows:
        print(
            f"{row['model']}: n={row['n_cases']}, "
            f"mean precision={row['mean_precision']}, "
            f"mean recall={row['mean_recall']}, "
            f"median precision={row['median_precision']}, "
            f"median recall={row['median_recall']}, "
            f"pooled precision={row['pooled_precision']}, "
            f"pooled recall={row['pooled_recall']}"
        )


if __name__ == "__main__":
    main()
