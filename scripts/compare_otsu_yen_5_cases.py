"""Build a matched Otsu-versus-Yen comparison for the five BraTS cases.

The five-case Yen export contains native-grid masks, while the completed
251-case Otsu evaluation retained its model-grid median-filtered anomaly maps
in the disk-streaming cache.  The first five cached subjects are the same five
subjects in ``scans_test_5_comparison.csv``.  This script thresholds those
shared score maps with both methods, verifies that the reconstructed Yen mask
is identical to the saved Yen mask, and exports matched native-grid masks and
comparison metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.anomaly.postprocess import threshold_anomaly_map


DEFAULT_SPLIT = Path("splits/BraTS21/scans_test_5_comparison.csv")
DEFAULT_FULL_SPLIT = Path("C:/ML/data/BraTS_2021_healthy_lmdb/scans_test.csv")
DEFAULT_CACHE = Path("outputs/eval_cache/empirical_spectrum233_epoch0232_brats21_full/mf")
DEFAULT_PREDICTIONS = Path(
    "outputs/predictions/"
    "brats21_5_empirical_spectrum233_epoch0232_empirical_noise_dual_yen_masks"
)
DEFAULT_OUTPUT = Path("outputs/comparisons/brats21_5_otsu_vs_yen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--full-split-csv", type=Path, default=DEFAULT_FULL_SPLIT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_subjects(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No subjects found in {path}")
    key = "BraTS21ID" if "BraTS21ID" in rows[0] else next(iter(rows[0]))
    subjects = [str(row[key]).strip() for row in rows]
    if any(not subject for subject in subjects):
        raise ValueError(f"Blank subject ID found in {path}")
    return subjects


def resize_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=shape, mode="nearest")[0, 0]
    return resized.numpy().astype(bool, copy=False)


def save_mask(mask: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8, copy=False), reference.affine, header),
        str(path),
    )


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    dice_denominator = 2 * tp + fp + fn
    return {
        "voxels": int(np.count_nonzero(prediction)),
        "dice": float(2 * tp / dice_denominator) if dice_denominator else 1.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
    }


def pair_metrics(otsu: np.ndarray, yen: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.count_nonzero(otsu & yen))
    union = int(np.count_nonzero(otsu | yen))
    otsu_voxels = int(np.count_nonzero(otsu))
    yen_voxels = int(np.count_nonzero(yen))
    total = otsu_voxels + yen_voxels
    return {
        "intersection_voxels": intersection,
        "union_voxels": union,
        "dice": float(2 * intersection / total) if total else 1.0,
        "jaccard": float(intersection / union) if union else 1.0,
        "otsu_only_voxels": int(np.count_nonzero(otsu & ~yen)),
        "yen_only_voxels": int(np.count_nonzero(yen & ~otsu)),
    }


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def best_slice(target: np.ndarray, otsu: np.ndarray, yen: np.ndarray) -> int:
    areas = target.sum(axis=(0, 1))
    if int(areas.max()) == 0:
        areas = (otsu | yen).sum(axis=(0, 1))
    return int(np.argmax(areas))


def save_figure(
    flair: np.ndarray,
    target: np.ndarray,
    otsu: np.ndarray,
    yen: np.ndarray,
    subject: str,
    output_path: Path,
) -> None:
    z_index = best_slice(target, otsu, yen)
    image = np.asarray(flair[:, :, z_index], dtype=np.float32)
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    scale_values = nonzero if nonzero.size else finite
    if scale_values.size:
        vmin, vmax = np.percentile(scale_values, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(image)), float(np.nanmax(image) + 1.0)

    slices = {
        "Ground truth": target[:, :, z_index],
        "Otsu": otsu[:, :, z_index],
        "Yen": yen[:, :, z_index],
    }
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for axis in axes:
        axis.imshow(image.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axis.axis("off")
    axes[0].set_title(f"FLAIR\nz={z_index}")
    colors = {"Ground truth": "lime", "Otsu": "red", "Yen": "cyan"}
    for axis, (name, mask) in zip(axes[1:4], slices.items()):
        if np.any(mask):
            axis.contour(mask.T, levels=[0.5], colors=[colors[name]], linewidths=1.0)
        axis.set_title(name)

    disagreement = np.zeros((*otsu.shape[:2], 3), dtype=np.float32)
    disagreement[otsu[:, :, z_index] & ~yen[:, :, z_index], 0] = 1.0
    disagreement[yen[:, :, z_index] & ~otsu[:, :, z_index], 2] = 1.0
    agreement = otsu[:, :, z_index] & yen[:, :, z_index]
    disagreement[agreement, 0] = 1.0
    disagreement[agreement, 1] = 1.0
    axes[4].imshow(disagreement.transpose(1, 0, 2), origin="lower", alpha=0.7)
    axes[4].set_title("Mask overlap\nyellow=both, red=Otsu, blue=Yen")
    fig.suptitle(subject)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Five-case Otsu vs Yen comparison",
        "",
        "Both masks were generated from the same median-filtered anomaly score. "
        "Thresholding is per subject and uses a strict `score > threshold` comparator.",
        "",
        "| Case | Otsu Dice vs GT | Yen Dice vs GT | Otsu/Yen Dice | Otsu voxels | Yen voxels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject_id']} | {row['otsu_dice_gt']:.4f} | "
            f"{row['yen_dice_gt']:.4f} | {row['mask_dice']:.4f} | "
            f"{row['otsu_voxels']} | {row['yen_voxels']} |"
        )
    mean_otsu = float(np.mean([row["otsu_dice_gt"] for row in rows]))
    mean_yen = float(np.mean([row["yen_dice_gt"] for row in rows]))
    mean_overlap = float(np.mean([row["mask_dice"] for row in rows]))
    lines.extend(
        [
            "",
            f"Mean Dice vs GT: Otsu **{mean_otsu:.4f}**, Yen **{mean_yen:.4f}**.",
            f"Mean Otsu/Yen mask Dice: **{mean_overlap:.4f}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    subjects = read_subjects(args.split_csv)
    full_subjects = read_subjects(args.full_split_csv)
    if full_subjects[: len(subjects)] != subjects:
        raise ValueError(
            "The five-case split is not the prefix of the full cached split; "
            "cached score-to-subject mapping cannot be verified."
        )

    cache_files = sorted(args.cache_dir.glob("*.npy"))[: len(subjects)]
    if len(cache_files) != len(subjects):
        raise FileNotFoundError(
            f"Expected {len(subjects)} cached score maps in {args.cache_dir}, found {len(cache_files)}."
        )

    cached_scores = [np.load(path, mmap_mode="r") for path in cache_files]
    dataset_min = min(float(score.min()) for score in cached_scores)
    dataset_max = max(float(score.max()) for score in cached_scores)
    dataset_range = dataset_max - dataset_min
    if dataset_range <= 0:
        raise ValueError("The five cached anomaly maps are constant.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for subject, cache_path, score in zip(subjects, cache_files, cached_scores):
        source_case_dir = args.prediction_dir / subject
        metadata = load_metadata(source_case_dir / "prediction_metadata.json")
        yen_reference = nib.load(str(source_case_dir / "lesion_mask_yen_mf.nii.gz"))
        saved_yen = np.asanyarray(yen_reference.dataobj).astype(bool)
        native_shape = tuple(int(value) for value in yen_reference.shape[:3])

        score_tensor = torch.from_numpy(np.array(score, copy=True))[None]
        otsu_model, otsu_threshold = threshold_anomaly_map(score_tensor, method="otsu")
        yen_model, yen_threshold = threshold_anomaly_map(score_tensor, method="yen")
        otsu_native = resize_mask(otsu_model[0].numpy(), native_shape)
        yen_native = resize_mask(yen_model[0].numpy(), native_shape)
        if not np.array_equal(yen_native, saved_yen):
            difference = int(np.count_nonzero(yen_native != saved_yen))
            raise ValueError(
                f"Reconstructed Yen mask for {subject} differs from the saved mask "
                f"at {difference} native-grid voxels."
            )

        segmentation_path = Path(str(metadata["segmentation_path"]))
        target_image = nib.load(str(segmentation_path))
        target = np.asanyarray(target_image.dataobj) > 0
        if target.shape != native_shape:
            raise ValueError(
                f"Ground-truth shape {target.shape} does not match mask shape {native_shape} for {subject}."
            )

        otsu_stats = prediction_metrics(otsu_native, target)
        yen_stats = prediction_metrics(yen_native, target)
        overlap = pair_metrics(otsu_native, yen_native)
        voxel_volume_mm3 = float(abs(np.linalg.det(yen_reference.affine[:3, :3])))
        otsu_threshold_value = float(otsu_threshold[0])
        yen_threshold_value = float(yen_threshold[0])
        otsu_threshold_normalized = (otsu_threshold_value - dataset_min) / dataset_range
        yen_threshold_normalized = (yen_threshold_value - dataset_min) / dataset_range
        saved_yen_threshold = float(metadata["yen_threshold_mf"])
        if not np.isclose(yen_threshold_normalized, saved_yen_threshold, atol=1e-5, rtol=0.0):
            raise ValueError(
                f"Normalized Yen threshold for {subject} ({yen_threshold_normalized}) does not "
                f"match metadata ({saved_yen_threshold})."
            )

        output_case_dir = args.output_dir / subject
        output_case_dir.mkdir(parents=True, exist_ok=True)
        save_mask(otsu_native, yen_reference, output_case_dir / "lesion_mask_otsu_mf.nii.gz")
        save_mask(yen_native, yen_reference, output_case_dir / "lesion_mask_yen_mf.nii.gz")

        flair_path = Path(str(metadata["input_paths"]["flair"]))
        flair = np.asanyarray(nib.load(str(flair_path)).dataobj)
        save_figure(
            flair,
            target,
            otsu_native,
            yen_native,
            subject,
            output_case_dir / "mask_comparison.png",
        )

        row = {
            "subject_id": subject,
            "otsu_threshold": otsu_threshold_normalized,
            "yen_threshold": yen_threshold_normalized,
            "otsu_voxels": otsu_stats["voxels"],
            "yen_voxels": yen_stats["voxels"],
            "otsu_volume_ml": float(otsu_stats["voxels"] * voxel_volume_mm3 / 1000.0),
            "yen_volume_ml": float(yen_stats["voxels"] * voxel_volume_mm3 / 1000.0),
            "otsu_dice_gt": otsu_stats["dice"],
            "yen_dice_gt": yen_stats["dice"],
            "otsu_sensitivity_gt": otsu_stats["sensitivity"],
            "yen_sensitivity_gt": yen_stats["sensitivity"],
            "otsu_precision_gt": otsu_stats["precision"],
            "yen_precision_gt": yen_stats["precision"],
            "mask_dice": overlap["dice"],
            "mask_jaccard": overlap["jaccard"],
            "otsu_only_voxels": overlap["otsu_only_voxels"],
            "yen_only_voxels": overlap["yen_only_voxels"],
        }
        rows.append(row)
        details.append(
            {
                **row,
                "cache_file": str(cache_path),
                "source_yen_mask": str(source_case_dir / "lesion_mask_yen_mf.nii.gz"),
                "ground_truth": str(segmentation_path),
                "yen_reconstruction_exact": True,
                "threshold_comparator": ">",
            }
        )

        (output_case_dir / "comparison_metrics.json").write_text(
            json.dumps(details[-1], indent=2), encoding="utf-8"
        )

    with (args.output_dir / "comparison_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "case_count": len(rows),
                "dataset_min": dataset_min,
                "dataset_max": dataset_max,
                "yen_reconstruction_exact_for_all_cases": True,
                "cases": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(markdown_summary(rows), encoding="utf-8")
    print(markdown_summary(rows))
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
