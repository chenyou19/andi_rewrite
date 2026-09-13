"""Registration metrics, automatic status rules, outliers, and visual QC."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else float(2 * np.count_nonzero(a & b) / denominator)


def affine_qc_metrics(forward_moving_to_static_world: np.ndarray) -> dict[str, float]:
    try:
        from scipy.linalg import polar
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Affine QC requires SciPy.") from exc

    affine = np.asarray(forward_moving_to_static_world, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("Affine QC requires a finite 4x4 matrix.")
    linear = affine[:3, :3]
    determinant = float(np.linalg.det(linear))
    rotation_matrix, stretch = polar(linear)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1
        stretch[-1, :] *= -1
    rotation = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    singular_values = np.sort(np.linalg.svd(linear, compute_uv=False))[::-1]
    diagonal = np.maximum(np.abs(np.diag(stretch)), 1.0e-12)
    normalized_stretch = stretch / np.sqrt(diagonal[:, None] * diagonal[None, :])
    shear_xy = float(normalized_stretch[0, 1])
    shear_xz = float(normalized_stretch[0, 2])
    shear_yz = float(normalized_stretch[1, 2])
    translation = affine[:3, 3]
    return {
        "tx": float(translation[0]),
        "ty": float(translation[1]),
        "tz": float(translation[2]),
        "translation_norm": float(np.linalg.norm(translation)),
        "rx": float(rotation[0]),
        "ry": float(rotation[1]),
        "rz": float(rotation[2]),
        "rotation_norm": float(np.linalg.norm(rotation)),
        "det": determinant,
        "sv1": float(singular_values[0]),
        "sv2": float(singular_values[1]),
        "sv3": float(singular_values[2]),
        "sv_min": float(singular_values.min()),
        "sv_max": float(singular_values.max()),
        "anisotropy_ratio": float(singular_values.max() / max(singular_values.min(), 1.0e-12)),
        "shear_xy": shear_xy,
        "shear_xz": shear_xz,
        "shear_yz": shear_yz,
        "shear_norm": float(math.sqrt(shear_xy**2 + shear_xz**2 + shear_yz**2)),
    }


def automatic_qc_status(metrics: Mapping[str, object]) -> tuple[str, list[str]]:
    flags: list[str] = []
    hard_arrays = ("image_finite", "mask_finite", "geometry_valid", "shape_valid")
    for name in hard_arrays:
        if not bool(metrics.get(name, False)):
            flags.append(f"FAIL:{name}")
    brain_volume = float(metrics.get("brain_volume_L", float("nan")))
    determinant = float(metrics.get("det", float("nan")))
    sv_min = float(metrics.get("sv_min", float("nan")))
    sv_max = float(metrics.get("sv_max", float("nan")))
    mask_dice = float(metrics.get("mask_dice", float("nan")))
    if not math.isfinite(brain_volume) or not 0.5 <= brain_volume <= 2.5:
        flags.append("FAIL:brain_volume_outside_0.5_2.5_L")
    if not math.isfinite(determinant) or determinant <= 0:
        flags.append("FAIL:nonpositive_or_nonfinite_determinant")
    if not math.isfinite(sv_min) or not math.isfinite(sv_max) or sv_min < 0.25 or sv_max > 4.0:
        flags.append("FAIL:catastrophic_singular_value")
    if not math.isfinite(mask_dice) or mask_dice < 0.70:
        flags.append("FAIL:mask_dice_below_0.70")
    for name in ("T1_MNI_MI_improvement", "FLAIR_T1_MI_improvement", "T2_T1_MI_improvement"):
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value) or value <= 0:
            flags.append(f"REVIEW:{name}_nonpositive")
    retention = float(metrics.get("guard_retention", float("nan")))
    if not math.isfinite(retention) or retention < 0.98:
        flags.append("REVIEW:guard_retention_below_0.98")
    if any(flag.startswith("FAIL:") for flag in flags):
        return "FAIL", flags
    if flags:
        return "REVIEW", flags
    return "PASS", []


def _modified_z(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if not np.any(finite):
        return result
    median = float(np.median(values[finite]))
    deviation = np.abs(values[finite] - median)
    mad = float(np.median(deviation))
    if mad <= 1.0e-12:
        result[finite] = np.where(deviation <= 1.0e-12, 0.0, np.inf)
    else:
        result[finite] = 0.67448975 * (values[finite] - median) / mad
    return result


def apply_dataset_outliers(
    rows: list[dict[str, object]],
    *,
    threshold: float = 3.5,
    suspicious_count: int = 15,
) -> list[dict[str, object]]:
    fields = (
        "mask_dice",
        "T1_MNI_MI_improvement",
        "FLAIR_T1_MI_improvement",
        "T2_T1_MI_improvement",
        "translation_norm",
        "rotation_norm",
        "det",
        "anisotropy_ratio",
        "shear_norm",
    )
    scores = np.zeros(len(rows), dtype=np.float64)
    worst_ranks = np.full(len(rows), len(rows) + 1, dtype=np.int64)
    per_field: dict[str, np.ndarray] = {}
    for field in fields:
        values = np.asarray([float(row.get(field, float("nan"))) for row in rows], dtype=np.float64)
        z = _modified_z(values)
        per_field[field] = z
        absolute = np.nan_to_num(np.abs(z), nan=0.0, posinf=1.0e12)
        scores = np.maximum(scores, absolute)
        field_order = np.argsort(-absolute, kind="stable")
        field_ranks = np.empty(len(rows), dtype=np.int64)
        field_ranks[field_order] = np.arange(1, len(rows) + 1)
        worst_ranks = np.minimum(worst_ranks, field_ranks)
    # A case's best (smallest) rank on any robust metric is its aggregate
    # suspicious rank; maximum |modified-z| breaks ties deterministically.
    ordering = np.lexsort((-scores, worst_ranks))
    suspicious = set(int(index) for index in ordering[: min(suspicious_count, len(rows))])
    for index, row in enumerate(rows):
        row["outlier_score"] = float(scores[index])
        row["worst_metric_rank"] = int(worst_ranks[index])
        row["suspicious_rank"] = (
            int(np.where(ordering == index)[0][0]) + 1 if index in suspicious else None
        )
        outlier_flags = [
            f"REVIEW:outlier_{field}_modified_z={per_field[field][index]:.3f}"
            for field in fields
            if np.isfinite(per_field[field][index]) and abs(per_field[field][index]) >= threshold
        ]
        existing = [item for item in str(row.get("QC_flags", "")).split(";") if item]
        combined = existing + outlier_flags
        if outlier_flags and row.get("QC_status") == "PASS":
            row["QC_status"] = "REVIEW"
        row["QC_flags"] = ";".join(combined)
    return [rows[int(index)] for index in ordering[: min(suspicious_count, len(rows))]]


def _display_scale(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(value)
    return np.clip((value - low) / (high - low), 0.0, 1.0)


def _center_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        return tuple(int(value // 2) for value in mask.shape)
    center = np.median(coordinates, axis=0)
    return tuple(int(round(value)) for value in center)


def save_registration_qc_figure(
    output_path: str | Path,
    static: np.ndarray,
    moving_on_static: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
    metrics_text: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    center = _center_indices(mask)
    views = (
        (static[center[0], :, :], moving_on_static[center[0], :, :], "sagittal"),
        (static[:, center[1], :], moving_on_static[:, center[1], :], "coronal"),
        (static[:, :, center[2]], moving_on_static[:, :, center[2]], "axial"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for column, (fixed, moving, view) in enumerate(views):
        fixed = _display_scale(fixed)
        moving = _display_scale(moving)
        checker = fixed.copy()
        tile = 12
        yy, xx = np.indices(checker.shape)
        selector = ((yy // tile) + (xx // tile)) % 2 == 1
        checker[selector] = moving[selector]
        axes[0, column].imshow(np.rot90(checker), cmap="gray")
        axes[0, column].set_title(f"{view} checkerboard")
        overlay = np.stack((fixed, moving, fixed), axis=-1)
        axes[1, column].imshow(np.rot90(overlay))
        axes[1, column].set_title(f"{view} overlay")
        for axis in axes[:, column]:
            axis.axis("off")
    figure.suptitle(f"{title}\n{metrics_text}", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_mask_qc_figure(
    output_path: str | Path,
    masks: Mapping[str, np.ndarray],
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 4), squeeze=False)
    for index, (name, mask) in enumerate(masks.items()):
        z = _center_indices(mask)[2]
        axes[0, index].imshow(np.rot90(np.asarray(mask)[:, :, z]), cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(f"{name} (z={z})")
        axes[0, index].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_flair_montage(
    output_path: str | Path,
    flair: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nonempty = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if nonempty.size == 0:
        raise ValueError("Cannot create a FLAIR montage from an empty mask.")
    positions = np.linspace(0, nonempty.size - 1, 9).round().astype(int)
    slices = nonempty[positions]
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    for axis, z in zip(axes.flat, slices):
        axis.imshow(np.rot90(_display_scale(flair[:, :, int(z)])), cmap="gray")
        axis.set_title(f"z={int(z)}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_edge_mask_overlay(
    output_path: str | Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = _center_indices(mask)[2]
    background = _display_scale(np.asarray(image)[:, :, z])
    mask_slice = np.asarray(mask, dtype=bool)[:, :, z]
    edge = mask_slice & ~binary_erosion(mask_slice)
    rgb = np.repeat(background[..., None], 3, axis=-1)
    rgb[edge] = (1.0, 0.1, 0.1)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(np.rot90(rgb))
    axis.set_title(f"{title} (z={z})")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_fixed_roi_slices(
    output_path: str | Path,
    slices: Iterable[tuple[int, np.ndarray]],
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = list(slices)
    if not values:
        raise ValueError("Cannot plot fixed ROI slices from an empty sequence.")
    selected = np.linspace(0, len(values) - 1, 3).round().astype(int)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, item_index in enumerate(selected):
        z, image = values[int(item_index)]
        for channel, modality in enumerate(("FLAIR", "T1", "T2")):
            axes[row, channel].imshow(np.rot90(_display_scale(image[channel])), cmap="gray")
            axes[row, channel].set_title(f"{modality} z={z}")
            axes[row, channel].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


__all__ = [
    "affine_qc_metrics",
    "apply_dataset_outliers",
    "automatic_qc_status",
    "dice_score",
    "save_edge_mask_overlay",
    "save_fixed_roi_slices",
    "save_flair_montage",
    "save_mask_qc_figure",
    "save_registration_qc_figure",
]
