"""Per-session LEMON preprocessing with one final interpolation per modality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..imaging import normalize_volume
from .geometry import FixedSquareROI, resize_roi_slice
from .manifest import MODALITY_ORDER, SessionRecord
from .qc import affine_qc_metrics, automatic_qc_status, dice_score
from .registration import (
    RegistrationSettings,
    compose_forward_transforms,
    load_canonical_nifti,
    register_affine_multistage,
    resample_with_forward_transform,
)


@dataclass
class ProcessedSession:
    record: SessionRecord
    slices: list[tuple[int, np.ndarray]]
    metrics: dict[str, Any]
    intensity_statistics: dict[str, Any]
    transforms: dict[str, np.ndarray]
    registration_images: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


def _require_same_geometry(
    name: str,
    shape: tuple[int, ...],
    affine: np.ndarray,
    reference_shape: tuple[int, ...],
    reference_affine: np.ndarray,
) -> None:
    if shape != reference_shape or not np.allclose(affine, reference_affine, rtol=0.0, atol=1.0e-4):
        raise ValueError(
            f"{name} geometry does not match native T1: shape={shape}/{reference_shape}, "
            f"affine_max_error={float(np.max(np.abs(affine - reference_affine))):.6g}."
        )


def _dilate_mask_mm(mask: np.ndarray, affine: np.ndarray, distance_mm: float) -> np.ndarray:
    try:
        import nibabel as nib
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Physical mask dilation requires nibabel and SciPy.") from exc
    value = np.asarray(mask, dtype=bool)
    spacing = nib.affines.voxel_sizes(np.asarray(affine, dtype=np.float64))
    distance = distance_transform_edt(~value, sampling=spacing)
    return (value | (distance <= float(distance_mm))).astype(np.uint8)


def _channel_statistics(
    normalized: np.ndarray,
    final_mask: np.ndarray,
    roi: FixedSquareROI,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {"channel_order": list(MODALITY_ORDER)}
    crop = np.zeros(
        (normalized.shape[0], *roi.crop_shape, normalized.shape[-1]), dtype=normalized.dtype
    )
    source_x_start = max(roi.x_start, 0)
    source_x_stop = min(roi.x_stop, normalized.shape[1])
    source_y_start = max(roi.y_start, 0)
    source_y_stop = min(roi.y_stop, normalized.shape[2])
    destination_x_start = source_x_start - roi.x_start
    destination_y_start = source_y_start - roi.y_start
    crop[
        :,
        destination_x_start : destination_x_start + source_x_stop - source_x_start,
        destination_y_start : destination_y_start + source_y_stop - source_y_start,
        :,
    ] = normalized[:, source_x_start:source_x_stop, source_y_start:source_y_stop, :]
    for index, modality in enumerate(MODALITY_ORDER):
        masked = normalized[index][final_mask]
        if masked.size == 0:
            raise ValueError(f"Final mask contains no values for {modality}.")
        statistics[modality] = {
            "masked_percentiles": {
                str(percentile): float(np.percentile(masked, percentile))
                for percentile in (0, 1, 25, 50, 75, 95, 99, 100)
            },
            "roi_fraction_lt_0": float(np.mean(crop[index] < 0)),
            "roi_fraction_gt_1": float(np.mean(crop[index] > 1)),
            "roi_fraction_eq_0": float(np.mean(crop[index] == 0)),
        }
    return statistics


def process_session(
    record: SessionRecord,
    brain_mask_path: str | Path,
    mni_t1: np.ndarray,
    mni_mask: np.ndarray,
    mni_affine: np.ndarray,
    roi: FixedSquareROI,
    *,
    registration_settings: RegistrationSettings,
    guard_dilation_mm: float = 5.0,
) -> ProcessedSession:
    """Register, mask, normalize, and slice one session.

    Registration QC may use intermediate resampling, but each exported modality is
    sampled directly from its canonical native grid to the MNI grid exactly once.
    """

    flair, flair_affine = load_canonical_nifti(record.flair_path)
    t1, t1_affine = load_canonical_nifti(record.t1_path)
    t2, t2_affine = load_canonical_nifti(record.t2_path)
    native_mask_data, native_mask_affine = load_canonical_nifti(brain_mask_path)
    native_mask = native_mask_data > 0.5
    if not np.any(native_mask):
        raise ValueError(f"HD-BET mask is empty for {record.session_id}.")
    _require_same_geometry(
        "HD-BET mask", native_mask.shape, native_mask_affine, t1.shape, t1_affine
    )

    mni_t1 = np.asarray(mni_t1, dtype=np.float32)
    mni_mask = np.asarray(mni_mask, dtype=bool)
    mni_affine = np.asarray(mni_affine, dtype=np.float64)
    if mni_t1.shape != mni_mask.shape or tuple(mni_t1.shape) != roi.template_shape:
        raise ValueError("MNI template, mask, and fixed ROI geometry disagree.")

    t1_brain = np.where(native_mask, t1, 0.0).astype(np.float32)
    static_mni = np.where(mni_mask, mni_t1, 0.0).astype(np.float32)
    # The native acquisitions share physical subject space but not necessarily
    # voxel grids. Project the trusted T1 HD-BET mask into each moving grid so
    # COM/MI is driven by brain anatomy rather than face/neck coverage. These
    # arrays are registration inputs only; final exports still resample the
    # original native modalities exactly once with the composed transform.
    flair_registration_mask = resample_with_forward_transform(
        native_mask.astype(np.float32),
        t1_affine,
        flair.shape,
        flair_affine,
        np.eye(4),
        interpolation="nearest",
    ).astype(bool)
    t2_registration_mask = resample_with_forward_transform(
        native_mask.astype(np.float32),
        t1_affine,
        t2.shape,
        t2_affine,
        np.eye(4),
        interpolation="nearest",
    ).astype(bool)
    if not np.any(flair_registration_mask) or not np.any(t2_registration_mask):
        raise ValueError(f"Projected moving registration mask is empty for {record.session_id}.")
    flair_brain = np.where(flair_registration_mask, flair, 0.0).astype(np.float32)
    t2_brain = np.where(t2_registration_mask, t2, 0.0).astype(np.float32)
    flair_t1 = register_affine_multistage(
        t1_brain,
        t1_affine,
        flair_brain,
        flair_affine,
        final_transform="rigid",
        settings=registration_settings,
        static_mask=native_mask.astype(np.uint8),
        moving_mask=flair_registration_mask.astype(np.uint8),
    )
    t2_t1 = register_affine_multistage(
        t1_brain,
        t1_affine,
        t2_brain,
        t2_affine,
        final_transform="rigid",
        settings=registration_settings,
        static_mask=native_mask.astype(np.uint8),
        moving_mask=t2_registration_mask.astype(np.uint8),
    )
    t1_mni = register_affine_multistage(
        static_mni,
        mni_affine,
        t1_brain,
        t1_affine,
        final_transform="affine",
        settings=registration_settings,
        static_mask=mni_mask.astype(np.uint8),
        moving_mask=native_mask.astype(np.uint8),
    )

    flair_to_mni = compose_forward_transforms(
        flair_t1.forward_moving_to_static_world,
        t1_mni.forward_moving_to_static_world,
    )
    t2_to_mni = compose_forward_transforms(
        t2_t1.forward_moving_to_static_world,
        t1_mni.forward_moving_to_static_world,
    )
    final_images = np.stack(
        [
            resample_with_forward_transform(
                flair,
                flair_affine,
                mni_t1.shape,
                mni_affine,
                flair_to_mni,
                interpolation="linear",
            ),
            resample_with_forward_transform(
                t1,
                t1_affine,
                mni_t1.shape,
                mni_affine,
                t1_mni.forward_moving_to_static_world,
                interpolation="linear",
            ),
            resample_with_forward_transform(
                t2,
                t2_affine,
                mni_t1.shape,
                mni_affine,
                t2_to_mni,
                interpolation="linear",
            ),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    warped_subject_mask = resample_with_forward_transform(
        native_mask.astype(np.float32),
        t1_affine,
        mni_t1.shape,
        mni_affine,
        t1_mni.forward_moving_to_static_world,
        interpolation="nearest",
    ).astype(bool)
    guard = _dilate_mask_mm(mni_mask, mni_affine, guard_dilation_mm).astype(bool)
    final_mask = warped_subject_mask & guard
    if not np.any(final_mask):
        raise ValueError(f"Final subject/template mask is empty for {record.session_id}.")

    final_images[:, ~final_mask] = 0.0
    normalized_tensor = normalize_volume(torch.from_numpy(final_images.copy()))
    normalized = normalized_tensor.cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"Normalization produced NaN/Inf for {record.session_id}.")

    nonempty_z = np.flatnonzero(np.any(final_mask, axis=(0, 1)))
    slices: list[tuple[int, np.ndarray]] = []
    for z in nonempty_z:
        value = resize_roi_slice(normalized[:, :, :, int(z)], roi, is_mask=False)
        slice_mask = resize_roi_slice(final_mask[:, :, int(z)], roi, is_mask=True).astype(bool)
        value[:, ~slice_mask] = 0.0
        if value.shape != (3, roi.output_size, roi.output_size):
            raise ValueError(f"Unexpected slice shape for {record.session_id}/z={z}: {value.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite exported slice for {record.session_id}/z={z}.")
        slices.append((int(z), value.astype(np.float32, copy=False)))
    if not slices:
        raise ValueError(f"No non-empty axial slices for {record.session_id}.")

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Brain volume QC requires nibabel.") from exc
    voxel_volume_mm3 = float(abs(np.linalg.det(t1_affine[:3, :3])))
    affine_metrics = affine_qc_metrics(t1_mni.forward_moving_to_static_world)
    mask_dice = dice_score(warped_subject_mask, mni_mask)
    guard_retention = float(final_mask.sum() / max(int(warped_subject_mask.sum()), 1))
    metrics: dict[str, Any] = {
        "csv_index": int(record.csv_index),
        "case_id": record.case_id,
        "session": record.session,
        "session_id": record.session_id,
        "split": record.split,
        "image_finite": bool(np.all(np.isfinite(final_images))),
        "mask_finite": True,
        "geometry_valid": True,
        "shape_valid": bool(final_images.shape == (3, *mni_t1.shape)),
        "brain_volume_L": float(native_mask.sum() * voxel_volume_mm3 / 1_000_000.0),
        "mask_dice": mask_dice,
        "guard_retention": guard_retention,
        "native_mask_voxels": int(native_mask.sum()),
        "warped_mask_voxels": int(warped_subject_mask.sum()),
        "guard_mask_voxels": int(guard.sum()),
        "final_mask_voxels": int(final_mask.sum()),
        "slice_count": len(slices),
        "first_z": int(nonempty_z.min()),
        "last_z": int(nonempty_z.max()),
        "FLAIR_T1_MI_before": flair_t1.mi_before,
        "FLAIR_T1_MI_after": flair_t1.mi_after,
        "FLAIR_T1_MI_improvement": flair_t1.mi_improvement,
        "T2_T1_MI_before": t2_t1.mi_before,
        "T2_T1_MI_after": t2_t1.mi_after,
        "T2_T1_MI_improvement": t2_t1.mi_improvement,
        "T1_MNI_MI_before": t1_mni.mi_before,
        "T1_MNI_MI_after": t1_mni.mi_after,
        "T1_MNI_MI_improvement": t1_mni.mi_improvement,
        **affine_metrics,
    }
    status, flags = automatic_qc_status(metrics)
    metrics["QC_status"] = status
    metrics["QC_flags"] = ";".join(flags)

    transforms = {
        "FLAIR_to_T1_world": flair_t1.forward_moving_to_static_world,
        "T2_to_T1_world": t2_t1.forward_moving_to_static_world,
        "T1_to_MNI_world": t1_mni.forward_moving_to_static_world,
        "FLAIR_to_MNI_world": flair_to_mni,
        "T2_to_MNI_world": t2_to_mni,
        "FLAIR_to_T1_dipy_pull": flair_t1.pull_static_to_moving_world,
        "T2_to_T1_dipy_pull": t2_t1.pull_static_to_moving_world,
        "T1_to_MNI_dipy_pull": t1_mni.pull_static_to_moving_world,
    }
    registration_images = {
        "native_t1_brain": t1_brain,
        "flair_on_t1_qc": flair_t1.transformed,
        "t2_on_t1_qc": t2_t1.transformed,
        "mni_t1_brain": static_mni,
        "t1_on_mni": final_images[1],
        "flair_on_mni": final_images[0],
        "t2_on_mni": final_images[2],
    }
    masks = {
        "native_hdbet": native_mask.astype(np.uint8),
        "warped_subject": warped_subject_mask.astype(np.uint8),
        "mni_guard_5mm": guard.astype(np.uint8),
        "final": final_mask.astype(np.uint8),
    }
    intensity_statistics = _channel_statistics(normalized, final_mask, roi)
    return ProcessedSession(
        record=record,
        slices=slices,
        metrics=metrics,
        intensity_statistics=intensity_statistics,
        transforms=transforms,
        registration_images=registration_images,
        masks=masks,
    )


def save_transform_artifacts(
    result: ProcessedSession,
    output_dir: str | Path,
    registration_settings: RegistrationSettings,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "world_transforms.npz"
    metadata_path = output_dir / "world_transforms.json"
    if matrix_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Transform artifacts already exist in {output_dir}.")
    np.savez_compressed(matrix_path, **result.transforms)
    metadata = {
        "case_id": result.record.case_id,
        "session": result.record.session,
        "convention": {
            "*_to_*_world": "moving-world to static-world forward matrix",
            "*_dipy_pull": "static-world to moving-world sampling matrix used by DIPY AffineMap",
            "final_resampling": "inverse(forward) used as pull matrix; image linear, mask nearest",
        },
        "composition": {
            "FLAIR_to_MNI": "T1_to_MNI @ FLAIR_to_T1",
            "T2_to_MNI": "T1_to_MNI @ T2_to_T1",
        },
        "registration": registration_settings.as_dict(),
        "stage_objectives_are_qc_only": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix_path, metadata_path


__all__ = ["ProcessedSession", "process_session", "save_transform_artifacts"]
