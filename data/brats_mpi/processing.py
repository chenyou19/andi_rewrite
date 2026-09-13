"""BraTS preprocessing for inference with MPI-trained models.

The cache deliberately keeps the categorical BraTS label map.  A binary whole
tumour target is derived only by the dataset adapter, never during spatial
preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch
from dipy.align.imaffine import AffineMap

from ..imaging import normalize_volume
from ..lemon_mip.registration import (
    RegistrationSettings,
    dipy_pull_to_forward,
    register_affine_multistage,
)

from .geometry import (
    FIXED_RAS_FOV,
    CropWindow,
    crop_resize_xy,
    fraction_inside_window,
    model_grid_affine,
)
from .manifest import BraTSMPIRecord, CHANNEL_ORDER, validate_record_geometry


MODES = frozenset({"mni_affine", "ras_fixed_fov"})
CACHE_SCHEMA_VERSION = 1
PROCESSING_IMPLEMENTATION_VERSION = "brats-mpi-v1-mni-precenter-v2"
VALID_BRATS_LABELS = frozenset({0, 1, 2, 4})


@dataclass(frozen=True)
class ProcessedBraTSSubject:
    record: BraTSMPIRecord
    mode: str
    image: np.ndarray
    segmentation: np.ndarray
    brain_mask: np.ndarray
    metadata: dict[str, Any]
    qc: dict[str, Any]


@dataclass(frozen=True)
class RegistrationEstimate:
    forward_moving_to_static_world: np.ndarray
    precenter_forward_world: np.ndarray
    residual_forward_world: np.ndarray
    mi_before: float
    mi_after: float
    stage_objectives: dict[str, float | None]


def _as_float_image(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("MRI volume must be a finite 3-D array")
    return data


def _as_categorical_segmentation(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    raw = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("segmentation must be a finite 3-D array")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1e-4):
        raise ValueError("segmentation contains non-categorical values")
    labels = {int(value) for value in np.unique(rounded)}
    unexpected = labels.difference(VALID_BRATS_LABELS)
    if unexpected:
        raise ValueError(f"unexpected BraTS labels: {sorted(unexpected)}")
    return rounded.astype(np.uint8, copy=False)


def _load_native_and_ras(record: BraTSMPIRecord) -> tuple[dict[str, nib.Nifti1Image], dict[str, nib.Nifti1Image]]:
    validate_record_geometry(record, check_voxels=True)

    paths = {
        "flair": record.flair_path,
        "t1": record.t1_path,
        "t2": record.t2_path,
        "seg": record.segmentation_path,
    }
    native: dict[str, nib.Nifti1Image] = {}
    ras: dict[str, nib.Nifti1Image] = {}
    for name, path in paths.items():
        loaded = nib.load(str(path))
        native[name] = loaded
        ras[name] = nib.as_closest_canonical(loaded)
    return native, ras


def _normalize_mri(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, volume, 0.0).astype(np.float32, copy=False)
    normalized = normalize_volume(torch.from_numpy(masked)[None])[0].cpu().numpy()
    normalized = np.asarray(normalized, dtype=np.float32)
    normalized[~mask] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("normalization produced non-finite values")
    return normalized


def _nonempty_z(mask: np.ndarray) -> tuple[int, int]:
    occupied = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if occupied.size == 0:
        raise ValueError("brain mask is empty")
    return int(occupied[0]), int(occupied[-1]) + 1


def _mask_volume_mm3(mask: np.ndarray, affine: np.ndarray) -> float:
    voxel_volume = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return float(np.count_nonzero(mask) * voxel_volume)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.count_nonzero(a & b) / denom)


def _mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    values_a = np.asarray(a, dtype=np.float32)
    values_b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        values_a = values_a[mask]
        values_b = values_b[mask]
    else:
        values_a = values_a.ravel()
        values_b = values_b.ravel()
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a, values_b = values_a[finite], values_b[finite]
    if values_a.size < 32:
        return 0.0
    hist, _, _ = np.histogram2d(values_a, values_b, bins=64)
    probability = hist / max(float(hist.sum()), 1.0)
    pa = probability.sum(axis=1, keepdims=True)
    pb = probability.sum(axis=0, keepdims=True)
    expected = pa @ pb
    valid = probability > 0
    return float(np.sum(probability[valid] * np.log(probability[valid] / expected[valid])))


def _dilate_mask_mm(mask: np.ndarray, affine: np.ndarray, radius_mm: float) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    spacing = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
    radii = np.maximum(np.ceil(float(radius_mm) / spacing).astype(int), 1)
    axes = [np.arange(-radius, radius + 1, dtype=np.float32) for radius in radii]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    footprint = (
        (xx * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (zz * spacing[2]) ** 2
        <= float(radius_mm) ** 2
    )
    return binary_dilation(np.asarray(mask, dtype=bool), structure=footprint)


def _resample_affine(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    moving_to_target: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    mapping = AffineMap(
        np.linalg.inv(np.asarray(moving_to_target, dtype=np.float64)),
        domain_grid_shape=tuple(int(v) for v in target_shape),
        domain_grid2world=np.asarray(target_affine, dtype=np.float64),
        codomain_grid_shape=tuple(int(v) for v in moving.shape),
        codomain_grid2world=np.asarray(moving_affine, dtype=np.float64),
    )
    interpolation = "nearest" if order == 0 else "linear"
    # DIPY's compiled nearest-neighbour kernel expects a floating input array;
    # categorical values remain exact and are rounded back to uint8 by callers.
    sampled = mapping.transform(np.asarray(moving, dtype=np.float32), interpolation=interpolation)
    return np.asarray(sampled)


def _register_affine(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    static: np.ndarray,
    static_affine: np.ndarray,
    settings: Mapping[str, Any],
    *,
    static_mask: np.ndarray,
    moving_mask: np.ndarray,
):
    from dipy.align.imaffine import transform_centers_of_mass

    locked = RegistrationSettings(
        nbins=int(settings.get("nbins", 32)),
        sampling_proportion=settings.get("sampling_proportion", None),
        level_iters=tuple(int(v) for v in settings.get("level_iters", [1000, 100, 10])),
        sigmas=tuple(float(v) for v in settings.get("sigmas", [3.0, 1.0, 0.0])),
        factors=tuple(int(v) for v in settings.get("factors", [4, 2, 1])),
    )
    center = transform_centers_of_mass(static, static_affine, moving, moving_affine)
    precenter_forward = dipy_pull_to_forward(center.affine)
    preview = _resample_affine(
        moving,
        moving_affine,
        tuple(int(value) for value in static.shape),
        static_affine,
        precenter_forward,
        order=1,
    ).astype(np.float32, copy=False)
    preview_mask = _resample_affine(
        moving_mask.astype(np.uint8),
        moving_affine,
        tuple(int(value) for value in static.shape),
        static_affine,
        precenter_forward,
        order=0,
    ) > 0.5
    residual = register_affine_multistage(
        static,
        static_affine,
        preview,
        static_affine,
        final_transform="affine",
        settings=locked,
        static_mask=static_mask,
        moving_mask=preview_mask,
    )
    residual_forward = residual.forward_moving_to_static_world
    return RegistrationEstimate(
        forward_moving_to_static_world=residual_forward @ precenter_forward,
        precenter_forward_world=precenter_forward,
        residual_forward_world=residual_forward,
        mi_before=float(residual.mi_before),
        mi_after=float(residual.mi_after),
        stage_objectives=residual.stage_objectives,
    )


def _find_single(root: Path, patterns: tuple[str, ...], *, exclude: tuple[str, ...] = ()) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    filtered = [
        path for path in unique if not any(token.lower() in path.name.lower() for token in exclude)
    ]
    if len(filtered) != 1:
        raise FileNotFoundError(
            f"expected one matching file under {root}, found {len(filtered)}: "
            f"{[str(path) for path in filtered[:8]]}"
        )
    return filtered[0]


def resolve_mni_resources(config: Mapping[str, Any]) -> dict[str, Path]:
    explicit = config.get("mni", {})
    root = Path(
        explicit.get(
            "root",
            Path(config["mpi_preprocess_root"]) / "templates" / "mni2009c",
        )
    )
    if not root.is_dir():
        raise FileNotFoundError(f"MNI resource directory does not exist: {root}")

    t1 = Path(explicit["t1_path"]) if explicit.get("t1_path") else root / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
    mask = Path(explicit["mask_path"]) if explicit.get("mask_path") else root / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"
    roi = Path(explicit["roi_path"]) if explicit.get("roi_path") else root / "fixed_roi.json"
    for path in (t1, mask, roi):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"root": root, "t1": t1, "mask": mask, "roi": roi}


def _read_roi(path: Path) -> CropWindow:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    for container_key in ("crop", "roi", "crop_window"):
        if isinstance(payload.get(container_key), Mapping):
            payload = payload[container_key]
            break
    aliases = {
        "x_start": ("x_start", "x0", "xmin"),
        "x_stop": ("x_stop", "x1", "xmax"),
        "y_start": ("y_start", "y0", "ymin"),
        "y_stop": ("y_stop", "y1", "ymax"),
    }
    values: dict[str, int] = {}
    for name, options in aliases.items():
        for option in options:
            if option in payload:
                values[name] = int(payload[option])
                break
    if len(values) != 4:
        # MPI's stored physical ROI is [-2, 196) x [16, 214) on a 1-mm MNI grid.
        if all(key in payload for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm")):
            values = {
                "x_start": int(round(float(payload["x_min_mm"]))),
                "x_stop": int(round(float(payload["x_max_mm"]))),
                "y_start": int(round(float(payload["y_min_mm"]))),
                "y_stop": int(round(float(payload["y_max_mm"]))),
            }
        else:
            raise ValueError(f"cannot read an XY crop window from {path}")
    return CropWindow(**values)


def _finalize_model_grid(
    images: list[np.ndarray],
    segmentation: np.ndarray,
    brain_mask: np.ndarray,
    affine: np.ndarray,
    window: CropWindow,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    z_start, z_stop = _nonempty_z(brain_mask)
    cropped_images = [
        crop_resize_xy(volume[:, :, z_start:z_stop], window, model_size, is_mask=False)
        for volume in images
    ]
    cropped_seg = crop_resize_xy(
        segmentation[:, :, z_start:z_stop], window, model_size, is_mask=True
    ).astype(np.uint8, copy=False)
    cropped_mask = crop_resize_xy(
        brain_mask[:, :, z_start:z_stop].astype(np.uint8), window, model_size, is_mask=True
    ).astype(np.uint8, copy=False)
    stacked = np.stack(cropped_images, axis=0).astype(np.float32, copy=False)
    out_affine = model_grid_affine(affine, window, model_size, z_start)
    return stacked, cropped_seg, cropped_mask, (z_start, z_stop), out_affine


def _fraction_inside_model_support(
    mask: np.ndarray,
    window: CropWindow,
    z_range: tuple[int, int],
) -> float:
    value = np.asarray(mask, dtype=bool)
    total = int(value.sum())
    if total == 0:
        return 1.0
    x_start = max(window.x_start, 0)
    x_stop = min(window.x_stop, value.shape[0])
    y_start = max(window.y_start, 0)
    y_stop = min(window.y_stop, value.shape[1])
    z_start = max(int(z_range[0]), 0)
    z_stop = min(int(z_range[1]), value.shape[2])
    retained = int(value[x_start:x_stop, y_start:y_stop, z_start:z_stop].sum())
    return float(retained / total)


def _transformed_mask_point_retention(
    mask: np.ndarray,
    moving_affine: np.ndarray,
    forward_world: np.ndarray,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> float:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        return 1.0
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    target_voxels = (
        np.linalg.inv(target_affine) @ forward_world @ moving_affine @ homogeneous.T
    ).T[:, :3]
    upper = np.asarray(target_shape, dtype=np.float64) - 0.5
    inside = np.all((target_voxels >= -0.5) & (target_voxels <= upper), axis=1)
    return float(np.mean(inside))


def _qc_status(metrics: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    return {"status": "PASS" if not failures else "EXCLUDED", "failures": failures, **metrics}


def _common_metadata(
    record: BraTSMPIRecord,
    mode: str,
    native_t1: nib.spatialimages.SpatialImage,
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    model_affine: np.ndarray,
    window: CropWindow,
    z_range: tuple[int, int],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "subject_id": record.subject_id,
        "mode": mode,
        "channel_order": list(CHANNEL_ORDER),
        "label_values": sorted(VALID_BRATS_LABELS),
        "source_shape": [int(v) for v in source_shape],
        "source_affine": np.asarray(source_affine, dtype=float).tolist(),
        "model_affine": np.asarray(model_affine, dtype=float).tolist(),
        "crop_window": window.as_dict(),
        "z_start": int(z_range[0]),
        "z_stop": int(z_range[1]),
        "native_reference_path": str(record.t1_path.resolve()),
        "segmentation_path": str(record.segmentation_path.resolve()),
        "input_paths": record.input_paths,
        "native_shape": [int(v) for v in native_t1.shape[:3]],
        "native_affine": np.asarray(native_t1.affine, dtype=float).tolist(),
    }


def _process_fixed(
    record: BraTSMPIRecord,
    native: Mapping[str, nib.Nifti1Image],
    ras: Mapping[str, nib.Nifti1Image],
    config: Mapping[str, Any],
) -> ProcessedBraTSSubject:
    image_data = {name: _as_float_image(ras[name]) for name in ("flair", "t1", "t2")}
    segmentation = _as_categorical_segmentation(ras["seg"])
    brain_mask = image_data["t1"] > 0
    images = [_normalize_mri(image_data[name], brain_mask) for name in ("flair", "t1", "t2")]
    window = FIXED_RAS_FOV
    model_size = int(config.get("model_size", 128))
    image, model_seg, model_mask, z_range, out_affine = _finalize_model_grid(
        images,
        segmentation,
        brain_mask,
        ras["t1"].affine,
        window,
        model_size,
    )

    foreground_retention = fraction_inside_window(brain_mask, window)
    lesion_retention = _fraction_inside_model_support(segmentation > 0, window, z_range)
    original_lesion = int(np.count_nonzero(segmentation))
    model_lesion = int(np.count_nonzero(model_seg))
    thresholds = config.get("qc", {})
    failures: list[str] = []
    if foreground_retention < float(thresholds.get("foreground_retention_min", 0.999)):
        failures.append("foreground_outside_fixed_fov")
    if lesion_retention < float(thresholds.get("lesion_retention_min", 1.0)) - 1e-8:
        failures.append("lesion_outside_fixed_fov")
    if original_lesion > 0 and model_lesion == 0:
        failures.append("nonempty_gt_became_empty")
    qc = _qc_status(
        {
            "foreground_retention": foreground_retention,
            "lesion_retention": lesion_retention,
            "original_lesion_voxels": original_lesion,
            "model_lesion_voxels": model_lesion,
        },
        failures,
    )
    metadata = _common_metadata(
        record,
        "ras_fixed_fov",
        native["t1"],
        tuple(int(v) for v in ras["t1"].shape[:3]),
        ras["t1"].affine,
        out_affine,
        window,
        z_range,
    )
    metadata["restore_kind"] = "canonical_ras_to_native"
    return ProcessedBraTSSubject(record, "ras_fixed_fov", image, model_seg, model_mask, metadata, qc)


def _process_mni(
    record: BraTSMPIRecord,
    native: Mapping[str, nib.Nifti1Image],
    ras: Mapping[str, nib.Nifti1Image],
    config: Mapping[str, Any],
) -> ProcessedBraTSSubject:
    resources = resolve_mni_resources(config)
    template_image = nib.as_closest_canonical(nib.load(str(resources["t1"])))
    template_mask_image = nib.as_closest_canonical(nib.load(str(resources["mask"])))
    template = _as_float_image(template_image)
    template_mask = _as_float_image(template_mask_image) > 0.5
    moving_data = {name: _as_float_image(ras[name]) for name in ("flair", "t1", "t2")}
    segmentation = _as_categorical_segmentation(ras["seg"])
    moving_mask = moving_data["t1"] > 0
    moving_t1 = _normalize_mri(moving_data["t1"], moving_mask)
    static_t1 = _normalize_mri(template, template_mask)

    registration_result = _register_affine(
        moving_t1,
        ras["t1"].affine,
        static_t1,
        template_image.affine,
        config.get("registration", {}),
        static_mask=template_mask,
        moving_mask=moving_mask,
    )
    native_to_mni = registration_result.forward_moving_to_static_world
    target_shape = tuple(int(v) for v in template_image.shape[:3])
    warped_images = [
        _resample_affine(
            moving_data[name],
            ras[name].affine,
            target_shape,
            template_image.affine,
            native_to_mni,
            order=1,
        ).astype(np.float32, copy=False)
        for name in ("flair", "t1", "t2")
    ]
    warped_brain = _resample_affine(
        moving_mask.astype(np.uint8),
        ras["t1"].affine,
        target_shape,
        template_image.affine,
        native_to_mni,
        order=0,
    ) > 0.5
    # No threshold/binarization is applied here: nearest-neighbour must retain 0/1/2/4.
    warped_seg = np.rint(
        _resample_affine(
            segmentation,
            ras["seg"].affine,
            target_shape,
            template_image.affine,
            native_to_mni,
            order=0,
        )
    ).astype(np.uint8)
    labels = {int(value) for value in np.unique(warped_seg)}
    if not labels.issubset(VALID_BRATS_LABELS):
        raise ValueError(f"nearest-neighbour GT resampling changed labels: {sorted(labels)}")

    guard_radius = float(config.get("mni", {}).get("guard_dilation_mm", 5.0))
    guard = _dilate_mask_mm(template_mask, template_image.affine, guard_radius)
    final_mask = warped_brain & guard
    images = [_normalize_mri(volume, final_mask) for volume in warped_images]
    window = _read_roi(resources["roi"])
    model_size = int(config.get("model_size", 128))
    image, model_seg, model_mask, z_range, out_affine = _finalize_model_grid(
        images,
        warped_seg,
        final_mask,
        template_image.affine,
        window,
        model_size,
    )

    mi_before = float(registration_result.mi_before)
    mi_after = float(registration_result.mi_after)
    affine_linear = native_to_mni[:3, :3]
    singular_values = np.linalg.svd(affine_linear, compute_uv=False)
    determinant = float(np.linalg.det(affine_linear))
    moving_volume = _mask_volume_mm3(moving_mask, ras["t1"].affine)
    warped_volume = _mask_volume_mm3(warped_brain, template_image.affine)
    volume_ratio = warped_volume / max(moving_volume, 1e-8)
    mask_dice = _dice(warped_brain, template_mask)
    guard_retention = float(np.count_nonzero(warped_brain & guard) / max(np.count_nonzero(warped_brain), 1))
    transform_support_retention = _transformed_mask_point_retention(
        segmentation > 0,
        ras["seg"].affine,
        native_to_mni,
        template_image.affine,
        target_shape,
    )
    lesion_retention = _fraction_inside_model_support(warped_seg > 0, window, z_range)
    original_lesion = int(np.count_nonzero(segmentation))
    warped_lesion = int(np.count_nonzero(warped_seg))
    model_lesion = int(np.count_nonzero(model_seg))

    thresholds = config.get("qc", {})
    failures: list[str] = []
    ratio_bounds = thresholds.get("brain_volume_ratio", [0.5, 2.5])
    sv_bounds = thresholds.get("affine_singular_values", [0.25, 4.0])
    if determinant <= 0:
        failures.append("nonpositive_affine_determinant")
    if not (float(ratio_bounds[0]) <= volume_ratio <= float(ratio_bounds[1])):
        failures.append("brain_volume_ratio")
    if singular_values.min() < float(sv_bounds[0]) or singular_values.max() > float(sv_bounds[1]):
        failures.append("affine_singular_values")
    if mask_dice < float(thresholds.get("mni_mask_dice_min", 0.70)):
        failures.append("low_mni_mask_dice")
    if mi_after <= mi_before + float(thresholds.get("mi_improvement_min", 0.0)):
        failures.append("mutual_information_not_improved")
    if guard_retention < float(thresholds.get("guard_retention_min", 0.98)):
        failures.append("brain_outside_mni_guard")
    if lesion_retention < float(thresholds.get("lesion_retention_min", 1.0)) - 1e-8:
        failures.append("lesion_outside_mpi_roi")
    if transform_support_retention < float(thresholds.get("lesion_retention_min", 1.0)) - 1e-8:
        failures.append("lesion_outside_mni_support")
    if original_lesion > 0 and (warped_lesion == 0 or model_lesion == 0):
        failures.append("nonempty_gt_became_empty")
    qc = _qc_status(
        {
            "affine_determinant": determinant,
            "affine_singular_values": singular_values.astype(float).tolist(),
            "brain_volume_ratio": volume_ratio,
            "mni_mask_dice": mask_dice,
            "mi_before": mi_before,
            "mi_after": mi_after,
            "mi_improvement": mi_after - mi_before,
            "registration_stage_objectives": registration_result.stage_objectives,
            "guard_retention": guard_retention,
            "lesion_retention": lesion_retention,
            "transform_support_retention": transform_support_retention,
            "original_lesion_voxels": original_lesion,
            "warped_lesion_voxels": warped_lesion,
            "model_lesion_voxels": model_lesion,
        },
        failures,
    )
    metadata = _common_metadata(
        record,
        "mni_affine",
        native["t1"],
        target_shape,
        template_image.affine,
        out_affine,
        window,
        z_range,
    )
    metadata.update(
        {
            "restore_kind": "mni_affine_to_native",
            "native_ras_shape": [int(v) for v in ras["t1"].shape[:3]],
            "native_ras_affine": np.asarray(ras["t1"].affine, dtype=float).tolist(),
            "native_ras_to_mni": native_to_mni.astype(float).tolist(),
            "registration_precenter_forward": registration_result.precenter_forward_world.astype(float).tolist(),
            "registration_residual_forward": registration_result.residual_forward_world.astype(float).tolist(),
            "mni_t1_path": str(resources["t1"]),
            "mni_mask_path": str(resources["mask"]),
            "mni_roi_path": str(resources["roi"]),
        }
    )
    return ProcessedBraTSSubject(record, "mni_affine", image, model_seg, model_mask, metadata, qc)


def process_subject(
    record: BraTSMPIRecord,
    mode: str,
    config: Mapping[str, Any],
) -> ProcessedBraTSSubject:
    """Process one subject without writing cache files."""

    if mode not in MODES:
        raise ValueError(f"unsupported preprocessing mode {mode!r}; expected one of {sorted(MODES)}")
    native, ras = _load_native_and_ras(record)
    if mode == "ras_fixed_fov":
        return _process_fixed(record, native, ras, config)
    return _process_mni(record, native, ras, config)
