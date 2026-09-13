"""Prediction-map postprocessing and NIfTI/JSON artifact export."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from andi_rewrite.anomaly.postprocess import PostprocessPolicy, PostprocessResult

from .fingerprints import json_safe
from .inputs import safe_subject_id


def resize_volume_to_shape(
    volume: torch.Tensor,
    shape: tuple[int, int, int],
    continuous: bool,
) -> torch.Tensor:
    """Restore one score or mask volume with the existing interpolation choices."""

    if tuple(volume.shape) == tuple(shape):
        return volume
    tensor = volume[None, None].float()
    if continuous:
        resized = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
    else:
        resized = F.interpolate(tensor, size=shape, mode="nearest")
    return resized[0, 0]


def load_reference_image(metadata: dict[str, Any]) -> Any | None:
    """Load the reference NIfTI image required for native-grid export."""

    reference_path = metadata.get("reference_path")
    if not reference_path:
        return None
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("prediction_output NIfTI export requires nibabel.") from exc
    path = Path(str(reference_path))
    if not path.exists():
        raise FileNotFoundError(f"Reference NIfTI for prediction export does not exist: {path}")
    return nib.load(str(path))


def save_nifti(array: np.ndarray, reference_image: Any, path: Path, dtype: Any) -> None:
    """Save one output map with the source affine/header and requested dtype."""

    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_image.header.copy()
    header.set_data_dtype(dtype)
    # Export arrays are already converted to their target dtype by the caller.
    # Avoid an unconditional full-volume copy here: a native BraTS float32
    # volume is ~34 MiB, and duplicating it can exhaust RAM after memory-heavy
    # empirical-spectrum inference.
    output_array = np.asarray(array, dtype=dtype)
    nib.save(nib.Nifti1Image(output_array, reference_image.affine, header), str(path))


def load_spatial_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Load a validated BraTS-MPI spatial sidecar when the dataset provides one."""

    path_value = metadata.get("spatial_metadata_path")
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"Spatial metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("qc", {}).get("status") != "PASS":
        raise ValueError(f"Spatial metadata is not a PASS cache entry: {path}")
    expected_subject = metadata.get("subject_id")
    if expected_subject and payload.get("subject_id") != expected_subject:
        raise ValueError(f"Spatial metadata subject mismatch: {path}")
    expected_fingerprint = metadata.get("cache_fingerprint")
    if expected_fingerprint and payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Spatial metadata fingerprint mismatch: {path}")
    return payload


def model_grid_reference_image(
    spatial_metadata: dict[str, Any], shape: tuple[int, int, int]
) -> Any:
    import nibabel as nib

    affine = np.asarray(spatial_metadata["model_affine"], dtype=np.float64)
    return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)


def restore_spatial_volume(
    volume: torch.Tensor,
    spatial_metadata: dict[str, Any],
    *,
    continuous: bool,
) -> torch.Tensor:
    """Undo MPI crop/resize and resample a prediction onto original BraTS grid."""

    import nibabel as nib
    from nibabel.processing import resample_from_to

    from andi_rewrite.data.brats_mpi.geometry import crop_window_from_mapping, restore_model_to_source

    source_shape = tuple(int(value) for value in spatial_metadata["source_shape"])
    window = crop_window_from_mapping(spatial_metadata["crop_window"])
    source = restore_model_to_source(
        volume.detach().cpu().numpy(),
        source_shape=source_shape,
        window=window,
        z_start=int(spatial_metadata["z_start"]),
        continuous=continuous,
    )
    source_affine = np.asarray(spatial_metadata["source_affine"], dtype=np.float64)
    restore_kind = str(spatial_metadata["restore_kind"])
    if restore_kind == "mni_affine_to_native":
        native_to_mni = np.asarray(spatial_metadata["native_ras_to_mni"], dtype=np.float64)
        source_affine = np.linalg.inv(native_to_mni) @ source_affine
    elif restore_kind != "canonical_ras_to_native":
        raise ValueError(f"Unsupported spatial restore kind: {restore_kind}")

    source_image = nib.Nifti1Image(np.asarray(source, dtype=np.float32), source_affine)
    native_image = nib.load(str(spatial_metadata["native_reference_path"]))
    restored = resample_from_to(
        source_image,
        (native_image.shape[:3], native_image.affine),
        order=1 if continuous else 0,
        mode="constant",
        cval=0.0,
    ).get_fdata(dtype=np.float32)
    if continuous:
        return torch.from_numpy(np.asarray(restored, dtype=np.float32))
    return torch.from_numpy(np.asarray(restored > 0.5, dtype=bool))


def prediction_postprocess(
    raw_maps: torch.Tensor,
    metric_processed: PostprocessResult | None,
    *,
    prediction_normalization_scope: str,
    process_raw_maps: Callable[..., PostprocessResult],
) -> PostprocessResult:
    """Materialize a prediction result while preserving subject-scope ordering."""

    if (
        metric_processed is not None
        and prediction_normalization_scope == metric_processed.normalization_scope
    ):
        return metric_processed
    if prediction_normalization_scope == "subject":
        return PostprocessResult.concatenate(
            [
                process_raw_maps(raw_maps[index : index + 1], normalization_scope="subject")
                for index in range(raw_maps.shape[0])
            ]
        )
    return process_raw_maps(raw_maps, normalization_scope="dataset")


def export_predictions(
    raw_maps: torch.Tensor,
    metadata_items: list[dict[str, Any]],
    *,
    processed: PostprocessResult,
    prediction_output: dict[str, Any],
    metric_threshold: float,
    model_config: dict[str, Any],
    detector: Any,
    postprocess_policy: PostprocessPolicy,
    prediction_normalization_scope: str,
    prediction_index: int,
    safe_subject_id_callback: Callable[[Any], str] | None = None,
    load_reference_image_callback: Callable[[dict[str, Any]], Any | None] | None = None,
    save_nifti_callback: Callable[[np.ndarray, Any, Path, Any], None] | None = None,
    resize_volume_callback: Callable[[torch.Tensor, tuple[int, int, int], bool], torch.Tensor] | None = None,
    json_safe_callback: Callable[[Any], Any] | None = None,
) -> int:
    """Export one batch of prediction products and return the next index."""

    output_dir = Path(prediction_output.get("directory", "outputs/predictions"))
    subject_id_for = safe_subject_id_callback or safe_subject_id
    load_reference = load_reference_image_callback or load_reference_image
    save_image = save_nifti_callback or save_nifti
    resize_volume = resize_volume_callback or resize_volume_to_shape
    make_json_safe = json_safe_callback or json_safe
    restore_native = bool(prediction_output.get("restore_native_grid", True))
    save_model_grid = bool(prediction_output.get("save_model_grid", False))
    binary_mask_source = str(
        prediction_output.get(
            "binary_mask_source",
            prediction_output.get("yen_source", "score_mf"),
        )
    ).strip().lower()
    use_raw_binary_mask = binary_mask_source in {"raw", "score_raw", "anomaly_score_raw"}
    selected_thresholds = processed.thresholds_raw if use_raw_binary_mask else processed.thresholds_mf
    threshold_method = processed.threshold_method
    threshold = float(prediction_output.get("threshold", metric_threshold))
    threshold_source = str(prediction_output.get("threshold_source", "score_mf")).lower()
    threshold_scores = (
        processed.score_raw
        if threshold_source in {"raw", "score_raw", "anomaly_score_raw"}
        else processed.score_mf
    )
    threshold_masks = postprocess_policy.fixed_threshold_mask(threshold_scores, threshold)
    batch_size = raw_maps.shape[0]
    for batch_index in range(batch_size):
        metadata = metadata_items[batch_index] if batch_index < len(metadata_items) else {}
        prediction_index += 1
        subject_id = subject_id_for(metadata.get("subject_id", f"subject_{prediction_index:04d}"))
        subject_root = output_dir / subject_id
        spatial_metadata = load_spatial_metadata(metadata)
        if spatial_metadata is not None and restore_native:
            subject_dir = subject_root / "native_grid"
        elif spatial_metadata is not None and save_model_grid:
            subject_dir = subject_root / "model_grid"
        else:
            subject_dir = subject_root
        reference_image = load_reference(metadata)
        if reference_image is None:
            if restore_native:
                raise ValueError("prediction_output.restore_native_grid requires metadata.reference_path.")
            try:
                import nibabel as nib
            except ImportError as exc:
                raise ImportError("prediction_output NIfTI export requires nibabel.") from exc
            reference_image = nib.Nifti1Image(
                np.zeros(tuple(raw_maps[batch_index].shape), dtype=np.float32),
                np.eye(4),
            )

        native_shape = tuple(int(item) for item in reference_image.shape[:3])
        model_shape = tuple(int(item) for item in raw_maps[batch_index].shape)
        model_reference = (
            model_grid_reference_image(spatial_metadata, model_shape)
            if spatial_metadata is not None
            else None
        )

        def restore(item: torch.Tensor, continuous: bool) -> torch.Tensor:
            if not restore_native:
                return item
            if spatial_metadata is not None:
                return restore_spatial_volume(item, spatial_metadata, continuous=continuous)
            return resize_volume(item, native_shape, continuous)

        raw_score_model = processed.score_raw[batch_index]
        mf_score_model = processed.score_mf[batch_index]
        binary_mask_raw_model = processed.binary_mask_raw_postprocessed[batch_index].bool()
        binary_mask_mf_model = processed.binary_mask_mf_postprocessed[batch_index].bool()
        binary_mask_model = binary_mask_raw_model if use_raw_binary_mask else binary_mask_mf_model
        threshold_mask_model = threshold_masks[batch_index].bool()

        raw_score = restore(raw_score_model, continuous=True)
        mf_score = restore(mf_score_model, continuous=True)
        binary_mask_raw = restore(
            binary_mask_raw_model.float(),
            continuous=False,
        ).bool()
        binary_mask_mf = restore(
            binary_mask_mf_model.float(),
            continuous=False,
        ).bool()
        binary_mask = binary_mask_raw if use_raw_binary_mask else binary_mask_mf
        threshold_mask = restore(threshold_masks[batch_index].float(), continuous=False).bool()

        finite_raw = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        finite_mf = torch.nan_to_num(mf_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        save_binary_mask = prediction_output.get("save_binary_mask")
        if save_binary_mask is None:
            save_binary_mask = prediction_output.get("save_yen_mask", True)

        def save_products(
            target_dir: Path,
            target_reference: Any,
            score_raw: torch.Tensor,
            score_mf: torch.Tensor,
            mask_raw: torch.Tensor,
            mask_mf: torch.Tensor,
            mask_selected: torch.Tensor,
            mask_threshold: torch.Tensor,
        ) -> None:
            if bool(prediction_output.get("save_raw_score", True)):
                save_image(
                    torch.nan_to_num(score_raw.float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
                    target_reference,
                    target_dir / "anomaly_score_raw.nii.gz",
                    np.float32,
                )
            if bool(prediction_output.get("save_median_filtered_score", True)):
                save_image(
                    torch.nan_to_num(score_mf.float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
                    target_reference,
                    target_dir / "anomaly_score_mf.nii.gz",
                    np.float32,
                )
            if bool(save_binary_mask):
                for suffix, value in (
                    (f"{threshold_method}_raw", mask_raw),
                    (f"{threshold_method}_mf", mask_mf),
                    (threshold_method, mask_selected),
                ):
                    save_image(
                        value.cpu().numpy().astype(np.uint8),
                        target_reference,
                        target_dir / f"lesion_mask_{suffix}.nii.gz",
                        np.uint8,
                    )
            if bool(prediction_output.get("save_threshold_mask", False)):
                save_image(
                    mask_threshold.cpu().numpy().astype(np.uint8),
                    target_reference,
                    target_dir / "lesion_mask_threshold.nii.gz",
                    np.uint8,
                )

        save_products(
            subject_dir,
            reference_image if restore_native else (model_reference or reference_image),
            finite_raw,
            finite_mf,
            binary_mask_raw,
            binary_mask_mf,
            binary_mask,
            threshold_mask,
        )
        if spatial_metadata is not None and save_model_grid and restore_native:
            save_products(
                subject_root / "model_grid",
                model_reference,
                raw_score_model,
                mf_score_model,
                binary_mask_raw_model,
                binary_mask_mf_model,
                binary_mask_model,
                threshold_mask_model,
            )

        payload = {
            "subject_id": subject_id,
            "location": metadata.get("location"),
            "split": metadata.get("split"),
            "input_paths": metadata.get("input_paths", {}),
            "native_shape": list(native_shape),
            "model_shape": list(model_shape),
            "export_shape": list(native_shape if restore_native else model_shape),
            "restored_to_native_grid": restore_native,
            "model_grid_saved": bool(spatial_metadata is not None and save_model_grid),
            "spatial_metadata_path": metadata.get("spatial_metadata_path"),
            "spatial_restore_kind": spatial_metadata.get("restore_kind") if spatial_metadata else None,
            "reference_modality": metadata.get("reference_modality"),
            "reference_path": metadata.get("reference_path"),
            "segmentation_path": metadata.get("segmentation_path"),
            "brain_mask_path": metadata.get("brain_mask_path"),
            "checkpoint": model_config.get("checkpoint"),
            "use_ema": model_config.get("use_ema"),
            "anomaly_timestep": {
                "t_lower": detector.t_lower,
                "t_upper": detector.t_upper,
            },
            "modality_mapping": metadata.get("modality_mapping", {}),
            "resampled_modalities": metadata.get("resampled_modalities", ""),
            "postprocess_mode": postprocess_policy.mode,
            "normalization_scope": processed.normalization_scope,
            "threshold_method": threshold_method,
            "binary_mask_source": "score_raw" if use_raw_binary_mask else "score_mf",
            "threshold_raw": processed.thresholds_raw[batch_index].item()
            if processed.thresholds_raw.numel() > batch_index
            else None,
            "threshold_mf": processed.thresholds_mf[batch_index].item()
            if processed.thresholds_mf.numel() > batch_index
            else None,
            "threshold": selected_thresholds[batch_index].item()
            if selected_thresholds.numel() > batch_index
            else None,
            "postprocessing": postprocess_policy.describe(),
            "prediction_output": {
                **prediction_output,
                "normalization_scope": prediction_normalization_scope,
            },
        }
        if threshold_method == "yen":
            payload.update(
                {
                    "yen_source": payload["binary_mask_source"],
                    "yen_threshold_raw": payload["threshold_raw"],
                    "yen_threshold_mf": payload["threshold_mf"],
                    "yen_threshold": payload["threshold"],
                }
            )
        metadata_path = (subject_root if spatial_metadata is not None else subject_dir) / "prediction_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")
    return prediction_index
