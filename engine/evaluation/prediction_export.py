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
        subject_dir = output_dir / subject_id
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

        def restore(item: torch.Tensor, continuous: bool) -> torch.Tensor:
            if not restore_native:
                return item
            return resize_volume(item, native_shape, continuous)

        raw_score = restore(processed.score_raw[batch_index], continuous=True)
        mf_score = restore(processed.score_mf[batch_index], continuous=True)
        binary_mask_raw = restore(
            processed.binary_mask_raw_postprocessed[batch_index].float(),
            continuous=False,
        ).bool()
        binary_mask_mf = restore(
            processed.binary_mask_mf_postprocessed[batch_index].float(),
            continuous=False,
        ).bool()
        binary_mask = binary_mask_raw if use_raw_binary_mask else binary_mask_mf
        threshold_mask = restore(threshold_masks[batch_index].float(), continuous=False).bool()

        finite_raw = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        finite_mf = torch.nan_to_num(mf_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if bool(prediction_output.get("save_raw_score", True)):
            save_image(
                finite_raw.numpy(),
                reference_image,
                subject_dir / "anomaly_score_raw.nii.gz",
                np.float32,
            )
        if bool(prediction_output.get("save_median_filtered_score", True)):
            save_image(
                finite_mf.numpy(),
                reference_image,
                subject_dir / "anomaly_score_mf.nii.gz",
                np.float32,
            )
        save_binary_mask = prediction_output.get("save_binary_mask")
        if save_binary_mask is None:
            save_binary_mask = prediction_output.get("save_yen_mask", True)
        if bool(save_binary_mask):
            save_image(
                binary_mask_raw.cpu().numpy().astype(np.uint8),
                reference_image,
                subject_dir / f"lesion_mask_{threshold_method}_raw.nii.gz",
                np.uint8,
            )
            save_image(
                binary_mask_mf.cpu().numpy().astype(np.uint8),
                reference_image,
                subject_dir / f"lesion_mask_{threshold_method}_mf.nii.gz",
                np.uint8,
            )
            save_image(
                binary_mask.cpu().numpy().astype(np.uint8),
                reference_image,
                subject_dir / f"lesion_mask_{threshold_method}.nii.gz",
                np.uint8,
            )
        if bool(prediction_output.get("save_threshold_mask", False)):
            save_image(
                threshold_mask.cpu().numpy().astype(np.uint8),
                reference_image,
                subject_dir / "lesion_mask_threshold.nii.gz",
                np.uint8,
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
        metadata_path = subject_dir / "prediction_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")
    return prediction_index
