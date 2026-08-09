from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.anomaly.postprocess import (
    BasePostprocessor,
    MedianFilterPostprocessor,
    NormalizePostprocessor,
    OriginalANDiPostprocessPolicy,
    PostprocessPolicy,
    PostprocessResult,
    RewritePostprocessPolicy,
    apply_postprocess_pipeline,
    build_postprocess_policy,
    sanitize_scores,
)
from andi_rewrite.engine.evaluation_cache import (
    DiskEvaluationCache,
    file_identity,
    stable_fingerprint,
)
from andi_rewrite.metrics.classification import (
    auprc,
    dice,
    external_auprc,
    resolve_auprc_mode,
)
from andi_rewrite.utils.progress import ProgressReporter


@dataclass(frozen=True)
class _DatasetPipelinePlan:
    prefix: list[BasePostprocessor]
    normalizer: NormalizePostprocessor | None
    suffix: list[BasePostprocessor]

    @classmethod
    def from_steps(
        cls,
        steps: list[BasePostprocessor],
        *,
        branch: str,
    ) -> "_DatasetPipelinePlan":
        positions = [
            index for index, step in enumerate(steps) if isinstance(step, NormalizePostprocessor)
        ]
        if len(positions) > 1:
            raise ValueError(
                "disk_streaming currently supports at most one dataset normalization "
                f"step per score branch; {branch!r} contains {len(positions)}."
            )
        if not positions:
            return cls(prefix=list(steps), normalizer=None, suffix=[])
        position = positions[0]
        return cls(
            prefix=list(steps[:position]),
            normalizer=steps[position],  # type: ignore[arg-type]
            suffix=list(steps[position + 1 :]),
        )

    def prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return apply_postprocess_pipeline(tensor, self.prefix)

    def finish(
        self,
        prepared: torch.Tensor,
        bounds: tuple[float, float] | None,
    ) -> torch.Tensor:
        output = sanitize_scores(prepared)
        if self.normalizer is not None:
            if bounds is None:
                raise RuntimeError("Dataset normalization bounds have not been prepared.")
            minimum, maximum = bounds
            denominator = torch.as_tensor(
                maximum - minimum,
                dtype=output.dtype,
                device=output.device,
            ).clamp_min(float(self.normalizer.eps))
            output = (output - float(minimum)) / denominator
        return apply_postprocess_pipeline(output, self.suffix)

    def process(
        self,
        tensor: torch.Tensor,
        bounds: tuple[float, float] | None,
    ) -> torch.Tensor:
        return self.finish(self.prepare(tensor), bounds)


class VolumeEvaluator:
    """完整 volume ANDi evaluator，輸出格式相容原版 CSV。

    evaluator 負責 volume/slice reshape、threshold sweep、後處理變體與 CSV 寫出。
    anomaly scoring 交給 ANDiDetector，讓 evaluation policy 與 detection math
    可以分開擴充。
    """

    def __init__(
        self,
        detector: ANDiDetector,
        config: dict[str, Any],
        accelerator: Any = None,
    ):
        self.detector = detector
        self.config = config
        self.accelerator = accelerator
        self.size_splits = int(config.get("size_splits", config.get("slice_batch_size", 155)))
        self.normalize_input = bool(config.get("normalize_input", True))
        self.output_csv = Path(config.get("output_csv", config.get("output", "outputs/metrics/ANDi.csv")))
        self.output_mf_csv = Path(config.get("output_mf_csv", config.get("output_mf", "outputs/metrics/ANDi_mf.csv")))
        self.threshold_start = float(config.get("thr_start", config.get("threshold_start", 0.01)))
        self.threshold_end = float(config.get("thr_end", config.get("threshold_end", 0.3)))
        self.threshold_step = float(config.get("thr_step", config.get("threshold_step", 0.001)))
        self.metric_threshold = float(config.get("threshold", 0.5))
        self.compute_auprc = bool(config.get("compute_auprc", True))
        auprc_max_samples = config.get("auprc_max_samples")
        self.auprc_mode, self.auprc_max_samples = resolve_auprc_mode(
            config.get("auprc_mode"),
            auprc_max_samples,
            enabled=self.compute_auprc,
        )
        self.auprc_seed = int(config.get("auprc_seed", 73))
        self.memory_mode = str(config.get("memory_mode", "in_memory")).strip().lower()
        if self.memory_mode not in {"in_memory", "disk_streaming"}:
            raise ValueError(
                "evaluation.memory_mode must be 'in_memory' or 'disk_streaming', "
                f"got {self.memory_mode!r}."
            )
        cache_config = config.get("cache", {})
        if not isinstance(cache_config, dict):
            raise TypeError("evaluation.cache must be a mapping.")
        self.cache_config = cache_config
        self.cache_directory = cache_config.get("directory")
        self.cache_resume = bool(cache_config.get("resume", True))
        self.cache_keep_on_success = bool(cache_config.get("keep_on_success", True))
        external_sort_config = config.get("external_sort", {})
        if not isinstance(external_sort_config, dict):
            raise TypeError("evaluation.external_sort must be a mapping.")
        self.external_sort_chunk_bytes = int(
            external_sort_config.get("chunk_bytes", 256 * 1024 * 1024)
        )
        if self.external_sort_chunk_bytes <= 0:
            raise ValueError("evaluation.external_sort.chunk_bytes must be positive.")
        if self.memory_mode == "disk_streaming":
            if accelerator is not None:
                raise ValueError(
                    "evaluation.memory_mode='disk_streaming' currently requires a single "
                    "process; disable runtime.accelerate/distributed."
                )
            if self.cache_directory in (None, ""):
                raise ValueError(
                    "evaluation.memory_mode='disk_streaming' requires "
                    "evaluation.cache.directory."
                )
        self.rank = int(config.get("rank", 3))
        self.connectivity = int(config.get("connectivity", 1))
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", config.get("median_enabled", True)))
        self.kernel_size = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        self.prediction_output = config.get("prediction_output", {})
        self.prediction_enabled = bool(self.prediction_output.get("enabled", False))
        self.model_config = config.get("model", {})
        self.anomaly_config = config.get("anomaly", getattr(detector, "config", {}))
        candidate_policy = build_postprocess_policy(
            config,
            anomaly_config=self.anomaly_config,
            warn_legacy=getattr(detector, "postprocess_policy", None) is None,
            legacy_profile="evaluator",
        )
        detector_policy = getattr(detector, "postprocess_policy", None)
        if (
            isinstance(detector_policy, PostprocessPolicy)
            and detector_policy.describe() == candidate_policy.describe()
        ):
            self.postprocess_policy = detector_policy
        else:
            self.postprocess_policy = candidate_policy
        if hasattr(detector, "set_postprocess_policy"):
            detector.set_postprocess_policy(self.postprocess_policy)
        else:
            detector.postprocess_policy = self.postprocess_policy  # type: ignore[attr-defined]
        self.metric_normalization_scope = self.postprocess_policy.normalization_scope
        self.threshold_method = self.postprocess_policy.threshold_method
        default_prediction_scope = (
            "dataset" if self.postprocess_policy.mode == "original_andi" else "subject"
        )
        prediction_scope_config = self.prediction_output.get(
            "normalization_scope", default_prediction_scope
        )
        if isinstance(prediction_scope_config, dict):
            prediction_scope_config = prediction_scope_config.get(
                self.postprocess_policy.mode,
                default_prediction_scope,
            )
        self.prediction_normalization_scope = str(prediction_scope_config).strip().lower()
        if self.prediction_normalization_scope not in {"dataset", "subject"}:
            raise ValueError(
                "prediction_output.normalization_scope must be 'dataset' or 'subject', "
                f"got {self.prediction_normalization_scope!r}."
            )
        self._prediction_index = 0
        self.last_processed: PostprocessResult | None = None
        self.last_prediction_processed: PostprocessResult | None = None

    def _progress_enabled(self) -> bool:
        progress_config = self.config.get("progress", True)
        if isinstance(progress_config, dict):
            return bool(progress_config.get("enabled", True)) and self.is_main_process
        return bool(progress_config) and self.is_main_process

    @property
    def is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def prepare(self, dataloader: Iterable) -> Iterable:
        if self.accelerator is None:
            return dataloader
        model, dataloader = self.accelerator.prepare(self.detector.model, dataloader)
        self.detector.model = model
        return dataloader

    def _slice_scores(self, images: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """計算 [N, C, H, W] slices 的 raw anomaly scores。"""

        scores = []
        chunks = list(torch.split(images, self.size_splits))
        progress_enabled = self._progress_enabled()
        volume_label = f"Volume {volume_index}" if volume_index is not None else "Volume"
        chunk_bar = ProgressReporter(
            len(chunks),
            f"{volume_label} chunks",
            enabled=progress_enabled,
            unit="chunk",
            leave=False,
        )
        try:
            for chunk_index, chunk in enumerate(chunks, start=1):
                deviations = self.detector.compute_deviation_stack(
                    chunk,
                    progress=progress_enabled,
                    progress_description=f"{volume_label} chunk {chunk_index} timesteps",
                    progress_leave=False,
                )
                per_modality = self.detector.aggregate_time(deviations)
                scores.append(self.detector.pool_modalities(per_modality).detach())
                chunk_bar.update()
        finally:
            chunk_bar.close()
        return torch.cat(scores, dim=0)

    def _volume_scores(self, image: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """將 [B, C, H, W, Z] 轉成 raw anomaly maps [B, H, W, Z]。"""

        image = image.to(self.detector.device)
        if self.normalize_input:
            image = image * 2.0 - 1.0
        batch_size, _, height, width, slices = image.shape
        # detector 以 2D slice 工作；先沿 slice 維度 flatten，逐 slice scoring 後再還原 volume。
        flat = image.permute(0, 4, 1, 2, 3).reshape(-1, image.shape[1], height, width)
        flat_scores = self._slice_scores(flat, volume_index=volume_index)
        return flat_scores.view(batch_size, slices, height, width).permute(0, 2, 3, 1).contiguous()

    def _split_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        if isinstance(batch, dict):
            image = batch.get("image")
            label = batch.get("label", batch.get("mask"))
            metadata = batch.get("metadata", batch.get("meta"))
            if image is None:
                raise ValueError("Dictionary batches must contain an 'image' key.")
            return image, label, metadata
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Empty evaluation batch.")
            image = batch[0]
            label = batch[1] if len(batch) > 1 else None
            metadata = batch[2] if len(batch) > 2 else None
            return image, label, metadata
        raise TypeError(f"Unsupported evaluation batch type: {type(batch)!r}")

    def _metadata_items(self, metadata: Any, batch_size: int) -> list[dict[str, Any]]:
        if metadata is None:
            return [{} for _ in range(batch_size)]
        return [self._metadata_item(metadata, index, batch_size) for index in range(batch_size)]

    def _metadata_item(self, value: Any, index: int, batch_size: int) -> Any:
        if isinstance(value, dict):
            return {key: self._metadata_item(item, index, batch_size) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            item = value
            if item.ndim > 0 and item.shape[0] == batch_size:
                item = item[index]
            if item.ndim == 0:
                return item.item()
            return item.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            item = value
            if item.ndim > 0 and item.shape[0] == batch_size:
                item = item[index]
            return item.item() if item.ndim == 0 else item.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                return self._metadata_item(value[index], 0, 1)
            return [self._metadata_item(item, index, batch_size) for item in value]
        return value

    @staticmethod
    def _truthy(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _safe_subject_id(value: Any) -> str:
        text = str(value or "subject").strip()
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
        return safe or "subject"

    @staticmethod
    def _shape_from_text(value: Any) -> tuple[int, ...] | None:
        if isinstance(value, str) and value:
            try:
                return tuple(int(part) for part in value.split(",") if part)
            except ValueError:
                return None
        if isinstance(value, (list, tuple)):
            try:
                return tuple(int(part) for part in value)
            except (TypeError, ValueError):
                return None
        return None

    def _has_label(self, label: torch.Tensor | None, metadata: dict[str, Any]) -> bool:
        if label is None:
            return False
        return self._truthy(metadata.get("has_label"), default=True)

    def process_raw_maps(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        """Run the shared Detector/Evaluator postprocessing policy once."""

        return self.postprocess_policy.process(
            raw_maps.float(),
            normalization_scope=normalization_scope or self.metric_normalization_scope,
        )

    def _processed_prediction_maps(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Backward-compatible dictionary view of the shared policy result."""

        processed = self.process_raw_maps(
            raw_maps,
            normalization_scope=normalization_scope or self.prediction_normalization_scope,
        )
        threshold = float(self.prediction_output.get("threshold", self.metric_threshold))
        threshold_source = str(self.prediction_output.get("threshold_source", "score_mf")).lower()
        threshold_score = (
            processed.score_raw
            if threshold_source in {"raw", "score_raw", "anomaly_score_raw"}
            else processed.score_mf
        )
        threshold_mask = self.postprocess_policy.fixed_threshold_mask(threshold_score, threshold)
        output = {
            "score_raw": processed.score_raw,
            "score_mf": processed.score_mf,
            "binary_mask": processed.binary_mask_mf_postprocessed,
            "thresholds": processed.thresholds_mf,
            "binary_mask_raw": processed.binary_mask_raw_postprocessed,
            "thresholds_raw": processed.thresholds_raw,
            "threshold_mask": threshold_mask,
        }
        if processed.threshold_method == "yen":
            output.update(
                {
                    "yen_mask": processed.binary_mask_mf_postprocessed,
                    "yen_thresholds": processed.thresholds_mf,
                    "yen_mask_raw": processed.binary_mask_raw_postprocessed,
                    "yen_thresholds_raw": processed.thresholds_raw,
                }
            )
        return output

    @staticmethod
    def _resize_volume_to_shape(volume: torch.Tensor, shape: tuple[int, int, int], continuous: bool) -> torch.Tensor:
        if tuple(volume.shape) == tuple(shape):
            return volume
        tensor = volume[None, None].float()
        if continuous:
            resized = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
        else:
            resized = F.interpolate(tensor, size=shape, mode="nearest")
        return resized[0, 0]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): VolumeEvaluator._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [VolumeEvaluator._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _load_reference_image(self, metadata: dict[str, Any]):
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

    def _save_nifti(self, array: np.ndarray, reference_image: Any, path: Path, dtype: Any) -> None:
        import nibabel as nib

        path.parent.mkdir(parents=True, exist_ok=True)
        header = reference_image.header.copy()
        header.set_data_dtype(dtype)
        # Export arrays are already converted to their target dtype by the
        # caller. Avoid an unconditional full-volume copy here: a native
        # BraTS float32 volume is ~34 MiB, and duplicating it can exhaust RAM
        # after memory-heavy empirical-spectrum inference.
        output_array = np.asarray(array, dtype=dtype)
        nib.save(nib.Nifti1Image(output_array, reference_image.affine, header), str(path))

    def _prediction_postprocess(
        self,
        raw_maps: torch.Tensor,
        metric_processed: PostprocessResult | None = None,
    ) -> PostprocessResult:
        if (
            metric_processed is not None
            and self.prediction_normalization_scope == metric_processed.normalization_scope
        ):
            return metric_processed
        if self.prediction_normalization_scope == "subject":
            return PostprocessResult.concatenate(
                [
                    self.process_raw_maps(raw_maps[index : index + 1], normalization_scope="subject")
                    for index in range(raw_maps.shape[0])
                ]
            )
        return self.process_raw_maps(raw_maps, normalization_scope="dataset")

    def _export_predictions(
        self,
        raw_maps: torch.Tensor,
        metadata_items: list[dict[str, Any]],
        processed: PostprocessResult | None = None,
    ) -> None:
        if not self.prediction_enabled or not self.is_main_process:
            return
        output_dir = Path(self.prediction_output.get("directory", "outputs/predictions"))
        restore_native = bool(self.prediction_output.get("restore_native_grid", True))
        processed = processed or self._prediction_postprocess(raw_maps.detach().cpu())
        self.last_prediction_processed = processed
        binary_mask_source = str(
            self.prediction_output.get(
                "binary_mask_source",
                self.prediction_output.get("yen_source", "score_mf"),
            )
        ).strip().lower()
        use_raw_binary_mask = binary_mask_source in {"raw", "score_raw", "anomaly_score_raw"}
        selected_thresholds = (
            processed.thresholds_raw if use_raw_binary_mask else processed.thresholds_mf
        )
        threshold_method = processed.threshold_method
        threshold = float(self.prediction_output.get("threshold", self.metric_threshold))
        threshold_source = str(self.prediction_output.get("threshold_source", "score_mf")).lower()
        threshold_scores = (
            processed.score_raw
            if threshold_source in {"raw", "score_raw", "anomaly_score_raw"}
            else processed.score_mf
        )
        threshold_masks = self.postprocess_policy.fixed_threshold_mask(threshold_scores, threshold)
        batch_size = raw_maps.shape[0]
        for batch_index in range(batch_size):
            metadata = metadata_items[batch_index] if batch_index < len(metadata_items) else {}
            self._prediction_index += 1
            subject_id = self._safe_subject_id(metadata.get("subject_id", f"subject_{self._prediction_index:04d}"))
            subject_dir = output_dir / subject_id
            reference_image = self._load_reference_image(metadata)
            if reference_image is None:
                if restore_native:
                    raise ValueError("prediction_output.restore_native_grid requires metadata.reference_path.")
                try:
                    import nibabel as nib
                except ImportError as exc:
                    raise ImportError("prediction_output NIfTI export requires nibabel.") from exc
                reference_image = nib.Nifti1Image(np.zeros(tuple(raw_maps[batch_index].shape), dtype=np.float32), np.eye(4))

            native_shape = tuple(int(item) for item in reference_image.shape[:3])
            model_shape = tuple(int(item) for item in raw_maps[batch_index].shape)

            def restore(item: torch.Tensor, continuous: bool) -> torch.Tensor:
                if not restore_native:
                    return item
                return self._resize_volume_to_shape(item, native_shape, continuous=continuous)

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
            if bool(self.prediction_output.get("save_raw_score", True)):
                self._save_nifti(finite_raw.numpy(), reference_image, subject_dir / "anomaly_score_raw.nii.gz", np.float32)
            if bool(self.prediction_output.get("save_median_filtered_score", True)):
                self._save_nifti(finite_mf.numpy(), reference_image, subject_dir / "anomaly_score_mf.nii.gz", np.float32)
            save_binary_mask = self.prediction_output.get("save_binary_mask")
            if save_binary_mask is None:
                save_binary_mask = self.prediction_output.get("save_yen_mask", True)
            if bool(save_binary_mask):
                self._save_nifti(
                    binary_mask_raw.cpu().numpy().astype(np.uint8),
                    reference_image,
                    subject_dir / f"lesion_mask_{threshold_method}_raw.nii.gz",
                    np.uint8,
                )
                self._save_nifti(
                    binary_mask_mf.cpu().numpy().astype(np.uint8),
                    reference_image,
                    subject_dir / f"lesion_mask_{threshold_method}_mf.nii.gz",
                    np.uint8,
                )
                self._save_nifti(
                    binary_mask.cpu().numpy().astype(np.uint8),
                    reference_image,
                    subject_dir / f"lesion_mask_{threshold_method}.nii.gz",
                    np.uint8,
                )
            if bool(self.prediction_output.get("save_threshold_mask", False)):
                self._save_nifti(
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
                "checkpoint": self.model_config.get("checkpoint"),
                "use_ema": self.model_config.get("use_ema"),
                "anomaly_timestep": {
                    "t_lower": self.detector.t_lower,
                    "t_upper": self.detector.t_upper,
                },
                "modality_mapping": metadata.get("modality_mapping", {}),
                "resampled_modalities": metadata.get("resampled_modalities", ""),
                "postprocess_mode": self.postprocess_policy.mode,
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
                "postprocessing": self.postprocess_policy.describe(),
                "prediction_output": {
                    **self.prediction_output,
                    "normalization_scope": self.prediction_normalization_scope,
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
            metadata_path.write_text(json.dumps(self._json_safe(payload), indent=2), encoding="utf-8")

    def collect(self, dataloader: Iterable) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
        maps = []
        labels = []
        label_flags: list[bool] = []
        collected_metadata: list[dict[str, Any]] = []
        try:
            total = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            total = 0
        volume_bar = ProgressReporter(
            total,
            "Evaluating volumes",
            enabled=self._progress_enabled(),
            unit="volume",
        )
        with torch.no_grad():
            try:
                for volume_index, batch in enumerate(dataloader, start=1):
                    image, label, metadata = self._split_batch(batch)
                    metadata_items = self._metadata_items(metadata, int(image.shape[0]))
                    anomaly_map = self._volume_scores(image, volume_index=volume_index)
                    label_available = [self._has_label(label, item) for item in metadata_items]
                    if label is not None:
                        label = label.to(anomaly_map.device).bool()
                    if self.accelerator is not None:
                        if label is not None:
                            anomaly_map, label = self.accelerator.gather_for_metrics((anomaly_map, label))
                        else:
                            anomaly_map = self.accelerator.gather_for_metrics(anomaly_map)
                        # Keep Python metadata in the same rank order as the
                        # gathered tensors. Truncation mirrors gather_for_metrics
                        # dropping duplicated padding from the final batch.
                        from accelerate.utils import gather_object

                        gathered_count = int(anomaly_map.shape[0])
                        metadata_items = list(gather_object(metadata_items))[:gathered_count]
                        label_available = list(gather_object(label_available))[:gathered_count]
                    maps.append(anomaly_map.detach().cpu())
                    if label is not None:
                        labels.append(label.detach().cpu())
                    label_flags.extend(label_available)
                    collected_metadata.extend(metadata_items)
                    volume_bar.update()
            finally:
                volume_bar.close()
        label_tensor = torch.cat(labels, dim=0) if labels and all(label_flags) else None
        return torch.cat(maps, dim=0), label_tensor, collected_metadata

    def threshold_values(self) -> list[float]:
        values = np.arange(self.threshold_start, self.threshold_end, self.threshold_step)
        return [round(float(value), 3) for value in values]

    def _dice_mean(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
        values = []
        for pred_item, label_item in zip(prediction, label):
            values.append(dice(pred_item, label_item))
        return float(np.mean(values)) if values else 0.0

    def _binary_rates(self, prediction: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        prediction = prediction.bool()
        label = label.bool()
        tp = torch.logical_and(prediction, label).sum().item()
        tn = torch.logical_and(~prediction, ~label).sum().item()
        fp = torch.logical_and(prediction, ~label).sum().item()
        fn = torch.logical_and(~prediction, label).sum().item()
        return {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
        }

    def _segmentation_metrics(self, segmentation: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        rates = self._binary_rates(segmentation, label)
        return {
            "dice": self._dice_mean(segmentation, label),
            "sensitivity": rates["sensitivity"],
            "precision": rates["precision"],
        }

    def _threshold_method_metrics(
        self,
        segmentation: torch.Tensor,
        thresholds: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, float]:
        metrics = self._segmentation_metrics(segmentation, label)
        metrics["threshold"] = float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        return metrics

    def _summarize_without_labels(
        self,
        processed: PostprocessResult,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        scores: dict[Any, Any] = {}
        scores_mf: dict[Any, Any] = {}
        for threshold in self.threshold_values():
            scores[threshold] = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}
            scores_mf[threshold] = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}

        method = processed.threshold_method
        scores[method] = "N/A"
        scores[f"{method}thr"] = (
            float(processed.thresholds_raw.float().mean().item())
            if processed.thresholds_raw.numel()
            else 0.0
        )
        scores[f"{method}sen"] = "N/A"
        scores[f"{method}pre"] = "N/A"
        scores_mf[method] = "N/A"
        scores_mf[f"{method}thr"] = (
            float(processed.thresholds_mf.float().mean().item())
            if processed.thresholds_mf.numel()
            else 0.0
        )
        scores_mf[f"{method}sen"] = "N/A"
        scores_mf[f"{method}pre"] = "N/A"
        if self.compute_auprc:
            scores["AUPRC"] = "N/A"
            scores_mf["AUPRC"] = "N/A"
            scores["AUPRC_mode"] = self.auprc_mode
            scores_mf["AUPRC_mode"] = self.auprc_mode
        scores.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
        scores_mf.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
        return scores, scores_mf

    def summarize(self, raw_maps: torch.Tensor, labels: torch.Tensor | None) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """針對 raw 與 MF map 計算 threshold sweep 與選定自適應 threshold 指標。"""

        processed = self.process_raw_maps(raw_maps, self.metric_normalization_scope)
        self.last_processed = processed
        return self.summarize_processed(processed, labels)

    def summarize_processed(
        self,
        processed: PostprocessResult,
        labels: torch.Tensor | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """Compute every metric from an already materialized shared result."""

        anomaly_map = processed.score_raw
        anomaly_map_mf = processed.score_mf

        if labels is None:
            return self._summarize_without_labels(processed)

        scores = {}
        scores_mf = {}
        thresholds = self.threshold_values()
        summary_bar = ProgressReporter(
            len(thresholds) + 3,
            "Summarizing metrics",
            enabled=self._progress_enabled(),
            unit="metric",
        )
        try:
            for threshold in thresholds:
                segmentation = self.postprocess_policy.fixed_threshold_mask(anomaly_map, threshold)
                segmentation_mf = self.postprocess_policy.fixed_threshold_mask(anomaly_map_mf, threshold)
                scores[threshold] = self._segmentation_metrics(segmentation, labels)
                scores_mf[threshold] = self._segmentation_metrics(segmentation_mf, labels)
                summary_bar.update(postfix={"thr": threshold})

            method = processed.threshold_method
            threshold_metrics = self._threshold_method_metrics(
                processed.binary_mask_raw_postprocessed,
                processed.thresholds_raw,
                labels,
            )
            threshold_metrics_mf = self._threshold_method_metrics(
                processed.binary_mask_mf_postprocessed,
                processed.thresholds_mf,
                labels,
            )
            scores[method] = threshold_metrics["dice"]
            scores[f"{method}thr"] = threshold_metrics["threshold"]
            scores[f"{method}sen"] = threshold_metrics["sensitivity"]
            scores[f"{method}pre"] = threshold_metrics["precision"]
            scores_mf[method] = threshold_metrics_mf["dice"]
            scores_mf[f"{method}thr"] = threshold_metrics_mf["threshold"]
            scores_mf[f"{method}sen"] = threshold_metrics_mf["sensitivity"]
            scores_mf[f"{method}pre"] = threshold_metrics_mf["precision"]
            summary_bar.update(postfix=method)
            if self.compute_auprc:
                scores["AUPRC"] = auprc(
                    anomaly_map,
                    labels,
                    max_samples=self.auprc_max_samples,
                    seed=self.auprc_seed,
                    mode=self.auprc_mode,
                )
                scores_mf["AUPRC"] = auprc(
                    anomaly_map_mf,
                    labels,
                    max_samples=self.auprc_max_samples,
                    seed=self.auprc_seed,
                    mode=self.auprc_mode,
                )
                scores["AUPRC_mode"] = self.auprc_mode
                scores_mf["AUPRC_mode"] = self.auprc_mode
                if self.auprc_mode == "sampled" and self.auprc_max_samples is not None:
                    sampled = min(int(anomaly_map.numel()), self.auprc_max_samples)
                    scores["AUPRC_samples"] = sampled
                    scores_mf["AUPRC_samples"] = sampled
                    scores["AUPRC_seed"] = self.auprc_seed
                    scores_mf["AUPRC_seed"] = self.auprc_seed
                summary_bar.update(postfix="AUPRC")

            rates = self._binary_rates(anomaly_map > self.metric_threshold, labels)
            rates_mf = self._binary_rates(anomaly_map_mf > self.metric_threshold, labels)
            scores.update(rates)
            scores_mf.update(rates_mf)
            summary_bar.update(postfix="rates")
        finally:
            summary_bar.close()
        return scores, scores_mf

    @staticmethod
    def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, value in scores.items():
            if isinstance(value, dict):
                rows.append(
                    {
                        "thr": key,
                        "value": value.get("value"),
                        "dice": value.get("dice"),
                        "sensitivity": value.get("sensitivity"),
                        "precision": value.get("precision"),
                    }
                )
            else:
                rows.append({"thr": key, "value": value, "dice": None, "sensitivity": None, "precision": None})
        frame = pd.DataFrame(rows, columns=["thr", "value", "dice", "sensitivity", "precision"])
        frame.to_csv(path, index=False)

    @staticmethod
    def _nested_value(mapping: dict[str, Any], *parts: str) -> Any:
        value: Any = mapping
        for part in parts:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def _cache_fingerprints(self) -> tuple[str, str]:
        run_config = self.config.get("_run_config")
        if not isinstance(run_config, dict):
            run_config = {
                "data": {
                    key: value
                    for key, value in self.config.items()
                    if key
                    in {
                        "type",
                        "dataset_path",
                        "path",
                        "path_to_csv",
                        "image_size",
                        "channels",
                        "modalities",
                        "histogram_normalization",
                        "shift_naming",
                        "normalize_input",
                        "size_splits",
                    }
                },
                "model": self.model_config,
                "anomaly": self.anomaly_config,
            }

        anomaly_config = copy.deepcopy(run_config.get("anomaly", {}))
        if isinstance(anomaly_config, dict):
            for key in ("threshold", "median_filter", "postprocess"):
                anomaly_config.pop(key, None)
        runtime_config = run_config.get("runtime", {})
        raw_payload = {
            "implementation": "disk_streaming_v1",
            "data": run_config.get("data", {}),
            "model": run_config.get("model", self.model_config),
            "diffusion": run_config.get("diffusion", {}),
            "noise": run_config.get("noise", {}),
            "anomaly": anomaly_config,
            "runtime": {
                key: runtime_config.get(key)
                for key in ("seed", "deterministic", "cudnn_benchmark")
                if isinstance(runtime_config, dict) and key in runtime_config
            },
            "file_identities": {
                "checkpoint": file_identity(
                    self._nested_value(run_config, "model", "checkpoint")
                ),
                "split_csv": file_identity(
                    self._nested_value(run_config, "data", "path_to_csv")
                ),
                "noise_statistics": file_identity(
                    self._nested_value(
                        run_config,
                        "noise",
                        "schedule",
                        "sampler",
                        "stats_path",
                    )
                ),
            },
        }
        raw_fingerprint = stable_fingerprint(self._json_safe(raw_payload))

        description = self.postprocess_policy.describe()
        score_payload = {
            "implementation": "disk_streaming_scores_v1",
            "raw_fingerprint": raw_fingerprint,
            "postprocess_mode": description.get("postprocess_mode"),
            "normalization_scope": description.get("normalization_scope"),
            "raw_score_pipeline": description.get("raw_score_pipeline"),
            "mf_score_pipeline": description.get("mf_score_pipeline"),
            "median_filter_settings": description.get("median_filter_settings"),
            "legacy_compatibility": description.get("legacy_compatibility"),
            "legacy_profile": description.get("legacy_profile"),
            "numerical_safety": description.get("numerical_safety"),
        }
        return raw_fingerprint, stable_fingerprint(self._json_safe(score_payload))

    def _open_disk_cache(self) -> DiskEvaluationCache:
        raw_fingerprint, score_fingerprint = self._cache_fingerprints()
        return DiskEvaluationCache(
            self.cache_directory,
            raw_fingerprint=raw_fingerprint,
            score_fingerprint=score_fingerprint,
            resume=self.cache_resume,
            keep_on_success=self.cache_keep_on_success,
        )

    def _streaming_pipeline_plans(
        self,
    ) -> tuple[_DatasetPipelinePlan, _DatasetPipelinePlan, str] | None:
        if self.metric_normalization_scope == "subject":
            return None
        policy = self.postprocess_policy
        if isinstance(policy, RewritePostprocessPolicy):
            return (
                _DatasetPipelinePlan.from_steps(
                    policy.raw_score_steps,
                    branch="score_raw",
                ),
                _DatasetPipelinePlan.from_steps(
                    policy.mf_score_steps,
                    branch="score_mf",
                ),
                "score_raw",
            )
        if isinstance(policy, OriginalANDiPostprocessPolicy):
            mf_steps: list[BasePostprocessor] = []
            if policy.median_enabled:
                mf_steps.append(
                    MedianFilterPostprocessor(
                        kernel_size=policy.median_kernel_size,
                        mode=policy.median_mode,
                    )
                )
            mf_steps.append(NormalizePostprocessor(eps=policy.eps))
            return (
                _DatasetPipelinePlan.from_steps(
                    [NormalizePostprocessor(eps=policy.eps)],
                    branch="score_raw",
                ),
                _DatasetPipelinePlan.from_steps(mf_steps, branch="score_mf"),
                "raw",
            )
        raise TypeError(
            "disk_streaming does not support this postprocess policy type: "
            f"{type(policy).__name__}."
        )

    @staticmethod
    def _prepared_bounds(tensor: torch.Tensor) -> tuple[float, float]:
        finite = sanitize_scores(tensor)
        if finite.numel() == 0:
            return 0.0, 0.0
        return float(finite.amin().item()), float(finite.amax().item())

    def _collect_to_disk_cache(
        self,
        dataloader: Iterable,
        cache: DiskEvaluationCache,
    ) -> tuple[int, int]:
        try:
            total = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            total = 0
        volume_bar = ProgressReporter(
            total,
            "Caching anomaly maps",
            enabled=self._progress_enabled(),
            unit="batch",
        )
        subject_index = 0
        cache_hits = 0
        inferred = 0
        with torch.no_grad():
            try:
                for volume_index, batch in enumerate(dataloader, start=1):
                    image, label, metadata = self._split_batch(batch)
                    batch_size = int(image.shape[0])
                    metadata_items = self._metadata_items(metadata, batch_size)
                    missing_local_indices: list[int] = []
                    subject_ids: list[str] = []
                    label_available: list[bool] = []
                    for local_index, metadata_item in enumerate(metadata_items):
                        current_index = subject_index + local_index
                        subject_id = self._safe_subject_id(
                            metadata_item.get(
                                "subject_id",
                                f"subject_{current_index + 1:04d}",
                            )
                        )
                        has_label = self._has_label(label, metadata_item)
                        subject_ids.append(subject_id)
                        label_available.append(has_label)
                        cached = cache.cached_entry(
                            current_index,
                            subject_id=subject_id,
                            has_label=has_label,
                        )
                        if cached is None:
                            missing_local_indices.append(local_index)
                        else:
                            cache_hits += 1

                    if missing_local_indices:
                        selected = torch.as_tensor(
                            missing_local_indices,
                            dtype=torch.long,
                            device=image.device,
                        )
                        selected_images = image.index_select(0, selected)
                        anomaly_maps = self._volume_scores(
                            selected_images,
                            volume_index=volume_index,
                        ).detach().cpu()
                        for score_index, local_index in enumerate(missing_local_indices):
                            current_index = subject_index + local_index
                            item_label: torch.Tensor | None = None
                            if label is not None and label_available[local_index]:
                                item_label = label[local_index].detach().cpu().bool()
                            cache.store_raw_entry(
                                current_index,
                                subject_id=subject_ids[local_index],
                                raw=anomaly_maps[score_index],
                                label=item_label,
                                metadata=self._json_safe(metadata_items[local_index]),
                            )
                            inferred += 1
                    subject_index += batch_size
                    volume_bar.update(
                        postfix={"cached": cache_hits, "inferred": inferred}
                    )
            finally:
                volume_bar.close()
        cache.finish_collection(subject_index)
        return cache_hits, inferred

    def _raw_score_from_entry(
        self,
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: _DatasetPipelinePlan,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = sanitize_scores(cache.load_raw(entry)[None])
        return raw, raw_plan.process(raw, raw_bounds)

    def _prepare_streaming_scores(
        self,
        cache: DiskEvaluationCache,
    ) -> tuple[
        _DatasetPipelinePlan | None,
        _DatasetPipelinePlan | None,
        str | None,
        tuple[float, float] | None,
    ]:
        plans = self._streaming_pipeline_plans()
        if plans is None:
            progress = ProgressReporter(
                len(cache.entries),
                "Preparing MF cache",
                enabled=self._progress_enabled(),
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    if not cache.has_mf(entry):
                        raw = cache.load_raw(entry)[None]
                        processed = self.process_raw_maps(raw, normalization_scope="subject")
                        cache.store_product(entry, key="mf", tensor=processed.score_mf[0])
                    progress.update()
            finally:
                progress.close()
            return None, None, None, None

        raw_plan, mf_plan, mf_source = plans
        raw_bounds = cache.get_bounds("raw_score_bounds")
        if raw_plan.normalizer is not None and raw_bounds is None:
            minimum = float("inf")
            maximum = float("-inf")
            progress = ProgressReporter(
                len(cache.entries),
                "Scanning raw score range",
                enabled=self._progress_enabled(),
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    raw = sanitize_scores(cache.load_raw(entry)[None])
                    prepared = raw_plan.prepare(raw)
                    item_minimum, item_maximum = self._prepared_bounds(prepared)
                    minimum = min(minimum, item_minimum)
                    maximum = max(maximum, item_maximum)
                    progress.update()
            finally:
                progress.close()
            if not cache.entries:
                minimum = maximum = 0.0
            cache.set_bounds("raw_score_bounds", minimum, maximum)
            raw_bounds = (minimum, maximum)

        if mf_plan.normalizer is None:
            progress = ProgressReporter(
                len(cache.entries),
                "Preparing MF cache",
                enabled=self._progress_enabled(),
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    if not cache.has_mf(entry):
                        raw, score_raw = self._raw_score_from_entry(
                            cache,
                            entry,
                            raw_plan,
                            raw_bounds,
                        )
                        source = raw if mf_source == "raw" else sanitize_scores(score_raw)
                        score_mf = mf_plan.process(source, None)
                        cache.store_product(entry, key="mf", tensor=score_mf[0])
                    progress.update()
            finally:
                progress.close()
            return raw_plan, mf_plan, mf_source, raw_bounds

        mf_bounds = cache.get_bounds("mf_score_bounds")
        if mf_bounds is None:
            if any(cache.has_mf(entry) for entry in cache.entries):
                raise RuntimeError(
                    "Evaluation cache contains final MF scores but no mf_score_bounds. "
                    "Use a new cache.directory; the existing cache was not modified."
                )
            progress = ProgressReporter(
                len(cache.entries),
                "Building MF staging cache",
                enabled=self._progress_enabled(),
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    if not cache.has_mf_pre(entry):
                        raw, score_raw = self._raw_score_from_entry(
                            cache,
                            entry,
                            raw_plan,
                            raw_bounds,
                        )
                        source = raw if mf_source == "raw" else sanitize_scores(score_raw)
                        prepared_mf = mf_plan.prepare(source)
                        cache.store_product(entry, key="mf_pre", tensor=prepared_mf[0])
                    progress.update()
            finally:
                progress.close()
            minimum = min(
                float(entry["mf_pre"]["min"]) for entry in cache.entries
            ) if cache.entries else 0.0
            maximum = max(
                float(entry["mf_pre"]["max"]) for entry in cache.entries
            ) if cache.entries else 0.0
            cache.set_bounds("mf_score_bounds", minimum, maximum)
            mf_bounds = (minimum, maximum)

        progress = ProgressReporter(
            len(cache.entries),
            "Finalizing MF cache",
            enabled=self._progress_enabled(),
            unit="subject",
        )
        try:
            for entry in cache.entries:
                if not cache.has_mf(entry):
                    prepared_mf = cache.load_product(entry, "mf_pre")[None]
                    score_mf = mf_plan.finish(prepared_mf, mf_bounds)
                    cache.store_product(entry, key="mf", tensor=score_mf[0])
                if entry.get("mf_pre") is not None:
                    cache.remove_mf_pre(entry)
                progress.update()
        finally:
            progress.close()
        return raw_plan, mf_plan, mf_source, raw_bounds

    @staticmethod
    def _empty_stream_stats() -> dict[str, float | int]:
        return {
            "dice_sum": 0.0,
            "subjects": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    @staticmethod
    def _update_stream_stats(
        stats: dict[str, float | int],
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        prediction = prediction.bool()
        label = label.bool()
        stats["dice_sum"] = float(stats["dice_sum"]) + dice(prediction, label)
        stats["subjects"] = int(stats["subjects"]) + 1
        stats["tp"] = int(stats["tp"]) + int(torch.logical_and(prediction, label).sum().item())
        stats["tn"] = int(stats["tn"]) + int(torch.logical_and(~prediction, ~label).sum().item())
        stats["fp"] = int(stats["fp"]) + int(torch.logical_and(prediction, ~label).sum().item())
        stats["fn"] = int(stats["fn"]) + int(torch.logical_and(~prediction, label).sum().item())

    @staticmethod
    def _finalize_stream_stats(stats: dict[str, float | int]) -> dict[str, float]:
        subjects = int(stats["subjects"])
        tp = int(stats["tp"])
        tn = int(stats["tn"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])
        return {
            "dice": float(stats["dice_sum"]) / max(subjects, 1),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
        }

    def _processed_stream_entry(
        self,
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, PostprocessResult]:
        raw = cache.load_raw(entry)[None]
        if raw_plan is None:
            return raw, self.process_raw_maps(raw, normalization_scope="subject")
        score_raw = raw_plan.process(sanitize_scores(raw), raw_bounds)
        score_mf = cache.load_product(entry, "mf")[None]
        processed = self.postprocess_policy._complete(
            score_raw,
            score_mf,
            self.metric_normalization_scope,
        )
        return raw, processed

    def _exact_auprc_chunks(
        self,
        cache: DiskEvaluationCache,
        branch: str,
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ):
        for entry in cache.entries:
            label = cache.load_label(entry)
            if label is None:
                continue
            if branch == "mf":
                score = cache.load_product(entry, "mf")
            else:
                raw = cache.load_raw(entry)[None]
                if raw_plan is None:
                    score = self.process_raw_maps(raw, normalization_scope="subject").score_raw[0]
                else:
                    score = raw_plan.process(sanitize_scores(raw), raw_bounds)[0]
            yield score.detach().cpu().numpy(), label.detach().cpu().numpy()

    def _summarize_streaming(
        self,
        cache: DiskEvaluationCache,
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        thresholds = self.threshold_values()
        labels_available = cache.labels_available
        raw_sweep = {threshold: self._empty_stream_stats() for threshold in thresholds}
        mf_sweep = {threshold: self._empty_stream_stats() for threshold in thresholds}
        raw_adaptive = self._empty_stream_stats()
        mf_adaptive = self._empty_stream_stats()
        raw_fixed = self._empty_stream_stats()
        mf_fixed = self._empty_stream_stats()
        raw_threshold_values: list[float] = []
        mf_threshold_values: list[float] = []

        sample_indices: np.ndarray | None = None
        sampled_raw: np.ndarray | None = None
        sampled_mf: np.ndarray | None = None
        sampled_labels: np.ndarray | None = None
        if labels_available and self.compute_auprc and self.auprc_mode == "sampled":
            if self.auprc_max_samples is None:
                raise RuntimeError("sampled AUPRC requires auprc_max_samples.")
            sample_count = min(cache.total_voxels, self.auprc_max_samples)
            if cache.total_voxels > self.auprc_max_samples:
                rng = np.random.default_rng(self.auprc_seed)
                sample_indices = rng.integers(
                    0,
                    cache.total_voxels,
                    size=sample_count,
                    dtype=np.int64,
                )
                sample_indices.sort()
            else:
                sample_indices = np.arange(sample_count, dtype=np.int64)
            sampled_raw = np.empty(sample_count, dtype=np.float32)
            sampled_mf = np.empty(sample_count, dtype=np.float32)
            sampled_labels = np.empty(sample_count, dtype=np.bool_)

        global_offset = 0
        progress = ProgressReporter(
            len(cache.entries),
            "Streaming metrics",
            enabled=self._progress_enabled(),
            unit="subject",
        )
        self._prediction_index = 0
        try:
            for entry in cache.entries:
                raw, processed = self._processed_stream_entry(
                    cache,
                    entry,
                    raw_plan,
                    raw_bounds,
                )
                self.last_processed = processed
                if self.prediction_enabled:
                    prediction_processed = self._prediction_postprocess(raw, processed)
                    self.last_prediction_processed = prediction_processed
                    self._export_predictions(
                        raw,
                        [entry.get("metadata", {})],
                        processed=prediction_processed,
                    )

                label = cache.load_label(entry)
                if labels_available and label is not None:
                    label_batch = label[None]
                    for threshold in thresholds:
                        raw_segmentation = self.postprocess_policy.fixed_threshold_mask(
                            processed.score_raw,
                            threshold,
                        )
                        mf_segmentation = self.postprocess_policy.fixed_threshold_mask(
                            processed.score_mf,
                            threshold,
                        )
                        self._update_stream_stats(
                            raw_sweep[threshold], raw_segmentation, label_batch
                        )
                        self._update_stream_stats(
                            mf_sweep[threshold], mf_segmentation, label_batch
                        )
                    self._update_stream_stats(
                        raw_adaptive,
                        processed.binary_mask_raw_postprocessed,
                        label_batch,
                    )
                    self._update_stream_stats(
                        mf_adaptive,
                        processed.binary_mask_mf_postprocessed,
                        label_batch,
                    )
                    self._update_stream_stats(
                        raw_fixed,
                        processed.score_raw > self.metric_threshold,
                        label_batch,
                    )
                    self._update_stream_stats(
                        mf_fixed,
                        processed.score_mf > self.metric_threshold,
                        label_batch,
                    )

                raw_threshold_values.extend(
                    float(value) for value in processed.thresholds_raw.detach().cpu().tolist()
                )
                mf_threshold_values.extend(
                    float(value) for value in processed.thresholds_mf.detach().cpu().tolist()
                )

                if sample_indices is not None:
                    next_offset = global_offset + int(entry["numel"])
                    lower = int(np.searchsorted(sample_indices, global_offset, side="left"))
                    upper = int(np.searchsorted(sample_indices, next_offset, side="left"))
                    if upper > lower:
                        local_indices = sample_indices[lower:upper] - global_offset
                        raw_flat = processed.score_raw.detach().cpu().numpy().reshape(-1)
                        mf_flat = processed.score_mf.detach().cpu().numpy().reshape(-1)
                        if label is None:
                            raise RuntimeError("A sampled AUPRC subject unexpectedly has no label.")
                        label_flat = label.detach().cpu().numpy().reshape(-1)
                        sampled_raw[lower:upper] = raw_flat[local_indices]  # type: ignore[index]
                        sampled_mf[lower:upper] = mf_flat[local_indices]  # type: ignore[index]
                        sampled_labels[lower:upper] = label_flat[local_indices]  # type: ignore[index]
                    global_offset = next_offset
                progress.update()
        finally:
            progress.close()

        scores: dict[Any, Any] = {}
        scores_mf: dict[Any, Any] = {}
        if labels_available:
            for threshold in thresholds:
                raw_values = self._finalize_stream_stats(raw_sweep[threshold])
                mf_values = self._finalize_stream_stats(mf_sweep[threshold])
                scores[threshold] = {
                    key: raw_values[key] for key in ("dice", "sensitivity", "precision")
                }
                scores_mf[threshold] = {
                    key: mf_values[key] for key in ("dice", "sensitivity", "precision")
                }
            raw_method = self._finalize_stream_stats(raw_adaptive)
            mf_method = self._finalize_stream_stats(mf_adaptive)
            fixed_raw = self._finalize_stream_stats(raw_fixed)
            fixed_mf = self._finalize_stream_stats(mf_fixed)
        else:
            for threshold in thresholds:
                scores[threshold] = {
                    "dice": "N/A",
                    "sensitivity": "N/A",
                    "precision": "N/A",
                }
                scores_mf[threshold] = {
                    "dice": "N/A",
                    "sensitivity": "N/A",
                    "precision": "N/A",
                }
            raw_method = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}
            mf_method = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}
            fixed_raw = {"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"}
            fixed_mf = {"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"}

        method = self.threshold_method
        scores[method] = raw_method["dice"]
        scores[f"{method}thr"] = (
            float(torch.tensor(raw_threshold_values, dtype=torch.float32).mean().item())
            if raw_threshold_values
            else 0.0
        )
        scores[f"{method}sen"] = raw_method["sensitivity"]
        scores[f"{method}pre"] = raw_method["precision"]
        scores_mf[method] = mf_method["dice"]
        scores_mf[f"{method}thr"] = (
            float(torch.tensor(mf_threshold_values, dtype=torch.float32).mean().item())
            if mf_threshold_values
            else 0.0
        )
        scores_mf[f"{method}sen"] = mf_method["sensitivity"]
        scores_mf[f"{method}pre"] = mf_method["precision"]

        if self.compute_auprc:
            if not labels_available:
                scores["AUPRC"] = "N/A"
                scores_mf["AUPRC"] = "N/A"
            elif self.auprc_mode == "sampled":
                if sampled_raw is None or sampled_mf is None or sampled_labels is None:
                    raise RuntimeError("Sampled AUPRC buffers were not initialized.")
                scores["AUPRC"] = auprc(sampled_raw, sampled_labels, mode="exact")
                scores_mf["AUPRC"] = auprc(sampled_mf, sampled_labels, mode="exact")
            else:
                print("Computing exact raw AUPRC with external sorting...")
                scores["AUPRC"] = external_auprc(
                    self._exact_auprc_chunks(
                        cache,
                        "raw",
                        raw_plan,
                        raw_bounds,
                    ),
                    cache.sort_directory / "raw",
                    chunk_bytes=self.external_sort_chunk_bytes,
                )
                print("Computing exact MF AUPRC with external sorting...")
                scores_mf["AUPRC"] = external_auprc(
                    self._exact_auprc_chunks(
                        cache,
                        "mf",
                        raw_plan,
                        raw_bounds,
                    ),
                    cache.sort_directory / "mf",
                    chunk_bytes=self.external_sort_chunk_bytes,
                )
            scores["AUPRC_mode"] = self.auprc_mode
            scores_mf["AUPRC_mode"] = self.auprc_mode
            if self.auprc_mode == "sampled" and sampled_raw is not None:
                scores["AUPRC_samples"] = int(sampled_raw.shape[0])
                scores_mf["AUPRC_samples"] = int(sampled_raw.shape[0])
                scores["AUPRC_seed"] = self.auprc_seed
                scores_mf["AUPRC_seed"] = self.auprc_seed

        scores.update(
            {key: fixed_raw[key] for key in ("sensitivity", "specificity", "precision")}
        )
        scores_mf.update(
            {key: fixed_mf[key] for key in ("sensitivity", "specificity", "precision")}
        )
        return scores, scores_mf

    def _evaluate_disk_streaming(self, dataloader: Iterable) -> dict[str, Any]:
        cache = self._open_disk_cache()
        print(
            "Evaluation memory mode: disk_streaming "
            f"(cache={cache.root}, resume={cache.resume}, "
            f"AUPRC={self.auprc_mode if self.compute_auprc else 'disabled'})"
        )
        cache_hits, inferred = self._collect_to_disk_cache(dataloader, cache)
        raw_plan, _, _, raw_bounds = self._prepare_streaming_scores(cache)
        scores, scores_mf = self._summarize_streaming(cache, raw_plan, raw_bounds)
        self.write_original_style_csv(scores, self.output_csv)
        self.write_original_style_csv(scores_mf, self.output_mf_csv)
        method = self.threshold_method
        result = {
            "output": str(self.output_csv),
            "output_mf": str(self.output_mf_csv),
            "subjects": len(cache.entries),
            "labels_available": cache.labels_available,
            "memory_mode": self.memory_mode,
            "cache_directory": str(cache.root),
            "cache_hits": cache_hits,
            "subjects_inferred": inferred,
            "AUPRC": scores.get("AUPRC"),
            "AUPRC_mf": scores_mf.get("AUPRC"),
            "AUPRC_mode": self.auprc_mode if self.compute_auprc else None,
            "AUPRC_samples": scores.get("AUPRC_samples"),
            "AUPRC_seed": (
                self.auprc_seed
                if self.compute_auprc and self.auprc_mode == "sampled"
                else None
            ),
            "threshold_method": method,
            "ThresholdDice": scores.get(method),
            "ThresholdDice_mf": scores_mf.get(method),
            "Threshold": scores.get(f"{method}thr"),
            "Threshold_mf": scores_mf.get(f"{method}thr"),
            "postprocess_mode": self.postprocess_policy.mode,
            "normalization_scope": self.metric_normalization_scope,
            "prediction_normalization_scope": self.prediction_normalization_scope,
            "postprocessing": self.postprocess_policy.describe(),
        }
        method_label = method.title()
        result.update(
            {
                f"Dice{method_label}": scores.get(method),
                f"Dice{method_label}_mf": scores_mf.get(method),
                f"{method_label}Thr": scores.get(f"{method}thr"),
                f"{method_label}Thr_mf": scores_mf.get(f"{method}thr"),
            }
        )
        cache.cleanup_after_success()
        return result

    def evaluate(self, dataloader: Iterable) -> dict[str, Any]:
        if self.memory_mode == "disk_streaming":
            return self._evaluate_disk_streaming(dataloader)
        raw_maps, labels, metadata = self.collect(dataloader)
        if not self.is_main_process:
            return {}
        processed = self.process_raw_maps(raw_maps, self.metric_normalization_scope)
        self.last_processed = processed
        prediction_processed: PostprocessResult | None = None
        if self.prediction_enabled:
            prediction_processed = self._prediction_postprocess(raw_maps, processed)
            self.last_prediction_processed = prediction_processed
            self._export_predictions(raw_maps, metadata, processed=prediction_processed)
        scores, scores_mf = self.summarize_processed(processed, labels)
        self.write_original_style_csv(scores, self.output_csv)
        self.write_original_style_csv(scores_mf, self.output_mf_csv)
        method = processed.threshold_method
        result = {
            "output": str(self.output_csv),
            "output_mf": str(self.output_mf_csv),
            "subjects": len(metadata),
            "labels_available": labels is not None,
            "AUPRC": scores.get("AUPRC"),
            "AUPRC_mf": scores_mf.get("AUPRC"),
            "AUPRC_mode": self.auprc_mode if self.compute_auprc else None,
            "AUPRC_samples": scores.get("AUPRC_samples"),
            "AUPRC_seed": (
                self.auprc_seed
                if self.compute_auprc and self.auprc_mode == "sampled"
                else None
            ),
            "memory_mode": self.memory_mode,
            "threshold_method": method,
            "ThresholdDice": scores.get(method),
            "ThresholdDice_mf": scores_mf.get(method),
            "Threshold": scores.get(f"{method}thr"),
            "Threshold_mf": scores_mf.get(f"{method}thr"),
            "postprocess_mode": self.postprocess_policy.mode,
            "normalization_scope": processed.normalization_scope,
            "prediction_normalization_scope": self.prediction_normalization_scope,
            "postprocessing": self.postprocess_policy.describe(),
        }
        method_label = method.title()
        result.update(
            {
                f"Dice{method_label}": scores.get(method),
                f"Dice{method_label}_mf": scores_mf.get(method),
                f"{method_label}Thr": scores.get(f"{method}thr"),
                f"{method_label}Thr_mf": scores_mf.get(f"{method}thr"),
            }
        )
        return result
