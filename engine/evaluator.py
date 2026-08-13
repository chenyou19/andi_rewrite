"""Stateful compatibility facade for volume-level ANDi evaluation.

Algorithms and artifact I/O live in ``engine.evaluation`` leaf modules.  This
class intentionally retains configuration state and the established overridable
hooks used by callers and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.anomaly.postprocess import (
    PostprocessPolicy,
    PostprocessResult,
    build_postprocess_policy,
)
from andi_rewrite.engine.evaluation_cache import DiskEvaluationCache
from andi_rewrite.metrics.classification import resolve_auprc_mode

from .evaluation import (
    collection,
    fingerprints,
    inference,
    inputs,
    metrics,
    output,
    prediction_export,
    streaming,
)


# Retain the old private module name for code that imported it incidentally.
_DatasetPipelinePlan = streaming.DatasetPipelinePlan


class VolumeEvaluator:
    """State/configuration facade for in-memory and disk-streaming evaluation."""

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

    # Inference and input compatibility wrappers.
    def _slice_scores(self, images: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        return inference.slice_scores(
            self.detector,
            images,
            size_splits=self.size_splits,
            progress_enabled=self._progress_enabled(),
            volume_index=volume_index,
        )

    def _volume_scores(self, image: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        return inference.volume_scores(
            self.detector,
            image,
            normalize_input=self.normalize_input,
            size_splits=self.size_splits,
            progress_enabled=self._progress_enabled(),
            volume_index=volume_index,
            slice_score_callback=self._slice_scores,
        )

    def _split_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        return inputs.split_batch(batch)

    def _metadata_items(self, metadata: Any, batch_size: int) -> list[dict[str, Any]]:
        return inputs.metadata_items(metadata, batch_size)

    def _metadata_item(self, value: Any, index: int, batch_size: int) -> Any:
        return inputs.metadata_item(value, index, batch_size)

    @staticmethod
    def _truthy(value: Any, default: bool = True) -> bool:
        return inputs.truthy(value, default)

    @staticmethod
    def _safe_subject_id(value: Any) -> str:
        return inputs.safe_subject_id(value)

    @staticmethod
    def _shape_from_text(value: Any) -> tuple[int, ...] | None:
        return inputs.shape_from_text(value)

    def _has_label(self, label: torch.Tensor | None, metadata: dict[str, Any]) -> bool:
        return inputs.has_label(label, metadata)

    # Shared policy orchestration remains facade-owned so callers can override it.
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
        result = {
            "score_raw": processed.score_raw,
            "score_mf": processed.score_mf,
            "binary_mask": processed.binary_mask_mf_postprocessed,
            "thresholds": processed.thresholds_mf,
            "binary_mask_raw": processed.binary_mask_raw_postprocessed,
            "thresholds_raw": processed.thresholds_raw,
            "threshold_mask": threshold_mask,
        }
        if processed.threshold_method == "yen":
            result.update(
                {
                    "yen_mask": processed.binary_mask_mf_postprocessed,
                    "yen_thresholds": processed.thresholds_mf,
                    "yen_mask_raw": processed.binary_mask_raw_postprocessed,
                    "yen_thresholds_raw": processed.thresholds_raw,
                }
            )
        return result

    # Prediction-export compatibility wrappers.
    @staticmethod
    def _resize_volume_to_shape(
        volume: torch.Tensor,
        shape: tuple[int, int, int],
        continuous: bool,
    ) -> torch.Tensor:
        return prediction_export.resize_volume_to_shape(volume, shape, continuous)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return fingerprints.json_safe(value)

    def _load_reference_image(self, metadata: dict[str, Any]) -> Any | None:
        return prediction_export.load_reference_image(metadata)

    def _save_nifti(self, array: Any, reference_image: Any, path: Path, dtype: Any) -> None:
        prediction_export.save_nifti(array, reference_image, path, dtype)

    def _prediction_postprocess(
        self,
        raw_maps: torch.Tensor,
        metric_processed: PostprocessResult | None = None,
    ) -> PostprocessResult:
        return prediction_export.prediction_postprocess(
            raw_maps,
            metric_processed,
            prediction_normalization_scope=self.prediction_normalization_scope,
            process_raw_maps=self.process_raw_maps,
        )

    def _export_predictions(
        self,
        raw_maps: torch.Tensor,
        metadata_items: list[dict[str, Any]],
        processed: PostprocessResult | None = None,
    ) -> None:
        if not self.prediction_enabled or not self.is_main_process:
            return
        processed = processed or self._prediction_postprocess(raw_maps.detach().cpu())
        self.last_prediction_processed = processed
        self._prediction_index = prediction_export.export_predictions(
            raw_maps,
            metadata_items,
            processed=processed,
            prediction_output=self.prediction_output,
            metric_threshold=self.metric_threshold,
            model_config=self.model_config,
            detector=self.detector,
            postprocess_policy=self.postprocess_policy,
            prediction_normalization_scope=self.prediction_normalization_scope,
            prediction_index=self._prediction_index,
            safe_subject_id_callback=self._safe_subject_id,
            load_reference_image_callback=self._load_reference_image,
            save_nifti_callback=self._save_nifti,
            resize_volume_callback=self._resize_volume_to_shape,
            json_safe_callback=self._json_safe,
        )

    def collect(self, dataloader: Iterable) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
        return collection.collect_volumes(
            dataloader,
            volume_scores=self._volume_scores,
            split_batch=self._split_batch,
            metadata_items=self._metadata_items,
            has_label=self._has_label,
            accelerator=self.accelerator,
            progress_enabled=self._progress_enabled(),
        )

    # In-memory metric compatibility wrappers.
    def threshold_values(self) -> list[float]:
        return metrics.threshold_values(self.threshold_start, self.threshold_end, self.threshold_step)

    def _dice_mean(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
        return metrics.dice_mean(prediction, label)

    def _binary_rates(self, prediction: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        return metrics.binary_rates(prediction, label)

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
        values = self._segmentation_metrics(segmentation, label)
        values["threshold"] = (
            float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        )
        return values

    def _summarize_without_labels(
        self,
        processed: PostprocessResult,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        return metrics.summarize_without_labels(
            processed,
            thresholds=self.threshold_values(),
            compute_auprc=self.compute_auprc,
            auprc_mode=self.auprc_mode,
        )

    def summarize(
        self,
        raw_maps: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        processed = self.process_raw_maps(raw_maps, self.metric_normalization_scope)
        self.last_processed = processed
        return self.summarize_processed(processed, labels)

    def summarize_processed(
        self,
        processed: PostprocessResult,
        labels: torch.Tensor | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        return metrics.summarize_processed(
            processed,
            labels,
            thresholds=self.threshold_values(),
            policy=self.postprocess_policy,
            metric_threshold=self.metric_threshold,
            compute_auprc=self.compute_auprc,
            auprc_mode=self.auprc_mode,
            auprc_max_samples=self.auprc_max_samples,
            auprc_seed=self.auprc_seed,
            progress_enabled=self._progress_enabled(),
            summarize_without_labels_callback=self._summarize_without_labels,
            segmentation_metrics_callback=self._segmentation_metrics,
            threshold_method_metrics_callback=self._threshold_method_metrics,
            binary_rates_callback=self._binary_rates,
        )

    @staticmethod
    def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
        output.write_original_style_csv(scores, path)

    # Cache/fingerprint compatibility wrappers.
    @staticmethod
    def _nested_value(mapping: dict[str, Any], *parts: str) -> Any:
        return fingerprints.nested_value(mapping, *parts)

    def _cache_fingerprints(self) -> tuple[str, str]:
        return fingerprints.cache_fingerprints(
            self.config,
            model_config=self.model_config,
            anomaly_config=self.anomaly_config,
            postprocess_policy=self.postprocess_policy,
        )

    def _open_disk_cache(self) -> DiskEvaluationCache:
        raw_fingerprint, score_fingerprint = self._cache_fingerprints()
        return DiskEvaluationCache(
            self.cache_directory,
            raw_fingerprint=raw_fingerprint,
            score_fingerprint=score_fingerprint,
            resume=self.cache_resume,
            keep_on_success=self.cache_keep_on_success,
        )

    def _streaming_helper(self) -> streaming.StreamingEvaluator:
        return streaming.StreamingEvaluator(
            policy=self.postprocess_policy,
            metric_normalization_scope=self.metric_normalization_scope,
            threshold_method=self.threshold_method,
            thresholds=self.threshold_values(),
            metric_threshold=self.metric_threshold,
            compute_auprc=self.compute_auprc,
            auprc_mode=self.auprc_mode,
            auprc_max_samples=self.auprc_max_samples,
            auprc_seed=self.auprc_seed,
            external_sort_chunk_bytes=self.external_sort_chunk_bytes,
            prediction_enabled=self.prediction_enabled,
            progress_enabled=self._progress_enabled(),
            volume_scores=self._volume_scores,
            split_batch=self._split_batch,
            metadata_items=self._metadata_items,
            safe_subject_id=self._safe_subject_id,
            has_label=self._has_label,
            json_safe=self._json_safe,
            process_raw_maps=self.process_raw_maps,
            prediction_postprocess=self._prediction_postprocess,
            export_predictions=self._export_predictions,
            on_processed=lambda processed: setattr(self, "last_processed", processed),
            on_prediction_processed=lambda processed: setattr(
                self, "last_prediction_processed", processed
            ),
            callbacks=streaming.StreamingCallbacks(
                streaming_pipeline_plans=self._streaming_pipeline_plans,
                prepared_bounds=self._prepared_bounds,
                raw_score_from_entry=self._raw_score_from_entry,
                empty_stream_stats=self._empty_stream_stats,
                update_stream_stats=self._update_stream_stats,
                finalize_stream_stats=self._finalize_stream_stats,
                processed_stream_entry=self._processed_stream_entry,
                exact_auprc_chunks=self._exact_auprc_chunks,
            ),
        )

    def _streaming_pipeline_plans(
        self,
    ) -> tuple[_DatasetPipelinePlan, _DatasetPipelinePlan, str] | None:
        return self._streaming_helper().streaming_pipeline_plans()

    @staticmethod
    def _prepared_bounds(tensor: torch.Tensor) -> tuple[float, float]:
        return streaming.StreamingEvaluator.prepared_bounds(tensor)

    def _collect_to_disk_cache(
        self,
        dataloader: Iterable,
        cache: DiskEvaluationCache,
    ) -> tuple[int, int]:
        return self._streaming_helper().collect_to_disk_cache(dataloader, cache)

    def _raw_score_from_entry(
        self,
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: _DatasetPipelinePlan,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return streaming.StreamingEvaluator.raw_score_from_entry(
            cache,
            entry,
            raw_plan,
            raw_bounds,
        )

    def _prepare_streaming_scores(
        self,
        cache: DiskEvaluationCache,
    ) -> tuple[
        _DatasetPipelinePlan | None,
        _DatasetPipelinePlan | None,
        str | None,
        tuple[float, float] | None,
    ]:
        return self._streaming_helper().prepare_scores(cache)

    @staticmethod
    def _empty_stream_stats() -> dict[str, float | int]:
        return streaming.StreamingEvaluator.empty_stream_stats()

    @staticmethod
    def _update_stream_stats(
        stats: dict[str, float | int],
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        streaming.StreamingEvaluator.update_stream_stats(stats, prediction, label)

    @staticmethod
    def _finalize_stream_stats(stats: dict[str, float | int]) -> dict[str, float]:
        return streaming.StreamingEvaluator.finalize_stream_stats(stats)

    def _processed_stream_entry(
        self,
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, PostprocessResult]:
        return self._streaming_helper().processed_stream_entry(
            cache,
            entry,
            raw_plan,
            raw_bounds,
        )

    def _exact_auprc_chunks(
        self,
        cache: DiskEvaluationCache,
        branch: str,
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ):
        return self._streaming_helper().exact_auprc_chunks(
            cache,
            branch,
            raw_plan,
            raw_bounds,
        )

    def _summarize_streaming(
        self,
        cache: DiskEvaluationCache,
        raw_plan: _DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        self._prediction_index = 0
        return self._streaming_helper().summarize(cache, raw_plan, raw_bounds)

    def _result(
        self,
        *,
        subjects: int,
        labels_available: bool,
        scores: dict[Any, Any],
        scores_mf: dict[Any, Any],
        normalization_scope: str,
        threshold_method: str | None = None,
        cache: DiskEvaluationCache | None = None,
        cache_hits: int | None = None,
        subjects_inferred: int | None = None,
    ) -> dict[str, Any]:
        resolved_threshold_method = threshold_method or self.threshold_method
        return output.build_evaluation_result(
            output_csv=self.output_csv,
            output_mf_csv=self.output_mf_csv,
            subjects=subjects,
            labels_available=labels_available,
            scores=scores,
            scores_mf=scores_mf,
            compute_auprc=self.compute_auprc,
            auprc_mode=self.auprc_mode,
            auprc_seed=self.auprc_seed,
            memory_mode=self.memory_mode,
            threshold_method=resolved_threshold_method,
            postprocess_mode=self.postprocess_policy.mode,
            normalization_scope=normalization_scope,
            prediction_normalization_scope=self.prediction_normalization_scope,
            postprocessing=self.postprocess_policy.describe(),
            cache_directory=cache.root if cache is not None else None,
            cache_hits=cache_hits,
            subjects_inferred=subjects_inferred,
        )

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
        result = self._result(
            subjects=len(cache.entries),
            labels_available=cache.labels_available,
            scores=scores,
            scores_mf=scores_mf,
            normalization_scope=self.metric_normalization_scope,
            cache=cache,
            cache_hits=cache_hits,
            subjects_inferred=inferred,
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
        if self.prediction_enabled:
            prediction_processed = self._prediction_postprocess(raw_maps, processed)
            self.last_prediction_processed = prediction_processed
            self._export_predictions(raw_maps, metadata, processed=prediction_processed)
        scores, scores_mf = self.summarize_processed(processed, labels)
        self.write_original_style_csv(scores, self.output_csv)
        self.write_original_style_csv(scores_mf, self.output_mf_csv)
        return self._result(
            subjects=len(metadata),
            labels_available=labels is not None,
            scores=scores,
            scores_mf=scores_mf,
            normalization_scope=processed.normalization_scope,
            threshold_method=processed.threshold_method,
        )
