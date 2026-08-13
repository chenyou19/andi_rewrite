"""Disk-streaming evaluation collection, score staging, and metric reduction.

The helper receives callbacks for facade-owned seams.  In particular, raw score
collection continues to call ``VolumeEvaluator._volume_scores`` and prediction
production continues through the facade postprocess/export wrappers.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from andi_rewrite.anomaly.postprocess import (
    BasePostprocessor,
    NormalizePostprocessor,
    PostprocessPolicy,
    PostprocessResult,
    apply_postprocess_pipeline,
    sanitize_scores,
)
from ..evaluation_cache import DiskEvaluationCache
from andi_rewrite.metrics.classification import auprc, dice, external_auprc
from andi_rewrite.utils.progress import ProgressReporter
from .metrics import (
    empty_stream_stats,
    finalize_stream_stats,
    fixed_rate_row,
    sweep_metric_row,
    update_stream_stats,
)


@dataclass(frozen=True)
class DatasetPipelinePlan:
    """Split one policy score pipeline around its dataset normalization step."""

    prefix: list[BasePostprocessor]
    normalizer: NormalizePostprocessor | None
    suffix: list[BasePostprocessor]

    @classmethod
    def from_steps(
        cls,
        steps: list[BasePostprocessor],
        *,
        branch: str,
    ) -> "DatasetPipelinePlan":
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


@dataclass(frozen=True)
class StreamingCallbacks:
    """Facade-owned compatibility seams used by streaming orchestration."""

    streaming_pipeline_plans: Callable[
        [], tuple[DatasetPipelinePlan, DatasetPipelinePlan, str] | None
    ]
    prepared_bounds: Callable[[torch.Tensor], tuple[float, float]]
    raw_score_from_entry: Callable[
        [DiskEvaluationCache, dict[str, Any], DatasetPipelinePlan, tuple[float, float] | None],
        tuple[torch.Tensor, torch.Tensor],
    ]
    empty_stream_stats: Callable[[], dict[str, float | int]]
    update_stream_stats: Callable[
        [dict[str, float | int], torch.Tensor, torch.Tensor], None
    ]
    finalize_stream_stats: Callable[
        [dict[str, float | int]], dict[str, float]
    ]
    processed_stream_entry: Callable[
        [
            DiskEvaluationCache,
            dict[str, Any],
            DatasetPipelinePlan | None,
            tuple[float, float] | None,
        ],
        tuple[torch.Tensor, PostprocessResult],
    ]
    exact_auprc_chunks: Callable[
        [
            DiskEvaluationCache,
            str,
            DatasetPipelinePlan | None,
            tuple[float, float] | None,
        ],
        Iterable[tuple[np.ndarray, np.ndarray]],
    ]


class StreamingEvaluator:
    """Cohesive disk-cache evaluator without a dependency on the facade class."""

    def __init__(
        self,
        *,
        policy: PostprocessPolicy,
        metric_normalization_scope: str,
        threshold_method: str,
        thresholds: list[float],
        metric_threshold: float,
        compute_auprc: bool,
        auprc_mode: str,
        auprc_max_samples: int | None,
        auprc_seed: int,
        external_sort_chunk_bytes: int,
        prediction_enabled: bool,
        progress_enabled: bool,
        volume_scores: Callable[..., torch.Tensor],
        split_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor | None, Any]],
        metadata_items: Callable[[Any, int], list[dict[str, Any]]],
        safe_subject_id: Callable[[Any], str],
        has_label: Callable[[torch.Tensor | None, dict[str, Any]], bool],
        json_safe: Callable[[Any], Any],
        process_raw_maps: Callable[..., PostprocessResult],
        prediction_postprocess: Callable[..., PostprocessResult],
        export_predictions: Callable[..., None],
        on_processed: Callable[[PostprocessResult], None],
        on_prediction_processed: Callable[[PostprocessResult], None],
        callbacks: StreamingCallbacks,
    ) -> None:
        self.policy = policy
        self.metric_normalization_scope = metric_normalization_scope
        self.threshold_method = threshold_method
        self.thresholds = thresholds
        self.metric_threshold = metric_threshold
        self.compute_auprc = compute_auprc
        self.auprc_mode = auprc_mode
        self.auprc_max_samples = auprc_max_samples
        self.auprc_seed = auprc_seed
        self.external_sort_chunk_bytes = external_sort_chunk_bytes
        self.prediction_enabled = prediction_enabled
        self.progress_enabled = progress_enabled
        self.volume_scores = volume_scores
        self.split_batch = split_batch
        self.metadata_items = metadata_items
        self.safe_subject_id = safe_subject_id
        self.has_label = has_label
        self.json_safe = json_safe
        self.process_raw_maps = process_raw_maps
        self.prediction_postprocess = prediction_postprocess
        self.export_predictions = export_predictions
        self.on_processed = on_processed
        self.on_prediction_processed = on_prediction_processed
        self.callbacks = callbacks

    def streaming_pipeline_plans(
        self,
    ) -> tuple[DatasetPipelinePlan, DatasetPipelinePlan, str] | None:
        if self.metric_normalization_scope == "subject":
            return None
        spec = self.policy.score_pipeline_spec()
        if spec is None:
            raise TypeError(
                "dataset-normalized disk_streaming requires the postprocess policy "
                "to provide score_pipeline_spec(); "
                f"{type(self.policy).__name__} did not opt in."
            )
        return (
            DatasetPipelinePlan.from_steps(list(spec.raw_steps), branch="score_raw"),
            DatasetPipelinePlan.from_steps(list(spec.mf_steps), branch="score_mf"),
            spec.mf_source,
        )

    @staticmethod
    def prepared_bounds(tensor: torch.Tensor) -> tuple[float, float]:
        finite = sanitize_scores(tensor)
        if finite.numel() == 0:
            return 0.0, 0.0
        return float(finite.amin().item()), float(finite.amax().item())

    def collect_to_disk_cache(
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
            enabled=self.progress_enabled,
            unit="batch",
        )
        subject_index = 0
        cache_hits = 0
        inferred = 0
        with torch.no_grad():
            try:
                for volume_index, batch in enumerate(dataloader, start=1):
                    image, label, metadata = self.split_batch(batch)
                    batch_size = int(image.shape[0])
                    items = self.metadata_items(metadata, batch_size)
                    missing_local_indices: list[int] = []
                    subject_ids: list[str] = []
                    label_available: list[bool] = []
                    for local_index, metadata_item in enumerate(items):
                        current_index = subject_index + local_index
                        subject_id = self.safe_subject_id(
                            metadata_item.get(
                                "subject_id",
                                f"subject_{current_index + 1:04d}",
                            )
                        )
                        has_label = self.has_label(label, metadata_item)
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
                        anomaly_maps = self.volume_scores(
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
                                metadata=self.json_safe(items[local_index]),
                            )
                            inferred += 1
                    subject_index += batch_size
                    volume_bar.update(postfix={"cached": cache_hits, "inferred": inferred})
            finally:
                volume_bar.close()
        cache.finish_collection(subject_index)
        return cache_hits, inferred

    @staticmethod
    def raw_score_from_entry(
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: DatasetPipelinePlan,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = sanitize_scores(cache.load_raw(entry)[None])
        return raw, raw_plan.process(raw, raw_bounds)

    def prepare_scores(
        self,
        cache: DiskEvaluationCache,
    ) -> tuple[
        DatasetPipelinePlan | None,
        DatasetPipelinePlan | None,
        str | None,
        tuple[float, float] | None,
    ]:
        plans = self.callbacks.streaming_pipeline_plans()
        if plans is None:
            progress = ProgressReporter(
                len(cache.entries),
                "Preparing MF cache",
                enabled=self.progress_enabled,
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
                enabled=self.progress_enabled,
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    raw = sanitize_scores(cache.load_raw(entry)[None])
                    prepared = raw_plan.prepare(raw)
                    item_minimum, item_maximum = self.callbacks.prepared_bounds(prepared)
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
                enabled=self.progress_enabled,
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    if not cache.has_mf(entry):
                        raw, score_raw = self.callbacks.raw_score_from_entry(
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
                enabled=self.progress_enabled,
                unit="subject",
            )
            try:
                for entry in cache.entries:
                    if not cache.has_mf_pre(entry):
                        raw, score_raw = self.callbacks.raw_score_from_entry(
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
            minimum = (
                min(float(entry["mf_pre"]["min"]) for entry in cache.entries)
                if cache.entries
                else 0.0
            )
            maximum = (
                max(float(entry["mf_pre"]["max"]) for entry in cache.entries)
                if cache.entries
                else 0.0
            )
            cache.set_bounds("mf_score_bounds", minimum, maximum)
            mf_bounds = (minimum, maximum)

        progress = ProgressReporter(
            len(cache.entries),
            "Finalizing MF cache",
            enabled=self.progress_enabled,
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

    # Keep the historical class-level helper seams while the formulas have one
    # owner shared with the in-memory metric module.
    empty_stream_stats = staticmethod(empty_stream_stats)
    update_stream_stats = staticmethod(update_stream_stats)
    finalize_stream_stats = staticmethod(finalize_stream_stats)

    def processed_stream_entry(
        self,
        cache: DiskEvaluationCache,
        entry: dict[str, Any],
        raw_plan: DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, PostprocessResult]:
        raw = cache.load_raw(entry)[None]
        if raw_plan is None:
            return raw, self.process_raw_maps(raw, normalization_scope="subject")
        score_raw = raw_plan.process(sanitize_scores(raw), raw_bounds)
        score_mf = cache.load_product(entry, "mf")[None]
        processed = self.policy.complete_scores(
            score_raw,
            score_mf,
            self.metric_normalization_scope,
        )
        return raw, processed

    def exact_auprc_chunks(
        self,
        cache: DiskEvaluationCache,
        branch: str,
        raw_plan: DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
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

    def summarize(
        self,
        cache: DiskEvaluationCache,
        raw_plan: DatasetPipelinePlan | None,
        raw_bounds: tuple[float, float] | None,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        labels_available = cache.labels_available
        raw_sweep = {
            threshold: self.callbacks.empty_stream_stats() for threshold in self.thresholds
        }
        mf_sweep = {
            threshold: self.callbacks.empty_stream_stats() for threshold in self.thresholds
        }
        raw_adaptive = self.callbacks.empty_stream_stats()
        mf_adaptive = self.callbacks.empty_stream_stats()
        raw_fixed = self.callbacks.empty_stream_stats()
        mf_fixed = self.callbacks.empty_stream_stats()
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
            enabled=self.progress_enabled,
            unit="subject",
        )
        try:
            for entry in cache.entries:
                raw, processed = self.callbacks.processed_stream_entry(
                    cache,
                    entry,
                    raw_plan,
                    raw_bounds,
                )
                self.on_processed(processed)
                if self.prediction_enabled:
                    prediction_processed = self.prediction_postprocess(raw, processed)
                    self.on_prediction_processed(prediction_processed)
                    self.export_predictions(
                        raw,
                        [entry.get("metadata", {})],
                        processed=prediction_processed,
                    )

                label = cache.load_label(entry)
                if labels_available and label is not None:
                    label_batch = label[None]
                    for threshold in self.thresholds:
                        raw_segmentation = self.policy.fixed_threshold_mask(
                            processed.score_raw,
                            threshold,
                        )
                        mf_segmentation = self.policy.fixed_threshold_mask(
                            processed.score_mf,
                            threshold,
                        )
                        self.callbacks.update_stream_stats(
                            raw_sweep[threshold], raw_segmentation, label_batch
                        )
                        self.callbacks.update_stream_stats(
                            mf_sweep[threshold], mf_segmentation, label_batch
                        )
                    self.callbacks.update_stream_stats(
                        raw_adaptive,
                        processed.binary_mask_raw_postprocessed,
                        label_batch,
                    )
                    self.callbacks.update_stream_stats(
                        mf_adaptive,
                        processed.binary_mask_mf_postprocessed,
                        label_batch,
                    )
                    self.callbacks.update_stream_stats(
                        raw_fixed,
                        processed.score_raw > self.metric_threshold,
                        label_batch,
                    )
                    self.callbacks.update_stream_stats(
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
            for threshold in self.thresholds:
                raw_values = self.callbacks.finalize_stream_stats(raw_sweep[threshold])
                mf_values = self.callbacks.finalize_stream_stats(mf_sweep[threshold])
                scores[threshold] = sweep_metric_row(raw_values)
                scores_mf[threshold] = sweep_metric_row(mf_values)
            raw_method = self.callbacks.finalize_stream_stats(raw_adaptive)
            mf_method = self.callbacks.finalize_stream_stats(mf_adaptive)
            fixed_raw = self.callbacks.finalize_stream_stats(raw_fixed)
            fixed_mf = self.callbacks.finalize_stream_stats(mf_fixed)
        else:
            for threshold in self.thresholds:
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
                    self.callbacks.exact_auprc_chunks(
                        cache, "raw", raw_plan, raw_bounds
                    ),
                    cache.sort_directory / "raw",
                    chunk_bytes=self.external_sort_chunk_bytes,
                )
                print("Computing exact MF AUPRC with external sorting...")
                scores_mf["AUPRC"] = external_auprc(
                    self.callbacks.exact_auprc_chunks(
                        cache, "mf", raw_plan, raw_bounds
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

        scores.update(fixed_rate_row(fixed_raw))
        scores_mf.update(fixed_rate_row(fixed_mf))
        return scores, scores_mf
