from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from andi_rewrite.anomaly.aggregation import (
    GeometricMeanAggregator,
    MaxAggregator,
    MeanAggregator,
    SumAggregator,
    build_aggregator,
)
from andi_rewrite.anomaly.inference_profiles import resolve_inference_profile
from andi_rewrite.anomaly.postprocess import (
    PostprocessPolicy,
    build_postprocess_policy,
    supported_threshold_methods,
)
from andi_rewrite.diffusion import DDPMDiffusion
from andi_rewrite.noise import NoisePlan
from andi_rewrite.utils.progress import ProgressReporter


_STREAMING_AGGREGATORS = {GeometricMeanAggregator, MeanAggregator, SumAggregator, MaxAggregator}


class ANDiDetector:
    """ANDi 推論流程，和 DDPM transition 數學保持分離。

    這個 class 只負責 ANDi 專屬邏輯：選擇 t 範圍、計算 transition deviation、
    聚合 anomaly map、產生 segmentation。DDPM 係數與 transition 計算留在
    diffusion/，metric 與後處理則維持 config-driven。
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: DDPMDiffusion,
        noise_plan: NoisePlan,
        config: dict[str, Any],
        device: torch.device | str,
        postprocess_policy: PostprocessPolicy | None = None,
        *,
        inference_profile: str = "original",
        amp_dtype: str = "none",
    ):
        self.config = dict(config)
        self.model = model
        self.diffusion = diffusion
        self.noise_plan = noise_plan
        self.device = torch.device(device)
        self.inference_profile = resolve_inference_profile(inference_profile)
        self.t_lower = int(config.get("t_lower", config.get("start", 75)))
        self.t_upper = int(config.get("t_upper", config.get("stop", 200)))
        self.num_timesteps = config.get("num_timesteps")
        streaming = config.get("streaming", False)
        if not isinstance(streaming, bool):
            raise ValueError("anomaly.streaming must be true or false.")
        self.streaming = streaming
        if amp_dtype not in {"none", "float16", "bfloat16"}:
            raise ValueError("runtime.amp_dtype must be 'none', 'float16', or 'bfloat16'.")
        if amp_dtype != "none" and self.device.type != "cuda":
            raise ValueError("Mixed precision inference requires a CUDA device.")
        self.amp_dtype = None if amp_dtype == "none" else getattr(torch, amp_dtype)
        self.attention_backend = getattr(model, "attention_backend", None)
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.median_cuda_device = (
            self.device if self.inference_profile != "original" and self.device.type == "cuda" else None
        )
        self.time_aggregator = build_aggregator(config.get("aggregation", "geometric"), default="geometric")
        self.modality_aggregator = build_aggregator(config.get("modality_pool", "max"), default="max")
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", True))
        self.median_kernel_size = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        self.median_mode = str(median_config.get("mode", "3d"))
        self.threshold = str(config.get("threshold", "yen")).lower()
        self.eps = float(config.get("eps", 1.0e-8))
        resolved_policy = postprocess_policy or build_postprocess_policy(
            config,
            anomaly_config=config,
            legacy_profile="detector",
            median_cuda_device=self.median_cuda_device,
        )
        self.set_postprocess_policy(resolved_policy)

    def set_postprocess_policy(self, policy: PostprocessPolicy) -> None:
        """Share the exact policy instance used by an evaluator."""

        self.postprocess_policy = policy
        if self.threshold in supported_threshold_methods():
            # Adaptive threshold selection belongs to the shared policy.
            self.threshold = policy.threshold_method
        elif policy.mode == "original_andi":
            warnings.warn(
                "original_andi uses per-subject adaptive thresholding; ignoring "
                f"anomaly.threshold={self.threshold!r} in favor of "
                f"threshold_method={policy.threshold_method!r}.",
                UserWarning,
                stacklevel=2,
            )
            self.threshold = policy.threshold_method

    def selected_timesteps(self) -> list[int]:
        """Return descending samples at the centers of equal-width bins."""

        if not 1 <= self.t_lower < self.t_upper <= self.diffusion.steps:
            raise ValueError("Require 1 <= t_lower < t_upper <= diffusion.steps.")
        span = self.t_upper - self.t_lower
        count = span if self.num_timesteps is None else self.num_timesteps
        if type(count) is not int or not 1 <= count <= span:
            raise ValueError("anomaly.num_timesteps must be an integer within the timestep interval.")
        return [
            self.t_lower + ((2 * index + 1) * span) // (2 * count)
            for index in reversed(range(count))
        ]

    def _iter_deviations(
        self,
        images: torch.Tensor,
        progress: bool,
        progress_description: str,
        progress_leave: bool,
    ) -> Iterator[torch.Tensor]:
        timesteps = self.selected_timesteps()
        images = images.to(self.device)
        if self.amp_dtype is not None:
            images = images.float()
        batch = images.shape[0]
        self.model.eval()
        timestep_bar = ProgressReporter(
            len(timesteps),
            progress_description,
            enabled=progress,
            unit="step",
            leave=progress_leave,
        )
        try:
            for timestep in timesteps:
                t = torch.full((batch,), timestep, device=self.device, dtype=torch.long)
                noise = self.noise_plan.sample(images.shape, self.device, images.dtype)
                x_t = self.diffusion.q_sample(images, t, noise)
                autocast = (
                    nullcontext() if self.amp_dtype is None else torch.autocast("cuda", dtype=self.amp_dtype)
                )
                with autocast:
                    predicted_noise = self.model(x_t, t)
                if self.amp_dtype is not None:
                    predicted_noise = predicted_noise.float()
                # Keep the small transition residuals in FP32 during mixed-precision inference.
                gt_transition = self.diffusion.posterior_mean_from_noise(x_t, noise, t)
                model_transition = self.diffusion.posterior_mean_from_noise(x_t, predicted_noise, t)
                yield (gt_transition - model_transition).square()
                timestep_bar.update(postfix={"t": timestep})
        finally:
            timestep_bar.close()

    @torch.no_grad()
    def compute_deviation_stack(
        self,
        images: torch.Tensor,
        progress: bool = False,
        progress_description: str = "ANDi timesteps",
        progress_leave: bool = False,
    ) -> torch.Tensor:
        """回傳 squared transition deviation，形狀為 [B, T, C, H, W]。"""

        deviations = list(self._iter_deviations(images, progress, progress_description, progress_leave))
        return torch.stack(deviations, dim=1)

    @property
    def streams_timesteps(self) -> bool:
        return self.streaming and type(self.time_aggregator) in _STREAMING_AGGREGATORS

    @torch.no_grad()
    def compute_scores_streaming(
        self,
        images: torch.Tensor,
        progress: bool = False,
        progress_description: str = "ANDi timesteps",
        progress_leave: bool = False,
    ) -> torch.Tensor:
        """Pool deviations without retaining a full timestep stack."""

        aggregator_type = type(self.time_aggregator)
        if aggregator_type not in _STREAMING_AGGREGATORS:
            stack = self.compute_deviation_stack(images, progress, progress_description, progress_leave)
            return self.pool_modalities(self.aggregate_time(stack))
        deviations = self._iter_deviations(images, progress, progress_description, progress_leave)
        first = next(deviations)
        # Preserve log(0) -> -inf -> zero, matching the stack aggregator.
        accumulator = first.log() if aggregator_type is GeometricMeanAggregator else first
        count = 1
        for deviation in deviations:
            if aggregator_type is GeometricMeanAggregator:
                accumulator.add_(deviation.log())
            elif aggregator_type is MaxAggregator:
                torch.maximum(accumulator, deviation, out=accumulator)
            else:
                accumulator.add_(deviation)
            count += 1
        if aggregator_type in {GeometricMeanAggregator, MeanAggregator}:
            accumulator.div_(count)
        if aggregator_type is GeometricMeanAggregator:
            accumulator.exp_()
        return self.pool_modalities(accumulator)

    def aggregate_time(self, deviations: torch.Tensor) -> torch.Tensor:
        return self.time_aggregator(deviations, dim=1)

    def pool_modalities(self, scores: torch.Tensor) -> torch.Tensor:
        return self.modality_aggregator(scores, dim=1)

    def postprocess(self, anomaly_map: torch.Tensor) -> dict[str, Any]:
        processed = self.postprocess_policy.process(anomaly_map)
        if self.threshold in supported_threshold_methods():
            segmentation = processed.binary_mask_mf_postprocessed
            thresholds = processed.thresholds_mf
        else:
            threshold = float(self.threshold)
            segmentation = self.postprocess_policy.fixed_threshold_mask(processed.score_mf, threshold)
            thresholds = torch.full(
                (processed.score_mf.shape[0],),
                threshold,
                device=processed.score_mf.device,
                dtype=processed.score_mf.dtype,
            )
        output: dict[str, Any] = {
            # Existing aliases remain stable for downstream callers.
            "anomaly_map": processed.score_raw,
            "anomaly_map_filtered": processed.score_mf,
            "segmentation": segmentation,
            "thresholds": thresholds,
            "threshold_method": processed.threshold_method,
            # Explicit products make raw/MF and pre/post-mask stages auditable.
            "score_raw": processed.score_raw,
            "score_mf": processed.score_mf,
            "thresholds_raw": processed.thresholds_raw,
            "thresholds_mf": processed.thresholds_mf,
            "binary_mask_raw": processed.binary_mask_raw,
            "binary_mask_mf": processed.binary_mask_mf,
            "binary_mask_raw_postprocessed": processed.binary_mask_raw_postprocessed,
            "binary_mask_mf_postprocessed": processed.binary_mask_mf_postprocessed,
        }
        if processed.threshold_method == "yen":
            output.update(
                {
                    "yen_thresholds_raw": processed.thresholds_raw,
                    "yen_thresholds_mf": processed.thresholds_mf,
                    "yen_mask_raw": processed.binary_mask_raw,
                    "yen_mask_mf": processed.binary_mask_mf,
                    "yen_mask_raw_postprocessed": processed.binary_mask_raw_postprocessed,
                    "yen_mask_mf_postprocessed": processed.binary_mask_mf_postprocessed,
                }
            )
        return output

    def detect(self, images: torch.Tensor) -> dict[str, Any]:
        deviations = self.compute_deviation_stack(images)
        per_modality = self.aggregate_time(deviations)
        anomaly_map = self.pool_modalities(per_modality)
        output = self.postprocess(anomaly_map)
        output["deviation_stack"] = deviations
        output["per_modality_scores"] = per_modality
        return output

    def describe_inference(self) -> dict[str, Any]:
        if not hasattr(self, "inference_profile"):
            return {}
        return {
            "profile": self.inference_profile,
            "t_lower": self.t_lower,
            "t_upper": self.t_upper,
            "timesteps": self.selected_timesteps(),
            "streaming": self.streams_timesteps,
            "amp_dtype": str(self.amp_dtype).removeprefix("torch.") if self.amp_dtype else "none",
            "attention_backend": self.attention_backend,
            "device": str(self.device),
            "cudnn_benchmark": self.cudnn_benchmark,
        }

    def inference_cache_options(self) -> dict[str, Any] | None:
        """Identify changed scoring options without invalidating baseline caches."""

        if not hasattr(self, "inference_profile"):
            return None
        if (
            self.inference_profile != "original"
            or self.num_timesteps is not None
            or self.streaming
            or self.amp_dtype is not None
            or self.attention_backend == "no_weights"
        ):
            return self.describe_inference()
        return None

    def describe(self) -> dict:
        postprocessing = self.postprocess_policy.describe()
        resolved_median = postprocessing.get("median_filter_settings", {})
        return {
            "t_lower": self.t_lower,
            "t_upper": self.t_upper,
            "aggregation": self.time_aggregator.describe(),
            "modality_pool": self.modality_aggregator.describe(),
            "inference_options": self.describe_inference(),
            "median_filter": {
                "enabled": resolved_median.get("enabled", self.median_enabled),
                "kernel_size": resolved_median.get("kernel_size", self.median_kernel_size),
                "mode": resolved_median.get("mode", self.median_mode),
            },
            "threshold": self.threshold,
            "threshold_method": self.postprocess_policy.threshold_method,
            "postprocessing": postprocessing,
        }
