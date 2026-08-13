from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from andi_rewrite.anomaly.aggregation import build_aggregator
from andi_rewrite.anomaly.postprocess import (
    PostprocessPolicy,
    build_postprocess_policy,
    supported_threshold_methods,
)
from andi_rewrite.diffusion import DDPMDiffusion
from andi_rewrite.noise import NoisePlan
from andi_rewrite.utils.progress import ProgressReporter


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
    ):
        self.config = dict(config)
        self.model = model
        self.diffusion = diffusion
        self.noise_plan = noise_plan
        self.device = torch.device(device)
        self.t_lower = int(config.get("t_lower", config.get("start", 75)))
        self.t_upper = int(config.get("t_upper", config.get("stop", 200)))
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

    def compute_deviation_stack(
        self,
        images: torch.Tensor,
        progress: bool = False,
        progress_description: str = "ANDi timesteps",
        progress_leave: bool = False,
    ) -> torch.Tensor:
        """回傳 squared transition deviation，形狀為 [B, T, C, H, W]。"""

        if self.t_lower <= 0:
            raise ValueError("ANDi t_lower must be >= 1 because x_0 has no previous transition.")
        if self.t_upper <= self.t_lower:
            raise ValueError("ANDi t_upper must be greater than t_lower.")

        images = images.to(self.device)
        batch = images.shape[0]
        deviations = []
        self.model.eval()
        timestep_bar = ProgressReporter(
            self.t_upper - self.t_lower,
            progress_description,
            enabled=progress,
            unit="step",
            leave=progress_leave,
        )
        with torch.no_grad():
            try:
                for timestep in reversed(range(self.t_lower, self.t_upper)):
                    t = torch.full((batch,), timestep, device=self.device, dtype=torch.long)
                    noise = self.noise_plan.sample(images.shape, self.device, images.dtype)
                    x_t = self.diffusion.q_sample(images, t, noise)
                    predicted_noise = self.model(x_t, t)
                    # 比較真實 DDPM backward transition 與模型預測的 normative denoising
                    # transition；ANDi 以兩者差異作為異常分數來源。
                    gt_transition = self.diffusion.posterior_mean_from_noise(x_t, noise, t)
                    model_transition = self.diffusion.posterior_mean_from_noise(x_t, predicted_noise, t)
                    deviations.append((gt_transition - model_transition).square())
                    timestep_bar.update(postfix={"t": timestep})
            finally:
                timestep_bar.close()
        return torch.stack(deviations, dim=1)

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

    def describe(self) -> dict:
        postprocessing = self.postprocess_policy.describe()
        resolved_median = postprocessing.get("median_filter_settings", {})
        return {
            "t_lower": self.t_lower,
            "t_upper": self.t_upper,
            "aggregation": self.time_aggregator.describe(),
            "modality_pool": self.modality_aggregator.describe(),
            "median_filter": {
                "enabled": resolved_median.get("enabled", self.median_enabled),
                "kernel_size": resolved_median.get("kernel_size", self.median_kernel_size),
                "mode": resolved_median.get("mode", self.median_mode),
            },
            "threshold": self.threshold,
            "threshold_method": self.postprocess_policy.threshold_method,
            "postprocessing": postprocessing,
        }
