from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from andi_rewrite.anomaly.aggregation import build_aggregator
from andi_rewrite.anomaly.postprocess import (
    apply_mask_postprocess,
    apply_score_postprocess,
    normalize_minmax,
    yen_threshold,
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
    ):
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
        postprocess_config = config.get("postprocess", {})
        self.score_postprocess = postprocess_config.get("score", {})
        self.filtered_score_postprocess = postprocess_config.get("score_mf")
        if self.filtered_score_postprocess is None:
            # 保留原版「median-filtered output」路徑；若需要新的 MF 流程，
            # 可以直接在 postprocess.score_mf 指定任意 score pipeline 取代。
            self.filtered_score_postprocess = (
                {
                    "pipeline": [
                        {
                            "type": "median_filter",
                            "kernel_size": self.median_kernel_size,
                            "mode": self.median_mode,
                        }
                    ]
                }
                if self.median_enabled
                else {}
            )
        self.mask_postprocess = postprocess_config.get("mask", {})

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

    def postprocess(self, anomaly_map: torch.Tensor) -> dict[str, torch.Tensor]:
        anomaly_map = normalize_minmax(anomaly_map)
        anomaly_map = apply_score_postprocess(anomaly_map, self.score_postprocess)
        filtered = apply_score_postprocess(anomaly_map, self.filtered_score_postprocess)
        if self.threshold == "yen":
            segmentation, thresholds = yen_threshold(filtered)
        else:
            threshold = float(self.threshold)
            segmentation = filtered > threshold
            thresholds = torch.full((filtered.shape[0],), threshold, device=filtered.device)
        segmentation = apply_mask_postprocess(segmentation, self.mask_postprocess)
        return {
            "anomaly_map": anomaly_map,
            "anomaly_map_filtered": filtered,
            "segmentation": segmentation,
            "thresholds": thresholds,
        }

    def detect(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        deviations = self.compute_deviation_stack(images)
        per_modality = self.aggregate_time(deviations)
        anomaly_map = self.pool_modalities(per_modality)
        output = self.postprocess(anomaly_map)
        output["deviation_stack"] = deviations
        output["per_modality_scores"] = per_modality
        return output

    def describe(self) -> dict:
        return {
            "t_lower": self.t_lower,
            "t_upper": self.t_upper,
            "aggregation": self.time_aggregator.describe(),
            "modality_pool": self.modality_aggregator.describe(),
            "median_filter": {
                "enabled": self.median_enabled,
                "kernel_size": self.median_kernel_size,
            },
            "threshold": self.threshold,
        }
