"""Threshold sweep 與 Yen metrics 的彙整。

把 VolumeEvaluator 中跟 metric 計算相關的邏輯集中起來：threshold range 產生、
raw/MF map 後處理、Dice/sensitivity/precision/specificity/AUPRC/Yen。
metric 計算公式與後處理順序與原版逐字一致。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from andi_rewrite.anomaly.postprocess import (
    apply_mask_postprocess,
    apply_score_postprocess,
    normalize_minmax,
    yen_threshold,
)
from andi_rewrite.metrics.classification import auprc, dice
from andi_rewrite.utils.progress import ProgressReporter

from .config import VolumeEvaluationConfig


class EvaluationMetricSummarizer:
    """從 raw anomaly maps 與 labels 算出原版相容的 scores / scores_mf。"""

    def __init__(
        self,
        config: VolumeEvaluationConfig,
        progress_enabled_fn: Callable[[], bool] | None = None,
    ):
        self.config = config
        self._progress_enabled_fn = progress_enabled_fn if progress_enabled_fn is not None else (lambda: False)

    def _progress_enabled(self) -> bool:
        return bool(self._progress_enabled_fn())

    def threshold_values(self) -> list[float]:
        values = np.arange(self.config.threshold_start, self.config.threshold_end, self.config.threshold_step)
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

    def _yen_metrics(self, anomaly_map: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        segmentation, thresholds = yen_threshold(anomaly_map)
        segmentation = apply_mask_postprocess(segmentation, self.config.yen_mask_postprocess)
        metrics = self._segmentation_metrics(segmentation, label)
        metrics["threshold"] = float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        return metrics

    def summarize(self, raw_maps: torch.Tensor, labels: torch.Tensor) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """針對 raw 與 MF map 版本計算 threshold-sweep 與 Yen metrics。"""

        anomaly_map = normalize_minmax(raw_maps)
        anomaly_map = apply_score_postprocess(anomaly_map, self.config.score_postprocess)
        anomaly_map = normalize_minmax(anomaly_map)

        anomaly_map_mf = apply_score_postprocess(anomaly_map, self.config.score_mf_postprocess)
        anomaly_map_mf = normalize_minmax(anomaly_map_mf)

        scores: dict[Any, Any] = {}
        scores_mf: dict[Any, Any] = {}
        thresholds = self.threshold_values()
        summary_bar = ProgressReporter(
            len(thresholds) + 3,
            "Summarizing metrics",
            enabled=self._progress_enabled(),
            unit="metric",
        )
        try:
            for threshold in thresholds:
                segmentation = apply_mask_postprocess(anomaly_map > threshold, self.config.threshold_mask_postprocess)
                segmentation_mf = apply_mask_postprocess(anomaly_map_mf > threshold, self.config.threshold_mask_postprocess)
                scores[threshold] = self._segmentation_metrics(segmentation, labels)
                scores_mf[threshold] = self._segmentation_metrics(segmentation_mf, labels)
                summary_bar.update(postfix={"thr": threshold})

            yen_metrics = self._yen_metrics(anomaly_map, labels)
            yen_metrics_mf = self._yen_metrics(anomaly_map_mf, labels)
            scores["yen"] = yen_metrics["dice"]
            scores["yenthr"] = yen_metrics["threshold"]
            scores["yensen"] = yen_metrics["sensitivity"]
            scores["yenpre"] = yen_metrics["precision"]
            scores_mf["yen"] = yen_metrics_mf["dice"]
            scores_mf["yenthr"] = yen_metrics_mf["threshold"]
            scores_mf["yensen"] = yen_metrics_mf["sensitivity"]
            scores_mf["yenpre"] = yen_metrics_mf["precision"]
            summary_bar.update(postfix="yen")
            scores["AUPRC"] = auprc(anomaly_map, labels)
            scores_mf["AUPRC"] = auprc(anomaly_map_mf, labels)
            summary_bar.update(postfix="AUPRC")

            rates = self._binary_rates(anomaly_map > self.config.metric_threshold, labels)
            rates_mf = self._binary_rates(anomaly_map_mf > self.config.metric_threshold, labels)
            scores.update(rates)
            scores_mf.update(rates_mf)
            summary_bar.update(postfix="rates")
        finally:
            summary_bar.close()
        return scores, scores_mf
