"""Metric reduction and threshold-sweep algorithms for volume evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from andi_rewrite.anomaly.postprocess import PostprocessPolicy, PostprocessResult
from andi_rewrite.metrics.classification import auprc, dice
from andi_rewrite.utils.progress import ProgressReporter


def threshold_values(start: float, end: float, step: float) -> list[float]:
    """Build the legacy rounded, half-open threshold sweep."""

    values = np.arange(start, end, step)
    return [round(float(value), 3) for value in values]


def dice_mean(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """Preserve the historical per-subject mean Dice reduction order."""

    values = []
    for pred_item, label_item in zip(prediction, label):
        values.append(dice(pred_item, label_item))
    return float(np.mean(values)) if values else 0.0


def binary_rates(prediction: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    """Compute micro voxel rates using the existing bool reductions."""

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


def segmentation_metrics(segmentation: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    """Return threshold sweep metrics with the historical schema."""

    rates = binary_rates(segmentation, label)
    return {
        "dice": dice_mean(segmentation, label),
        "sensitivity": rates["sensitivity"],
        "precision": rates["precision"],
    }


def threshold_method_metrics(
    segmentation: torch.Tensor,
    thresholds: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, float]:
    """Reduce an adaptive threshold branch without changing float dtype/order."""

    values = segmentation_metrics(segmentation, label)
    values["threshold"] = (
        float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
    )
    return values


def summarize_without_labels(
    processed: PostprocessResult,
    *,
    thresholds: list[float],
    compute_auprc: bool,
    auprc_mode: str,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Produce the legacy N/A score tables for unlabeled data."""

    scores: dict[Any, Any] = {}
    scores_mf: dict[Any, Any] = {}
    for threshold in thresholds:
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
    if compute_auprc:
        scores["AUPRC"] = "N/A"
        scores_mf["AUPRC"] = "N/A"
        scores["AUPRC_mode"] = auprc_mode
        scores_mf["AUPRC_mode"] = auprc_mode
    scores.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
    scores_mf.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
    return scores, scores_mf


def summarize_processed(
    processed: PostprocessResult,
    labels: torch.Tensor | None,
    *,
    thresholds: list[float],
    policy: PostprocessPolicy,
    metric_threshold: float,
    compute_auprc: bool,
    auprc_mode: str,
    auprc_max_samples: int | None,
    auprc_seed: int,
    progress_enabled: bool,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Compute every in-memory metric from an already materialized policy result."""

    anomaly_map = processed.score_raw
    anomaly_map_mf = processed.score_mf

    if labels is None:
        return summarize_without_labels(
            processed,
            thresholds=thresholds,
            compute_auprc=compute_auprc,
            auprc_mode=auprc_mode,
        )

    scores: dict[Any, Any] = {}
    scores_mf: dict[Any, Any] = {}
    summary_bar = ProgressReporter(
        len(thresholds) + 3,
        "Summarizing metrics",
        enabled=progress_enabled,
        unit="metric",
    )
    try:
        for threshold in thresholds:
            segmentation = policy.fixed_threshold_mask(anomaly_map, threshold)
            segmentation_mf = policy.fixed_threshold_mask(anomaly_map_mf, threshold)
            scores[threshold] = segmentation_metrics(segmentation, labels)
            scores_mf[threshold] = segmentation_metrics(segmentation_mf, labels)
            summary_bar.update(postfix={"thr": threshold})

        method = processed.threshold_method
        threshold_metrics = threshold_method_metrics(
            processed.binary_mask_raw_postprocessed,
            processed.thresholds_raw,
            labels,
        )
        threshold_metrics_mf = threshold_method_metrics(
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
        if compute_auprc:
            scores["AUPRC"] = auprc(
                anomaly_map,
                labels,
                max_samples=auprc_max_samples,
                seed=auprc_seed,
                mode=auprc_mode,
            )
            scores_mf["AUPRC"] = auprc(
                anomaly_map_mf,
                labels,
                max_samples=auprc_max_samples,
                seed=auprc_seed,
                mode=auprc_mode,
            )
            scores["AUPRC_mode"] = auprc_mode
            scores_mf["AUPRC_mode"] = auprc_mode
            if auprc_mode == "sampled" and auprc_max_samples is not None:
                sampled = min(int(anomaly_map.numel()), auprc_max_samples)
                scores["AUPRC_samples"] = sampled
                scores_mf["AUPRC_samples"] = sampled
                scores["AUPRC_seed"] = auprc_seed
                scores_mf["AUPRC_seed"] = auprc_seed
            summary_bar.update(postfix="AUPRC")

        rates = binary_rates(anomaly_map > metric_threshold, labels)
        rates_mf = binary_rates(anomaly_map_mf > metric_threshold, labels)
        scores.update(rates)
        scores_mf.update(rates_mf)
        summary_bar.update(postfix="rates")
    finally:
        summary_bar.close()
    return scores, scores_mf
