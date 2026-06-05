from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.anomaly.postprocess import (
    apply_mask_postprocess,
    apply_score_postprocess,
    normalize_minmax,
    yen_threshold,
)
from andi_rewrite.metrics.classification import auprc, dice
from andi_rewrite.utils.progress import ProgressReporter


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
        self.rank = int(config.get("rank", 3))
        self.connectivity = int(config.get("connectivity", 1))
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", config.get("median_enabled", True)))
        self.kernel_size = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        postprocess_config = config.get("postprocess", {})
        self.score_postprocess = postprocess_config.get("score", {})
        self.score_mf_postprocess = postprocess_config.get("score_mf")
        if self.score_mf_postprocess is None:
            # 與原版 ANDi_mf.csv 相容的預設路徑；新實驗可用
            # postprocess.score_mf.pipeline 換成任意 MF 後處理流程。
            self.score_mf_postprocess = (
                {
                    "pipeline": [
                        {
                            "type": "median_filter",
                            "kernel_size": self.kernel_size,
                            "mode": str(median_config.get("mode", "3d")),
                        }
                    ]
                }
                if self.median_enabled
                else {}
            )
        self.threshold_mask_postprocess = postprocess_config.get("threshold_mask", {})
        self.yen_mask_postprocess = postprocess_config.get(
            "yen_mask",
            {
                "binary_dilation": {
                    "enabled": bool(config.get("yen_binary_dilation", True)),
                    "rank": self.rank,
                    "connectivity": self.connectivity,
                    "iterations": 1,
                }
            },
        )

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

    def collect(self, dataloader: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        maps = []
        labels = []
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
                for volume_index, (image, label) in enumerate(dataloader, start=1):
                    anomaly_map = self._volume_scores(image, volume_index=volume_index)
                    label = label.to(anomaly_map.device).bool()
                    if self.accelerator is not None:
                        anomaly_map, label = self.accelerator.gather_for_metrics((anomaly_map, label))
                    maps.append(anomaly_map.detach().cpu())
                    labels.append(label.detach().cpu())
                    volume_bar.update()
            finally:
                volume_bar.close()
        return torch.cat(maps, dim=0), torch.cat(labels, dim=0)

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

    def _yen_metrics(self, anomaly_map: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        segmentation, thresholds = yen_threshold(anomaly_map)
        segmentation = apply_mask_postprocess(segmentation, self.yen_mask_postprocess)
        metrics = self._segmentation_metrics(segmentation, label)
        metrics["threshold"] = float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        return metrics

    def summarize(self, raw_maps: torch.Tensor, labels: torch.Tensor) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """針對 raw 與 MF map 版本計算 threshold-sweep 與 Yen metrics。"""

        anomaly_map = normalize_minmax(raw_maps)
        anomaly_map = apply_score_postprocess(anomaly_map, self.score_postprocess)
        anomaly_map = normalize_minmax(anomaly_map)

        anomaly_map_mf = apply_score_postprocess(anomaly_map, self.score_mf_postprocess)
        anomaly_map_mf = normalize_minmax(anomaly_map_mf)

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
                segmentation = apply_mask_postprocess(anomaly_map > threshold, self.threshold_mask_postprocess)
                segmentation_mf = apply_mask_postprocess(anomaly_map_mf > threshold, self.threshold_mask_postprocess)
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

    def evaluate(self, dataloader: Iterable) -> dict[str, Any]:
        raw_maps, labels = self.collect(dataloader)
        if not self.is_main_process:
            return {}
        scores, scores_mf = self.summarize(raw_maps, labels)
        self.write_original_style_csv(scores, self.output_csv)
        self.write_original_style_csv(scores_mf, self.output_mf_csv)
        return {
            "output": str(self.output_csv),
            "output_mf": str(self.output_mf_csv),
            "AUPRC": scores["AUPRC"],
            "AUPRC_mf": scores_mf["AUPRC"],
            "DiceYen": scores["yen"],
            "DiceYen_mf": scores_mf["yen"],
            "YenThr": scores["yenthr"],
            "YenThr_mf": scores_mf["yenthr"],
        }
