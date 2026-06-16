from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.engine.evaluation import (
    EvaluationMetricSummarizer,
    VolumeEvaluationConfig,
    VolumeScoreCollector,
    write_original_style_csv,
)


class VolumeEvaluator:
    """完整 volume ANDi evaluator，輸出格式相容原版 CSV。

    evaluator 現在只負責 orchestration：config 解析、accelerator prepare，
    以及把 collect -> summarize -> write 串起來。實際的 volume scoring、metric
    彙整與 CSV 寫出分別交給 VolumeScoreCollector / EvaluationMetricSummarizer /
    write_original_style_csv，anomaly scoring 仍交給 ANDiDetector。
    """

    def __init__(
        self,
        detector: ANDiDetector,
        config: dict[str, Any],
        accelerator: Any = None,
    ):
        # detector 同時當作 SliceAnomalyScorer；保留 self.detector 供 report 與
        # 外部程式使用，collector 則只透過 scorer 介面互動。
        self.detector = detector
        self.scorer = detector
        self.raw_config = config
        self.config = VolumeEvaluationConfig.from_dict(config)
        self.accelerator = accelerator
        self.collector = VolumeScoreCollector(
            scorer=self.scorer,
            config=self.config,
            accelerator=accelerator,
            progress_enabled_fn=self._progress_enabled,
        )
        self.summarizer = EvaluationMetricSummarizer(
            config=self.config,
            progress_enabled_fn=self._progress_enabled,
        )
        self.writer = write_original_style_csv

    @property
    def output_csv(self) -> Path:
        return self.config.output_csv

    @property
    def output_mf_csv(self) -> Path:
        return self.config.output_mf_csv

    def _progress_enabled(self) -> bool:
        progress_config = self.config.progress
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

    def collect(self, dataloader: Iterable) -> tuple[Any, Any]:
        return self.collector.collect(dataloader)

    def summarize(self, raw_maps: Any, labels: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
        return self.summarizer.summarize(raw_maps, labels)

    @staticmethod
    def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
        return write_original_style_csv(scores, path)

    def evaluate(self, dataloader: Iterable) -> dict[str, Any]:
        raw_maps, labels = self.collect(dataloader)
        if not self.is_main_process:
            return {}
        scores, scores_mf = self.summarize(raw_maps, labels)
        self.write_original_style_csv(scores, self.config.output_csv)
        self.write_original_style_csv(scores_mf, self.config.output_mf_csv)
        return {
            "output": str(self.config.output_csv),
            "output_mf": str(self.config.output_mf_csv),
            "AUPRC": scores["AUPRC"],
            "AUPRC_mf": scores_mf["AUPRC"],
            "DiceYen": scores["yen"],
            "DiceYen_mf": scores_mf["yen"],
            "YenThr": scores["yenthr"],
            "YenThr_mf": scores_mf["yenthr"],
        }
