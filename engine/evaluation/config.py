"""Volume evaluation 的 config 解析。

把原本散在 VolumeEvaluator.__init__ 的 config 讀取與 fallback 邏輯集中成一個
dataclass，方便閱讀與測試。所有預設值與 fallback 行為與原版逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VolumeEvaluationConfig:
    size_splits: int
    normalize_input: bool
    output_csv: Path
    output_mf_csv: Path
    threshold_start: float
    threshold_end: float
    threshold_step: float
    metric_threshold: float
    rank: int
    connectivity: int
    median_enabled: bool
    kernel_size: int
    score_postprocess: dict
    score_mf_postprocess: dict
    threshold_mask_postprocess: dict
    yen_mask_postprocess: dict
    progress: Any

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "VolumeEvaluationConfig":
        median_config = config.get("median_filter", {})
        median_enabled = bool(median_config.get("enabled", config.get("median_enabled", True)))
        kernel_size = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        rank = int(config.get("rank", 3))
        connectivity = int(config.get("connectivity", 1))

        postprocess_config = config.get("postprocess", {})
        score_postprocess = postprocess_config.get("score", {})
        score_mf_postprocess = postprocess_config.get("score_mf")
        if score_mf_postprocess is None:
            # 與原版 ANDi_mf.csv 相容的預設路徑；新實驗可用
            # postprocess.score_mf.pipeline 換成任意 MF 後處理流程。
            score_mf_postprocess = (
                {
                    "pipeline": [
                        {
                            "type": "median_filter",
                            "kernel_size": kernel_size,
                            "mode": str(median_config.get("mode", "3d")),
                        }
                    ]
                }
                if median_enabled
                else {}
            )
        threshold_mask_postprocess = postprocess_config.get("threshold_mask", {})
        yen_mask_postprocess = postprocess_config.get(
            "yen_mask",
            {
                "binary_dilation": {
                    "enabled": bool(config.get("yen_binary_dilation", True)),
                    "rank": rank,
                    "connectivity": connectivity,
                    "iterations": 1,
                }
            },
        )

        return cls(
            size_splits=int(config.get("size_splits", config.get("slice_batch_size", 155))),
            normalize_input=bool(config.get("normalize_input", True)),
            output_csv=Path(config.get("output_csv", config.get("output", "outputs/metrics/ANDi.csv"))),
            output_mf_csv=Path(config.get("output_mf_csv", config.get("output_mf", "outputs/metrics/ANDi_mf.csv"))),
            threshold_start=float(config.get("thr_start", config.get("threshold_start", 0.01))),
            threshold_end=float(config.get("thr_end", config.get("threshold_end", 0.3))),
            threshold_step=float(config.get("thr_step", config.get("threshold_step", 0.001))),
            metric_threshold=float(config.get("threshold", 0.5)),
            rank=rank,
            connectivity=connectivity,
            median_enabled=median_enabled,
            kernel_size=kernel_size,
            score_postprocess=score_postprocess,
            score_mf_postprocess=score_mf_postprocess,
            threshold_mask_postprocess=threshold_mask_postprocess,
            yen_mask_postprocess=yen_mask_postprocess,
            progress=config.get("progress", True),
        )
