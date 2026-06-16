"""Volume evaluation 的子職責模組（config / scoring / metrics / writers）。

VolumeEvaluator 仍是對外入口，這個 package 只是把它原本過度集中的職責拆開，
方便後續擴充 metric、輸出格式與 score collector。
"""

from .config import VolumeEvaluationConfig
from .metrics import EvaluationMetricSummarizer
from .volume_scoring import VolumeScoreCollector
from .writers import write_original_style_csv

__all__ = [
    "EvaluationMetricSummarizer",
    "VolumeEvaluationConfig",
    "VolumeScoreCollector",
    "write_original_style_csv",
]
