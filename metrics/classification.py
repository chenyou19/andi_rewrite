"""metric 工具，並提供輕量 dependency fallback。

evaluator 會同時用這些函式評估 raw 與 postprocessed anomaly map。
函式可接受 torch tensor 或 numpy array，讓 engine code 保持簡潔。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def auprc(scores: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """在所有 voxel/pixel 上計算 average precision。"""

    y_score = _to_numpy(scores).reshape(-1)
    y_true = _to_numpy(target).astype(bool).reshape(-1)
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        # fallback 版本讓沒有 sklearn 的環境仍可計算 AUPRC。
        order = np.argsort(-y_score)
        y_true = y_true[order]
        tp = np.cumsum(y_true)
        fp = np.cumsum(~y_true)
        precision = tp / np.maximum(tp + fp, 1)
        recall_step = y_true / max(y_true.sum(), 1)
        return float(np.sum(precision * recall_step))


def dice(prediction: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, eps: float = 1.0e-8) -> float:
    pred = _to_numpy(prediction).astype(bool)
    truth = _to_numpy(target).astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    denominator = pred.sum() + truth.sum()
    return float((2.0 * intersection) / max(float(denominator), eps))


def confusion_rates(prediction: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> dict[str, float]:
    """由 binary prediction 計算 sensitivity 與 specificity。"""

    pred = _to_numpy(prediction).astype(bool)
    truth = _to_numpy(target).astype(bool)
    tp = np.logical_and(pred, truth).sum()
    tn = np.logical_and(~pred, ~truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    return {
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def dice_yen(scores: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    score_array = _to_numpy(scores)
    try:
        from skimage.filters import threshold_yen

        threshold = threshold_yen(score_array)
    except ImportError:
        threshold = float(score_array.mean() + score_array.std())
    return dice(score_array > threshold, target)


def compute_metrics(
    scores: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    prediction = _to_numpy(scores) > threshold
    rates = confusion_rates(prediction, target)
    return {
        "AUPRC": auprc(scores, target),
        "Dice": dice(prediction, target),
        "DiceYen": dice_yen(scores, target),
        "sensitivity": rates["sensitivity"],
        "specificity": rates["specificity"],
    }


def write_metrics_csv(metrics: dict[str, Any], output_path: str | Path) -> None:
    """將 metric dict 寫成穩定的雙欄 CSV 格式。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
