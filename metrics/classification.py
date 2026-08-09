"""metric 工具，並提供輕量 dependency fallback。

evaluator 會同時用這些函式評估 raw 與 postprocessed anomaly map。
函式可接受 torch tensor 或 numpy array，讓 engine code 保持簡潔。
"""

from __future__ import annotations

import csv
import heapq
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


AUPRC_MODES = ("sampled", "exact")


def resolve_auprc_mode(
    mode: str | None,
    max_samples: int | None,
    *,
    enabled: bool = True,
) -> tuple[str, int | None]:
    """Resolve an explicit AUPRC policy while preserving legacy configs."""

    normalized_max = (
        int(max_samples)
        if max_samples not in (None, "", 0, False)
        else None
    )
    normalized_mode = str(mode).strip().lower() if mode not in (None, "") else None
    if normalized_mode is None:
        normalized_mode = "sampled" if normalized_max is not None else "exact"
    if normalized_mode not in AUPRC_MODES:
        supported = ", ".join(AUPRC_MODES)
        raise ValueError(
            f"Unknown AUPRC mode: {mode!r}. Supported modes: {supported}."
        )
    if enabled and normalized_mode == "sampled" and normalized_max is None:
        raise ValueError(
            "auprc_mode='sampled' requires a positive auprc_max_samples value."
        )
    if enabled and normalized_mode == "exact" and normalized_max is not None:
        raise ValueError(
            "auprc_mode='exact' cannot be combined with auprc_max_samples. "
            "Remove auprc_max_samples or select auprc_mode='sampled'."
        )
    return normalized_mode, normalized_max


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def auprc(
    scores: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    max_samples: int | None = None,
    seed: int = 73,
    mode: str | None = None,
) -> float:
    """在所有 voxel/pixel 上計算 average precision。"""

    resolved_mode, resolved_max_samples = resolve_auprc_mode(
        mode,
        max_samples,
        enabled=True,
    )
    y_score = _to_numpy(scores).reshape(-1)
    y_true = _to_numpy(target).astype(bool).reshape(-1)
    if (
        resolved_mode == "sampled"
        and resolved_max_samples is not None
        and y_score.shape[0] > resolved_max_samples
    ):
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(
            0,
            y_score.shape[0],
            size=int(resolved_max_samples),
            dtype=np.int64,
        )
        y_score = y_score[sample_indices]
        y_true = y_true[sample_indices]
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


def external_auprc(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    work_directory: str | Path,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
    cleanup: bool = True,
) -> float:
    """Compute exact all-voxel AP using bounded-memory sorted disk runs.

    Equal float32 scores are grouped across run boundaries before precision is
    updated, matching sklearn's binary non-interpolated average precision.
    """

    work_path = Path(work_directory)
    if int(chunk_bytes) <= 0:
        raise ValueError("external_sort.chunk_bytes must be a positive integer.")
    if work_path.exists():
        shutil.rmtree(work_path)
    work_path.mkdir(parents=True, exist_ok=False)

    # Sorting needs an int64 order plus input/output arrays.  The conservative
    # divisor keeps all live arrays inside the requested memory budget.
    max_records = max(1, int(chunk_bytes) // 24)
    pending_scores: list[np.ndarray] = []
    pending_labels: list[np.ndarray] = []
    pending_count = 0
    run_paths: list[tuple[Path, Path]] = []
    total_positive = 0
    total_count = 0
    run_scores: list[np.ndarray] = []
    run_labels: list[np.ndarray] = []

    def flush_run() -> None:
        nonlocal pending_count
        if pending_count == 0:
            return
        scores_array = np.concatenate(pending_scores).astype(np.float32, copy=False)
        labels_array = np.concatenate(pending_labels).astype(np.bool_, copy=False)
        order = np.argsort(scores_array, kind="mergesort")[::-1]
        run_index = len(run_paths)
        score_path = work_path / f"run_{run_index:06d}_scores.npy"
        label_path = work_path / f"run_{run_index:06d}_labels.npy"
        np.save(score_path, scores_array[order], allow_pickle=False)
        np.save(label_path, labels_array[order], allow_pickle=False)
        run_paths.append((score_path, label_path))
        pending_scores.clear()
        pending_labels.clear()
        pending_count = 0

    try:
        for scores, labels in chunks:
            score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
            label_values = np.asarray(labels, dtype=np.bool_).reshape(-1)
            if score_values.shape[0] != label_values.shape[0]:
                raise ValueError(
                    "AUPRC score and label chunks must contain the same number of values."
                )
            score_values = np.nan_to_num(
                score_values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            total_positive += int(label_values.sum(dtype=np.int64))
            total_count += int(score_values.shape[0])
            offset = 0
            while offset < score_values.shape[0]:
                available = max_records - pending_count
                take = min(available, score_values.shape[0] - offset)
                end = offset + take
                pending_scores.append(np.array(score_values[offset:end], copy=True))
                pending_labels.append(np.array(label_values[offset:end], copy=True))
                pending_count += take
                offset = end
                if pending_count >= max_records:
                    flush_run()
        flush_run()

        if total_count == 0 or total_positive == 0:
            return 0.0

        run_scores = [np.load(path, mmap_mode="r") for path, _ in run_paths]
        run_labels = [np.load(path, mmap_mode="r") for _, path in run_paths]
        positions = [0 for _ in run_paths]
        heap: list[tuple[float, int]] = []
        for run_index, values in enumerate(run_scores):
            if values.size:
                heapq.heappush(heap, (-float(values[0]), run_index))

        cumulative_positive = 0
        cumulative_total = 0
        average_precision = 0.0
        while heap:
            score = -heap[0][0]
            group_positive = 0
            group_total = 0
            while heap and -heap[0][0] == score:
                _, run_index = heapq.heappop(heap)
                values = run_scores[run_index]
                labels = run_labels[run_index]
                start = positions[run_index]
                end = start + 1
                while end < values.shape[0] and float(values[end]) == score:
                    end += 1
                group_total += end - start
                group_positive += int(labels[start:end].sum(dtype=np.int64))
                positions[run_index] = end
                if end < values.shape[0]:
                    heapq.heappush(heap, (-float(values[end]), run_index))

            cumulative_positive += group_positive
            cumulative_total += group_total
            if group_positive:
                precision = cumulative_positive / cumulative_total
                average_precision += (group_positive / total_positive) * precision
        return float(average_precision)
    finally:
        for array in [*run_scores, *run_labels]:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if cleanup and work_path.exists():
            shutil.rmtree(work_path)


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
