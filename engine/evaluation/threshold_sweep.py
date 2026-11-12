"""Exact fixed-threshold sweep counts for CPU float32 score volumes."""

from __future__ import annotations

from math import isfinite

import numpy as np
import torch


_CHUNK_VOXELS = 1 << 20
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def prepare_cutoffs(thresholds: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    """Cast thresholds as in a Torch float32 scalar comparison, then sort."""

    if not thresholds or any(
        not isfinite(threshold) or abs(threshold) > _FLOAT32_MAX for threshold in thresholds
    ):
        return None
    cutoffs = np.asarray(thresholds, dtype=np.float32)
    order = np.argsort(cutoffs, kind="stable")
    return cutoffs[order], order


def supports_inputs(
    score_raw: torch.Tensor,
    score_mf: torch.Tensor,
    label: torch.Tensor,
) -> bool:
    """Check that bucketing uses the same shapes and scalar precision as masks."""

    return all(
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and tensor.shape == label.shape
        and not tensor.is_neg()
        and not tensor.is_conj()
        for tensor in (score_raw, score_mf, label)
    ) and score_raw.dtype == score_mf.dtype == torch.float32


def update_bucket_stats(
    sweep: dict[float, dict[str, float | int]],
    thresholds: list[float],
    cutoffs: tuple[np.ndarray, np.ndarray],
    score: torch.Tensor,
    label_mask: np.ndarray,
) -> None:
    """Accumulate one subject's sweep counts in bounded voxel chunks."""

    sorted_cutoffs, order = cutoffs
    total = np.zeros(sorted_cutoffs.size + 1, dtype=np.int64)
    positive = np.zeros_like(total)
    chunks = np.nditer(
        [score.detach().numpy(), label_mask],
        flags=["buffered", "external_loop", "zerosize_ok"],
        buffersize=_CHUNK_VOXELS,
    )
    for score_chunk, label_chunk in chunks:
        # Match the fixed mask's nonfinite-to-zero sanitization.
        finite_score = np.nan_to_num(score_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        # A voxel contributes only to cutoffs strictly below its score.
        buckets = np.searchsorted(sorted_cutoffs, finite_score, side="left")
        total += np.bincount(buckets, minlength=total.size)
        positive += np.bincount(buckets[label_chunk], minlength=total.size)

    predicted = np.empty_like(order)
    true_positive = np.empty_like(order)
    predicted[order] = total[:0:-1].cumsum()[::-1]
    true_positive[order] = positive[:0:-1].cumsum()[::-1]
    label_positive = int(positive.sum())
    voxel_count = score.numel()

    # Preserve list order and repeated updates to duplicate threshold keys.
    for threshold, predicted_count, tp_count in zip(thresholds, predicted, true_positive):
        stats = sweep[threshold]
        predicted_count = int(predicted_count)
        tp = int(tp_count)
        fp = predicted_count - tp
        fn = label_positive - tp
        tn = voxel_count - tp - fp - fn
        dice_value = 2.0 * tp / max(float(predicted_count + label_positive), 1.0e-8)
        stats["dice_sum"] = float(stats["dice_sum"]) + dice_value
        stats["subjects"] = int(stats["subjects"]) + 1
        stats["tp"] = int(stats["tp"]) + tp
        stats["tn"] = int(stats["tn"]) + tn
        stats["fp"] = int(stats["fp"]) + fp
        stats["fn"] = int(stats["fn"]) + fn
