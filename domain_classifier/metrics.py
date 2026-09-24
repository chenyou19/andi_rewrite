"""Slice and subject-level statistics for the domain-classifier audit.

The functions in this module are deliberately independent of torch models.
They operate on held-out scores and manifest metadata, which makes the
analysis reproducible and keeps test uncertainty separate from training-seed
variation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import numpy as np


def _vector(values: Sequence[object] | np.ndarray, *, dtype: object | None = None) -> np.ndarray:
    array = np.asarray(values if not isinstance(values, np.ndarray) else values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except ImportError:  # pragma: no cover - sklearn is part of the audit env
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        if positives.size == 0 or negatives.size == 0:
            return float("nan")
        comparisons = (positives[:, None] > negatives[None, :]).mean()
        ties = (positives[:, None] == negatives[None, :]).mean()
        return float(comparisons + 0.5 * ties)


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or int(labels.sum()) == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(labels, scores))
    except ImportError:  # pragma: no cover - sklearn is part of the audit env
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        cumulative = np.cumsum(sorted_labels)
        precision = cumulative / np.arange(1, labels.size + 1)
        return float(np.sum(precision * sorted_labels) / max(int(labels.sum()), 1))


def _binary_cross_entropy(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(np.float64), 1.0e-7, 1.0 - 1.0e-7)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)))


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute thresholded and ranking metrics for binary scores.

    ``scores`` are probabilities for class 1.  The returned confusion counts
    and rates are kept explicit so reports cannot accidentally conflate slice
    and subject denominators.
    """

    raw_labels = _vector(labels)
    y_score = _vector(scores, dtype=np.float64)
    if raw_labels.size:
        try:
            numeric_labels = raw_labels.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must contain only 0 and 1.") from exc
        if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0.0, 1.0)).all():
            raise ValueError("labels must contain only finite 0 and 1 values.")
        y_true = numeric_labels.astype(np.int64)
    else:
        y_true = raw_labels.astype(np.int64)
    if y_true.size != y_score.size:
        raise ValueError("labels and scores must have the same length.")
    if y_score.size and (not np.isfinite(y_score).all() or np.any(y_score < 0.0) or np.any(y_score > 1.0)):
        raise ValueError("scores must contain finite probabilities in [0, 1].")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    predicted = (y_score >= float(threshold)).astype(np.int64)
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    accuracy = float((tp + tn) / max(y_true.size, 1))
    return {
        "n": int(y_true.size),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "roc_auc": _safe_auc(y_true, y_score),
        "average_precision": _safe_average_precision(y_true, y_score),
        "bce": _binary_cross_entropy(y_true, y_score) if y_true.size else float("nan"),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def aggregate_subject_predictions(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    participant_ids: Sequence[object] | np.ndarray,
    *,
    pair_ids: Sequence[object] | np.ndarray | None = None,
    method: str = "mean",
) -> dict[str, np.ndarray]:
    """Aggregate all slices from each participant into one score.

    A participant with multiple slices contributes one subject observation to
    subject-level metrics.  ``pair_id`` is retained when supplied so matched
    pair bootstrap and swap tests can resample complete pairs.
    """

    raw_labels = _vector(labels)
    try:
        numeric_labels = raw_labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain only 0 and 1.") from exc
    if numeric_labels.size and (not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0.0, 1.0)).all()):
        raise ValueError("labels must contain only finite 0 and 1 values.")
    y_true = numeric_labels.astype(np.int64)
    y_score = _vector(scores, dtype=np.float64)
    participants = _vector(participant_ids, dtype=object)
    if not (y_true.size == y_score.size == participants.size):
        raise ValueError("labels, scores, and participant_ids must have the same length.")
    if y_score.size and (not np.isfinite(y_score).all() or np.any(y_score < 0.0) or np.any(y_score > 1.0)):
        raise ValueError("scores must contain finite probabilities in [0, 1].")
    pair_ids_supplied = pair_ids is not None
    if pair_ids is None:
        pairs = np.asarray([None] * y_true.size, dtype=object)
    else:
        pairs = _vector(pair_ids, dtype=object)
        if pairs.size != y_true.size:
            raise ValueError("pair_ids must have the same length as labels.")
        if any(value is None or str(value).strip() == "" for value in pairs.tolist()):
            raise ValueError("pair_ids must be present and non-empty when supplied.")
    normalized_method = str(method).strip().lower()
    if normalized_method not in {"mean", "median", "max"}:
        raise ValueError("method must be one of: mean, median, max.")

    groups: "OrderedDict[object, list[int]]" = OrderedDict()
    for index, participant in enumerate(participants.tolist()):
        groups.setdefault(participant, []).append(index)

    output_labels: list[int] = []
    output_scores: list[float] = []
    output_participants: list[object] = []
    output_pairs: list[object] = []
    output_slice_counts: list[int] = []
    reducer = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
    }[normalized_method]
    for participant, indices in groups.items():
        values = y_true[indices]
        if np.unique(values).size > 1:
            raise ValueError(f"participant {participant!r} has conflicting binary labels.")
        output_labels.append(int(values[0]))
        output_scores.append(float(reducer(y_score[indices])))
        output_participants.append(participant)
        candidate_pairs = [value for value in pairs[indices].tolist() if value is not None]
        if pair_ids_supplied and len(set(candidate_pairs)) != 1:
            raise ValueError(f"participant {participant!r} maps to conflicting pair_ids.")
        output_pairs.append(candidate_pairs[0] if candidate_pairs else None)
        output_slice_counts.append(len(indices))
    return {
        "labels": np.asarray(output_labels, dtype=np.int64),
        "scores": np.asarray(output_scores, dtype=np.float64),
        "participant_ids": np.asarray(output_participants, dtype=object),
        "pair_ids": np.asarray(output_pairs, dtype=object),
        "slice_counts": np.asarray(output_slice_counts, dtype=np.int64),
    }


def compute_dataset_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    participant_ids: Sequence[object] | np.ndarray | None = None,
    pair_ids: Sequence[object] | np.ndarray | None = None,
    threshold: float = 0.5,
    subject_method: str = "mean",
) -> dict[str, object]:
    """Return separate slice and subject metric dictionaries."""

    result: dict[str, object] = {
        "slice": compute_binary_metrics(labels, scores, threshold=threshold),
        "subject": None,
    }
    if participant_ids is not None:
        aggregated = aggregate_subject_predictions(
            labels,
            scores,
            participant_ids,
            pair_ids=pair_ids,
            method=subject_method,
        )
        result["subject"] = compute_binary_metrics(
            aggregated["labels"],
            aggregated["scores"],
            threshold=threshold,
        )
        result["subject_predictions"] = aggregated
    return result


def _resampling_groups(
    labels: np.ndarray,
    *,
    pair_ids: Sequence[object] | np.ndarray | None,
) -> list[np.ndarray]:
    if pair_ids is None:
        return [np.asarray([index], dtype=np.int64) for index in range(labels.size)]
    pairs = _vector(pair_ids, dtype=object)
    if pairs.size != labels.size:
        raise ValueError("pair_ids must have the same length as labels.")
    if any(value is None or str(value).strip() == "" for value in pairs.tolist()):
        raise ValueError("pair_ids must be present and non-empty for paired bootstrap.")
    groups: "OrderedDict[object, list[int]]" = OrderedDict()
    for index, pair in enumerate(pairs.tolist()):
        groups.setdefault(pair, []).append(index)
    output: list[np.ndarray] = []
    for pair, indices in groups.items():
        if len(indices) != 2 or sorted(int(labels[index]) for index in indices) != [0, 1]:
            raise ValueError(
                f"pair_id {pair!r} must contain exactly one class-0 and one class-1 subject."
            )
        output.append(np.asarray(indices, dtype=np.int64))
    return output


def bootstrap_subject_auc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 73,
    pair_ids: Sequence[object] | np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict[str, float | int | list[float] | str]:
    """Bootstrap subject AUC, resampling complete subjects or matched pairs."""

    raw_labels = _vector(labels)
    try:
        numeric_labels = raw_labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain only 0 and 1.") from exc
    if numeric_labels.size and (not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0.0, 1.0)).all()):
        raise ValueError("labels must contain only finite 0 and 1 values.")
    y_true = numeric_labels.astype(np.int64)
    y_score = _vector(scores, dtype=np.float64)
    if y_true.size != y_score.size:
        raise ValueError("labels and scores must have the same length.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    groups = _resampling_groups(y_true, pair_ids=pair_ids)
    observed = _safe_auc(y_true, y_score)
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    if len(groups) and int(n_bootstrap) > 0:
        for _ in range(int(n_bootstrap)):
            chosen = rng.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in chosen])
            auc = _safe_auc(y_true[indices], y_score[indices])
            if np.isfinite(auc):
                samples.append(float(auc))
    alpha = (1.0 - float(confidence)) / 2.0
    if samples:
        lower, upper = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha])
        ci_low, ci_high = float(lower), float(upper)
    else:
        ci_low = ci_high = float("nan")
    return {
        "observed_auc": float(observed),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "n_valid": int(len(samples)),
        "resampling_unit": "matched_pair" if pair_ids is not None else "subject",
        "samples": samples,
    }


def paired_heldout_swap_test(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    pair_ids: Sequence[object] | np.ndarray,
    *,
    n_swaps: int = 1000,
    seed: int = 73,
) -> dict[str, float | int | str | bool]:
    """Two-sided held-out whole-pair label swap test for a fixed classifier.

    Each valid pair must contain exactly one class-0 and one class-1 subject.
    A replicate flips both labels in a pair together, preserving the pair
    structure.  The +1 correction is applied to the two-sided tail count.
    This is a conditional matched-sample association test, not an
    unconditional permutation test for the full source populations.
    """

    y_true = _vector(labels, dtype=np.int64)
    y_score = _vector(scores, dtype=np.float64)
    pairs = _vector(pair_ids, dtype=object)
    if not (y_true.size == y_score.size == pairs.size):
        raise ValueError("labels, scores, and pair_ids must have the same length.")
    if y_score.size and (not np.isfinite(y_score).all() or np.any(y_score < 0.0) or np.any(y_score > 1.0)):
        raise ValueError("scores must contain finite probabilities in [0, 1].")
    if any(value is None or str(value).strip() == "" for value in pairs.tolist()):
        raise ValueError("pair_ids must be present and non-empty.")
    if int(n_swaps) <= 0:
        raise ValueError("n_swaps must be a positive integer.")
    groups: "OrderedDict[object, tuple[int, int]]" = OrderedDict()
    invalid_pairs: list[object] = []
    grouped: "OrderedDict[object, list[int]]" = OrderedDict()
    for index, pair in enumerate(pairs.tolist()):
        grouped.setdefault(pair, []).append(index)
    for pair, indices in grouped.items():
        positives = [index for index in indices if y_true[index] == 1]
        negatives = [index for index in indices if y_true[index] == 0]
        if len(indices) == 2 and len(positives) == 1 and len(negatives) == 1:
            groups[pair] = (positives[0], negatives[0])
        else:
            invalid_pairs.append(pair)
    if invalid_pairs:
        raise ValueError(
            "Every pair_id must contain exactly one class-0 and one class-1 subject; "
            f"invalid pairs={invalid_pairs[:5]!r}."
        )
    valid_indices = np.asarray(
        [index for pair in groups.values() for index in pair],
        dtype=np.int64,
    )
    observed = _safe_auc(y_true[valid_indices], y_score[valid_indices]) if valid_indices.size else float("nan")
    rng = np.random.default_rng(int(seed))
    null_values: list[float] = []
    pair_values = list(groups.values())
    for _ in range(int(n_swaps)):
        permuted = y_true[valid_indices].copy()
        local_positions = {original: local for local, original in enumerate(valid_indices.tolist())}
        for positive_index, negative_index in pair_values:
            if rng.integers(0, 2):
                positive_local = local_positions[positive_index]
                negative_local = local_positions[negative_index]
                permuted[positive_local], permuted[negative_local] = (
                    permuted[negative_local],
                    permuted[positive_local],
                )
        value = _safe_auc(permuted, y_score[valid_indices])
        if np.isfinite(value):
            null_values.append(float(value))
    deviation = abs(float(observed) - 0.5) if np.isfinite(observed) else float("nan")
    if np.isfinite(observed) and null_values:
        extreme = sum(abs(value - 0.5) >= deviation - 1.0e-15 for value in null_values)
        p_value = float((1 + extreme) / (1 + len(null_values)))
    else:
        p_value = float("nan")
    return {
        "observed_auc": float(observed),
        "two_sided_deviation": float(deviation),
        "p_value": p_value,
        "n_swaps": int(n_swaps),
        "n_valid": int(len(null_values)),
        "n_pairs": int(len(groups)),
        "n_invalid_pairs": int(len(invalid_pairs)),
        "status": "complete" if len(null_values) == int(n_swaps) else "incomplete",
        "null_auc_mean": float(np.mean(null_values)) if null_values else float("nan"),
        "null_ci_low": float(np.quantile(null_values, 0.025)) if null_values else float("nan"),
        "null_ci_high": float(np.quantile(null_values, 0.975)) if null_values else float("nan"),
        "null_auc_values": [float(value) for value in null_values],
        "conditional_on_matched_pairs": True,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family adjustment for finite p-values."""

    finite = [(name, float(value)) for name, value in p_values.items() if np.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {name: float("nan") for name in p_values}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


# Explicit alias used in reports and by callers that prefer the shorter name.
paired_swap_test = paired_heldout_swap_test


__all__ = [
    "aggregate_subject_predictions",
    "bootstrap_subject_auc",
    "compute_binary_metrics",
    "compute_dataset_metrics",
    "holm_adjust",
    "paired_heldout_swap_test",
    "paired_swap_test",
]
