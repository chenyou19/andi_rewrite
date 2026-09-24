from __future__ import annotations

import numpy as np
import pytest

from andi_rewrite.domain_classifier.metrics import (
    aggregate_subject_predictions,
    bootstrap_subject_auc,
    compute_binary_metrics,
    paired_heldout_swap_test,
)


def test_binary_metrics_expose_slice_confusion_and_ranking_values() -> None:
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)


def test_metrics_reject_nonbinary_labels_and_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([0.7, 1], [0.1, 0.9])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [-0.1, 1.1])


def test_subject_aggregation_counts_each_participant_once() -> None:
    aggregated = aggregate_subject_predictions(
        [0, 0, 1, 1, 1],
        [0.1, 0.3, 0.7, 0.9, 0.8],
        ["n0", "n0", "p1", "p1", "p1"],
        pair_ids=["pair0", "pair0", "pair1", "pair1", "pair1"],
    )
    assert aggregated["labels"].tolist() == [0, 1]
    assert aggregated["scores"].tolist() == pytest.approx([0.2, 0.8])
    assert aggregated["slice_counts"].tolist() == [2, 3]


def test_matched_pair_bootstrap_and_swap_are_reproducible() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    scores = np.asarray([0.2, 0.8, 0.4, 0.6, 0.45, 0.55])
    pairs = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    bootstrap = bootstrap_subject_auc(labels, scores, pair_ids=pairs, n_bootstrap=200, seed=73)
    assert bootstrap["observed_auc"] == pytest.approx(1.0)
    assert bootstrap["n_valid"] == 200
    swap = paired_heldout_swap_test(labels, scores, pairs, n_swaps=100, seed=73)
    assert swap["n_pairs"] == 3
    assert swap["status"] == "complete"
    assert len(swap["null_auc_values"]) == 100
    assert 0.0 <= swap["p_value"] <= 1.0


def test_pair_controls_fail_closed_on_malformed_pairs() -> None:
    with pytest.raises(ValueError):
        bootstrap_subject_auc([0, 1, 1], [0.2, 0.8, 0.7], pair_ids=["a", "a", "a"], n_bootstrap=10)
    with pytest.raises(ValueError):
        paired_heldout_swap_test([0, 1], [0.2, 0.8], [None, None], n_swaps=10)
