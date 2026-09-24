"""Regression tests for immutable held-out label sanity evaluation.

These fixtures contain no images and never fit a model.  They exercise the
join, pair-flip, immutability, and scoring contracts of the read-only helper.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from domain_classifier.metrics import compute_dataset_metrics
from scripts.evaluate_immutable_test_label_sanity import (
    DEFAULT_SWAP_SEEDS,
    evaluate_immutable_test_labels,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for pair_index in range(4):
        pair_id = f"pair-{pair_index}"
        for label in (0, 1):
            record_index = len(manifest_rows)
            participant_id = f"subject-{pair_index}-{label}"
            case_id = f"case-{pair_index}-{label}"
            probability = 0.10 + 0.02 * pair_index if label == 0 else 0.90 - 0.02 * pair_index
            manifest_rows.append(
                {
                    "record_index": record_index,
                    "participant_id": participant_id,
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "label": label,
                }
            )
            prediction_rows.append(
                {
                    "record_index": record_index,
                    "participant_id": participant_id,
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "label": label,
                    "probability": probability,
                }
            )
    manifest_path = root / "test_manifest.jsonl"
    predictions_path = root / "predictions.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(predictions_path, prediction_rows)
    return manifest_path, predictions_path, manifest_rows, prediction_rows


def _run(root: Path, manifest_path: Path, predictions_path: Path, name: str = "run") -> dict[str, object]:
    return evaluate_immutable_test_labels(
        predictions_path=predictions_path,
        test_manifest=manifest_path,
        output_root=root / name,
        bootstrap_replicates=40,
        swap_replicates=40,
    )


def test_true_predictions_and_labels_are_preserved(tmp_path: Path) -> None:
    manifest_path, predictions_path, manifest_rows, prediction_rows = _fixture(tmp_path)
    original_prediction_bytes = predictions_path.read_bytes()

    result = _run(tmp_path, manifest_path, predictions_path)

    assert predictions_path.read_bytes() == original_prediction_bytes
    true_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "test_predictions_true.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["true_label"] for row in true_rows] == [row["label"] for row in manifest_rows]
    assert [row["probability"] for row in true_rows] == [row["probability"] for row in prediction_rows]
    assert result["true_test"]["test"]["subject"]["roc_auc"] == pytest.approx(1.0)
    assert result["true_test"]["test_statistics"]["heldout_pair_swap"]["status"] == "complete"


def test_random_labels_are_whole_pair_flips_and_are_scored_separately(tmp_path: Path) -> None:
    manifest_path, predictions_path, manifest_rows, prediction_rows = _fixture(tmp_path)
    result = _run(tmp_path, manifest_path, predictions_path)
    true_auc = float(result["true_test"]["test"]["subject"]["roc_auc"])
    changed_auc = False

    for random_result in result["random_test_label_results"]:
        seed = int(random_result["seed"])
        output_path = tmp_path / "run" / f"test_predictions_random_seed_{seed}.jsonl"
        random_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in random_rows:
            by_pair[str(row["pair_id"])].append(row)
        for pair_rows in by_pair.values():
            assert len(pair_rows) == 2
            original = {str(row["participant_id"]): int(row["true_label"]) for row in pair_rows}
            random = {str(row["participant_id"]): int(row["random_label"]) for row in pair_rows}
            assert all(random[participant] == 1 - label for participant, label in original.items()) or all(
                random[participant] == label for participant, label in original.items()
            )

        labels = [int(row["random_label"]) for row in random_rows]
        scores = [float(row["probability"]) for row in random_rows]
        participants = [str(row["participant_id"]) for row in random_rows]
        pairs = [str(row["pair_id"]) for row in random_rows]
        expected = compute_dataset_metrics(
            labels,
            scores,
            participant_ids=participants,
            pair_ids=pairs,
        )["subject"]["roc_auc"]
        observed = float(random_result["test"]["subject"]["roc_auc"])
        assert observed == pytest.approx(float(expected))
        changed_auc = changed_auc or observed != pytest.approx(true_auc)

    assert changed_auc
    assert tuple(int(item) for item in result["swap_seeds"]) == DEFAULT_SWAP_SEEDS


def test_three_frozen_swap_seeds_are_reproducible(tmp_path: Path) -> None:
    manifest_path, predictions_path, _manifest_rows, _prediction_rows = _fixture(tmp_path)
    first = _run(tmp_path, manifest_path, predictions_path, name="run_a")
    second = _run(tmp_path, manifest_path, predictions_path, name="run_b")

    assert first["swap_seeds"] == second["swap_seeds"] == list(DEFAULT_SWAP_SEEDS)
    assert first["true_test"] == second["true_test"]
    assert first["random_test_label_results"] == second["random_test_label_results"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("participant_id", "wrong-subject"),
        ("pair_id", "wrong-pair"),
        ("label", 1),
    ),
)
def test_prediction_manifest_join_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    manifest_path, predictions_path, _manifest_rows, prediction_rows = _fixture(tmp_path)
    prediction_rows[0][field] = replacement
    _write_jsonl(predictions_path, prediction_rows)

    with pytest.raises(ValueError, match="mismatch"):
        _run(tmp_path, manifest_path, predictions_path, name=f"bad_{field}")


@pytest.mark.parametrize("field", ("participant_id", "pair_id"))
def test_missing_subject_or_pair_identity_fails_closed(tmp_path: Path, field: str) -> None:
    manifest_path, predictions_path, manifest_rows, prediction_rows = _fixture(tmp_path)
    manifest_rows[0][field] = ""
    prediction_rows[0][field] = ""
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(predictions_path, prediction_rows)

    with pytest.raises(ValueError, match=f"{field} is missing"):
        _run(tmp_path, manifest_path, predictions_path, name=f"missing_{field}")
