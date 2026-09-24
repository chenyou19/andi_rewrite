from __future__ import annotations

import numpy as np
import torch

from andi_rewrite.data.domain_classifier.records import record_from_paths
from andi_rewrite.domain_classifier.runner import (
    DomainClassifierRunner,
    ManifestDataset,
    TrainConfig,
    extract_statistical_features,
    fit_statistical_logistic,
    permute_pair_labels,
    subject_balanced_bce_with_logits,
)


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(2):
        for label in (0, 1):
            participant = f"p{pair_index}_{label}"
            for slice_index in range(2):
                rows.append(
                    {
                        "image": np.full((3, 16, 16), float(label) + 0.05 * slice_index, dtype=np.float32),
                        "label": label,
                        "participant_id": participant,
                        "pair_id": f"pair{pair_index}",
                        "case_id": f"case_{pair_index}_{label}_{slice_index}",
                    }
                )
    return rows


def test_subject_balanced_bce_equalizes_participant_slice_counts() -> None:
    logits = torch.zeros(3)
    labels = torch.tensor([0.0, 1.0, 1.0])
    value = subject_balanced_bce_with_logits(logits, labels, ["short", "long", "long"])
    expected = torch.tensor(np.log(2.0), dtype=value.dtype)
    assert torch.allclose(value, expected)


def test_tiny_runner_returns_final_state_and_predictions() -> None:
    rows = _records()
    dataset = ManifestDataset(rows)
    config = TrainConfig(
        in_channels=3,
        max_epochs=2,
        patience=1,
        batch_size=4,
        device="cpu",
        tiny=True,
    )
    result = DomainClassifierRunner(config).train_one_seed(dataset, dataset, dataset, seed=73)
    assert result["epochs_completed"] == 2
    assert result["best_epoch"] == 2
    assert len(result["test_predictions"]) == len(rows)
    assert len(result["test_subject_predictions"]) == 4
    assert result["train_final"] is not None
    assert result["state_dict"] is not None


def test_runner_accepts_data_agent_slice_records_with_stage_loader() -> None:
    records = []
    for pair_index in range(2):
        for label in (0, 1):
            records.append(
                record_from_paths(
                    split="train",
                    label=label,
                    domain="healthy" if label == 0 else "brats",
                    participant_id=f"subject_{pair_index}_{label}",
                    session_id="session",
                    case_id=f"case_{pair_index}_{label}",
                    z=1,
                    geometry_shape=(16, 16, 4),
                    source_dataset="synthetic",
                    image_paths={"flair": "unused", "t1": "unused", "t2": "unused"},
                    pair_id=f"pair_{pair_index}",
                )
            )

    def loader(record, *, stage):
        return torch.full((3, 16, 16), float(record.label), dtype=torch.float32)

    dataset = ManifestDataset(records, loader=loader, stage="final")
    result = DomainClassifierRunner(
        TrainConfig(in_channels=3, max_epochs=1, batch_size=2, device="cpu", tiny=True)
    ).train_one_seed(dataset, dataset, dataset, seed=73)
    assert result["epochs_completed"] == 1
    assert len(result["test_predictions"]) == 4


def test_pair_label_permutation_is_whole_participant_not_slice_random() -> None:
    rows = _records()
    permuted = permute_pair_labels(rows, seed=73)
    original = {(row["pair_id"], row["participant_id"]): row["label"] for row in rows}
    updated = {(row["pair_id"], row["participant_id"]): row["label"] for row in permuted}
    for key, old_label in original.items():
        assert updated[key] in (old_label, 1 - old_label)
    for pair in {row["pair_id"] for row in rows}:
        for participant in {row["participant_id"] for row in rows if row["pair_id"] == pair}:
            values = [row["label"] for row in permuted if row["pair_id"] == pair and row["participant_id"] == participant]
            assert len(set(values)) == 1


def test_statistical_control_fits_scaler_on_train_rows_only() -> None:
    train = np.concatenate(
        [np.zeros((2, 3, 16, 16), dtype=np.float32), np.ones((2, 3, 16, 16), dtype=np.float32)],
        axis=0,
    )
    test = np.concatenate(
        [np.full((1, 3, 16, 16), -3.0, dtype=np.float32), np.full((1, 3, 16, 16), 3.0, dtype=np.float32)],
        axis=0,
    )
    train_features = extract_statistical_features(train)
    test_features = extract_statistical_features(test)
    fitted = fit_statistical_logistic(train_features, [0, 0, 1, 1], test_features, seed=73)
    assert np.allclose(fitted["train_feature_mean"], train_features.mean(axis=0))
    assert fitted["scores"].shape == (2,)
