from __future__ import annotations

import numpy as np

from andi_rewrite.domain_classifier.runner import (
    DomainClassifierRunner,
    ManifestDataset,
    TrainConfig,
    evaluate_negative_gate,
    evaluate_positive_gate,
    evaluate_tiny_gate,
    formal_expansion_plan,
    make_same_cohort_negative_records,
    materialize_registered_dataset,
    materialize_tensor_dataset,
    materialize_tensor_datasets,
    seed_matrix_gate,
    shuffle_participant_labels,
    subject_balanced_bce_with_logits,
)
from scripts.run_domain_classifier_controls import _powered_shuffle_datasets, _tiny_datasets


def _records(n: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        for slice_index in range(1 + (index % 3)):
            rows.append(
                {
                    "split": "train",
                    "label": 0,
                    "participant_id": f"healthy_{index}",
                    "pair_id": f"original_{index}",
                    "case_id": f"case_{index}_{slice_index}",
                    "z": slice_index,
                    "z_bin": slice_index,
                    "source_split": "source_train",
                }
            )
    return rows


def _result(seed: int, auc: float = 0.5, low: float = 0.4, high: float = 0.6) -> dict[str, object]:
    return {
        "seed": seed,
        "test": {"subject": {"roc_auc": auc}},
        "test_statistics": {
            "subject_bootstrap": {"ci_low": low, "ci_high": high},
            "heldout_pair_swap": {
                "p_value": 0.5,
                "two_sided_deviation": abs(auc - 0.5),
            },
        },
    }


def test_fixed_subject_normalizer_keeps_batch_denominator_global() -> None:
    import torch

    value = subject_balanced_bce_with_logits(
        torch.zeros(2),
        torch.tensor([0.0, 1.0]),
        ["p0", "p1"],
        subject_slice_counts={"p0": 1, "p1": 1, "p2": 1},
        subject_normalizer=3,
    )
    assert torch.allclose(value, torch.tensor(2.0 * np.log(2.0) / 3.0, dtype=value.dtype))


def test_registered_materializer_matches_path_reader(tmp_path) -> None:
    import nibabel as nib
    from andi_rewrite.data.domain_classifier.readers import load_slice

    paths: dict[str, str] = {}
    for channel, offset in zip(("flair", "t1", "t2"), (0.0, 10.0, 20.0)):
        values = np.empty((8, 8, 2), dtype=np.float32)
        values[..., 0] = offset + 1.0
        values[..., 1] = offset + 2.0
        path = tmp_path / f"{channel}.nii.gz"
        nib.save(nib.Nifti1Image(values, np.eye(4)), str(path))
        paths[channel] = str(path)
    rows = [
        {
            "split": "train",
            "label": 0,
            "participant_id": "healthy",
            "case_id": "case",
            "source_dataset": "synthetic",
            "source_split": "train",
            "source_key": "0",
            "z": z,
            "z_norm": float(z),
            "z_bin": 0 if z == 0 else 19,
            "registered_paths": paths,
        }
        for z in (0, 1)
    ]
    source = ManifestDataset(
        rows,
        loader=lambda record, *, stage: load_slice(record, stage=stage),
        stage="registered",
    )
    cached = materialize_registered_dataset(source)
    for index, row in enumerate(rows):
        expected = load_slice(row, stage="registered")
        actual = cached[index]["image"]
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=0.0)


def test_tensor_materializer_preserves_final_samples_and_metadata() -> None:
    rows = [
        {
            "image": np.full((3, 8, 8), float(index), dtype=np.float32),
            "label": index % 2,
            "participant_id": f"p{index}",
            "source_dataset": "synthetic",
            "source_split": "train",
            "source_key": str(index),
            "case_id": f"case{index}",
            "z": index,
            "pair_id": f"pair{index}",
        }
        for index in range(4)
    ]
    source = ManifestDataset(rows, stage="final")
    cached = materialize_tensor_dataset(source)
    assert cached.stage == "final"
    for index, row in enumerate(rows):
        sample = cached[index]
        assert int(sample["label"]) == int(row["label"])
        assert sample["participant_id"] == row["participant_id"]
        np.testing.assert_array_equal(sample["image"].numpy(), row["image"])


def test_training_label_shuffle_is_whole_participant() -> None:
    rows = _records()
    for row in rows:
        if str(row["participant_id"]).endswith(("4", "5", "6", "7")):
            row["label"] = 1
    shuffled = shuffle_participant_labels(rows, seed=73)
    by_participant: dict[str, set[int]] = {}
    for row in shuffled:
        by_participant.setdefault(str(row["participant_id"]), set()).add(int(row["label"]))
    assert all(len(labels) == 1 for labels in by_participant.values())
    assert sum(next(iter(labels)) for labels in by_participant.values()) == 4


def test_powered_shuffle_splits_pairs_and_preserves_true_test_labels() -> None:
    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
    original: dict[tuple[str, str], int] = {}
    original_source: dict[str, str] = {}
    for pair_index in range(20):
        source_split = ("train", "val", "test")[pair_index % 3]
        pair_id = f"pair_{pair_index}"
        original_source[pair_id] = source_split
        for label in (0, 1):
            participant = f"p{pair_index}_{label}"
            for z in (2, 3):
                row = {
                    "split": source_split,
                    "source_split": source_split,
                    "label": label,
                    "participant_id": participant,
                    "pair_id": pair_id,
                    "case_id": f"case_{pair_index}_{label}_{z}",
                    "z": z,
                    "z_bin": z,
                }
                rows_by_split[source_split].append(row)
                original[(pair_id, participant)] = label
    datasets = {split: ManifestDataset(rows) for split, rows in rows_by_split.items()}
    shuffled = _powered_shuffle_datasets(datasets, seed=73)
    assert {split: len(dataset.records) for split, dataset in shuffled.items()} == {
        "train": 32,
        "val": 8,
        "test": 40,
    }
    for split, dataset in shuffled.items():
        by_pair: dict[str, list[dict[str, object]]] = {}
        for row in dataset.records:
            by_pair.setdefault(str(row["pair_id"]), []).append(row)
            assert row["source_split"] == original_source[str(row["pair_id"])]
            assert row["metadata"]["test_labels_true"] is (split == "test")
        for pair_rows in by_pair.values():
            assert {int(row["label"]) for row in pair_rows} == {0, 1}
            if split == "test":
                for row in pair_rows:
                    key = (str(row["pair_id"]), str(row["participant_id"]))
                    assert int(row["label"]) == original[key]


def test_shared_tensor_cache_loads_rows_moved_between_split_templates() -> None:
    """Powered reassignment must retain exact tensors across source splits."""

    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
    expected: dict[tuple[str, str, str, str, str, int], np.ndarray] = {}
    for pair_index in range(12):
        source_split = ("train", "val", "test")[pair_index % 3]
        pair_id = f"pair_{pair_index}"
        for label in (0, 1):
            participant = f"p{pair_index}_{label}"
            case_id = f"case_{pair_index}_{label}"
            source_key = f"{pair_index}_{label}"
            image = np.full((3, 8, 8), float(pair_index * 2 + label), dtype=np.float32)
            row = {
                "split": source_split,
                "source_dataset": "synthetic",
                "source_split": source_split,
                "source_key": source_key,
                "label": label,
                "participant_id": participant,
                "pair_id": pair_id,
                "case_id": case_id,
                "z": 3,
                "z_bin": 3,
                "image": image,
            }
            rows_by_split[source_split].append(row)
            expected[("synthetic", source_split, source_key, participant, case_id, 3)] = image

    source = {split: ManifestDataset(rows) for split, rows in rows_by_split.items()}
    cached = materialize_tensor_datasets(source)
    shuffled = _powered_shuffle_datasets(cached, seed=73)
    moved = 0
    for target_split, dataset in shuffled.items():
        for index, row in enumerate(dataset.records):
            original_split = str(row["source_split"])
            if original_split != target_split:
                moved += 1
            key = (
                "synthetic",
                original_split,
                str(row["source_key"]),
                str(row["participant_id"]),
                str(row["case_id"]),
                int(row["z"]),
            )
            np.testing.assert_array_equal(dataset[index]["image"].numpy(), expected[key])
    assert moved > 0


def test_same_cohort_negative_control_pairs_balanced_groups() -> None:
    rows = make_same_cohort_negative_records(_records(), seed=73)
    assert len(rows) <= len(_records())
    labels_by_pair: dict[str, set[int]] = {}
    for row in rows:
        labels_by_pair.setdefault(str(row["pair_id"]), set()).add(int(row["label"]))
    assert labels_by_pair
    assert all(labels == {0, 1} for labels in labels_by_pair.values())
    assert all(row["domain"] == "same_cohort_negative" for row in rows)
    assert all(row["source_split"] == "source_train" for row in rows)
    for pair_id in labels_by_pair:
        bins_by_label = {
            label: sorted(row["z_bin"] for row in rows if row["pair_id"] == pair_id and row["label"] == label)
            for label in (0, 1)
        }
        assert bins_by_label[0] == bins_by_label[1]


def test_gates_do_not_call_wide_chance_ci_a_pass() -> None:
    result = evaluate_negative_gate([_result(73, low=0.10, high=0.90), _result(173, low=0.20, high=0.80)])
    assert result["status"] == "INCONCLUSIVE"


def test_tiny_and_positive_gate_thresholds_are_explicit() -> None:
    tiny = evaluate_tiny_gate(
        [{"seed": 73, "train_final": {"slice": {"accuracy": 0.995, "bce": 0.01}}}]
    )
    positive = evaluate_positive_gate([_result(73, auc=0.85, low=0.70)])
    assert tiny["status"] == "PASS"
    assert positive["status"] == "PASS"


def test_expansion_plan_and_seed_matrix_gate() -> None:
    plan = formal_expansion_plan()
    assert len(plan) == 112
    assert seed_matrix_gate(plan, expected_plan=plan)["status"] == "PASS"
    assert seed_matrix_gate(plan[:-1], expected_plan=plan)["status"] == "INCOMPLETE"


def test_synthetic_tiny_and_negative_one_epoch_have_complete_pairs() -> None:
    from andi_rewrite.domain_classifier.runner import make_same_cohort_negative_records

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "val", "test"):
        rows: list[dict[str, object]] = []
        for pair_index in range(10):
            for label in (0, 1):
                for z_bin in range(4):
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "participant_id": f"{split}_p{pair_index}_{label}",
                            "pair_id": f"{split}_pair{pair_index}",
                            "case_id": f"{split}_case{pair_index}_{label}_{z_bin}",
                            "z": z_bin,
                            "z_bin": z_bin,
                            "image": np.full((3, 16, 16), float(label), dtype=np.float32),
                        }
                    )
        rows_by_split[split] = rows
    datasets = {split: ManifestDataset(rows) for split, rows in rows_by_split.items()}
    tiny = _tiny_datasets(
        datasets,
        subjects_per_label=10,
        max_slices=128,
        min_slices=64,
        seed=73,
    )
    result = DomainClassifierRunner(
        TrainConfig(in_channels=3, max_epochs=1, batch_size=16, device="cpu", tiny=True, bootstrap_replicates=20, swap_replicates=20)
    ).train_one_seed(tiny["train"], tiny["val"], tiny["test"], seed=73)
    assert len(result["test_subject_predictions"]) == 20
    assert result["test_statistics"]["heldout_pair_swap"]["status"] == "complete"

    healthy_rows = {
        split: [row for row in rows if int(row["label"]) == 0]
        for split, rows in rows_by_split.items()
    }
    negative = {
        split: ManifestDataset(make_same_cohort_negative_records(rows, seed=73))
        for split, rows in healthy_rows.items()
    }
    negative_result = DomainClassifierRunner(
        TrainConfig(in_channels=3, max_epochs=1, batch_size=16, device="cpu", bootstrap_replicates=20, swap_replicates=20)
    ).train_one_seed(negative["train"], negative["val"], negative["test"], seed=73)
    assert negative_result["test_statistics"]["heldout_pair_swap"]["status"] == "complete"
