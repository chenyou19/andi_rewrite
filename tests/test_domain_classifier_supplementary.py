from __future__ import annotations

import numpy as np

from andi_rewrite.domain_classifier.runner import ManifestDataset, materialize_tensor_datasets
from scripts.run_domain_classifier_supplementary import _balanced_label_rows, _pair_counts


def _key(row: dict[str, object]) -> tuple[str, str, str, str, str, int]:
    return (
        str(row["source_dataset"]),
        str(row["source_split"]),
        str(row["source_key"]),
        str(row["participant_id"]),
        str(row["case_id"]),
        int(row["z"]),
    )


def _synthetic_source() -> tuple[dict[str, ManifestDataset], dict[tuple[str, str, str, str, str, int], int]]:
    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
    true_labels: dict[tuple[str, str, str, str, str, int], int] = {}
    for pair_index in range(20):
        source_split = ("train", "val", "test")[pair_index % 3]
        pair_id = f"pair_{pair_index}"
        for label in (0, 1):
            participant = f"p{pair_index}_{label}"
            row = {
                "split": source_split,
                "source_dataset": "synthetic",
                "source_split": source_split,
                "source_key": f"{pair_index}_{label}",
                "label": label,
                "domain": "synthetic",
                "participant_id": participant,
                "pair_id": pair_id,
                "case_id": f"case_{pair_index}_{label}",
                "z": 3,
                "z_bin": 3,
                "image": np.full((3, 8, 8), float(pair_index * 2 + label), dtype=np.float32),
            }
            rows_by_split[source_split].append(row)
            true_labels[_key(row)] = label
    source = {split: ManifestDataset(rows) for split, rows in rows_by_split.items()}
    return materialize_tensor_datasets(source), true_labels


def test_balanced_protocol_freezes_split_keys_and_exact_flip_counts() -> None:
    source, true_labels = _synthetic_source()
    outputs = [
        _balanced_label_rows(source, seed=seed, split_seed=73, true_labels=true_labels)
        for seed in (73, 173, 273)
    ]
    first_keys = {
        split: [_key(row) for row in outputs[0][split].records]
        for split in ("train", "val", "test")
    }
    first_pairs = {
        split: [str(row["pair_id"]) for row in outputs[0][split].records]
        for split in ("train", "val", "test")
    }
    for datasets in outputs:
        assert {split: [_key(row) for row in datasets[split].records] for split in first_keys} == first_keys
        assert {split: [str(row["pair_id"]) for row in datasets[split].records] for split in first_pairs} == first_pairs
        counts = _pair_counts(datasets)
        assert counts["train"]["pairs"] == 8
        assert counts["train"]["flipped_pairs"] == 4
        assert counts["val"]["pairs"] == 2
        assert counts["val"]["flipped_pairs"] == 1
        assert counts["test"]["pairs"] == 10
        assert counts["test"]["flipped_pairs"] == 0


def test_balanced_protocol_preserves_true_test_labels_and_changes_only_train_val() -> None:
    source, true_labels = _synthetic_source()
    datasets = _balanced_label_rows(source, seed=273, split_seed=73, true_labels=true_labels)
    for split in ("train", "val", "test"):
        for row in datasets[split].records:
            true = true_labels[_key(row)]
            if split == "test":
                assert int(row["label"]) == true
            else:
                assert int(row["label"]) in (true, 1 - true)
            assert str(row["source_split"]) in {"train", "val", "test"}
