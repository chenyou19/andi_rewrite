from __future__ import annotations

import numpy as np

from andi_rewrite.domain_classifier.runner import ManifestDataset, TrainConfig, DomainClassifierRunner, materialize_tensor_datasets
from scripts.run_domain_classifier_calibration import _label_stream_seeds, permuted_datasets


def _paired_datasets() -> dict[str, ManifestDataset]:
    output: dict[str, ManifestDataset] = {}
    for split in ("train", "val", "test"):
        rows: list[dict[str, object]] = []
        for pair_index in range(20):
            for label in (0, 1):
                rows.append(
                    {
                        "split": split,
                        "source_split": split,
                        "source_dataset": "synthetic",
                        "source_key": f"{split}:{pair_index}:{label}",
                        "case_id": f"case_{split}_{pair_index}_{label}",
                        "z": 0,
                        "label": label,
                        "participant_id": f"{split}_participant_{pair_index}_{label}",
                        "pair_id": f"{split}_pair_{pair_index}",
                        "image": np.full((3, 8, 8), float(label + pair_index), dtype=np.float32),
                    }
                )
        output[split] = ManifestDataset(rows, stage="final")
    return output


def test_label_streams_are_deterministic_and_independent() -> None:
    first = _label_stream_seeds(0)
    second = _label_stream_seeds(1)
    assert first == _label_stream_seeds(0)
    assert len(set(first.values())) == 3
    assert first != second


def test_full_null_swaps_each_split_using_fixed_cached_images() -> None:
    source = _paired_datasets()
    cached = materialize_tensor_datasets(source)
    permuted, streams = permuted_datasets(cached, index=0, init_seed=73)
    assert len(streams) == 3
    assert set(streams) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        before = {(str(row["participant_id"]), int(row["label"])) for row in source[split].records}
        after = {(str(row["participant_id"]), int(row["label"])) for row in permuted[split].records}
        assert {str(row["pair_id"]) for row in permuted[split].records} == {
            f"{split}_pair_{index}" for index in range(20)
        }
        assert all(
            {int(row["label"]) for row in permuted[split].records if row["pair_id"] == pair} == {0, 1}
            for pair in {row["pair_id"] for row in permuted[split].records}
        )
        assert any(before != after for _ in [0])
        for index in range(len(permuted[split])):
            np.testing.assert_array_equal(permuted[split][index]["image"], cached[split][index]["image"])


def test_zero_null_statistics_are_omitted_without_affecting_fit() -> None:
    source = _paired_datasets()
    cached = materialize_tensor_datasets(source)
    cfg = TrainConfig(
        model="small_cnn",
        in_channels=3,
        max_epochs=1,
        patience=1,
        batch_size=16,
        device="cpu",
        bootstrap_replicates=0,
        swap_replicates=0,
    )
    result = DomainClassifierRunner(cfg).train_one_seed(
        cached["train"], cached["val"], cached["test"], seed=73
    )
    assert result["test_statistics"] == {}
    assert result["config"]["bootstrap_replicates"] == 0
    assert result["config"]["swap_replicates"] == 0
