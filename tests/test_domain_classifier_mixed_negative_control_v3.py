from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from andi_rewrite.domain_classifier.runner import ManifestDataset, materialize_tensor_datasets
from scripts.run_domain_classifier_mixed_negative_control_v3 import (
    SOURCES,
    build_source_stratified_negative_datasets,
    build_source_stratified_negative_records,
)


def _mixed_rows(participants_per_source: int = 21) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        for participant_index in range(participants_per_source):
            participant = f"{source}:subject_{participant_index:03d}"
            # Keep a source participant together in the source manifests while
            # exercising all three original LMDB split values.
            source_split = ("train", "val", "test")[participant_index % 3]
            for z_bin in (2, 5, 8):
                rows.append(
                    {
                        "split": source_split,
                        "source_split": source_split,
                        "source_dataset": "mixed",
                        "source_key": f"{source}_{participant_index}_{z_bin}",
                        "case_id": f"case_{participant_index}",
                        "z": z_bin * 7,
                        "z_norm": z_bin / 20.0,
                        "z_bin": z_bin,
                        "label": 0,
                        "domain": "mixed",
                        "participant_id": participant,
                        "pair_id": f"original:{source}:{participant_index}",
                        "metadata": {
                            "underlying_source_dataset": source,
                            "underlying_source_split": source_split,
                            "underlying_source_key": f"{source}_{participant_index}_{z_bin}",
                        },
                        "image": np.full(
                            (3, 8, 8),
                            float(SOURCES.index(source) + participant_index / 100.0 + z_bin),
                            dtype=np.float32,
                        ),
                    }
                )
    return rows


def test_mixed_negative_is_source_stratified_and_keeps_source_split() -> None:
    rows, audit = build_source_stratified_negative_records(
        _mixed_rows(), split_seed=73, pseudo_seed=173
    )
    assert rows
    assert audit["control_type"] == "mixed_source_stratified"
    assert audit["pair_source_crossings"] == 0
    assert audit["pair_split_crossings"] == 0
    assert set(audit["source_assignment"]) == set(SOURCES)
    for source in SOURCES:
        source_audit = audit["source_assignment"][source]
        # 21 participants become 8/2/10 target participants; the odd source
        # participant is explicitly recorded rather than moved to another split.
        assert source_audit["target_split_counts"] == {"train": 8, "val": 2, "test": 10}
        assert len(source_audit["split_assignment_omissions"]) == 1
        for split in ("train", "val", "test"):
            split_audit = source_audit["splits"][split]
            assert split_audit["pseudo_label_participants"]["0"] == split_audit["pseudo_label_participants"]["1"]
            assert split_audit["output_slice_counts"]["0"] == split_audit["output_slice_counts"]["1"]
    for row in rows:
        assert row["source_dataset"] == "mixed"
        assert row["source_split"] in {"train", "val", "test"}
        assert row["metadata"]["source_split_immutable"] == row["source_split"]
        assert str(row["pair_id"]).startswith("negative:mixed:")
    by_pair: dict[str, set[tuple[str, int, int]]] = {}
    for row in rows:
        pair = str(row["pair_id"])
        source = str(row["metadata"]["underlying_source_dataset"])
        by_pair.setdefault(pair, set()).add((source, int(row["label"]), int(row["z_bin"])))
    for pair, values in by_pair.items():
        assert len({value[0] for value in values}) == 1, pair
        assert {value[1] for value in values} == {0, 1}, pair


def test_mixed_negative_pseudo_seed_changes_labels_but_not_source_split_assignment() -> None:
    rows = _mixed_rows()
    first, audit_first = build_source_stratified_negative_records(
        rows, split_seed=73, pseudo_seed=73
    )
    second, audit_second = build_source_stratified_negative_records(
        rows, split_seed=73, pseudo_seed=273
    )
    first_assignment = {
        (str(row["participant_id"]), str(row["split"]))
        for row in first
    }
    second_assignment = {
        (str(row["participant_id"]), str(row["split"]))
        for row in second
    }
    assert first_assignment == second_assignment
    assert audit_first["split_seed"] == audit_second["split_seed"] == 73
    assert audit_first["pseudo_seed"] == 73
    assert audit_second["pseudo_seed"] == 273
    assert any(
        (str(a["participant_id"]), int(a["label"]))
        != (str(b["participant_id"]), int(b["label"]))
        for a, b in zip(first, second)
    )


def test_mixed_negative_dataset_transform_reuses_exact_tensor_cache() -> None:
    rows = _mixed_rows(participants_per_source=20)
    by_split = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        by_split[str(row["split"])].append(row)
    original = {
        split: ManifestDataset(split_rows, modalities=("flair", "t1", "t2"))
        for split, split_rows in by_split.items()
    }
    cached = materialize_tensor_datasets(original)
    transformed, audit = build_source_stratified_negative_datasets(
        cached, split_seed=73, pseudo_seed=73
    )
    assert audit["output_rows"] == sum(len(dataset.records) for dataset in transformed.values())
    assert all(dataset.loader is cached["train"].loader for dataset in transformed.values())
    for split, dataset in transformed.items():
        for index, record in enumerate(dataset.records):
            sample = dataset[index]
            assert tuple(sample["image"].shape) == (3, 8, 8)
            # The output row's target split can differ from the immutable source
            # split, so the cache key must be source-key based rather than target
            # split based.
            assert str(record["split"]) == split
            assert int(sample["label"]) in (0, 1)


def test_mixed_negative_rejects_nonmixed_or_nonzero_input() -> None:
    rows = _mixed_rows(participants_per_source=20)
    rows[0]["source_dataset"] = "fomo45k"
    import pytest

    with pytest.raises(ValueError, match="source_dataset='mixed'"):
        build_source_stratified_negative_records(rows)
    rows = _mixed_rows(participants_per_source=20)
    rows[0]["label"] = 2
    with pytest.raises(ValueError, match="finite binary"):
        build_source_stratified_negative_records(rows)
