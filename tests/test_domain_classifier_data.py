"""Focused Stage1 data-contract tests for the domain-classifier layer."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from andi_rewrite.data.domain_classifier import (
    DomainClassifierDataset,
    SliceRecord,
    audit_records,
    build_pairs,
    load_slice,
    participant_split_map,
    validate_tumor_free,
    write_records,
)
from andi_rewrite.data.robust_normalization import robust_normalize_volume
from andi_rewrite.data.robust_normalization import ROBUST_SPEC


def _record(
    participant: str,
    label: int,
    z: int,
    *,
    split: str = "train",
    case: str | None = None,
    image_paths: dict[str, str] | None = None,
    seg_path: str = "",
    model_mask_path: str = "",
) -> SliceRecord:
    case = case or f"{participant}/ses-01"
    z_norm = z / 19.0
    return SliceRecord(
        split=split,
        label=label,
        domain="healthy" if label == 0 else "brats21",
        participant_id=participant,
        session_id="ses-01",
        case_id=case,
        z=z,
        z_norm=z_norm,
        z_bin=min(19, int(z_norm * 20)),
        source_dataset="healthy" if label == 0 else "brats21",
        source_key=f"{participant}-{z}",
        source_split=split,
        image_paths=image_paths or {},
        seg_path=seg_path,
        model_mask_path=model_mask_path,
        geometry_shape=(4, 4, 20),
        model_shape=(3, 128, 128),
        native_seg_voxels=0 if label == 1 and seg_path else None,
    )


def test_participant_split_is_disjoint_and_reproducible() -> None:
    ids = [f"p-{index:02d}" for index in range(20)]
    first = participant_split_map(ids, seed=73)
    second = participant_split_map(reversed(ids), seed=73)
    assert first == second
    groups = [{participant for participant, split in first.items() if split == name} for name in ("train", "val", "test")]
    assert all(groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_pair_matching_has_one_primary_case_equal_histogram_and_cap() -> None:
    healthy = [_record("healthy:h", 0, z, case="healthy:h/primary") for z in (0, 1, 10, 11, 19)]
    healthy.extend(_record("healthy:h", 0, z, case="healthy:h/secondary") for z in (0, 1, 2, 3))
    brats = [_record("brats21:b", 1, z, case="brats21:b/primary") for z in (0, 2, 10, 12, 19)]
    result = build_pairs(healthy, brats, comparison="test", seed=73, cap_per_bin=2)
    assert result.records
    assert len(result.pairs) == 1
    rows = list(result.records)
    assert {row.case_id for row in rows if row.label == 0} == {"healthy:h/primary"}
    assert {row.case_id for row in rows if row.label == 1} == {"brats21:b/primary"}
    audit = audit_records(rows, require_pairs=True, cap_per_bin=2)
    assert audit["pair_count"] == 1
    assert len([row for row in rows if row.label == 0]) == len([row for row in rows if row.label == 1])


def test_path_reader_uses_full_volume_robust_iqr_and_exact_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(73)
    arrays = {}
    for index, modality in enumerate(("flair", "t1", "t2")):
        value = rng.uniform(0.1, 100.0, size=(4, 4, 2)).astype(np.float32)
        value[0, :, :] = 0.0
        path = tmp_path / f"{modality}.npy"
        np.save(path, value)
        arrays[modality] = (value, path)
    record = SliceRecord(
        split="train",
        label=0,
        domain="synthetic",
        participant_id="synthetic:p",
        session_id="ses-01",
        case_id="synthetic:p/ses-01",
        z=1,
        z_norm=1.0,
        z_bin=19,
        source_dataset="synthetic",
        source_key="slice-1",
        source_split="train",
        image_paths={key: str(value[1]) for key, value in arrays.items()},
        geometry_shape=(4, 4, 2),
        model_shape=(3, 128, 128),
    )
    actual = load_slice(record)
    expected_volume = robust_normalize_volume(torch.from_numpy(np.stack([arrays[m][0] for m in ("flair", "t1", "t2")], axis=0)))
    from torchvision.transforms import Resize

    expected = Resize(128, antialias=True)(expected_volume[..., 1])
    assert tuple(actual.shape) == (3, 128, 128)
    assert torch.equal(actual, expected)


def test_lmdb_final_reader_is_exact_and_dataset_returns_label(tmp_path: Path, request) -> None:
    import lmdb

    lmdb_path = tmp_path / "train"
    env = lmdb.open(str(lmdb_path), map_size=2 * 1024 * 1024)
    request.addfinalizer(env.close)
    expected = np.arange(3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)
    with env.begin(write=True) as txn:
        assert txn.put(b"00000000", pickle.dumps(expected, protocol=pickle.HIGHEST_PROTOCOL))
    (lmdb_path / "normalization.json").write_text(json.dumps(ROBUST_SPEC), encoding="utf-8")
    record = SliceRecord(
        split="train",
        label=0,
        domain="fomo45k",
        participant_id="fomo45k:p",
        session_id="ses-01",
        case_id="fomo45k:p/ses-01",
        z=0,
        z_norm=0.0,
        z_bin=0,
        source_dataset="fomo45k",
        source_key="00000000",
        source_split="train",
        geometry_shape=(240, 240, 155),
        model_shape=(3, 128, 128),
        metadata={"lmdb_path": str(lmdb_path)},
    )
    manifest = tmp_path / "manifest.jsonl"
    write_records(manifest, [record])
    dataset = DomainClassifierDataset.from_manifest(manifest, return_metadata=True)
    image, label, metadata = dataset[0]
    assert torch.equal(image, torch.from_numpy(expected))
    assert label == 0
    assert metadata["source_key"] == "00000000"


def test_tumor_free_requires_native_and_model_masks_to_be_zero(tmp_path: Path) -> None:
    native = np.zeros((4, 4, 2), dtype=np.int16)
    model = np.zeros((128, 128, 2), dtype=np.uint8)
    native_path = tmp_path / "seg.npy"
    model_path = tmp_path / "model_mask.npy"
    np.save(native_path, native)
    np.save(model_path, model)
    record = _record(
        "brats21:p",
        1,
        1,
        image_paths={},
        seg_path=str(native_path),
        model_mask_path=str(model_path),
    )
    result = validate_tumor_free(record)
    assert result["native_seg_voxels"] == 0
    assert result["model_mask_voxels"] == 0
    native[0, 0, 1] = 1
    np.save(native_path, native)
    with pytest.raises(ValueError, match="Native segmentation"):
        validate_tumor_free(record)
