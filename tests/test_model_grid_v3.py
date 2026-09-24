from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from andi_rewrite.data.domain_classifier.records import SliceRecord
from scripts.build_model_grid_v3 import (
    ATLAS_DEPTH,
    MODEL_SHAPE,
    SUPPORT_EPS,
    SUPPORT_THRESHOLD,
    check_brats_zero,
    check_expected_participant_map,
    check_map_reuse,
    check_source_joins,
    check_support,
    check_z_norm,
    _candidate_pool_overlap,
    validate_brats_csv_map,
    read_mixed_rows,
    source_lmdb_path,
    support_fraction,
)


def _record(**updates: object) -> SliceRecord:
    payload = dict(
        split="test",
        label=0,
        domain="fomo45k",
        participant_id="fomo45k:healthy-1",
        session_id="ses-1",
        case_id="healthy-1/ses-1",
        z=145,
        z_norm=145 / 154,
        z_bin=18,
        source_dataset="fomo45k",
        source_key="00000001",
        source_split="train",
        model_shape=MODEL_SHAPE,
        geometry_shape=(240, 240, 155),
        stage="final",
        foreground_fraction=(0.2, 0.2, 0.2),
        metadata={
            "source_participant_id": "healthy-1",
            "model_grid_support_fraction": [0.2, 0.2, 0.2],
        },
    )
    payload.update(updates)
    return SliceRecord(**payload)


def test_support_fraction_is_exact_three_channel_abs_background_rule() -> None:
    tensor = torch.full(MODEL_SHAPE, -1.0)
    tensor[:, :13, :] = -1.0 + 2.0 * SUPPORT_EPS
    values = support_fraction(tensor)
    assert values == pytest.approx((13 / 128, 13 / 128, 13 / 128))
    assert all(value >= SUPPORT_THRESHOLD for value in values)
    below = tensor.clone()
    below[:, :12, :] = -1.0 + SUPPORT_EPS / 2
    assert all(value < SUPPORT_THRESHOLD for value in support_fraction(below))


def test_z_norm_uses_full_registered_depth_not_max_available_slice() -> None:
    row = _record(z=145, z_norm=145 / (ATLAS_DEPTH - 1), z_bin=18)
    audit = check_z_norm([row])
    assert audit["status"] == "PASS"
    assert audit["denominator_depth"] == 155
    assert row.z_norm != pytest.approx(1.0)


def test_mixed_split_local_key_join_checks_underlying_metadata() -> None:
    record = _record(
        domain="mixed",
        source_dataset="mixed",
        participant_id="fomo45k:healthy-1",
        source_key="00000007",
        source_split="train",
        metadata={
            "source_participant_id": "healthy-1",
            "model_grid_support_fraction": [0.2, 0.2, 0.2],
        },
    )
    source_rows = {
        "fomo45k": [
            {
                "source_split": "train",
                "source_key": "00000001",
                "participant_id": "healthy-1",
                "case_id": "healthy-1/ses-1",
                "session_id": "ses-1",
                "z": 145,
            }
        ],
        "mpi": [],
        "oasis3": [],
        "mixed": [
            {
                "source_split": "train",
                "source_key": "00000001",
                "mixed_local_key": "00000007",
                "underlying_source_dataset": "fomo45k",
            }
        ],
    }
    audit = check_source_joins([record], "mixed", source_rows)
    assert audit["status"] == "PASS"
    bad = _record(
        domain="mixed",
        source_dataset="mixed",
        source_key="00000007",
        metadata={"source_participant_id": "other", "model_grid_support_fraction": [0.2, 0.2, 0.2]},
    )
    assert check_source_joins([bad], "mixed", source_rows)["status"] == "FAIL"


def test_read_mixed_rows_retains_source_dataset_and_split_local_keys(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    (root / "validation").mkdir(parents=True)
    (root / "source_entries.jsonl").write_text(json.dumps({"key": "00000000", "source_dataset": "mpi", "source_key": "00000003", "source_split": "train"}) + "\n", encoding="utf-8")
    (root / "validation" / "source_entries.jsonl").write_text(json.dumps({"key": "00000000", "source_dataset": "oasis3", "source_key": "00000004", "source_split": "val"}) + "\n", encoding="utf-8")
    rows = read_mixed_rows(root)
    assert rows[0]["source_dataset"] == "mpi"
    assert rows[0]["underlying_source_dataset"] == "mpi"
    assert rows[0]["source_key"] == "00000003"
    assert rows[0]["mixed_local_key"] == "00000000"
    assert rows[1]["source_split"] == "val"
    assert source_lmdb_path("mixed", "val").name == "val"


def test_seed73_map_reuse_is_checked_for_fomo_and_brats() -> None:
    healthy = _record(participant_id="fomo45k:healthy-1")
    brats = _record(
        label=1,
        domain="brats21",
        participant_id="brats21:BraTS2021_00001",
        source_dataset="brats21",
        native_seg_voxels=0,
        model_mask_voxels=0,
        seg_path="C:/seg.nii.gz",
        metadata={"model_grid_support_fraction": [0.2, 0.2, 0.2], "native_segmentation_checked": True, "model_mask_checked": True},
    )
    maps = {"fomo45k": {"fomo45k:healthy-1": "test"}, "brats21": {"brats21:BraTS2021_00001": "test"}}
    assert check_map_reuse([healthy, brats], "fomo45k", maps)["status"] == "PASS"
    changed = _record(participant_id="fomo45k:healthy-1", split="train")
    assert check_map_reuse([changed, brats], "fomo45k", maps)["status"] == "FAIL"


def test_brats_zero_gate_requires_explicit_native_and_model_checks() -> None:
    good = _record(
        label=1,
        domain="brats21",
        participant_id="brats21:BraTS2021_00001",
        source_dataset="brats21",
        native_seg_voxels=0,
        model_mask_voxels=0,
        seg_path=__file__,
        metadata={"model_grid_support_fraction": [0.2, 0.2, 0.2], "native_segmentation_checked": True, "model_mask_checked": True},
    )
    assert check_brats_zero([good])["status"] == "PASS"
    missing_model_check = good.with_updates(metadata={"model_grid_support_fraction": [0.2, 0.2, 0.2], "native_segmentation_checked": True})
    assert check_brats_zero([missing_model_check])["status"] == "FAIL"


def test_support_audit_fails_any_channel_below_threshold() -> None:
    row = _record(metadata={"model_grid_support_fraction": [0.2, 0.099, 0.2]})
    audit = check_support([row])
    assert audit["status"] == "FAIL"
    assert audit["expression"] == f"abs(x + 1) > {SUPPORT_EPS:g}"


def test_full_candidate_identity_accepts_csv_map_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "scans_train.csv"
    csv_path.write_text("BraTS21ID\nS1\nS2\n", encoding="utf-8")
    audit = validate_brats_csv_map(csv_path, {"brats21:S1": "train", "brats21:S2": "test"})
    assert audit["csv_rows"] == 2
    assert audit["id_set_equal"] is True
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("BraTS21ID\nS1\nS1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity validation failed"):
        validate_brats_csv_map(duplicate, {"brats21:S1": "train", "brats21:S2": "test"})


def test_full_candidate_pool_audit_records_rows_outside_legacy_selected_pool() -> None:
    legacy = [{"participant_id": "brats21:S1", "z": 10}]
    full = [
        {"participant_id": "brats21:S1", "z": 10},
        {"participant_id": "brats21:S1", "z": 11},
        {"participant_id": "brats21:S2", "z": 10},
    ]
    audit = _candidate_pool_overlap(full, legacy)
    assert audit["candidate_keys"] == 3
    assert audit["overlap_keys"] == 1
    assert audit["candidate_keys_outside_legacy"] == 2


def test_mixed_map_uses_standalone_union_and_allows_unselected_participants() -> None:
    rows = [
        _record(domain="mixed", source_dataset="mixed", participant_id="mpi:p1", split="train"),
        _record(domain="mixed", source_dataset="mixed", participant_id="oasis3:p2", split="test"),
    ]
    expected = {
        "fomo45k:p0": "val",
        "mpi:p1": "train",
        "oasis3:p2": "test",
    }
    assert check_expected_participant_map(rows, expected)["status"] == "PASS"
    wrong = rows[1].with_updates(split="train")
    assert check_expected_participant_map([rows[0], wrong], expected)["status"] == "FAIL"
    extra = rows[0].with_updates(participant_id="unknown:p3")
    assert check_expected_participant_map([extra], expected)["status"] == "FAIL"
