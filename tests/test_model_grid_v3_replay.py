from __future__ import annotations

import tempfile
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.replay_model_grid_v3 import (
    SEMANTIC_FIELDS,
    compare_candidate_ledger,
    compare_candidate_rows,
    compare_semantic_rows,
    tensor_sha256,
    verify_healthy_tensor_hash_ledger,
    verify_healthy_source_tensors,
    verify_source_fingerprint_continuity,
    verify_selected_brats_tensor_hashes,
)
from scripts.replay_model_grid_v3 import _load_optional_healthy_hash_ledger, _new_output_root
import scripts.replay_model_grid_v3 as replay_module


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_dataset": "fomo45k",
        "source_split": "train",
        "source_key": "0001",
        "participant_id": "fomo45k:h1",
        "case_id": "case-1",
        "session_id": "ses-1",
        "split": "train",
        "pair_id": "fomo45k:train:p1",
        "z": 20,
        "z_bin": 2,
        "label": 0,
    }
    row.update(updates)
    return row


def test_semantic_manifest_comparison_is_multiset_and_field_explicit() -> None:
    expected = [_row(), _row(source_key="0002", z=21, z_bin=2)]
    assert compare_semantic_rows(expected, expected, comparison="fomo45k")["status"] == "PASS"

    changed = [dict(expected[0]), dict(expected[1], pair_id="fomo45k:train:changed")]
    audit = compare_semantic_rows(expected, changed, comparison="fomo45k")
    assert audit["status"] == "FAIL"
    assert audit["missing_count"] == 1
    assert audit["extra_count"] == 1
    assert "pair_id" in audit["semantic_fields"]
    assert set(SEMANTIC_FIELDS) == set(audit["semantic_fields"])


def _candidate_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "participant_id": "brats21:S1",
        "source_participant_id": "S1",
        "case_id": "S1",
        "session_id": "S1",
        "z": 7,
        "z_bin": 1,
        "z_norm": 7 / 154,
        "source_dataset": "brats21",
        "source_key": "S1:7",
        "source_split": "train",
        "logical_split": "test",
        "native_seg_voxels": 0,
        "model_mask_voxels": 0,
        "_support_fraction": [0.25, 0.25, 0.25],
        "metadata": {
            "native_segmentation_checked": True,
            "model_mask_checked": True,
            "full_candidate_inventory": True,
            "model_grid_support_fraction": [0.25, 0.25, 0.25],
            "model_grid_support_threshold": 0.1,
        },
        "provenance": {"brats_csv_sha256": "csv", "build_protocol": "model_grid_v3_fullcandidate_v1"},
    }
    row.update(updates)
    return row


def test_candidate_replay_rejects_same_count_with_different_identity_or_support() -> None:
    expected = [_candidate_row()]
    changed_identity = [_candidate_row(source_key="S1:8", z=8, z_bin=2)]
    identity_audit = compare_candidate_rows(expected, changed_identity)
    assert identity_audit["status"] == "FAIL"
    assert identity_audit["missing_identity_count"] == 1
    assert identity_audit["extra_identity_count"] == 1

    changed_support = [_candidate_row(_support_fraction=[0.35, 0.25, 0.25])]
    support_audit = compare_candidate_rows(expected, changed_support)
    assert support_audit["status"] == "FAIL"
    assert support_audit["missing_full_row_count"] == 1
    assert support_audit["extra_full_row_count"] == 1
    assert support_audit["full_row_fields_compared"] == "all"


def test_candidate_completion_ledger_compares_full_subject_audit() -> None:
    expected = [{"subject": "S1", "status": "COMPLETE", "resume_fingerprint": "old", "audit": {"support_pass_slices": 4, "support_fail_slices": 2}}]
    same = [{"subject": "S1", "status": "COMPLETE", "resume_fingerprint": "new", "audit": {"support_pass_slices": 4, "support_fail_slices": 2}}]
    assert compare_candidate_ledger(expected, same)["status"] == "PASS"
    changed = [{"subject": "S1", "status": "COMPLETE", "resume_fingerprint": "new", "audit": {"support_pass_slices": 3, "support_fail_slices": 3}}]
    assert compare_candidate_ledger(expected, changed)["status"] == "FAIL"


def test_selected_brats_hashes_use_one_canonical_volume_pass() -> None:
    volume = np.zeros((3, 128, 128, 155), dtype=np.float32)
    expected_hash = tensor_sha256(volume[..., 7])
    rows = [
        {
            "label": 1,
            "participant_id": "brats21:S1",
            "z": 7,
            "metadata": {"source_participant_id": "S1"},
            "provenance": {"model_input_sha256": expected_hash},
        },
        # The same participant/z appears in another comparison; it must be
        # checked once in the union, while both cached hashes are required.
        {
            "label": 1,
            "participant_id": "brats21:S1",
            "z": 7,
            "metadata": {"source_participant_id": "S1"},
            "provenance": {"model_input_sha256": expected_hash},
        },
    ]
    calls: list[str] = []

    def reader(subject: str):
        calls.append(subject)
        return volume, np.zeros((128, 128, 155), dtype=np.uint8)

    audit = verify_selected_brats_tensor_hashes(rows, reader)
    assert audit["status"] == "PASS"
    assert audit["unique_subject_z"] == 1
    assert audit["checked_subject_z"] == 1
    assert calls == ["S1"]

    bad = [dict(rows[0], provenance={"model_input_sha256": "bad"})]
    assert verify_selected_brats_tensor_hashes(bad, reader)["status"] == "FAIL"


def test_selected_brats_hashes_fail_closed_when_existing_cache_hash_is_missing() -> None:
    row = {
        "label": 1,
        "participant_id": "brats21:S1",
        "z": 7,
        "metadata": {"source_participant_id": "S1"},
        "provenance": {},
    }
    audit = verify_selected_brats_tensor_hashes(
        [row],
        lambda _subject: (
            np.zeros((3, 128, 128, 155), dtype=np.float32),
            np.zeros((128, 128, 155), dtype=np.uint8),
        ),
    )
    assert audit["status"] == "FAIL"
    assert audit["missing_hash_count"] == 1
    assert audit["checked_subject_z"] == 0


def test_selected_brats_hashes_reject_conflicting_expected_hashes() -> None:
    volume = np.zeros((3, 128, 128, 155), dtype=np.float32)
    good = tensor_sha256(volume[..., 7])
    row = {
        "label": 1,
        "participant_id": "brats21:S1",
        "z": 7,
        "metadata": {"source_participant_id": "S1"},
    }
    rows = [
        dict(row, provenance={"model_input_sha256": good}),
        dict(row, provenance={"model_input_sha256": "different"}),
    ]
    audit = verify_selected_brats_tensor_hashes(
        rows,
        lambda _subject: (volume, np.zeros((128, 128, 155), dtype=np.uint8)),
    )
    assert audit["status"] == "FAIL"
    assert audit["conflicting_expected_hash_count"] == 1
    assert any(item["reason"] == "conflicting_expected_model_input_sha256" for item in audit["mismatch_examples"])


def test_selected_brats_hashes_fail_closed_when_union_is_empty() -> None:
    audit = verify_selected_brats_tensor_hashes([], lambda _subject: None)
    assert audit["status"] == "FAIL"
    assert audit["unique_subject_z"] == 0


def test_replay_output_must_be_fresh_and_cannot_equal_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source"
        source.mkdir()
        with pytest.raises(ValueError, match="cannot be overwritten"):
            _new_output_root(source, source)
        occupied = Path(directory) / "occupied"
        occupied.mkdir()
        (occupied / "audit.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError, match="new and empty"):
            _new_output_root(source, occupied)


class _Reader:
    def __init__(self, values: dict[int, np.ndarray]) -> None:
        self.values = values

    def __getitem__(self, key: int) -> np.ndarray:
        return self.values[int(key)]


def test_healthy_and_mixed_source_hash_audit_checks_exact_concat() -> None:
    tensor = np.arange(3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)
    rows = {
        "fomo45k": [_row()],
        "mpi": [],
        "oasis3": [],
        "mixed": [
            _row(
                source_dataset="mixed",
                source_split="train",
                source_key="9",
                participant_id="mpi:m1",
                metadata={
                    "underlying_source_dataset": "mpi",
                    "underlying_source_split": "train",
                    "underlying_source_key": "4",
                },
            )
        ],
    }
    values = {
        ("fomo45k", "train"): _Reader({1: tensor}),
        ("mpi", "train"): _Reader({4: tensor}),
        ("mixed", "train"): _Reader({9: tensor.copy()}),
    }
    audit = verify_healthy_source_tensors(rows, reader_factory=lambda dataset, split: values[(dataset, split)])
    assert audit["status"] == "PASS"
    assert audit["checked_healthy_rows"] == 2
    assert audit["mixed_rows_checked"] == 1

    values[("mixed", "train")] = _Reader({9: tensor + 1})
    failed = verify_healthy_source_tensors(rows, reader_factory=lambda dataset, split: values[(dataset, split)])
    assert failed["status"] == "FAIL"
    assert any(item["reason"] == "mixed_tensor_not_exact_source" for item in failed["failure_examples"])


def test_source_fingerprint_continuity_rejects_changed_sidecar_and_reports_missing_history() -> None:
    with tempfile.TemporaryDirectory() as directory:
        sidecar = Path(directory) / "entries.jsonl"
        sidecar.write_text('{"key":"1"}\n', encoding="utf-8")
        digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        row = _row(
            metadata={"source_sidecar_sha256": digest},
            provenance={"source_sidecar": str(sidecar)},
        )
        rows = {"fomo45k": [row], "mpi": [], "oasis3": [], "mixed": []}
        assert verify_source_fingerprint_continuity(rows, {"sidecars": {"fomo45k:train": digest}})["status"] == "PASS"
        sidecar.write_text('{"key":"changed"}\n', encoding="utf-8")
        changed = verify_source_fingerprint_continuity(rows, {"sidecars": {"fomo45k:train": digest}})
        assert changed["status"] == "FAIL"
        assert any(item["reason"] == "source_sidecar_changed_since_historical_fingerprint" for item in changed["failure_examples"])
        missing = verify_source_fingerprint_continuity(rows, {})
        assert missing["status"] == "INCONCLUSIVE_NO_HISTORICAL_FINGERPRINT"


def test_healthy_tensor_hash_ledger_accepts_writer_canonical_rows_and_rejects_changed_hash() -> None:
    tensor = np.zeros((3, 128, 128), dtype=np.float32)
    digest = tensor_sha256(tensor)
    row = _row(
        source_dataset="fomo45k",
        source_key="0001",
        provenance={"model_input_sha256": digest},
    )
    ledger = [{
        "canonical_source_dataset": "fomo45k",
        "canonical_source_split": "train",
        "canonical_source_key": "0001",
        "z": 20,
        "tensor_sha256": digest,
        "memberships": [{"comparison": "fomo45k", "local_source_key": "0001"}],
    }]
    assert verify_healthy_tensor_hash_ledger({"fomo45k": [row], "mpi": [], "oasis3": [], "mixed": []}, ledger)["status"] == "PASS"
    bad = [dict(ledger[0], tensor_sha256="bad")]
    assert verify_healthy_tensor_hash_ledger({"fomo45k": [row], "mpi": [], "oasis3": [], "mixed": []}, bad)["status"] == "FAIL"
    source_reader = _Reader({1: tensor})
    source_audit = verify_healthy_source_tensors(
        {"fomo45k": [row], "mpi": [], "oasis3": [], "mixed": []},
        reader_factory=lambda _dataset, _split: source_reader,
        tensor_hash_ledger=ledger,
    )
    assert source_audit["status"] == "PASS"
    assert source_audit["tensor_hash_ledger_status"] == "PASS"


def test_healthy_hash_ledger_loader_binds_summary_status_path_and_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binding = root / "observed_input_binding_v1"
        binding.mkdir()
        ledger = binding / "selected_healthy_source_tensor_ledger.jsonl"
        ledger.write_text('{"canonical_source_dataset":"fomo45k","z":1}\n', encoding="utf-8")
        digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
        summary = {
            "status": "PASS",
            "ledger_path": str(ledger.resolve()),
            "ledger_sha256": digest,
        }
        (binding / "selected_healthy_source_tensor_ledger_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        entries, info = _load_optional_healthy_hash_ledger(root)
        assert len(entries) == 1
        assert info["status"] == "PASS"
        assert info["ledger_sha256_match"] is True
        assert info["actual_ledger_sha256"] == digest
        assert info["summary_status"] == "PASS"

        ledger.write_text('{"canonical_source_dataset":"fomo45k","z":2}\n', encoding="utf-8")
        _entries, changed = _load_optional_healthy_hash_ledger(root)
        assert changed["status"] == "FAIL_LEDGER_DIGEST_MISMATCH"
        assert changed["ledger_sha256_match"] is False

        summary["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        summary["status"] = "RUNNING"
        (binding / "selected_healthy_source_tensor_ledger_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        _entries, non_pass = _load_optional_healthy_hash_ledger(root)
        assert non_pass["status"] == "FAIL_LEDGER_SUMMARY_NOT_PASS"

        summary.pop("ledger_sha256")
        summary["status"] = "PASS"
        (binding / "selected_healthy_source_tensor_ledger_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        _entries, missing_digest = _load_optional_healthy_hash_ledger(root)
        assert missing_digest["status"] == "INCONCLUSIVE_LEDGER_DIGEST_MISSING"


def test_run_replay_fixture_wires_mapping_and_fail_closed_ledger_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the complete audit wiring without opening MRI/LMDB data."""

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        source = workspace / "frozen"
        for comparison in replay_module.COMPARISONS:
            for split in ("train", "val", "test"):
                path = source / "manifests" / comparison / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
        (source / "build_summary.json").write_text("{}", encoding="utf-8")
        (source / "source_fingerprints.json").write_text("{}", encoding="utf-8")
        binding = source / "observed_input_binding_v1"
        binding.mkdir()
        ledger = binding / "selected_healthy_source_tensor_ledger.jsonl"
        ledger.write_text('{"canonical_source_dataset":"fomo45k","z":1}\n', encoding="utf-8")
        ledger_digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
        (binding / "selected_healthy_source_tensor_ledger_summary.json").write_text(
            json.dumps({"status": "PASS", "ledger_path": str(ledger.resolve()), "ledger_sha256": ledger_digest}),
            encoding="utf-8",
        )
        candidate = {
            "participant_id": "brats21:S1",
            "source_participant_id": "S1",
            "case_id": "S1",
            "source_key": "S1:1",
            "z": 1,
            "native_seg_voxels": 0,
            "model_mask_voxels": 0,
            "_support_fraction": [0.2, 0.2, 0.2],
            "metadata": {"full_candidate_inventory": True},
        }
        completion = {"subject": "S1", "status": "COMPLETE", "audit": {"eligible": 1}}
        (source / "brats_candidate_rows.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        (source / "brats_candidate_completion.jsonl").write_text(json.dumps(completion) + "\n", encoding="utf-8")

        monkeypatch.setattr(replay_module, "EXPECTED_MANIFEST_TOTAL", 0)
        monkeypatch.setattr(replay_module, "EXPECTED_CANDIDATE_ROWS", 1)
        monkeypatch.setattr(replay_module, "EXPECTED_CANDIDATE_SUBJECTS", 1)
        monkeypatch.setattr(replay_module, "EXPECTED_SOURCE_CSV_ROWS", 1)
        monkeypatch.setattr(replay_module, "read_old_split_maps", lambda: {name: {} for name in (*replay_module.COMPARISONS, "brats21")})
        monkeypatch.setattr(replay_module, "canonical_source_rows", lambda: {name: [] for name in replay_module.COMPARISONS})
        monkeypatch.setattr(replay_module, "_source_hashes", lambda _rows: {})
        monkeypatch.setattr(replay_module, "make_healthy_records", lambda *args, **kwargs: ([], []))
        monkeypatch.setattr(replay_module, "make_brats_records", lambda *args, **kwargs: ([], []))
        monkeypatch.setattr(replay_module, "build_pairs", lambda *args, **kwargs: SimpleNamespace(records=[]))
        monkeypatch.setattr(replay_module, "records_by_split", lambda _records: {split: [] for split in ("train", "val", "test")})
        monkeypatch.setattr(replay_module, "_make_mixed_healthy_records", lambda *args, **kwargs: [])

        def fake_candidate_loader(*args: object, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
            output_root = Path(str(args[1]))
            replay_module._write_jsonl(output_root / "brats_candidate_rows.jsonl", [candidate])
            replay_module._write_jsonl(output_root / "brats_candidate_completion.jsonl", [completion])
            return [candidate], {
                "status": "PASS",
                "full_candidate_coverage": {"verified": True},
                "model_grid_counts": {"candidate_rows": 1, "subjects": 1},
                "source": {"csv_rows_verified": 1},
            }

        monkeypatch.setattr(replay_module, "load_brats_candidate_rows", fake_candidate_loader)
        monkeypatch.setattr(replay_module, "verify_selected_brats_tensor_hashes", lambda *args, **kwargs: {"status": "PASS"})
        monkeypatch.setattr(replay_module, "verify_source_fingerprint_continuity", lambda *args, **kwargs: {"status": "PASS"})
        monkeypatch.setattr(
            replay_module,
            "verify_healthy_source_tensors",
            lambda *args, **kwargs: {
                "status": "PASS",
                "tensor_hash_ledger_status": "PASS",
                "tensor_hash_ledger_expected_entries": 1,
                "tensor_hash_ledger_checked_entries": 1,
                "tensor_hash_ledger_unobserved_entries": 0,
                "tensor_hash_ledger_failure_count": 0,
            },
        )

        happy = replay_module.run_replay(source, workspace / "replay_pass")
        assert happy["status"] == "PASS"
        assert happy["healthy_tensor_hash_ledger"]["status"] == "PASS"
        assert happy["healthy_tensor_hash_ledger"]["actual_ledger_sha256"] == ledger_digest
        assert json.loads((workspace / "replay_pass" / "replay_audit.json").read_text(encoding="utf-8"))["status"] == "PASS"

        (binding / "selected_healthy_source_tensor_ledger_summary.json").write_text(
            json.dumps({"status": "PASS", "ledger_path": str(ledger.resolve()), "ledger_sha256": "wrong"}),
            encoding="utf-8",
        )
        failed = replay_module.run_replay(source, workspace / "replay_digest_fail")
        assert failed["status"] == "FAIL_CLOSED"
        assert failed["healthy_tensor_hash_ledger"]["status"] == "FAIL_LEDGER_DIGEST_MISMATCH"
        assert "healthy_tensor_hash_ledger_inconclusive_or_mismatch" in failed["failures"]
