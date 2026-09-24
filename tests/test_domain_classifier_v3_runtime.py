from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import numpy as np

from andi_rewrite.domain_classifier.runner import TrainConfig
from andi_rewrite.domain_classifier.v3_runtime import (
    MODEL_SHAPE,
    SPLITS,
    V3InputValidationError,
    permute_cached_v3_labels,
    set_single_thread_runtime,
    sha256_file,
    validate_and_materialize_v3_inputs,
    verify_source_freeze,
)
import andi_rewrite.domain_classifier.v3_runtime as v3_runtime
import scripts.run_domain_classifier_primary_null_v3 as primary_v3


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def frozen_fixture(tmp_path: Path) -> dict[str, Any]:
    """A small complete manifest/ledger with an injectable canonical reader."""

    build_root = tmp_path / "build"
    manifest_root = build_root / "manifests" / "fomo45k"
    manifest_root.mkdir(parents=True)
    tensors: dict[tuple[str, str, str, str], torch.Tensor] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    ledger_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        rows: list[dict[str, Any]] = []
        for pair_index in range(2):
            pair_id = f"fomo45k:{split}:pair-{pair_index}"
            z = split_index * 10 + pair_index
            healthy_participant = f"fomo45k:healthy-{split}-{pair_index}"
            healthy_key = f"healthy-{split}-{pair_index}"
            healthy = torch.full(MODEL_SHAPE, float(10 + split_index + pair_index), dtype=torch.float32)
            healthy_row: dict[str, Any] = {
                "split": split,
                "label": 0,
                "domain": "fomo45k",
                "source_dataset": "fomo45k",
                "source_split": split,
                "source_key": healthy_key,
                "participant_id": healthy_participant,
                "pair_id": pair_id,
                "case_id": f"healthy-case-{split}-{pair_index}",
                "session_id": "ses-1",
                "z": z,
                "z_bin": z,
                "model_shape": list(MODEL_SHAPE),
                "metadata": {"source_participant_id": healthy_participant.split(":", 1)[1]},
                "provenance": {
                    "source_split_immutable": split,
                    "source_key_immutable": healthy_key,
                },
            }
            healthy_identity = ("healthy", "fomo45k", split, healthy_key)
            tensors[healthy_identity] = healthy
            ledger_rows.append(
                {
                    "canonical_source_dataset": "fomo45k",
                    "canonical_source_split": split,
                    "canonical_source_key": healthy_key,
                    "dtype": "float32",
                    "shape": list(MODEL_SHAPE),
                    "tensor_sha256": _tensor_digest(healthy),
                    "memberships": [
                        {
                            "comparison": "fomo45k",
                            "target_split": split,
                            "manifest_split": split,
                            "participant_id": healthy_participant,
                            "pair_id": pair_id,
                            "z": z,
                            "local_source_dataset": "fomo45k",
                            "local_source_split": split,
                            "local_source_key": healthy_key,
                            "mixed_local_key": "",
                            "underlying_source_dataset": "fomo45k",
                            "underlying_source_split": split,
                            "underlying_source_key": healthy_key,
                        }
                    ],
                }
            )
            brats_participant = f"brats21:BraTS2021_TEST_{split_index}_{pair_index}"
            brats = torch.full(MODEL_SHAPE, float(30 + split_index + pair_index), dtype=torch.float32)
            brats_row: dict[str, Any] = {
                "split": split,
                "label": 1,
                "domain": "brats21",
                "source_dataset": "brats21",
                "source_split": "test",
                "source_key": f"brats-{split}-{pair_index}",
                "participant_id": brats_participant,
                "pair_id": pair_id,
                "case_id": f"brats-case-{split}-{pair_index}",
                "session_id": "ses-1",
                "z": z,
                "z_bin": z,
                "model_shape": list(MODEL_SHAPE),
                "metadata": {"source_participant_id": brats_participant},
                "provenance": {"model_input_sha256": _tensor_digest(brats)},
            }
            brats_identity = ("brats21", brats_participant.split(":", 1)[1], "z", str(z))
            tensors[brats_identity] = brats
            rows.extend((healthy_row, brats_row))
        rows_by_split[split] = rows
        _write_jsonl(manifest_root / f"{split}.jsonl", rows)
    ledger_path = build_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl"
    _write_jsonl(ledger_path, ledger_rows)
    ledger_sha = sha256_file(ledger_path)
    summary_path = ledger_path.with_name("selected_healthy_source_tensor_ledger_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "ledger_rows": len(ledger_rows),
                "ledger_sha256": ledger_sha,
                "manifest_identity": {
                    f"fomo45k/{split}": {
                        "path": str((manifest_root / f"{split}.jsonl").resolve()),
                        "sha256": sha256_file(manifest_root / f"{split}.jsonl"),
                    }
                    for split in SPLITS
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, str, str]] = []

    def loader(row: dict[str, Any], **_: Any) -> torch.Tensor:
        source = row.get("source_dataset")
        if source == "fomo45k":
            identity = ("healthy", "fomo45k", str(row["source_split"]), str(row["source_key"]))
        else:
            identity = ("brats21", str(row["participant_id"]).split(":", 1)[1], "z", str(row["z"]))
        calls.append(identity)
        return tensors[identity]

    return {
        "build_root": build_root,
        "manifest_root": manifest_root,
        "ledger_path": ledger_path,
        "summary_path": summary_path,
        "ledger_sha": ledger_sha,
        "tensors": tensors,
        "rows_by_split": rows_by_split,
        "loader": loader,
        "calls": calls,
    }


def _materialize(fixture: dict[str, Any]):
    return validate_and_materialize_v3_inputs(
        fixture["build_root"],
        comparison="fomo45k",
        expected_ledger_sha256=fixture["ledger_sha"],
        loader_factory=fixture["loader"],
    )


def test_materialization_binds_ledger_and_projects_without_rereading(frozen_fixture: dict[str, Any]) -> None:
    cached = _materialize(frozen_fixture)
    assert cached.input_binding_audit["status"] == "PASS"
    assert cached.input_binding_audit["rows_checked"] == 12
    assert cached.input_binding_audit["unique_canonical_tensors"] == 12
    assert cached.input_binding_audit["manifest_prelaunch_anchor"]["status"] == "PASS"
    assert len(frozen_fixture["calls"]) == 12
    projected = cached.projected(("t1",))
    assert tuple(projected["test"].tensors[0].shape) == MODEL_SHAPE
    assert tuple(projected["test"][0]["image"].shape) == (1, 128, 128)
    assert projected["test"].tensors[0].data_ptr() == cached.datasets["test"].tensors[0].data_ptr()


def test_materialization_rejects_wrong_ledger_or_brats_digest(frozen_fixture: dict[str, Any]) -> None:
    with pytest.raises(V3InputValidationError, match="frozen prelaunch ledger"):
        validate_and_materialize_v3_inputs(
            frozen_fixture["build_root"],
            comparison="fomo45k",
            expected_ledger_sha256="0" * 64,
            loader_factory=frozen_fixture["loader"],
        )
    test_path = frozen_fixture["manifest_root"] / "test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["provenance"]["model_input_sha256"] = "bad"
    _write_jsonl(test_path, rows)
    summary = json.loads(frozen_fixture["summary_path"].read_text(encoding="utf-8"))
    summary["manifest_identity"]["fomo45k/test"]["sha256"] = sha256_file(test_path)
    frozen_fixture["summary_path"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(V3InputValidationError, match="tensor hash mismatch"):
        _materialize(frozen_fixture)


def test_materialization_rejects_empty_participant_or_pair_identity(frozen_fixture: dict[str, Any]) -> None:
    path = frozen_fixture["manifest_root"] / "val.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["participant_id"] = ""
    _write_jsonl(path, rows)
    with pytest.raises(V3InputValidationError, match="empty_participant_id"):
        _materialize(frozen_fixture)
    rows[0]["participant_id"] = "fomo45k:healthy-val-0"
    rows[0]["pair_id"] = ""
    _write_jsonl(path, rows)
    with pytest.raises(V3InputValidationError, match="empty_pair_id"):
        _materialize(frozen_fixture)


def test_mixed_membership_uses_underlying_tensor_and_local_membership_key(
    frozen_fixture: dict[str, Any],
) -> None:
    mixed_root = frozen_fixture["build_root"] / "manifests" / "mixed"
    mixed_root.mkdir(parents=True)
    mixed_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    mixed_ledger: list[dict[str, Any]] = []
    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        for row in frozen_fixture["rows_by_split"][split]:
            value = json.loads(json.dumps(row))
            if int(value["label"]) == 0:
                underlying_key = value["source_key"]
                local_key = f"mixed-local-{split}-{underlying_key}"
                healthy_tensor = frozen_fixture["tensors"][(
                    "healthy",
                    "fomo45k",
                    split,
                    underlying_key,
                )]
                value["source_dataset"] = "mixed"
                value["domain"] = "mixed"
                value["source_key"] = local_key
                value["metadata"].update(
                    {
                        "underlying_source_dataset": "fomo45k",
                        "underlying_source_split": split,
                        "underlying_source_key": underlying_key,
                        "mixed_local_key": local_key,
                    }
                )
                value["provenance"].update(
                    {
                        "source_key_immutable": underlying_key,
                        "mixed_local_key_immutable": local_key,
                    }
                )
                mixed_ledger.append(
                    {
                        "canonical_source_dataset": "fomo45k",
                        "canonical_source_split": split,
                        "canonical_source_key": underlying_key,
                        "dtype": "float32",
                        "shape": list(MODEL_SHAPE),
                        "tensor_sha256": _tensor_digest(healthy_tensor),
                        "memberships": [
                            {
                                "comparison": "mixed",
                                "target_split": split,
                                "manifest_split": split,
                                "participant_id": value["participant_id"],
                                "pair_id": value["pair_id"],
                                "z": value["z"],
                                "local_source_dataset": "mixed",
                                "local_source_split": split,
                                "local_source_key": local_key,
                                "mixed_local_key": local_key,
                                "underlying_source_dataset": "fomo45k",
                                "underlying_source_split": split,
                                "underlying_source_key": underlying_key,
                            }
                        ],
                    }
                )
            rows.append(value)
        mixed_rows_by_split[split] = rows
        _write_jsonl(mixed_root / f"{split}.jsonl", rows)
    mixed_ledger_path = frozen_fixture["build_root"] / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl"
    _write_jsonl(mixed_ledger_path, mixed_ledger)
    mixed_sha = sha256_file(mixed_ledger_path)
    mixed_summary_path = mixed_ledger_path.with_name("selected_healthy_source_tensor_ledger_summary.json")
    mixed_summary_path.write_text(
        json.dumps({"status": "PASS", "ledger_rows": len(mixed_ledger), "ledger_sha256": mixed_sha}),
        encoding="utf-8",
    )

    def mixed_loader(row: dict[str, Any], **_: Any) -> torch.Tensor:
        if row["source_dataset"] == "mixed":
            identity = (
                "healthy",
                "fomo45k",
                row["metadata"]["underlying_source_split"],
                row["metadata"]["underlying_source_key"],
            )
        else:
            identity = ("brats21", str(row["participant_id"]).split(":", 1)[1], "z", str(row["z"]))
        return frozen_fixture["tensors"][identity]

    cached = validate_and_materialize_v3_inputs(
        frozen_fixture["build_root"],
        comparison="mixed",
        expected_ledger_sha256=mixed_sha,
        loader_factory=mixed_loader,
    )
    assert cached.input_binding_audit["status"] == "PASS"
    assert cached.input_binding_audit["healthy_ledger"]["scoped_memberships"] == 6


def test_source_freeze_rejects_manifest_drift(frozen_fixture: dict[str, Any]) -> None:
    cached = _materialize(frozen_fixture)
    path = frozen_fixture["manifest_root"] / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(V3InputValidationError, match="frozen source/config/data bytes drifted"):
        verify_source_freeze(cached.source_freeze)


def test_pair_permutation_is_deterministic_and_keeps_cached_tensors(frozen_fixture: dict[str, Any]) -> None:
    cached = _materialize(frozen_fixture)
    first, streams_first, detail_first = permute_cached_v3_labels(cached, index=17)
    second, streams_second, detail_second = permute_cached_v3_labels(cached, index=17)
    assert streams_first == streams_second
    assert detail_first == detail_second
    for split in SPLITS:
        original = cached.records_by_split[split]
        permuted = first[split].records
        assert [row["label"] for row in permuted] == [row["label"] for row in second[split].records]
        by_pair: dict[str, list[tuple[int, int]]] = {}
        for before, after in zip(original, permuted):
            by_pair.setdefault(str(before["pair_id"]), []).append((int(before["label"]), int(after["label"])))
        for values in by_pair.values():
            assert sorted(before for before, _ in values) == [0, 1]
            assert sorted(after for _, after in values) == [0, 1]
            assert len({after != before for before, after in values}) == 1
        assert all(
            first[split].tensors[index].data_ptr() == cached.datasets[split].tensors[index].data_ptr()
            for index in range(len(first[split]))
        )


def test_thread_contract_is_measured_in_a_fresh_process() -> None:
    code = (
        "import json; "
        "from andi_rewrite.domain_classifier.v3_runtime import set_single_thread_runtime; "
        "print(json.dumps(set_single_thread_runtime()))"
    )
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    project_parent = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project_parent), str(project_parent / "andi_rewrite")]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(result.stdout)
    assert snapshot["status"] == "PASS"
    assert snapshot["torch_num_threads"] == 1
    assert snapshot["torch_num_interop_threads"] == 1


def test_null_draw_resume_checks_reconstructed_labels_predictions_and_artifact_hashes(
    frozen_fixture: dict[str, Any], tmp_path: Path
) -> None:
    cached = _materialize(frozen_fixture)
    permuted, streams, detail = permute_cached_v3_labels(cached, index=0)
    draw = tmp_path / "permutation_0000"
    draw.mkdir()
    labels = primary_v3._label_rows(cached, permuted, fingerprint="fp", index=0)
    primary_v3._write_jsonl(draw / "labels.jsonl", labels)
    predictions = []
    for index, row in enumerate(permuted["test"].records):
        predictions.append(
            {
                "record_index": index,
                "label": int(row["label"]),
                "probability": 0.5,
                "participant_id": row["participant_id"],
                "pair_id": row["pair_id"],
                "case_id": row["case_id"],
            }
        )
    primary_v3._write_jsonl(draw / "test_predictions.jsonl", predictions)
    (draw / "model_best.pt").write_bytes(b"fixture-checkpoint")
    result = {
        "run_fingerprint": "fp",
        "result_kind": "v3_primary_full_retrained_pair_null",
        "permutation_index": 0,
        "init_seed": 73,
        "config": {"model": "small_cnn"},
        "label_stream_seeds": streams,
        "swap_counts": detail["swap_counts"],
        "test": {"subject": {"roc_auc": 0.5}},
        "artifact_hashes": {
            "labels": sha256_file(draw / "labels.jsonl"),
            "test_predictions": sha256_file(draw / "test_predictions.jsonl"),
            "model_best": sha256_file(draw / "model_best.pt"),
        },
    }
    (draw / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert primary_v3._null_draw_complete(draw, fingerprint="fp", cached=cached, index=0)
    labels = primary_v3._read_jsonl(draw / "labels.jsonl")
    labels[0]["label"] = 1 - int(labels[0]["label"])
    primary_v3._write_jsonl(draw / "labels.jsonl", labels, overwrite=True)
    with pytest.raises(primary_v3.V3RuntimeError, match="artifact bytes drifted"):
        primary_v3._null_draw_complete(draw, fingerprint="fp", cached=cached, index=0)


def test_primary_dry_run_writes_binding_without_calling_fit(
    frozen_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = _materialize(frozen_fixture)
    monkeypatch.setattr(primary_v3, "validate_and_materialize_v3_inputs", lambda *args, **kwargs: cached)
    monkeypatch.setattr(
        primary_v3,
        "set_single_thread_runtime",
        lambda: {
            "status": "PASS",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "errors": [],
        },
    )
    fit_called = {"count": 0}

    def forbidden_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        fit_called["count"] += 1
        raise AssertionError("dry run must not fit")

    monkeypatch.setattr(primary_v3, "fit_cached_v3_cell", forbidden_fit)
    result = primary_v3.run_primary_null_v3(
        comparison="fomo45k",
        manifest_root=frozen_fixture["build_root"],
        output_root=tmp_path / "dry-run",
        expected_ledger_sha256=frozen_fixture["ledger_sha"],
        device="cpu",
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert fit_called["count"] == 0
    assert (tmp_path / "dry-run" / "input_binding_audit.json").is_file()
    assert json.loads((tmp_path / "dry-run" / "dry_run.json").read_text())[
        "thread_contract"
    ]["torch_num_threads"] == 1


def test_reference_config_must_match_primary_architecture(tmp_path: Path) -> None:
    path = tmp_path / "reference.yaml"
    path.write_text("training:\n  model: resnet18\n  max_epochs: 40\n", encoding="utf-8")
    with pytest.raises(primary_v3.V3RuntimeError, match="reference config disagrees"):
        primary_v3._validate_reference_config(path, primary_v3._primary_config(device="cpu", init_seed=73))


def test_primary_init_seed_is_frozen_to_73(
    frozen_fixture: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(primary_v3.V3RuntimeError, match="frozen to init_seed=73"):
        primary_v3.run_primary_null_v3(
            comparison="fomo45k",
            manifest_root=frozen_fixture["build_root"],
            output_root=tmp_path / "wrong-seed",
            expected_ledger_sha256=frozen_fixture["ledger_sha"],
            device="cpu",
            init_seed=173,
            observed_only=True,
        )


def test_primary_observed_resume_reuses_artifacts_without_refit(
    frozen_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = _materialize(frozen_fixture)
    monkeypatch.setattr(primary_v3, "validate_and_materialize_v3_inputs", lambda *args, **kwargs: cached)
    monkeypatch.setattr(
        primary_v3,
        "set_single_thread_runtime",
        lambda: {
            "status": "PASS",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "errors": [],
        },
    )
    calls = {"count": 0}

    def fake_fit(cached_input: Any, *, config: Any, seed: int) -> dict[str, Any]:
        calls["count"] += 1
        def prediction_rows(split: str) -> list[dict[str, Any]]:
            return [
                {
                    "record_index": index,
                    "label": int(row["label"]),
                    "probability": 0.5,
                    "participant_id": row["participant_id"],
                    "pair_id": row["pair_id"],
                    "case_id": row["case_id"],
                }
                for index, row in enumerate(cached_input.records_by_split[split])
            ]
        return {
            "seed": seed,
            "config": {"model": "small_cnn"},
            "validation": {"subject": {"roc_auc": 0.5}},
            "test": {"subject": {"roc_auc": 0.5}},
            "validation_predictions": prediction_rows("val"),
            "test_predictions": prediction_rows("test"),
            "state_dict": {"fixture": torch.tensor([1.0])},
        }

    monkeypatch.setattr(primary_v3, "fit_cached_v3_cell", fake_fit)
    output = tmp_path / "observed"
    first = primary_v3.run_primary_null_v3(
        comparison="fomo45k",
        manifest_root=frozen_fixture["build_root"],
        output_root=output,
        expected_ledger_sha256=frozen_fixture["ledger_sha"],
        device="cpu",
        observed_only=True,
    )
    second = primary_v3.run_primary_null_v3(
        comparison="fomo45k",
        manifest_root=frozen_fixture["build_root"],
        output_root=output,
        expected_ledger_sha256=frozen_fixture["ledger_sha"],
        device="cpu",
        observed_only=True,
    )
    assert first["status"] == "observed_complete"
    assert second["status"] == "observed_reused"
    assert calls["count"] == 1


def test_logistic_cached_cell_fits_once_and_keeps_pair_bootstrap(
    frozen_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = _materialize(frozen_fixture)
    calls = {"count": 0}

    class FakeScaler:
        def __init__(self, columns: int) -> None:
            self.mean_ = np.zeros(columns, dtype=np.float64)
            self.scale_ = np.ones(columns, dtype=np.float64)

        def transform(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

    class FakeModel:
        def __init__(self, columns: int) -> None:
            self.coef_ = np.zeros((1, columns), dtype=np.float64)
            self.intercept_ = np.zeros(1, dtype=np.float64)
            self.n_iter_ = np.asarray([3], dtype=np.int64)
            self.max_iter = 1000

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            count = int(np.asarray(values).shape[0])
            return np.column_stack([np.full(count, 0.5), np.full(count, 0.5)])

    def fake_fit(train_features: np.ndarray, train_labels: Any, eval_features: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        columns = int(np.asarray(train_features).shape[1])
        model = FakeModel(columns)
        return {
            "model": model,
            "scaler": FakeScaler(columns),
            "scores": np.full(int(np.asarray(eval_features).shape[0]), 0.5),
            "train_feature_mean": model.coef_[0],
            "train_feature_scale": np.ones(columns),
        }

    monkeypatch.setattr(v3_runtime, "fit_statistical_logistic", fake_fit)
    config = TrainConfig(
        model="logistic",
        modalities=("flair", "t1", "t2"),
        in_channels=3,
        device="cpu",
        bootstrap_replicates=4,
        swap_replicates=0,
    )
    result = v3_runtime.fit_cached_v3_cell(cached, config=config, seed=73)
    assert calls["count"] == 1
    assert result["logistic"]["single_train_fit_for_val_and_test"] is True
    assert result["test_statistics"]["subject_bootstrap"]["n_bootstrap"] == 4
    assert result["test_statistics"]["subject_bootstrap"]["resampling_unit"] == "matched_pair"
    assert "heldout_pair_swap" not in result["test_statistics"]
    assert result["logistic"]["convergence"]["n_iter"] == [3]
