from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from andi_rewrite.domain_classifier.runner import TrainConfig
from andi_rewrite.domain_classifier.v3_runtime import V3CachedInputs
import scripts.run_domain_classifier_cell_v3 as cell_v3


def _config_payload(model: str = "statistical_logistic", modalities: list[str] | None = None) -> dict[str, Any]:
    values = modalities or ["flair"]
    training: dict[str, Any] = {
        "model": model,
        "in_channels": len(values),
        "modalities": values,
        "num_classes": 1,
        "widths": [32, 64, 128, 128],
        "groupnorm_groups": 8,
        "dropout": 0.0,
        "learning_rate": None if "logistic" in model else (0.001 if model == "small_cnn" else 0.0003),
        "weight_decay": 0.0 if "logistic" in model else 0.0001,
        "max_epochs": 40,
        "patience": 8,
        "batch_size": 32,
        "num_workers": 0,
        "threshold": 0.5,
        "split_seed": 73,
        "stage": "final",
        "device": "cpu",
        "subject_method": "mean",
        "no_augmentation": True,
        "tiny": False,
        "early_stopping": False if "logistic" in model else True,
        "bootstrap_replicates": 2000,
        "swap_replicates": 0,
        "permutation_replicates": 0,
        "permutation_mode": "none",
    }
    if "logistic" in model:
        training.update(
            {
                "logistic_C": 1.0,
                "logistic_solver": "lbfgs",
                "train_only_standardizer": True,
                "inverse_slice_count_weights": True,
            }
        )
    return {"comparison": "mpi", "training": training}


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _rows() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        output[split] = [
            {
                "participant_id": f"healthy-{split}",
                "pair_id": f"pair-{split}",
                "case_id": f"healthy-case-{split}",
                "label": 0,
            },
            {
                "participant_id": f"brats-{split}",
                "pair_id": f"pair-{split}",
                "case_id": f"brats-case-{split}",
                "label": 1,
            },
        ]
    return output


def _fake_cached(tmp_path: Path) -> V3CachedInputs:
    rows = _rows()
    tensor = torch.zeros((3, 128, 128), dtype=torch.float32)
    identity = ("healthy", "mpi", "train", "healthy")
    return V3CachedInputs(
        comparison="mpi",
        build_root=tmp_path / "build",
        manifest_root=tmp_path / "manifests",
        manifest_paths={split: tmp_path / "manifests" / f"{split}.jsonl" for split in rows},
        records_by_split=rows,
        datasets={},
        tensors_by_identity={identity: tensor},
        tensor_digests_by_identity={identity: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()},
        input_binding_audit={
            "status": "PASS",
            "comparison": "mpi",
            "manifest_identity": {},
            "rows_checked": 6,
        },
        source_freeze={"status": "PASS", "files": []},
        ledger=SimpleNamespace(ledger_sha256="ledger-sha"),
        manifest_fingerprints={"train": "train-sha", "val": "val-sha", "test": "test-sha"},
    )


def _prediction_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_index": index,
            "label": int(row["label"]),
            "probability": 0.25 if int(row["label"]) == 0 else 0.75,
            "participant_id": row["participant_id"],
            "pair_id": row["pair_id"],
            "case_id": row["case_id"],
        }
        for index, row in enumerate(records)
    ]


def _subject_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "subject_index": index,
            "label": int(row["label"]),
            "probability": 0.25 if int(row["label"]) == 0 else 0.75,
            "participant_id": row["participant_id"],
            "pair_id": row["pair_id"],
            "slice_count": 1,
        }
        for index, row in enumerate(records)
    ]


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, cached: V3CachedInputs, *, model: str = "statistical_logistic") -> dict[str, int]:
    monkeypatch.setattr(cell_v3, "validate_and_materialize_v3_inputs", lambda *args, **kwargs: cached)
    monkeypatch.setattr(
        cell_v3,
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
    calls = {"fit": 0}

    def fake_fit(cached_input: Any, *, config: TrainConfig, seed: int) -> dict[str, Any]:
        calls["fit"] += 1
        result: dict[str, Any] = {
            "seed": seed,
            "split_seed": 73,
            "config": dict(cell_v3.asdict(config)),
            "device": "cpu",
            "best_epoch": 1,
            "epochs_completed": 1,
            "history": [{"epoch": 1, "train_loss": 0.1}],
            "validation": {"subject": {"roc_auc": 0.75}},
            "test": {"subject": {"roc_auc": 0.75}},
            "test_statistics": {"subject_bootstrap": {"n_bootstrap": 2000}},
            "validation_predictions": _prediction_rows(cached_input.records_by_split["val"]),
            "validation_subject_predictions": _subject_rows(cached_input.records_by_split["val"]),
            "test_predictions": _prediction_rows(cached_input.records_by_split["test"]),
            "test_subject_predictions": _subject_rows(cached_input.records_by_split["test"]),
            "train_final_predictions": [],
            "train_final_subject_predictions": [],
            "model": {"class": "LogisticRegression" if "logistic" in model else model},
        }
        if "logistic" in model:
            result["state_dict"] = None
            result["logistic"] = {
                "C": 1.0,
                "train_only_standardizer": True,
                "single_train_fit_for_val_and_test": True,
                "model_coef": [[0.0]],
                "model_intercept": [0.0],
                "convergence": {"n_iter": [3], "max_iter": 1000, "converged": True, "warnings": []},
            }
        else:
            result["state_dict"] = {"weight": torch.ones(1)}
        return result

    monkeypatch.setattr(cell_v3, "fit_cached_v3_cell", fake_fit)
    return calls


def test_cli_contract_requires_one_seed_and_supports_matrix_flags() -> None:
    args = cell_v3.build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--manifest-root",
            "manifests/mpi",
            "--output-dir",
            "out",
            "--stage",
            "final",
            "--device",
            "cuda",
            "--modalities",
            "flair",
            "--seeds",
            "73",
            "--comparison",
            "mpi",
            "--build-root",
            "build",
            "--dry-run",
        ]
    )
    assert args.stage == "final"
    assert args.modalities == ["flair"]
    assert args.seeds == [73]
    with pytest.raises(cell_v3.V3RuntimeError, match="exactly one seed"):
        cell_v3.run_cell_v3(
            config_path="c",
            manifest_root="m",
            output_dir="o",
            seeds=[73, 173],
        )


def test_config_contract_has_expected_models_and_seed_rules(tmp_path: Path) -> None:
    payload = _config_payload("resnet18", ["t1"])
    config = cell_v3._config_from_yaml(payload, stage="final", device="cpu", modalities=["t1"], seed=173)
    assert config.model == "resnet18"
    assert config.modalities == ("t1",)
    assert config.in_channels == 1
    assert config.learning_rate == 0.0003
    logistic = cell_v3._config_from_yaml(_config_payload(), stage="final", device="cpu", modalities=["flair"], seed=73)
    assert logistic.model == "statistical_logistic"
    assert logistic.logistic_C == 1.0  # type: ignore[attr-defined]
    with pytest.raises(cell_v3.V3RuntimeError, match="logistic cells are frozen"):
        cell_v3._config_from_yaml(_config_payload(), stage="final", device="cpu", modalities=["flair"], seed=173)
    with pytest.raises(cell_v3.V3RuntimeError, match="primary cell"):
        cell_v3._config_from_yaml(_config_payload("small_cnn", ["flair", "t1", "t2"]), stage="final", device="cpu", modalities=None, seed=73)


def test_dry_run_materializes_contract_without_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _fake_cached(tmp_path)
    calls = _patch_runtime(monkeypatch, cached)
    monkeypatch.setattr(cell_v3, "fit_cached_v3_cell", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry run fitted")))
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_payload())
    result = cell_v3.run_cell_v3(
        config_path=config_path,
        manifest_root=tmp_path / "manifests",
        output_dir=tmp_path / "out",
        comparison="mpi",
        device="cpu",
        modalities=["flair"],
        seeds=[73],
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert calls["fit"] == 0
    assert json.loads((tmp_path / "out" / "dry_run.json").read_text())["training_started"] is False
    assert json.loads((tmp_path / "out" / "input_binding_audit.json").read_text())["status"] == "PASS"


def test_logistic_fit_persists_state_and_resume_does_not_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _fake_cached(tmp_path)
    calls = _patch_runtime(monkeypatch, cached)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_payload())
    kwargs = {
        "config_path": config_path,
        "manifest_root": tmp_path / "manifests",
        "output_dir": tmp_path / "out",
        "comparison": "mpi",
        "device": "cpu",
        "modalities": ["flair"],
        "seeds": [73],
    }
    first = cell_v3.run_cell_v3(**kwargs)
    second = cell_v3.run_cell_v3(**kwargs)
    assert first["status"] == "COMPLETE"
    assert second["status"] == "REUSED"
    assert calls["fit"] == 1
    assert (tmp_path / "out" / "logistic_state.json").is_file()
    assert (tmp_path / "out" / "fit_started.json").is_file()
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["formal_final"] is True
    assert result["control_mode"] is None
    assert result["artifact_hashes"]["logistic_state"]
    assert result["artifact_hashes"]["fit_started"]
    assert result["test_statistics"]["subject_bootstrap"]["n_bootstrap"] == 2000


def test_started_sidecar_makes_interrupted_fit_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _fake_cached(tmp_path)
    _patch_runtime(monkeypatch, cached)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_payload())

    def interrupted_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cell_v3, "fit_cached_v3_cell", interrupted_fit)
    kwargs = {
        "config_path": config_path,
        "manifest_root": tmp_path / "manifests",
        "output_dir": tmp_path / "out",
        "comparison": "mpi",
        "device": "cpu",
        "modalities": ["flair"],
        "seeds": [73],
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cell_v3.run_cell_v3(**kwargs)
    assert (tmp_path / "out" / "fit_started.json").is_file()
    with pytest.raises(cell_v3.V3RuntimeError, match="partial"):
        cell_v3.run_cell_v3(**kwargs)


def test_tampered_prediction_checkpoint_and_code_are_rejected_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _fake_cached(tmp_path)
    _patch_runtime(monkeypatch, cached, model="small_cnn")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_payload("small_cnn", ["flair"]))
    kwargs = {
        "config_path": config_path,
        "manifest_root": tmp_path / "manifests",
        "output_dir": tmp_path / "out",
        "comparison": "mpi",
        "device": "cpu",
        "modalities": ["flair"],
        "seeds": [173],
    }
    cell_v3.run_cell_v3(**kwargs)
    prediction = tmp_path / "out" / "test_predictions.jsonl"
    prediction.write_text(prediction.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(cell_v3.V3RuntimeError, match="artifact bytes drifted"):
        cell_v3.run_cell_v3(**kwargs)

    # Rebuild a separate immutable output to exercise checkpoint and snapshot
    # checks independently of the prediction hash check.
    output2 = tmp_path / "out2"
    kwargs["output_dir"] = output2
    cell_v3.run_cell_v3(**kwargs)
    checkpoint = output2 / "model_best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(cell_v3.V3RuntimeError, match="artifact bytes drifted"):
        cell_v3.run_cell_v3(**kwargs)

    output3 = tmp_path / "out3"
    kwargs["output_dir"] = output3
    cell_v3.run_cell_v3(**kwargs)
    snapshot_file = output3 / "code_snapshot" / "domain_classifier" / "v3_runtime.py"
    snapshot_file.write_bytes(snapshot_file.read_bytes() + b"tamper")
    with pytest.raises(cell_v3.V3RuntimeError, match="code snapshot differs"):
        cell_v3.run_cell_v3(**kwargs)


def test_manifest_binding_change_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _fake_cached(tmp_path)
    calls = _patch_runtime(monkeypatch, cached)
    original_validator = cell_v3.validate_and_materialize_v3_inputs
    manifest = tmp_path / "manifests" / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("frozen\n", encoding="utf-8")
    first_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    state = {"hash": first_hash}

    def validator(*args: Any, **kwargs: Any) -> V3CachedInputs:
        current = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if current != state["hash"]:
            raise cell_v3.V3InputValidationError("manifest bytes drifted")
        return cached

    monkeypatch.setattr(cell_v3, "validate_and_materialize_v3_inputs", validator)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_payload())
    kwargs = {
        "config_path": config_path,
        "manifest_root": tmp_path / "manifests",
        "output_dir": tmp_path / "out",
        "comparison": "mpi",
        "device": "cpu",
        "modalities": ["flair"],
        "seeds": [73],
    }
    cell_v3.run_cell_v3(**kwargs)
    manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(cell_v3.V3InputValidationError, match="manifest bytes drifted"):
        cell_v3.run_cell_v3(**kwargs)
    assert calls["fit"] == 1
    del original_validator
