from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_domain_classifier_matrix_v3 import (
    ACTUAL_MODELS,
    COHORTS,
    FAMILY_SEEDS,
    INPUT_TYPES,
    MatrixProtocolError,
    _control_identity,
    _control_requirement,
    _cell_control_requirements,
    _cell_execution_gate,
    _cell_config,
    build_matrix_plan,
    execution_gate,
    execute_plan,
    write_plan_artifacts,
)


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": "model_grid_v3_training_v1",
        "matrix": {
            "cohorts": list(COHORTS),
            "input_types": ["flair", "t1", "t2", "flair+t1+t2"],
            "families": {
                family: {"seeds": list(seeds), "count": len(seeds)}
                for family, seeds in FAMILY_SEEDS.items()
            },
            "total_fits": 112,
        },
    }


def _amendment() -> dict[str, object]:
    return {
        "hyperparameters": {
            "small_cnn": {"lr": 1.0e-3},
            "resnet18_scratch_gn": {"lr": 3.0e-4},
            "logistic": {"C": 1.0, "selection": "none; fixed C=1 train fit only"},
        }
    }


def _write_manifests(build_root: Path) -> None:
    for cohort in COHORTS:
        root = build_root / "manifests" / cohort
        root.mkdir(parents=True)
        for split in ("train", "val", "test"):
            row = {
                "participant_id": f"{cohort}:{split}:subject",
                "pair_id": f"{cohort}:{split}:pair",
                "label": 0,
            }
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_plan_is_exactly_112_and_uses_family_seed_contract(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    plan = build_matrix_plan(
        _protocol(),
        amendment=_amendment(),
        build_root=build_root,
        output_root=tmp_path / "matrix",
        device="cpu",
    )
    assert plan["fit_count"] == 112
    assert len(plan["cells"]) == 112
    assert {cell["family"] for cell in plan["cells"]} == set(FAMILY_SEEDS)
    for family, seeds in FAMILY_SEEDS.items():
        found = {int(cell["seed"]) for cell in plan["cells"] if cell["family"] == family}
        assert found == set(seeds)
        assert all(cell["model"] == ACTUAL_MODELS[family] for cell in plan["cells"] if cell["family"] == family)
    assert {tuple(cell["modalities"]) for cell in plan["cells"]} == set(INPUT_TYPES)
    assert plan["secondary_group_count"] == 48
    assert len(plan["secondary_groups"]) == 48
    primary = next(
        cell
        for cell in plan["cells"]
        if cell["cohort"] == "fomo45k"
        and cell["family"] == "small_cnn"
        and tuple(cell["modalities"]) == ("flair", "t1", "t2")
        and cell["seed"] == 73
    )
    assert primary["analysis_roles"]["primary_full_retrained_null"] is True
    assert primary["analysis_roles"]["primary_null_replicates"] == 199
    assert primary["existing_fit"]["kind"] == "observed_calibration"
    assert primary["primary_null"]["required"] is True
    assert primary["primary_null"]["reuse_existing"] is True
    assert primary["execution_route"] == "reuse_existing_primary_observed_and_full_null"
    ordinary = next(
        cell
        for cell in plan["cells"]
        if cell["cohort"] == "mpi"
        and cell["family"] == "small_cnn"
        and tuple(cell["modalities"]) == ("flair",)
        and cell["seed"] == 173
    )
    assert ordinary["execution_route"] == "run_domain_classifier_cell_v3"
    assert ordinary["command"][1].endswith("run_domain_classifier_cell_v3.py")
    assert "--ledger-path" in ordinary["command"]
    mpi_primary = next(
        cell
        for cell in plan["cells"]
        if cell["cohort"] == "mpi"
        and cell["family"] == "small_cnn"
        and tuple(cell["modalities"]) == ("flair", "t1", "t2")
        and cell["seed"] == 73
    )
    assert mpi_primary["primary_null"]["command"][-1] == "73"
    assert mpi_primary["primary_null"]["entrypoint"].endswith("run_domain_classifier_primary_null_v3.py")
    assert "--build-root" in mpi_primary["primary_null"]["command"]
    assert mpi_primary["execution_route"] == "run_primary_null_v3"
    assert mpi_primary["command"][1].endswith("run_domain_classifier_primary_null_v3.py")
    assert "--ledger-path" in mpi_primary["primary_null"]["command"]
    secondary = next(cell for cell in plan["cells"] if cell["family"] == "small_cnn" and cell["seed"] == 173)
    assert secondary["analysis_roles"]["primary_full_retrained_null"] is False
    assert secondary["analysis_roles"]["secondary_group_swap_replicates"] == 9999
    assert all("sha256" in record for record in plan["runner_code_hashes"].values())


def test_cell_config_keeps_resnet_alias_and_logistic_train_only_settings(tmp_path: Path) -> None:
    cell = {
        "cell_id": "mpi__resnet18_scratch_gn__flair__seed73",
        "cohort": "mpi",
        "family": "resnet18_scratch_gn",
        "model": "resnet18",
        "modalities": ["flair"],
        "seed": 73,
        "output_dir": str(tmp_path / "fit"),
        "manifests": {split: str(tmp_path / f"{split}.jsonl") for split in ("train", "val", "test")},
    }
    base = {"training": {"batch_size": 32}}
    resnet = _cell_config(cell, base_config=base, device="cuda")
    assert resnet["training"]["model"] == "resnet18"
    assert resnet["training"]["learning_rate"] == pytest.approx(3.0e-4)
    assert resnet["training"]["modalities"] == ["flair"]

    logistic_cell = dict(cell, family="logistic", model="statistical_logistic")
    logistic = _cell_config(logistic_cell, base_config=base, device="cpu")
    assert logistic["training"]["model"] == "statistical_logistic"
    assert logistic["training"]["learning_rate"] is None
    assert logistic["training"]["weight_decay"] == 0.0
    assert logistic["training"]["logistic_C"] == 1.0
    assert logistic["training"]["logistic_solver"] == "lbfgs"
    assert logistic["training"]["train_only_standardizer"] is True
    assert logistic["training"]["batch_size"] == 32

    primary_cell = dict(cell)
    primary_cell.update(
        {
            "cell_id": "fomo45k__small_cnn__flair_t1_t2__seed73",
            "family": "small_cnn",
            "model": "small_cnn",
            "modalities": ["flair", "t1", "t2"],
            "analysis_roles": {"primary_observed_heldout_pair_swap_replicates": 1000},
        }
    )
    primary_config = _cell_config(primary_cell, base_config=base, device="cuda")
    assert primary_config["training"]["swap_replicates"] == 1000
    secondary_config = _cell_config(cell, base_config=base, device="cuda")
    assert secondary_config["training"]["swap_replicates"] == 0


def test_control_requirements_bind_actual_architecture_and_shared_negative_input(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    single = _cell_control_requirements(
        build_root,
        cohort="mpi",
        family="resnet18_scratch_gn",
        modalities=("flair",),
    )
    assert single[0]["expected_identity"]["model"] == "resnet18"
    assert single[0]["expected_identity"]["modalities"] == ["flair"]
    assert single[1]["expected_identity"]["modalities"] == ["flair"]
    # The preregistered negative control is one joint three-channel fit per
    # cohort/architecture, shared by the four input cells.
    assert single[2]["expected_identity"]["model"] == "resnet18"
    assert single[2]["expected_identity"]["modalities"] == list(("flair", "t1", "t2"))
    assert single[2]["expected_identity"]["cohort"] == "mpi"


def test_control_identity_rejects_gate_or_manifest_binding_drift(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    manifest_root = build_root / "control_manifests"
    manifest_root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (manifest_root / f"{split}.jsonl").write_text(
            "{\"participant_id\": \"fomo45k:healthy\", \"pair_id\": \"pair:0\", \"label\": 0}\n"
            "{\"participant_id\": \"brats21:case\", \"pair_id\": \"pair:0\", \"label\": 1}\n",
            encoding="utf-8",
        )
    gate_path = build_root / "control" / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate = {
        "mode": "tiny",
        "gate": {"name": "tiny_overfit", "status": "PASS"},
        "control_manifest": {"manifest_dir": str(manifest_root)},
        "run_fingerprint": "frozen-control-v1",
        "results": [
            {
                "config": {"model": "resnet18", "modalities": ["flair"], "stage": "final"},
                "test_predictions": [
                    {"label": 0, "participant_id": "fomo45k:healthy"},
                    {"label": 1, "participant_id": "brats21:case"},
                ],
            }
        ],
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    requirement = _control_requirement(
        "tiny",
        gate_path,
        expected_identity={
            "mode": "tiny",
            "gate_name": "tiny_overfit",
            "model": "resnet18",
            "modalities": ["flair"],
            "stage": "final",
            "cohort": "fomo45k",
        },
        source_build_root=build_root,
    )
    assert requirement["binding_capture_status"] == "PASS"
    assert _control_identity(gate_path, requirement)["status"] == "PASS"
    changed = dict(gate)
    changed["run_fingerprint"] = "changed-control"
    gate_path.write_text(json.dumps(changed), encoding="utf-8")
    drift = _control_identity(gate_path, requirement)
    assert drift["status"] == "MISMATCH"
    assert "run_fingerprint" in drift["failures"]


def test_mixed_negative_summary_aliases_bind_to_registry_identity(tmp_path: Path) -> None:
    """The dedicated Mixed helper's common manifest schema is registry-ready."""

    build_root = tmp_path / "build"
    manifest_root = build_root / "stage_b_controls" / "mixed" / "small_cnn" / "negative" / "common"
    manifest_root.mkdir(parents=True)
    rows_by_split = {
        "train": [
            {"participant_id": "fomo45k:h0", "pair_id": "negative:mixed:fomo45k:train:pair0", "label": 0},
            {"participant_id": "fomo45k:h1", "pair_id": "negative:mixed:fomo45k:train:pair0", "label": 1},
        ],
        "val": [
            {"participant_id": "mpi:h0", "pair_id": "negative:mixed:mpi:val:pair0", "label": 0},
            {"participant_id": "mpi:h1", "pair_id": "negative:mixed:mpi:val:pair0", "label": 1},
        ],
        "test": [
            {"participant_id": "oasis3:h0", "pair_id": "negative:mixed:oasis3:test:pair0", "label": 0},
            {"participant_id": "oasis3:h1", "pair_id": "negative:mixed:oasis3:test:pair0", "label": 1},
        ],
    }
    for split, rows in rows_by_split.items():
        (manifest_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    gate_path = build_root / "stage_b_controls" / "mixed" / "small_cnn" / "negative" / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate = {
        "mode": "negative",
        "comparison": "mixed",
        "run_fingerprint": "mixed-frozen-v3",
        "gate": {"name": "negative_or_shuffle", "status": "PASS"},
        # These are the aliases emitted by the dedicated Mixed helper.  The
        # registry must bind this same common manifest for all three fits.
        "control_manifest": {
            "manifest_dir": str(manifest_root),
            "manifest_hashes": {},
            "reused_across_fit_seeds": True,
        },
        "results": [
            {
                "config": {
                    "model": "small_cnn",
                    "modalities": ["flair", "t1", "t2"],
                    "stage": "final",
                },
                "test_predictions": [
                    {"label": 0, "participant_id": "oasis3:h0"},
                    {"label": 1, "participant_id": "oasis3:h1"},
                ],
            }
        ],
    }
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    requirement = _control_requirement(
        "negative__small_cnn__flair_t1_t2__mixed",
        gate_path,
        expected_identity={
            "mode": "negative",
            "gate_name": "negative_or_shuffle",
            "model": "small_cnn",
            "modalities": ["flair", "t1", "t2"],
            "stage": "final",
            "cohort": "mixed",
        },
        source_build_root=build_root,
    )
    assert requirement["binding_capture_status"] == "PASS"
    binding = _control_identity(gate_path, requirement)
    assert binding["status"] == "PASS", binding
    assert binding["checks"]["mode"]["status"] == "PASS"
    assert binding["checks"]["cohort"]["status"] == "PASS"
    assert binding["checks"]["control_manifest_dir"]["status"] == "PASS"


def test_plan_artifacts_are_json_yaml_and_written_per_cell(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    plan = build_matrix_plan(_protocol(), amendment=_amendment(), build_root=build_root, output_root=tmp_path / "matrix")
    plan_path = tmp_path / "matrix" / "matrix_plan.json"
    write_plan_artifacts(plan, base_config={"training": {}}, plan_path=plan_path)
    assert json.loads(plan_path.read_text(encoding="utf-8"))["fit_count"] == 112
    configs = list((tmp_path / "matrix" / "cells").glob("*/config.yaml"))
    assert len(configs) == 112
    assert json.loads(configs[0].read_text(encoding="utf-8"))["matrix_cell"]["cell_id"]


def test_execution_gate_blocks_incomplete_null_without_historical_shuffle(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    stage = build_root / "stage_a_fomo45k_observed_v3_20260917"
    (build_root / "audits").mkdir(parents=True)
    (stage / "immutable_test_label_sanity").mkdir(parents=True)
    (stage / "retrained_null").mkdir(parents=True)
    (build_root / "build_summary.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    for cohort in COHORTS:
        (build_root / "audits" / f"{cohort}.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    for path in (
        build_root / "stage_a_fomo45k" / "tiny" / "tiny" / "gate.json",
        build_root / "stage_a_fomo45k_acl_retry_20260917" / "positive" / "gate.json",
        build_root / "stage_a_fomo45k_negative_v3_20260917" / "negative" / "gate.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"gate": {"status": "PASS"}}), encoding="utf-8")
    (stage / "input_binding_audit.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (stage / "immutable_test_label_sanity" / "gate_random_test_labels.json").write_text(
        json.dumps({"status": "INCONCLUSIVE"}), encoding="utf-8"
    )
    (stage / "retrained_null" / "status.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    (stage / "retrained_null" / "summary.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    gate = execution_gate(build_root=build_root, stage_a_root=stage, primary_null_root=stage)
    assert gate["status"] == "BLOCKED"
    assert "primary_null_status" in gate["failures"]
    assert "primary_null_integrity" in gate["failures"]
    assert gate["historical_shuffle_gate_required"] is False


def test_execution_gate_rejects_missing_heldout_sanity(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    stage = build_root / "stage"
    stage.mkdir(parents=True)
    gate = execution_gate(build_root=build_root, stage_a_root=stage, primary_null_root=stage)
    assert gate["status"] == "BLOCKED"
    assert "heldout_label_sanity" in gate["failures"]


def test_cell_gate_rehashes_manifests_and_blocks_missing_neural_controls(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    for name in ("protocol.json", "source_fingerprints.json", "build_summary.json"):
        (build_root / name).write_text("{}", encoding="utf-8")
    plan = build_matrix_plan(
        _protocol(), amendment=_amendment(), build_root=build_root, output_root=tmp_path / "matrix", device="cpu"
    )
    write_plan_artifacts(plan, base_config={"training": {}}, plan_path=tmp_path / "matrix" / "matrix_plan.json")
    logistic = next(cell for cell in plan["cells"] if cell["cohort"] == "mpi" and cell["family"] == "logistic")
    assert _cell_execution_gate(logistic, plan)["status"] == "PASS"
    neural = next(
        cell
        for cell in plan["cells"]
        if cell["cohort"] == "mpi" and cell["family"] == "small_cnn" and cell["seed"] == 73
    )
    check = _cell_execution_gate(neural, plan)
    assert check["status"] == "PENDING"
    assert {"control:tiny", "control:positive", "control:negative"}.issubset(check["failures"])
    manifest = Path(neural["manifests"]["train"])
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    drifted = _cell_execution_gate(logistic, plan)
    assert "manifest:train" in drifted["failures"]


def test_execute_plan_skips_pending_cell_and_runs_ready_cell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    for name in ("protocol.json", "source_fingerprints.json", "build_summary.json"):
        (build_root / name).write_text("{}", encoding="utf-8")
    plan = build_matrix_plan(
        _protocol(), amendment=_amendment(), build_root=build_root, output_root=tmp_path / "matrix", device="cpu"
    )
    write_plan_artifacts(plan, base_config={"training": {}}, plan_path=tmp_path / "matrix" / "matrix_plan.json")
    ready = next(cell for cell in plan["cells"] if cell["cohort"] == "mpi" and cell["family"] == "logistic")
    pending = next(cell for cell in plan["cells"] if cell["cohort"] == "mpi" and cell["family"] == "small_cnn")
    monkeypatch.setattr(
        "scripts.run_domain_classifier_matrix_v3.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    results = execute_plan(plan, env={}, cell_ids=[ready["cell_id"], pending["cell_id"]])
    assert [result["status"] for result in results] == ["PASS", "PENDING"]
    assert results[1]["skipped"] is True


def test_protocol_rejects_wrong_input_count_or_family_seed(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_manifests(build_root)
    wrong = _protocol()
    wrong["matrix"] = dict(wrong["matrix"], total_fits=112, input_types=["flair", "t1"])
    with pytest.raises(MatrixProtocolError, match="input types"):
        build_matrix_plan(wrong, build_root=build_root, output_root=tmp_path / "matrix")
