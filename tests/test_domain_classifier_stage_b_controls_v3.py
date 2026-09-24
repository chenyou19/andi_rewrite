import json
from pathlib import Path

from scripts.plan_domain_classifier_stage_b_controls_v3 import (
    ARCHITECTURES,
    COHORTS,
    FIT_SEEDS,
    INPUT_TYPES,
    append_control_completion_event,
    build_stage_b_plan,
)


def _write_manifest_root(root: Path, prefix: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [
            {"participant_id": f"{prefix}:healthy", "pair_id": f"{prefix}:{split}:pair", "label": 0, "z": 10},
            {"participant_id": "brats21:case", "pair_id": f"{prefix}:{split}:pair", "label": 1, "z": 10},
        ]
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _write_build(tmp_path: Path) -> tuple[Path, Path]:
    build_root = tmp_path / "build"
    for cohort in COHORTS:
        _write_manifest_root(build_root / "manifests" / cohort, cohort)
    _write_manifest_root(build_root / "fomo45k_registered_paths_v1" / "manifests" / "fomo45k", "fomo45k")
    for name in (
        "protocol.json",
        "source_fingerprints.json",
        "build_summary.json",
        "training_protocol.json",
        "training_protocol_amendment_20260917.json",
    ):
        (build_root / name).write_text("{}", encoding="utf-8")
    config = tmp_path / "base.yaml"
    config.write_text("training:\n  model: small_cnn\n  widths: [32, 64, 128, 128]\n", encoding="utf-8")
    return build_root, config


def test_stage_b_plan_freezes_24_jobs_and_40_fits_without_training(tmp_path: Path) -> None:
    build_root, config = _write_build(tmp_path)
    plan = build_stage_b_plan(
        build_root=build_root,
        plan_root=tmp_path / "stage_b_plan",
        base_config_path=config,
        device="cuda",
    )
    assert plan["status"] == "PLANNED_NO_TRAINING"
    assert plan["training_started"] is False
    assert plan["counts"] == {
        "tiny_jobs": 8,
        "tiny_fits": 8,
        "positive_jobs": 8,
        "positive_fits": 8,
        "negative_jobs": 8,
        "negative_fits": 24,
        "jobs": 24,
        "fits": 40,
        "reused_jobs": 0,
        "reused_fits": 0,
        "planned_jobs": 24,
        "planned_fits": 40,
    }
    assert len(plan["jobs"]) == 24
    assert plan["source_freeze"]["status"] == "PASS"
    assert all(job["environment"]["OMP_NUM_THREADS"] == "1" for job in plan["jobs"])
    assert all(job["environment"]["MKL_NUM_THREADS"] == "1" for job in plan["jobs"])
    assert all(job["command"][0].endswith(r"miniconda3\envs\ANDi\python.exe") for job in plan["jobs"])
    tiny = [job for job in plan["jobs"] if job["mode"] == "tiny"]
    positive = [job for job in plan["jobs"] if job["mode"] == "positive"]
    negative = [job for job in plan["jobs"] if job["mode"] == "negative"]
    assert len(tiny) == len(positive) == 8
    assert len(negative) == 8
    assert all(job["fit_seeds"] == [73] for job in tiny + positive)
    assert all(job["fit_seeds"] == list(FIT_SEEDS) for job in negative)
    assert all(job["control_input_modalities"] == ["flair", "t1", "t2"] for job in negative)
    assert all(job["environment"]["requires_elevated_raw_read"] for job in positive)
    assert all("--fit-positive-shared-scalar" in job["command"] for job in positive)
    assert all("--tiny-max-slices" in job["command"] for job in tiny)
    mixed_negative = [job for job in negative if job["source_cohort"] == "mixed"]
    assert len(mixed_negative) == 2
    assert all(
        str(job["command"][1]).endswith("run_domain_classifier_mixed_negative_control_v3.py")
        for job in mixed_negative
    )
    assert all(job["implementation"] == "mixed_source_stratified_v3" for job in mixed_negative)
    assert all(job["command"][-6:] == ["--split-seed", "73", "--seeds", "73", "173", "273"] for job in mixed_negative)
    assert len((tmp_path / "stage_b_plan" / "control_registry.jsonl").read_text(encoding="utf-8").splitlines()) == 24


def test_stage_b_plan_refuses_existing_root_and_has_all_arch_input_coordinates(tmp_path: Path) -> None:
    build_root, config = _write_build(tmp_path)
    plan_root = tmp_path / "stage_b_plan"
    plan = build_stage_b_plan(build_root=build_root, plan_root=plan_root, base_config_path=config)
    coordinates = {
        (job["mode"], job["family"], tuple(job["modalities"]))
        for job in plan["jobs"]
        if job["mode"] in {"tiny", "positive"}
    }
    assert coordinates == {
        (mode, family, modalities)
        for mode in ("tiny", "positive")
        for family in ARCHITECTURES
        for modalities in INPUT_TYPES
    }
    # A repeated invocation cannot refresh a frozen source/config binding.
    import pytest

    with pytest.raises(RuntimeError, match="already contains files"):
        build_stage_b_plan(build_root=build_root, plan_root=plan_root, base_config_path=config)


def test_completion_registry_appends_verified_mixed_binding_once(tmp_path: Path) -> None:
    build_root, config = _write_build(tmp_path)
    plan_root = tmp_path / "stage_b_plan"
    plan = build_stage_b_plan(build_root=build_root, plan_root=plan_root, base_config_path=config)
    job = next(
        item
        for item in plan["jobs"]
        if item["job_id"] == "negative__small_cnn__flair_t1_t2__mixed"
    )
    manifest_root = build_root / "stage_b_controls" / "mixed" / "small_cnn" / "negative" / "common"
    _write_manifest_root(manifest_root, "mixed")
    gate_path = Path(job["expected_gate_path"])
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {
                "mode": "negative",
                "comparison": "mixed",
                "run_fingerprint": "mixed-negative-frozen-v3",
                "gate": {"name": "negative_or_shuffle", "status": "PASS"},
                "control_manifest": {"manifest_dir": str(manifest_root)},
                "results": [
                    {
                        "config": {
                            "model": "small_cnn",
                            "modalities": ["flair", "t1", "t2"],
                            "stage": "final",
                        }
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry_path = plan_root / "control_registry.jsonl"
    event = append_control_completion_event(
        registry_path,
        job=job,
        gate_path=gate_path,
        build_root=build_root,
    )
    assert event["event"] == "COMPLETED_BOUND"
    assert event["binding_status"] == "BOUND"
    rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 25
    assert rows[-1]["job_id"] == job["job_id"]
    import pytest

    with pytest.raises(RuntimeError, match="already recorded"):
        append_control_completion_event(
            registry_path,
            job=job,
            gate_path=gate_path,
            build_root=build_root,
        )
