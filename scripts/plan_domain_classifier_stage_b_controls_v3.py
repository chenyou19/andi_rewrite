"""Freeze executable Stage-B control jobs without starting training.

The v3 matrix has 112 formal cells, but neural cells require capability
controls before they can be launched.  This plan materializes those control
jobs and their exact manifests/configs/commands while keeping execution
opt-in outside this script:

* eight FOMO tiny jobs (four inputs by SmallCNN/ResNet18),
* eight FOMO registered-positive jobs with the train-only shared scalar, and
* eight same-cohort negative jobs (four cohorts by two architectures), each
  fitting seeds 73, 173, and 273.

The resulting 24 jobs represent 40 fits.  Existing FOMO joint SmallCNN
artifacts are bound as explicit reuse records.  The accidentally broad tiny
run (seeds 73/173/273) is retained as a historical diagnostic; only its
predeclared seed-73 result is reused for the Stage-B capability cell.

This command only writes a frozen job registry and source snapshot.  It never
invokes a training subprocess.  Later execution must validate the frozen
source hashes and, for registered jobs, run in the elevated raw-reader
context.  All command entries use the ANDi interpreter and the threaded
controls wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - the ANDi environment pins PyYAML
    raise RuntimeError("The Stage-B planner requires PyYAML.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
ANDI_PYTHON = Path(r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe")
BUILD_ROOT_DEFAULT = (
    REPO_ROOT
    / "outputs"
    / "diagnostics"
    / "domain_classifier"
    / "model_grid_v3_fullcandidate_20260917_final"
)
PLAN_ROOT_DEFAULT = BUILD_ROOT_DEFAULT / "stage_b_control_jobs_v1_20260917"
COHORTS = ("fomo45k", "mpi", "oasis3", "mixed")
INPUT_TYPES: tuple[tuple[str, ...], ...] = (
    ("flair",),
    ("t1",),
    ("t2",),
    ("flair", "t1", "t2"),
)
JOINT_MODALITIES = ("flair", "t1", "t2")
ARCHITECTURES = ("small_cnn", "resnet18_scratch_gn")
FIT_SEEDS = (73, 173, 273)
SPLITS = ("train", "val", "test")
ACTUAL_MODELS = {"small_cnn": "small_cnn", "resnet18_scratch_gn": "resnet18"}


class StageBPlanError(RuntimeError):
    """Raised when a Stage-B job cannot be frozen fail-closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_record(path: Path) -> dict[str, Any]:
    """Return a fail-closed path/hash record for completion artifacts."""

    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "status": "MISSING", "sha256": None}
    return {"path": str(resolved), "status": "PASS", "sha256": _sha256(resolved)}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_payload(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_json(path: Path, value: Any, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise StageBPlanError(f"Refusing to overwrite existing frozen artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json_payload(value), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_text(path: Path, payload: str, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise StageBPlanError(f"Refusing to overwrite existing frozen artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise StageBPlanError(f"Cannot read YAML config {path}") from exc
    if not isinstance(value, dict):
        raise StageBPlanError(f"Config {path} must contain a mapping.")
    return value


def _count_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "status": "MISSING", "rows": 0, "sha256": None}
    rows = 0
    participants: set[str] = set()
    pairs: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StageBPlanError(f"Invalid manifest JSONL {path}:{line_number}") from exc
                if not isinstance(value, Mapping):
                    raise StageBPlanError(f"Manifest row {path}:{line_number} is not an object.")
                participant = str(value.get("participant_id", "")).strip()
                pair = str(value.get("pair_id", "")).strip()
                if not participant or not pair:
                    raise StageBPlanError(f"Manifest row {path}:{line_number} has empty IDs.")
                rows += 1
                participants.add(participant)
                pairs.add(pair)
    except OSError as exc:
        raise StageBPlanError(f"Cannot read manifest {path}") from exc
    return {
        "path": str(path.resolve()),
        "status": "PASS",
        "rows": rows,
        "unique_participants": len(participants),
        "unique_pairs": len(pairs),
        "sha256": _sha256(path),
    }


def _manifest_bindings(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result = {split: _count_manifest(root / f"{split}.jsonl") for split in SPLITS}
    if any(item["status"] != "PASS" for item in result.values()):
        missing = [split for split, item in result.items() if item["status"] != "PASS"]
        raise StageBPlanError(f"Manifest root {root} is incomplete: {missing}")
    return {"root": str(root), "splits": result}


def append_control_completion_event(
    registry_path: Path,
    *,
    job: Mapping[str, Any],
    gate_path: Path,
    build_root: Path,
    result_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Append one verified completion event to a frozen Stage-B registry.

    The initial registry event and frozen plan are never rewritten.  A
    completed gate is accepted only when its declared mode/cohort/model/input
    identity agrees with the planned job, its common manifest is complete and
    inside the frozen build root, and every manifest hash is recomputed at the
    completion boundary.  This is the binding step consumed by a later matrix
    plan; an unfamiliar PASS artifact cannot self-register.

    ``result_paths`` is optional because the helper's aggregate gate already
    embeds per-seed result metadata.  When supplied, those paths are hashed
    into the append-only event as additional provenance.
    """

    registry_path = registry_path.resolve()
    gate_path = gate_path.resolve()
    build_root = build_root.resolve()
    if not registry_path.is_file():
        raise StageBPlanError(f"Frozen control registry is missing: {registry_path}")
    expected_job_id = str(job.get("job_id", "")).strip()
    if not expected_job_id:
        raise StageBPlanError("Cannot bind a completion without a job_id.")
    expected_gate = Path(str(job.get("expected_gate_path", ""))).resolve()
    if expected_gate != gate_path:
        raise StageBPlanError(
            f"Completion gate does not match frozen job path: {gate_path} != {expected_gate}"
        )
    try:
        gate_value = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPlanError(f"Cannot read completed gate {gate_path}") from exc
    if not isinstance(gate_value, Mapping):
        raise StageBPlanError(f"Completed gate is not an object: {gate_path}")
    nested_gate = gate_value.get("gate") if isinstance(gate_value.get("gate"), Mapping) else {}
    gate_status = str(nested_gate.get("status") or gate_value.get("status") or "").strip()
    gate_name = str(nested_gate.get("name") or "").strip()
    expected_identity = job.get("expected_identity") if isinstance(job.get("expected_identity"), Mapping) else {}
    failures: list[str] = []
    if gate_status not in {"PASS", "INCONCLUSIVE", "FAIL"}:
        failures.append("gate_status")
    if expected_identity.get("gate_name") is not None and gate_name != str(expected_identity["gate_name"]):
        failures.append("gate_name")
    expected_mode = str(expected_identity.get("mode", "")).strip().lower()
    actual_mode = str(gate_value.get("mode", "")).strip().lower()
    if expected_mode and actual_mode != expected_mode:
        failures.append("mode")
    expected_cohort = str(expected_identity.get("cohort", "")).strip().lower()
    declared_cohorts = {
        str(gate_value.get(key, "")).strip().lower()
        for key in ("cohort", "comparison", "source_cohort")
        if str(gate_value.get(key, "")).strip()
    }
    if expected_cohort and expected_cohort not in declared_cohorts:
        failures.append("cohort")
    expected_model = str(expected_identity.get("model", "")).strip()
    expected_modalities = tuple(str(value) for value in expected_identity.get("modalities", ()))
    expected_stage = str(expected_identity.get("stage", "")).strip()
    results = gate_value.get("results")
    if not isinstance(results, list) or not results:
        failures.append("results")
        results = []
    result_checks: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        config = result.get("config") if isinstance(result, Mapping) and isinstance(result.get("config"), Mapping) else {}
        item = {
            "index": index,
            "model": config.get("model"),
            "modalities": list(config.get("modalities", ())),
            "stage": config.get("stage"),
        }
        result_checks.append(item)
        if expected_model and str(config.get("model", "")) != expected_model:
            failures.append(f"results[{index}].model")
        if expected_modalities and tuple(config.get("modalities", ())) != expected_modalities:
            failures.append(f"results[{index}].modalities")
        if expected_stage and str(config.get("stage", "")) != expected_stage:
            failures.append(f"results[{index}].stage")
    control_manifest = gate_value.get("control_manifest")
    if not isinstance(control_manifest, Mapping):
        control_manifest = gate_value.get("common_control_manifest")
    manifest_dir_value = control_manifest.get("manifest_dir") if isinstance(control_manifest, Mapping) else None
    if not manifest_dir_value:
        failures.append("control_manifest.manifest_dir")
        manifest_binding = {"status": "MISSING", "root": None, "splits": {}}
    else:
        manifest_root = Path(str(manifest_dir_value)).resolve()
        try:
            manifest_root.relative_to(build_root)
            in_build = True
        except ValueError:
            in_build = False
        if not in_build:
            failures.append("control_manifest.root")
            manifest_binding = {"status": "MISMATCH", "root": str(manifest_root), "splits": {}}
        else:
            try:
                manifest_binding = _manifest_bindings(manifest_root)
                manifest_binding["status"] = "PASS"
            except StageBPlanError:
                failures.append("control_manifest.files")
                manifest_binding = {"status": "INVALID", "root": str(manifest_root), "splits": {}}
    try:
        registry_lines = [
            json.loads(line)
            for line in registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPlanError(f"Cannot read frozen control registry {registry_path}") from exc
    matching_events = [
        event
        for event in registry_lines
        if isinstance(event, Mapping) and str(event.get("job_id", "")) == expected_job_id
    ]
    if not matching_events:
        raise StageBPlanError(f"Job {expected_job_id!r} is absent from frozen registry.")
    if any(str(event.get("event", "")).upper().startswith("COMPLETED") for event in matching_events):
        raise StageBPlanError(f"Completion for {expected_job_id!r} is already recorded; refusing duplicate append.")
    if failures:
        raise StageBPlanError(
            f"Completed gate identity does not match frozen job {expected_job_id}: {', '.join(sorted(set(failures)))}"
        )
    result_hashes: dict[str, Any] = {}
    for name, path in (result_paths or {}).items():
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(build_root)
        except ValueError as exc:
            raise StageBPlanError(f"Result artifact is outside build root: {resolved}") from exc
        result_hashes[str(name)] = _hash_record(resolved)
        if result_hashes[str(name)]["status"] != "PASS":
            raise StageBPlanError(f"Result artifact is missing: {resolved}")
    event = {
        "event": "COMPLETED_BOUND" if gate_status == "PASS" else "COMPLETED_UNBOUND",
        "job_id": expected_job_id,
        "status": gate_status,
        "binding_status": "BOUND" if gate_status == "PASS" else "NOT_BOUND",
        "gate_path": str(gate_path),
        "gate_sha256": _sha256(gate_path),
        "run_fingerprint": gate_value.get("run_fingerprint"),
        "expected_identity": dict(expected_identity),
        "result_identity_checks": result_checks,
        "control_manifest": manifest_binding,
        "result_hashes": result_hashes,
        "source_freeze_sha256": job.get("source_freeze", {}).get("sha256") if isinstance(job.get("source_freeze"), Mapping) else None,
        "base_manifest_sha256": {
            split: job.get("manifest_binding", {}).get("splits", {}).get(split, {}).get("sha256")
            for split in SPLITS
        },
    }
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    return event


def _job_id(mode: str, family: str, modalities: Sequence[str], cohort: str | None = None) -> str:
    input_name = "_".join(modalities)
    prefix = f"{mode}__{family}__{input_name}"
    return f"{prefix}__{cohort}" if cohort is not None else prefix


def _source_files(build_root: Path, plan_root: Path) -> list[tuple[str, Path]]:
    """Return code/config/map files whose bytes are copied into the freeze."""

    repo_files: set[Path] = {
        REPO_ROOT / "configs" / "domain_classifier.yaml",
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "metrics.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
        REPO_ROOT / "domain_classifier" / "v3_runtime.py",
        REPO_ROOT / "scripts" / "run_domain_classifier.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_controls.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_controls_threaded.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_controls_v3.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_mixed_negative_control_v3.py",
        REPO_ROOT / "scripts" / "audit_v3_cached_input_binding.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_primary_null_v3.py",
        REPO_ROOT / "scripts" / "run_domain_classifier_matrix_v3.py",
        REPO_ROOT / "scripts" / "plan_domain_classifier_stage_b_controls_v3.py",
    }
    # Freeze all local Python dependencies used by the manifest reader and
    # canonical image adapters without copying MRI/LMDB/NPZ payloads.
    for root_name in ("data",):
        root = REPO_ROOT / root_name
        if root.is_dir():
            repo_files.update(path for path in root.rglob("*.py") if path.is_file())
    build_files = {
        build_root / "protocol.json",
        build_root / "source_fingerprints.json",
        build_root / "build_summary.json",
        build_root / "training_protocol.json",
        build_root / "training_protocol_amendment_20260917.json",
        REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "split_maps.json",
        REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_protocol_survey" / "survey.json",
        REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_protocol_survey" / "survey.md",
    }
    entries: list[tuple[str, Path]] = []
    for path in sorted(repo_files):
        entries.append((str(Path("repo") / path.relative_to(REPO_ROOT)), path))
    for path in sorted(build_files):
        if path.is_relative_to(build_root):
            relative = Path("build_root") / path.relative_to(build_root)
        else:
            relative = Path("diagnostics") / path.relative_to(REPO_ROOT / "outputs" / "diagnostics")
        entries.append((str(relative), path))
    # A future invocation must never accidentally include its own prior
    # output.  The planner source is a repository file and is intentionally
    # copied above; generated plan bytes stay outside this list.
    del plan_root
    return entries


def _freeze_sources(build_root: Path, freeze_root: Path) -> dict[str, Any]:
    source_dir = freeze_root / "source"
    # The new controls-v3 adapter is optional until its owner lands it; when
    # present it is copied into the next freeze.  Older plans remain
    # immutable, and the legacy controls implementation is already captured
    # by the existing scripts below.
    optional_sources = {
        (REPO_ROOT / "scripts" / "run_domain_classifier_controls_v3.py").resolve(),
    }
    records: list[dict[str, Any]] = []
    for relative, source in _source_files(build_root, freeze_root.parent):
        target = source_dir / relative
        if not source.is_file():
            records.append({"source": str(source.resolve()), "snapshot": str(target), "status": "MISSING"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_sha = _sha256(source)
        target_sha = _sha256(target)
        records.append(
            {
                "source": str(source.resolve()),
                "snapshot": str(target.resolve()),
                "status": "PASS" if source_sha == target_sha else "MISMATCH",
                "source_sha256": source_sha,
                "snapshot_sha256": target_sha,
            }
        )
    critical_prefixes = (
        str((REPO_ROOT / "configs" / "domain_classifier.yaml").resolve()),
        str((REPO_ROOT / "domain_classifier").resolve()),
        str((REPO_ROOT / "scripts" / "run_domain_classifier_controls.py").resolve()),
        str((REPO_ROOT / "scripts" / "run_domain_classifier_controls_threaded.py").resolve()),
        str((REPO_ROOT / "domain_classifier" / "v3_runtime.py").resolve()),
        str((REPO_ROOT / "scripts" / "run_domain_classifier_primary_null_v3.py").resolve()),
        str((build_root / "training_protocol.json").resolve()),
    )
    critical_missing = [
        record["source"]
        for record in records
        if Path(record["source"]).resolve() not in optional_sources
        if record["status"] != "PASS"
        and any(record["source"] == prefix or record["source"].startswith(prefix + os.sep) for prefix in critical_prefixes)
    ]
    optional_missing = [
        record["source"]
        for record in records
        if Path(record["source"]).resolve() in optional_sources and record["status"] != "PASS"
    ]
    status = "PASS" if not critical_missing and all(record["status"] == "PASS" for record in records if "source_sha256" in record) else "INCOMPLETE"
    manifest = {
        "schema_version": 1,
        "protocol": "model_grid_v3_stage_b_source_freeze_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(REPO_ROOT),
        "build_root": str(build_root.resolve()),
        "freeze_root": str(freeze_root.resolve()),
        "critical_missing": critical_missing,
        "optional_missing": optional_missing,
        "records": records,
    }
    _write_json(freeze_root / "manifest.json", manifest, refuse_existing=True)
    return {
        "path": str((freeze_root / "manifest.json").resolve()),
        "sha256": _sha256(freeze_root / "manifest.json"),
        "status": status,
        "record_count": len(records),
        "critical_missing": critical_missing,
        "optional_missing": optional_missing,
    }


def _runtime_environment() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
    except Exception as exc:  # pragma: no cover - environment probe only
        torch_info = {"error": type(exc).__name__}
    return {
        "python": str(Path(sys.executable).resolve()),
        "required_python": str(ANDI_PYTHON),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch": torch_info,
        "policy": "future execution sets OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and uses controls_threaded.py",
    }


def _load_training_config(base_config_path: Path) -> dict[str, Any]:
    value = _read_yaml(base_config_path)
    training = value.get("training", value)
    if not isinstance(training, Mapping):
        training = {}
    return dict(training)


def _job_config(
    base_training: Mapping[str, Any],
    *,
    model: str,
    modalities: Sequence[str],
    stage: str,
    output_root: Path,
) -> dict[str, Any]:
    training = dict(base_training)
    learning_rate = 1.0e-3 if model == "small_cnn" else 3.0e-4
    training.update(
        {
            "model": model,
            "in_channels": len(modalities),
            "modalities": list(modalities),
            "stage": stage,
            "device": "cuda",
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-4,
            "max_epochs": 40,
            "patience": 8,
            "early_stopping": True,
            "dropout": 0.0,
            "no_augmentation": True,
            "batch_size": 32,
            "num_workers": 0,
            "threshold": 0.5,
            "split_seed": 73,
            "bootstrap_replicates": 2000,
            "swap_replicates": 1000,
            "permutation_replicates": 0,
            "permutation_mode": "none",
            "tiny": False,
            "shared_train_scalar": None,
        }
    )
    return {
        "output_dir": str(output_root.resolve()),
        "training": training,
    }


def _control_command(
    *,
    config_path: Path,
    comparison: str,
    manifest_root: Path,
    output_root: Path,
    mode: str,
    modalities: Sequence[str],
    seeds: Sequence[int],
    device: str,
) -> list[str]:
    command = [
        str(ANDI_PYTHON),
        str((REPO_ROOT / "scripts" / "run_domain_classifier_controls_threaded.py").resolve()),
        "--config",
        str(config_path.resolve()),
        "--comparison",
        str(comparison),
        "--manifest-root",
        str(manifest_root.resolve()),
        "--output-dir",
        str(output_root.resolve()),
        "--mode",
        str(mode),
        "--device",
        str(device),
        "--modalities",
        *[str(value) for value in modalities],
        "--seeds",
        *[str(int(seed)) for seed in seeds],
        "--retrained-permutations",
        "0",
    ]
    if mode == "tiny":
        command.extend(["--tiny", "--tiny-seed", "73", "--tiny-max-slices", "128", "--tiny-min-slices", "64"])
    elif mode == "positive":
        command.extend(["--stage", "registered", "--fit-positive-shared-scalar"])
    elif mode == "negative":
        command.extend(["--stage", "final", "--tiny-seed", "73"])
    return command


def _mixed_negative_command(
    *,
    config_path: Path,
    manifest_root: Path,
    output_root: Path,
    build_root: Path,
    family: str,
    device: str,
) -> list[str]:
    """Return the isolated Mixed source-stratified negative entry point.

    The ordinary controls CLI makes one global participant assignment.  A
    Mixed control must allocate source cohorts independently, so the planner
    freezes a dedicated helper command instead of silently invoking the
    global assignment path.
    """

    return [
        str(ANDI_PYTHON),
        str((REPO_ROOT / "scripts" / "run_domain_classifier_mixed_negative_control_v3.py").resolve()),
        "--config",
        str(config_path.resolve()),
        "--manifest-root",
        str(manifest_root.resolve()),
        "--output-dir",
        str(output_root.resolve()),
        "--build-root",
        str(build_root.resolve()),
        "--family",
        str(family),
        "--device",
        str(device),
        "--split-seed",
        "73",
        "--seeds",
        *[str(int(seed)) for seed in FIT_SEEDS],
    ]


def _existing_reuse(
    build_root: Path,
    *,
    mode: str,
    family: str,
    modalities: Sequence[str],
    cohort: str | None,
) -> dict[str, Any] | None:
    joint = tuple(modalities) == JOINT_MODALITIES
    if mode in {"tiny", "positive"} and family == "small_cnn" and joint:
        if mode == "tiny":
            gate = build_root / "stage_a_fomo45k" / "tiny" / "tiny" / "gate.json"
        else:
            gate = build_root / "stage_a_fomo45k_acl_retry_20260917" / "positive" / "gate.json"
        if not gate.is_file():
            return None
        return {
            "status": "REUSED_EXISTING",
            "gate_path": str(gate.resolve()),
            "gate_sha256": _sha256(gate) if gate.is_file() else None,
            "primary_seed": 73,
            "artifact_seeds": [73, 173, 273] if mode == "tiny" else [73],
            "historical_seed_exception": (
                "The initial tiny CLI omitted --seeds73 and produced diagnostic seeds 73/173/273; "
                "only predeclared seed73 is reused for this one Stage-B capability cell."
                if mode == "tiny"
                else None
            ),
        }
    if mode == "negative" and cohort == "fomo45k" and family == "small_cnn":
        gate = build_root / "stage_a_fomo45k_negative_v3_20260917" / "negative" / "gate.json"
        if not gate.is_file():
            return None
        return {
            "status": "REUSED_EXISTING",
            "gate_path": str(gate.resolve()),
            "gate_sha256": _sha256(gate) if gate.is_file() else None,
            "fit_seeds": list(FIT_SEEDS),
            "primary_seed": 73,
            "historical_seed_exception": "Existing FOMO SmallCNN negative is the reviewed 3-seed Stage-A control.",
        }
    return None


def build_stage_b_plan(
    *,
    build_root: Path,
    plan_root: Path,
    base_config_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    build_root = build_root.resolve()
    plan_root = plan_root.resolve()
    base_config_path = base_config_path.resolve()
    if plan_root.exists() and any(plan_root.iterdir()):
        raise StageBPlanError(f"Plan root already contains files; choose a new immutable root: {plan_root}")
    if Path(sys.executable).resolve() != ANDI_PYTHON.resolve():
        raise StageBPlanError(f"Stage-B planning must use {ANDI_PYTHON}; current interpreter is {sys.executable}")
    if not base_config_path.is_file():
        raise StageBPlanError(f"Missing base config: {base_config_path}")
    base_training = _load_training_config(base_config_path)
    build_protocol_paths = [
        build_root / "protocol.json",
        build_root / "source_fingerprints.json",
        build_root / "build_summary.json",
        build_root / "training_protocol.json",
        build_root / "training_protocol_amendment_20260917.json",
    ]
    missing_build = [str(path) for path in build_protocol_paths if not path.is_file()]
    if missing_build:
        raise StageBPlanError("Missing frozen build artifacts: " + "; ".join(missing_build))
    source_freeze = _freeze_sources(build_root, plan_root / "source_freeze_v1")
    if source_freeze["status"] != "PASS":
        raise StageBPlanError("Source freeze is incomplete: " + "; ".join(source_freeze["critical_missing"]))
    final_manifests = {
        cohort: _manifest_bindings(build_root / "manifests" / cohort)
        for cohort in COHORTS
    }
    registered_root = build_root / "fomo45k_registered_paths_v1" / "manifests" / "fomo45k"
    registered_manifests = _manifest_bindings(registered_root)
    jobs: list[dict[str, Any]] = []
    job_configs: dict[str, str] = {}

    def add_job(
        *,
        mode: str,
        family: str,
        modalities: Sequence[str],
        source_cohort: str,
        manifest_root: Path,
        output_root: Path,
        fit_seeds: Sequence[int],
        expected_stage: str,
        comparison_arg: str,
        applies_to_cohorts: Sequence[str] = (),
        raw_read_required: bool = False,
    ) -> None:
        job_id = _job_id(mode, family, modalities, source_cohort if mode == "negative" else None)
        if any(job["job_id"] == job_id for job in jobs):
            raise StageBPlanError(f"Duplicate Stage-B job ID: {job_id}")
        job_dir = plan_root / "jobs" / job_id
        config_path = job_dir / "config.yaml"
        config = _job_config(
            base_training,
            model=ACTUAL_MODELS[family],
            modalities=modalities,
            stage=expected_stage,
            output_root=output_root,
        )
        _write_text(config_path, _json_payload(config), refuse_existing=True)
        manifest_binding = _manifest_bindings(manifest_root)
        if mode == "negative" and source_cohort == "mixed":
            command = _mixed_negative_command(
                config_path=config_path,
                manifest_root=manifest_root,
                output_root=output_root,
                build_root=build_root,
                family=family,
                device=device,
            )
        else:
            command = _control_command(
                config_path=config_path,
                comparison=comparison_arg,
                manifest_root=manifest_root,
                output_root=output_root,
                mode=mode,
                modalities=modalities,
                seeds=fit_seeds,
                device=device,
            )
        reuse = _existing_reuse(
            build_root,
            mode=mode,
            family=family,
            modalities=modalities,
            cohort=source_cohort if mode == "negative" else None,
        )
        bound_gate_path = (
            Path(str(reuse["gate_path"])).resolve()
            if reuse is not None and reuse.get("gate_path")
            else (output_root / mode / "gate.json").resolve()
        )
        expected_identity = {
            "mode": mode,
            "gate_name": {
                "tiny": "tiny_overfit",
                "positive": "positive_registered",
                "negative": "negative_or_shuffle",
            }[mode],
            "model": ACTUAL_MODELS[family],
            "modalities": list(JOINT_MODALITIES if mode == "negative" else modalities),
            "stage": expected_stage,
            "cohort": source_cohort,
        }
        job = {
            "job_id": job_id,
            "status": "REUSED_EXISTING" if reuse is not None else "PLANNED",
            "mode": mode,
            "source_cohort": source_cohort,
            "applies_to_cohorts": list(applies_to_cohorts),
            "family": family,
            "model": ACTUAL_MODELS[family],
            "modalities": list(modalities),
            "control_input_modalities": list(JOINT_MODALITIES if mode == "negative" else modalities),
            "stage": expected_stage,
            "fit_seeds": [int(seed) for seed in fit_seeds],
            "fit_count": len(tuple(fit_seeds)),
            "manifest_binding": manifest_binding,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "output_root": str(output_root.resolve()),
            "expected_gate_path": str(bound_gate_path),
            "expected_identity": expected_identity,
            "command": command,
            "environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "wrapper": (
                    "scripts/run_domain_classifier_mixed_negative_control_v3.py"
                    if mode == "negative" and source_cohort == "mixed"
                    else "scripts/run_domain_classifier_controls_threaded.py"
                ),
                "requires_elevated_raw_read": bool(raw_read_required),
            },
            "implementation": (
                "mixed_source_stratified_v3"
                if mode == "negative" and source_cohort == "mixed"
                else "controls_threaded_v3"
            ),
            "source_freeze": source_freeze,
            "reuse": reuse,
            "binding_after_execution": {
                "required": reuse is None,
                "gate_sha256": "captured after completed run",
                "run_fingerprint": "captured after completed run",
                "control_manifest_hashes": "captured after completed run",
                "matrix_planner_control_requirement": "must match expected_identity and frozen source build",
            },
        }
        jobs.append(job)
        job_configs[job_id] = str(config_path.resolve())

    # Eight FOMO capacity jobs: one for each input/architecture combination.
    for family in ARCHITECTURES:
        for modalities in INPUT_TYPES:
            add_job(
                mode="tiny",
                family=family,
                modalities=modalities,
                source_cohort="fomo45k",
                manifest_root=build_root / "manifests" / "fomo45k",
                output_root=build_root / "stage_b_fomo_controls" / family / "_".join(modalities),
                fit_seeds=(73,),
                expected_stage="final",
                comparison_arg="fomo",
                applies_to_cohorts=COHORTS,
            )
            add_job(
                mode="positive",
                family=family,
                modalities=modalities,
                source_cohort="fomo45k",
                manifest_root=registered_root,
                output_root=build_root / "stage_b_fomo_controls" / family / "_".join(modalities),
                fit_seeds=(73,),
                expected_stage="registered",
                comparison_arg="fomo",
                applies_to_cohorts=COHORTS,
                raw_read_required=True,
            )
    # Eight negative jobs produce 24 fits: each job keeps one joint input and
    # all three preregistered fit seeds.  Its result is shared by the four
    # input cells for that cohort/architecture.
    for cohort in COHORTS:
        for family in ARCHITECTURES:
            add_job(
                mode="negative",
                family=family,
                modalities=JOINT_MODALITIES,
                source_cohort=cohort,
                manifest_root=build_root / "manifests" / cohort,
                output_root=build_root / "stage_b_controls" / cohort / family,
                fit_seeds=FIT_SEEDS,
                expected_stage="final",
                comparison_arg=cohort,
                applies_to_cohorts=(cohort,),
            )
    if len(jobs) != 24:
        raise StageBPlanError(f"Expected 24 Stage-B jobs, found {len(jobs)}")
    fit_count = sum(int(job["fit_count"]) for job in jobs)
    if fit_count != 40:
        raise StageBPlanError(f"Expected 40 Stage-B fits, found {fit_count}")
    reused_jobs = [job for job in jobs if job["status"] == "REUSED_EXISTING"]
    planned_jobs = [job for job in jobs if job["status"] == "PLANNED"]
    source_build_hashes = {
        str(path.name): {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in build_protocol_paths
    }
    plan = {
        "schema_version": 1,
        "protocol_id": "model_grid_v3_stage_b_control_jobs_v1",
        "status": "PLANNED_NO_TRAINING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "executable": str(ANDI_PYTHON),
        "device": str(device),
        "build_root": str(build_root),
        "plan_root": str(plan_root),
        "base_config": {"path": str(base_config_path), "sha256": _sha256(base_config_path)},
        "source_build_hashes": source_build_hashes,
        "source_freeze": source_freeze,
        "runtime_environment_at_freeze": _runtime_environment(),
        "counts": {
            "tiny_jobs": 8,
            "tiny_fits": 8,
            "positive_jobs": 8,
            "positive_fits": 8,
            "negative_jobs": 8,
            "negative_fits": 24,
            "jobs": len(jobs),
            "fits": fit_count,
            "reused_jobs": len(reused_jobs),
            "reused_fits": sum(int(job["fit_count"]) for job in reused_jobs),
            "planned_jobs": len(planned_jobs),
            "planned_fits": sum(int(job["fit_count"]) for job in planned_jobs),
        },
        "reuse_summary": {
            "tiny_primary_seed73": 1,
            "tiny_historical_extra_seeds173_273_retained": True,
            "positive_primary_seed73": 1,
            "negative_fomo_small_cnn_three_seeds": 3,
        },
        "final_manifest_bindings": final_manifests,
        "registered_fomo_manifest_binding": registered_manifests,
        "jobs": jobs,
        "job_configs": job_configs,
        "execution_contract": {
            "plan_only": True,
            "no_training_subprocess_started": True,
            "positive_requires_elevated_raw_reader": True,
            "tiny_total_slices_target": 128,
            "tiny_total_slices_minimum": 64,
            "tiny_no_early_stopping_or_weight_decay": True,
            "negative_joint_input_only": True,
            "negative_pair_assignment_seed": 73,
            "mixed_negative_implementation": "source-stratified participant split and pseudo-label pairing; one common seed-73 manifest reused across fit seeds",
            "matrix_identity": "A later matrix plan must verify each completed gate against this registry; no unfamiliar PASS is auto-registered.",
        },
    }
    plan_payload = _json_payload(plan)
    plan_hash = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
    plan["plan_sha256"] = plan_hash
    _write_json(plan_root / "stage_b_control_plan.json", plan, refuse_existing=True)
    events = []
    for job in jobs:
        events.append(
            {
                "event": "PLANNED" if job["status"] == "PLANNED" else "REUSED_BOUND",
                "job_id": job["job_id"],
                "status": job["status"],
                "fit_count": job["fit_count"],
                "expected_gate_path": job["expected_gate_path"],
                "expected_identity": job["expected_identity"],
                "source_freeze_sha256": source_freeze["sha256"],
                "config_sha256": job["config_sha256"],
                "manifest_sha256": {
                    split: job["manifest_binding"]["splits"][split]["sha256"] for split in SPLITS
                },
            }
        )
    _write_text(
        plan_root / "control_registry.jsonl",
        "".join(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n" for event in events),
        refuse_existing=True,
    )
    readme = (
        "# Stage-B control job freeze\n\n"
        f"Plan: `{plan_root / 'stage_b_control_plan.json'}`\n\n"
        "This is a plan-only artifact; no training subprocess was started. It contains 24 jobs and 40 fits. "
        "The eight tiny and eight registered-positive jobs use FOMO manifests; the eight negative jobs use "
        "joint three-channel inputs for each cohort/architecture and fit seeds 73/173/273. Mixed uses the dedicated source-stratified helper with one seed-73 pair/selection manifest shared by the three initialization seeds.\n\n"
        "Before execution, verify every source-freeze hash and run registered jobs in the elevated raw-reader context. "
        "Existing FOMO joint SmallCNN tiny/positive/negative outputs are explicit reuse records; the accidental "
        "tiny 173/273 runs remain diagnostic and are not selected as replacements.\n"
    )
    _write_text(plan_root / "README.md", readme, refuse_existing=True)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=BUILD_ROOT_DEFAULT)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT_DEFAULT)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "domain_classifier.yaml")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_root = args.build_root if args.build_root.is_absolute() else REPO_ROOT / args.build_root
    plan_root = args.plan_root if args.plan_root.is_absolute() else REPO_ROOT / args.plan_root
    base_config = args.base_config if args.base_config.is_absolute() else REPO_ROOT / args.base_config
    plan = build_stage_b_plan(
        build_root=build_root,
        plan_root=plan_root,
        base_config_path=base_config,
        device=str(args.device),
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_root": plan["plan_root"],
                "jobs": plan["counts"]["jobs"],
                "fits": plan["counts"]["fits"],
                "source_freeze": plan["source_freeze"],
                "training_started": plan["training_started"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except StageBPlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
