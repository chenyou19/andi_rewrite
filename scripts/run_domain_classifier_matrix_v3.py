"""Freeze and, after explicit gates, execute the v3 classifier matrix.

The v3 matrix is deliberately kept in this small orchestration layer instead
of modifying the Stage-A calibration runner.  With no ``--execute`` argument
the command only validates the frozen protocol, resolves the four manifest
roots, writes one deterministic cell plan, and creates JSON-as-YAML config
files.  ``--execute`` is guarded by the completed Stage-A controls and the
completed primary null integrity audit; it then invokes the frozen-input cell
CLI one cell at a time with a fresh output directory.  The FOMO joint
SmallCNN seed-73 primary coordinate is explicitly reused from its separate
observed/199-null run and is never sent through the ordinary cell CLI.

The plan has 112 fits: one fixed-C train-only Logistic fit and three seeds for
each of SmallCNN and ResNet18-Scratch-GN, across four cohorts and four input
sets.  The family name ``resnet18_scratch_gn`` is translated to the runner's
actual ``resnet18`` model name.  The four cohort/input/family coordinates are
retained in every artifact so a result cannot be mistaken for a different
cell.  Primary full retrained pair-swap nulls are recorded as a separate
analysis requirement; this script does not substitute the older shuffle
control for those nulls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ANDI_PYTHON = Path(r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe")
COHORTS = ("fomo45k", "mpi", "oasis3", "mixed")
INPUT_TYPES: tuple[tuple[str, ...], ...] = (
    ("flair",),
    ("t1",),
    ("t2",),
    ("flair", "t1", "t2"),
)
JOINT_MODALITIES = ("flair", "t1", "t2")
FAMILY_SEEDS: dict[str, tuple[int, ...]] = {
    "logistic": (73,),
    "small_cnn": (73, 173, 273),
    "resnet18_scratch_gn": (73, 173, 273),
}
ACTUAL_MODELS = {
    "logistic": "statistical_logistic",
    "small_cnn": "small_cnn",
    "resnet18_scratch_gn": "resnet18",
}
SPLITS = ("train", "val", "test")
REQUIRED_STAGE_A_CONTROL_NAMES = ("tiny", "positive", "negative")
MATRIX_CODE_PATHS = (
    REPO_ROOT / "domain_classifier" / "models.py",
    REPO_ROOT / "domain_classifier" / "metrics.py",
    REPO_ROOT / "domain_classifier" / "runner.py",
    REPO_ROOT / "domain_classifier" / "v3_runtime.py",
    REPO_ROOT / "scripts" / "run_domain_classifier.py",
    REPO_ROOT / "scripts" / "run_domain_classifier_cell_v3.py",
    REPO_ROOT / "scripts" / "run_domain_classifier_primary_null_v3.py",
)


class MatrixProtocolError(RuntimeError):
    """Raised when a matrix plan or execution gate is not auditable."""


@dataclass(frozen=True)
class MatrixPaths:
    """Resolved immutable paths used by one cohort."""

    cohort: str
    root: Path
    manifests: dict[str, Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_record(path: Path) -> dict[str, Any]:
    """Return a fail-closed path/hash record without hiding missing inputs."""

    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "status": "MISSING", "sha256": None}
    return {"path": str(resolved), "status": "PASS", "sha256": _sha256(resolved)}


def _hash_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {str(name): _hash_record(path) for name, path in paths.items()}


def _json_read(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixProtocolError(f"Cannot read JSON artifact {path}.") from exc


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = _json_payload(value)
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _json_payload(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _protocol_matrix(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    matrix = protocol.get("matrix")
    if not isinstance(matrix, Mapping):
        raise MatrixProtocolError("training_protocol.json is missing matrix.")
    return matrix


def _normalized_inputs(value: Iterable[str] | str) -> tuple[str, ...]:
    # The frozen JSON protocol stores these as ``"flair+t1+t2"`` strings,
    # while the internal plan uses tuples so command construction is explicit.
    values = value.split("+") if isinstance(value, str) else value
    result = tuple(str(item).strip().lower() for item in values)
    if result not in INPUT_TYPES:
        raise MatrixProtocolError(f"Unsupported frozen input set: {result!r}.")
    return result


def _validate_frozen_protocol(protocol: Mapping[str, Any], amendment: Mapping[str, Any] | None = None) -> None:
    if protocol.get("protocol_id") != "model_grid_v3_training_v1":
        raise MatrixProtocolError(f"Unexpected protocol_id: {protocol.get('protocol_id')!r}.")
    matrix = _protocol_matrix(protocol)
    cohorts = tuple(str(value) for value in matrix.get("cohorts", ()))
    if cohorts != COHORTS:
        raise MatrixProtocolError(f"Frozen cohorts differ from {COHORTS!r}: {cohorts!r}.")
    inputs = tuple(_normalized_inputs(value) for value in matrix.get("input_types", ()))
    if inputs != INPUT_TYPES:
        raise MatrixProtocolError(f"Frozen input types differ from {INPUT_TYPES!r}: {inputs!r}.")
    families = matrix.get("families")
    if not isinstance(families, Mapping):
        raise MatrixProtocolError("Frozen matrix is missing families.")
    for family, expected_seeds in FAMILY_SEEDS.items():
        entry = families.get(family)
        if not isinstance(entry, Mapping):
            raise MatrixProtocolError(f"Frozen matrix is missing family {family!r}.")
        seeds = tuple(int(seed) for seed in entry.get("seeds", ()))
        count = int(entry.get("count", -1))
        if seeds != expected_seeds or count != len(expected_seeds):
            raise MatrixProtocolError(
                f"Family {family!r} must use seeds {expected_seeds!r} and count {len(expected_seeds)}; "
                f"found seeds={seeds!r}, count={count}."
            )
    if int(matrix.get("total_fits", -1)) != 112:
        raise MatrixProtocolError(f"Frozen total_fits must be 112, found {matrix.get('total_fits')!r}.")
    if amendment is not None:
        hyper = amendment.get("hyperparameters")
        if not isinstance(hyper, Mapping):
            raise MatrixProtocolError("training_protocol_amendment_20260917.json is missing hyperparameters.")
        if float(hyper.get("small_cnn", {}).get("lr", float("nan"))) != 1.0e-3:
            raise MatrixProtocolError("Frozen SmallCNN learning rate must be 1e-3.")
        if float(hyper.get("resnet18_scratch_gn", {}).get("lr", float("nan"))) != 3.0e-4:
            raise MatrixProtocolError("Frozen ResNet18 learning rate must be 3e-4.")
        logistic = hyper.get("logistic", {})
        if float(logistic.get("C", float("nan"))) != 1.0:
            raise MatrixProtocolError("Frozen Logistic C must be 1.")
        if str(logistic.get("selection", "")).startswith("none;") is False:
            raise MatrixProtocolError("Logistic must have no validation checkpoint selection.")


def _read_jsonl_count(path: Path) -> tuple[int, int, int]:
    """Return rows and unique participant/pair counts without loading images."""

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
                    raise MatrixProtocolError(f"Invalid JSONL {path}:{line_number}.") from exc
                if not isinstance(value, Mapping):
                    raise MatrixProtocolError(f"Manifest row {path}:{line_number} is not an object.")
                participant = str(value.get("participant_id", "")).strip()
                pair = str(value.get("pair_id", "")).strip()
                if not participant or not pair:
                    raise MatrixProtocolError(f"Manifest row {path}:{line_number} has empty participant/pair ID.")
                rows += 1
                participants.add(participant)
                pairs.add(pair)
    except OSError as exc:
        raise MatrixProtocolError(f"Cannot read manifest {path}.") from exc
    return rows, len(participants), len(pairs)


def resolve_cohort_paths(build_root: Path, cohort: str) -> MatrixPaths:
    name = str(cohort).strip().lower()
    if name not in COHORTS:
        raise MatrixProtocolError(f"Unknown cohort {cohort!r}.")
    root = (build_root / "manifests" / name).resolve()
    manifests = {split: root / f"{split}.jsonl" for split in SPLITS}
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise MatrixProtocolError(f"Missing {name} manifest(s): {'; '.join(missing)}")
    return MatrixPaths(cohort=name, root=root, manifests=manifests)


def _cell_id(cohort: str, family: str, modalities: Sequence[str], seed: int) -> str:
    input_name = "_".join(modalities)
    return f"{cohort}__{family}__{input_name}__seed{int(seed)}"


def _input_name(modalities: Sequence[str]) -> str:
    return "_".join(str(value).strip().lower() for value in modalities)


def _capture_control_bindings(path: Path) -> dict[str, Any]:
    """Capture immutable bindings exposed by a completed control artifact.

    A control gate's location under the build directory is only a path check;
    it does not establish which manifest bytes produced the result.  When a
    control already exists while the matrix plan is frozen, retain the gate
    digest, run fingerprint, and the hashes of its three control manifests.
    A later preflight compares all of those values.  Missing controls are
    recorded as ``MISSING`` and remain pending until a new plan is frozen after
    the control has been produced.
    """

    resolved = path.resolve()
    if not resolved.is_file():
        return {"status": "MISSING", "path": str(resolved)}
    try:
        value = _json_read(resolved)
    except MatrixProtocolError:
        return {"status": "INVALID", "path": str(resolved)}
    if not isinstance(value, Mapping):
        return {"status": "INVALID", "path": str(resolved)}
    control_manifest = value.get("control_manifest")
    manifest_dir = control_manifest.get("manifest_dir") if isinstance(control_manifest, Mapping) else None
    if not manifest_dir:
        return {
            "status": "INVALID",
            "path": str(resolved),
            "artifact_sha256": _sha256(resolved),
            "reason": "control_manifest.manifest_dir is missing",
        }
    manifest_root = Path(str(manifest_dir)).resolve()
    manifest_hashes = _hash_records({split: manifest_root / f"{split}.jsonl" for split in SPLITS})
    manifest_status = "PASS" if all(item["status"] == "PASS" for item in manifest_hashes.values()) else "MISSING"
    return {
        "status": manifest_status,
        "path": str(resolved),
        "artifact_sha256": _sha256(resolved),
        "run_fingerprint": value.get("run_fingerprint"),
        "manifest_dir": str(manifest_root),
        "manifest_hashes": manifest_hashes,
    }


def _control_requirement(
    name: str,
    path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    source_build_root: Path | None = None,
) -> dict[str, Any]:
    value = {
        "name": str(name),
        "path": str(path.resolve()),
        "allowed_statuses": ["PASS"],
        "nested_gate": True,
    }
    if expected_identity is not None:
        value["expected_identity"] = dict(expected_identity)
    if source_build_root is not None:
        value["source_build_root"] = str(source_build_root.resolve())
    # Freeze the actual gate and control-manifest bindings when this plan is
    # created.  For a missing Stage-B control this deliberately remains
    # incomplete; a later plan revision must be frozen once that control is
    # available instead of accepting an unbound PASS artifact.
    binding = _capture_control_bindings(path)
    value["binding_capture_status"] = binding.get("status")
    for key in ("artifact_sha256", "run_fingerprint", "manifest_dir", "manifest_hashes"):
        if key in binding:
            value[f"frozen_{key}"] = binding[key]
    return value


def _cell_control_requirements(build_root: Path, *, cohort: str, family: str, modalities: Sequence[str]) -> list[dict[str, Any]]:
    """Resolve the preregistered neural control gates for one matrix cell.

    Logistic cells have no neural overfit or registered/raw capacity gate.  For
    neural cells, FOMO joint SmallCNN uses the already completed Stage-A
    controls for the joint three-channel SmallCNN capability and its
    same-cohort negative.  Every other neural coordinate must acquire its own
    Stage-B control artifacts before that coordinate is executed; a missing
    artifact is therefore a deliberate block rather than an implicit pass.
    """

    if family == "logistic":
        return []
    input_name = _input_name(modalities)
    # Tiny/positive Stage-B artifacts are trained at the cell's actual input
    # width, whereas the shared FOMO Stage-A artifacts are the frozen joint
    # three-channel SmallCNN capability controls.  Negative controls are
    # preregistered as one three-channel fit per cohort/architecture and are
    # consequently shared across that architecture's four input cells.
    tiny_identity = {
        "mode": "tiny",
        "gate_name": "tiny_overfit",
        "model": ACTUAL_MODELS[family] if not (family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES) else "small_cnn",
        "modalities": list(modalities) if not (family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES) else list(JOINT_MODALITIES),
        "stage": "final",
        "cohort": "fomo45k",
    }
    positive_identity = {
        "mode": "positive",
        "gate_name": "positive_registered",
        "model": ACTUAL_MODELS[family] if not (family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES) else "small_cnn",
        "modalities": list(modalities) if not (family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES) else list(JOINT_MODALITIES),
        "stage": "registered",
        "cohort": "fomo45k",
    }
    negative_identity = {
        "mode": "negative",
        "gate_name": "negative_or_shuffle",
        "model": ACTUAL_MODELS[family],
        "modalities": list(JOINT_MODALITIES),
        "stage": "final",
        "cohort": cohort,
    }
    shared_fomo_capability = family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES
    if shared_fomo_capability:
        tiny = build_root / "stage_a_fomo45k" / "tiny" / "tiny" / "gate.json"
        positive = build_root / "stage_a_fomo45k_acl_retry_20260917" / "positive" / "gate.json"
    else:
        tiny = build_root / "stage_b_fomo_controls" / family / input_name / "tiny" / "gate.json"
        positive = build_root / "stage_b_fomo_controls" / family / input_name / "positive" / "gate.json"
    if cohort == "fomo45k" and family == "small_cnn":
        negative = build_root / "stage_a_fomo45k_negative_v3_20260917" / "negative" / "gate.json"
    else:
        negative = build_root / "stage_b_controls" / cohort / family / "negative" / "gate.json"
    return [
        _control_requirement("tiny", tiny, expected_identity=tiny_identity, source_build_root=build_root),
        _control_requirement("positive", positive, expected_identity=positive_identity, source_build_root=build_root),
        _control_requirement("negative", negative, expected_identity=negative_identity, source_build_root=build_root),
    ]


def _existing_observed_fit_descriptor(build_root: Path, *, cohort: str, family: str, modalities: Sequence[str], seed: int) -> dict[str, Any] | None:
    """Describe the one completed observed fit eligible for strict reuse."""

    if not (
        cohort == "fomo45k"
        and family == "small_cnn"
        and tuple(modalities) == JOINT_MODALITIES
        and int(seed) == 73
    ):
        return None
    root = build_root / "stage_a_fomo45k_observed_v3_20260917"
    result_path = root / "observed" / "result.json"
    expected_fingerprint: str | None = None
    if result_path.is_file():
        try:
            value = _json_read(result_path)
            if isinstance(value, Mapping):
                candidate = value.get("run_fingerprint")
                if candidate is not None:
                    expected_fingerprint = str(candidate)
        except MatrixProtocolError:
            # The descriptor remains present and the strict verifier will
            # report INVALID rather than silently treating a bad artifact as
            # reusable.
            expected_fingerprint = None
    return {
        "kind": "observed_calibration",
        "root": str(root.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str((root / "observed" / "model_best.pt").resolve()),
        "predictions": str((root / "observed" / "test_predictions.jsonl").resolve()),
        "run_identity": str((root / "run_identity.json").resolve()),
        "input_binding_audit": str((root / "input_binding_audit.json").resolve()),
        "source_cache_digest": str((root / "source_cache_digest.json").resolve()),
        "expected_run_fingerprint": expected_fingerprint,
        "required_input_binding_status": "PASS",
        "reuse_equivalence": "strict_run_fingerprint_config_manifest_checkpoint_predictions_input_binding",
    }


def _primary_null_spec(
    build_root: Path,
    cell_dir: Path,
    *,
    cohort: str,
    family: str,
    modalities: Sequence[str],
    seed: int,
    manifest_root: Path,
    device: str,
) -> dict[str, Any]:
    """Record an executable, cohort-specific primary null contract."""

    is_primary = family == "small_cnn" and tuple(modalities) == JOINT_MODALITIES and int(seed) == 73
    base = {
        "required": is_primary,
        "replicates": 199 if is_primary else 0,
        "init_seed": 73 if is_primary else None,
        "label_scope": "TRAIN/VAL/TEST whole participant-pair swaps",
        "validation_selection": "draw-specific permuted validation labels",
        "entrypoint": None,
        "output_dir": None,
        "command": None,
        "reuse_existing": False,
    }
    if not is_primary:
        return base
    if cohort == "fomo45k":
        existing_root = build_root / "stage_a_fomo45k_observed_v3_20260917"
        base.update(
            {
                "entrypoint": "existing_calibration",
                "output_dir": str(existing_root.resolve()),
                "reuse_existing": True,
                "existing_run_fingerprint": "64fff1630c7fc63cf595d8671330bd14974c67b8f26c66784c33265e655d0137",
                "command": None,
            }
        )
        return base
    output_dir = cell_dir / "primary_null"
    entrypoint = REPO_ROOT / "scripts" / "run_domain_classifier_primary_null_v3.py"
    base.update(
        {
            "entrypoint": str(entrypoint.resolve()),
            "output_dir": str(output_dir.resolve()),
            "command": [
                str(ANDI_PYTHON),
                str(entrypoint),
                "--comparison",
                cohort,
                "--manifest-root",
                str(manifest_root.resolve()),
                "--build-root",
                str(build_root.resolve()),
                "--config-path",
                str((REPO_ROOT / "configs" / "domain_classifier.yaml").resolve()),
                "--ledger-path",
                str((build_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl").resolve()),
                "--ledger-summary-path",
                str((build_root / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger_summary.json").resolve()),
                "--output-root",
                str(output_dir.resolve()),
                "--replicates",
                "199",
                "--device",
                str(device),
                "--init-seed",
                "73",
            ],
        }
    )
    return base


def _source_build_hashes(build_root: Path) -> dict[str, dict[str, Any]]:
    return _hash_records(
        {
            "build_protocol": build_root / "protocol.json",
            "source_fingerprints": build_root / "source_fingerprints.json",
            "build_summary": build_root / "build_summary.json",
        }
    )


def build_matrix_plan(
    protocol: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any] | None = None,
    build_root: Path,
    output_root: Path,
    device: str = "cuda",
    executable: Path = ANDI_PYTHON,
) -> dict[str, Any]:
    """Build the deterministic 112-cell plan and validate each manifest root."""

    _validate_frozen_protocol(protocol, amendment)
    resolved_build = build_root.resolve()
    resolved_output = output_root.resolve()
    cells: list[dict[str, Any]] = []
    cohort_summaries: dict[str, Any] = {}
    source_build_hashes = _source_build_hashes(resolved_build)
    code_hashes = _hash_records({path.name: path for path in MATRIX_CODE_PATHS})
    for cohort in COHORTS:
        paths = resolve_cohort_paths(resolved_build, cohort)
        split_summary: dict[str, Any] = {}
        for split, manifest in paths.manifests.items():
            rows, participants, pairs = _read_jsonl_count(manifest)
            split_summary[split] = {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "rows": rows,
                "unique_participants": participants,
                "unique_pairs": pairs,
            }
        cohort_summaries[cohort] = {"manifest_root": str(paths.root), "splits": split_summary}
        for modalities in INPUT_TYPES:
            for family, seeds in FAMILY_SEEDS.items():
                for seed in seeds:
                    actual_model = ACTUAL_MODELS[family]
                    cell_id = _cell_id(cohort, family, modalities, seed)
                    cell_dir = resolved_output / "cells" / cell_id
                    config_path = cell_dir / "config.yaml"
                    is_primary = (
                        family == "small_cnn"
                        and tuple(modalities) == JOINT_MODALITIES
                        and int(seed) == 73
                    )
                    secondary_group_id = f"{cohort}__{family}__{_input_name(modalities)}"
                    primary_null = _primary_null_spec(
                        resolved_build,
                        cell_dir,
                        cohort=cohort,
                        family=family,
                        modalities=modalities,
                        seed=seed,
                        manifest_root=paths.root,
                        device=device,
                    )
                    # Ordinary cells use the v3 binding boundary.  It
                    # materializes canonical all-channel tensors once,
                    # checks the frozen healthy/BraTS identities, projects
                    # the requested input in memory, and then fits.  The
                    # primary coordinate is still represented here for
                    # auditability; its completed observed/199-null fit is
                    # reused by the execution gate before any subprocess is
                    # considered.
                    cell_entrypoint = REPO_ROOT / "scripts" / "run_domain_classifier_cell_v3.py"
                    if is_primary and cohort != "fomo45k":
                        # Non-FOMO primary cells own the full TRAIN/VAL/TEST
                        # 199-draw retrained null and must use its dedicated
                        # entrypoint.  The ordinary cell CLI intentionally
                        # refuses this coordinate to prevent a duplicate or
                        # an observed-only fit.
                        command = list(primary_null["command"] or ())
                    else:
                        command = [
                            str(executable),
                            str(cell_entrypoint.resolve()),
                            "--config",
                            str(config_path),
                            "--manifest-root",
                            str(paths.root),
                            "--output-dir",
                            str(cell_dir / "fit"),
                            "--comparison",
                            str(cohort),
                            "--stage",
                            "final",
                            "--device",
                            str(device),
                            "--modalities",
                            *modalities,
                            "--seeds",
                            str(seed),
                            "--build-root",
                            str(resolved_build),
                            "--ledger-path",
                            str((resolved_build / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger.jsonl").resolve()),
                            "--ledger-summary-path",
                            str((resolved_build / "observed_input_binding_v1" / "selected_healthy_source_tensor_ledger_summary.json").resolve()),
                        ]
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "comparison": cohort,
                            "cohort": cohort,
                            "family": family,
                            "model": actual_model,
                            "modalities": list(modalities),
                            "input_type": "+".join(modalities),
                            "in_channels": len(modalities),
                            "seed": int(seed),
                            "manifest_root": str(paths.root),
                            "manifests": {split: str(path) for split, path in paths.manifests.items()},
                            "config_path": str(config_path),
                            "output_dir": str(cell_dir / "fit"),
                            "manifest_hashes": {
                                split: split_summary[split]["sha256"] for split in SPLITS
                            },
                            "code_hashes": code_hashes,
                            "source_build_hashes": source_build_hashes,
                            "control_requirements": _cell_control_requirements(
                                resolved_build,
                                cohort=cohort,
                                family=family,
                                modalities=modalities,
                            ),
                            "existing_fit": _existing_observed_fit_descriptor(
                                resolved_build,
                                cohort=cohort,
                                family=family,
                                modalities=modalities,
                                seed=seed,
                            ),
                            "primary_null": primary_null,
                            "command": command,
                            "execution_route": (
                                "reuse_existing_primary_observed_and_full_null"
                                if is_primary
                                and cohort == "fomo45k"
                                else "run_primary_null_v3"
                                if is_primary
                                else "run_domain_classifier_cell_v3"
                            ),
                            "analysis_roles": {
                                "primary_observed_fit": is_primary,
                                "primary_null_replicates": 199 if is_primary else 0,
                                "primary_observed_heldout_pair_swap_replicates": 1000 if is_primary else 0,
                                "secondary_fixed_classifier_pair_swaps": True,
                                "secondary_group_id": secondary_group_id,
                                "secondary_group_swap_replicates": 9999,
                                "primary_full_retrained_null": is_primary,
                                "historical_shuffle_control_substitute": False,
                            },
                        }
                    )
    _validate_plan_cells(cells)
    secondary_groups: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for modalities in INPUT_TYPES:
            for family in FAMILY_SEEDS:
                group_id = f"{cohort}__{family}__{_input_name(modalities)}"
                members = [
                    cell["cell_id"]
                    for cell in cells
                    if cell["analysis_roles"]["secondary_group_id"] == group_id
                ]
                secondary_groups.append(
                    {
                        "group_id": group_id,
                        "cohort": cohort,
                        "family": family,
                        "modalities": list(modalities),
                        "member_cell_ids": members,
                        "expected_member_count": len(FAMILY_SEEDS[family]),
                        "aggregate": (
                            "mean_subject_probabilities_before_auc"
                            if family != "logistic"
                            else "single_logistic_seed73_subject_probabilities"
                        ),
                        "fixed_classifier_pair_swap_replicates": 9999,
                        "holm_family": "secondary_48",
                        "holm_alpha": 0.05,
                        "conditional_interpretation": "matched-sample diagnostic",
                    }
                )
    protocol_hash = hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest()
    amendment_hash = None if amendment is None else hashlib.sha256(_canonical_json(amendment).encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "plan_id": "model_grid_v3_matrix_112_plan_v2",
        "status": "PLANNED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "executable": str(executable),
        "executable_required": str(ANDI_PYTHON),
        "device": str(device),
        "source_build_root": str(resolved_build),
        "output_root": str(resolved_output),
        "source_build_hashes": source_build_hashes,
        "runner_code_hashes": code_hashes,
        "training_protocol_sha256": protocol_hash,
        "training_protocol_amendment_sha256": amendment_hash,
        "cohorts": list(COHORTS),
        "input_types": [list(values) for values in INPUT_TYPES],
        "family_seeds": {key: list(value) for key, value in FAMILY_SEEDS.items()},
        "fit_count": len(cells),
        "secondary_group_count": len(secondary_groups),
        "secondary_groups": secondary_groups,
        "cohort_summaries": cohort_summaries,
        "cells": cells,
        "execution_gate": {
            "status": "REQUIRES_COMPLETED_STAGE_A_AND_PRIMARY_NULL",
            "historical_shuffle_gate_required": False,
            "primary_null_required_before_execute": True,
        },
    }


def _validate_plan_cells(cells: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (cohort, family, modalities, seed)
        for cohort in COHORTS
        for modalities in INPUT_TYPES
        for family, seeds in FAMILY_SEEDS.items()
        for seed in seeds
    }
    observed = {
        (
            str(cell.get("cohort", "")),
            str(cell.get("family", "")),
            tuple(cell.get("modalities", ())),
            int(cell.get("seed", -1)),
        )
        for cell in cells
    }
    if len(cells) != len(expected) or observed != expected:
        raise MatrixProtocolError(
            f"Matrix plan identity mismatch: expected {len(expected)} unique cells, got {len(cells)} rows/{len(observed)} unique."
        )
    ids = [str(cell.get("cell_id", "")) for cell in cells]
    if not all(ids) or len(set(ids)) != len(ids):
        raise MatrixProtocolError("Matrix cell IDs must be non-empty and unique.")
    for cell in cells:
        family = str(cell["family"])
        expected_model = ACTUAL_MODELS[family]
        if str(cell.get("model")) != expected_model:
            raise MatrixProtocolError(f"Family {family} must use runner model {expected_model!r}.")
        modalities = tuple(cell.get("modalities", ()))
        is_primary = family == "small_cnn" and modalities == JOINT_MODALITIES and int(cell.get("seed", -1)) == 73
        roles = cell.get("analysis_roles") if isinstance(cell.get("analysis_roles"), Mapping) else {}
        if bool(roles.get("primary_full_retrained_null", False)) != is_primary:
            raise MatrixProtocolError(f"Cell {cell.get('cell_id')} has an invalid primary-null role.")
        if int(roles.get("primary_null_replicates", 0)) != (199 if is_primary else 0):
            raise MatrixProtocolError(f"Cell {cell.get('cell_id')} has an invalid primary-null replicate count.")
        null_spec = cell.get("primary_null") if isinstance(cell.get("primary_null"), Mapping) else {}
        if bool(null_spec.get("required", False)) != is_primary:
            raise MatrixProtocolError(f"Cell {cell.get('cell_id')} has an invalid primary-null command contract.")
    group_members: Counter[str] = Counter()
    for cell in cells:
        roles = cell.get("analysis_roles") if isinstance(cell.get("analysis_roles"), Mapping) else {}
        group_id = str(roles.get("secondary_group_id", ""))
        if not group_id:
            raise MatrixProtocolError(f"Cell {cell.get('cell_id')} is missing its secondary group ID.")
        group_members[group_id] += 1
    if len(group_members) != 48:
        raise MatrixProtocolError(f"Expected 48 secondary groups, found {len(group_members)}.")
    for group_id, count in group_members.items():
        family = group_id.split("__", 2)[1] if "__" in group_id else ""
        if count != len(FAMILY_SEEDS.get(family, ())):
            raise MatrixProtocolError(f"Secondary group {group_id} has {count} members, expected family seed count.")


def _artifact_status(path: Path, *, nested_gate: bool = False) -> str:
    if not path.is_file():
        return "MISSING"
    try:
        value = _json_read(path)
    except MatrixProtocolError:
        return "INVALID"
    if not isinstance(value, Mapping):
        return "INVALID"
    if nested_gate and isinstance(value.get("gate"), Mapping):
        value = value["gate"]
    status = value.get("status")
    return str(status) if status is not None else "MISSING_STATUS"


def _control_identity(path: Path, requirement: Mapping[str, Any]) -> dict[str, Any]:
    """Check the control's declared mode/config, beyond a bare PASS string."""

    expected = requirement.get("expected_identity")
    if not isinstance(expected, Mapping):
        return {"status": "PASS", "checks": {}, "reason": "no identity contract supplied"}
    resolved_path = path.resolve()
    source_root_value = requirement.get("source_build_root")
    if source_root_value:
        source_root = Path(str(source_root_value)).resolve()
        try:
            resolved_path.relative_to(source_root)
            source_path_status = "PASS"
        except ValueError:
            source_path_status = "MISMATCH"
        if source_path_status != "PASS":
            return {
                "status": "MISMATCH",
                "checks": {"source_build_root": {"expected": str(source_root), "actual": str(resolved_path)}},
                "failures": ["source_build_root"],
            }
    try:
        value = _json_read(path)
    except MatrixProtocolError:
        return {"status": "INVALID", "checks": {}, "reason": "invalid JSON"}
    if not isinstance(value, Mapping):
        return {"status": "INVALID", "checks": {}, "reason": "control artifact is not an object"}
    failures: list[str] = []
    checks: dict[str, Any] = {}
    # A frozen plan binds the actual gate bytes and the control manifests that
    # produced them.  A PASS gate with a later swapped JSON/manifest is a
    # protocol mismatch even when it remains below the same build directory.
    capture_status = str(requirement.get("binding_capture_status", "MISSING"))
    checks["binding_capture"] = {"status": capture_status}
    if capture_status != "PASS":
        failures.append("binding_capture")
    expected_artifact_sha = requirement.get("frozen_artifact_sha256")
    if expected_artifact_sha is not None:
        actual_artifact_sha = _sha256(resolved_path)
        item = {
            "expected_sha256": str(expected_artifact_sha),
            "actual_sha256": actual_artifact_sha,
            "status": "PASS" if actual_artifact_sha == str(expected_artifact_sha) else "MISMATCH",
        }
        checks["artifact_sha256"] = item
        if item["status"] != "PASS":
            failures.append("artifact_sha256")
    else:
        checks["artifact_sha256"] = {"status": "MISSING_EXPECTATION"}
        failures.append("artifact_sha256")
    expected_run_fingerprint = requirement.get("frozen_run_fingerprint")
    actual_run_fingerprint = value.get("run_fingerprint")
    run_item = {
        "expected": expected_run_fingerprint,
        "actual": actual_run_fingerprint,
        "status": "PASS"
        if expected_run_fingerprint is not None and actual_run_fingerprint == expected_run_fingerprint
        else "MISMATCH",
    }
    checks["run_fingerprint"] = run_item
    if run_item["status"] != "PASS":
        failures.append("run_fingerprint")
    if source_root_value:
        checks["source_build_root"] = {"status": "PASS", "path": str(resolved_path)}
    control_manifest = value.get("control_manifest")
    manifest_dir = control_manifest.get("manifest_dir") if isinstance(control_manifest, Mapping) else None
    if manifest_dir:
        manifest_path = Path(str(manifest_dir)).resolve()
        try:
            manifest_path.relative_to(Path(str(source_root_value)).resolve()) if source_root_value else manifest_path
            manifest_status = "PASS"
        except ValueError:
            manifest_status = "MISMATCH"
        checks["control_manifest_root"] = {"status": manifest_status, "path": str(manifest_path)}
        if manifest_status != "PASS":
            failures.append("control_manifest_root")
    else:
        failures.append("control_manifest")
    expected_manifest_dir = requirement.get("frozen_manifest_dir")
    if expected_manifest_dir is None or manifest_dir is None:
        checks["control_manifest_dir"] = {
            "expected": expected_manifest_dir,
            "actual": manifest_dir,
            "status": "MISMATCH",
        }
        failures.append("control_manifest_dir")
    else:
        manifest_dir_item = {
            "expected": str(expected_manifest_dir),
            "actual": str(Path(str(manifest_dir)).resolve()),
            "status": "PASS"
            if Path(str(manifest_dir)).resolve() == Path(str(expected_manifest_dir)).resolve()
            else "MISMATCH",
        }
        checks["control_manifest_dir"] = manifest_dir_item
        if manifest_dir_item["status"] != "PASS":
            failures.append("control_manifest_dir")
    expected_manifest_hashes = requirement.get("frozen_manifest_hashes")
    current_manifest_hashes: dict[str, Any] = {}
    if manifest_dir:
        manifest_root = Path(str(manifest_dir)).resolve()
        current_manifest_hashes = _hash_records(
            {split: manifest_root / f"{split}.jsonl" for split in SPLITS}
        )
    manifest_hash_items: dict[str, Any] = {}
    if not isinstance(expected_manifest_hashes, Mapping):
        failures.append("control_manifest_hashes")
    else:
        for split in SPLITS:
            expected_item = expected_manifest_hashes.get(split)
            actual_item = current_manifest_hashes.get(split, {"status": "MISSING", "sha256": None})
            expected_sha = expected_item.get("sha256") if isinstance(expected_item, Mapping) else None
            actual_sha = actual_item.get("sha256") if isinstance(actual_item, Mapping) else None
            item = {
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "status": "PASS"
                if expected_sha is not None
                and actual_item.get("status") == "PASS"
                and actual_sha == expected_sha
                else "MISMATCH",
            }
            manifest_hash_items[split] = item
            if item["status"] != "PASS":
                failures.append(f"control_manifest_hashes:{split}")
    checks["control_manifest_hashes"] = manifest_hash_items
    expected_cohort = expected.get("cohort")
    if expected_cohort is not None:
        expected_cohort = str(expected_cohort).strip().lower()
        declared_values: list[str] = []
        for source in (value, value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}):
            for key in ("cohort", "comparison", "source_cohort"):
                candidate = source.get(key) if isinstance(source, Mapping) else None
                if candidate is not None and str(candidate).strip():
                    declared_values.append(str(candidate).strip().lower())
        healthy_prefixes: set[str] = set()
        inspected_rows = 0
        results_for_cohort = value.get("results")
        if isinstance(results_for_cohort, list):
            for result in results_for_cohort:
                if not isinstance(result, Mapping):
                    continue
                for prediction_key in (
                    "test_predictions",
                    "test_subject_predictions",
                    "validation_predictions",
                    "validation_subject_predictions",
                    "train_final_predictions",
                    "train_final_subject_predictions",
                ):
                    rows = result.get(prediction_key)
                    if not isinstance(rows, list):
                        continue
                    for row in rows[:256]:
                        if not isinstance(row, Mapping):
                            continue
                        inspected_rows += 1
                        try:
                            label = int(row.get("label"))
                        except (TypeError, ValueError):
                            continue
                        if label != 0:
                            continue
                        participant = str(row.get("participant_id", "")).strip()
                        if ":" in participant:
                            healthy_prefixes.add(participant.split(":", 1)[0].strip().lower())
        allowed_prefixes = {expected_cohort}
        if expected_cohort == "mixed":
            allowed_prefixes = {"fomo45k", "mpi", "oasis3"}
        if declared_values:
            cohort_status = "PASS" if any(candidate == expected_cohort for candidate in declared_values) else "MISMATCH"
        else:
            cohort_status = (
                "PASS"
                if healthy_prefixes and healthy_prefixes.issubset(allowed_prefixes)
                else "MISMATCH"
            )
        checks["cohort"] = {
            "expected": expected_cohort,
            "declared": declared_values,
            "healthy_participant_prefixes": sorted(healthy_prefixes),
            "inspected_prediction_rows": inspected_rows,
            "allowed_prefixes": sorted(allowed_prefixes),
            "status": cohort_status,
        }
        if cohort_status != "PASS":
            failures.append("cohort")
    mode = expected.get("mode")
    if mode is not None:
        actual_mode = value.get("mode")
        checks["mode"] = {"expected": mode, "actual": actual_mode, "status": "PASS" if actual_mode == mode else "MISMATCH"}
        if actual_mode != mode:
            failures.append("mode")
    gate = value.get("gate") if isinstance(value.get("gate"), Mapping) else {}
    gate_name = expected.get("gate_name")
    if gate_name is not None:
        actual_name = gate.get("name")
        checks["gate_name"] = {"expected": gate_name, "actual": actual_name, "status": "PASS" if actual_name == gate_name else "MISMATCH"}
        if actual_name != gate_name:
            failures.append("gate_name")
    results = value.get("results")
    if not isinstance(results, list) or not results:
        failures.append("results")
        results = []
    expected_model = expected.get("model")
    expected_modalities = tuple(expected.get("modalities", ()))
    expected_stage = expected.get("stage")
    config_checks: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        config = result.get("config") if isinstance(result, Mapping) and isinstance(result.get("config"), Mapping) else {}
        model = config.get("model")
        modalities = tuple(config.get("modalities", ()))
        stage = config.get("stage")
        item = {
            "index": index,
            "model": "PASS" if model == expected_model else "MISMATCH",
            "modalities": "PASS" if modalities == expected_modalities else "MISMATCH",
            "stage": "PASS" if expected_stage is None or stage == expected_stage else "MISMATCH",
        }
        config_checks.append(item)
        if item["model"] != "PASS":
            failures.append(f"results[{index}].model")
        if item["modalities"] != "PASS":
            failures.append(f"results[{index}].modalities")
        if item["stage"] != "PASS":
            failures.append(f"results[{index}].stage")
    checks["result_configs"] = config_checks
    return {
        "status": "PASS" if not failures else "MISMATCH",
        "checks": checks,
        "failures": failures,
    }


def _verify_file_hash(path: Path, expected: str | None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "status": "MISSING", "expected_sha256": expected, "actual_sha256": None}
    actual = _sha256(resolved)
    status = "PASS" if expected and actual == str(expected) else "MISMATCH"
    return {"path": str(resolved), "status": status, "expected_sha256": expected, "actual_sha256": actual}


def _verify_existing_fit(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the completed observed FOMO fit before permitting reuse."""

    descriptor = cell.get("existing_fit")
    if not isinstance(descriptor, Mapping):
        return {"status": "NOT_APPLICABLE", "failures": []}
    required = {
        "result": Path(str(descriptor.get("result", ""))),
        "checkpoint": Path(str(descriptor.get("checkpoint", ""))),
        "predictions": Path(str(descriptor.get("predictions", ""))),
        "run_identity": Path(str(descriptor.get("run_identity", ""))),
        "input_binding_audit": Path(str(descriptor.get("input_binding_audit", ""))),
        "source_cache_digest": Path(str(descriptor.get("source_cache_digest", ""))),
    }
    failures: list[str] = []
    path_status = {name: _hash_record(path) for name, path in required.items()}
    failures.extend(name for name, record in path_status.items() if record["status"] != "PASS")
    values: dict[str, Any] = {}
    if not failures:
        try:
            result = _json_read(required["result"])
            identity = _json_read(required["run_identity"])
            binding = _json_read(required["input_binding_audit"])
            cache = _json_read(required["source_cache_digest"])
            if not all(isinstance(value, Mapping) for value in (result, identity, binding, cache)):
                failures.append("invalid_json_schema")
            else:
                values = {"result": result, "identity": identity, "binding": binding, "cache": cache}
        except MatrixProtocolError:
            failures.append("invalid_json")
    result = values.get("result", {})
    identity = values.get("identity", {})
    binding = values.get("binding", {})
    cache = values.get("cache", {})
    expected_fp = descriptor.get("expected_run_fingerprint")
    if expected_fp is None:
        failures.append("missing_expected_run_fingerprint")
    for name, value in (("result", result), ("identity", identity)):
        if expected_fp is not None and value.get("run_fingerprint") != expected_fp:
            failures.append(f"{name}.run_fingerprint")
    model_value = result.get("model")
    model_name = model_value.get("class") if isinstance(model_value, Mapping) else model_value
    if str(model_name or "").lower() not in {"smallcnn", "small_cnn"}:
        failures.append("result.model")
    try:
        result_seed = int(result.get("seed", -1))
        cell_seed = int(cell.get("seed", -2))
    except (TypeError, ValueError):
        result_seed, cell_seed = -1, -2
    if result_seed != cell_seed:
        failures.append("result.seed")
    config = result.get("config") if isinstance(result.get("config"), Mapping) else {}
    if str(config.get("stage", "")) != "final":
        failures.append("result.config.stage")
    if tuple(config.get("modalities", ())) != tuple(cell.get("modalities", ())):
        failures.append("result.config.modalities")
    try:
        split_seed = int(config.get("split_seed", -1))
    except (TypeError, ValueError):
        split_seed = -1
    if split_seed != 73:
        failures.append("result.config.split_seed")
    if str(binding.get("status", "")) != "PASS":
        failures.append("input_binding_audit.status")
    if str(cache.get("sha256", "")) != "e1605d2fd4af5fe8018bd130b0c428d30e3f3bebbe5cfd0b4f29552d0a3a3308":
        # This digest is the completed observed source-cache binding.  A
        # future observed fit must publish a new descriptor rather than being
        # silently attached to this matrix cell.
        failures.append("source_cache_digest.sha256")
    manifest_identity = identity.get("manifest_identity") if isinstance(identity, Mapping) else {}
    identity_hashes = manifest_identity.get("sha256") if isinstance(manifest_identity, Mapping) else {}
    expected_manifests = cell.get("manifest_hashes") if isinstance(cell.get("manifest_hashes"), Mapping) else {}
    for split in SPLITS:
        if identity_hashes.get(split) != expected_manifests.get(split):
            failures.append(f"run_identity.manifest:{split}")
    return {
        "status": "PASS" if not failures else "BLOCKED",
        "reusable": not failures,
        "failures": sorted(set(failures)),
        "paths": path_status,
        "expected_run_fingerprint": expected_fp,
    }


def _cell_execution_gate(cell: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on control, input, and source drift for one matrix cell."""

    checks: dict[str, Any] = {}
    failures: list[str] = []
    for requirement in cell.get("control_requirements", ()):
        if not isinstance(requirement, Mapping):
            failures.append("invalid_control_requirement")
            continue
        path = Path(str(requirement.get("path", "")))
        try:
            status = _artifact_status(path, nested_gate=bool(requirement.get("nested_gate", False)))
        except MatrixProtocolError:
            status = "INVALID"
        name = str(requirement.get("name", path.name))
        checks[f"control:{name}"] = {"path": str(path), "status": status}
        allowed = {str(value) for value in requirement.get("allowed_statuses", ("PASS",))}
        if status not in allowed:
            failures.append(f"control:{name}")
        elif status == "PASS":
            identity = _control_identity(path, requirement)
            checks[f"control_identity:{name}"] = identity
            if identity["status"] != "PASS":
                failures.append(f"control_identity:{name}")
    manifest_checks: dict[str, Any] = {}
    expected_manifests = cell.get("manifest_hashes")
    if not isinstance(expected_manifests, Mapping):
        failures.append("manifest_hashes")
    else:
        manifest_paths = cell.get("manifests") if isinstance(cell.get("manifests"), Mapping) else {}
        for split in SPLITS:
            path = Path(str(manifest_paths.get(split, "")))
            item = _verify_file_hash(path, str(expected_manifests.get(split)) if expected_manifests.get(split) else None)
            manifest_checks[split] = item
            if item["status"] != "PASS":
                failures.append(f"manifest:{split}")
    checks["manifests"] = manifest_checks
    code_checks: dict[str, Any] = {}
    expected_code = cell.get("code_hashes")
    if not isinstance(expected_code, Mapping):
        failures.append("code_hashes")
    else:
        for name, expected in expected_code.items():
            if not isinstance(expected, Mapping):
                failures.append(f"code:{name}")
                continue
            item = _verify_file_hash(Path(str(expected.get("path", ""))), expected.get("sha256"))
            code_checks[str(name)] = item
            if item["status"] != "PASS":
                failures.append(f"code:{name}")
    checks["code"] = code_checks
    source_checks: dict[str, Any] = {}
    expected_source = plan.get("source_build_hashes")
    if not isinstance(expected_source, Mapping):
        failures.append("source_build_hashes")
    else:
        for name, expected in expected_source.items():
            if not isinstance(expected, Mapping):
                failures.append(f"source:{name}")
                continue
            item = _verify_file_hash(Path(str(expected.get("path", ""))), expected.get("sha256"))
            source_checks[str(name)] = item
            if item["status"] != "PASS":
                failures.append(f"source:{name}")
    checks["source_build"] = source_checks
    config_path = Path(str(cell.get("config_path", "")))
    config_check = _verify_file_hash(config_path, cell.get("config_sha256"))
    checks["config"] = config_check
    if config_check["status"] != "PASS":
        failures.append("config")
    plan_source_checks: dict[str, Any] = {}
    for path_key, hash_key in (
        ("base_config_path", "base_config_sha256"),
        ("training_protocol_path", "training_protocol_file_sha256"),
        ("training_protocol_amendment_path", "training_protocol_amendment_file_sha256"),
    ):
        path_value = plan.get(path_key)
        hash_value = plan.get(hash_key)
        if path_value is None and hash_value is None:
            continue
        item = _verify_file_hash(Path(str(path_value or "")), hash_value)
        plan_source_checks[path_key] = item
        if item["status"] != "PASS":
            failures.append(path_key)
    checks["plan_sources"] = plan_source_checks
    existing = _verify_existing_fit(cell)
    checks["existing_fit"] = existing
    if existing.get("status") == "BLOCKED":
        failures.append("existing_fit")
    pending_only = bool(failures) and all(
        failure.startswith("control:")
        and checks.get(failure, {}).get("status") in {"MISSING", "MISSING_STATUS"}
        for failure in failures
    )
    return {
        "status": "PASS" if not failures else "PENDING" if pending_only else "BLOCKED",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "checks": checks,
    }


def execution_gate(
    *,
    build_root: Path,
    stage_a_root: Path,
    primary_null_root: Path,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed execution gate for future matrix training."""

    build_root = build_root.resolve()
    stage_a_root = stage_a_root.resolve()
    primary_null_root = primary_null_root.resolve()
    checks: dict[str, Any] = {}
    build_summary = build_root / "build_summary.json"
    checks["build_summary"] = {
        "path": str(build_summary),
        "status": _artifact_status(build_summary),
    }
    for cohort in COHORTS:
        audit_path = build_root / "audits" / f"{cohort}.json"
        checks[f"audit:{cohort}"] = {"path": str(audit_path), "status": _artifact_status(audit_path)}
    controls = {
        "tiny": build_root / "stage_a_fomo45k" / "tiny" / "tiny" / "gate.json",
        "positive": build_root / "stage_a_fomo45k_acl_retry_20260917" / "positive" / "gate.json",
        "negative": build_root / "stage_a_fomo45k_negative_v3_20260917" / "negative" / "gate.json",
    }
    for name, path in controls.items():
        checks[f"control:{name}"] = {"path": str(path), "status": _artifact_status(path, nested_gate=True)}
    input_binding = stage_a_root / "input_binding_audit.json"
    checks["input_binding"] = {"path": str(input_binding), "status": _artifact_status(input_binding)}
    sanity = stage_a_root / "immutable_test_label_sanity" / "gate_random_test_labels.json"
    checks["heldout_label_sanity"] = {"path": str(sanity), "status": _artifact_status(sanity)}
    null_status = primary_null_root / "retrained_null" / "status.json"
    null_summary = primary_null_root / "retrained_null" / "summary.json"
    checks["primary_null_status"] = {"path": str(null_status), "status": _artifact_status(null_status)}
    checks["primary_null_summary"] = {"path": str(null_summary), "status": _artifact_status(null_summary)}
    checks["primary_null_integrity"] = {
        "path": str(primary_null_root / "calibration_integrity_audit.json"),
        "status": _artifact_status(primary_null_root / "calibration_integrity_audit.json"),
    }
    failures: list[str] = []
    if checks["build_summary"]["status"] != "PASS":
        failures.append("build_summary")
    failures.extend(key for key in (f"audit:{cohort}" for cohort in COHORTS) if checks[key]["status"] != "PASS")
    failures.extend(key for key in ("control:tiny", "control:positive", "control:negative") if checks[key]["status"] != "PASS")
    if checks["input_binding"]["status"] != "PASS":
        failures.append("input_binding")
    # INCONCLUSIVE is allowed to continue the predeclared null; only a
    # completed PASS or an explicit INCONCLUSIVE is an admissible status.
    # Missing/invalid artifacts must never be treated as a chance result.
    if checks["heldout_label_sanity"]["status"] not in {"PASS", "INCONCLUSIVE"}:
        failures.append("heldout_label_sanity")
    if checks["primary_null_status"]["status"] != "complete":
        failures.append("primary_null_status")
    if checks["primary_null_summary"]["status"] != "complete":
        failures.append("primary_null_summary")
    if checks["primary_null_integrity"]["status"] != "PASS":
        failures.append("primary_null_integrity")
    cell_gates: dict[str, Any] = {}
    ready_cells: list[str] = []
    pending_cells: list[str] = []
    blocked_cells: list[str] = []
    if plan is not None:
        for cell in plan.get("cells", ()):
            if not isinstance(cell, Mapping):
                cell_gates["INVALID"] = {"status": "BLOCKED", "failures": ["invalid_cell"]}
                blocked_cells.append("INVALID")
                continue
            cell_id = str(cell.get("cell_id", ""))
            cell_gate = _cell_execution_gate(cell, plan)
            cell_gates[cell_id] = cell_gate
            if cell_gate["status"] == "PASS":
                ready_cells.append(cell_id)
            elif cell_gate["status"] == "PENDING":
                pending_cells.append(cell_id)
            else:
                blocked_cells.append(cell_id)
        checks["cell_gates"] = cell_gates
    return {
        "schema_version": 1,
        "name": "model_grid_v3_matrix_execution_gate",
        "status": "PASS" if not failures else "BLOCKED",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "ready_cells": ready_cells,
        "pending_cells": pending_cells,
        "blocked_cells": blocked_cells,
        "cell_gate_summary": {
            "ready": len(ready_cells),
            "pending": len(pending_cells),
            "blocked": len(blocked_cells),
            "total": len(ready_cells) + len(pending_cells) + len(blocked_cells),
            "interpretation": "Cell-local PENDING/BLOCKED controls are recorded and skipped; they do not block independent ready cells once global prerequisites pass.",
        },
        "historical_shuffle_gate_required": False,
        "reason": "All frozen Stage-A and primary-null prerequisites are complete." if not failures else "Matrix execution is blocked until every listed prerequisite passes.",
    }


def _cell_config(
    cell: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Create a runner config without mutating the repository config."""

    base_training = base_config.get("training", base_config)
    if not isinstance(base_training, Mapping):
        base_training = {}
    training = dict(base_training)
    family = str(cell["family"])
    modalities = tuple(str(value) for value in cell["modalities"])
    training.update(
        {
            "model": ACTUAL_MODELS[family],
            "in_channels": len(modalities),
            "modalities": list(modalities),
            "stage": "final",
            "device": device,
            "bootstrap_replicates": 2000,
            # The 9,999 secondary swaps are an aggregate over the 48
            # classifier/input groups.  Individual fits only retain the
            # primary observed fit's 1,000 diagnostic swaps.
            "swap_replicates": int(
                cell.get("analysis_roles", {}).get("primary_observed_heldout_pair_swap_replicates", 0)
            ),
            "permutation_replicates": 0,
            "permutation_mode": "none",
            "tiny": False,
            "early_stopping": True,
            "dropout": 0.0,
            "no_augmentation": True,
            "threshold": 0.5,
            "batch_size": 32,
            "split_seed": 73,
        }
    )
    if family == "small_cnn":
        training.update({"learning_rate": 1.0e-3, "weight_decay": 1.0e-4, "max_epochs": 40, "patience": 8})
    elif family == "resnet18_scratch_gn":
        training.update({"learning_rate": 3.0e-4, "weight_decay": 1.0e-4, "max_epochs": 40, "patience": 8})
    elif family == "logistic":
        training.update(
            {
                "model": "statistical_logistic",
                "learning_rate": None,
                "weight_decay": 0.0,
                # These fields are also persisted in the cell config even
                # though the runner's fixed implementation does not use a
                # neural optimizer for this family.
                "logistic_C": 1.0,
                "logistic_solver": "lbfgs",
                "logistic_max_iter": 1000,
                "train_only_standardizer": True,
                "inverse_slice_count_weights": True,
                "checkpoint_selection": "none; fixed C=1 train fit only",
                "early_stopping": False,
            }
        )
    else:  # pragma: no cover - plan validation catches this first
        raise MatrixProtocolError(f"Unknown family {family!r}.")
    manifests = {str(split): str(path) for split, path in cell["manifests"].items()}
    return {
        "comparison": str(cell["cohort"]),
        "output_dir": str(cell["output_dir"]),
        "manifests": manifests,
        "training": training,
        "matrix_cell": {
            "cell_id": str(cell["cell_id"]),
            "cohort": str(cell["cohort"]),
            "family": family,
            "model": ACTUAL_MODELS[family],
            "modalities": list(modalities),
            "seed": int(cell["seed"]),
            "primary_observed_fit": bool(cell.get("analysis_roles", {}).get("primary_observed_fit", False)),
            "secondary_group_id": str(cell.get("analysis_roles", {}).get("secondary_group_id", "")),
        },
    }


def write_plan_artifacts(
    plan: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    plan_path: Path,
) -> None:
    """Persist a new plan, refusing to refresh an existing frozen plan.

    The wall-clock creation timestamp is allowed to differ on a repeated
    preflight, but all protocol, source, cell, and generated-config bindings
    must remain identical.  Existing files are compared byte-for-byte before
    any write so a resumed execution cannot silently acquire a new freeze.
    """

    plan_path = plan_path.resolve()
    existing_plan: Mapping[str, Any] | None = None
    if plan_path.is_file():
        loaded = _json_read(plan_path)
        if not isinstance(loaded, Mapping):
            raise MatrixProtocolError(f"Existing matrix plan {plan_path} is not a JSON object.")
        existing_plan = loaded
    for cell in plan["cells"]:
        config_path = Path(str(cell["config_path"]))
        expected_payload = _json_payload(_cell_config(cell, base_config=base_config, device=str(plan["device"])))
        expected_digest = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
        if config_path.is_file():
            actual_digest = _sha256(config_path)
            if actual_digest != expected_digest:
                raise MatrixProtocolError(
                    f"Existing cell config differs from the frozen plan: {config_path}; choose a new output root."
                )
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = config_path.with_name(config_path.name + ".tmp")
            temporary.write_text(expected_payload, encoding="utf-8", newline="\n")
            temporary.replace(config_path)
        # The generated config is an input binding, so store its digest after
        # writing it.  It is intentionally not included in the config itself
        # (which would create a self-referential hash).
        cell["config_sha256"] = expected_digest
    plan["config_hashes_complete"] = True
    if existing_plan is not None:
        immutable_keys = (
            "schema_version",
            "plan_id",
            "executable",
            "executable_required",
            "device",
            "source_build_root",
            "output_root",
            "base_config_path",
            "base_config_sha256",
            "training_protocol_path",
            "training_protocol_file_sha256",
            "training_protocol_amendment_path",
            "training_protocol_amendment_file_sha256",
            "source_build_hashes",
            "runner_code_hashes",
            "training_protocol_sha256",
            "training_protocol_amendment_sha256",
            "cohorts",
            "input_types",
            "family_seeds",
            "fit_count",
            "secondary_group_count",
            "secondary_groups",
            "cohort_summaries",
            "cells",
        )
        for key in immutable_keys:
            if existing_plan.get(key) != plan.get(key):
                raise MatrixProtocolError(
                    f"Existing matrix plan differs at {key!r}; choose a new output root instead of refreshing a freeze."
                )
        # Preserve the original freeze timestamp and execution state.  The
        # gate is regenerated separately from the immutable plan.
        return
    _json_write(plan_path, dict(plan))


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise MatrixProtocolError("PyYAML is required to read the base config.") from exc
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise MatrixProtocolError(f"Cannot read base config {path}.") from exc
    if not isinstance(value, dict):
        raise MatrixProtocolError(f"Base config {path} must contain a mapping.")
    return value


def execute_plan(
    plan: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    cell_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute every cell sequentially after a complete fail-closed preflight."""

    if Path(sys.executable).resolve() != ANDI_PYTHON.resolve():
        raise MatrixProtocolError(
            f"Matrix execution must use {ANDI_PYTHON}; current interpreter is {Path(sys.executable).resolve()}."
        )
    all_cells = [cell for cell in plan.get("cells", ()) if isinstance(cell, Mapping)]
    by_id = {str(cell.get("cell_id", "")): cell for cell in all_cells}
    selected_ids = [str(value) for value in cell_ids] if cell_ids else list(by_id)
    unknown = [value for value in selected_ids if value not in by_id]
    if unknown:
        raise MatrixProtocolError("Unknown matrix cell ID(s): " + ", ".join(unknown))
    if len(set(selected_ids)) != len(selected_ids):
        raise MatrixProtocolError("--cells must contain each cell ID at most once.")
    preflight: dict[str, Any] = {}
    for cell in all_cells:
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id", ""))
        preflight[cell_id] = _cell_execution_gate(cell, plan)
    results: list[dict[str, Any]] = []
    for cell_id in selected_ids:
        cell = by_id[cell_id]
        # Re-read every binding immediately before a subprocess launch.  The
        # initial snapshot is useful for audit context, but a control/config
        # or source file changing while a long matrix run is in progress must
        # affect the cell at its actual launch boundary.
        check = _cell_execution_gate(cell, plan)
        preflight[cell_id] = check
        if check["status"] != "PASS":
            # Local control prerequisites are explicit pending/blocked cells.
            # They are never reported as a successful matrix fit and do not
            # prevent unrelated ready cells (for example Logistic) from run.
            results.append(
                {
                    "cell_id": cell_id,
                    "command": [str(value) for value in cell["command"]],
                    "returncode": None,
                    "status": str(check["status"]),
                    "skipped": True,
                    "failures": list(check.get("failures", [])),
                    "preflight": check,
                }
            )
            continue
        existing = _verify_existing_fit(cell)
        if existing.get("status") == "PASS":
            results.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "command": [str(value) for value in cell["command"]],
                    "returncode": 0,
                    "status": "REUSED_EXISTING",
                    "reuse": existing,
                }
            )
            continue
        command = [str(value) for value in cell["command"]]
        cell_dir = Path(str(cell["output_dir"]))
        cell_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=dict(env or os.environ), check=False)
        record = {
            "cell_id": str(cell["cell_id"]),
            "command": command,
            "returncode": int(result.returncode),
            "status": "PASS" if result.returncode == 0 else "FAIL",
        }
        results.append(record)
        if result.returncode != 0:
            raise MatrixProtocolError(f"Matrix cell {cell['cell_id']} failed with exit code {result.returncode}.")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_ROOT / "outputs/diagnostics/domain_classifier/model_grid_v3_fullcandidate_20260917_final",
    )
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--amendment", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs/domain_classifier.yaml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Plan/output directory (defaults to <build-root>/matrix_v3_orchestration).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Run every cell whose local gate is ready; record blocked cells as skipped.")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Optional explicit cell IDs to execute from the frozen 112-cell plan.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan summary after writing deterministic artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_root = args.build_root if args.build_root.is_absolute() else REPO_ROOT / args.build_root
    build_root = build_root.resolve()
    protocol_path = args.protocol or build_root / "training_protocol.json"
    amendment_path = args.amendment or build_root / "training_protocol_amendment_20260917.json"
    output_root = args.output_root or build_root / "matrix_v3_orchestration"
    output_root = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    output_root = output_root.resolve()
    protocol = _json_read(protocol_path.resolve())
    amendment = _json_read(amendment_path.resolve()) if amendment_path.is_file() else None
    if not isinstance(protocol, Mapping) or amendment is not None and not isinstance(amendment, Mapping):
        raise MatrixProtocolError("Protocol and amendment must be JSON objects.")
    plan = build_matrix_plan(
        protocol,
        amendment=amendment,
        build_root=build_root,
        output_root=output_root,
        device=args.device,
    )
    base_config_path = args.base_config if args.base_config.is_absolute() else REPO_ROOT / args.base_config
    base_config = _load_yaml_or_json(base_config_path.resolve())
    plan["base_config_path"] = str(base_config_path.resolve())
    plan["base_config_sha256"] = _sha256(base_config_path.resolve())
    plan["training_protocol_path"] = str(protocol_path.resolve())
    plan["training_protocol_file_sha256"] = _sha256(protocol_path.resolve())
    plan["training_protocol_amendment_path"] = str(amendment_path.resolve()) if amendment_path.is_file() else None
    plan["training_protocol_amendment_file_sha256"] = _sha256(amendment_path.resolve()) if amendment_path.is_file() else None
    plan_path = output_root / "matrix_plan.json"
    write_plan_artifacts(plan, base_config=base_config, plan_path=plan_path)
    gate = execution_gate(
        build_root=build_root,
        stage_a_root=build_root / "stage_a_fomo45k_observed_v3_20260917",
        primary_null_root=build_root / "stage_a_fomo45k_observed_v3_20260917",
        plan=plan,
    )
    gate_path = output_root / "execution_gate.json"
    _json_write(gate_path, gate)
    if args.execute:
        if gate["status"] != "PASS":
            raise MatrixProtocolError(
                "Matrix execution is blocked: " + ", ".join(str(value) for value in gate["failures"])
            )
        results = execute_plan(
            plan,
            cell_ids=args.cells,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        completed = dict(plan)
        successful = sum(1 for result in results if result["status"] in {"PASS", "REUSED_EXISTING"})
        pending = sum(1 for result in results if result["status"] == "PENDING")
        blocked = sum(1 for result in results if result["status"] == "BLOCKED")
        selected_count = len(results)
        all_cells_selected = selected_count == int(plan["fit_count"])
        completed["status"] = (
            "COMPLETE"
            if all_cells_selected and successful == selected_count
            else "COMPLETE_SUBSET"
            if blocked == 0 and selected_count > 0
            else "PARTIAL"
        )
        completed["training_started"] = successful > 0
        completed["selected_cell_ids"] = [str(result["cell_id"]) for result in results]
        completed["execution_summary"] = {
            "selected": selected_count,
            "successful_or_reused": successful,
            "pending_skipped": pending,
            "blocked_skipped": blocked,
            "matrix_complete": completed["status"] == "COMPLETE",
        }
        completed["results"] = results
        _json_write(output_root / "matrix_execution.json", completed)
        print(json.dumps({"status": completed["status"], "selected_count": selected_count, "fit_count": successful, "pending": pending, "blocked": blocked, "output_root": str(output_root)}, indent=2))
        return 0
    summary = {
        "status": "PLANNED",
        "fit_count": int(plan["fit_count"]),
        "output_root": str(output_root),
        "matrix_plan": str(plan_path),
        "execution_gate": gate,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except MatrixProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
