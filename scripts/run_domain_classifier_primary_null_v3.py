"""Run one v3 primary observed fit and its fixed 199-draw paired null.

This entrypoint is intentionally separate from the active matrix orchestrator
and the historical calibration runner.  It validates a frozen v3 cohort
manifest, materializes canonical ``[3,128,128]`` tensors once, and gives the
same cache to the observed fit and every whole-pair label permutation.  A
completed observed fit and each completed null draw are immutable; interrupted
runs fail closed on partial artifacts rather than retraining or overwriting
them.

The command is suitable for the three new cohort primary cells.  The existing
FOMO Stage-A run has its own output identity and should not be replaced by this
entrypoint while it is active.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Set these before importing torch so a command-line invocation has the
# reproducible single-thread contract even when the caller did not export it.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import TrainConfig  # noqa: E402
from andi_rewrite.domain_classifier.v3_runtime import (  # noqa: E402
    CANONICAL_MODALITIES,
    DEFAULT_HEALTHY_LEDGER_SHA256,
    DEFAULT_INIT_SEED,
    DEFAULT_LABEL_STREAM_ROOT,
    DEFAULT_PRIMARY_REPLICATES,
    SPLITS,
    V3CachedInputs,
    V3InputValidationError,
    V3RuntimeError,
    fit_cached_v3_cell,
    permute_cached_v3_labels,
    run_fingerprint,
    set_single_thread_runtime,
    sha256_file,
    validate_and_materialize_v3_inputs,
    verify_source_freeze,
    write_v3_json,
)


PROTOCOL_ID = "domain_classifier_primary_null_v3"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_reference_config(path: Path | None, config: TrainConfig) -> dict[str, Any] | None:
    """Check the caller's frozen reference config against the primary contract.

    ``--config-path`` is a provenance/reference input.  The v3 primary runner
    resolves its own fixed execution config so a legacy smoke field such as
    ``permutation_replicates: 19`` cannot silently shorten the 199-draw null.
    Core architecture, split, optimization, and evaluation fields still must
    agree when they are present in the reference file.
    """

    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise V3RuntimeError(f"reference config is missing: {path}")
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError as exc:  # pragma: no cover - environment contract
        raise V3RuntimeError("PyYAML is required to validate --config-path") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise V3RuntimeError(f"cannot read reference config {path}") from exc
    if not isinstance(payload, Mapping):
        raise V3RuntimeError(f"reference config is not a mapping: {path}")
    training = payload.get("training", payload)
    if not isinstance(training, Mapping):
        raise V3RuntimeError(f"reference config has no training mapping: {path}")
    expected = {
        "model": str(config.model),
        "in_channels": int(config.in_channels),
        "num_classes": int(config.num_classes),
        "widths": list(config.widths),
        "groupnorm_groups": int(config.groupnorm_groups),
        "dropout": float(config.dropout),
        "weight_decay": float(config.weight_decay),
        "max_epochs": int(config.max_epochs),
        "patience": int(config.patience),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "threshold": float(config.threshold),
        "split_seed": int(config.split_seed),
        "stage": str(config.stage),
        "subject_method": str(config.subject_method),
        "no_augmentation": bool(config.no_augmentation),
        "bootstrap_replicates": int(config.bootstrap_replicates),
        "swap_replicates": int(config.swap_replicates),
    }
    mismatches: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key not in training:
            continue
        actual = training.get(key)
        if key == "widths":
            actual = list(actual) if isinstance(actual, (list, tuple)) else actual
        if isinstance(expected_value, float):
            try:
                agrees = float(actual) == expected_value
            except (TypeError, ValueError):
                agrees = False
        else:
            agrees = actual == expected_value
        if not agrees:
            mismatches[key] = {"expected": expected_value, "actual": actual}
    if mismatches:
        raise V3RuntimeError(
            "reference config disagrees with fixed v3 primary contract: "
            + _canonical_json(mismatches)
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "resolved_training": dict(training),
        "execution_config": asdict(config),
        "execution_fields_fixed_by_runner": [
            "seed",
            "modalities",
            "device",
            "permutation_replicates",
            "permutation_mode",
        ],
    }


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    import torch

    torch.save(value, temporary)
    temporary.replace(destination)


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe({key: value for key, value in result.items() if key != "state_dict"})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise V3RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a committed JSONL artifact and reject malformed rows."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise V3RuntimeError(f"JSONL row is not an object at {path}:{line_number}")
                rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3RuntimeError(f"cannot read JSONL artifact {path}") from exc
    return rows


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash committed artifacts for resume-time byte identity checks."""

    output: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise V3RuntimeError(f"expected committed artifact is missing: {path}")
        output[str(name)] = sha256_file(path)
    return output


def _snapshot_code(output_root: Path, code_paths: Sequence[Path]) -> dict[str, Any]:
    """Persist the small executable/source snapshot used by this run.

    Source MRI/LMDB/NPZ bytes are intentionally excluded.  Their selected
    tensor bytes are bound by ``input_binding_audit.json`` and large source
    files by the explicit prelaunch/metadata freeze.  Keeping the Python
    bytes here makes a later resume auditable even if the working tree moves.
    """

    snapshot_root = output_root / "code_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    normalized_paths = [Path(path).resolve() for path in code_paths if Path(path).is_file()]
    entries: list[dict[str, Any]] = []
    for path in sorted({str(item).lower(): item for item in normalized_paths}.values(), key=str):
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative = path.name
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = path.read_bytes()
        if destination.is_file():
            if destination.read_bytes() != source_bytes:
                raise V3RuntimeError(f"code snapshot differs from the current source: {path}")
        else:
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(source_bytes)
            temporary.replace(destination)
        entries.append({
            "source_path": str(path),
            "snapshot_path": str(destination),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size": len(source_bytes),
        })
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "entrypoint plus v3 runtime, model/metric/runner and canonical data readers; no MRI/LMDB/NPZ copied",
        "rows": entries,
    }
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V3RuntimeError(f"cannot read existing code snapshot {manifest_path}") from exc
        if _canonical_json(existing) != _canonical_json(manifest):
            raise V3RuntimeError("current executable/source bytes differ from the frozen code snapshot")
    else:
        write_v3_json(manifest_path, manifest, overwrite=False)
    return manifest


def _default_code_paths() -> list[Path]:
    """Return all Python implementations that can affect canonical inputs."""

    return [
        Path(__file__).resolve(),
        REPO_ROOT / "domain_classifier" / "v3_runtime.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "metrics.py",
        REPO_ROOT / "data" / "domain_classifier" / "readers.py",
        REPO_ROOT / "data" / "domain_classifier" / "records.py",
        REPO_ROOT / "data" / "domain_classifier" / "matching.py",
        REPO_ROOT / "data" / "domain_classifier" / "audit.py",
        REPO_ROOT / "data" / "robust_normalization.py",
        REPO_ROOT / "data" / "datasets" / "brats.py",
        REPO_ROOT / "data" / "datasets" / "imaging.py",
        REPO_ROOT / "data" / "datasets" / "common.py",
        REPO_ROOT / "data" / "datasets" / "lmdb.py",
        REPO_ROOT / "data" / "lmdb_io.py",
        REPO_ROOT / "data" / "imaging.py",
        REPO_ROOT / "data" / "healthy_slices.py",
        REPO_ROOT / "data" / "subject_splits.py",
    ]


def _verify_artifact_hashes(
    result: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    required: Sequence[str],
) -> None:
    recorded = result.get("artifact_hashes")
    if not isinstance(recorded, Mapping):
        raise V3RuntimeError("committed result has no artifact_hashes")
    for name in required:
        path = paths.get(name)
        expected = recorded.get(name)
        if path is None or not path.is_file() or not isinstance(expected, str) or not expected:
            raise V3RuntimeError(f"committed artifact hash is missing for {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise V3RuntimeError(f"committed artifact bytes drifted for {name}: {actual} != {expected}")


def _validate_prediction_rows(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_labels: Sequence[int] | None = None,
) -> None:
    """Check prediction-to-frozen-record joins before a result is reusable."""

    rows = _read_jsonl(path)
    if len(rows) != len(records):
        raise V3RuntimeError(
            f"prediction row count differs at {path}: {len(rows)} != {len(records)}"
        )
    if expected_labels is not None and len(expected_labels) != len(records):
        raise V3RuntimeError("expected prediction labels have an incompatible length")
    seen: set[int] = set()
    for position, (prediction, record) in enumerate(zip(rows, records)):
        try:
            record_index = int(prediction.get("record_index"))
        except (TypeError, ValueError) as exc:
            raise V3RuntimeError(f"prediction row {position} has no valid record_index") from exc
        if record_index != position or record_index in seen:
            raise V3RuntimeError(f"prediction record_index join failed at row {position}")
        seen.add(record_index)
        for field in ("participant_id", "pair_id", "case_id"):
            expected = record.get(field, position if field == "case_id" else None)
            if prediction.get(field) != expected:
                raise V3RuntimeError(f"prediction {field} join failed at row {position}")
        try:
            label = int(prediction.get("label"))
            probability = float(prediction.get("probability"))
        except (TypeError, ValueError) as exc:
            raise V3RuntimeError(f"prediction label/probability is invalid at row {position}") from exc
        if label not in (0, 1) or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise V3RuntimeError(f"prediction label/probability is out of range at row {position}")
        if expected_labels is not None and label != int(expected_labels[position]):
            raise V3RuntimeError(f"prediction label join failed at row {position}")


def _validate_subject_auc(result: Mapping[str, Any], *, context: str) -> float:
    test = result.get("test")
    subject = test.get("subject") if isinstance(test, Mapping) else None
    auc = subject.get("roc_auc") if isinstance(subject, Mapping) else None
    try:
        value = float(auc)
    except (TypeError, ValueError) as exc:
        raise V3RuntimeError(f"{context} has no finite subject AUC") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise V3RuntimeError(f"{context} subject AUC is outside [0,1]")
    return value


def _primary_config(*, device: str, init_seed: int, bootstrap_replicates: int = 2000) -> TrainConfig:
    return TrainConfig(
        model="small_cnn",
        in_channels=3,
        modalities=CANONICAL_MODALITIES,
        max_epochs=40,
        patience=8,
        early_stopping=True,
        dropout=0.0,
        weight_decay=1.0e-4,
        batch_size=32,
        num_workers=0,
        split_seed=73,
        seed=int(init_seed),
        stage="final",
        device=str(device),
        subject_method="mean",
        no_augmentation=True,
        bootstrap_replicates=int(bootstrap_replicates),
        swap_replicates=1000 if int(bootstrap_replicates) else 0,
        permutation_replicates=DEFAULT_PRIMARY_REPLICATES,
        permutation_mode="full_retrained_pair_swap_all_splits",
    )


def _freeze_identity(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in freeze.items() if key != "created_at_utc"}


def _ensure_protocol(
    output_root: Path,
    *,
    identity: Mapping[str, Any],
    cached: V3CachedInputs,
    config: TrainConfig,
    thread_contract: Mapping[str, Any],
    reference_config: Mapping[str, Any] | None,
    code_snapshot: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    """Write the immutable protocol once and reject source/config drift."""

    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_INPUTS_READY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": cached.comparison,
        "manifest_root": str(cached.manifest_root),
        "manifest_identity": cached.input_binding_audit.get("manifest_identity", {}),
        "config": asdict(config),
        "selection": "subject ROC-AUC then BCE on true validation labels",
        "primary_null": {
            "replicates": DEFAULT_PRIMARY_REPLICATES,
            "init_seed": int(config.seed),
            "label_stream_root": DEFAULT_LABEL_STREAM_ROOT,
            "scope": "TRAIN/VAL/TEST whole participant-pair swaps",
            "validation_selection": "draw-specific permuted validation labels",
            "statistic": "T=abs(subject_mean_score_roc_auc-0.5)",
            "p_value": "(1 + count(T_null >= T_observed)) / 200",
            "optional_stopping": False,
        },
        "input_binding_audit": cached.input_binding_audit,
        "source_freeze": cached.source_freeze,
        "thread_contract": dict(thread_contract),
        "reference_config": dict(reference_config) if reference_config is not None else None,
        "code_snapshot": {
            "manifest_path": str((protocol_path.parent / "code_snapshot" / "manifest.json").resolve()),
            "sha256": identity.get("code_snapshot_sha256"),
            "entries": len(code_snapshot.get("rows", [])),
        },
        "run_fingerprint": identity["run_fingerprint"],
        "training_started": False,
    }
    if protocol_path.is_file():
        try:
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V3RuntimeError(f"cannot read existing protocol {protocol_path}") from exc
        if existing.get("run_fingerprint") != identity["run_fingerprint"]:
            raise V3RuntimeError("existing v3 protocol has a different run fingerprint")
        if existing.get("comparison") != cached.comparison:
            raise V3RuntimeError("existing v3 protocol has a different comparison")
        previous_freeze = existing.get("source_freeze")
        if not isinstance(previous_freeze, Mapping):
            raise V3RuntimeError("existing v3 protocol has no source/config/data freeze")
        if _canonical_json(_freeze_identity(previous_freeze)) != _canonical_json(_freeze_identity(cached.source_freeze)):
            raise V3RuntimeError("current source/config/data freeze differs from existing protocol")
        previous_threads = existing.get("thread_contract")
        if isinstance(previous_threads, Mapping) and _canonical_json(previous_threads) != _canonical_json(thread_contract):
            raise V3RuntimeError("current thread contract differs from existing protocol")
        previous_code = existing.get("code_snapshot")
        if isinstance(previous_code, Mapping) and previous_code.get("sha256") != identity.get("code_snapshot_sha256"):
            raise V3RuntimeError("current code snapshot differs from existing protocol")
        previous_reference = existing.get("reference_config")
        if _canonical_json(previous_reference) != _canonical_json(reference_config):
            raise V3RuntimeError("current reference config differs from existing protocol")
        verify_source_freeze(previous_freeze)
        return dict(existing)
    write_v3_json(protocol_path, protocol, overwrite=False)
    return protocol


def _ensure_run_identity(output_root: Path, identity: Mapping[str, Any]) -> None:
    path = output_root / "run_identity.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V3RuntimeError(f"cannot read existing run identity {path}") from exc
        for key in ("run_fingerprint", "comparison", "requested_replicates", "init_seed"):
            if existing.get(key) != identity.get(key):
                raise V3RuntimeError(f"existing run identity differs at {key}")
        return
    write_v3_json(path, identity, overwrite=False)


def _cache_manifest(cached: V3CachedInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for identity in sorted(cached.tensors_by_identity, key=repr):
        rows.append({
            "identity": list(identity),
            "tensor_sha256": cached.tensor_digests_by_identity[identity],
            "shape": [3, 128, 128],
            "dtype": "torch.float32",
        })
    return {
        "schema_version": 1,
        "comparison": cached.comparison,
        "materialized_once": True,
        "canonical_modalities": list(CANONICAL_MODALITIES),
        "canonical_shape": [3, 128, 128],
        "canonical_dtype": "torch.float32",
        "unique_tensor_count": len(rows),
        "rows": rows,
        "source_cache_digest": hashlib.sha256(
            _canonical_json(rows).encode("utf-8")
        ).hexdigest(),
    }


def _write_input_artifacts(output_root: Path, cached: V3CachedInputs, protocol: Mapping[str, Any]) -> None:
    write_v3_json(output_root / "input_binding_audit.json", cached.input_binding_audit, overwrite=False)
    write_v3_json(output_root / "source_freeze.json", cached.source_freeze, overwrite=False)
    write_v3_json(output_root / "cache_manifest.json", _cache_manifest(cached), overwrite=False)
    # Keep a compact top-level identity that consumers can inspect without
    # opening the complete row audit.
    write_v3_json(
        output_root / "preflight.json",
        {
            "status": "PASS",
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "run_fingerprint": protocol.get("run_fingerprint"),
            "input_binding_status": cached.input_binding_audit.get("status"),
            "source_freeze_status": cached.source_freeze.get("status"),
            "ledger_sha256": cached.ledger.ledger_sha256,
            "unique_canonical_tensors": len(cached.tensors_by_identity),
        },
        overwrite=False,
    )


def _observed_paths(output_root: Path) -> dict[str, Path]:
    observed = output_root / "observed"
    return {
        "root": observed,
        "result": observed / "result.json",
        "checkpoint": observed / "model_best.pt",
        "predictions": observed / "test_predictions.jsonl",
        "validation_predictions": observed / "validation_predictions.jsonl",
    }


def _fit_observed(
    cached: V3CachedInputs,
    *,
    output_root: Path,
    config: TrainConfig,
    fingerprint: str,
) -> tuple[dict[str, Any], bool]:
    paths = _observed_paths(output_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["result"].is_file():
        try:
            value = json.loads(paths["result"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V3RuntimeError(f"cannot read observed result {paths['result']}") from exc
        if value.get("run_fingerprint") != fingerprint:
            raise V3RuntimeError("observed result belongs to a different frozen run")
        if not paths["predictions"].is_file() or not paths["validation_predictions"].is_file():
            raise V3RuntimeError("observed result exists but prediction artifacts are incomplete")
        if str(config.model).lower() not in {"logistic", "statistical", "statistical_logistic"} and not paths["checkpoint"].is_file():
            raise V3RuntimeError("observed neural result exists but model_best.pt is missing")
        required_hashes = ["test_predictions", "validation_predictions"]
        hash_paths = {
            "test_predictions": paths["predictions"],
            "validation_predictions": paths["validation_predictions"],
        }
        if str(config.model).lower() not in {"logistic", "statistical", "statistical_logistic"}:
            required_hashes.append("model_best")
            hash_paths["model_best"] = paths["checkpoint"]
        _verify_artifact_hashes(value, hash_paths, required=required_hashes)
        _validate_prediction_rows(
            paths["predictions"],
            cached.records_by_split["test"],
        )
        _validate_prediction_rows(
            paths["validation_predictions"],
            cached.records_by_split["val"],
        )
        _validate_subject_auc(value, context="observed result")
        return value, True
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = fit_cached_v3_cell(cached, config=config, seed=int(config.seed))
    elapsed = float(time.perf_counter() - started)
    state_dict = result.get("state_dict")
    if state_dict is not None:
        _atomic_torch_save(state_dict, paths["checkpoint"])
    _write_jsonl(paths["predictions"], result.get("test_predictions", []), overwrite=False)
    _write_jsonl(paths["validation_predictions"], result.get("validation_predictions", []), overwrite=False)
    hash_paths = {
        "test_predictions": paths["predictions"],
        "validation_predictions": paths["validation_predictions"],
    }
    if state_dict is not None:
        hash_paths["model_best"] = paths["checkpoint"]
    serializable = _safe_result(result)
    serializable.update({
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "run_fingerprint": fingerprint,
        "result_kind": "v3_primary_observed",
        "formal_final": False,
        "classification": "V3_PRIMARY_OBSERVED_EXCLUDED_FROM_CONFIRMATORY_UNTIL_NULL_AND_HOLM",
        "timing": {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
        },
        "input_binding": {
            "status": cached.input_binding_audit.get("status"),
            "cache_manifest": str((output_root / "cache_manifest.json").resolve()),
        },
        "artifact_hashes": _artifact_hashes(hash_paths),
    })
    write_v3_json(paths["result"], serializable, overwrite=False)
    return serializable, False


def _null_draw_dir(output_root: Path, index: int) -> Path:
    return output_root / "retrained_null" / f"permutation_{int(index):04d}"


def _null_draw_complete(
    path: Path,
    *,
    fingerprint: str,
    cached: V3CachedInputs | None = None,
    index: int | None = None,
    init_seed: int = DEFAULT_INIT_SEED,
) -> bool:
    result_path = path / "result.json"
    labels_path = path / "labels.jsonl"
    predictions_path = path / "test_predictions.jsonl"
    if not path.exists():
        return False
    present = [item.exists() for item in (result_path, labels_path, predictions_path)]
    if any(present) and not all(present):
        raise V3RuntimeError(f"partial null artifact requires inspection: {path}")
    if not all(present):
        return False
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3RuntimeError(f"cannot read null result {result_path}") from exc
    if value.get("run_fingerprint") != fingerprint:
        raise V3RuntimeError(f"null result belongs to a different frozen run: {result_path}")
    if index is None:
        try:
            index = int(path.name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise V3RuntimeError(f"cannot infer permutation index from {path}") from exc
    if int(value.get("permutation_index", -1)) != int(index):
        raise V3RuntimeError(f"null result index does not match its directory: {path}")
    if int(value.get("init_seed", -1)) != int(init_seed):
        raise V3RuntimeError(f"null result init seed does not match frozen init seed: {path}")
    if value.get("result_kind") != "v3_primary_full_retrained_pair_null":
        raise V3RuntimeError(f"null result has an unexpected result_kind: {path}")
    _validate_subject_auc(value, context=f"null result {path}")
    model_name = str((value.get("config") or {}).get("model", "small_cnn")).lower()
    required = ["labels", "test_predictions"]
    artifact_paths = {
        "labels": labels_path,
        "test_predictions": predictions_path,
    }
    if model_name not in {"logistic", "statistical", "statistical_logistic"}:
        required.append("model_best")
        checkpoint = path / "model_best.pt"
        artifact_paths["model_best"] = checkpoint
    _verify_artifact_hashes(value, artifact_paths, required=required)
    if cached is not None:
        permuted, streams, _detail = permute_cached_v3_labels(
            cached,
            index=int(index),
            label_stream_root=DEFAULT_LABEL_STREAM_ROOT,
        )
        expected_labels = _label_rows(
            cached,
            permuted,
            fingerprint=fingerprint,
            index=int(index),
        )
        actual_labels = _read_jsonl(labels_path)
        if _canonical_json(actual_labels) != _canonical_json(expected_labels):
            raise V3RuntimeError(f"null labels cannot be reconstructed from frozen pair streams: {labels_path}")
        if value.get("label_stream_seeds") != streams:
            raise V3RuntimeError(f"null label streams differ from deterministic seed streams: {path}")
        expected_swap_counts = _detail.get("swap_counts", {})
        if value.get("swap_counts") != expected_swap_counts:
            raise V3RuntimeError(f"null swap counts differ from deterministic pair streams: {path}")
        expected_label_values = [int(row.get("label")) for row in permuted["test"].records]
        _validate_prediction_rows(
            predictions_path,
            cached.records_by_split["test"],
            expected_labels=expected_label_values,
        )
    return True


def _label_rows(cached: V3CachedInputs, datasets: Mapping[str, Any], *, fingerprint: str, index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        dataset = datasets[split]
        for row_index, row in enumerate(dataset.records):
            rows.append({
                "run_fingerprint": fingerprint,
                "permutation_index": int(index),
                "split": split,
                "record_index": int(row_index),
                "participant_id": row.get("participant_id"),
                "pair_id": row.get("pair_id"),
                "case_id": row.get("case_id", row_index),
                "source_dataset": row.get("source_dataset"),
                "source_split": row.get("source_split"),
                "source_key": row.get("source_key"),
                "z": row.get("z"),
                "label": row.get("label"),
            })
    return rows


def _null_status(output_root: Path, *, completed: Sequence[int], fingerprint: str) -> dict[str, Any]:
    done = sorted(int(index) for index in completed)
    complete_indices = done == list(range(DEFAULT_PRIMARY_REPLICATES))
    return {
        "status": "complete" if complete_indices else "incomplete",
        "completed": len(done),
        "requested": DEFAULT_PRIMARY_REPLICATES,
        "completed_indices": done,
        "remaining": DEFAULT_PRIMARY_REPLICATES - len(done),
        "run_fingerprint": fingerprint,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "permutation_mode": "full_retrained_pair_swap_all_splits",
        "unit": "whole_pair_label_swap",
        "conditional_on": "frozen matched pairs",
    }


def _null_summary(output_root: Path, *, observed: Mapping[str, Any], completed: Sequence[int], fingerprint: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    indices = sorted(int(value) for value in completed)
    for index in sorted(int(value) for value in completed):
        path = _null_draw_dir(output_root, index) / "result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            validation_errors.append(f"cannot_read_result:{index}:{exc}")
            continue
        test = value.get("test") if isinstance(value.get("test"), Mapping) else {}
        subject = test.get("subject") if isinstance(test.get("subject"), Mapping) else {}
        auc = subject.get("roc_auc")
        try:
            numeric_auc = float(auc)
        except (TypeError, ValueError):
            numeric_auc = float("nan")
        if not np.isfinite(numeric_auc) or not 0.0 <= numeric_auc <= 1.0:
            validation_errors.append(f"invalid_subject_auc:{index}")
        rows.append({
            "index": index,
            "init_seed": value.get("init_seed", value.get("seed")),
            "label_stream_seeds": value.get("label_stream_seeds"),
            "swap_counts": value.get("swap_counts"),
            "epochs_completed": value.get("epochs_completed"),
            "selected_epoch": value.get("best_epoch"),
            "subject_auc": numeric_auc,
            "slice_auc": (test.get("slice") or {}).get("roc_auc") if isinstance(test.get("slice"), Mapping) else None,
        })
    observed_test = observed.get("test") if isinstance(observed.get("test"), Mapping) else {}
    observed_subject = observed_test.get("subject") if isinstance(observed_test.get("subject"), Mapping) else {}
    observed_auc = observed_subject.get("roc_auc")
    try:
        observed_auc_value = float(observed_auc)
    except (TypeError, ValueError):
        observed_auc_value = float("nan")
    if not np.isfinite(observed_auc_value) or not 0.0 <= observed_auc_value <= 1.0:
        validation_errors.append("invalid_observed_subject_auc")
    statistic: dict[str, Any] | None = None
    expected_indices = list(range(DEFAULT_PRIMARY_REPLICATES))
    complete_rows = (
        indices == expected_indices
        and len(rows) == DEFAULT_PRIMARY_REPLICATES
        and len({int(row["index"]) for row in rows}) == DEFAULT_PRIMARY_REPLICATES
        and not validation_errors
    )
    if complete_rows:
        observed_t = abs(observed_auc_value - 0.5)
        null_t = [abs(float(row["subject_auc"]) - 0.5) for row in rows]
        extreme = int(sum(value >= observed_t - 1.0e-15 for value in null_t))
        statistic = {
            "name": "full_retrained_pair_swap_test",
            "statistic": "T=abs(subject_mean_score_roc_auc-0.5)",
            "observed_subject_auc": observed_auc_value,
            "observed_T": observed_t,
            "null_T_values": null_t,
            "n_null": len(null_t),
            "n_null_at_least_observed_T": extreme,
            "p_plus_one": float((1 + extreme) / (1 + len(null_t))),
            "minimum_attainable_p": float(1.0 / (1 + len(null_t))),
            "conditional_on_fixed_matched_pairs": True,
            "label_scope": "train_val_test_complete_pair_swaps",
        }
    if len(indices) != len(set(indices)):
        validation_errors.append("duplicate_completed_indices")
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparison": observed.get("comparison"),
        "run_fingerprint": fingerprint,
        "status": "complete" if complete_rows else "incomplete",
        "requested": DEFAULT_PRIMARY_REPLICATES,
        "completed": len(rows),
        "statistics": statistic,
        "validation_errors": validation_errors,
        "rows": rows,
    }
    return summary


def run_primary_null_v3(
    *,
    comparison: str,
    manifest_root: str | Path,
    output_root: str | Path,
    build_root: str | Path | None = None,
    config_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    ledger_summary_path: str | Path | None = None,
    expected_ledger_sha256: str | None = DEFAULT_HEALTHY_LEDGER_SHA256,
    device: str = "cuda",
    init_seed: int = DEFAULT_INIT_SEED,
    replicates: int = DEFAULT_PRIMARY_REPLICATES,
    observed_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or resume one fixed v3 primary cohort output."""

    if int(replicates) != DEFAULT_PRIMARY_REPLICATES:
        raise V3RuntimeError(
            f"v3 primary null has a fixed {DEFAULT_PRIMARY_REPLICATES} replicates; optional stopping is not allowed"
        )
    if int(init_seed) != DEFAULT_INIT_SEED:
        raise V3RuntimeError(
            f"v3 primary null is frozen to init_seed={DEFAULT_INIT_SEED}; got {init_seed}"
        )
    thread_contract = set_single_thread_runtime()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _primary_config(device=device, init_seed=int(init_seed), bootstrap_replicates=2000)
    reference_config = _validate_reference_config(
        Path(config_path).resolve() if config_path is not None else None,
        config,
    )
    cached = validate_and_materialize_v3_inputs(
        manifest_root,
        comparison=comparison,
        build_root=build_root,
        ledger_path=ledger_path,
        ledger_summary_path=ledger_summary_path,
        expected_ledger_sha256=expected_ledger_sha256,
        config_path=config_path,
    )
    code_paths = _default_code_paths()
    # Freeze source bytes before any observed/null fit.  Re-running the
    # binding audit above is read-only; this snapshot is the immutable code
    # provenance boundary for both the first fit and every resume.
    code_snapshot = _snapshot_code(Path(output_root).resolve(), code_paths)
    fingerprint = run_fingerprint(
        comparison=cached.comparison,
        config=config,
        cached=cached,
        code_paths=code_paths,
        protocol_id=PROTOCOL_ID,
    )
    identity = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "run_fingerprint": fingerprint,
        "requested_replicates": DEFAULT_PRIMARY_REPLICATES,
        "init_seed": int(init_seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_identity": cached.input_binding_audit.get("manifest_identity", {}),
        "healthy_ledger_sha256": cached.ledger.ledger_sha256,
        "thread_contract": thread_contract,
        "reference_config": reference_config,
        "code_snapshot_sha256": hashlib.sha256(
            _canonical_json(code_snapshot).encode("utf-8")
        ).hexdigest(),
    }
    _ensure_run_identity(output, identity)
    protocol = _ensure_protocol(
        output,
        identity=identity,
        cached=cached,
        config=config,
        thread_contract=thread_contract,
        reference_config=reference_config,
        code_snapshot=code_snapshot,
        protocol_path=output / "protocol.json",
    )
    if not (output / "input_binding_audit.json").is_file():
        _write_input_artifacts(output, cached, protocol)
    elif (output / "source_freeze.json").is_file():
        existing_freeze = json.loads((output / "source_freeze.json").read_text(encoding="utf-8"))
        verify_source_freeze(existing_freeze)
    if dry_run:
        write_v3_json(
            output / "dry_run.json",
            {
                "status": "DRY_RUN",
                "protocol_id": PROTOCOL_ID,
                "comparison": cached.comparison,
                "run_fingerprint": fingerprint,
                "training_started": False,
                "null_started": False,
                "input_binding_status": cached.input_binding_audit.get("status"),
                "thread_contract": thread_contract,
                "reference_config": reference_config,
                "unique_canonical_tensors": len(cached.tensors_by_identity),
            },
            overwrite=True,
        )
        return {
            "status": "DRY_RUN",
            "comparison": cached.comparison,
            "output_root": str(output),
            "run_fingerprint": fingerprint,
            "training_started": False,
        }
    observed, reused = _fit_observed(cached, output_root=output, config=config, fingerprint=fingerprint)
    if observed_only:
        return {
            "status": "observed_reused" if reused else "observed_complete",
            "comparison": cached.comparison,
            "output_root": str(output),
            "run_fingerprint": fingerprint,
            "observed_result": str((_observed_paths(output))["result"]),
            "observed_reused": reused,
            "null_started": False,
        }
    null_root = output / "retrained_null"
    null_root.mkdir(parents=True, exist_ok=True)
    completed: list[int] = []
    for index in range(DEFAULT_PRIMARY_REPLICATES):
        if _null_draw_complete(
            _null_draw_dir(output, index),
            fingerprint=fingerprint,
            cached=cached,
            index=index,
            init_seed=int(init_seed),
        ):
            completed.append(index)
    write_v3_json(output / "retrained_null" / "status.json", _null_status(output, completed=completed, fingerprint=fingerprint), overwrite=True)
    null_config = replace(config, bootstrap_replicates=0, swap_replicates=0, permutation_replicates=0)
    for index in range(DEFAULT_PRIMARY_REPLICATES):
        if index in completed:
            continue
        draw_dir = _null_draw_dir(output, index)
        if draw_dir.exists() and any(draw_dir.iterdir()):
            raise V3RuntimeError(f"null draw directory is non-empty but incomplete: {draw_dir}")
        draw_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        permuted, streams, detail = permute_cached_v3_labels(
            cached,
            index=index,
            label_stream_root=DEFAULT_LABEL_STREAM_ROOT,
        )
        permuted_cached = copy.copy(cached)
        permuted_cached.datasets = permuted
        permuted_cached.records_by_split = {split: list(permuted[split].records) for split in SPLITS}
        result = fit_cached_v3_cell(permuted_cached, config=null_config, seed=int(init_seed))
        elapsed = float(time.perf_counter() - started)
        state_dict = result.get("state_dict")
        if state_dict is not None:
            _atomic_torch_save(state_dict, draw_dir / "model_best.pt")
        _write_jsonl(draw_dir / "test_predictions.jsonl", result.get("test_predictions", []), overwrite=False)
        _write_jsonl(draw_dir / "labels.jsonl", _label_rows(cached, permuted, fingerprint=fingerprint, index=index), overwrite=False)
        artifact_paths = {
            "labels": draw_dir / "labels.jsonl",
            "test_predictions": draw_dir / "test_predictions.jsonl",
        }
        if state_dict is not None:
            artifact_paths["model_best"] = draw_dir / "model_best.pt"
        serializable = _safe_result(result)
        serializable.update({
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "run_fingerprint": fingerprint,
            "result_kind": "v3_primary_full_retrained_pair_null",
            "formal_final": False,
            "permutation_index": int(index),
            "init_seed": int(init_seed),
            "label_stream_seeds": streams,
            "swap_counts": detail.get("swap_counts", {}),
            "label_scope": detail.get("label_scope"),
            "statistics_omitted": {"bootstrap_replicates": 0, "swap_replicates": 0},
            "timing": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
            },
            "labels_artifact": str((draw_dir / "labels.jsonl").resolve()),
            "artifact_hashes": _artifact_hashes(artifact_paths),
        })
        # Result is committed last so a crash cannot make an incomplete draw
        # look reusable on the next invocation.
        write_v3_json(draw_dir / "result.json", serializable, overwrite=False)
        completed.append(index)
        completed.sort()
        write_v3_json(output / "retrained_null" / "status.json", _null_status(output, completed=completed, fingerprint=fingerprint), overwrite=True)
    summary = _null_summary(output, observed=observed, completed=completed, fingerprint=fingerprint)
    write_v3_json(output / "retrained_null" / "summary.json", summary, overwrite=True)
    final = {
        "status": "complete" if len(completed) == DEFAULT_PRIMARY_REPLICATES else "incomplete",
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "output_root": str(output),
        "run_fingerprint": fingerprint,
        "observed_result": str((_observed_paths(output))["result"]),
        "observed_reused": reused,
        "null_status": _null_status(output, completed=completed, fingerprint=fingerprint),
        "null_summary": str((output / "retrained_null" / "summary.json").resolve()),
        "conditional_on": "frozen matched pairs",
    }
    write_v3_json(output / "summary.json", final, overwrite=True)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, choices=("fomo45k", "mpi", "oasis3", "mixed"))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--ledger-summary-path", type=Path)
    parser.add_argument("--expected-ledger-sha256", default=DEFAULT_HEALTHY_LEDGER_SHA256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-seed", type=int, default=DEFAULT_INIT_SEED)
    parser.add_argument("--replicates", type=int, default=DEFAULT_PRIMARY_REPLICATES)
    parser.add_argument("--observed-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    def resolve(value: Path | None) -> Path | None:
        if value is None:
            return None
        return value if value.is_absolute() else REPO_ROOT / value
    try:
        result = run_primary_null_v3(
            comparison=str(args.comparison),
            manifest_root=resolve(args.manifest_root) or args.manifest_root,
            output_root=resolve(args.output_root) or args.output_root,
            build_root=resolve(args.build_root),
            config_path=resolve(args.config_path),
            ledger_path=resolve(args.ledger_path),
            ledger_summary_path=resolve(args.ledger_summary_path),
            expected_ledger_sha256=str(args.expected_ledger_sha256) if args.expected_ledger_sha256 else None,
            device=str(args.device),
            init_seed=int(args.init_seed),
            replicates=int(args.replicates),
            observed_only=bool(args.observed_only),
            dry_run=bool(args.dry_run),
        )
    except (V3RuntimeError, V3InputValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") in {"complete", "observed_complete", "observed_reused", "DRY_RUN"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
