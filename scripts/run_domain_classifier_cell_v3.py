"""Run one frozen-input v3 domain-classifier cell.

The v3 matrix has one output directory per ``cohort/model/modalities/seed``.
This entrypoint is the small cell boundary used for the ordinary (non-primary
null) fits.  It binds a frozen manifest to the selected healthy tensor ledger,
materialises the canonical three-channel tensors once, projects modalities in
memory, and then delegates the fit to :mod:`domain_classifier.v3_runtime`.

The primary four cells have a separate 199-draw entrypoint and must not be
silently duplicated here.  A cell result is committed last; all source,
configuration, code, prediction, checkpoint, and logistic-state artifacts are
hashed so a resume either reuses a complete immutable fit or fails closed.

Example (the matrix writes one config per cell)::

    C:\\Users\\E-118-3\\miniconda3\\envs\\ANDi\\python.exe \\
      scripts\\run_domain_classifier_cell_v3.py \\
      --config outputs\\...\\config.yaml \\
      --manifest-root outputs\\...\\manifests\\mpi \\
      --output-dir outputs\\...\\cells\\mpi__logistic__flair__73\\fit \\
      --comparison mpi --stage final --device cuda \\
      --modalities flair --seeds 73
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Set process-level limits before importing torch through v3_runtime.  The
# helper also measures the resulting PyTorch pools and rejects a non-1 state.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from andi_rewrite.domain_classifier.runner import TrainConfig  # noqa: E402
from andi_rewrite.domain_classifier.v3_runtime import (  # noqa: E402
    CANONICAL_MODALITIES,
    DEFAULT_HEALTHY_LEDGER_SHA256,
    MODEL_SHAPE,
    SPLITS,
    V3CachedInputs,
    V3InputValidationError,
    V3RuntimeError,
    fit_cached_v3_cell,
    run_fingerprint,
    set_single_thread_runtime,
    sha256_file,
    validate_and_materialize_v3_inputs,
    verify_source_freeze,
    write_v3_json,
)


PROTOCOL_ID = "domain_classifier_cell_v3"
FIXED_SPLIT_SEED = 73
FIXED_BOOTSTRAP_REPLICATES = 2000
FIXED_BATCH_SIZE = 32
FIXED_EPOCHS = 40
FIXED_PATIENCE = 8
FIXED_WEIGHT_DECAY = 1.0e-4
FIXED_MODELS = {
    "statistical_logistic": "logistic",
    "logistic": "logistic",
    "statistical": "logistic",
    "small_cnn": "small_cnn",
    "cnn": "small_cnn",
    "small": "small_cnn",
    "resnet18": "resnet18",
    "resnet": "resnet18",
    "resnet_18": "resnet18",
}
ALLOWED_SEEDS = (73, 173, 273)


def _json_safe(value: Any) -> Any:
    """Convert numpy/torch/path values while dropping model tensors."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise V3RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3RuntimeError(f"cannot read JSON artifact {path}") from exc


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise V3RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise V3RuntimeError(f"missing JSONL artifact {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
    output: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise V3RuntimeError(f"expected artifact is missing: {path}")
        output[str(name)] = sha256_file(path)
    return output


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment contract
        raise V3RuntimeError("PyYAML is required to load --config") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise V3RuntimeError(f"cannot read YAML config {path}") from exc
    if not isinstance(value, Mapping):
        raise V3RuntimeError(f"config must contain a mapping at its root: {path}")
    return dict(value)


def _training_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    training = payload.get("training", payload)
    if not isinstance(training, Mapping):
        raise V3RuntimeError("config has no training mapping")
    return training


def _normalise_model(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    try:
        return FIXED_MODELS[key]
    except KeyError as exc:
        raise V3RuntimeError(f"unsupported v3 cell model {value!r}") from exc


def _normalise_modalities(value: Sequence[Any] | str | None) -> tuple[str, ...]:
    if value is None:
        return CANONICAL_MODALITIES
    if isinstance(value, str):
        values = tuple(item.strip().lower() for item in value.replace(",", " ").split() if item.strip())
    else:
        values = tuple(str(item).strip().lower() for item in value)
    if values not in {
        ("flair",),
        ("t1",),
        ("t2",),
        CANONICAL_MODALITIES,
    }:
        raise V3RuntimeError("modalities must be flair, t1, t2, or all three in canonical order")
    return values


def _coerce_train_kwargs(training: Mapping[str, Any]) -> dict[str, Any]:
    allowed = set(TrainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: item for key, item in training.items() if key in allowed}
    if "widths" in kwargs:
        kwargs["widths"] = tuple(int(item) for item in kwargs["widths"])
    if "modalities" in kwargs:
        kwargs["modalities"] = _normalise_modalities(kwargs["modalities"])
    return kwargs


def _config_from_yaml(
    payload: Mapping[str, Any],
    *,
    stage: str,
    device: str,
    modalities: Sequence[str] | None,
    seed: int,
) -> TrainConfig:
    """Resolve a matrix cell config and apply only the CLI cell identity."""

    training = _training_mapping(payload)
    kwargs = _coerce_train_kwargs(training)
    config_modalities = _normalise_modalities(modalities if modalities is not None else kwargs.get("modalities"))
    model = _normalise_model(kwargs.get("model", "small_cnn"))
    kwargs.update(
        {
            "model": "statistical_logistic" if model == "logistic" else model,
            "modalities": config_modalities,
            "in_channels": len(config_modalities),
            "stage": str(stage),
            "device": str(device),
            "seed": int(seed),
            "split_seed": int(kwargs.get("split_seed", FIXED_SPLIT_SEED)),
        }
    )
    config = TrainConfig(**kwargs)
    # Logistic C is not a TrainConfig field in the shared runner, so retain it
    # as a normal attribute for v3 contract validation and provenance.
    if "logistic_C" in training:
        try:
            config.logistic_C = float(training["logistic_C"])  # type: ignore[attr-defined]
        except (TypeError, ValueError) as exc:
            raise V3RuntimeError("logistic_C must be numeric") from exc
    _validate_cell_contract(config, training=training, seed=int(seed))
    return config


def _effective_learning_rate(config: TrainConfig) -> float:
    return float(config.resolved_learning_rate())


def _validate_cell_contract(
    config: TrainConfig,
    *,
    training: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> None:
    """Reject a config that would silently define another matrix cell."""

    model = _normalise_model(config.model)
    actual_seed = int(config.seed if seed is None else seed)
    if actual_seed not in ALLOWED_SEEDS:
        raise V3RuntimeError(f"cell seed must be one of {ALLOWED_SEEDS}; got {actual_seed}")
    if model == "logistic" and actual_seed != 73:
        raise V3RuntimeError("logistic cells are frozen to seed 73")
    if str(config.stage).strip().lower() != "final":
        raise V3RuntimeError("v3 cell stage is frozen to final")
    if int(config.split_seed) != FIXED_SPLIT_SEED:
        raise V3RuntimeError("v3 cell split_seed is frozen to 73")
    if int(config.batch_size) != FIXED_BATCH_SIZE:
        raise V3RuntimeError("v3 cell batch_size is frozen to 32")
    if len(config.modalities) != int(config.in_channels):
        raise V3RuntimeError("in_channels must equal the selected modality count")
    if bool(config.no_augmentation) is not True:
        raise V3RuntimeError("v3 cells require no_augmentation=true")
    if float(config.dropout) != 0.0:
        raise V3RuntimeError("v3 cells require dropout=0")
    if int(config.bootstrap_replicates) != FIXED_BOOTSTRAP_REPLICATES:
        raise V3RuntimeError("ordinary v3 cells require 2,000 subject/pair bootstrap replicates")
    if int(config.swap_replicates) != 0:
        raise V3RuntimeError("individual held-out swaps are disabled for ordinary cells; group swaps are separate")
    if int(config.permutation_replicates) != 0 or str(config.permutation_mode).lower() not in {"none", ""}:
        raise V3RuntimeError("ordinary v3 cells cannot run a per-cell permutation null")
    if model == "logistic":
        if float(getattr(config, "logistic_C", 1.0)) != 1.0:
            raise V3RuntimeError("logistic cells require fixed C=1")
        if float(config.weight_decay) != 0.0:
            raise V3RuntimeError("logistic cells require no neural weight decay")
        if bool(config.early_stopping):
            raise V3RuntimeError("logistic cells do not use neural early stopping")
    else:
        expected_lr = 1.0e-3 if model == "small_cnn" else 3.0e-4
        if not np.isclose(_effective_learning_rate(config), expected_lr, rtol=0.0, atol=1.0e-12):
            raise V3RuntimeError(f"{model} learning_rate must be {expected_lr}")
        if float(config.weight_decay) != FIXED_WEIGHT_DECAY:
            raise V3RuntimeError("neural v3 cells require weight_decay=1e-4")
        if int(config.max_epochs) != FIXED_EPOCHS or int(config.patience) != FIXED_PATIENCE:
            raise V3RuntimeError("neural v3 cells require 40 epochs and patience 8")
        if bool(config.early_stopping) is not True:
            raise V3RuntimeError("neural v3 cells require validation checkpoint selection")
    # The all-modality SmallCNN seed-73 coordinate is one of the primary four
    # cells and has a separate full-retrained-null protocol.  Refusing it here
    # prevents a matrix typo from producing an untracked duplicate fit.
    if model == "small_cnn" and tuple(config.modalities) == CANONICAL_MODALITIES and actual_seed == 73:
        raise V3RuntimeError(
            "the SmallCNN/all3/seed73 primary cell belongs to "
            "run_domain_classifier_primary_null_v3.py"
        )
    # If a config explicitly contains a fixed field, verify that it was not
    # silently replaced by a runner default.
    if training is not None:
        explicit_ints = {
            "batch_size": FIXED_BATCH_SIZE,
            "split_seed": FIXED_SPLIT_SEED,
            "bootstrap_replicates": FIXED_BOOTSTRAP_REPLICATES,
            "swap_replicates": 0,
            "permutation_replicates": 0,
        }
        for key, expected in explicit_ints.items():
            if key in training and int(training[key]) != expected:
                raise V3RuntimeError(f"config field {key} disagrees with the frozen cell contract")
        if "stage" in training and str(training["stage"]).lower() != "final":
            raise V3RuntimeError("config stage disagrees with the frozen cell contract")


def _code_paths() -> list[Path]:
    """Return code whose bytes can affect this cell or its canonical input."""

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


def _snapshot_code(output: Path, paths: Sequence[Path]) -> dict[str, Any]:
    snapshot_root = output / "code_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    required = {Path(__file__).resolve(), (REPO_ROOT / "domain_classifier" / "v3_runtime.py").resolve()}
    normalized = [Path(item).resolve() for item in paths if Path(item).is_file()]
    missing_required = sorted(str(path) for path in required if not path.is_file())
    if missing_required:
        raise V3RuntimeError("required executable source is missing: " + "; ".join(missing_required))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sorted(normalized, key=lambda item: str(item).lower()):
        key = str(source).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative = source.name
        destination = snapshot_root / relative
        source_bytes = source.read_bytes()
        if destination.is_file() and destination.read_bytes() != source_bytes:
            raise V3RuntimeError(f"code snapshot differs from current source: {source}")
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(source_bytes)
            temporary.replace(destination)
        entries.append(
            {
                "source_path": str(source),
                "snapshot_path": str(destination),
                "sha256": _sha256_bytes(source_bytes),
                "size": len(source_bytes),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "cell entrypoint plus v3 runtime/model/metric/runner and canonical readers; no MRI/LMDB/NPZ copied",
        "rows": entries,
    }
    if manifest_path.is_file():
        if _canonical_json(_read_json(manifest_path)) != _canonical_json(manifest):
            raise V3RuntimeError("current executable/source bytes differ from frozen code snapshot")
    else:
        _write_json(manifest_path, manifest)
    return manifest


def _freeze_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key not in {"created_at_utc", "updated_at_utc"}}


def _cache_manifest(cached: V3CachedInputs) -> dict[str, Any]:
    rows = [
        {
            "identity": list(identity),
            "tensor_sha256": cached.tensor_digests_by_identity[identity],
            "shape": list(MODEL_SHAPE),
            "dtype": "torch.float32",
        }
        for identity in sorted(cached.tensors_by_identity, key=repr)
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "materialized_once": True,
        "canonical_modalities": list(CANONICAL_MODALITIES),
        "canonical_shape": list(MODEL_SHAPE),
        "canonical_dtype": "torch.float32",
        "unique_tensor_count": len(rows),
        "rows": rows,
        "source_cache_digest": _sha256_bytes(_canonical_json(rows).encode("utf-8")),
    }


def _input_artifacts(output: Path) -> dict[str, Path]:
    return {
        "run_identity": output / "run_identity.json",
        "protocol": output / "protocol.json",
        "input_binding_audit": output / "input_binding_audit.json",
        "source_freeze": output / "source_freeze.json",
        "cache_manifest": output / "cache_manifest.json",
        "preflight": output / "preflight.json",
        "fingerprint": output / "fingerprint.json",
    }


def _prediction_paths(output: Path) -> dict[str, Path]:
    return {
        "test_predictions": output / "test_predictions.jsonl",
        "validation_predictions": output / "validation_predictions.jsonl",
        "train_final_predictions": output / "train_final_predictions.jsonl",
        "test_subject_predictions": output / "test_subject_predictions.jsonl",
        "validation_subject_predictions": output / "validation_subject_predictions.jsonl",
        "train_final_subject_predictions": output / "train_final_subject_predictions.jsonl",
    }


def _required_fit_paths(output: Path, *, logistic: bool) -> dict[str, Path]:
    paths = _prediction_paths(output)
    # This marker is committed once immediately before the fit begins.  It is
    # part of the immutable result bundle so a crash cannot leave a directory
    # that looks like frozen preparation only.
    paths["fit_started"] = output / "fit_started.json"
    paths["history"] = output / "history.json"
    paths["config"] = output / "config.json"
    if logistic:
        paths["logistic_state"] = output / "logistic_state.json"
    else:
        paths["model_best"] = output / "model_best.pt"
    return paths


def _prediction_records_valid(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> None:
    rows = _read_jsonl(path)
    # The shared neural runner only computes ``train_final`` for its explicit
    # tiny control.  Ordinary cells therefore persist an empty train-final
    # artifact as an intentional, hashed absence of that evaluation.
    if allow_empty and not rows:
        return
    if len(rows) != len(records):
        raise V3RuntimeError(f"prediction row count differs from frozen records: {path}")
    seen: set[int] = set()
    for index, (prediction, record) in enumerate(zip(rows, records)):
        try:
            record_index = int(prediction.get("record_index"))
            label = int(prediction.get("label"))
            probability = float(prediction.get("probability"))
        except (TypeError, ValueError) as exc:
            raise V3RuntimeError(f"invalid prediction join at {path}:{index}") from exc
        if record_index != index or record_index in seen:
            raise V3RuntimeError(f"prediction record_index join failed at {path}:{index}")
        seen.add(record_index)
        if label not in (0, 1) or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise V3RuntimeError(f"prediction label/probability is invalid at {path}:{index}")
        for field in ("participant_id", "pair_id", "case_id"):
            expected = record.get(field, index if field == "case_id" else None)
            if prediction.get(field) != expected:
                raise V3RuntimeError(f"prediction {field} join failed at {path}:{index}")
        try:
            expected_label = int(record.get("label"))
        except (TypeError, ValueError) as exc:
            raise V3RuntimeError(f"frozen record label is invalid at {path}:{index}") from exc
        if label != expected_label:
            raise V3RuntimeError(f"prediction label differs from frozen manifest at {path}:{index}")


def _subject_predictions_valid(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> None:
    rows = _read_jsonl(path)
    if allow_empty and not rows:
        return
    expected: dict[str, tuple[Any, Any]] = {}
    for record in records:
        participant = str(record.get("participant_id", "")).strip()
        if not participant:
            raise V3RuntimeError("frozen record has empty participant_id")
        label = int(record.get("label"))
        pair = record.get("pair_id")
        previous = expected.get(participant)
        if previous is not None and previous != (label, pair):
            raise V3RuntimeError(f"participant has inconsistent frozen label/pair: {participant}")
        expected[participant] = (label, pair)
    seen: set[str] = set()
    for row in rows:
        participant = str(row.get("participant_id", "")).strip()
        if not participant or participant in seen or participant not in expected:
            raise V3RuntimeError(f"subject prediction join failed at {path}: {participant!r}")
        seen.add(participant)
        if int(row.get("label")) != int(expected[participant][0]):
            raise V3RuntimeError(f"subject prediction label join failed at {path}:{participant}")
        if row.get("pair_id") != expected[participant][1]:
            raise V3RuntimeError(f"subject prediction pair join failed at {path}:{participant}")
        probability = float(row.get("probability"))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise V3RuntimeError(f"subject prediction probability is invalid at {path}:{participant}")


def _verify_committed_fit(
    output: Path,
    *,
    result: Mapping[str, Any],
    cached: V3CachedInputs,
    config: TrainConfig,
    fingerprint: str,
) -> dict[str, Any]:
    if result.get("protocol_id") != PROTOCOL_ID:
        raise V3RuntimeError("committed result has an unexpected protocol_id")
    if result.get("comparison") != cached.comparison:
        raise V3RuntimeError("committed result comparison differs from frozen input")
    if result.get("run_fingerprint") != fingerprint:
        raise V3RuntimeError("committed result belongs to a different frozen cell")
    if int(result.get("seed", -1)) != int(config.seed):
        raise V3RuntimeError("committed result seed differs from frozen cell")
    result_config = result.get("config")
    if not isinstance(result_config, Mapping):
        raise V3RuntimeError("committed result has no resolved config")
    if _canonical_json(result_config) != _canonical_json(asdict(config)):
        raise V3RuntimeError("committed result config differs from frozen cell")
    logistic = _normalise_model(config.model) == "logistic"
    paths = _required_fit_paths(output, logistic=logistic)
    recorded = result.get("artifact_hashes")
    if not isinstance(recorded, Mapping):
        raise V3RuntimeError("committed result has no artifact_hashes")
    for name, path in paths.items():
        expected = recorded.get(name)
        if not isinstance(expected, str) or not expected or not path.is_file():
            raise V3RuntimeError(f"committed artifact is missing or unhashed: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise V3RuntimeError(f"committed artifact bytes drifted for {name}: {actual} != {expected}")
    _prediction_records_valid(paths["test_predictions"], cached.records_by_split["test"])
    _prediction_records_valid(paths["validation_predictions"], cached.records_by_split["val"])
    _prediction_records_valid(paths["train_final_predictions"], cached.records_by_split["train"], allow_empty=True)
    _subject_predictions_valid(paths["test_subject_predictions"], cached.records_by_split["test"])
    _subject_predictions_valid(paths["validation_subject_predictions"], cached.records_by_split["val"])
    _subject_predictions_valid(paths["train_final_subject_predictions"], cached.records_by_split["train"], allow_empty=True)
    test = result.get("test") if isinstance(result.get("test"), Mapping) else {}
    subject = test.get("subject") if isinstance(test, Mapping) else {}
    try:
        auc = float(subject.get("roc_auc"))
    except (TypeError, ValueError) as exc:
        raise V3RuntimeError("committed result has no finite test subject AUC") from exc
    if not np.isfinite(auc) or not 0.0 <= auc <= 1.0:
        raise V3RuntimeError("committed result test subject AUC is outside [0,1]")
    return dict(result)


def _ensure_preflight_artifacts(
    output: Path,
    *,
    cached: V3CachedInputs,
    config: TrainConfig,
    identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    artifacts = _input_artifacts(output)
    expected = {
        "input_binding_audit": cached.input_binding_audit,
        "source_freeze": cached.source_freeze,
        "cache_manifest": _cache_manifest(cached),
        "preflight": {
            "status": "PASS",
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "run_fingerprint": identity["run_fingerprint"],
            "input_binding_status": cached.input_binding_audit.get("status"),
            "source_freeze_status": cached.source_freeze.get("status"),
            "ledger_sha256": cached.ledger.ledger_sha256,
            "unique_canonical_tensors": len(cached.tensors_by_identity),
            "training_started": False,
        },
        "fingerprint": {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_fingerprint": identity["run_fingerprint"],
            "comparison": cached.comparison,
            "config_sha256": _sha256_bytes(_canonical_json(asdict(config)).encode("utf-8")),
            "manifest_sha256": cached.manifest_fingerprints,
            "ledger_sha256": cached.ledger.ledger_sha256,
            "code_snapshot_sha256": identity["code_snapshot_sha256"],
        },
    }
    for key in ("input_binding_audit", "source_freeze", "cache_manifest", "preflight", "fingerprint"):
        path = artifacts[key]
        if not path.is_file():
            _write_json(path, expected[key])
            continue
        existing = _read_json(path)
        if key == "source_freeze":
            same = _canonical_json(_freeze_identity(existing)) == _canonical_json(_freeze_identity(expected[key]))
        else:
            same = _canonical_json(existing) == _canonical_json(expected[key])
        if not same:
            raise V3RuntimeError(f"frozen {key} differs from existing cell artifact")
    # Source bytes are checked on every invocation; large source files retain
    # the explicit metadata/selected-tensor scope supplied by v3_runtime.
    verify_source_freeze(cached.source_freeze)


def _ensure_identity_and_protocol(
    output: Path,
    *,
    cached: V3CachedInputs,
    config: TrainConfig,
    identity: Mapping[str, Any],
    thread_contract: Mapping[str, Any],
    code_snapshot: Mapping[str, Any],
    reference_config: Mapping[str, Any],
) -> None:
    identity_path = output / "run_identity.json"
    protocol_path = output / "protocol.json"
    expected_identity = dict(identity)
    if identity_path.is_file():
        existing = _read_json(identity_path)
        for key in ("protocol_id", "comparison", "seed", "model", "modalities", "run_fingerprint", "split_seed"):
            if existing.get(key) != expected_identity.get(key):
                raise V3RuntimeError(f"existing run identity differs at {key}")
    else:
        _write_json(identity_path, expected_identity)
    expected_protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_INPUTS_READY",
        "comparison": cached.comparison,
        "stage": config.stage,
        "model": config.model,
        "modalities": list(config.modalities),
        "seed": int(config.seed),
        "split_seed": int(config.split_seed),
        "config": asdict(config),
        "selection": "subject ROC-AUC then BCE on true validation labels",
        "bootstrap": {
            "replicates": FIXED_BOOTSTRAP_REPLICATES,
            "unit": "subjects; matched pairs when pair IDs are available",
        },
        "individual_pair_swap_replicates": 0,
        "input_binding_audit": cached.input_binding_audit,
        "source_freeze": cached.source_freeze,
        "thread_contract": dict(thread_contract),
        "reference_config": dict(reference_config),
        "code_snapshot": {
            "manifest_path": str((output / "code_snapshot" / "manifest.json").resolve()),
            "sha256": identity["code_snapshot_sha256"],
            "entries": len(code_snapshot.get("rows", [])),
        },
        "run_fingerprint": identity["run_fingerprint"],
        "training_started": False,
        "primary_cell": False,
    }
    if protocol_path.is_file():
        existing = _read_json(protocol_path)
        for key in ("protocol_id", "comparison", "stage", "model", "modalities", "seed", "run_fingerprint"):
            if existing.get(key) != expected_protocol.get(key):
                raise V3RuntimeError(f"existing cell protocol differs at {key}")
        previous_freeze = existing.get("source_freeze")
        if not isinstance(previous_freeze, Mapping) or _canonical_json(_freeze_identity(previous_freeze)) != _canonical_json(_freeze_identity(cached.source_freeze)):
            raise V3RuntimeError("existing cell source/config/data freeze differs")
        previous_threads = existing.get("thread_contract")
        if isinstance(previous_threads, Mapping) and _canonical_json(previous_threads) != _canonical_json(thread_contract):
            raise V3RuntimeError("existing cell thread contract differs")
        previous_code = existing.get("code_snapshot")
        if isinstance(previous_code, Mapping) and previous_code.get("sha256") != identity["code_snapshot_sha256"]:
            raise V3RuntimeError("existing cell code snapshot differs")
        verify_source_freeze(previous_freeze)
    else:
        _write_json(protocol_path, expected_protocol)


def _has_known_fit_artifacts(output: Path) -> bool:
    names = {
        "result.json",
        "model_best.pt",
        "logistic_state.json",
        "test_predictions.jsonl",
        "validation_predictions.jsonl",
        "history.json",
        "config.json",
        "fit_started.json",
    }
    return any((output / name).exists() for name in names)


def _reference_config(path: Path, config: TrainConfig) -> dict[str, Any]:
    payload = _load_yaml(path)
    training = _training_mapping(payload)
    # The resolved config is the execution contract; preserving the complete
    # source mapping makes it possible to audit fields not consumed by the
    # shared dataclass.
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source": payload,
        "resolved_training": dict(training),
        "execution_config": asdict(config),
    }


def run_cell_v3(
    *,
    config_path: str | Path,
    manifest_root: str | Path,
    output_dir: str | Path,
    stage: str = "final",
    device: str = "cuda",
    modalities: Sequence[str] | None = None,
    seeds: Sequence[int] = (73,),
    comparison: str | None = None,
    build_root: str | Path | None = None,
    ledger_path: str | Path | None = None,
    ledger_summary_path: str | Path | None = None,
    expected_ledger_sha256: str | None = DEFAULT_HEALTHY_LEDGER_SHA256,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run/resume one ordinary v3 cell, or bind it without fitting in dry-run."""

    if len(tuple(seeds)) != 1:
        raise V3RuntimeError("--seeds must specify exactly one seed for one cell")
    seed = int(tuple(seeds)[0])
    config_file = _resolve_path(config_path)
    if config_file is None or not config_file.is_file():
        raise V3RuntimeError(f"config is missing: {config_file}")
    payload = _load_yaml(config_file)
    config = _config_from_yaml(
        payload,
        stage=str(stage),
        device=str(device),
        modalities=modalities,
        seed=seed,
    )
    reference = _reference_config(config_file, config)
    thread_contract = set_single_thread_runtime()
    output = _resolve_path(output_dir)
    if output is None:
        raise V3RuntimeError("output-dir is required")
    output.mkdir(parents=True, exist_ok=True)
    existing_entries = list(output.iterdir())
    if existing_entries and not (output / "protocol.json").is_file() and not (output / "dry_run.json").is_file():
        raise V3RuntimeError(f"output directory is non-empty without a frozen protocol: {output}")
    code_snapshot = _snapshot_code(output, _code_paths())
    code_snapshot_sha = _sha256_bytes(_canonical_json(code_snapshot).encode("utf-8"))
    # A dry-run output is valid frozen preparation and may be continued; any
    # other partial fit artifacts are rejected before materialising new data.
    if existing_entries and not (output / "result.json").is_file() and not (output / "dry_run.json").is_file() and _has_known_fit_artifacts(output):
        raise V3RuntimeError(f"cell output is partial and cannot be resumed: {output}")
    cached = validate_and_materialize_v3_inputs(
        _resolve_path(manifest_root) or manifest_root,
        comparison=comparison,
        build_root=_resolve_path(build_root),
        ledger_path=_resolve_path(ledger_path),
        ledger_summary_path=_resolve_path(ledger_summary_path),
        expected_ledger_sha256=expected_ledger_sha256,
        config_path=config_file,
    )
    # Refuse the one primary coordinate after resolving canonical input, so a
    # caller cannot route a typo to this ordinary-cell protocol.
    _validate_cell_contract(config, seed=seed)
    fingerprint = run_fingerprint(
        comparison=cached.comparison,
        config=config,
        cached=cached,
        code_paths=_code_paths(),
        protocol_id=PROTOCOL_ID,
    )
    identity = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "model": config.model,
        "modalities": list(config.modalities),
        "seed": seed,
        "split_seed": int(config.split_seed),
        "stage": config.stage,
        "run_fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_identity": cached.input_binding_audit.get("manifest_identity", {}),
        "healthy_ledger_sha256": cached.ledger.ledger_sha256,
        "thread_contract": thread_contract,
        "reference_config": reference,
        "code_snapshot_sha256": code_snapshot_sha,
    }
    _ensure_identity_and_protocol(
        output,
        cached=cached,
        config=config,
        identity=identity,
        thread_contract=thread_contract,
        code_snapshot=code_snapshot,
        reference_config=reference,
    )
    _ensure_preflight_artifacts(output, cached=cached, config=config, identity=identity, protocol={})
    if dry_run:
        _write_json(
            output / "dry_run.json",
            {
                "status": "DRY_RUN",
                "protocol_id": PROTOCOL_ID,
                "comparison": cached.comparison,
                "model": config.model,
                "modalities": list(config.modalities),
                "seed": seed,
                "run_fingerprint": fingerprint,
                "training_started": False,
                "input_binding_status": cached.input_binding_audit.get("status"),
                "thread_contract": thread_contract,
                "unique_canonical_tensors": len(cached.tensors_by_identity),
                "projection": "fit_cached_v3_cell will use the same canonical tensor objects; no image reread",
            },
            overwrite=True,
        )
        return {
            "status": "DRY_RUN",
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "output_dir": str(output),
            "run_fingerprint": fingerprint,
            "training_started": False,
        }
    result_path = output / "result.json"
    if result_path.is_file():
        result = _verify_committed_fit(
            output,
            result=_read_json(result_path),
            cached=cached,
            config=config,
            fingerprint=fingerprint,
        )
        return {
            "status": "REUSED",
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "output_dir": str(output),
            "run_fingerprint": fingerprint,
            "result_path": str(result_path.resolve()),
            "seed": seed,
            "result": result,
        }
    if any((output / name).exists() for name in ("dry_run.json",)):
        # The dry-run marker is preparation only.  It is safe to continue after
        # the complete input artifacts have been checked above.
        pass
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    # Commit the execution boundary before invoking torch/sklearn.  This file
    # is deliberately never overwritten: an incomplete fit is inspected and
    # rejected on resume rather than silently retrained.
    fit_paths = _required_fit_paths(output, logistic=_normalise_model(config.model) == "logistic")
    _write_json(
        fit_paths["fit_started"],
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "model": config.model,
            "modalities": list(config.modalities),
            "seed": seed,
            "run_fingerprint": fingerprint,
            "started_at_utc": started_at,
            "source_freeze_status": cached.source_freeze.get("status"),
            "input_binding_status": cached.input_binding_audit.get("status"),
        },
    )
    fit_result = fit_cached_v3_cell(cached, config=config, seed=seed)
    elapsed = float(time.perf_counter() - started)
    if not isinstance(fit_result, Mapping):
        raise V3RuntimeError("fit_cached_v3_cell did not return a mapping")
    logistic = _normalise_model(config.model) == "logistic"
    # ``fit_paths`` includes the already committed execution marker.
    state_dict = fit_result.get("state_dict")
    if logistic:
        logistic_state = fit_result.get("logistic")
        if not isinstance(logistic_state, Mapping):
            raise V3RuntimeError("logistic fit did not return persisted logistic state")
        _write_json(fit_paths["logistic_state"], logistic_state)
    else:
        if state_dict is None:
            raise V3RuntimeError("neural fit did not return model state_dict")
        temporary = fit_paths["model_best"].with_name(fit_paths['model_best'].name + ".tmp")
        torch.save(state_dict, temporary)
        temporary.replace(fit_paths["model_best"])
    prediction_names = (
        "test_predictions",
        "validation_predictions",
        "train_final_predictions",
        "test_subject_predictions",
        "validation_subject_predictions",
        "train_final_subject_predictions",
    )
    for name in prediction_names:
        _write_jsonl(fit_paths[name], fit_result.get(name, []))
    _write_json(fit_paths["history"], fit_result.get("history", []))
    _write_json(
        fit_paths["config"],
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "resolved": asdict(config),
            "reference": reference,
        },
    )
    artifact_hashes = _artifact_hashes(fit_paths)
    serializable = _json_safe({key: value for key, value in fit_result.items() if key != "state_dict"})
    serializable.update(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "run_fingerprint": fingerprint,
            "result_kind": "v3_cell_observed",
            "formal_final": True,
            "control_mode": None,
            "classification": "V3_ORDINARY_CELL_OBSERVED",
            "model": serializable.get("model", {"class": config.model}),
            "seed": seed,
            "stage": config.stage,
            "modalities": list(config.modalities),
            "cell_identity": {
                "comparison": cached.comparison,
                "model": config.model,
                "modalities": list(config.modalities),
                "seed": seed,
                "split_seed": int(config.split_seed),
            },
            "timing": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
            },
            "input_binding": {
                "status": cached.input_binding_audit.get("status"),
                "cache_manifest": str((output / "cache_manifest.json").resolve()),
                "unique_canonical_tensors": len(cached.tensors_by_identity),
            },
            "artifact_hashes": artifact_hashes,
            "artifact_paths": {name: str(path.resolve()) for name, path in fit_paths.items()},
        }
    )
    # Validate the in-memory serialised rows before the final commit.  This
    # catches a runner/schema mismatch without leaving a result that appears
    # resumable.
    _verify_committed_fit_candidate(output, result=serializable, cached=cached, config=config, paths=fit_paths)
    _write_json(result_path, serializable)
    # The protocol's status is updated only after result.json is committed; a
    # crash before this point remains visibly FROZEN_INPUTS_READY.
    protocol_path = output / "protocol.json"
    protocol = _read_json(protocol_path)
    protocol["training_started"] = True
    protocol["status"] = "COMPLETE"
    protocol["result_path"] = str(result_path.resolve())
    protocol["result_artifact_hashes"] = dict(artifact_hashes)
    _write_json(protocol_path, protocol, overwrite=True)
    return {
        "status": "COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "comparison": cached.comparison,
        "output_dir": str(output),
        "run_fingerprint": fingerprint,
        "result_path": str(result_path.resolve()),
        "seed": seed,
        "training_started": True,
        "result": serializable,
    }


def _verify_committed_fit_candidate(
    output: Path,
    *,
    result: Mapping[str, Any],
    cached: V3CachedInputs,
    config: TrainConfig,
    paths: Mapping[str, Path],
) -> None:
    """Run the same join checks used by resume before result commit."""

    _prediction_records_valid(paths["test_predictions"], cached.records_by_split["test"])
    _prediction_records_valid(paths["validation_predictions"], cached.records_by_split["val"])
    _prediction_records_valid(paths["train_final_predictions"], cached.records_by_split["train"], allow_empty=True)
    _subject_predictions_valid(paths["test_subject_predictions"], cached.records_by_split["test"])
    _subject_predictions_valid(paths["validation_subject_predictions"], cached.records_by_split["val"])
    _subject_predictions_valid(paths["train_final_subject_predictions"], cached.records_by_split["train"], allow_empty=True)
    test = result.get("test") if isinstance(result.get("test"), Mapping) else {}
    subject = test.get("subject") if isinstance(test, Mapping) else {}
    try:
        auc = float(subject.get("roc_auc"))
    except (TypeError, ValueError) as exc:
        raise V3RuntimeError("fit result has no finite test subject AUC") from exc
    if not np.isfinite(auc) or not 0.0 <= auc <= 1.0:
        raise V3RuntimeError("fit result test subject AUC is outside [0,1]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("final",), default="final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modalities", nargs="+", choices=CANONICAL_MODALITIES)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--comparison", choices=("fomo45k", "mpi", "oasis3", "mixed"))
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--ledger-summary-path", type=Path)
    parser.add_argument("--expected-ledger-sha256", default=DEFAULT_HEALTHY_LEDGER_SHA256)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_cell_v3(
            config_path=args.config,
            manifest_root=args.manifest_root,
            output_dir=args.output_dir,
            stage=args.stage,
            device=args.device,
            modalities=args.modalities,
            seeds=args.seeds,
            comparison=args.comparison,
            build_root=args.build_root,
            ledger_path=args.ledger_path,
            ledger_summary_path=args.ledger_summary_path,
            expected_ledger_sha256=args.expected_ledger_sha256,
            dry_run=bool(args.dry_run),
        )
    except (V3RuntimeError, V3InputValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") in {"COMPLETE", "REUSED", "DRY_RUN"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
