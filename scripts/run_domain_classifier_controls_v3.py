"""Run the frozen Stage-B controls against the v3 input contract.

This is a bounded adapter around the existing control selection and training
code.  Final tiny and same-cohort-negative controls first pass through
``validate_and_materialize_v3_inputs``.  The validator reads each canonical
model-grid tensor once, checks the prelaunch healthy ledger and BraTS hashes,
and exposes in-memory modality projections.  Control selection then changes
only manifest rows; it never calls an image reader again.

Registered positive controls intentionally use the registered NIfTI paths.
They are a separate pre-IQR diagnostic and therefore record train-only scalar
normalisation and per-row pre/post digests without comparing those tensors to
the final healthy ledger.

The command accepts the old Stage-B planner's arguments, including the
compatibility ``--tiny`` flag and ``--retrained-permutations 0``.  It does not
run Mixed negative controls; those use the source-stratified v3 helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# These variables must be set before importing NumPy/PyTorch.  The same
# values are measured again by v3_runtime.set_single_thread_runtime().
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    DomainClassifierRunner,
    ManifestDataset,
    TrainConfig,
    dataset_from_manifest,
    dataset_with_shared_train_scalar,
    evaluate_negative_gate,
    evaluate_positive_gate,
    evaluate_tiny_gate,
    fit_shared_train_scalar,
    materialize_dataset,
    materialize_registered_dataset,
    read_jsonl_manifest,
    write_prediction_rows,
)
from andi_rewrite.domain_classifier.v3_runtime import (  # noqa: E402
    CANONICAL_MODALITIES,
    DEFAULT_HEALTHY_LEDGER_SHA256,
    MODEL_SHAPE,
    SPLITS,
    V3CachedInputs,
    V3InputValidationError,
    V3RuntimeError,
    run_fingerprint as v3_run_fingerprint,
    set_single_thread_runtime,
    sha256_file,
    validate_and_materialize_v3_inputs,
)

# These functions are the frozen Stage-A selection rules.  Keeping the
# selector itself shared avoids a second, subtly different tiny/negative
# sampling implementation in the Stage-B adapter.
from scripts.run_domain_classifier_controls import (  # noqa: E402
    _normal_neural_config,
    _same_cohort_negative_datasets,
    _tiny_datasets,
)


ANDI_PYTHON = Path(r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe")
PROTOCOL_ID = "model_grid_v3_stage_b_controls_v3"
STAGE_B_PROTOCOL = "model_grid_v3_stage_b_control_jobs_v1"
JOINT_MODALITIES = tuple(CANONICAL_MODALITIES)
FIXED_SPLIT_SEED = 73
TINY_SEED = 73
TINY_SUBJECTS_PER_LABEL = 10
TINY_MAX_SLICES = 128
TINY_MIN_SLICES = 64
TINY_MAX_EPOCHS = 300
FIXED_MAX_EPOCHS = 40
FIXED_PATIENCE = 8
FIXED_BATCH_SIZE = 32
FIXED_WEIGHT_DECAY = 1.0e-4
FIXED_BOOTSTRAP = 2000
FIXED_CONTROL_SEEDS = (73, 173, 273)
FAMILIES = {"small_cnn", "resnet18", "resnet", "resnet_18"}
MODEL_ALIASES = {
    "small": "small_cnn",
    "cnn": "small_cnn",
    "small_cnn": "small_cnn",
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
}


class ControlsV3Error(V3RuntimeError):
    """Raised when a Stage-B control cannot be bound fail-closed."""


class _RecordsOnlyDataset:
    """Manifest-only view used by the frozen selection helpers.

    The selector needs records but must not invoke an image reader.  The
    returned rows are mapped back to the validated canonical cache below.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(row) for row in records]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool = False, allow_nan: bool = False) -> None:
    if path.exists() and not overwrite:
        raise ControlsV3Error(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_immutable_json(path: Path, value: Any) -> None:
    """Create a sidecar once, or require a byte-equivalent canonical value."""

    normalized = _json_safe(value)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControlsV3Error(f"cannot read immutable artifact: {path}") from exc
        if _canonical_json(existing) != _canonical_json(normalized):
            raise ControlsV3Error(f"existing immutable artifact differs: {path}")
        return
    _write_json(path, normalized)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    lines = "".join(
        json.dumps(_json_safe(dict(row)), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != lines:
            raise ControlsV3Error(f"existing immutable manifest differs: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(lines, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return sha256_file(path)


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if callable(converter):
        return dict(converter())
    raise TypeError(f"control record must be a mapping, got {type(record).__name__}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ControlsV3Error("PyYAML is required for the Stage-B adapter") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ControlsV3Error(f"cannot read config {path}") from exc
    if not isinstance(payload, Mapping):
        raise ControlsV3Error(f"config must contain a mapping: {path}")
    return dict(payload)


def _training_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    training = payload.get("training", payload)
    if not isinstance(training, Mapping):
        raise ControlsV3Error("config has no training mapping")
    return training


def _normalise_model(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        raise ControlsV3Error(f"unsupported Stage-B neural model {value!r}") from exc


def _normalise_modalities(value: Sequence[Any] | str | None) -> tuple[str, ...]:
    if value is None:
        return JOINT_MODALITIES
    if isinstance(value, str):
        values = tuple(item.strip().lower() for item in value.replace(",", " ").split() if item.strip())
    else:
        values = tuple(str(item).strip().lower() for item in value)
    if values not in {
        ("flair",),
        ("t1",),
        ("t2",),
        JOINT_MODALITIES,
    }:
        raise ControlsV3Error("modalities must be flair, t1, t2, or all three in canonical order")
    return values


def _resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _infer_build_root(manifest_root: Path) -> Path:
    current = manifest_root.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "training_protocol.json").is_file() and (candidate / "protocol.json").is_file():
            return candidate
    if current.parent.name == "manifests":
        return current.parent.parent
    return current


def _config_from_payload(
    payload: Mapping[str, Any],
    *,
    mode: str,
    modalities: Sequence[str],
    stage: str,
    device: str,
    seed: int,
) -> TrainConfig:
    training = _training_mapping(payload)
    allowed = set(TrainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: item for key, item in training.items() if key in allowed}
    if "widths" in kwargs:
        kwargs["widths"] = tuple(int(item) for item in kwargs["widths"])
    model = _normalise_model(kwargs.get("model", "small_cnn"))
    values = _normalise_modalities(modalities)
    kwargs.update(
        {
            "model": model,
            "modalities": values,
            "in_channels": len(values),
            "stage": str(stage),
            "device": str(device),
            "seed": int(seed),
            "split_seed": FIXED_SPLIT_SEED,
            "batch_size": FIXED_BATCH_SIZE,
            "num_workers": 0,
            "no_augmentation": True,
            "dropout": 0.0,
            "bootstrap_replicates": FIXED_BOOTSTRAP,
            "permutation_replicates": 0,
            "permutation_mode": "none",
            "threshold": 0.5,
        }
    )
    if model == "small_cnn" and kwargs.get("learning_rate") is None:
        kwargs["learning_rate"] = 1.0e-3
    if model == "resnet18" and kwargs.get("learning_rate") is None:
        kwargs["learning_rate"] = 3.0e-4
    if mode == "tiny":
        kwargs.update(
            {
                "tiny": True,
                "early_stopping": False,
                "max_epochs": TINY_MAX_EPOCHS,
                "patience": TINY_MAX_EPOCHS + 1,
                "weight_decay": 0.0,
            }
        )
    else:
        kwargs.update(
            {
                "tiny": False,
                "early_stopping": True,
                "max_epochs": FIXED_MAX_EPOCHS,
                "patience": FIXED_PATIENCE,
                "weight_decay": FIXED_WEIGHT_DECAY,
            }
        )
    return TrainConfig(**kwargs)


def _validate_contract(
    *,
    mode: str,
    comparison: str,
    config: TrainConfig,
    modalities: Sequence[str],
    seeds: Sequence[int],
    tiny_seed: int,
    tiny_subjects_per_label: int,
    tiny_max_slices: int,
    tiny_min_slices: int,
    stage: str,
    fit_positive_shared_scalar: bool,
    retrained_permutations: int,
) -> None:
    if mode not in {"tiny", "positive", "negative"}:
        raise ControlsV3Error("Stage-B v3 adapter supports tiny, positive, and negative modes")
    if comparison == "mixed":
        raise ControlsV3Error(
            "Mixed negative controls use the dedicated source-stratified v3 helper "
            "run_domain_classifier_mixed_negative_control_v3.py"
        )
    if mode == "positive" and comparison != "fomo45k":
        raise ControlsV3Error("registered positive controls are frozen to the FOMO45K source")
    expected_stage = "registered" if mode == "positive" else "final"
    if str(stage).lower() != expected_stage:
        raise ControlsV3Error(f"{mode} control requires stage={expected_stage}")
    if config.stage != expected_stage:
        raise ControlsV3Error(f"resolved config stage must be {expected_stage}")
    model = _normalise_model(config.model)
    if model not in {"small_cnn", "resnet18"}:
        raise ControlsV3Error(f"unsupported control model {config.model!r}")
    if tuple(config.modalities) != tuple(modalities):
        raise ControlsV3Error("resolved modality projection differs from the requested control input")
    if int(config.split_seed) != FIXED_SPLIT_SEED or int(config.batch_size) != FIXED_BATCH_SIZE:
        raise ControlsV3Error("Stage-B controls require split_seed=73 and batch_size=32")
    if not bool(config.no_augmentation) or float(config.dropout) != 0.0:
        raise ControlsV3Error("Stage-B controls require no augmentation and dropout=0")
    if int(config.bootstrap_replicates) != FIXED_BOOTSTRAP:
        raise ControlsV3Error("Stage-B controls require 2,000 bootstrap replicates")
    if int(config.permutation_replicates) != 0 or str(config.permutation_mode).lower() not in {"none", ""}:
        raise ControlsV3Error("Stage-B controls do not run per-cell retrained permutations")
    if int(retrained_permutations) != 0:
        raise ControlsV3Error("Stage-B controls freeze --retrained-permutations to 0")
    if mode == "tiny":
        if tuple(int(seed) for seed in seeds) != (73,):
            raise ControlsV3Error("tiny control is frozen to fit seed 73")
        if int(tiny_seed) != TINY_SEED:
            raise ControlsV3Error("tiny selection is frozen to seed 73")
        if int(tiny_subjects_per_label) != TINY_SUBJECTS_PER_LABEL:
            raise ControlsV3Error("tiny selection is frozen to 10 subjects per label")
        if int(tiny_max_slices) != TINY_MAX_SLICES or int(tiny_min_slices) != TINY_MIN_SLICES:
            raise ControlsV3Error("tiny selection is frozen to 128 total slices and 64 minimum")
        if bool(config.early_stopping) or int(config.max_epochs) != TINY_MAX_EPOCHS or float(config.weight_decay) != 0.0:
            raise ControlsV3Error("tiny control requires 300 epochs without early stopping or weight decay")
    elif mode == "negative":
        if tuple(_normalise_modalities(modalities)) != JOINT_MODALITIES:
            raise ControlsV3Error("same-cohort negative control is frozen to all three modalities")
        if tuple(int(seed) for seed in seeds) != FIXED_CONTROL_SEEDS:
            raise ControlsV3Error("negative control is frozen to fit seeds 73, 173, and 273")
        if bool(config.tiny) or not bool(config.early_stopping) or int(config.max_epochs) != FIXED_MAX_EPOCHS:
            raise ControlsV3Error("negative control requires the ordinary 40-epoch validation protocol")
    else:
        if tuple(int(seed) for seed in seeds) != (73,):
            raise ControlsV3Error("registered positive control is frozen to fit seed 73")
        if not fit_positive_shared_scalar:
            raise ControlsV3Error("registered positive control requires the train-only shared scalar")
        if bool(config.tiny) or not bool(config.early_stopping) or int(config.max_epochs) != FIXED_MAX_EPOCHS:
            raise ControlsV3Error("registered positive control requires the ordinary 40-epoch protocol")


def _code_paths(config_path: Path) -> list[Path]:
    candidates = [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "run_domain_classifier_controls.py",
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
        config_path,
    ]
    return [path.resolve() for path in candidates if path.is_file()]


def _path_freeze(paths: Sequence[Path], *, metadata_only: Sequence[Path] = ()) -> dict[str, Any]:
    metadata_set = {path.resolve() for path in metadata_only}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((Path(value).resolve() for value in paths), key=lambda value: str(value).lower()):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise ControlsV3Error(f"freeze source is missing: {path}")
        stat = path.stat()
        if path in metadata_set:
            rows.append(
                {
                    "path": str(path),
                    "status": "PASS",
                    "sha256": None,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "hash_source": "metadata_only_registered_input",
                }
            )
        else:
            rows.append(
                {
                    "path": str(path),
                    "status": "PASS",
                    "sha256": sha256_file(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "hash_source": "bytes_read_before_fit",
                }
            )
    return {"status": "PASS", "rows": rows}


def _freeze_compare(value: Any) -> Any:
    """Drop timestamps from a freeze for immutable resume comparison."""

    if isinstance(value, Mapping):
        return {
            str(key): _freeze_compare(item)
            for key, item in value.items()
            if str(key) not in {"created_at_utc", "finished_at_utc", "started_at_utc"}
        }
    if isinstance(value, list):
        return [_freeze_compare(item) for item in value]
    return value


def _verify_path_freeze(freeze: Mapping[str, Any]) -> None:
    failures: list[str] = []
    for item in freeze.get("rows", []):
        if not isinstance(item, Mapping):
            failures.append("invalid_entry")
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            failures.append(f"missing:{path}")
            continue
        stat = path.stat()
        if int(item.get("size", -1)) != int(stat.st_size) or int(item.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            failures.append(f"metadata_drift:{path}")
            continue
        expected = item.get("sha256")
        if expected:
            actual = sha256_file(path)
            if actual != str(expected):
                failures.append(f"sha256_drift:{path}")
    if failures:
        raise ControlsV3Error("frozen Stage-B source drifted: " + "; ".join(failures[:8]))


def _write_or_verify_freeze(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_json(_freeze_compare(existing)) != _canonical_json(_freeze_compare(value)):
            raise ControlsV3Error(f"existing source freeze differs: {path}")
        _verify_path_freeze(existing)
        return dict(existing)
    _write_json(path, value)
    return dict(value)


def _records_by_split_from_cache(cached: V3CachedInputs) -> dict[str, _RecordsOnlyDataset]:
    return {
        split: _RecordsOnlyDataset(cached.records_by_split[split])
        for split in SPLITS
    }


def _cached_projected_view(
    cached: V3CachedInputs,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    modalities: Sequence[str],
) -> dict[str, Any]:
    transformed = cached.with_label_rows(rows_by_split)
    return {
        split: transformed[split].projected(tuple(modalities))
        for split in SPLITS
    }


def build_cached_tiny_datasets(
    cached: V3CachedInputs,
    *,
    subjects_per_label: int = TINY_SUBJECTS_PER_LABEL,
    max_slices: int = TINY_MAX_SLICES,
    min_slices: int = TINY_MIN_SLICES,
    seed: int = TINY_SEED,
    modalities: Sequence[str] = JOINT_MODALITIES,
) -> dict[str, Any]:
    """Run the frozen tiny selector over records and map it to cached tensors."""

    selected = _tiny_datasets(
        _records_by_split_from_cache(cached),
        subjects_per_label=int(subjects_per_label),
        max_slices=int(max_slices),
        min_slices=int(min_slices),
        seed=int(seed),
    )
    rows_by_split = {
        split: [_record_mapping(row) for row in selected[split].records]
        for split in SPLITS
    }
    return _cached_projected_view(cached, rows_by_split, modalities)


def build_cached_negative_datasets(
    cached: V3CachedInputs,
    *,
    seed: int = TINY_SEED,
    modalities: Sequence[str] = JOINT_MODALITIES,
) -> dict[str, Any]:
    """Run the existing non-Mixed same-cohort negative selector on cache rows."""

    selected = _same_cohort_negative_datasets(
        _records_by_split_from_cache(cached),
        seed=int(seed),
    )
    rows_by_split = {
        split: [_record_mapping(row) for row in selected[split].records]
        for split in SPLITS
    }
    return _cached_projected_view(cached, rows_by_split, modalities)


def _control_manifest_audit(datasets: Mapping[str, Any]) -> dict[str, Any]:
    split_audit: dict[str, Any] = {}
    for split in SPLITS:
        records = [_record_mapping(row) for row in getattr(datasets[split], "records", [])]
        participant_labels: dict[str, set[int]] = {}
        pair_ids: set[str] = set()
        for row in records:
            participant = str(row.get("participant_id", ""))
            participant_labels.setdefault(participant, set()).add(int(row.get("label", -1)))
            pair = str(row.get("pair_id", ""))
            if pair:
                pair_ids.add(pair)
        split_audit[split] = {
            "records": len(records),
            "participants": len(participant_labels),
            "pairs": len(pair_ids),
            "label_counts": {
                str(label): sum(1 for labels in participant_labels.values() if labels == {label})
                for label in (0, 1)
            },
            "conflicting_participants": sorted(
                participant for participant, labels in participant_labels.items() if len(labels) != 1
            ),
        }
    return split_audit


def _persist_control_manifests(destination: Path, datasets: Mapping[str, Any]) -> dict[str, Any]:
    manifest_dir = destination / "manifests"
    hashes: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = [_record_mapping(row) for row in getattr(datasets[split], "records", [])]
        digest = _write_jsonl(manifest_dir / f"{split}.jsonl", rows)
        hashes[split] = {"status": "PASS", "sha256": digest, "rows": len(rows)}
    audit = _control_manifest_audit(datasets)
    _write_immutable_json(destination / "split_audit.json", audit)
    return {
        "manifest_dir": str(manifest_dir.resolve()),
        "manifest_hashes": hashes,
        "split_audit": audit,
        "reused_across_fit_seeds": True,
    }


def _cache_artifacts(destination: Path, cached: V3CachedInputs) -> dict[str, str]:
    audit_path = destination / "input_binding_audit.json"
    _write_immutable_json(audit_path, cached.input_binding_audit)
    digest_rows = [
        {
            "identity": list(identity),
            "tensor_sha256": digest,
            "shape": list(MODEL_SHAPE),
            "dtype": "torch.float32",
        }
        for identity, digest in sorted(cached.tensor_digests_by_identity.items(), key=lambda item: repr(item[0]))
    ]
    digest_path = destination / "input_tensor_digest_ledger.json"
    _write_immutable_json(
        digest_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "comparison": cached.comparison,
            "source": "v3_runtime.validate_and_materialize_v3_inputs",
            "materialized_once": True,
            "unique_canonical_tensors": len(digest_rows),
            "rows": digest_rows,
        },
    )
    cache_path = destination / "cache_manifest.json"
    _write_immutable_json(
        cache_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "comparison": cached.comparison,
            "materialized_once": True,
            "unique_canonical_tensors": len(digest_rows),
            "rows": digest_rows,
        },
    )
    return {
        "input_binding_audit": str(audit_path.resolve()),
        "input_tensor_digest_ledger": str(digest_path.resolve()),
        "cache_manifest": str(cache_path.resolve()),
    }


def _row_tensor_digest(dataset: Any, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(getattr(dataset, "records", [])):
        sample = dataset[index]
        if not isinstance(sample, Mapping) or "image" not in sample:
            raise ControlsV3Error(f"registered cache sample is missing image: {split}[{index}]")
        tensor = torch.as_tensor(sample["image"], dtype=torch.float32).detach().cpu().contiguous()
        if tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
            raise ControlsV3Error(f"registered sample is not finite [C,H,W]: {split}[{index}]")
        row = _record_mapping(record)
        identity = [
            split,
            str(row.get("source_dataset", "")),
            str(row.get("source_split", "")),
            str(row.get("source_key", "")),
            str(row.get("participant_id", "")),
            str(row.get("case_id", "")),
            int(row.get("z", 0)),
        ]
        rows.append(
            {
                "identity": identity,
                "record_index": int(index),
                "tensor_sha256": _sha256_bytes(tensor.numpy().tobytes()),
                "shape": list(tensor.shape),
                "dtype": "torch.float32",
            }
        )
    return rows


def _aggregate_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(row) for row in rows), key=lambda row: _canonical_json(row.get("identity", [])))
    return _sha256_bytes(_canonical_json(ordered).encode("utf-8"))


def _registered_digest_audit(
    raw_datasets: Mapping[str, Any],
    scaled_datasets: Mapping[str, Any],
    *,
    scalar: float,
    modalities: Sequence[str],
) -> dict[str, Any]:
    pre_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    by_split: dict[str, Any] = {}
    for split in SPLITS:
        pre = _row_tensor_digest(raw_datasets[split], split)
        post = _row_tensor_digest(scaled_datasets[split], split)
        if len(pre) != len(post):
            raise ControlsV3Error(f"registered pre/post row count differs in {split}")
        merged: list[dict[str, Any]] = []
        for before, after in zip(pre, post):
            if before["identity"] != after["identity"]:
                raise ControlsV3Error(f"registered pre/post identity differs in {split}")
            merged.append(
                {
                    "identity": before["identity"],
                    "pre_scalar_sha256": before["tensor_sha256"],
                    "post_scalar_sha256": after["tensor_sha256"],
                    "shape": before["shape"],
                    "dtype": before["dtype"],
                }
            )
        by_split[split] = {"rows": merged, "pre_aggregate_sha256": _aggregate_digest(pre), "post_aggregate_sha256": _aggregate_digest(post)}
        pre_rows.extend(pre)
        post_rows.extend(post)
    return {
        "schema_version": 1,
        "status": "PASS",
        "stage": "registered",
        "modalities": list(modalities),
        "scalar": float(scalar),
        "scalar_fit": {"source_split": "train", "quantile": 0.995, "train_only": True},
        "rows_by_split": by_split,
        "pre_scalar_aggregate_sha256": _aggregate_digest(pre_rows),
        "post_scalar_aggregate_sha256": _aggregate_digest(post_rows),
        "final_ledger_comparison": "NOT_PERFORMED_BY_DESIGN_REGISTERED_PRE_IQR_CONTROL",
    }


def _registered_inputs_from_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Path]:
    paths: list[Path] = []
    for rows in rows_by_split.values():
        for row in rows:
            candidates = row.get("registered_paths") or row.get("image_paths")
            if not isinstance(candidates, Mapping):
                raise ControlsV3Error("registered control row lacks registered_paths/image_paths")
            for value in candidates.values():
                if value:
                    path = Path(str(value)).resolve()
                    if not path.is_file():
                        raise ControlsV3Error(f"registered input path is missing: {path}")
                    paths.append(path)
    return paths


def _build_source_freeze(
    *,
    mode: str,
    config_path: Path,
    manifest_paths: Mapping[str, Path],
    build_root: Path,
    cached: V3CachedInputs | None,
    registered_input_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    code = _path_freeze(_code_paths(config_path))
    manifest = _path_freeze(list(manifest_paths.values()))
    build_candidates = [
        build_root / name
        for name in (
            "protocol.json",
            "source_fingerprints.json",
            "build_summary.json",
            "training_protocol.json",
            "training_protocol_amendment_20260917.json",
        )
        if (build_root / name).is_file()
    ]
    registered = _path_freeze(registered_input_paths, metadata_only=registered_input_paths) if registered_input_paths else {"status": "NOT_APPLICABLE", "rows": []}
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "manifest": manifest,
        "build": _path_freeze(build_candidates) if build_candidates else {"status": "NOT_APPLICABLE", "rows": []},
        "registered_inputs": registered,
        "v3_source_freeze": cached.source_freeze if cached is not None else None,
        "scope": "executable/config/manifest/build bytes and registered input metadata frozen before fit; selected final tensor bytes bound by v3 ledger",
    }


def _control_fingerprint(
    *,
    mode: str,
    comparison: str,
    config: TrainConfig,
    source_freeze: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
    cached: V3CachedInputs | None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage_b_protocol": STAGE_B_PROTOCOL,
        "mode": mode,
        "comparison": comparison,
        "config": _json_safe(config.__dict__),
        "source_freeze": _freeze_compare(source_freeze),
        "control_manifest": _freeze_compare(control_manifest),
        "cache_manifest_sha256": (
            _sha256_bytes(_canonical_json(cached.tensor_digests_by_identity).encode("utf-8"))
            if cached is not None
            else None
        ),
        "extra": _json_safe(extra or {}),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _identity_path_check(path: Path, identity: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        keys = ("protocol_id", "mode", "comparison", "model", "modalities", "stage", "run_fingerprint")
        for key in keys:
            if existing.get(key) != identity.get(key):
                raise ControlsV3Error(f"existing Stage-B identity differs at {key}")
    else:
        _write_json(path, identity)


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ControlsV3Error(f"fit artifact is missing: {path}")
        output[str(name)] = sha256_file(path)
    return output


def _fit_paths(destination: Path) -> dict[str, Path]:
    return {
        "model": destination / "model_best.pt",
        "predictions": destination / "test_predictions.jsonl",
        "validation_predictions": destination / "validation_predictions.jsonl",
        "train_final_predictions": destination / "train_final_predictions.jsonl",
        "subject_predictions": destination / "test_subject_predictions.jsonl",
        "validation_subject_predictions": destination / "validation_subject_predictions.jsonl",
        "train_final_subject_predictions": destination / "train_final_subject_predictions.jsonl",
        "history": destination / "history.json",
        "config": destination / "config.json",
    }


def _verify_existing_seed(
    artifact: Path,
    *,
    fingerprint: str,
    checkpoint: Path,
    predictions: Path,
) -> dict[str, Any]:
    try:
        result = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlsV3Error(f"cannot read existing control result: {artifact}") from exc
    if result.get("run_fingerprint") != fingerprint:
        raise ControlsV3Error(f"existing control result belongs to another run: {artifact}")
    if not checkpoint.is_file() or not predictions.is_file():
        raise ControlsV3Error(f"existing control result is incomplete: {artifact}")
    recorded = result.get("artifact_hashes")
    if isinstance(recorded, Mapping):
        for name, path in (("model", checkpoint), ("predictions", predictions)):
            expected = recorded.get(name)
            if expected and sha256_file(path) != str(expected):
                raise ControlsV3Error(f"existing control artifact drifted: {path}")
    return dict(result)


def _fit_neural_seeds(
    datasets: Mapping[str, Any],
    *,
    config: TrainConfig,
    seeds: Sequence[int],
    destination: Path,
    fingerprint: str,
    identity: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    source_freeze_path: Path,
) -> list[dict[str, Any]]:
    runner = DomainClassifierRunner(config)
    results: list[dict[str, Any]] = []
    for seed in (int(value) for value in seeds):
        artifact = destination / f"seed_{seed}.json"
        checkpoint = destination / f"model_seed_{seed}.pt"
        predictions = destination / f"seed_{seed}_test_predictions.jsonl"
        if artifact.exists():
            results.append(_verify_existing_seed(artifact, fingerprint=fingerprint, checkpoint=checkpoint, predictions=predictions))
            continue
        if checkpoint.exists() or predictions.exists():
            raise ControlsV3Error(f"partial control artifacts exist for seed {seed}: {destination}")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        fit = runner.train_one_seed(
            datasets["train"],
            datasets["val"],
            datasets["test"],
            seed=seed,
        )
        state = fit.get("state_dict")
        if state is None:
            raise ControlsV3Error(f"control fit returned no model state for seed {seed}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        write_prediction_rows(predictions, fit.get("test_predictions", []))
        result = _json_safe({key: value for key, value in fit.items() if key != "state_dict"})
        result.update(
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "control_protocol": STAGE_B_PROTOCOL,
                "control_type": str(identity["control_type"]),
                "mode": str(identity["mode"]),
                "comparison": str(identity["comparison"]),
                "cohort": str(identity["cohort"]),
                "model": str(identity["model"]),
                "modalities": list(identity["modalities"]),
                "stage": str(identity["stage"]),
                "fit_seed": seed,
                "run_fingerprint": fingerprint,
                "control_manifest": dict(control_manifest),
                "input_binding": dict(input_binding),
                "source_freeze": str(source_freeze_path.resolve()),
                "timing": {
                    "started_at_utc": started_at,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": float(time.perf_counter() - started),
                },
            }
        )
        artifact_hashes = {
            "model": sha256_file(checkpoint),
            "predictions": sha256_file(predictions),
        }
        result["artifact_hashes"] = artifact_hashes
        _write_json(artifact, result, allow_nan=True)
        results.append(result)
    return results


def _registered_fit_datasets(
    *,
    manifest_paths: Mapping[str, Path],
    modalities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    base = {
        split: dataset_from_manifest(
            manifest_paths[split],
            stage="registered",
            modalities=tuple(modalities),
            shared_train_scalar=None,
        )
        for split in SPLITS
    }
    raw = {
        split: materialize_registered_dataset(base[split])
        for split in SPLITS
    }
    train = materialize_dataset(raw["train"])
    scalar = fit_shared_train_scalar(train["images"])
    scaled = {
        split: dataset_with_shared_train_scalar(raw[split], scalar)
        for split in SPLITS
    }
    digest_audit = _registered_digest_audit(raw, scaled, scalar=scalar, modalities=modalities)
    return raw, scaled, scalar, digest_audit


def run_control(
    *,
    config_path: str | Path,
    manifest_root: str | Path,
    output_dir: str | Path,
    mode: str,
    comparison: str,
    build_root: str | Path | None = None,
    ledger_path: str | Path | None = None,
    ledger_summary_path: str | Path | None = None,
    expected_ledger_sha256: str | None = DEFAULT_HEALTHY_LEDGER_SHA256,
    stage: str | None = None,
    device: str = "cuda",
    modalities: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    tiny_seed: int = TINY_SEED,
    tiny_subjects_per_label: int = TINY_SUBJECTS_PER_LABEL,
    tiny_max_slices: int = TINY_MAX_SLICES,
    tiny_min_slices: int = TINY_MIN_SLICES,
    fit_positive_shared_scalar: bool = False,
    retrained_permutations: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    if Path(sys.executable).resolve() != ANDI_PYTHON.resolve():
        raise ControlsV3Error(f"Stage-B controls require the ANDi interpreter: {ANDI_PYTHON}")
    normalized_mode = str(mode).strip().lower()
    normalized_comparison = {"fomo": "fomo45k", "fomo45k": "fomo45k", "mpi": "mpi", "oasis": "oasis3", "oasis3": "oasis3", "mixed": "mixed"}.get(str(comparison).lower())
    if normalized_comparison is None:
        raise ControlsV3Error(f"unsupported comparison {comparison!r}")
    manifest_dir = _resolve_path(manifest_root)
    if not manifest_dir.is_dir():
        raise ControlsV3Error(f"manifest root is not a directory: {manifest_dir}")
    manifest_paths = {split: manifest_dir / f"{split}.jsonl" for split in SPLITS}
    if any(not path.is_file() for path in manifest_paths.values()):
        raise ControlsV3Error("Stage-B manifest root is missing one or more split JSONL files")
    config_file = _resolve_path(config_path)
    if not config_file.is_file():
        raise ControlsV3Error(f"config is missing: {config_file}")
    payload = _load_yaml(config_file)
    requested_modalities = _normalise_modalities(modalities)
    resolved_stage = "registered" if normalized_mode == "positive" else "final"
    if stage is not None and str(stage).lower() != resolved_stage:
        raise ControlsV3Error(f"{normalized_mode} control requires --stage {resolved_stage}")
    selected_seeds = tuple(
        int(value)
        for value in (
            seeds
            if seeds is not None
            else ((73,) if normalized_mode in {"tiny", "positive"} else FIXED_CONTROL_SEEDS)
        )
    )
    config = _config_from_payload(
        payload,
        mode=normalized_mode,
        modalities=requested_modalities,
        stage=resolved_stage,
        device=device,
        seed=selected_seeds[0] if selected_seeds else 73,
    )
    _validate_contract(
        mode=normalized_mode,
        comparison=normalized_comparison,
        config=config,
        modalities=requested_modalities,
        seeds=selected_seeds,
        tiny_seed=int(tiny_seed),
        tiny_subjects_per_label=int(tiny_subjects_per_label),
        tiny_max_slices=int(tiny_max_slices),
        tiny_min_slices=int(tiny_min_slices),
        stage=resolved_stage,
        fit_positive_shared_scalar=bool(fit_positive_shared_scalar),
        retrained_permutations=int(retrained_permutations),
    )
    output = _resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_build = _resolve_path(build_root) if build_root is not None else _infer_build_root(manifest_dir)
    thread_contract = set_single_thread_runtime()
    if thread_contract.get("status") != "PASS":
        raise ControlsV3Error("single-thread runtime contract did not pass")

    cached: V3CachedInputs | None = None
    registered_digest_audit: dict[str, Any] | None = None
    input_binding_paths: dict[str, str] = {}
    if normalized_mode in {"tiny", "negative"}:
        if normalized_comparison == "mixed":
            raise ControlsV3Error("Mixed negative controls use the dedicated source-stratified v3 helper")
        validation_kwargs: dict[str, Any] = {
            "comparison": normalized_comparison,
            "build_root": resolved_build,
            "config_path": config_file,
            "expected_ledger_sha256": expected_ledger_sha256,
        }
        if ledger_path is not None:
            validation_kwargs["ledger_path"] = _resolve_path(ledger_path)
        if ledger_summary_path is not None:
            validation_kwargs["ledger_summary_path"] = _resolve_path(ledger_summary_path)
        cached = validate_and_materialize_v3_inputs(manifest_dir, **validation_kwargs)
        input_binding_paths = _cache_artifacts(output, cached)
        source_freeze_value = _build_source_freeze(
            mode=normalized_mode,
            config_path=config_file,
            manifest_paths=manifest_paths,
            build_root=resolved_build,
            cached=cached,
        )
        source_freeze_path = output / "source_freeze.json"
        source_freeze = _write_or_verify_freeze(source_freeze_path, source_freeze_value)
        if normalized_mode == "tiny":
            datasets = build_cached_tiny_datasets(
                cached,
                subjects_per_label=tiny_subjects_per_label,
                max_slices=tiny_max_slices,
                min_slices=tiny_min_slices,
                seed=tiny_seed,
                modalities=requested_modalities,
            )
            control_type = "tiny_overfit"
            extra = {
                "selection": "existing_frozen_tiny_selector",
                "tiny_seed": int(tiny_seed),
                "subjects_per_label": int(tiny_subjects_per_label),
                "tiny_max_slices": int(tiny_max_slices),
                "tiny_min_slices": int(tiny_min_slices),
                "canonical_cache_materialized_once": True,
            }
        else:
            datasets = build_cached_negative_datasets(cached, seed=tiny_seed, modalities=requested_modalities)
            control_type = "same_cohort_negative"
            extra = {
                "selection": "existing_frozen_same_cohort_negative_selector",
                "negative_seed": int(tiny_seed),
                "source_label": 0,
                "participant_split_fractions": [0.40, 0.10, 0.50],
                "canonical_cache_materialized_once": True,
                "same_control_manifest_reused_across_fit_seeds": True,
            }
    else:
        # Freeze code/config/manifests and registered source metadata before
        # reading any NIfTI.  The registered path reader is intentionally a
        # separate positive-control data path.
        source_rows = {split: read_jsonl_manifest(path) for split, path in manifest_paths.items()}
        registered_paths = _registered_inputs_from_rows(source_rows)
        source_freeze_value = _build_source_freeze(
            mode=normalized_mode,
            config_path=config_file,
            manifest_paths=manifest_paths,
            build_root=resolved_build,
            cached=None,
            registered_input_paths=registered_paths,
        )
        source_freeze_path = output / "source_freeze.json"
        source_freeze = _write_or_verify_freeze(source_freeze_path, source_freeze_value)
        raw, datasets, scalar, registered_digest_audit = _registered_fit_datasets(
            manifest_paths=manifest_paths,
            modalities=requested_modalities,
        )
        del raw
        config = TrainConfig(**{**config.__dict__, "shared_train_scalar": float(scalar)})
        _write_immutable_json(output / "registered_input_digest_ledger.json", registered_digest_audit)
        control_type = "positive_registered"
        extra = {
            "selection": "registered_paths_materialize_once",
            "registered_cache_materialized_once": True,
            "shared_train_scalar": float(scalar),
            "scalar_fit_train_only": True,
            "final_ledger_comparison": "NOT_PERFORMED_BY_DESIGN_REGISTERED_PRE_IQR_CONTROL",
        }
        input_binding_paths = {
            "registered_input_digest_ledger": str((output / "registered_input_digest_ledger.json").resolve()),
        }

    control_destination = output / normalized_mode
    control_destination.mkdir(parents=True, exist_ok=True)
    control_manifest = _persist_control_manifests(control_destination, datasets)
    fingerprint = _control_fingerprint(
        mode=normalized_mode,
        comparison=normalized_comparison,
        config=config,
        source_freeze=source_freeze,
        control_manifest=control_manifest,
        cached=cached,
        extra=extra,
    )
    identity = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "control_protocol": STAGE_B_PROTOCOL,
        "control_type": control_type,
        "mode": normalized_mode,
        "comparison": normalized_comparison,
        "cohort": normalized_comparison,
        "model": config.model,
        "modalities": list(JOINT_MODALITIES if normalized_mode == "negative" else requested_modalities),
        "stage": resolved_stage,
        "split_seed": FIXED_SPLIT_SEED,
        "fit_seeds": list(selected_seeds),
        "run_fingerprint": fingerprint,
        "control_manifest": control_manifest,
        "input_binding": input_binding_paths,
        "source_freeze": str(source_freeze_path.resolve()),
        "thread_contract": thread_contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _identity_path_check(output / "run_identity.json", identity)
    _write_immutable_json(output / "control_manifest.json", control_manifest)
    _write_immutable_json(output / "config.json", {"schema_version": 1, "resolved": config.__dict__, "source": payload})

    if dry_run:
        _write_json(
            control_destination / "dry_run.json",
            {
                "status": "PLANNED_NO_TRAINING",
                "training_started": False,
                "protocol_id": PROTOCOL_ID,
                "control_type": control_type,
                "mode": normalized_mode,
                "comparison": normalized_comparison,
                "model": config.model,
                "modalities": identity["modalities"],
                "stage": resolved_stage,
                "fit_seeds": list(selected_seeds),
                "run_fingerprint": fingerprint,
                "counts": {split: len(getattr(datasets[split], "records", [])) for split in SPLITS},
                "thread_contract": thread_contract,
                "input_binding": input_binding_paths,
                "source_freeze": str(source_freeze_path.resolve()),
                "control_manifest": control_manifest,
            },
            overwrite=True,
        )
        return {
            "status": "PLANNED_NO_TRAINING",
            "training_started": False,
            "protocol_id": PROTOCOL_ID,
            "mode": normalized_mode,
            "comparison": normalized_comparison,
            "output_dir": str(output.resolve()),
            "run_fingerprint": fingerprint,
        }

    # The source freeze, input binding, transformed manifest, and identity all
    # exist before this first call to train_one_seed.
    results = _fit_neural_seeds(
        datasets,
        config=config,
        seeds=selected_seeds,
        destination=control_destination,
        fingerprint=fingerprint,
        identity=identity,
        control_manifest=control_manifest,
        input_binding=input_binding_paths,
        source_freeze_path=source_freeze_path,
    )
    if normalized_mode == "tiny":
        gate = evaluate_tiny_gate(results)
    elif normalized_mode == "positive":
        gate = evaluate_positive_gate(results)
    else:
        gate = evaluate_negative_gate(results)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "control_protocol": STAGE_B_PROTOCOL,
        "control_type": control_type,
        "mode": normalized_mode,
        "comparison": normalized_comparison,
        "cohort": normalized_comparison,
        "model": config.model,
        "modalities": identity["modalities"],
        "stage": resolved_stage,
        "split_seed": FIXED_SPLIT_SEED,
        "fit_seeds": list(selected_seeds),
        "run_fingerprint": fingerprint,
        "status": gate.get("status", "INCONCLUSIVE"),
        "training_started": True,
        "thread_contract": thread_contract,
        "config": _json_safe(config.__dict__),
        "control_manifest": control_manifest,
        "input_binding": input_binding_paths,
        "source_freeze": str(source_freeze_path.resolve()),
        "metadata": extra,
        "results": results,
        "gate": gate,
    }
    _write_json(control_destination / "gate.json", summary, overwrite=True, allow_nan=True)
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison", choices=("fomo", "fomo45k", "mpi", "oasis3", "mixed"), required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--ledger-summary-path", type=Path, default=None)
    parser.add_argument("--expected-ledger-sha256", default=DEFAULT_HEALTHY_LEDGER_SHA256)
    parser.add_argument("--mode", choices=("tiny", "positive", "negative"), required=True)
    parser.add_argument("--stage", choices=("final", "registered"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modalities", nargs="+", choices=CANONICAL_MODALITIES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--tiny-seed", type=int, default=TINY_SEED)
    parser.add_argument("--tiny-subjects-per-label", type=int, default=TINY_SUBJECTS_PER_LABEL)
    parser.add_argument("--tiny-max-slices", type=int, default=TINY_MAX_SLICES)
    parser.add_argument("--tiny-min-slices", type=int, default=TINY_MIN_SLICES)
    parser.add_argument("--fit-positive-shared-scalar", action="store_true")
    parser.add_argument("--retrained-permutations", type=int, default=0)
    # Kept solely for the v4 planner's command compatibility.  Mode and
    # contract validation remain authoritative.
    parser.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_control(
            config_path=args.config,
            manifest_root=args.manifest_root,
            output_dir=args.output_dir,
            mode=args.mode,
            comparison=args.comparison,
            build_root=args.build_root,
            ledger_path=args.ledger_path,
            ledger_summary_path=args.ledger_summary_path,
            expected_ledger_sha256=args.expected_ledger_sha256,
            stage=args.stage,
            device=args.device,
            modalities=args.modalities,
            seeds=args.seeds,
            tiny_seed=args.tiny_seed,
            tiny_subjects_per_label=args.tiny_subjects_per_label,
            tiny_max_slices=args.tiny_max_slices,
            tiny_min_slices=args.tiny_min_slices,
            fit_positive_shared_scalar=args.fit_positive_shared_scalar,
            retrained_permutations=args.retrained_permutations,
            dry_run=args.dry_run,
        )
    except (ControlsV3Error, V3InputValidationError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True))
    return 0 if result.get("status") in {"COMPLETE", "REUSED", "PLANNED_NO_TRAINING"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
