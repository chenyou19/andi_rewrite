"""Run one observed FOMO fit and its fixed-data full retrained pair null.

This is a bounded calibration protocol after the negative-control failures.  It
uses the original 70/15/15 FOMO manifests and one three-channel SmallCNN
configuration.  The observed fit uses the ordinary validation-selected
checkpoint (40 epochs maximum, patience 8).  The predeclared 199 null fits
swap labels independently within complete matched pairs in train, validation,
and test, rerun the same fitting and validation selection, and keep the model
initialization seed fixed at 73.  Images are materialized once and shared by
all fits.  Every null fit is written atomically so an interrupted run resumes
without replacing completed artifacts.

The null deliberately omits bootstrap and held-out swap resampling budgets;
each null still stores its fitted predictions, validation selection, test
metrics, label streams, configuration, and run fingerprint.  The observed fit
retains the preregistered 2000/1000 statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    _strip_prediction_arrays,
    atomic_write_json,
    dataset_from_manifest,
    materialize_tensor_datasets,
    permute_pair_labels,
    read_jsonl_manifest,
    run_fingerprint,
    write_prediction_rows,
)


SPLITS = ("train", "val", "test")
DEFAULT_REPLICATES = 199
DEFAULT_INIT_SEED = 73
DEFAULT_LABEL_STREAM_ROOT = 0xC2A57E


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
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
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if not callable(converter):
        raise TypeError(f"Expected mapping or to_dict record, got {type(record).__name__}.")
    return dict(converter())


def _row_key(record: Any) -> tuple[str, str, str, str, str, int]:
    row = _record_mapping(record)
    return (
        str(row.get("source_dataset", "")),
        str(row.get("source_split", "")),
        str(row.get("source_key", "")),
        str(row.get("participant_id", "")),
        str(row.get("case_id", "")),
        int(row.get("z", 0)),
    )


def _copy_rows(dataset: Any) -> list[dict[str, Any]]:
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Calibration requires manifest-backed datasets.")
    return [_record_mapping(record) for record in records]


def _label_stream_seeds(index: int, *, root: int = DEFAULT_LABEL_STREAM_ROOT) -> dict[str, int]:
    """Derive independent deterministic label streams for one replicate."""

    parent = np.random.SeedSequence([int(root), int(index)])
    children = parent.spawn(3)
    values = {
        split: int(child.generate_state(1, dtype=np.uint32)[0])
        for split, child in zip(SPLITS, children)
    }
    if len(set(values.values())) != 3:
        raise AssertionError("Independent calibration label streams collided.")
    return values


def permuted_datasets(
    cached: Mapping[str, Any],
    *,
    index: int,
    init_seed: int = DEFAULT_INIT_SEED,
) -> tuple[dict[str, ManifestDataset], dict[str, int]]:
    """Swap each split's complete matched pairs with independent streams."""

    stream_seeds = _label_stream_seeds(index)
    output: dict[str, ManifestDataset] = {}
    for split in SPLITS:
        rows = permute_pair_labels(_copy_rows(cached[split]), seed=stream_seeds[split])
        source = cached[split]
        output[split] = ManifestDataset(
            rows,
            loader=getattr(source, "loader", None),
            stage=getattr(source, "stage", "final"),
            modalities=getattr(source, "modalities", ("flair", "t1", "t2")),
            shared_train_scalar=getattr(source, "shared_train_scalar", None),
        )
    return output, stream_seeds


def _manifest_identity(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "paths": {split: str(paths[split]) for split in SPLITS},
        "sha256": {
            split: hashlib.sha256(paths[split].read_bytes()).hexdigest()
            for split in SPLITS
        },
    }


def _cache_digest(cached: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for split in SPLITS:
        records = getattr(cached[split], "records")
        if len(records) != len(cached[split]):
            raise ValueError(f"Cached {split} record count does not match dataset length.")
        for index, record in enumerate(records):
            sample = cached[split][index]
            image = sample["image"].detach().cpu().contiguous()
            digest.update(split.encode("utf-8"))
            digest.update(repr(_row_key(record)).encode("utf-8"))
            digest.update(image.numpy().tobytes())
            count += 1
    return {"rows": count, "sha256": digest.hexdigest(), "materialized_once": True, "stage": "final"}


def _copy_code_snapshot(output_root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    snapshot_root = output_root / "code_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for source in paths:
        if not source.is_file():
            continue
        relative = source.relative_to(REPO_ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(source.read_bytes()).hexdigest():
                raise RuntimeError(f"Existing code snapshot differs: {destination}")
        else:
            shutil.copy2(source, destination)
        entries.append({
            "source": str(source),
            "snapshot": str(destination),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    atomic_write_json(output_root / "code_snapshot" / "snapshot_manifest.json", {"files": entries}, overwrite=False)
    return {"root": str(snapshot_root), "files": entries}


def _ensure_identity(output_root: Path, identity: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "run_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != identity.get("run_fingerprint"):
            raise RuntimeError("Existing calibration output has a different run identity; choose a new root.")
        if int(existing.get("requested_replicates", -1)) != int(identity.get("requested_replicates", -2)):
            raise RuntimeError("Existing calibration output has a different requested replicate count.")
    else:
        atomic_write_json(path, identity, overwrite=False)


def _null_status(output_root: Path, *, requested: int, fingerprint: str) -> dict[str, Any]:
    null_root = output_root / "retrained_null"
    completed: list[int] = []
    partial: list[int] = []
    for index in range(int(requested)):
        destination = null_root / f"permutation_{index:04d}.json"
        predictions = null_root / f"permutation_{index:04d}_test_predictions.jsonl"
        if destination.exists() and predictions.exists():
            try:
                value = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot read null artifact {destination}") from exc
            if value.get("run_fingerprint") != fingerprint:
                raise RuntimeError(f"Null artifact {destination} belongs to another run.")
            completed.append(index)
        elif destination.exists() or predictions.exists():
            partial.append(index)
    if partial:
        raise RuntimeError(f"Partial null artifacts require inspection before resume: {partial[:5]!r}")
    return {
        "requested": int(requested),
        "completed": len(completed),
        "completed_indices": completed,
        "remaining": int(requested) - len(completed),
        "status": "complete" if len(completed) == int(requested) else "incomplete",
        "run_fingerprint": fingerprint,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_null_status(output_root: Path, status: Mapping[str, Any]) -> None:
    atomic_write_json(output_root / "retrained_null" / "status.json", _json_safe(dict(status)), overwrite=True)


def _compact_null_summary(output_root: Path, *, requested: int, fingerprint: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(int(requested)):
        path = output_root / "retrained_null" / f"permutation_{index:04d}.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        test = value.get("test", {})
        rows.append({
            "index": int(index),
            "init_seed": value.get("init_seed"),
            "label_stream_seeds": value.get("label_stream_seeds"),
            "epochs_completed": value.get("epochs_completed"),
            "selected_epoch": value.get("best_epoch"),
            "subject_auc": (test.get("subject") or {}).get("roc_auc"),
            "slice_auc": (test.get("slice") or {}).get("roc_auc"),
            "test_loss": test.get("loss"),
            "elapsed_seconds": (value.get("timing") or {}).get("elapsed_seconds"),
        })
    observed_path = output_root / "observed" / "result.json"
    observed_auc = None
    if observed_path.exists():
        observed_value = json.loads(observed_path.read_text(encoding="utf-8"))
        observed_auc = (observed_value.get("test", {}).get("subject") or {}).get("roc_auc")
    null_aucs = [float(row["subject_auc"]) for row in rows if row.get("subject_auc") is not None and np.isfinite(float(row["subject_auc"]))]
    t_observed = abs(float(observed_auc) - 0.5) if observed_auc is not None and np.isfinite(float(observed_auc)) else None
    null_t_values = [abs(value - 0.5) for value in null_aucs]
    null_extreme = int(sum(value >= t_observed - 1.0e-15 for value in null_t_values)) if t_observed is not None else None
    p_plus_one = float((1 + null_extreme) / (1 + len(null_t_values))) if null_extreme is not None else None
    statistic = {
        "name": "full_retrained_pair_swap_test",
        "statistic": "T=abs(subject_mean_score_roc_auc-0.5)",
        "observed_subject_auc": observed_auc,
        "observed_T": t_observed,
        "null_T_values": null_t_values,
        "n_null": len(null_t_values),
        "n_null_at_least_observed_T": null_extreme,
        "p_plus_one": p_plus_one,
        "minimum_attainable_p": float(1.0 / (1 + len(null_t_values))) if null_t_values else None,
        "conditional_on_fixed_matched_pairs": True,
        "label_scope": "train_val_test_complete_pair_swaps",
    }
    summary = {
        "requested": int(requested),
        "completed": len(rows),
        "status": "complete" if len(rows) == int(requested) else "incomplete",
        "run_fingerprint": fingerprint,
        "statistics_budget": {"bootstrap_replicates": 0, "swap_replicates": 0},
        "full_null_statistic": statistic,
        "rows": rows,
    }
    atomic_write_json(output_root / "retrained_null" / "summary.json", _json_safe(summary), overwrite=True)
    atomic_write_json(output_root / "retrained_null" / "statistic.json", _json_safe(statistic), overwrite=True)
    return summary


def run_calibration(
    *,
    manifest_root: Path,
    output_root: Path,
    replicates: int = DEFAULT_REPLICATES,
    device: str = "cuda",
    init_seed: int = DEFAULT_INIT_SEED,
    observed_only: bool = False,
) -> dict[str, Any]:
    if int(replicates) < 0:
        raise ValueError("replicates must be non-negative.")
    paths = {split: manifest_root / f"{split}.jsonl" for split in SPLITS}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Missing calibration manifest under {manifest_root}.")
    observed_cfg = TrainConfig(
        model="small_cnn",
        in_channels=3,
        modalities=("flair", "t1", "t2"),
        max_epochs=40,
        patience=8,
        early_stopping=True,
        dropout=0.0,
        weight_decay=1.0e-4,
        batch_size=32,
        split_seed=73,
        seed=int(init_seed),
        stage="final",
        device=str(device),
        bootstrap_replicates=2000,
        swap_replicates=1000,
        permutation_replicates=int(replicates),
        permutation_mode="full_retrained_pair_swap_all_splits",
    )
    code_paths = [
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "metrics.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
        Path(__file__).resolve(),
    ]
    fingerprint = run_fingerprint(observed_cfg, paths, code_paths=code_paths)
    identity = {
        "protocol": "fomo45k_calibration_smallcnn_full_retrained_pair_null",
        "run_fingerprint": fingerprint,
        "requested_replicates": int(replicates),
        "init_seed": int(init_seed),
        "manifest_identity": _manifest_identity(paths),
        "config": asdict(observed_cfg),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _ensure_identity(output_root, identity)
    source_datasets = {
        split: dataset_from_manifest(
            paths[split],
            stage="final",
            modalities=observed_cfg.modalities,
            shared_train_scalar=observed_cfg.shared_train_scalar,
        )
        for split in SPLITS
    }
    cached = materialize_tensor_datasets(source_datasets)
    cache_info = _cache_digest(cached)
    protocol_path = output_root / "protocol.json"
    protocol = {
        **identity,
        "status": "PRE_REGISTERED",
        "fixed_split_seed": 73,
        "model": "small_cnn",
        "modalities": list(observed_cfg.modalities),
        "shape": [3, 128, 128],
        "selection": "subject AUC then BCE on true validation for observed; same validation procedure on each permuted null",
        "observed_statistics": {"bootstrap_replicates": 2000, "swap_replicates": 1000},
        "null_statistics": {"bootstrap_replicates": 0, "swap_replicates": 0, "omitted_by_design": True},
        "null_label_scope": "complete matched-pair swaps in train, val, and test",
        "null_initialization": "fixed seed 73 for every replicate; only label-stream seeds vary",
        "null_label_stream_root": int(DEFAULT_LABEL_STREAM_ROOT),
        "requested_replicates": int(replicates),
        "tensor_cache": cache_info,
        "manifest_identity": _manifest_identity(paths),
        "historical_failed_controls": [
            str(REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "fomo45k_cached2" / "shuffle" / "gate.json"),
            str(REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "fomo45k_supplementary_balanced_v2" / "gate_true_test.json"),
        ],
    }
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol.get("run_fingerprint") != fingerprint or int(existing_protocol.get("requested_replicates", -1)) != int(replicates):
            raise RuntimeError("Existing protocol differs from this calibration request.")
    else:
        atomic_write_json(protocol_path, _json_safe(protocol), overwrite=False)
        atomic_write_json(output_root / "source_cache_digest.json", _json_safe(cache_info), overwrite=False)
        _copy_code_snapshot(output_root, code_paths)

    # The ordinary observed fit is written once and retained on resume.
    observed_root = output_root / "observed"
    observed_root.mkdir(parents=True, exist_ok=True)
    observed_result_path = observed_root / "result.json"
    if not observed_result_path.exists():
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        result = DomainClassifierRunner(observed_cfg).train_one_seed(
            cached["train"], cached["val"], cached["test"], seed=int(init_seed)
        )
        elapsed = float(time.perf_counter() - started)
        state_dict = result.get("state_dict")
        if state_dict is None:
            raise RuntimeError("Observed calibration returned no selected state dict.")
        _atomic_torch_save(state_dict, observed_root / "model_best.pt")
        write_prediction_rows(observed_root / "test_predictions.jsonl", result.get("test_predictions", []))
        serializable = _json_safe({
            "run_fingerprint": fingerprint,
            "protocol": protocol["protocol"],
            "init_seed": int(init_seed),
            "config": asdict(observed_cfg),
            "timing": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
            },
            **{key: value for key, value in result.items() if key != "state_dict"},
        })
        atomic_write_json(observed_result_path, serializable, overwrite=False)
    elif not (observed_root / "model_best.pt").exists() or not (observed_root / "test_predictions.jsonl").exists():
        raise RuntimeError("Observed calibration result exists but checkpoint/predictions are missing.")

    if observed_only:
        return {"status": "observed_complete", "output_root": str(output_root), "run_fingerprint": fingerprint}

    null_root = output_root / "retrained_null"
    null_root.mkdir(parents=True, exist_ok=True)
    null_cfg = replace(observed_cfg, bootstrap_replicates=0, swap_replicates=0)
    status = _null_status(output_root, requested=int(replicates), fingerprint=fingerprint)
    _write_null_status(output_root, status)
    for index in range(int(replicates)):
        destination = null_root / f"permutation_{index:04d}.json"
        predictions_destination = null_root / f"permutation_{index:04d}_test_predictions.jsonl"
        if index in set(status["completed_indices"]):
            continue
        if destination.exists() or predictions_destination.exists():
            raise RuntimeError(f"Partial null artifact exists for replicate {index}; refusing overwrite.")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        permuted, stream_seeds = permuted_datasets(cached, index=int(index), init_seed=int(init_seed))
        result = DomainClassifierRunner(null_cfg).train_one_seed(
            permuted["train"], permuted["val"], permuted["test"], seed=int(init_seed)
        )
        elapsed = float(time.perf_counter() - started)
        predictions = result.get("test_predictions", [])
        write_prediction_rows(predictions_destination, predictions, overwrite=False)
        serializable = _json_safe({
            "run_fingerprint": fingerprint,
            "protocol": protocol["protocol"],
            "permutation_index": int(index),
            "init_seed": int(init_seed),
            "label_stream_seeds": stream_seeds,
            "config": asdict(null_cfg),
            "label_scope": "train_val_test_complete_pair_swaps",
            "statistics_omitted": {"bootstrap_replicates": 0, "swap_replicates": 0},
            "timing": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
            },
            **{key: value for key, value in result.items() if key != "state_dict"},
        })
        atomic_write_json(destination, serializable, overwrite=False)
        status = _null_status(output_root, requested=int(replicates), fingerprint=fingerprint)
        _write_null_status(output_root, status)
        _compact_null_summary(output_root, requested=int(replicates), fingerprint=fingerprint)

    status = _null_status(output_root, requested=int(replicates), fingerprint=fingerprint)
    _write_null_status(output_root, status)
    null_summary = _compact_null_summary(output_root, requested=int(replicates), fingerprint=fingerprint)
    final = {
        "status": "complete" if status["status"] == "complete" else "incomplete",
        "protocol": protocol["protocol"],
        "output_root": str(output_root),
        "run_fingerprint": fingerprint,
        "observed_result": str(observed_result_path),
        "null_status": status,
        "null_summary": null_summary,
        "historical_controls_preserved": protocol["historical_failed_controls"],
    }
    atomic_write_json(output_root / "summary.json", _json_safe(final), overwrite=True)
    return _json_safe(final)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-seed", type=int, default=DEFAULT_INIT_SEED)
    parser.add_argument("--observed-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    summary = run_calibration(
        manifest_root=manifest_root.resolve(),
        output_root=output_root.resolve(),
        replicates=int(args.replicates),
        device=str(args.device),
        init_seed=int(args.init_seed),
        observed_only=bool(args.observed_only),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"complete", "observed_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
