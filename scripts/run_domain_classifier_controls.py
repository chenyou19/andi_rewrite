"""Run the staged controls for the ANDi domain-classifier audit.

The controls are deliberately separate from report rendering.  Every mode
uses fixed participant manifests, writes one seed artifact at a time, and
records an explicit gate status.  The command is intended to be run with the
ANDi interpreter, for example::

    C:\\Users\\E-118-3\\miniconda3\\envs\\ANDi\\python.exe \\
      scripts\\run_domain_classifier_controls.py \\
      --config configs\\domain_classifier.yaml --comparison fomo --mode tiny

Use ``--dry-run`` to inspect paths and the planned matrix without training.
The ``all`` mode runs tiny, registered positive, same-cohort negative, and
training-label shuffle controls sequentially.  ``expansion`` is intentionally
opt-in because it executes the full 112-cell cohort/modality/classifier/seed
matrix after the Stage1 gates have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    DomainClassifierRunner,
    ManifestDataset,
    TrainConfig,
    atomic_write_json,
    dataset_from_manifest,
    dataset_with_modalities,
    dataset_with_shared_train_scalar,
    dataset_with_stage,
    evaluate_negative_gate,
    evaluate_positive_gate,
    evaluate_tiny_gate,
    fit_shared_train_scalar,
    formal_expansion_plan,
    iter_retrained_pair_permutation_control,
    make_same_cohort_negative_records,
    materialize_dataset,
    materialize_registered_dataset,
    materialize_tensor_dataset,
    materialize_tensor_datasets,
    permute_pair_labels,
    read_jsonl_manifest,
    run_fingerprint,
    run_statistical_control,
    seed_matrix_gate,
    select_tiny_records,
    shuffle_participant_labels,
    subset_dataset,
    write_prediction_rows,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("The controls CLI requires PyYAML.") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config {path} must contain a mapping at its root.")
    return value


def _config_from_yaml(value: Mapping[str, Any], *, stage: str | None, device: str | None) -> TrainConfig:
    training = value.get("training", value)
    if not isinstance(training, Mapping):
        training = {}
    allowed = set(TrainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: item for key, item in training.items() if key in allowed}
    if "widths" in kwargs:
        kwargs["widths"] = tuple(int(item) for item in kwargs["widths"])
    if "modalities" in kwargs:
        kwargs["modalities"] = tuple(str(item).strip().lower() for item in kwargs["modalities"])
    if stage is not None:
        kwargs["stage"] = stage
    if device is not None:
        kwargs["device"] = device
    return TrainConfig(**kwargs)


def _comparison_name(value: str) -> str:
    return {"fomo": "fomo45k", "fomo45k": "fomo45k", "mpi": "mpi", "oasis3": "oasis3", "mixed": "mixed"}[value]


def _manifest_paths(config_value: Mapping[str, Any], *, comparison: str | None, manifest_root: Path | None) -> dict[str, Path]:
    if comparison is not None:
        name = _comparison_name(comparison)
        root = manifest_root or (REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "manifests" / name)
        return {split: (root / f"{split}.jsonl").resolve() for split in ("train", "val", "test")}
    manifests = config_value.get("manifests", {})
    if not isinstance(manifests, Mapping):
        manifests = {}
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidate = manifests.get(split)
        if candidate is None and manifest_root is not None:
            candidate = manifest_root / f"{split}.jsonl"
        if candidate is None:
            raise ValueError(f"Missing {split} manifest in config or --manifest-root.")
        path = Path(candidate)
        paths[split] = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "state_dict"}
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


def _ensure_identity(destination: Path, fingerprint: str, details: Mapping[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "run_identity.json"
    if identity_path.exists():
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read existing control identity: {identity_path}") from exc
        if existing.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing control output {destination} belongs to another run; choose a new output directory."
            )
        return
    atomic_write_json(destination / "run_identity.json", {"run_fingerprint": fingerprint, **dict(details)})


def _read_existing_result(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read existing control artifact: {path}") from exc
    if result.get("run_fingerprint") != fingerprint:
        raise RuntimeError(f"Existing artifact {path} belongs to another run.")
    return result


def _clone_with_records(dataset: Any, records: Sequence[Any]) -> ManifestDataset:
    return subset_dataset(dataset, records)


def _tiny_datasets(
    datasets: Mapping[str, Any],
    *,
    subjects_per_label: int,
    max_slices: int,
    min_slices: int,
    seed: int,
) -> dict[str, ManifestDataset]:
    def paired_selection(records: Sequence[Any], *, seed_value: int) -> list[Any] | None:
        by_pair: dict[str, list[Any]] = {}
        for row in records:
            pair = row.get("pair_id") if isinstance(row, Mapping) else getattr(row, "pair_id", None)
            if pair is None or str(pair).strip() == "":
                return None
            by_pair.setdefault(str(pair), []).append(row)
        valid: list[tuple[str, list[Any]]] = []
        for pair, pair_rows in by_pair.items():
            labels = {
                int(row.get("label")) if isinstance(row, Mapping) else int(getattr(row, "label"))
                for row in pair_rows
            }
            participants = {
                str(row.get("participant_id")) if isinstance(row, Mapping) else str(getattr(row, "participant_id"))
                for row in pair_rows
            }
            if labels == {0, 1} and len(participants) >= 2:
                valid.append((pair, pair_rows))
        if len(valid) < int(subjects_per_label):
            return None
        rng = np.random.default_rng(int(seed_value))
        chosen_indices = rng.permutation(len(valid))[: int(subjects_per_label)].tolist()
        selected: list[Any] = []
        for index in chosen_indices:
            pair, pair_rows = valid[int(index)]
            selected.extend(
                sorted(
                    pair_rows,
                    key=lambda row: (
                        int(row.get("z", 0)) if isinstance(row, Mapping) else int(getattr(row, "z", 0)),
                        str(row.get("case_id", "")) if isinstance(row, Mapping) else str(getattr(row, "case_id", "")),
                    ),
                )
            )
        if len(selected) <= int(max_slices):
            return selected
        # Retain at least one slice for both participants of every selected
        # pair, then consume the deterministic z/case ordering up to the cap.
        kept: list[Any] = []
        remaining: list[Any] = []
        for index in chosen_indices:
            _pair, pair_rows = valid[int(index)]
            by_label: dict[int, list[Any]] = {0: [], 1: []}
            for row in pair_rows:
                label = int(row.get("label")) if isinstance(row, Mapping) else int(getattr(row, "label"))
                by_label[label].append(row)
            kept.extend([by_label[0][0], by_label[1][0]])
            remaining.extend(by_label[0][1:] + by_label[1][1:])
        return kept + remaining[: max(0, int(max_slices) - len(kept))]

    def complete_pairs(rows: Sequence[Any], *, max_count: int) -> list[Any]:
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            if isinstance(row, Mapping):
                pair = row.get("pair_id")
                label = row.get("label")
                participant = row.get("participant_id")
            else:
                pair = getattr(row, "pair_id", None)
                label = getattr(row, "label", None)
                participant = getattr(row, "participant_id", None)
            if pair is None or str(pair).strip() == "":
                continue
            grouped.setdefault(str(pair), []).append(row)
        output: list[Any] = []
        for pair in sorted(grouped):
            pair_rows = grouped[pair]
            by_label: dict[int, list[Any]] = {0: [], 1: []}
            for row in pair_rows:
                label = int(getattr(row, "label", row.get("label"))) if isinstance(row, Mapping) else int(getattr(row, "label"))
                by_label.setdefault(label, []).append(row)
            if not by_label.get(0) or not by_label.get(1):
                continue
            candidate = by_label[0] + by_label[1]
            if len(output) + len(candidate) <= int(max_count):
                output.extend(candidate)
            else:
                # Preserve both participants when the final pair meets the
                # cap; a single slice per participant is sufficient for the
                # held-out subject bootstrap/swap statistics.
                if len(output) + 2 <= int(max_count):
                    output.extend([by_label[0][0], by_label[1][0]])
                break
        return output

    output: dict[str, ManifestDataset] = {}
    for offset, split in enumerate(("train", "val", "test")):
        source = datasets[split]
        records = getattr(source, "records", None)
        if records is None:
            raise TypeError("Tiny controls require manifest-backed datasets.")
        selected = paired_selection(records, seed_value=int(seed) + offset)
        if selected is None:
            selected = select_tiny_records(
                records,
                subjects_per_label=int(subjects_per_label),
                max_slices=int(max_slices),
                seed=int(seed) + offset,
            )
            if split in {"val", "test"}:
                selected = complete_pairs(selected, max_count=int(max_slices))
        if len(selected) < int(min_slices):
            raise ValueError(
                f"Tiny {split} subset contains {len(selected)} slices; expected at least {min_slices}."
            )
        output[split] = _clone_with_records(source, selected)
    return output


def _same_cohort_negative_datasets(
    datasets: Mapping[str, Any],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.40, 0.10, 0.50),
) -> dict[str, ManifestDataset]:
    """Create a separately split 40/10/50 same-cohort negative control.

    The ordinary positive/domain audit keeps its preregistered 70/15/15
    participant split.  This negative control starts from all label-0 cohort
    participants, randomly assigns whole participants to 40/10/50 train/val/
    test groups, then creates balanced pseudo-label pairs inside each group.
    With 240 healthy FOMO participants this yields 60 participants per test
    pseudo-label, as specified by the control plan.
    """

    if len(fractions) != 3 or any(float(value) <= 0.0 for value in fractions):
        raise ValueError("Negative split fractions must contain three positive values.")
    total_fraction = float(sum(fractions))
    if not np.isclose(total_fraction, 1.0):
        raise ValueError("Negative split fractions must sum to one.")
    participant_rows: dict[object, list[dict[str, Any]]] = {}
    templates = dict(datasets)
    for split in ("train", "val", "test"):
        records = getattr(datasets[split], "records", None)
        if records is None:
            raise TypeError("Negative controls require manifest-backed datasets.")
        for raw in records:
            if isinstance(raw, Mapping):
                record = dict(raw)
            else:
                converter = getattr(raw, "to_dict", None)
                if not callable(converter):
                    raise TypeError("Negative-control records must be mappings or expose to_dict().")
                record = dict(converter())
            if int(float(record.get("label", -1))) != 0:
                continue
            participant = record.get("participant_id")
            if participant is None or str(participant).strip() == "":
                raise ValueError("Negative-control records require participant_id.")
            participant_rows.setdefault(participant, []).append(record)
    participants = sorted(participant_rows, key=lambda value: str(value))
    if len(participants) < 6:
        raise ValueError("Negative control needs at least six same-cohort participants.")
    order = np.random.default_rng(int(seed)).permutation(len(participants)).tolist()
    shuffled = [participants[int(index)] for index in order]
    counts = [int(round(len(shuffled) * float(value))) for value in fractions[:2]]
    counts.append(len(shuffled) - sum(counts))
    if min(counts) < 2:
        raise ValueError(f"Negative split participant counts are too small: {counts!r}")
    split_names = ("train", "val", "test")
    target_records: list[dict[str, Any]] = []
    start = 0
    for split, count in zip(split_names, counts):
        for participant in shuffled[start : start + count]:
            for record in participant_rows[participant]:
                updated = dict(record)
                updated["split"] = split
                # Keep source_split tied to the original source LMDB/cache;
                # only the control split assignment changes.
                target_records.append(updated)
        start += count
    negative = make_same_cohort_negative_records(target_records, seed=int(seed), source_label=0)
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    for record in negative:
        by_split[str(record.get("split", ""))].append(record)
    return {
        split: _clone_with_records(templates[split], by_split[split])
        for split in split_names
    }


def _shuffle_training_dataset(dataset: Any, *, seed: int) -> ManifestDataset:
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Training-label shuffle requires a manifest-backed dataset.")
    return _clone_with_records(dataset, shuffle_participant_labels(records, seed=int(seed)))


def _powered_shuffle_datasets(
    datasets: Mapping[str, Any],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.40, 0.10, 0.50),
) -> dict[str, ManifestDataset]:
    """Create the powered 40/10/50 pair-label-swap shuffle control.

    Original matched pairs are assigned as whole units to a new participant
    split.  For each seed, labels are swapped independently once per pair in
    the new train and validation groups; the held-out test group retains its
    original labels.  This avoids using true validation labels to choose a
    checkpoint and keeps every test pair in the fixed estimand.
    """

    if len(fractions) != 3 or any(float(value) <= 0.0 for value in fractions):
        raise ValueError("Powered shuffle fractions must contain three positive values.")
    if not np.isclose(float(sum(fractions)), 1.0):
        raise ValueError("Powered shuffle fractions must sum to one.")
    all_rows: list[dict[str, Any]] = []
    templates = dict(datasets)
    for split in ("train", "val", "test"):
        records = getattr(datasets[split], "records", None)
        if records is None:
            raise TypeError("Powered shuffle requires manifest-backed datasets.")
        for record in records:
            row = _record_mapping_for_controls(record)
            row["_original_split"] = split
            all_rows.append(row)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        pair = str(row.get("pair_id", "")).strip()
        if not pair:
            raise ValueError("Powered shuffle requires a non-empty pair_id on every row.")
        by_pair.setdefault(pair, []).append(row)
    pair_ids = sorted(by_pair)
    if len(pair_ids) < 6:
        raise ValueError("Powered shuffle requires at least six matched pairs.")
    order = np.random.default_rng(int(seed)).permutation(len(pair_ids)).tolist()
    shuffled_pairs = [pair_ids[int(index)] for index in order]
    counts = [int(round(len(shuffled_pairs) * float(value))) for value in fractions[:2]]
    counts.append(len(shuffled_pairs) - sum(counts))
    split_names = ("train", "val", "test")
    pair_split: dict[str, str] = {}
    cursor = 0
    for split, count in zip(split_names, counts):
        for pair in shuffled_pairs[cursor : cursor + count]:
            pair_split[pair] = split
        cursor += count
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    for pair in pair_ids:
        split = pair_split[pair]
        for row in by_pair[pair]:
            updated = dict(row)
            updated.pop("_original_split", None)
            updated["split"] = split
            rows_by_split[split].append(updated)

    # Swap labels independently for all train/validation pairs using the
    # per-seed RNG in permute_pair_labels.  The test records are copied after
    # this operation, so their labels remain the true held-out labels.
    train_val_rows = permute_pair_labels(
        rows_by_split["train"] + rows_by_split["val"],
        seed=int(seed),
    )
    rows_by_split["train"] = [row for row in train_val_rows if str(row.get("split")) == "train"]
    rows_by_split["val"] = [row for row in train_val_rows if str(row.get("split")) == "val"]
    for split in split_names:
        for row in rows_by_split[split]:
            metadata = dict(row.get("metadata", {}) or {})
            metadata.update(
                {
                    "shuffle_control": "powered_pair_split_40_10_50",
                    "shuffle_seed": int(seed),
                    "pair_labels_swapped": split in {"train", "val"},
                    "test_labels_true": split == "test",
                }
            )
            row["metadata"] = metadata
    return {
        split: _clone_with_records(templates[split], rows_by_split[split])
        for split in split_names
    }


def _record_mapping_for_controls(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if not callable(converter):
        raise TypeError("Control records must be mappings or expose to_dict().")
    return dict(converter())


def _fit_optional_shared_scalar(
    datasets: Mapping[str, Any],
    *,
    requested: float | None,
    fit_from_train: bool,
) -> tuple[dict[str, Any], float | None]:
    scalar = None if requested is None else float(requested)
    if fit_from_train:
        train_values = materialize_dataset(datasets["train"])
        scalar = fit_shared_train_scalar(train_values["images"])
    if scalar is None:
        return dict(datasets), None
    return {
        split: dataset_with_shared_train_scalar(dataset, scalar)
        for split, dataset in datasets.items()
    }, scalar


def _persist_control_manifests(destination: Path, datasets: Mapping[str, Any]) -> dict[str, Any]:
    """Persist transformed control rows and a compact participant audit."""

    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_audit: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        dataset = datasets[split]
        records = getattr(dataset, "records", None)
        if records is None:
            continue
        rows: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, Mapping):
                row = dict(record)
            else:
                converter = getattr(record, "to_dict", None)
                if not callable(converter):
                    raise TypeError("Control records must be mappings or expose to_dict().")
                row = dict(converter())
            rows.append(_json_safe(row))
        path = manifest_dir / f"{split}.jsonl"
        if not path.exists():
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            temporary.replace(path)
        participant_labels: dict[str, set[int]] = {}
        pair_bins: dict[str, dict[int, int]] = {}
        for row in rows:
            participant = str(row.get("participant_id", ""))
            participant_labels.setdefault(participant, set()).add(int(row.get("label", -1)))
            pair = str(row.get("pair_id", ""))
            if pair:
                pair_bins.setdefault(pair, {})[int(row.get("z_bin", row.get("z", 0)))] = pair_bins.setdefault(pair, {}).get(int(row.get("z_bin", row.get("z", 0))), 0) + 1
        split_audit[split] = {
            "records": len(rows),
            "participants": len(participant_labels),
            "label_counts": {
                str(label): sum(1 for labels in participant_labels.values() if labels == {label})
                for label in (0, 1)
            },
            "conflicting_participants": sorted(participant for participant, labels in participant_labels.items() if len(labels) > 1),
            "pair_count": len(pair_bins),
            "pair_z_bin_histograms": pair_bins,
        }
    atomic_write_json(destination / "split_audit.json", split_audit, overwrite=True)
    return {"manifest_dir": str(manifest_dir), "split_audit": split_audit}


def _normal_neural_config(config: TrainConfig, *, mode: str, stage: str | None = None) -> TrainConfig:
    updates: dict[str, Any] = {"tiny": False, "stage": stage or config.stage}
    if mode == "tiny":
        updates.update(
            {
                "tiny": True,
                "early_stopping": False,
                "max_epochs": 300,
                "patience": 301,
                "weight_decay": 0.0,
                "dropout": 0.0,
            }
        )
    elif mode == "positive":
        updates.update({"max_epochs": 40, "early_stopping": True})
    return replace(config, **updates)


def _run_neural_seeds(
    datasets: Mapping[str, Any],
    *,
    config: TrainConfig,
    seeds: Sequence[int],
    destination: Path,
    fingerprint: str,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _ensure_identity(destination, fingerprint, identity)
    runner = DomainClassifierRunner(config)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_value = int(seed)
        artifact = destination / f"seed_{seed_value}.json"
        checkpoint = destination / f"model_seed_{seed_value}.pt"
        predictions = destination / f"seed_{seed_value}_test_predictions.jsonl"
        existing = _read_existing_result(artifact, fingerprint)
        if existing is not None:
            if not checkpoint.exists() or not predictions.exists():
                raise RuntimeError(f"Control seed {seed_value} has incomplete artifacts: {artifact}")
            results.append(existing)
            continue
        if checkpoint.exists() or predictions.exists():
            raise RuntimeError(f"Partial control artifacts exist for seed {seed_value}: {destination}")
        result = runner.train_one_seed(
            datasets["train"],
            datasets["val"],
            datasets["test"],
            seed=seed_value,
        )
        _atomic_torch_save(result.get("state_dict"), checkpoint)
        write_prediction_rows(predictions, result.get("test_predictions", []))
        serializable = _json_safe({"run_fingerprint": fingerprint, **result})
        atomic_write_json(artifact, serializable)
        results.append(serializable)
    return results


def _run_shuffle_seeds(
    datasets: Mapping[str, Any],
    *,
    config: TrainConfig,
    seeds: Sequence[int],
    destination: Path,
    fingerprint: str,
    identity: Mapping[str, Any],
    design: str,
) -> list[dict[str, Any]]:
    """Fit per-seed shuffle controls with seed-specific label manifests."""

    _ensure_identity(destination, fingerprint, identity)
    runner = DomainClassifierRunner(config)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_value = int(seed)
        if design == "powered_pair_split_40_10_50":
            seed_datasets = _powered_shuffle_datasets(datasets, seed=seed_value)
        elif design == "supplemental_train_only":
            seed_datasets = dict(datasets)
            seed_datasets["train"] = _shuffle_training_dataset(seed_datasets["train"], seed=seed_value)
        else:
            raise ValueError(f"Unknown shuffle design {design!r}.")
        seed_destination = destination / f"seed_{seed_value}"
        control_manifest = _persist_control_manifests(seed_destination, seed_datasets)
        artifact = destination / f"seed_{seed_value}.json"
        checkpoint = destination / f"model_seed_{seed_value}.pt"
        predictions = destination / f"seed_{seed_value}_test_predictions.jsonl"
        existing = _read_existing_result(artifact, fingerprint)
        if existing is not None:
            if not checkpoint.exists() or not predictions.exists():
                raise RuntimeError(f"Shuffle seed {seed_value} has incomplete artifacts: {artifact}")
            results.append(existing)
            continue
        if checkpoint.exists() or predictions.exists():
            raise RuntimeError(f"Partial shuffle artifacts exist for seed {seed_value}: {destination}")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        result = runner.train_one_seed(
            seed_datasets["train"],
            seed_datasets["val"],
            seed_datasets["test"],
            seed=seed_value,
        )
        elapsed = float(time.perf_counter() - started)
        finished_at = datetime.now(timezone.utc).isoformat()
        _atomic_torch_save(result.get("state_dict"), checkpoint)
        write_prediction_rows(predictions, result.get("test_predictions", []))
        serializable = _json_safe(
            {
                "run_fingerprint": fingerprint,
                "shuffle_design": design,
                "shuffle_seed": seed_value,
                "control_manifest": control_manifest,
                "timing": {
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "elapsed_seconds": elapsed,
                },
                **result,
            }
        )
        atomic_write_json(artifact, serializable)
        results.append(serializable)
    return results


def _run_logistic_seeds(
    datasets: Mapping[str, Any],
    *,
    config: TrainConfig,
    seeds: Sequence[int],
    destination: Path,
    fingerprint: str,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _ensure_identity(destination, fingerprint, identity)
    values = {split: materialize_dataset(datasets[split]) for split in ("train", "val", "test")}
    results: list[dict[str, Any]] = []
    background = 0.0 if config.stage == "registered" else -1.0
    for seed in seeds:
        seed_value = int(seed)
        artifact = destination / f"seed_{seed_value}.json"
        checkpoint = destination / f"model_seed_{seed_value}.pt"
        predictions = destination / f"seed_{seed_value}_test_predictions.jsonl"
        existing = _read_existing_result(artifact, fingerprint)
        if existing is not None:
            if not checkpoint.exists() or not predictions.exists():
                raise RuntimeError(f"Control logistic seed {seed_value} has incomplete artifacts: {artifact}")
            results.append(existing)
            continue
        train = values["train"]
        test = values["test"]
        control = run_statistical_control(
            train["images"],
            train["labels"],
            test["images"],
            test["labels"],
            train_participant_ids=train["participant_ids"],
            eval_participant_ids=test["participant_ids"],
            eval_pair_ids=test["pair_ids"],
            eval_case_ids=test["case_ids"],
            background_value=background,
            threshold=config.threshold,
            seed=seed_value,
            bootstrap_replicates=config.bootstrap_replicates,
            swap_replicates=config.swap_replicates,
        )
        result = {
            "run_fingerprint": fingerprint,
            "seed": seed_value,
            "split_seed": int(config.split_seed),
            "config": asdict(config),
            "model": "statistical_logistic",
            "device": "cpu",
            "test": control["metrics"],
            "test_statistics": {
                key: control[key]
                for key in ("subject_bootstrap", "heldout_pair_swap")
                if key in control
            },
            "test_predictions": control.get("prediction_rows", []),
            "test_subject_predictions": control.get("subject_prediction_rows", []),
            "statistical_control": control,
        }
        _atomic_torch_save(
            {"coef": control.get("model_coef"), "intercept": control.get("model_intercept")},
            checkpoint,
        )
        write_prediction_rows(predictions, result["test_predictions"])
        atomic_write_json(artifact, _json_safe(result))
        results.append(_json_safe(result))
    return results


def _run_retrained_permutations(
    datasets: Mapping[str, Any],
    *,
    config: TrainConfig,
    destination: Path,
    fingerprint: str,
    requested: int,
    permute_test_labels: bool = True,
    observed_auc: float | None = None,
) -> dict[str, Any]:
    """Run resumable whole-pair retraining controls when explicitly requested."""

    permutation_dir = destination / "retrained_permutations"
    permutation_dir.mkdir(parents=True, exist_ok=True)
    requested_value = int(max(0, requested))
    pending_indices: list[int] = []
    completed = 0
    for index in range(requested_value):
        artifact = permutation_dir / f"permutation_{index:04d}.json"
        prediction = permutation_dir / f"permutation_{index:04d}_test_predictions.jsonl"
        if artifact.exists() and prediction.exists():
            existing = _read_existing_result(artifact, fingerprint)
            if existing is None:
                raise RuntimeError(f"Existing permutation artifact lacks matching fingerprint: {artifact}")
            completed += 1
        elif artifact.exists() or prediction.exists():
            raise RuntimeError(f"Partial retrained-permutation artifacts exist: {artifact}")
        else:
            pending_indices.append(index)
    for index, result in iter_retrained_pair_permutation_control(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        config=config,
        replicates=int(requested),
        seed=int(config.seed),
        permute_test_labels=bool(permute_test_labels),
        indices=pending_indices,
    ):
        artifact = permutation_dir / f"permutation_{index:04d}.json"
        prediction = permutation_dir / f"permutation_{index:04d}_test_predictions.jsonl"
        atomic_write_json(
            artifact,
            _json_safe({"run_fingerprint": fingerprint, "permutation_index": index, **result}),
        )
        write_prediction_rows(prediction, result.get("test_predictions", []))
        completed += 1
    status = {
        "requested": requested_value,
        "completed": int(completed),
        "remaining": int(requested_value - completed),
        # Zero means the expensive null was omitted, so preserve an explicit
        # incomplete status instead of implying that a null with zero draws
        # was completed.
        "status": "complete" if requested_value > 0 and completed >= requested_value else "incomplete",
        "unit": "whole_pair_label_swap",
        "selection_and_scaler_refit": True,
        "test_labels_true": not bool(permute_test_labels),
        "test_labels_permuted": bool(permute_test_labels),
    }
    null_auc_values: list[float] = []
    for artifact in sorted(permutation_dir.glob("permutation_*.json")):
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            subject = payload.get("test", {}).get("subject", {})
            value = float(subject.get("roc_auc", float("nan")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            null_auc_values.append(value)
    status["null_auc_values"] = null_auc_values
    status["null_auc_mean"] = float(np.mean(null_auc_values)) if null_auc_values else float("nan")
    status["null_ci_low"] = float(np.quantile(null_auc_values, 0.025)) if null_auc_values else float("nan")
    status["null_ci_high"] = float(np.quantile(null_auc_values, 0.975)) if null_auc_values else float("nan")
    observed = float(observed_auc) if observed_auc is not None else float("nan")
    status["observed_auc"] = observed
    if np.isfinite(observed) and null_auc_values:
        deviation = abs(observed - 0.5)
        extreme = sum(abs(value - 0.5) >= deviation - 1.0e-15 for value in null_auc_values)
        status["null_two_sided_p_value"] = float((1 + extreme) / (1 + len(null_auc_values)))
    else:
        status["null_two_sided_p_value"] = float("nan")
    atomic_write_json(permutation_dir / "status.json", status)
    return status


def _mode_fingerprint(base: str, mode: str, config: TrainConfig, extra: Mapping[str, Any] | None = None) -> str:
    payload = {"base": base, "mode": mode, "config": asdict(config), "extra": dict(extra or {})}
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")).hexdigest()


def _assert_expansion_prerequisites(output_root: Path, comparisons: Sequence[str]) -> None:
    """Require Stage1 audit and all four reviewed controls before expansion."""

    required_modes = ("tiny", "positive", "negative", "shuffle")
    missing: list[str] = []
    for mode in required_modes:
        gate_path = output_root / mode / "gate.json"
        if not gate_path.is_file():
            missing.append(str(gate_path))
            continue
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read prerequisite gate {gate_path}") from exc
        gate = payload.get("gate", payload)
        if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
            raise RuntimeError(
                f"Expansion blocked: prerequisite {mode} gate is {gate.get('status') if isinstance(gate, Mapping) else 'MISSING'}, not PASS."
            )
    for comparison in comparisons:
        audit_candidates = [
            output_root / "comparisons" / f"{comparison}.json",
            output_root / "audit.json",
        ]
        audit_path = next((path for path in audit_candidates if path.is_file()), None)
        if audit_path is None:
            missing.append(str(audit_candidates[0]))
            continue
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read Stage1 audit {audit_path}") from exc
        status = audit.get("status")
        if status != "PASS":
            raise RuntimeError(f"Expansion blocked: Stage1 audit {audit_path} has status {status!r}.")
    if missing:
        raise RuntimeError("Expansion blocked: required Stage1/control artifacts are missing: " + "; ".join(missing))


def run_mode(
    mode: str,
    *,
    config: TrainConfig,
    datasets: Mapping[str, Any],
    paths: Mapping[str, Path],
    seeds: Sequence[int],
    output_root: Path,
    base_fingerprint: str,
    subjects_per_label: int,
    tiny_max_slices: int,
    tiny_min_slices: int,
    tiny_seed: int,
    fit_positive_scalar: bool,
    requested_retrained_permutations: int,
    permute_test_labels: bool = True,
    shuffle_design: str = "powered_pair_split_40_10_50",
    comparison_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode == "expansion":
        if comparison_name is None:
            raise ValueError("Expansion execution requires an explicit comparison_name.")
        plan = formal_expansion_plan(comparisons=(comparison_name,))
        if dry_run:
            return {"mode": mode, "status": "planned", "plan": plan, "seed_matrix": seed_matrix_gate([], expected_plan=plan)}
        expansion_results: list[dict[str, Any]] = []
        expansion_root = output_root / "expansion"
        for index, cell in enumerate(plan):
            modalities = tuple(cell["modalities"])
            cell_config = replace(
                config,
                model=str(cell["model"]),
                modalities=modalities,
                in_channels=len(modalities),
                tiny=False,
                early_stopping=True,
            )
            cell_datasets = {
                split: dataset_with_modalities(dataset, modalities)
                for split, dataset in datasets.items()
            }
            cell_name = f"cell_{index:03d}_{cell['comparison']}_{cell['model']}_{'-'.join(modalities)}_seed_{cell['seed']}"
            cell_dir = expansion_root / cell_name
            cell_fp = _mode_fingerprint(base_fingerprint, cell_name, cell_config, cell)
            if str(cell["model"]).lower() in {"statistical", "statistical_logistic", "logistic"}:
                cell_config = replace(cell_config, model="statistical_logistic")
                cell_results = _run_logistic_seeds(
                    cell_datasets,
                    config=cell_config,
                    seeds=(int(cell["seed"]),),
                    destination=cell_dir,
                    fingerprint=cell_fp,
                    identity={"mode": mode, "cell": cell, "manifests": {key: str(value) for key, value in paths.items()}},
                )
            else:
                cell_results = _run_neural_seeds(
                    cell_datasets,
                    config=cell_config,
                    seeds=(int(cell["seed"]),),
                    destination=cell_dir,
                    fingerprint=cell_fp,
                    identity={"mode": mode, "cell": cell, "manifests": {key: str(value) for key, value in paths.items()}},
                )
            expansion_results.extend({**result, **cell} for result in cell_results)
        return {
            "mode": mode,
            "status": "complete" if seed_matrix_gate(expansion_results, expected_plan=plan)["status"] == "PASS" else "INCOMPLETE",
            "results": expansion_results,
            "seed_matrix": seed_matrix_gate(expansion_results, expected_plan=plan),
        }
    if mode not in {"tiny", "positive", "negative", "shuffle", "logistic"}:
        raise ValueError(f"Unsupported control mode: {mode}")

    mode_config = config
    mode_datasets = dict(datasets)
    extra: dict[str, Any] = {}
    if mode == "tiny":
        mode_config = _normal_neural_config(config, mode="tiny", stage="final")
        mode_datasets = _tiny_datasets(
            mode_datasets,
            subjects_per_label=subjects_per_label,
            max_slices=tiny_max_slices,
            min_slices=tiny_min_slices,
            seed=tiny_seed,
        )
        extra.update({"subjects_per_label": subjects_per_label, "tiny_max_slices": tiny_max_slices, "tiny_seed": tiny_seed})
    elif mode == "positive":
        mode_config = _normal_neural_config(config, mode="positive", stage="registered")
        mode_datasets = {
            split: dataset_with_stage(dataset, "registered")
            for split, dataset in mode_datasets.items()
        }
        # The registered source is pre-IQR NIfTI rather than the final LMDB
        # cache.  Cache model-grid slices once per split before fitting the
        # shared train scalar so each epoch reuses exact values in memory.
        mode_datasets = {
            split: materialize_registered_dataset(dataset)
            for split, dataset in mode_datasets.items()
        }
        mode_datasets, scalar = _fit_optional_shared_scalar(
            mode_datasets,
            requested=config.shared_train_scalar,
            fit_from_train=fit_positive_scalar,
        )
        extra.update({
            "stage": "registered",
            "shared_train_scalar": scalar,
            "fit_from_train": bool(fit_positive_scalar),
            "registered_materialized_once": True,
        })
    elif mode == "negative":
        mode_config = _normal_neural_config(config, mode="negative", stage="final")
        # Read each immutable source slice once before constructing the
        # pseudo-domain pairs.  The negative control reassigns participants
        # and labels while retaining source keys; a shared cache keeps those
        # cloned records exact and avoids rereading the LMDB/NPZ source for
        # every epoch and seed.
        mode_datasets = materialize_tensor_datasets(mode_datasets)
        mode_datasets = _same_cohort_negative_datasets(mode_datasets, seed=tiny_seed)
        extra.update({
            "same_cohort": True,
            "negative_seed": tiny_seed,
            "source_label": 0,
            "shared_tensor_cache": True,
            "tensor_cache_materialized_once": True,
            "participant_split_fractions": [0.40, 0.10, 0.50],
        })
    elif mode == "shuffle":
        if shuffle_design not in {"powered_pair_split_40_10_50", "supplemental_train_only"}:
            raise ValueError(f"Unsupported shuffle design {shuffle_design!r}.")
        mode_config = replace(
            _normal_neural_config(config, mode="shuffle", stage="final"),
            max_epochs=40,
            early_stopping=False,
            patience=41,
        )
        # Label transforms are applied independently for each requested seed
        # inside _run_shuffle_seeds.  Keeping the unmodified base manifests
        # here makes the pre-outcome design and provenance explicit.
        # Pair-level powered shuffling moves rows between the original
        # train/validation/test templates.  Materialize all source splits
        # behind one immutable cache so a row moved from (say) val to test
        # remains loadable through the new target split.
        mode_datasets = materialize_tensor_datasets(mode_datasets)
        extra.update(
            {
                "shuffle_design": shuffle_design,
                "unit": "whole_pair_label_swap" if shuffle_design == "powered_pair_split_40_10_50" else "whole_participant",
                "participant_split_fractions": [0.40, 0.10, 0.50] if shuffle_design == "powered_pair_split_40_10_50" else None,
                "train_val_pair_labels_swapped": shuffle_design == "powered_pair_split_40_10_50",
                "validation_labels_true": shuffle_design == "supplemental_train_only",
                "test_labels_true": True,
                "checkpoint_selection": "final_epoch_no_true_validation_selection",
            }
        )
    elif mode == "logistic":
        mode_config = replace(config, model="statistical_logistic")
        extra.update({"feature_control": "intensity_foreground_gradient_sharpness_radial", "train_only_scaler": True})

    mode_fp = _mode_fingerprint(base_fingerprint, mode, mode_config, extra)
    destination = output_root / mode
    identity = {
        "mode": mode,
        "config": asdict(mode_config),
        "manifests": {split: str(path) for split, path in paths.items()},
        "metadata": extra,
    }
    if dry_run:
        return {
            "mode": mode,
            "status": "planned",
            "config": asdict(mode_config),
            "counts": {split: len(getattr(mode_datasets[split], "records", [])) for split in ("train", "val", "test")},
            "output_dir": str(destination),
            "run_fingerprint": mode_fp,
            "metadata": extra,
        }
    if mode == "shuffle":
        design_path = destination / "shuffle_design.json"
        design_payload = {
            "status": "PRE_REGISTERED",
            "design": extra,
            "seeds": [int(seed) for seed in seeds],
            "max_epochs": int(mode_config.max_epochs),
            "early_stopping": bool(mode_config.early_stopping),
            "true_test_labels": True,
            "original_manifest_split_preserved": True,
            "rationale": (
                "Powered shuffle assigns complete original matched pairs to a new 40/10/50 split, "
                "swaps pair labels independently in train and validation for each seed, and leaves "
                "the held-out test labels unchanged. The final epoch is evaluated without using true "
                "validation labels for checkpoint selection."
                if shuffle_design == "powered_pair_split_40_10_50"
                else "Supplemental control shuffles whole training participant labels only and evaluates the final 40-epoch state."
            ),
        }
        if not design_path.exists():
            atomic_write_json(design_path, _json_safe(design_payload))
        control_manifest = {
            "base_manifest": _persist_control_manifests(destination / "base", mode_datasets),
            "seed_manifests": "written_before_each_seed_fit",
        }
        results = _run_shuffle_seeds(
            mode_datasets,
            config=mode_config,
            seeds=seeds,
            destination=destination,
            fingerprint=mode_fp,
            identity=identity,
            design=shuffle_design,
        )
    elif mode == "logistic":
        control_manifest = _persist_control_manifests(destination, mode_datasets)
        results = _run_logistic_seeds(
            mode_datasets,
            config=mode_config,
            seeds=seeds,
            destination=destination,
            fingerprint=mode_fp,
            identity=identity,
        )
    else:
        control_manifest = _persist_control_manifests(destination, mode_datasets)
        results = _run_neural_seeds(
            mode_datasets,
            config=mode_config,
            seeds=seeds,
            destination=destination,
            fingerprint=mode_fp,
            identity=identity,
        )
    if mode == "tiny":
        gate = evaluate_tiny_gate(results)
    elif mode == "positive":
        gate = evaluate_positive_gate(results)
    elif mode == "logistic":
        # Logistic is a formal capacity, not a negative/chance control.  Its
        # result must never be converted into a low-separability gate.
        gate = {
            "name": "statistical_logistic_formal_capacity",
            "status": "COMPLETE" if results else "INCONCLUSIVE",
            "passed": None,
            "n_seeds": len(results),
            "reason": "Formal statistical capacity recorded; no chance-band gate applied.",
        }
    else:
        gate = evaluate_negative_gate(results)
    permutation = None
    if mode in {"negative", "positive"} and int(requested_retrained_permutations) >= 0:
        observed_auc_values: list[float] = []
        for result in results:
            try:
                value = float(result.get("test", {}).get("subject", {}).get("roc_auc", float("nan")))
            except (AttributeError, TypeError, ValueError):
                value = float("nan")
            if np.isfinite(value):
                observed_auc_values.append(value)
        permutation = _run_retrained_permutations(
            mode_datasets,
            config=mode_config,
            destination=destination,
            fingerprint=mode_fp,
            requested=int(requested_retrained_permutations),
            permute_test_labels=bool(permute_test_labels),
            observed_auc=(float(np.mean(observed_auc_values)) if observed_auc_values else None),
        )
    summary = {
        "mode": mode,
        "status": gate["status"],
        "gate": gate,
        "results": results,
        "run_fingerprint": mode_fp,
        "output_dir": str(destination),
        "metadata": extra,
        "control_manifest": control_manifest,
    }
    if permutation is not None:
        summary["retrained_permutations"] = permutation
    # Gate summaries are the resumable aggregate for a mode.  Re-running a
    # completed mode must be able to refresh this aggregate while preserving
    # the immutable per-seed/checkpoint/prediction artifacts.
    atomic_write_json(destination / "gate.json", _json_safe(summary), overwrite=True)
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison", choices=("fomo", "fomo45k", "mpi", "oasis3", "mixed"), default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("tiny", "positive", "negative", "shuffle", "logistic", "expansion", "all"), default="tiny")
    parser.add_argument("--stage", choices=("raw", "final", "registered"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modalities", nargs="+", choices=("flair", "t1", "t2"), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(73, 173, 273))
    parser.add_argument("--tiny-subjects-per-label", type=int, default=10)
    parser.add_argument("--tiny-max-slices", type=int, default=128)
    parser.add_argument("--tiny-min-slices", type=int, default=64)
    parser.add_argument("--tiny-seed", type=int, default=73)
    parser.add_argument("--fit-positive-shared-scalar", action="store_true")
    parser.add_argument(
        "--shuffle-design",
        choices=("powered_pair_split_40_10_50", "supplemental_train_only"),
        default="powered_pair_split_40_10_50",
        help="Shuffle control design; powered pair split is the preregistered default.",
    )
    parser.add_argument("--retrained-permutations", type=int, default=0, help="Whole-pair retraining controls (0 records an explicit incomplete/omitted status).")
    parser.add_argument("--permute-test-labels", dest="permute_test_labels", action="store_true", default=True, help="Full retrained C2ST null: swap whole-pair labels in train/val/test (default).")
    parser.add_argument("--keep-test-labels", dest="permute_test_labels", action="store_false", help="Diagnostic train/validation-only label control; preserve true test labels.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config_path = config_path.resolve()
    config_value = _load_yaml(config_path)
    cfg = _config_from_yaml(config_value, stage=args.stage, device=args.device)
    if args.modalities:
        cfg = replace(cfg, modalities=tuple(args.modalities), in_channels=len(args.modalities))
    paths = _manifest_paths(config_value, comparison=args.comparison, manifest_root=args.manifest_root)
    counts = {split: len(read_jsonl_manifest(path)) for split, path in paths.items()}
    output_root = args.output_dir or Path(config_value.get("output_dir", "outputs/diagnostics/domain_classifier"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root = output_root.resolve()
    code_paths = [
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "metrics.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
        Path(__file__).resolve(),
        config_path,
    ]
    base_fp = run_fingerprint(cfg, paths, code_paths=code_paths)
    if args.dry_run:
        planned = {"mode": args.mode, "base_config": asdict(cfg), "manifests": {key: str(path) for key, path in paths.items()}, "counts": counts, "output_root": str(output_root), "base_fingerprint": base_fp}
        if args.mode == "expansion":
            planned["plan"] = formal_expansion_plan()
        elif args.mode == "all":
            planned["modes"] = ["tiny", "positive", "negative", "shuffle", "logistic"]
        print(json.dumps(_json_safe(planned), indent=2, sort_keys=True))
        return 0
    if args.mode == "expansion":
        comparison_names = [_comparison_name(args.comparison)] if args.comparison is not None else ["fomo45k", "mpi", "oasis3", "mixed"]
        _assert_expansion_prerequisites(output_root, comparison_names)
        reports: list[dict[str, Any]] = []
        for comparison in comparison_names:
            comparison_paths = _manifest_paths(
                config_value,
                comparison=comparison,
                manifest_root=args.manifest_root,
            )
            comparison_fp = run_fingerprint(cfg, comparison_paths, code_paths=code_paths)
            comparison_datasets = {
                split: dataset_from_manifest(
                    path,
                    stage=cfg.stage,
                    modalities=cfg.modalities,
                    shared_train_scalar=cfg.shared_train_scalar,
                )
                for split, path in comparison_paths.items()
            }
            reports.append(
                run_mode(
                    "expansion",
                    config=cfg,
                    datasets=comparison_datasets,
                    paths=comparison_paths,
                    seeds=args.seeds,
                    output_root=output_root,
                    base_fingerprint=comparison_fp,
                    subjects_per_label=args.tiny_subjects_per_label,
                    tiny_max_slices=args.tiny_max_slices,
                    tiny_min_slices=args.tiny_min_slices,
                    tiny_seed=args.tiny_seed,
                    fit_positive_scalar=args.fit_positive_shared_scalar,
                    requested_retrained_permutations=args.retrained_permutations,
                    permute_test_labels=args.permute_test_labels,
                    shuffle_design=args.shuffle_design,
                    comparison_name=comparison,
                )
            )
        all_results = [row for report in reports for row in report.get("results", [])]
        expected_plan = formal_expansion_plan(comparisons=comparison_names)
        summary = {
            "mode": "expansion",
            "reports": reports,
            "seed_matrix": seed_matrix_gate(all_results, expected_plan=expected_plan),
        }
        atomic_write_json(output_root / "expansion_summary.json", _json_safe(summary), overwrite=True)
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
        return 0
    datasets = {
        split: dataset_from_manifest(
            path,
            stage=cfg.stage,
            modalities=cfg.modalities,
            shared_train_scalar=cfg.shared_train_scalar,
        )
        for split, path in paths.items()
    }
    modes = ["tiny", "positive", "negative", "shuffle", "logistic"] if args.mode == "all" else [args.mode]
    reports: list[dict[str, Any]] = []
    halted_on_fail = False
    for mode in modes:
        report = run_mode(
                mode,
                config=cfg,
                datasets=datasets,
                paths=paths,
                seeds=args.seeds,
                output_root=output_root,
                base_fingerprint=base_fp,
                subjects_per_label=args.tiny_subjects_per_label,
                tiny_max_slices=args.tiny_max_slices,
                tiny_min_slices=args.tiny_min_slices,
                tiny_seed=args.tiny_seed,
                fit_positive_scalar=args.fit_positive_shared_scalar,
                requested_retrained_permutations=args.retrained_permutations,
                permute_test_labels=args.permute_test_labels,
                shuffle_design=args.shuffle_design,
            )
        reports.append(report)
        if report.get("status") == "FAIL":
            halted_on_fail = True
            break
    summary = {"mode": args.mode, "manifests": {key: str(path) for key, path in paths.items()}, "counts": counts, "reports": reports, "halted_on_fail": halted_on_fail}
    atomic_write_json(output_root / "controls_summary.json", _json_safe(summary), overwrite=True)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
