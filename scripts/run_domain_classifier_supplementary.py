"""Run the post-failure balanced powered-shuffle diagnostic.

This protocol is a new diagnostic and never replaces the historical random
powered-shuffle failure.  It fixes the train/validation pair-flip counts at
exactly 48/96 and 12/24, keeps the held-out test labels true for fitting, and
also evaluates independent random test-label swaps as a separate sanity gate.
All source slices are materialized once through the existing exact final
reader and shared across the three saved fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
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
    bootstrap_subject_auc,
    compute_dataset_metrics,
    dataset_from_manifest,
    dataset_with_modalities,
    evaluate_negative_gate,
    materialize_tensor_datasets,
    paired_heldout_swap_test,
    permute_pair_labels,
    read_jsonl_manifest,
    subset_dataset,
    write_prediction_rows,
)
from scripts.run_domain_classifier_controls import (  # noqa: E402
    _atomic_torch_save,
    _json_safe,
    _persist_control_manifests,
    _powered_shuffle_datasets,
)


DEFAULT_SEEDS = (73, 173, 273)
DEFAULT_TEST_SWAP_SEEDS = (10073, 10173, 10273)
SPLITS = ("train", "val", "test")


def _record_value(record: Any, field: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        value = record.get(field, default)
    else:
        value = getattr(record, field, default)
    return default if value is None else value


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if not callable(converter):
        raise TypeError(f"Expected mapping or to_dict record, got {type(record).__name__}.")
    return dict(converter())


def _row_key(row: Any) -> tuple[str, str, str, str, str, int]:
    return (
        str(_record_value(row, "source_dataset", "")),
        str(_record_value(row, "source_split", "")),
        str(_record_value(row, "source_key", "")),
        str(_record_value(row, "participant_id", "")),
        str(_record_value(row, "case_id", "")),
        int(_record_value(row, "z", 0)),
    )


def _copy_rows(dataset: Any) -> list[dict[str, Any]]:
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Supplementary control requires manifest-backed datasets.")
    return [_record_mapping(record) for record in records]


def _balanced_label_rows(
    datasets: Mapping[str, Any],
    *,
    seed: int,
    split_seed: int = 73,
    true_labels: Mapping[tuple[str, str, str, str, str, int], int],
) -> dict[str, ManifestDataset]:
    """Assign powered pairs then flip exactly half train and validation pairs."""

    # Freeze pair-to-split assignment across protocol-v2 fits.  Only model
    # initialization and the explicit label-flip stream vary by fit seed.
    provisional = _powered_shuffle_datasets(datasets, seed=int(split_seed))
    output: dict[str, ManifestDataset] = {}
    for split_index, split in enumerate(SPLITS):
        rows = _copy_rows(provisional[split])
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_pair[str(row.get("pair_id", ""))].append(row)
        pair_ids = sorted(pair for pair in by_pair if pair)
        if split in {"train", "val"}:
            if len(pair_ids) % 2:
                raise ValueError(f"Balanced {split} pair count must be even, got {len(pair_ids)}.")
            flip_count = len(pair_ids) // 2
            stream = np.random.SeedSequence([int(seed), 0xBA1A, int(split_index)])
            rng = np.random.default_rng(stream)
            flip_pairs = {
                pair_ids[int(index)]
                for index in rng.choice(len(pair_ids), size=flip_count, replace=False).tolist()
            }
        else:
            flip_pairs = set()
        for row in rows:
            pair = str(row.get("pair_id", ""))
            true_label = int(true_labels[_row_key(row)])
            flipped = pair in flip_pairs
            row["label"] = 1 - true_label if flipped else true_label
            metadata = dict(row.get("metadata", {}) or {})
            metadata.update(
                {
                    "supplementary_control": "balanced_powered_shuffle_v2",
                    "supplementary_seed": int(seed),
                    "split_seed": int(split_seed),
                    "pair_labels_flipped": bool(flipped),
                    "test_labels_true": split == "test",
                    "flip_stream": [int(seed), 0xBA1A, int(split_index)],
                }
            )
            row["metadata"] = metadata
        output[split] = ManifestDataset(
            rows,
            loader=getattr(provisional[split], "loader", None),
            stage=getattr(provisional[split], "stage", "final"),
            modalities=getattr(provisional[split], "modalities", ("flair", "t1", "t2")),
        )
        observed_flips = sum(
            1
            for pair in pair_ids
            if pair in flip_pairs
        )
        expected_flips = len(pair_ids) // 2 if split in {"train", "val"} else 0
        if observed_flips != expected_flips:
            raise AssertionError(f"Balanced {split} flip count {observed_flips} != {expected_flips}.")
    return output


def _pair_counts(datasets: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in SPLITS:
        rows = _copy_rows(datasets[split])
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_pair[str(row.get("pair_id", ""))].append(row)
        flipped = 0
        for pair_rows in by_pair.values():
            values = {bool((row.get("metadata") or {}).get("pair_labels_flipped", False)) for row in pair_rows}
            if len(values) != 1:
                raise ValueError(f"Pair has inconsistent flip metadata in {split}.")
            flipped += int(next(iter(values)))
        report[split] = {
            "pairs": len(by_pair),
            "flipped_pairs": int(flipped),
            "unflipped_pairs": int(len(by_pair) - flipped),
            "records": len(rows),
        }
    return report


def _metrics_with_statistics(
    labels: Sequence[int],
    scores: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    participants = [str(row["participant_id"]) for row in rows]
    pair_ids = [str(row["pair_id"]) for row in rows]
    cases = [str(row.get("case_id", index)) for index, row in enumerate(rows)]
    metrics = compute_dataset_metrics(
        labels,
        scores,
        participant_ids=participants,
        pair_ids=pair_ids,
        threshold=0.5,
    )
    subject = metrics.get("subject_predictions")
    if not isinstance(subject, Mapping):
        raise RuntimeError("Supplementary test metrics require subject predictions.")
    subject_pairs = subject["pair_ids"]
    statistics = {
        "subject_bootstrap": bootstrap_subject_auc(
            subject["labels"],
            subject["scores"],
            pair_ids=subject_pairs,
            n_bootstrap=2000,
            seed=int(bootstrap_seed),
        ),
        "heldout_pair_swap": paired_heldout_swap_test(
            subject["labels"],
            subject["scores"],
            subject_pairs,
            n_swaps=1000,
            seed=int(seed),
        ),
    }
    prediction_rows = [
        {
            "record_index": int(index),
            "label": int(labels[index]),
            "probability": float(scores[index]),
            "participant_id": participants[index],
            "pair_id": pair_ids[index],
            "case_id": cases[index],
        }
        for index in range(len(rows))
    ]
    return {
        "test": _strip_prediction_arrays(metrics),
        "test_statistics": _strip_prediction_arrays(statistics),
        "test_predictions": prediction_rows,
    }


def _random_test_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    return permute_pair_labels(rows, seed=int(seed))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _base_identity(base_paths: Mapping[str, Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for split, path in base_paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[split] = digest
    return {"paths": {split: str(path) for split, path in base_paths.items()}, "sha256": files}


def run_supplementary(
    *,
    manifest_root: Path,
    output_root: Path,
    config: TrainConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_swap_seeds: Sequence[int] = DEFAULT_TEST_SWAP_SEEDS,
) -> dict[str, Any]:
    if len(tuple(seeds)) != 3 or len(tuple(test_swap_seeds)) != 3:
        raise ValueError("Supplementary protocol requires exactly three fit and test-swap seeds.")
    base_paths = {
        split: manifest_root / f"{split}.jsonl" for split in SPLITS
    }
    if any(not path.is_file() for path in base_paths.values()):
        raise FileNotFoundError(f"Missing supplementary manifest under {manifest_root}.")
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: read_jsonl_manifest(path) for split, path in base_paths.items()}
    true_labels = {_row_key(row): int(row["label"]) for rows in rows_by_split.values() for row in rows}
    source_datasets = {
        split: dataset_from_manifest(
            path,
            stage="final",
            modalities=config.modalities,
            shared_train_scalar=config.shared_train_scalar,
        )
        for split, path in base_paths.items()
    }
    cached = materialize_tensor_datasets(source_datasets)
    cache_digest = hashlib.sha256()
    cache_count = 0
    for split in SPLITS:
        for index, record in enumerate(getattr(cached[split], "records")):
            tensor = cached[split][index]["image"].detach().cpu().contiguous()
            cache_digest.update(repr(_row_key(record)).encode("utf-8"))
            cache_digest.update(tensor.numpy().tobytes())
            cache_count += 1
    protocol = {
        "status": "PRE_REGISTERED",
        "protocol": "balanced_powered_shuffle_v2",
        "historical_failed_control": str(output_root.parent / "fomo45k_cached2" / "shuffle" / "gate.json"),
        "fit_seeds": [int(seed) for seed in seeds],
        "test_swap_seeds": [int(seed) for seed in test_swap_seeds],
        "pair_split_fractions": [0.40, 0.10, 0.50],
        "split_seed": 73,
        "exact_pair_flips": {"train": "48/96", "val": "12/24", "test": "0/120"},
        "max_epochs": 40,
        "early_stopping": False,
        "test_labels_true_for_fit": True,
        "random_test_label_sanity_separate_gate": True,
        "chance_ci_band": [0.35, 0.65],
        "alpha": 0.01,
        "retrained_permutation_null": "not_run",
        "source_manifest_identity": _base_identity(base_paths),
        "tensor_cache": {
            "rows": cache_count,
            "sha256": cache_digest.hexdigest(),
            "materialized_once": True,
            "reader_stage": "final",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_root / "protocol.json", _json_safe(protocol), overwrite=False)
    atomic_write_json(output_root / "source_cache_digest.json", _json_safe(protocol["tensor_cache"]), overwrite=False)
    true_results: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    runner_config = replace(config, max_epochs=40, early_stopping=False, patience=41, tiny=False, stage="final")
    for seed, test_swap_seed in zip(tuple(seeds), tuple(test_swap_seeds)):
        seed = int(seed)
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        seed_datasets = _balanced_label_rows(cached, seed=seed, split_seed=73, true_labels=true_labels)
        manifest_info = _persist_control_manifests(seed_root, seed_datasets)
        pair_info = _pair_counts(seed_datasets)
        atomic_write_json(seed_root / "pair_flip_audit.json", _json_safe(pair_info), overwrite=False)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        result = DomainClassifierRunner(runner_config).train_one_seed(
            seed_datasets["train"],
            seed_datasets["val"],
            seed_datasets["test"],
            seed=seed,
        )
        elapsed = float(time.perf_counter() - started)
        finished_at = datetime.now(timezone.utc).isoformat()
        checkpoint = result.get("state_dict")
        if checkpoint is None:
            raise RuntimeError(f"Seed {seed} returned no final state_dict.")
        _atomic_torch_save(checkpoint, seed_root / f"model_seed_{seed}.pt")
        true_predictions = result.get("test_predictions", [])
        write_prediction_rows(seed_root / "test_predictions_true.jsonl", true_predictions)
        random_rows = _random_test_rows(_copy_rows(seed_datasets["test"]), seed=int(test_swap_seed))
        score_by_index = {
            int(row["record_index"]): float(row["probability"])
            for row in true_predictions
        }
        random_labels = [int(row["label"]) for row in random_rows]
        random_scores = [score_by_index[index] for index in range(len(random_rows))]
        random_metrics = _metrics_with_statistics(
            random_labels,
            random_scores,
            random_rows,
            seed=seed,
            bootstrap_seed=int(test_swap_seed),
        )
        random_predictions = []
        for index, row in enumerate(random_rows):
            random_predictions.append(
                {
                    "record_index": index,
                    "true_label": int(_record_value(seed_datasets["test"].records[index], "label")),
                    "random_label": int(row["label"]),
                    "probability": float(random_scores[index]),
                    "participant_id": str(row["participant_id"]),
                    "pair_id": str(row["pair_id"]),
                    "case_id": str(row.get("case_id", index)),
                }
            )
        _write_rows(seed_root / f"test_predictions_random_seed_{int(test_swap_seed)}.jsonl", random_predictions)
        serializable = _json_safe(
            {
                "seed": seed,
                "test_swap_seed": int(test_swap_seed),
                "config": asdict(runner_config),
                "epochs_completed": result.get("epochs_completed"),
                "history": result.get("history"),
                "test": result.get("test"),
                "test_statistics": result.get("test_statistics"),
                "test_predictions": true_predictions,
                "true_test": {
                    "metrics": result.get("test"),
                    "statistics": result.get("test_statistics"),
                },
                "random_test": random_metrics,
                "control_manifest": manifest_info,
                "pair_flip_audit": pair_info,
                "timing": {
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "elapsed_seconds": elapsed,
                },
                "retrained_permutation_null": {"status": "not_run", "requested": 0},
            }
        )
        atomic_write_json(seed_root / "result.json", serializable, overwrite=False)
        true_results.append(
            {
                "seed": seed,
                "test": result.get("test", {}),
                "test_statistics": result.get("test_statistics", {}),
            }
        )
        random_results.append(
            {
                "seed": seed,
                "test": random_metrics["test"],
                "test_statistics": random_metrics["test_statistics"],
            }
        )
    true_gate = evaluate_negative_gate(true_results)
    random_gate = evaluate_negative_gate(random_results)
    summary = {
        "protocol": "balanced_powered_shuffle_v2",
        "status": "PASS" if true_gate["status"] == "PASS" and random_gate["status"] == "PASS" else "FAIL" if "FAIL" in {true_gate["status"], random_gate["status"]} else "INCONCLUSIVE",
        "true_test_gate": true_gate,
        "random_test_label_gate": random_gate,
        "true_test_results": true_results,
        "random_test_label_results": random_results,
        "retrained_permutation_null": {"status": "not_run", "requested": 0},
        "historical_gate_preserved": str(output_root.parent / "fomo45k_cached2" / "shuffle" / "gate.json"),
        "output_root": str(output_root),
    }
    atomic_write_json(output_root / "gate_true_test.json", _json_safe(true_gate), overwrite=False)
    atomic_write_json(output_root / "gate_random_test_labels.json", _json_safe(random_gate), overwrite=False)
    atomic_write_json(output_root / "gate.json", _json_safe(summary), overwrite=False)
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    config = TrainConfig(
        model="small_cnn",
        in_channels=3,
        modalities=("flair", "t1", "t2"),
        max_epochs=40,
        early_stopping=False,
        patience=41,
        device=str(args.device),
        bootstrap_replicates=2000,
        swap_replicates=1000,
        permutation_mode="none",
        permutation_replicates=0,
    )
    summary = run_supplementary(
        manifest_root=manifest_root.resolve(),
        output_root=output_root.resolve(),
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "INCONCLUSIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
