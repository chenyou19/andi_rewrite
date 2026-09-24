"""Score one fixed observed classifier against independent pair-flipped test labels.

This helper deliberately does no fitting.  It joins an observed fit's immutable
slice predictions to the v3 test manifest, verifies the true labels and pair
metadata, and then evaluates the same probabilities against three independent
whole-pair label-flip streams.  The random-label results are a separate sanity
gate; they are not the historical train-label shuffle control and do not alter
the true test labels or the saved probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.metrics import (  # noqa: E402
    bootstrap_subject_auc,
    compute_dataset_metrics,
    paired_heldout_swap_test,
)
from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    _strip_prediction_arrays,
    atomic_write_json,
    evaluate_negative_gate,
    permute_pair_labels,
    read_jsonl_manifest,
)


DEFAULT_SWAP_SEEDS = (10073, 10173, 10273)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Prediction row {path}:{line_number} is not an object.")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if value is None else value


def _validate_join(
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[float]]:
    if len(manifest_rows) != len(prediction_rows):
        raise ValueError(
            "Prediction/manifest row count mismatch: "
            f"predictions={len(prediction_rows)}, manifest={len(manifest_rows)}."
        )
    labels: list[int] = []
    scores: list[float] = []
    for index, (manifest, prediction) in enumerate(zip(manifest_rows, prediction_rows)):
        if int(_value(prediction, "record_index", -1)) != index:
            raise ValueError(f"Prediction record_index at row {index} is not the manifest order.")
        for field in ("participant_id", "pair_id", "case_id"):
            manifest_value = _value(manifest, field, "")
            prediction_value = _value(prediction, field, "")
            if field in {"participant_id", "pair_id"} and (
                manifest_value is None
                or prediction_value is None
                or str(manifest_value).strip() == ""
                or str(prediction_value).strip() == ""
            ):
                raise ValueError(f"Prediction/manifest {field} is missing at row {index}.")
            if str(prediction_value) != str(manifest_value):
                raise ValueError(f"Prediction/manifest {field} mismatch at row {index}.")
        manifest_label = int(_value(manifest, "label", -1))
        prediction_label = int(_value(prediction, "label", -1))
        if manifest_label not in (0, 1) or prediction_label != manifest_label:
            raise ValueError(f"True test label mismatch at row {index}.")
        score = float(_value(prediction, "probability", float("nan")))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Prediction probability at row {index} is not finite in [0, 1].")
        labels.append(manifest_label)
        scores.append(score)
    return labels, scores


def _metrics_and_statistics(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    bootstrap_seed: int,
    swap_seed: int,
    bootstrap_replicates: int,
    swap_replicates: int,
) -> dict[str, Any]:
    participants = [str(_value(row, "participant_id", "")) for row in rows]
    pair_ids = [str(_value(row, "pair_id", "")) for row in rows]
    cases = [str(_value(row, "case_id", index)) for index, row in enumerate(rows)]
    metrics = compute_dataset_metrics(
        labels,
        scores,
        participant_ids=participants,
        pair_ids=pair_ids,
        threshold=0.5,
    )
    subject = metrics.get("subject_predictions")
    if not isinstance(subject, Mapping):
        raise ValueError("Test manifest must contain participant IDs for subject metrics.")
    subject_labels = subject["labels"]
    subject_scores = subject["scores"]
    subject_pairs = subject["pair_ids"]
    statistics = {
        "subject_bootstrap": bootstrap_subject_auc(
            subject_labels,
            subject_scores,
            pair_ids=subject_pairs,
            n_bootstrap=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        ),
        "heldout_pair_swap": paired_heldout_swap_test(
            subject_labels,
            subject_scores,
            subject_pairs,
            n_swaps=int(swap_replicates),
            seed=int(swap_seed),
        ),
    }
    return {
        "test": _strip_prediction_arrays(metrics),
        "test_statistics": _strip_prediction_arrays(statistics),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_immutable_test_labels(
    *,
    predictions_path: Path,
    test_manifest: Path,
    output_root: Path,
    swap_seeds: Sequence[int] = DEFAULT_SWAP_SEEDS,
    bootstrap_replicates: int = 2000,
    swap_replicates: int = 1000,
) -> dict[str, Any]:
    swap_seeds = tuple(int(seed) for seed in swap_seeds)
    if swap_seeds != DEFAULT_SWAP_SEEDS:
        raise ValueError(f"The frozen protocol requires swap seeds {DEFAULT_SWAP_SEEDS!r}.")
    if int(bootstrap_replicates) <= 0 or int(swap_replicates) <= 0:
        raise ValueError("Bootstrap and held-out swap replicate counts must be positive.")
    if not predictions_path.is_file() or not test_manifest.is_file():
        raise FileNotFoundError("Both immutable predictions and the test manifest are required.")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl_manifest(test_manifest)
    prediction_rows = _read_jsonl(predictions_path)
    true_labels, scores = _validate_join(manifest_rows, prediction_rows)
    pair_ids = [str(_value(row, "pair_id", "")) for row in manifest_rows]
    if any(not pair for pair in pair_ids):
        raise ValueError("Every held-out test row requires a non-empty pair_id.")

    true_result = _metrics_and_statistics(
        manifest_rows,
        true_labels,
        scores,
        bootstrap_seed=73,
        swap_seed=73,
        bootstrap_replicates=bootstrap_replicates,
        swap_replicates=swap_replicates,
    )
    _write_jsonl(
        output_root / "test_predictions_true.jsonl",
        [
            {
                "record_index": index,
                "true_label": int(true_labels[index]),
                "probability": float(scores[index]),
                "participant_id": _value(manifest_rows[index], "participant_id"),
                "pair_id": _value(manifest_rows[index], "pair_id"),
                "case_id": _value(manifest_rows[index], "case_id", index),
            }
            for index in range(len(manifest_rows))
        ],
    )

    random_results: list[dict[str, Any]] = []
    for swap_seed in swap_seeds:
        swapped_rows = permute_pair_labels(manifest_rows, seed=int(swap_seed))
        random_labels = [int(row["label"]) for row in swapped_rows]
        result = _metrics_and_statistics(
            manifest_rows,
            random_labels,
            scores,
            bootstrap_seed=int(swap_seed),
            swap_seed=int(swap_seed) + 1,
            bootstrap_replicates=bootstrap_replicates,
            swap_replicates=swap_replicates,
        )
        result["seed"] = int(swap_seed)
        random_results.append(result)
        _write_jsonl(
            output_root / f"test_predictions_random_seed_{int(swap_seed)}.jsonl",
            [
                {
                    "record_index": index,
                    "true_label": int(true_labels[index]),
                    "random_label": int(random_labels[index]),
                    "probability": float(scores[index]),
                    "participant_id": _value(manifest_rows[index], "participant_id"),
                    "pair_id": _value(manifest_rows[index], "pair_id"),
                    "case_id": _value(manifest_rows[index], "case_id", index),
                }
                for index in range(len(manifest_rows))
            ],
        )

    gate = evaluate_negative_gate(random_results, alpha=0.01)
    protocol = {
        "protocol": "immutable_observed_test_random_pair_labels_v1",
        "status": "COMPLETE",
        "fit_retrained": False,
        "probabilities_immutable": True,
        "true_test_labels_preserved": True,
        "swap_seeds": [int(seed) for seed in swap_seeds],
        "bootstrap_replicates": int(bootstrap_replicates),
        "heldout_swap_replicates": int(swap_replicates),
        "predictions_path": str(predictions_path.resolve()),
        "test_manifest": str(test_manifest.resolve()),
        "predictions_sha256": _sha256(predictions_path),
        "test_manifest_sha256": _sha256(test_manifest),
        "n_test_slices": len(manifest_rows),
        "n_test_participants": len({str(_value(row, "participant_id", "")) for row in manifest_rows}),
        "true_test": true_result,
        "random_test_label_gate": gate,
        "random_test_label_results": random_results,
    }
    atomic_write_json(output_root / "protocol.json", _json_safe(protocol), overwrite=False)
    atomic_write_json(output_root / "gate_random_test_labels.json", _json_safe(gate), overwrite=False)
    atomic_write_json(output_root / "result.json", _json_safe(protocol), overwrite=False)
    return _json_safe(protocol)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--swap-seeds", nargs=3, type=int, default=DEFAULT_SWAP_SEEDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--heldout-swaps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "predictions": (args.predictions if args.predictions.is_absolute() else REPO_ROOT / args.predictions).resolve(),
        "test_manifest": (args.test_manifest if args.test_manifest.is_absolute() else REPO_ROOT / args.test_manifest).resolve(),
        "output_root": (args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root).resolve(),
    }
    if args.dry_run:
        print(json.dumps(_json_safe({
            "protocol": "immutable_observed_test_random_pair_labels_v1",
            "paths": {key: str(value) for key, value in paths.items()},
            "swap_seeds": [int(seed) for seed in args.swap_seeds],
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "heldout_swap_replicates": int(args.heldout_swaps),
            "training_started": False,
        }), indent=2, sort_keys=True))
        return 0
    result = evaluate_immutable_test_labels(
        predictions_path=paths["predictions"],
        test_manifest=paths["test_manifest"],
        output_root=paths["output_root"],
        swap_seeds=args.swap_seeds,
        bootstrap_replicates=args.bootstrap_replicates,
        swap_replicates=args.heldout_swaps,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["random_test_label_gate"]["status"] in {"PASS", "INCONCLUSIVE"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
