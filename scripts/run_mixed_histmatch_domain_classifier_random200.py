"""Randomly sample 200 supported mixed histmatch train slices for domain inference.

The frozen observed SmallCNN estimates the BraTS21 *domain* probability. Its
output is neither a lesion probability nor an accuracy measurement for these
unlabelled mixed slices. The stored float32 LMDB tensor is passed unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from andi_rewrite.scripts.run_healthy_domain_classifier_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MODEL_CHANNELS,
    MODEL_SHAPE,
    SUPPORT_THRESHOLD,
    SUPPORT_TOLERANCE,
    _load_observed_model,
    _predict,
    _runtime_versions,
    _summary_stats,
    file_fingerprint,
    sha256_file,
)


DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888"
)
DEFAULT_OUTPUT_PARENT = REPO_ROOT / "outputs/diagnostics/domain_classifier"
SOURCES = frozenset(("mpi", "oasis3", "fomo45k"))
KEY_PATTERN = re.compile(r"[0-9]{8}\Z")
EXPECTED_NORMALIZATION = {
    "type": "andi_histmatch_mean888_robust_iqr",
    "version": 1,
    "source_normalization": "robust_iqr_v1",
    "reference": "equal_case_mean_BraTS_train888_healthy_tissue_quantiles",
    "operation": "SimpleITK.HistogramMatching_default_parameters_per_case_per_modality",
    "background": -1.0,
    "model_normalize_input": False,
}


class InferenceAuditError(RuntimeError):
    """An input, sampling, or inference contract was not satisfied."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _load_source_entries(path: Path, expected_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InferenceAuditError(f"invalid source manifest JSON at line {index + 1}") from exc
            if not isinstance(row, dict):
                raise InferenceAuditError(f"source manifest row {index + 1} is not an object")
            key = f"{index:08d}"
            if (
                row.get("key") != key
                or row.get("source_split") != "train"
                or row.get("source_dataset") not in SOURCES
                or not isinstance(row.get("source_key"), str)
                or KEY_PATTERN.fullmatch(row["source_key"]) is None
            ):
                raise InferenceAuditError(f"source manifest contract mismatch at key {key}: {row}")
            rows.append({name: str(row[name]) for name in ("key", "source_dataset", "source_key", "source_split")})
    if len(rows) != expected_count:
        raise InferenceAuditError(f"source manifest has {len(rows)} rows, expected {expected_count}")
    return rows


def _read_array(raw: bytes, key: str) -> np.ndarray:
    value = pickle.loads(raw)
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != MODEL_SHAPE:
        raise InferenceAuditError(
            f"invalid LMDB tensor at {key}: type={type(value).__name__}, "
            f"dtype={getattr(value, 'dtype', None)}, shape={getattr(value, 'shape', None)}"
        )
    if not bool(np.isfinite(value).all()):
        raise InferenceAuditError(f"non-finite LMDB tensor at {key}")
    return value


def _support_fraction(value: np.ndarray) -> tuple[float, float, float]:
    fractions = (np.abs(value + np.float32(1.0)) > SUPPORT_TOLERANCE).mean(axis=(1, 2))
    return tuple(float(fraction) for fraction in fractions)  # type: ignore[return-value]


def _sample_eligible(
    eligible: Sequence[tuple[int, tuple[float, float, float], bytes]],
    *,
    sample_size: int,
    seed: int,
) -> list[tuple[int, tuple[float, float, float], bytes]]:
    if len(eligible) < sample_size:
        raise InferenceAuditError(f"only {len(eligible)} slices pass the support gate; need {sample_size}")
    indices = np.random.default_rng(seed).choice(len(eligible), size=sample_size, replace=False)
    selected = sorted((eligible[int(index)] for index in indices), key=lambda row: row[0])
    if len(selected) != sample_size or len({row[0] for row in selected}) != sample_size:
        raise InferenceAuditError("random sample is not unique or has the wrong size")
    return selected


def _scan_lmdb(
    split_dir: Path,
    source_rows: Sequence[dict[str, str]],
) -> tuple[list[tuple[int, tuple[float, float, float], bytes]], str, dict[str, int]]:
    """Hash every raw LMDB value and retain support metadata for eligible rows."""
    eligible: list[tuple[int, tuple[float, float, float], bytes]] = []
    counts: Counter[str] = Counter()
    records_hash = hashlib.sha256()
    env = lmdb.open(str(split_dir), readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    try:
        with env.begin(write=False) as txn:
            if txn.stat()["entries"] != len(source_rows):
                raise InferenceAuditError("LMDB entry count differs from the source manifest")
            for index, (key_bytes, raw) in enumerate(txn.cursor()):
                if index >= len(source_rows):
                    raise InferenceAuditError("LMDB has more rows than the source manifest")
                key = f"{index:08d}"
                if key_bytes != key.encode("ascii"):
                    raise InferenceAuditError(f"LMDB key order mismatch: expected {key}, got {key_bytes!r}")
                value = _read_array(raw, key)
                support = _support_fraction(value)
                value_digest = hashlib.sha256(raw).digest()
                records_hash.update(key_bytes)
                records_hash.update(len(raw).to_bytes(8, "little"))
                records_hash.update(value_digest)
                counts["scanned"] += 1
                if all(fraction >= SUPPORT_THRESHOLD for fraction in support):
                    eligible.append((index, support, value_digest))
                    counts["eligible"] += 1
                    counts[f"eligible_{source_rows[index]['source_dataset']}"] += 1
                else:
                    counts["support_failed"] += 1
            if counts["scanned"] != len(source_rows):
                raise InferenceAuditError("LMDB has fewer rows than the source manifest")
    finally:
        env.close()
    return eligible, records_hash.hexdigest(), dict(counts)


def _predict_selected(
    split_dir: Path,
    selected: Sequence[tuple[int, tuple[float, float, float], bytes]],
    source_rows: Sequence[dict[str, str]],
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    env = lmdb.open(str(split_dir), readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    try:
        with env.begin(write=False) as txn:
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                tensors: list[torch.Tensor] = []
                base_rows: list[dict[str, Any]] = []
                for index, original_support, original_digest in batch:
                    source = source_rows[index]
                    key = source["key"]
                    raw = txn.get(key.encode("ascii"))
                    if raw is None or hashlib.sha256(raw).digest() != original_digest:
                        raise InferenceAuditError(f"selected LMDB value changed or is missing: {key}")
                    value = _read_array(raw, key)
                    support = _support_fraction(value)
                    if support != original_support or any(fraction < SUPPORT_THRESHOLD for fraction in support):
                        raise InferenceAuditError(f"selected slice no longer passes support gate: {key}")
                    tensors.append(torch.from_numpy(np.ascontiguousarray(value)))
                    base_rows.append({
                        "key": key,
                        "split": "train",
                        "source_dataset": source["source_dataset"],
                        "source_key": source["source_key"],
                        "source_split": source["source_split"],
                        "support_fraction": dict(zip(MODEL_CHANNELS, support)),
                        "tensor_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                        "tensor_shape": list(MODEL_SHAPE),
                        "tensor_dtype": "float32",
                    })
                logits, probabilities = _predict(model, tensors, device)
                if len(logits) != len(batch) or len(probabilities) != len(batch):
                    raise InferenceAuditError("classifier prediction count differs from batch size")
                for base, logit, probability in zip(base_rows, logits, probabilities):
                    if not (0.0 <= probability <= 1.0):
                        raise InferenceAuditError(f"invalid classifier probability at {base['key']}")
                    selection_rows.append(base)
                    prediction_rows.append({
                        **base,
                        "logit": logit,
                        "probability_brats21_domain": probability,
                        "predicted_domain_at_0_5": "BraTS21-like" if probability >= 0.5 else "FOMO-like",
                    })
    finally:
        env.close()
    return selection_rows, prediction_rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / "train"
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    stage_dir = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or stage_dir.exists():
        raise InferenceAuditError(f"output or staging directory already exists: {output_dir}")
    if not split_dir.is_dir() or not checkpoint.is_file():
        raise InferenceAuditError("input train LMDB or observed checkpoint is missing")

    report_path = dataset_root / "build_report.json"
    reference_path = dataset_root / "reference.npz"
    normalization_path = split_dir / "normalization.json"
    source_path = dataset_root / "source_entries.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("entries", {}).get("train") != 84341:
        raise InferenceAuditError("histmatch dataset build report is not a passing 84,341-row build")
    if {name: normalization.get(name) for name in EXPECTED_NORMALIZATION} != EXPECTED_NORMALIZATION:
        raise InferenceAuditError("histmatch normalization contract differs from the supported model input")
    reference_sha256 = sha256_file(reference_path)
    if reference_sha256 != report.get("reference_sha256") or reference_sha256 != normalization.get("reference_sha256"):
        raise InferenceAuditError("BraTS reference hash differs across dataset metadata")
    source_rows = _load_source_entries(source_path, expected_count=84341)
    input_fingerprints = {
        "build_report": file_fingerprint(report_path),
        "reference": file_fingerprint(reference_path),
        "normalization": file_fingerprint(normalization_path),
        "source_entries": file_fingerprint(source_path),
        "data_mdb_stat": file_fingerprint(split_dir / "data.mdb", content=False),
    }
    checkpoint_fingerprint = file_fingerprint(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise InferenceAuditError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, model_contract = _load_observed_model(checkpoint, device)

    stage_dir.mkdir(parents=True)
    try:
        _write_json(stage_dir / "run_status.json", {"status": "RUNNING", "started_at_utc": _now()})
        eligible, records_sha256, scan_counts = _scan_lmdb(split_dir, source_rows)
        selected = _sample_eligible(eligible, sample_size=args.sample_size, seed=args.seed)
        # Recompute the draw to audit exact seed and choice semantics.
        replay = _sample_eligible(eligible, sample_size=args.sample_size, seed=args.seed)
        if [row[0] for row in selected] != [row[0] for row in replay]:
            raise InferenceAuditError("random sample is not reproducible")
        selection_rows, prediction_rows = _predict_selected(
            split_dir, selected, source_rows, model=model, device=device, batch_size=args.batch_size
        )
        if len(selection_rows) != args.sample_size or len(prediction_rows) != args.sample_size:
            raise InferenceAuditError("selection or inference output does not contain the requested sample size")
        if len({row["key"] for row in selection_rows}) != args.sample_size:
            raise InferenceAuditError("selection output contains duplicate LMDB keys")
        final_data_stat = file_fingerprint(split_dir / "data.mdb", content=False)
        if final_data_stat != input_fingerprints["data_mdb_stat"]:
            raise InferenceAuditError("train LMDB file changed during inference")

        selection_path = stage_dir / "selection_manifest.jsonl"
        prediction_path = stage_dir / "slice_predictions.jsonl"
        _write_jsonl(selection_path, selection_rows)
        _write_jsonl(prediction_path, prediction_rows)
        output_fingerprints = {
            "selection_manifest": file_fingerprint(selection_path),
            "slice_predictions": file_fingerprint(prediction_path),
        }
        # The staging directory is renamed below; published provenance must
        # point at the durable file locations rather than the temporary ones.
        output_fingerprints["selection_manifest"]["path"] = str(output_dir / selection_path.name)
        output_fingerprints["slice_predictions"]["path"] = str(output_dir / prediction_path.name)
        selected_counts = dict(Counter(row["source_dataset"] for row in selection_rows))
        source_probability_stats = {
            source: _summary_stats([
                row["probability_brats21_domain"] for row in prediction_rows
                if row["source_dataset"] == source
            ])
            for source in sorted(SOURCES)
        }
        summary = {
            "status": "PASS",
            "completed_at_utc": _now(),
            "elapsed_seconds": time.time() - started,
            "dataset_root": str(dataset_root),
            "split": "train",
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "scan_counts": scan_counts,
            "selected_source_counts": selected_counts,
            "predicted_domain_counts_at_0_5": dict(Counter(
                row["predicted_domain_at_0_5"] for row in prediction_rows
            )),
            "probability_brats21_domain_stats": _summary_stats([
                row["probability_brats21_domain"] for row in prediction_rows
            ]),
            "source_probability_stats": source_probability_stats,
            "prediction_count": len(prediction_rows),
            "label_status": "NO_COMPARABLE_DOMAIN_LABELS_FOR_MIXED_HISTMATCH_SLICES",
            "auc_status": "NOT_COMPUTED",
            "output_sha256": {name: record["sha256"] for name, record in output_fingerprints.items()},
        }
        provenance = {
            "schema_version": 1,
            "status": "PASS",
            "purpose": "inference_only_fomo_vs_brats21_domain_classifier_on_random_mixed_histmatch_train_slices",
            "input": {"dataset_root": str(dataset_root), "split": "train", "fingerprints": input_fingerprints},
            "train_lmdb_records_sha256": records_sha256,
            "train_lmdb_records_hash_scheme": "sha256(concat(key_ascii_8, raw_value_length_uint64_le, sha256(raw_lmdb_value))) in LMDB key order",
            "checkpoint": {**checkpoint_fingerprint, **model_contract},
            "sampling": {
                "eligible_rule": "each channel mean(abs(x + 1) > 1e-6) >= 0.10",
                "population": "all eligible train slice keys",
                "method": "numpy.random.default_rng(seed).choice(eligible_count, size=sample_size, replace=False)",
                "seed": args.seed,
                "sample_size": args.sample_size,
                "ordering_after_selection": "ascending train LMDB key",
                "source_balancing": False,
            },
            "model_input": {
                "channel_order": list(MODEL_CHANNELS),
                "dtype": "float32",
                "shape": list(MODEL_SHAPE),
                "normalization": normalization,
                "additional_transform": None,
            },
            "runtime": {**_runtime_versions(), "device": str(device), "batch_size": args.batch_size},
            "code_fingerprints": [
                file_fingerprint(Path(__file__)),
                file_fingerprint(REPO_ROOT / "scripts/run_healthy_domain_classifier_inference.py"),
                file_fingerprint(REPO_ROOT / "domain_classifier/models.py"),
            ],
            "summary": summary,
        }
        audit = {
            "status": "PASS",
            "checks": {
                "build_report_pass": True,
                "normalization_and_reference_hash_match": True,
                "manifest_and_lmdb_exactly_84341_contiguous_keys": True,
                "all_tensors_float32_3x128x128_and_finite": True,
                "support_gate_before_sampling": True,
                "sample_unique_and_seed_replay_equal": True,
                "selected_values_match_scanned_digests": True,
                "checkpoint_strict_load": True,
                "no_additional_input_transform": True,
                "prediction_count_and_probability_range": True,
                "lmdb_file_stat_unchanged": True,
            },
            "selection_rows": len(selection_rows),
            "prediction_rows": len(prediction_rows),
            "output_fingerprints": output_fingerprints,
        }
        _write_json(stage_dir / "summary.json", summary)
        _write_json(stage_dir / "provenance.json", provenance)
        _write_json(stage_dir / "audit.json", audit)
        _write_json(stage_dir / "run_status.json", {"status": "PASS", "completed_at_utc": summary["completed_at_utc"]})
        stage_dir.rename(output_dir)
        return summary
    except Exception as exc:
        _write_json(stage_dir / "failure.json", {"status": "FAIL", "completed_at_utc": _now(), "error": str(exc)})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.sample_size < 1 or args.batch_size < 1:
        parser.error("--sample-size and --batch-size must be positive")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_PARENT / f"mixed_histmatch_mean888_train_random{args.sample_size}_seed{args.seed}"
    try:
        summary = _run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
