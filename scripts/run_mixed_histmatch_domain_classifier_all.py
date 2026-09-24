"""Infer every mixed train slice with the observed domain classifier.

All 84,341 LMDB entries are inferred, including entries below the classifier's
usual 10% per-channel support threshold. Each prediction records that gate's
status so the low-support subset can be examined separately. The probability
is for the BraTS21 domain, not for a lesion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    _load_observed_model,
    _predict,
    _runtime_versions,
    _summary_stats,
    file_fingerprint,
    sha256_file,
)
from andi_rewrite.scripts.run_mixed_histmatch_domain_classifier_random200 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    EXPECTED_NORMALIZATION,
    InferenceAuditError,
    _load_source_entries,
    _read_array,
    _support_fraction,
)
from andi_rewrite.data.robust_normalization import ROBUST_SPEC  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/diagnostics/domain_classifier/mixed_histmatch_mean888_train_all84341"
)
EXPECTED_COUNT = 84341


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / "train"
    output_dir = args.output_dir.resolve()
    stage_dir = output_dir.with_name(output_dir.name + ".staging")
    checkpoint = args.checkpoint.resolve()
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
    if report.get("status") != "PASS":
        raise InferenceAuditError("mixed dataset build report did not pass")
    if normalization.get("type") == "andi_histmatch_mean888_robust_iqr":
        dataset_kind = "andi_histmatch_mean888_robust_iqr"
        if report.get("entries", {}).get("train") != EXPECTED_COUNT:
            raise InferenceAuditError("histmatch build report does not record 84,341 train rows")
        if {name: normalization.get(name) for name in EXPECTED_NORMALIZATION} != EXPECTED_NORMALIZATION:
            raise InferenceAuditError("histmatch normalization contract differs from model input")
        reference_sha256 = sha256_file(reference_path)
        if reference_sha256 != report.get("reference_sha256") or reference_sha256 != normalization.get("reference_sha256"):
            raise InferenceAuditError("BraTS reference hash differs across dataset metadata")
    elif normalization.get("type") == "robust_iqr":
        dataset_kind = "robust_iqr"
        if normalization != ROBUST_SPEC:
            raise InferenceAuditError("old mixed dataset robust-IQR contract differs from the classifier input")
        if (
            report.get("total_entries") != EXPECTED_COUNT
            or report.get("shape") != list(MODEL_SHAPE)
            or report.get("dtype") != "float32"
            or report.get("channel_order") != ["FLAIR", "T1", "T2"]
        ):
            raise InferenceAuditError("old mixed dataset build report does not match the 84,341-row model input")
    else:
        raise InferenceAuditError(f"unsupported mixed dataset normalization: {normalization.get('type')!r}")
    source_rows = _load_source_entries(source_path, EXPECTED_COUNT)
    input_fingerprints = {
        "build_report": file_fingerprint(report_path),
        "normalization": file_fingerprint(normalization_path),
        "source_entries": file_fingerprint(source_path),
        "data_mdb_stat": file_fingerprint(split_dir / "data.mdb", content=False),
    }
    if dataset_kind == "andi_histmatch_mean888_robust_iqr":
        input_fingerprints["reference"] = file_fingerprint(reference_path)
    checkpoint_fingerprint = file_fingerprint(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise InferenceAuditError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, model_contract = _load_observed_model(checkpoint, device)

    stage_dir.mkdir(parents=True)
    try:
        _write_json(stage_dir / "run_status.json", {"status": "RUNNING", "started_at_utc": _now()})
        predictions_path = stage_dir / "slice_predictions.jsonl"
        records_hash = hashlib.sha256()
        source_counts: Counter[str] = Counter()
        support_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        support_domain_counts: Counter[str] = Counter()
        all_probabilities: list[float] = []
        source_probabilities: dict[str, list[float]] = {"mpi": [], "oasis3": [], "fomo45k": []}
        support_probabilities: dict[str, list[float]] = {"pass": [], "fail": []}
        prediction_count = 0
        model_batch_seconds = 0.0
        model_batches = 0

        def write_batch(
            handle: Any,
            batch_tensors: list[torch.Tensor],
            batch_rows: list[dict[str, Any]],
        ) -> None:
            nonlocal prediction_count, model_batch_seconds, model_batches
            if not batch_rows:
                return
            batch_started = time.perf_counter()
            logits, probabilities = _predict(model, batch_tensors, device)
            model_batch_seconds += time.perf_counter() - batch_started
            model_batches += 1
            if len(logits) != len(batch_rows) or len(probabilities) != len(batch_rows):
                raise InferenceAuditError("classifier prediction count differs from input batch")
            for base, logit, probability in zip(batch_rows, logits, probabilities):
                if not (0.0 <= probability <= 1.0):
                    raise InferenceAuditError(f"invalid probability at key {base['key']}")
                domain = "BraTS21-like" if probability >= 0.5 else "FOMO-like"
                row = {
                    **base,
                    "logit": logit,
                    "probability_brats21_domain": probability,
                    "predicted_domain_at_0_5": domain,
                }
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                source = base["source_dataset"]
                gate = "pass" if base["support_gate_pass"] else "fail"
                all_probabilities.append(probability)
                source_probabilities[source].append(probability)
                support_probabilities[gate].append(probability)
                domain_counts[domain] += 1
                support_domain_counts[f"{gate}_{domain}"] += 1
                prediction_count += 1

        env = lmdb.open(str(split_dir), readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
        try:
            with env.begin(write=False) as txn, predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
                if txn.stat()["entries"] != EXPECTED_COUNT:
                    raise InferenceAuditError("train LMDB entry count differs from build report")
                batch_tensors: list[torch.Tensor] = []
                batch_rows: list[dict[str, Any]] = []
                scanned = 0
                for index, (key_bytes, raw) in enumerate(txn.cursor()):
                    if index >= EXPECTED_COUNT:
                        raise InferenceAuditError("train LMDB has extra rows")
                    key = f"{index:08d}"
                    if key_bytes != key.encode("ascii"):
                        raise InferenceAuditError(f"LMDB key order mismatch at {key}: {key_bytes!r}")
                    value = _read_array(raw, key)
                    support = _support_fraction(value)
                    gate_pass = all(fraction >= SUPPORT_THRESHOLD for fraction in support)
                    source = source_rows[index]
                    value_digest = hashlib.sha256(raw).digest()
                    records_hash.update(key_bytes)
                    records_hash.update(len(raw).to_bytes(8, "little"))
                    records_hash.update(value_digest)
                    source_counts[source["source_dataset"]] += 1
                    support_counts["pass" if gate_pass else "fail"] += 1
                    batch_tensors.append(torch.from_numpy(np.ascontiguousarray(value)))
                    batch_rows.append({
                        "key": key,
                        "split": "train",
                        "source_dataset": source["source_dataset"],
                        "source_key": source["source_key"],
                        "source_split": source["source_split"],
                        "support_fraction": dict(zip(MODEL_CHANNELS, support)),
                        "support_gate_pass": gate_pass,
                        "lmdb_value_sha256": value_digest.hex(),
                        "tensor_shape": list(MODEL_SHAPE),
                        "tensor_dtype": "float32",
                    })
                    scanned += 1
                    if len(batch_rows) == args.batch_size:
                        write_batch(handle, batch_tensors, batch_rows)
                        batch_tensors.clear()
                        batch_rows.clear()
                    if scanned % 8192 == 0:
                        print(f"INFER {scanned}/{EXPECTED_COUNT} elapsed={time.perf_counter() - started:.1f}s", flush=True)
                        _write_json(stage_dir / "run_status.json", {
                            "status": "RUNNING", "scanned": scanned, "predicted": prediction_count,
                        })
                write_batch(handle, batch_tensors, batch_rows)
                if scanned != EXPECTED_COUNT or prediction_count != EXPECTED_COUNT:
                    raise InferenceAuditError(f"expected {EXPECTED_COUNT} predictions; scanned={scanned}, predicted={prediction_count}")
        finally:
            env.close()
        if file_fingerprint(split_dir / "data.mdb", content=False) != input_fingerprints["data_mdb_stat"]:
            raise InferenceAuditError("train LMDB file changed during inference")
        prediction_fingerprint = file_fingerprint(predictions_path)
        prediction_fingerprint["path"] = str(output_dir / predictions_path.name)
        summary = {
            "status": "PASS",
            "completed_at_utc": _now(),
            "dataset_root": str(dataset_root),
            "dataset_kind": dataset_kind,
            "split": "train",
            "checkpoint": str(checkpoint),
            "total_elapsed_seconds": time.perf_counter() - started,
            "model_batch_inference_seconds": model_batch_seconds,
            "model_batch_inference_definition": "sum of batched input transfer, forward, sigmoid and CPU result extraction; excludes LMDB read, validation and JSONL write",
            "model_batches": model_batches,
            "batch_size": args.batch_size,
            "prediction_count": prediction_count,
            "source_counts": dict(source_counts),
            "support_gate_counts": dict(support_counts),
            "predicted_domain_counts_at_0_5": dict(domain_counts),
            "support_by_predicted_domain_counts": dict(support_domain_counts),
            "probability_brats21_domain_stats": _summary_stats(all_probabilities),
            "source_probability_stats": {
                name: _summary_stats(values) for name, values in source_probabilities.items()
            },
            "support_probability_stats": {
                name: _summary_stats(values) for name, values in support_probabilities.items()
            },
            "label_status": "NO_COMPARABLE_DOMAIN_LABELS_FOR_MIXED_SLICES",
            "auc_status": "NOT_COMPUTED",
            "slice_predictions_sha256": prediction_fingerprint["sha256"],
        }
        provenance = {
            "schema_version": 1,
            "status": "PASS",
            "purpose": f"inference_only_fomo_vs_brats21_domain_classifier_on_all_mixed_{dataset_kind}_train_slices",
            "input": {"dataset_root": str(dataset_root), "split": "train", "fingerprints": input_fingerprints},
            "train_lmdb_records_sha256": records_hash.hexdigest(),
            "train_lmdb_records_hash_scheme": "sha256(concat(key_ascii_8, raw_value_length_uint64_le, sha256(raw_lmdb_value))) in LMDB key order",
            "checkpoint": {**checkpoint_fingerprint, **model_contract},
            "model_input": {
                "channel_order": list(MODEL_CHANNELS),
                "shape": list(MODEL_SHAPE),
                "dtype": "float32",
                "normalization": normalization,
                "additional_transform": None,
            },
            "selection": "all train LMDB rows, including low-support rows",
            "support_gate": "each channel mean(abs(x + 1) > 1e-6) >= 0.10; recorded but not used to filter",
            "runtime": {**_runtime_versions(), "device": str(device), "batch_size": args.batch_size},
            "code_fingerprints": [
                file_fingerprint(Path(__file__)),
                file_fingerprint(REPO_ROOT / "scripts/run_mixed_histmatch_domain_classifier_random200.py"),
                file_fingerprint(REPO_ROOT / "scripts/run_healthy_domain_classifier_inference.py"),
                file_fingerprint(REPO_ROOT / "domain_classifier/models.py"),
            ],
            "outputs": {"slice_predictions": prediction_fingerprint},
            "summary": summary,
        }
        audit = {
            "status": "PASS",
            "checks": {
                "build_report_and_dataset_contract": True,
                "normalization_contract": True,
                "source_manifest_and_lmdb_84341_contiguous_keys": True,
                "all_tensors_float32_3x128x128_and_finite": True,
                "all_rows_inferred_including_low_support": True,
                "checkpoint_state_dict_strict_load": True,
                "no_additional_input_transform": True,
                "all_logits_finite_and_probabilities_in_unit_interval": True,
                "lmdb_file_stat_unchanged": True,
            },
            "prediction_rows": prediction_count,
            "prediction_fingerprint": prediction_fingerprint,
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        summary = _run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
