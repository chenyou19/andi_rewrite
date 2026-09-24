"""Run the observed FOMO-vs-BraTS21 classifier on healthy LMDB cohorts.

This is an inference-only adapter for the completed MPI and OASIS3 healthy
preprocessing outputs.  It reads the canonical train/val LMDB tensors without
re-normalising, resizing, or re-running skull stripping.  The frozen v3
model-grid support proxy is recorded for every tensor; only rows passing the
same 10-percent-per-channel support gate are sent to the classifier.

The checkpoint's sigmoid output is the probability of the BraTS21 domain
(label 1 in the observed FOMO-vs-BraTS21 experiment).  MPI/OASIS3 do not carry
a comparable domain label here, so this script reports distributions and
participant-level predictions rather than an AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Match the audited domain-classifier runtime before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from andi_rewrite.data.datasets.lmdb import LMDBSliceDataset  # noqa: E402
from andi_rewrite.data.robust_normalization import ROBUST_SPEC  # noqa: E402
from andi_rewrite.domain_classifier.models import build_model  # noqa: E402


SCHEMA_VERSION = 1
MODEL_SHAPE = (3, 128, 128)
MODEL_CHANNELS = ("flair", "t1", "t2")
SUPPORT_TOLERANCE = 1.0e-6
SUPPORT_THRESHOLD = 0.1
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "diagnostics"
    / "domain_classifier"
    / "model_grid_v3_fullcandidate_20260917_final"
    / "stage_a_fomo45k_observed_v3_20260917"
    / "observed"
    / "model_best.pt"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier"
)
DATASET_CONFIG = {
    "mpi": {
        "display_name": "MPI",
        "dataset_root": REPO_ROOT / "outputs" / "datasets" / "mpi_sri24_robust_iqr",
    },
    "oasis3": {
        "display_name": "OASIS3",
        "dataset_root": REPO_ROOT / "outputs" / "datasets" / "oasis3_sri24_robust_iqr",
    },
}


class InferenceAuditError(RuntimeError):
    """Raised when an exact input or output contract cannot be proved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _append_jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().to(dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def file_fingerprint(path: Path, *, content: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise InferenceAuditError(f"missing source file: {resolved}")
    stat = resolved.stat()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content:
        payload["sha256"] = sha256_file(resolved)
    return payload


def _load_entries(entries_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not entries_path.is_file():
        raise InferenceAuditError(f"missing entries manifest: {entries_path}")
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    with entries_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InferenceAuditError(
                    f"invalid entries JSON at {entries_path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise InferenceAuditError(f"entries row is not an object at line {line_number}")
            split = str(row.get("split", ""))
            if split not in grouped:
                raise InferenceAuditError(f"unexpected healthy split {split!r}")
            required = ("key", "case_id", "participant_id", "z")
            missing = [key for key in required if key not in row]
            if missing:
                raise InferenceAuditError(f"entries row missing {missing}: {row}")
            grouped[split].append({str(key): row[key] for key in row})
    for split, rows in grouped.items():
        for expected, row in enumerate(rows):
            try:
                actual = int(str(row["key"]))
            except (TypeError, ValueError) as exc:
                raise InferenceAuditError(f"non-integer {split} key: {row['key']!r}") from exc
            if actual != expected:
                raise InferenceAuditError(
                    f"{split} entries are not contiguous from zero: expected {expected}, found {actual}"
                )
            if not str(row["participant_id"]).strip() or not str(row["case_id"]).strip():
                raise InferenceAuditError(f"empty participant/case identity in {split}: {row}")
    return grouped


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": str(torch.version.cuda),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }
    try:
        import numpy

        versions["numpy"] = str(numpy.__version__)
    except Exception:
        versions["numpy"] = None
    try:
        import lmdb

        versions["lmdb"] = str(getattr(lmdb, "__version__", "unknown"))
    except Exception:
        versions["lmdb"] = None
    try:
        import torchvision

        versions["torchvision"] = str(torchvision.__version__)
    except Exception:
        versions["torchvision"] = None
    return versions


def _load_observed_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, Mapping):
        raise InferenceAuditError("observed checkpoint is not a state_dict mapping")
    model = build_model(
        "small_cnn",
        in_channels=3,
        num_classes=1,
        widths=(32, 64, 128, 128),
        groupnorm_groups=8,
        dropout=0.0,
    )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise InferenceAuditError(
            f"observed checkpoint does not match SmallCNN contract: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    return model, {
        "architecture": {
            "model": "small_cnn",
            "in_channels": 3,
            "num_classes": 1,
            "widths": [32, 64, 128, 128],
            "groupnorm_groups": 8,
            "dropout": 0.0,
        },
        "state_dict_keys": sorted(str(key) for key in state),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _validate_robust_contract(normalization_path: Path) -> None:
    if not normalization_path.is_file():
        raise InferenceAuditError(f"missing LMDB normalization contract: {normalization_path}")
    try:
        actual = json.loads(normalization_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceAuditError(f"invalid normalization JSON: {normalization_path}") from exc
    if dict(actual) != dict(ROBUST_SPEC):
        raise InferenceAuditError(
            f"LMDB normalization contract mismatch at {normalization_path}: {actual}"
        )


def _predict(model: torch.nn.Module, tensors: list[torch.Tensor], device: torch.device) -> tuple[list[float], list[float]]:
    if not tensors:
        return [], []
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        logits_tensor = model(batch).reshape(-1)
        probabilities_tensor = torch.sigmoid(logits_tensor)
    if not bool(torch.isfinite(logits_tensor).all()) or not bool(torch.isfinite(probabilities_tensor).all()):
        raise InferenceAuditError("model produced non-finite logits or probabilities")
    logits = [float(value) for value in logits_tensor.detach().cpu().tolist()]
    probabilities = [float(value) for value in probabilities_tensor.detach().cpu().tolist()]
    return logits, probabilities


def _dataset_source_fingerprints(dataset_root: Path, split_names: Sequence[str]) -> dict[str, Any]:
    compact_names = (
        "entries.jsonl",
        "split.json",
        "configuration.json",
        "build_report.json",
        "atlas_provenance.json",
        "tool_versions_and_parameters.json",
    )
    records: dict[str, Any] = {}
    for name in compact_names:
        path = dataset_root / name
        if path.is_file():
            records[name] = file_fingerprint(path, content=True)
    records["lmdb_splits"] = {}
    for split in split_names:
        split_dir = dataset_root / split
        records["lmdb_splits"][split] = {
            "directory": str(split_dir.resolve()),
            # data.mdb is a multi-gigabyte sparse map.  Every tensor consumed
            # below is hashed row-by-row; retain the map identity without
            # forcing a second full-file hash.
            "data_mdb": file_fingerprint(split_dir / "data.mdb", content=False),
            "lock_mdb": file_fingerprint(split_dir / "lock.mdb", content=False),
            "normalization": file_fingerprint(split_dir / "normalization.json", content=True),
        }
    return records


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_key = str(args.dataset).lower()
    if dataset_key not in DATASET_CONFIG:
        raise InferenceAuditError(f"unsupported dataset {args.dataset!r}")
    config = DATASET_CONFIG[dataset_key]
    display_name = str(config["display_name"])
    dataset_root = Path(config["dataset_root"]).resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.is_dir():
        raise InferenceAuditError(f"healthy dataset root does not exist: {dataset_root}")
    if not checkpoint.is_file():
        raise InferenceAuditError(f"observed checkpoint does not exist: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InferenceAuditError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = output_dir / "run_status.json"
    _write_json(status_path, {"status": "RUNNING", "started_at_utc": _now()})

    entries_path = dataset_root / "entries.jsonl"
    grouped = _load_entries(entries_path)
    split_names = ("train", "val")
    for split in split_names:
        _validate_robust_contract(dataset_root / split / "normalization.json")
    source_fingerprints = _dataset_source_fingerprints(dataset_root, split_names)
    checkpoint_fingerprint = file_fingerprint(checkpoint, content=True)
    result_json = checkpoint.parent / "result.json"
    if result_json.is_file():
        checkpoint_fingerprint["result_json"] = file_fingerprint(result_json, content=True)
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "data" / "datasets" / "lmdb.py",
        REPO_ROOT / "data" / "robust_normalization.py",
        REPO_ROOT / "domain_classifier" / "models.py",
    ]
    code_fingerprints = [file_fingerprint(path, content=True) for path in code_paths]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise InferenceAuditError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    model, model_contract = _load_observed_model(checkpoint, device)
    if dict(ROBUST_SPEC) != {
        "type": "robust_iqr",
        "version": 1,
        "scope": "per_volume_per_modality",
        "foreground": "input > 0",
        "center": "median",
        "scale": "q75-q25",
        "background": -1.0,
        "clip": False,
        "eps": 1e-8,
        "model_normalize_input": False,
    }:
        raise InferenceAuditError(f"ROBUST_SPEC differs from frozen v3 contract: {ROBUST_SPEC}")

    selection_path = output_dir / "selection_manifest.jsonl"
    prediction_path = output_dir / "slice_predictions.jsonl"
    subject_path = output_dir / "subject_predictions.jsonl"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": _now(),
        "purpose": "inference_only_observed_fomo_vs_brats21_smallcnn_on_preprocessed_healthy_cohort",
        "source": {
            "dataset": display_name,
            "dataset_key": dataset_key,
            "dataset_root": str(dataset_root),
            "entries": file_fingerprint(entries_path, content=True),
            "splits_included": list(split_names),
            "source_fingerprints": source_fingerprints,
            "entry_counts": {split: len(grouped[split]) for split in split_names},
            "participant_counts": {
                split: len({str(row["participant_id"]) for row in grouped[split]})
                for split in split_names
            },
        },
        "checkpoint": {
            **checkpoint_fingerprint,
            "path": str(checkpoint),
            **model_contract,
        },
        "preprocessing": {
            "reader": "andi_rewrite.data.datasets.lmdb.LMDBSliceDataset",
            "lmdb_image_size_argument": None,
            "normalization": dict(ROBUST_SPEC),
            "model_input_shape": list(MODEL_SHAPE),
            "channel_order": list(MODEL_CHANNELS),
            "additional_classifier_transform": None,
            "source_tensors_already_model_space": True,
        },
        "selection": {
            "support_proxy": "abs(model_input + 1) > 1e-6",
            "support_threshold_each_channel": SUPPORT_THRESHOLD,
            "support_denominator_pixels": 128 * 128,
            "all_three_channels_required": True,
            "selected_rows_are_preprocessed_lmdb_entries": True,
        },
        "runtime": {
            **_runtime_versions(),
            "device": str(device),
            "batch_size": int(args.batch_size),
        },
        "code_fingerprints": code_fingerprints,
        "outputs": {
            "selection_manifest": str(selection_path.resolve()),
            "slice_predictions": str(prediction_path.resolve()),
            "subject_predictions": str(subject_path.resolve()),
        },
    }
    _write_json(output_dir / "provenance.json", provenance)

    counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in split_names}
    participant_state: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    prediction_count = 0
    scanned_count = 0
    selected_count = 0

    with (
        selection_path.open("w", encoding="utf-8", newline="\n") as selection_handle,
        prediction_path.open("w", encoding="utf-8", newline="\n") as prediction_handle,
    ):
        for split in split_names:
            rows = grouped[split]
            lmdb_path = dataset_root / split
            dataset = LMDBSliceDataset(lmdb_path, image_size=None)
            try:
                if len(dataset) != len(rows):
                    raise InferenceAuditError(
                        f"{display_name} {split} LMDB length {len(dataset)} != entries {len(rows)}"
                    )
                for start in range(0, len(rows), int(args.batch_size)):
                    batch_rows = rows[start : start + int(args.batch_size)]
                    tensors: list[torch.Tensor] = []
                    selected_flags: list[bool] = []
                    support_values: list[list[float]] = []
                    tensor_digests: list[str] = []
                    for row in batch_rows:
                        key = int(str(row["key"]))
                        tensor = dataset[key]
                        if tensor.dtype != torch.float32 or tuple(tensor.shape) != MODEL_SHAPE:
                            raise InferenceAuditError(
                                f"invalid {display_name} {split} tensor key={key}: "
                                f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
                            )
                        if not bool(torch.isfinite(tensor).all()):
                            raise InferenceAuditError(f"non-finite tensor at {display_name}/{split}/{key}")
                        support_fraction = (
                            (torch.abs(tensor + 1.0) > SUPPORT_TOLERANCE)
                            .to(dtype=torch.float32)
                            .mean(dim=(1, 2))
                        )
                        support = [float(value) for value in support_fraction.tolist()]
                        selected = bool(all(value >= SUPPORT_THRESHOLD for value in support))
                        tensors.append(tensor.contiguous())
                        selected_flags.append(selected)
                        support_values.append(support)
                        tensor_digests.append(tensor_sha256(tensor))

                    selected_tensors = [
                        tensor for tensor, selected in zip(tensors, selected_flags) if selected
                    ]
                    logits, probabilities = _predict(model, selected_tensors, device)
                    prediction_iter = iter(zip(logits, probabilities))
                    batch_prediction_rows: list[dict[str, Any] | None] = []
                    for row, tensor, selected, support, digest in zip(
                        batch_rows, tensors, selected_flags, support_values, tensor_digests
                    ):
                        participant = str(row["participant_id"])
                        state = participant_state.setdefault(
                            participant,
                            {
                                "participant_id": participant,
                                "case_ids": [],
                                "splits": [],
                                "scanned_slice_count": 0,
                                "selected_slice_count": 0,
                                "probabilities": [],
                                "logits": [],
                                "selected_z": [],
                            },
                        )
                        case_id = str(row["case_id"])
                        if case_id not in state["case_ids"]:
                            state["case_ids"].append(case_id)
                        if split not in state["splits"]:
                            state["splits"].append(split)
                        state["scanned_slice_count"] += 1
                        scanned_count += 1
                        counts["scanned_slices"] += 1
                        split_counts[split]["scanned_slices"] += 1
                        selected_info: dict[str, Any] | None = None
                        if selected:
                            logit, probability = next(prediction_iter)
                            selected_info = {
                                "prediction_index": prediction_count,
                                "logit": float(logit),
                                "probability_brats21_domain": float(probability),
                                "predicted_domain_at_0_5": (
                                    "BraTS21-like" if probability >= 0.5 else "FOMO-like"
                                ),
                            }
                            prediction_count += 1
                            selected_count += 1
                            counts["selected_slices"] += 1
                            split_counts[split]["selected_slices"] += 1
                            state["selected_slice_count"] += 1
                            state["probabilities"].append(float(probability))
                            state["logits"].append(float(logit))
                            state["selected_z"].append(int(row["z"]))
                        else:
                            counts["support_failed_slices"] += 1
                            split_counts[split]["support_failed_slices"] += 1
                        base_row = {
                            "source_dataset": display_name,
                            "dataset_key": dataset_key,
                            "split": split,
                            "source_key": str(row["key"]),
                            "case_id": case_id,
                            "participant_id": participant,
                            "z": int(row["z"]),
                            "z_norm": float(int(row["z"]) / 154.0),
                            "support_fraction": dict(zip(MODEL_CHANNELS, support)),
                            "support_gate_pass": selected,
                            "tensor_shape": list(tensor.shape),
                            "tensor_dtype": str(tensor.dtype).replace("torch.", ""),
                            "tensor_sha256": digest,
                            "label_available": False,
                            "auc_available": False,
                        }
                        selection_row = {
                            **base_row,
                            "selected_for_inference": selected,
                        }
                        _append_jsonl(selection_handle, selection_row)
                        if selected_info is not None:
                            _append_jsonl(
                                prediction_handle,
                                {**base_row, **selected_info},
                            )
                    if any(selected_flags) and next(prediction_iter, None) is not None:
                        raise InferenceAuditError("prediction count/order mismatch in batch")
            finally:
                transaction = getattr(dataset, "txn", None)
                if transaction is not None:
                    transaction.abort()
                    dataset.txn = None
                environment = getattr(dataset, "env", None)
                if environment is not None:
                    environment.close()
                    dataset.env = None

    if selected_count != prediction_count:
        raise InferenceAuditError(
            f"selected slice count {selected_count} differs from predictions {prediction_count}"
        )

    subject_rows: list[dict[str, Any]] = []
    all_participant_means: list[float] = []
    with subject_path.open("w", encoding="utf-8", newline="\n") as subject_handle:
        for subject_index, state in enumerate(participant_state.values()):
            probabilities = np.asarray(state["probabilities"], dtype=np.float64)
            mean_probability = float(probabilities.mean()) if probabilities.size else None
            if mean_probability is not None:
                all_participant_means.append(mean_probability)
            row = {
                "subject_index": subject_index,
                "participant_id": state["participant_id"],
                "source_dataset": display_name,
                "case_ids": state["case_ids"],
                "splits": state["splits"],
                "scanned_slice_count": int(state["scanned_slice_count"]),
                "selected_slice_count": int(state["selected_slice_count"]),
                "selected_z": state["selected_z"],
                "aggregation": "mean_selected_slice_probability",
                "mean_probability": mean_probability,
                "median_probability": float(np.median(probabilities)) if probabilities.size else None,
                "min_probability": float(probabilities.min()) if probabilities.size else None,
                "max_probability": float(probabilities.max()) if probabilities.size else None,
                "std_probability": float(probabilities.std()) if probabilities.size else None,
                "predicted_domain": (
                    "BraTS21-like" if mean_probability is not None and mean_probability >= 0.5
                    else "FOMO-like" if mean_probability is not None else None
                ),
                "label_available": False,
                "auc_available": False,
            }
            _append_jsonl(subject_handle, row)
            subject_rows.append(row)

    output_paths = {
        "selection_manifest": selection_path,
        "slice_predictions": prediction_path,
        "subject_predictions": subject_path,
    }
    output_fingerprints = {
        name: file_fingerprint(path, content=True)
        for name, path in output_paths.items()
    }
    slice_probabilities: list[float] = []
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                slice_probabilities.append(float(json.loads(line)["probability_brats21_domain"]))
    completed_at = _now()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": completed_at,
        "elapsed_seconds": float(time.time() - started),
        "dataset": display_name,
        "dataset_key": dataset_key,
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "splits_included": list(split_names),
        "subjects_processed": len(subject_rows),
        "subjects_with_selected_slices": int(
            sum(int(row["selected_slice_count"]) > 0 for row in subject_rows)
        ),
        "total_z_slices_scanned": int(scanned_count),
        "selected_slice_count": int(selected_count),
        "prediction_count": int(prediction_count),
        "counts": dict(counts),
        "split_counts": {split: dict(values) for split, values in split_counts.items()},
        "slice_probability_stats": _summary_stats(slice_probabilities),
        "participant_mean_probability_stats": _summary_stats(all_participant_means),
        "participant_predicted_domain_counts": dict(
            Counter(str(row["predicted_domain"]) for row in subject_rows)
        ),
        "label_status": "MPI/OASIS3 healthy inputs have no comparable BraTS-vs-FOMO ground-truth label",
        "auc_status": "NOT_COMPUTED_NO_COMPARABLE_LABELS",
        "output_fingerprints": output_fingerprints,
        "checks": {
            "source_metadata_and_lmdb_stat_bound": True,
            "entries_order_and_lmdb_key_alignment": True,
            "healthy_lmdb_normalization_contract": "PASS",
            "support_gate": "PASS",
            "all_tensor_hashes_finite_float32_shape_3x128x128": True,
            "checkpoint_state_dict_strict_load": "PASS",
            "no_additional_classifier_transform": True,
        },
    }
    provenance["status"] = "PASS"
    provenance["completed_at_utc"] = completed_at
    provenance["summary"] = summary
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "audit.json",
        {
            "status": "PASS",
            "completed_at_utc": completed_at,
            "selection_and_prediction_summary": summary,
            "subject_rows": len(subject_rows),
            "slice_rows": prediction_count,
            "output_fingerprints": output_fingerprints,
        },
    )
    _write_json(status_path, {"status": "PASS", "completed_at_utc": completed_at, "summary": summary})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIG), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT_ROOT
            / f"{args.dataset.lower()}_inference_fomo_observed_20260918"
        )
    try:
        summary = _run(args)
    except Exception as exc:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL",
            "completed_at_utc": _now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / "failure.json", failure)
        _write_json(output_dir / "run_status.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
