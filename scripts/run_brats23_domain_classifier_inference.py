"""Run the observed FOMO-vs-BraTS21 domain classifier on BraTS23 slices.

This is an inference-only adapter.  It deliberately reuses the repository's
``MRIDataVolume`` implementation for the complete-volume robust-IQR
normalization, model-grid resize, and nearest-exact segmentation resize.  The
BraTS23 names are mapped explicitly to the classifier's channels::

    t2f -> FLAIR, t1n -> T1, t2w -> T2

The contrast-enhanced ``t1c`` image is not loaded.  A slice is selected only
when its native segmentation and model-grid segmentation are both empty and
all three final model-space channels have at least 10% non-background support,
matching the frozen v3 model-grid proxy.  No domain AUC is computed because
BraTS23 inference has no healthy-domain labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Match the single-threaded runtime used by the audited domain-classifier
# runners.  This is set before importing torch.
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

from andi_rewrite.data.datasets.brats import MRIDataVolume  # noqa: E402
from andi_rewrite.data.robust_normalization import ROBUST_SPEC  # noqa: E402
from andi_rewrite.domain_classifier.models import build_model  # noqa: E402


SCHEMA_VERSION = 1
MODEL_SHAPE = (3, 128, 128)
MODEL_CHANNELS = ("flair", "t1", "t2")
BRATS23_MODALITIES = ("t2f", "t1n", "t2w")
MODALITY_MAPPING = {"t2f": "flair", "t1n": "t1", "t2w": "t2"}
SUPPORT_TOLERANCE = 1.0e-6
DEFAULT_DATASET_ROOT = Path(r"C:\ML\data\BraTS23")
DEFAULT_SPLIT_CSV = REPO_ROOT / "splits" / "BraTS23" / "scans_test_250.csv"
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
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "diagnostics"
    / "domain_classifier"
    / "brats23_inference_fomo_observed_20260918"
)


class InferenceAuditError(RuntimeError):
    """Raised when exact preprocessing or source validation cannot be proved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _read_subject_ids(split_csv: Path) -> list[str]:
    if not split_csv.is_file():
        raise InferenceAuditError(f"split CSV does not exist: {split_csv}")
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise InferenceAuditError(f"split CSV is empty: {split_csv}")
    fields = set(rows[0])
    column = "subject_id" if "subject_id" in fields else next(iter(fields), "")
    if column != "subject_id":
        raise InferenceAuditError(
            f"BraTS23 split must expose subject_id; found columns={sorted(fields)}"
        )
    subjects = [str(row[column]).strip() for row in rows]
    if any(not value for value in subjects):
        raise InferenceAuditError("BraTS23 split contains an empty subject_id")
    if len(set(subjects)) != len(subjects):
        duplicates = [key for key, count in Counter(subjects).items() if count > 1]
        raise InferenceAuditError(f"BraTS23 split contains duplicate subjects: {duplicates[:5]}")
    return subjects


def _subject_paths(dataset_root: Path, subject: str) -> dict[str, Path]:
    subject_dir = dataset_root / subject
    return {
        **{
            modality: subject_dir / f"{subject}-{modality}.nii.gz"
            for modality in BRATS23_MODALITIES
        },
        "seg": subject_dir / f"{subject}-seg.nii.gz",
    }


def _source_fingerprints(dataset_root: Path, subjects: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subject in subjects:
        paths = _subject_paths(dataset_root, subject)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise InferenceAuditError(
                f"BraTS23 subject {subject} is missing required files: {missing}"
            )
        records.append(
            {
                "subject_id": subject,
                "files": {name: file_fingerprint(path, content=True) for name, path in paths.items()},
            }
        )
    return records


def _load_native_segmentation(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment contract
        raise InferenceAuditError("BraTS23 inference requires nibabel") from exc
    image = nib.load(str(path))
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim != 3 or not bool(np.isfinite(values).all()):
        raise InferenceAuditError(f"invalid native segmentation: {path}")
    return values


def _validate_dataset_contract(dataset: MRIDataVolume, subjects: Sequence[str]) -> None:
    actual_subjects = [str(value) for value in dataset.df.iloc[:, 0].tolist()]
    if actual_subjects != list(subjects):
        raise InferenceAuditError("MRIDataVolume subject ordering differs from frozen CSV ordering")
    if tuple(dataset.modalities) != BRATS23_MODALITIES:
        raise InferenceAuditError(f"unexpected loader modalities: {dataset.modalities}")
    if str(dataset.filename_separator) != "-":
        raise InferenceAuditError("BraTS23 loader did not use '-' filename separator")
    if int(dataset.image_size) != 128:
        raise InferenceAuditError("BraTS23 loader image_size is not 128")
    if str(dataset.intensity_normalization) != "robust_iqr":
        raise InferenceAuditError("BraTS23 loader is not using robust_iqr")


def _load_observed_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
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
    except Exception as exc:  # pragma: no cover - fail-closed checkpoint gate
        raise InferenceAuditError(f"observed checkpoint does not match SmallCNN contract: {exc}") from exc
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
        import nibabel

        versions["nibabel"] = str(nibabel.__version__)
    except Exception:
        versions["nibabel"] = None
    try:
        import torchvision

        versions["torchvision"] = str(torchvision.__version__)
    except Exception:
        versions["torchvision"] = None
    try:
        import numpy

        versions["numpy"] = str(numpy.__version__)
    except Exception:
        versions["numpy"] = None
    return versions


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = output_dir / "run_status.json"
    _write_json(status_path, {"status": "RUNNING", "started_at_utc": _now()})

    dataset_root = args.dataset_root.resolve()
    split_csv = args.split_csv.resolve()
    checkpoint = args.checkpoint.resolve()
    if not dataset_root.is_dir():
        raise InferenceAuditError(f"BraTS23 dataset root does not exist: {dataset_root}")
    subjects = _read_subject_ids(split_csv)
    if args.expected_subjects is not None and len(subjects) != int(args.expected_subjects):
        raise InferenceAuditError(
            f"split subject count {len(subjects)} != expected {args.expected_subjects}"
        )
    if not checkpoint.is_file():
        raise InferenceAuditError(f"observed checkpoint does not exist: {checkpoint}")

    source_fingerprints = _source_fingerprints(dataset_root, subjects)
    checkpoint_fingerprint = file_fingerprint(checkpoint, content=True)
    split_fingerprint = file_fingerprint(split_csv, content=True)
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "data" / "datasets" / "brats.py",
        REPO_ROOT / "data" / "robust_normalization.py",
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
    ]
    code_fingerprints = [file_fingerprint(path, content=True) for path in code_paths]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise InferenceAuditError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    model, model_contract = _load_observed_model(checkpoint, device)

    dataset = MRIDataVolume(
        csv_path=split_csv,
        dataset_path=dataset_root,
        image_size=128,
        modalities=list(BRATS23_MODALITIES),
        segmentation_suffix="seg",
        filename_separator="-",
        return_metadata=True,
        intensity_normalization="robust_iqr",
    )
    _validate_dataset_contract(dataset, subjects)

    expected_spec = {
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
    }
    if dict(ROBUST_SPEC) != expected_spec:
        raise InferenceAuditError(f"ROBUST_SPEC differs from frozen v3 contract: {ROBUST_SPEC}")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": _now(),
        "purpose": "inference_only_observed_fomo_vs_brats21_smallcnn_on_brats23_tumor_free_slices",
        "source": {
            "dataset": "BraTS23",
            "dataset_root": str(dataset_root),
            "split_csv": split_fingerprint,
            "subject_count": len(subjects),
            "subject_order_sha256": hashlib.sha256(
                "\n".join(subjects).encode("utf-8")
            ).hexdigest(),
            "subject_ids": subjects,
            "file_fingerprints": source_fingerprints,
        },
        "checkpoint": {
            **checkpoint_fingerprint,
            "path": str(checkpoint),
            "result_json": str(checkpoint.parent / "result.json"),
            "result_json_sha256": (
                sha256_file(checkpoint.parent / "result.json")
                if (checkpoint.parent / "result.json").is_file()
                else None
            ),
            **model_contract,
        },
        "channel_mapping": {
            "source_modalities_in_loader_order": list(BRATS23_MODALITIES),
            "classifier_channels_in_order": list(MODEL_CHANNELS),
            "mapping": MODALITY_MAPPING,
            "excluded_source_modality": "t1c",
        },
        "preprocessing": {
            "reader": "andi_rewrite.data.datasets.brats.MRIDataVolume",
            "normalization": dict(ROBUST_SPEC),
            "image_size": 128,
            "model_input_shape": list(MODEL_SHAPE),
            "resize": "torchvision.transforms.Resize(size=128, antialias=True) per axial slice",
            "segmentation_model_grid_resize": "torch.nn.functional.interpolate(mode='nearest-exact', size=(128,128,depth))",
            "additional_classifier_transform": None,
        },
        "selection": {
            "native_segmentation_voxels": 0,
            "model_grid_segmentation_voxels": 0,
            "support_proxy": "abs(model_input + 1) > 1e-6",
            "support_threshold_each_channel": float(args.support_threshold),
            "support_denominator_pixels": 128 * 128,
            "all_three_channels_required": True,
        },
        "runtime": {
            **_runtime_versions(),
            "device": str(device),
            "batch_size": int(args.batch_size),
        },
        "code_fingerprints": code_fingerprints,
        "outputs": {
            "slice_predictions": str((output_dir / "slice_predictions.jsonl").resolve()),
            "subject_predictions": str((output_dir / "subject_predictions.jsonl").resolve()),
            "selection_manifest": str((output_dir / "selection_manifest.jsonl").resolve()),
        },
    }
    _write_json(output_dir / "provenance.json", provenance)

    counts: Counter[str] = Counter()
    subject_summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_selected = 0
    prediction_count = 0

    selection_path = output_dir / "selection_manifest.jsonl"
    prediction_path = output_dir / "slice_predictions.jsonl"
    subject_path = output_dir / "subject_predictions.jsonl"
    with (
        selection_path.open("w", encoding="utf-8", newline="\n") as selection_handle,
        prediction_path.open("w", encoding="utf-8", newline="\n") as prediction_handle,
        subject_path.open("w", encoding="utf-8", newline="\n") as subject_handle,
    ):
        for dataset_index, subject in enumerate(subjects):
            paths = _subject_paths(dataset_root, subject)
            native_seg = _load_native_segmentation(paths["seg"])
            volume, model_mask, metadata = dataset[dataset_index]
            if not isinstance(volume, torch.Tensor) or not isinstance(model_mask, torch.Tensor):
                raise InferenceAuditError(f"dataset returned non-tensor data for {subject}")
            if tuple(volume.shape[:3]) != MODEL_SHAPE or volume.ndim != 4:
                raise InferenceAuditError(f"unexpected model volume shape for {subject}: {tuple(volume.shape)}")
            if tuple(model_mask.shape) != (128, 128, volume.shape[-1]):
                raise InferenceAuditError(f"unexpected model mask shape for {subject}: {tuple(model_mask.shape)}")
            native_shape_text = str(metadata["native_shape"]).replace("x", ",")
            if tuple(native_seg.shape) != tuple(int(value) for value in native_shape_text.split(",")):
                raise InferenceAuditError(f"native segmentation shape mismatch for {subject}")
            if native_seg.shape[-1] != volume.shape[-1]:
                raise InferenceAuditError(f"native/model depth mismatch for {subject}")
            if volume.dtype != torch.float32 or not bool(torch.isfinite(volume).all()):
                raise InferenceAuditError(f"non-finite or non-float32 model input for {subject}")
            if not bool(torch.isfinite(model_mask).all()):
                raise InferenceAuditError(f"non-finite model mask for {subject}")

            native_voxels_by_z = np.count_nonzero(native_seg != 0, axis=(0, 1)).astype(np.int64)
            model_mask_np = model_mask.detach().cpu().numpy().astype(bool, copy=False)
            model_voxels_by_z = np.count_nonzero(model_mask_np, axis=(0, 1)).astype(np.int64)
            support_np = (torch.abs(volume + 1.0) > SUPPORT_TOLERANCE).detach().cpu().numpy()
            # volume is [C,H,W,Z]; preserve the explicit channel/z axes.
            support_fraction_by_channel_z = support_np.mean(axis=(1, 2))
            native_zero = native_voxels_by_z == 0
            model_zero = model_voxels_by_z == 0
            support_pass = np.all(support_fraction_by_channel_z >= float(args.support_threshold), axis=0)
            selected_z = np.flatnonzero(native_zero & model_zero & support_pass).astype(int).tolist()
            total_rows += int(volume.shape[-1])
            counts["native_zero_slices"] += int(native_zero.sum())
            counts["model_zero_slices"] += int(model_zero.sum())
            counts["both_mask_zero_slices"] += int(np.logical_and(native_zero, model_zero).sum())
            counts["support_pass_slices"] += int(support_pass.sum())
            counts["selected_slices"] += len(selected_z)
            if not selected_z:
                counts["subjects_without_selected_slices"] += 1

            selected_tensor = volume[..., selected_z].permute(3, 0, 1, 2).contiguous() if selected_z else None
            probabilities: list[float] = []
            logits: list[float] = []
            if selected_tensor is not None:
                if tuple(selected_tensor.shape[1:]) != MODEL_SHAPE:
                    raise InferenceAuditError(f"selected tensor shape mismatch for {subject}")
                for start in range(0, int(selected_tensor.shape[0]), int(args.batch_size)):
                    batch = selected_tensor[start : start + int(args.batch_size)].to(device)
                    with torch.inference_mode():
                        batch_logits = model(batch).reshape(-1)
                        batch_probabilities = torch.sigmoid(batch_logits)
                    if not bool(torch.isfinite(batch_logits).all()) or not bool(torch.isfinite(batch_probabilities).all()):
                        raise InferenceAuditError(f"non-finite model prediction for {subject}")
                    logits.extend(float(value) for value in batch_logits.detach().cpu().tolist())
                    probabilities.extend(float(value) for value in batch_probabilities.detach().cpu().tolist())
                if len(probabilities) != len(selected_z):
                    raise InferenceAuditError(f"prediction count mismatch for {subject}")

            subject_probability_array = np.asarray(probabilities, dtype=np.float64)
            subject_summary = {
                "subject_index": dataset_index,
                "subject_id": subject,
                "source_dataset": "BraTS23",
                "native_shape": list(native_seg.shape),
                "model_shape": [int(value) for value in volume.shape[1:]],
                "total_z_slices": int(volume.shape[-1]),
                "native_zero_slice_count": int(native_zero.sum()),
                "model_zero_slice_count": int(model_zero.sum()),
                "both_mask_zero_slice_count": int(np.logical_and(native_zero, model_zero).sum()),
                "support_pass_slice_count": int(support_pass.sum()),
                "selected_slice_count": int(len(selected_z)),
                "selected_z": selected_z,
                "aggregation": "mean_selected_slice_probability",
                "mean_probability": float(subject_probability_array.mean()) if probabilities else None,
                "median_probability": float(np.median(subject_probability_array)) if probabilities else None,
                "min_probability": float(subject_probability_array.min()) if probabilities else None,
                "max_probability": float(subject_probability_array.max()) if probabilities else None,
                "std_probability": float(subject_probability_array.std()) if probabilities else None,
                "predicted_domain": (
                    "BraTS21-like" if probabilities and float(subject_probability_array.mean()) >= 0.5
                    else "FOMO-like" if probabilities else None
                ),
                "label_available": False,
                "auc_available": False,
                "metadata": metadata,
            }
            _append_jsonl(subject_handle, subject_summary)
            subject_summaries.append(subject_summary)

            for local_index, (z, probability, logit) in enumerate(zip(selected_z, probabilities, logits)):
                slice_tensor = volume[..., int(z)].contiguous()
                digest = tensor_sha256(slice_tensor)
                support = [float(value) for value in support_fraction_by_channel_z[:, int(z)]]
                row = {
                    "prediction_index": prediction_count,
                    "subject_index": dataset_index,
                    "subject_id": subject,
                    "source_dataset": "BraTS23",
                    "z": int(z),
                    "z_norm_native": float(z / max(1, volume.shape[-1] - 1)),
                    "native_seg_voxels": int(native_voxels_by_z[int(z)]),
                    "model_grid_seg_voxels": int(model_voxels_by_z[int(z)]),
                    "support_fraction": dict(zip(MODEL_CHANNELS, support)),
                    "tensor_shape": list(slice_tensor.shape),
                    "tensor_dtype": str(slice_tensor.dtype).replace("torch.", ""),
                    "tensor_sha256": digest,
                    "logit": float(logit),
                    "probability_brats21_domain": float(probability),
                    "predicted_domain_at_0_5": "BraTS21-like" if probability >= 0.5 else "FOMO-like",
                    "aggregation_subject_row_index": dataset_index,
                }
                _append_jsonl(selection_handle, {key: value for key, value in row.items() if key not in {"logit", "probability_brats21_domain", "predicted_domain_at_0_5"}})
                _append_jsonl(prediction_handle, row)
                prediction_count += 1
            total_selected += len(selected_z)

    # Re-read the emitted JSONL byte streams so the final audit binds both the
    # row counts and their exact bytes to this completed run.
    output_fingerprints = {
        name: file_fingerprint(path, content=True)
        for name, path in {
            "selection_manifest": selection_path,
            "slice_predictions": prediction_path,
            "subject_predictions": subject_path,
        }.items()
    }
    if total_selected != prediction_count:
        raise InferenceAuditError("selected slice count differs from prediction count")
    completed_at = _now()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": completed_at,
        "elapsed_seconds": float(time.time() - started),
        "dataset": "BraTS23",
        "checkpoint": str(checkpoint),
        "subjects_requested": len(subjects),
        "subjects_processed": len(subject_summaries),
        "subjects_with_selected_slices": int(sum(row["selected_slice_count"] > 0 for row in subject_summaries)),
        "total_z_slices_scanned": int(total_rows),
        "selected_slice_count": int(total_selected),
        "prediction_count": int(prediction_count),
        "counts": dict(counts),
        "label_status": "BraTS23 has no healthy-domain ground-truth label in this inference task",
        "auc_status": "NOT_COMPUTED_NO_COMPARABLE_LABELS",
        "output_fingerprints": output_fingerprints,
        "checks": {
            "source_files_present_and_sha256_bound": True,
            "split_unique_and_order_bound": True,
            "loader_contract": "PASS",
            "robust_iqr_contract": "PASS",
            "native_and_model_grid_mask_gates": "PASS",
            "support_gate": "PASS",
            "all_tensor_hashes_finite_float32_shape_3x128x128": True,
            "checkpoint_state_dict_strict_load": "PASS",
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
            "subject_rows": len(subject_summaries),
            "slice_rows": prediction_count,
            "output_fingerprints": output_fingerprints,
        },
    )
    _write_json(status_path, {"status": "PASS", "completed_at_utc": completed_at, "summary": summary})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-threshold", type=float, default=0.1)
    parser.add_argument("--expected-subjects", type=int, default=250)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0.0 < args.support_threshold <= 1.0:
        parser.error("--support-threshold must lie in (0,1]")
    try:
        summary = _run(args)
    except Exception as exc:
        output_dir = args.output_dir.resolve()
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
