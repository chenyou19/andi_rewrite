"""Read-only parity check for existing domain-classifier diagnostic references.

The checker consumes the already-written ``comparison.json`` files.  For each
selected pair it loads the production reader output and independently rebuilds
the expected tensor from the recorded registered full volumes with the current
robust-IQR function and 128-pixel resize.  It never creates caches, manifests,
LMDBs, or replacement source data.

All source failures are retained in ``parity.json``.  In particular, a missing
raw source makes a pair ``INCOMPLETE`` even when the existing LMDB tensor can
still be read; an LMDB-vs-LMDB comparison is not called raw-input parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

MODALITIES = ("FLAIR", "T1", "T2")
COMPARISON_PATHS = {
    "FOMO": REPO_ROOT / "outputs/diagnostics/robust_b/input_comparison_tumor_free_20/comparison.json",
    "MPI": REPO_ROOT / "outputs/diagnostics/robust_b/input_comparison_mpi_tumor_free_20/comparison.json",
    "OASIS3": REPO_ROOT / "outputs/diagnostics/robust_b/input_comparison_oasis3_tumor_free_20/comparison.json",
}
LMDB_PATHS = {
    "FOMO": REPO_ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/train",
    "MPI": REPO_ROOT / "outputs/datasets/mpi_sri24_robust_iqr/train",
    "OASIS3": REPO_ROOT / "outputs/datasets/oasis3_sri24_robust_iqr/train",
}
BRATS_ROOT = Path(r"C:\ML\data\BraTS_2021")


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _file_state(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        exists = target.is_file()
    except OSError as exc:
        # Keep an ACL/access failure as source evidence.  A metadata audit
        # must not turn an unavailable registered volume into an uncaught
        # checker failure or silently treat it as an available file.
        return {
            "path": str(target),
            "exists": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    state: dict[str, Any] = {"path": str(target), "exists": exists}
    if not exists:
        state["reason"] = "FileNotFoundError"
        return state
    try:
        stat = target.stat()
    except OSError as exc:
        state.update({"exists": False, "reason": f"{type(exc).__name__}: {exc}"})
        return state
    state.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return state


def _fingerprint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    state = _file_state(target)
    if not state["exists"]:
        return state
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    state["sha256"] = digest.hexdigest()
    return state


def _comparison_fingerprint(path: Path) -> dict[str, Any]:
    return _fingerprint(path)


def _load_comparison(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("pairs"), list):
        raise ValueError(f"comparison.json must contain an object with pairs: {path}")
    return value


def _healthy_metadata(cohort: str, pair: Mapping[str, Any], slice_index: int) -> dict[str, Any]:
    cohort_lower = cohort.lower()
    if cohort == "FOMO":
        participant_case = str(pair["fomo"])
        participant, _, session = participant_case.partition("/")
        paths = pair["fomo_paths"]
        key = str(pair["lmdb_indices"][slice_index])
        case_id = participant_case
    else:
        participant = str(pair["source_participant"])
        case_id = str(pair["source_case"])
        session = case_id.split("/", 1)[1] if "/" in case_id else ""
        paths = pair["source_paths"]
        key = str(pair["source_lmdb_keys"][slice_index])
    return {
        "participant_id": participant,
        "session_id": session,
        "case_id": case_id,
        "paths": dict(paths),
        "source_key": key,
        "lmdb_path": str(LMDB_PATHS[cohort]),
        "source_dataset": cohort_lower,
    }


def _brats_metadata(pair: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(pair["brats"] if "brats" in pair else pair["brats_case"])
    return {
        "participant_id": case_id,
        "session_id": "",
        "case_id": case_id,
        "paths": dict(pair["brats_paths"]),
        "segmentation": str(pair.get("segmentation", pair.get("brats_segmentation", ""))),
        "source_dataset": "brats",
    }


def _record(meta: Mapping[str, Any], *, z: int, label: int, repo_root: Path):
    from andi_rewrite.data.domain_classifier.records import SliceRecord

    z_norm = float(z) / 154.0
    source_dataset = str(meta["source_dataset"])
    source_key = str(meta.get("source_key", ""))
    metadata: dict[str, Any] = {}
    if meta.get("lmdb_path"):
        metadata.update({"lmdb_path": str(meta["lmdb_path"]), "input_kind": "lmdb"})
    else:
        metadata["dataset_path"] = str(BRATS_ROOT)
    return SliceRecord(
        split="train",
        label=label,
        domain="healthy" if label == 0 else "brats",
        participant_id=str(meta["participant_id"]),
        session_id=str(meta.get("session_id", "")),
        case_id=str(meta["case_id"]),
        z=int(z),
        z_norm=z_norm,
        z_bin=min(19, int(z_norm * 20)),
        source_dataset=source_dataset,
        source_key=source_key,
        source_split="train",
        image_paths=dict(meta["paths"]),
        seg_path=str(meta.get("segmentation", "")),
        geometry_shape=(240, 240, 155),
        model_shape=(3, 128, 128),
        metadata=metadata,
    )


def _read_registered_expected(paths: Mapping[str, str], z: int):
    import nibabel as nib
    import torch
    from torchvision.transforms import Resize

    arrays = []
    images: dict[str, Any] = {}
    for modality in MODALITIES:
        path = Path(str(paths[modality]))
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(str(path))
        array = np.asarray(image.dataobj, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"{path} shape is not 3-D: {array.shape}")
        images[modality] = image
        arrays.append(array)
    shape = tuple(int(value) for value in arrays[0].shape)
    if any(tuple(array.shape) != shape for array in arrays[1:]):
        raise ValueError(f"registered modality geometry mismatch: {[array.shape for array in arrays]}")
    if not 0 <= int(z) < shape[-1]:
        raise IndexError(f"z={z} outside registered depth={shape[-1]}")
    from andi_rewrite.data.robust_normalization import robust_normalize_volume

    volume = robust_normalize_volume(torch.from_numpy(np.stack(arrays, axis=0)))
    expected = Resize(128, antialias=True)(volume[..., int(z)]).contiguous()
    return expected, images, shape


def _affine_audit(images: Mapping[str, Any], reference: Any, expected_orientation: Sequence[Any]) -> dict[str, Any]:
    import nibabel as nib

    affines = {modality: np.asarray(image.affine, dtype=np.float64).tolist() for modality, image in images.items()}
    affine_ok = all(np.allclose(np.asarray(image.affine), np.asarray(reference), atol=1e-6, rtol=0.0) for image in images.values())
    orientations = {modality: list(nib.aff2axcodes(image.affine)) for modality, image in images.items()}
    orientation_ok = all(tuple(value) == tuple(expected_orientation) for value in orientations.values())
    return {
        "affines": affines,
        "reference_affine": reference,
        "affine_matches_reference": bool(affine_ok),
        "orientations": orientations,
        "reference_orientation": list(expected_orientation),
        "orientation_matches_reference": bool(orientation_ok),
    }


def _mask_audit(segmentation_path: str, z: int) -> dict[str, Any]:
    import nibabel as nib
    import torch
    import torch.nn.functional as F

    path = Path(segmentation_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(nib.load(str(path)).dataobj)
    if array.ndim != 3:
        raise ValueError(f"segmentation shape is not 3-D: {array.shape}")
    if not 0 <= int(z) < array.shape[-1]:
        raise IndexError(f"z={z} outside segmentation depth={array.shape[-1]}")
    native = np.asarray(array[..., int(z)] > 0, dtype=bool)
    model = F.interpolate(
        torch.from_numpy((array > 0).astype(np.float32))[None, None],
        size=(128, 128, int(array.shape[-1])),
        mode="nearest-exact",
    )[0, 0, ..., int(z)].bool().numpy()
    return {
        "native_shape": [int(value) for value in array.shape],
        "model_shape": [128, 128, int(array.shape[-1])],
        "native_lesion_voxels": int(np.count_nonzero(native)),
        "model_lesion_voxels": int(np.count_nonzero(model)),
        "native_zero": bool(not np.any(native)),
        "model_zero": bool(not np.any(model)),
    }


def _check_slice(cohort: str, pair: Mapping[str, Any], z: int, slice_index: int) -> dict[str, Any]:
    import torch
    from andi_rewrite.data.domain_classifier.readers import load_slice

    record_results: dict[str, Any] = {"z": int(z), "slice_index": int(slice_index), "status": "INCOMPLETE"}
    healthy = _healthy_metadata(cohort, pair, slice_index)
    brats = _brats_metadata(pair)
    record_results["healthy"] = {
        "participant_id": healthy["participant_id"],
        "case_id": healthy["case_id"],
        "source_key": healthy["source_key"],
        "lmdb_path": healthy["lmdb_path"],
        "lmdb_state": _file_state(Path(healthy["lmdb_path"]) / "data.mdb"),
        "raw_paths": {modality: _file_state(path) for modality, path in healthy["paths"].items()},
    }
    record_results["brats"] = {
        "participant_id": brats["participant_id"],
        "case_id": brats["case_id"],
        "raw_paths": {modality: _file_state(path) for modality, path in brats["paths"].items()},
        "segmentation": _file_state(brats["segmentation"]),
    }
    checks: dict[str, Any] = {
        "shape": [3, 128, 128],
        "modalities": list(MODALITIES),
        "torch_equal": False,
        "max_abs_error": None,
        "no_2x_minus_1": False,
    }
    try:
        # Read the production outputs first, even when the independently
        # recorded raw volume is unavailable.  This proves the LMDB/key and
        # the BraTS dataset path are the actual reader inputs; missing raw data
        # then becomes an explicit INCOMPLETE expected-side result.
        healthy_tensor = load_slice(
            _record(healthy, z=z, label=0, repo_root=REPO_ROOT), stage="final", base_dir=REPO_ROOT
        )
        brats_tensor = load_slice(
            _record(brats, z=z, label=1, repo_root=REPO_ROOT), stage="final", base_dir=REPO_ROOT
        )
        checks["healthy_actual_shape"] = list(healthy_tensor.shape)
        checks["brats_actual_shape"] = list(brats_tensor.shape)
        if tuple(healthy_tensor.shape) != (3, 128, 128) or tuple(brats_tensor.shape) != (3, 128, 128):
            raise ValueError("production reader did not return [3,128,128]")
    except Exception as exc:
        # A production-reader/LMDB/BraTS failure is a checker failure.  It is
        # distinct from an unavailable independent registered-volume source.
        record_results["checks"] = checks
        record_results["status"] = "FAIL"
        record_results["reason"] = f"production_reader_{type(exc).__name__}: {exc}"
        return _safe_json(record_results)

    try:
        expected_healthy, healthy_images, healthy_shape = _read_registered_expected(healthy["paths"], z)
        expected_brats, brats_images, brats_shape = _read_registered_expected(brats["paths"], z)
        checks["healthy_registered_shape"] = list(healthy_shape)
        checks["brats_registered_shape"] = list(brats_shape)
        checks["healthy_affine"] = _affine_audit(healthy_images, pair["affine"], pair.get("orientation", []))
        checks["brats_affine"] = _affine_audit(brats_images, pair["affine"], pair.get("orientation", []))
        for name, actual, expected in (
            ("healthy", healthy_tensor, expected_healthy),
            ("brats", brats_tensor, expected_brats),
        ):
            if tuple(actual.shape) != (3, 128, 128):
                raise ValueError(f"{name} production tensor shape={tuple(actual.shape)}")
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"{name} production tensor contains NaN/Inf")
            error = float(torch.max(torch.abs(actual - expected)))
            checks[f"{name}_max_abs_error"] = error
            checks[f"{name}_torch_equal"] = bool(torch.equal(actual, expected))
            if error != 0.0 or not bool(torch.equal(actual, expected)):
                raise AssertionError(f"{name} production reader differs from independent expected: max={error}")
        checks["torch_equal"] = True
        checks["max_abs_error"] = 0.0
        checks["no_2x_minus_1"] = True
        checks["normalization_contract"] = "robust_iqr/per_volume_per_modality/background=-1/clip=false/model_normalize_input=false"
        checks["healthy_mask"] = {"not_applicable": True}
        checks["brats_mask"] = _mask_audit(brats["segmentation"], z)
        if not checks["brats_mask"]["native_zero"] or not checks["brats_mask"]["model_zero"]:
            raise AssertionError("BraTS slice is not tumor-free in native/model mask")
        record_results["checks"] = checks
        record_results["status"] = "PASS"
    except OSError as exc:
        # Missing or inaccessible registered source files are honest
        # INCOMPLETE evidence.  The production tensors above remain recorded,
        # but an LMDB-vs-LMDB result is not raw-input parity.
        record_results["checks"] = checks
        record_results["reason"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # retain exact failure in durable audit
        record_results["checks"] = checks
        record_results["status"] = "FAIL"
        record_results["reason"] = f"{type(exc).__name__}: {exc}"
    return _safe_json(record_results)


def _run(cohort: str, comparison_path: Path, requested_pairs: int, *, threads: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(threads))
    started = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "cohort": cohort,
        "comparison_path": str(comparison_path),
        "comparison_fingerprint": _comparison_fingerprint(comparison_path),
        "requested_pairs": int(requested_pairs),
        "requested_slices_per_pair": 3,
        "started_at_utc": started,
        "status": "INCOMPLETE",
        "pairs": [],
    }
    try:
        comparison = _load_comparison(comparison_path)
        result["comparison_seed"] = comparison.get("seed")
        result["comparison_shape"] = comparison.get("shape")
        result["comparison_modalities"] = comparison.get("modalities")
        pairs = comparison["pairs"][: int(requested_pairs)]
        result["selected_pairs"] = len(pairs)
        for pair in pairs:
            pair_result = {
                "pair": int(pair.get("pair", len(result["pairs"]) + 1)),
                "healthy_participant": pair.get("fomo", pair.get("source_participant")),
                "brats_participant": pair.get("brats", pair.get("brats_case")),
                "z": [int(value) for value in pair.get("z", [])],
                "slices": [],
            }
            for index, z in enumerate(pair_result["z"]):
                pair_result["slices"].append(_check_slice(cohort, pair, z, index))
            pair_result["status"] = "PASS" if pair_result["slices"] and all(item["status"] == "PASS" for item in pair_result["slices"]) else (
                "FAIL" if any(item["status"] == "FAIL" for item in pair_result["slices"]) else "INCOMPLETE"
            )
            result["pairs"].append(pair_result)
        result["pass_pairs"] = sum(item["status"] == "PASS" for item in result["pairs"])
        result["fail_pairs"] = sum(item["status"] == "FAIL" for item in result["pairs"])
        result["incomplete_pairs"] = sum(item["status"] == "INCOMPLETE" for item in result["pairs"])
        result["actual_pair_count"] = len(result["pairs"])
        result["actual_slice_count"] = sum(len(item["slices"]) for item in result["pairs"])
        result["status"] = "PASS" if result["pairs"] and result["pass_pairs"] == len(result["pairs"]) else (
            "FAIL" if result["fail_pairs"] else "INCOMPLETE"
        )
    except Exception as exc:
        result["status"] = "FAIL"
        result["reason"] = f"{type(exc).__name__}: {exc}"
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return _safe_json(result)


def _write_result(output: Path, command_log: Path, run: dict[str, Any], command: str, append: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command_log.parent.mkdir(parents=True, exist_ok=True)
    with command_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(command + "\n")
    if append and output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        runs = list(existing.get("runs", [])) if isinstance(existing, Mapping) else []
        # Preserve the first non-append invocation when a later invocation is
        # appended.  This avoids losing the initial smoke evidence.
        if isinstance(existing, Mapping) and isinstance(existing.get("run"), Mapping):
            runs.insert(0, existing["run"])
        runs.append(run)
        payload = {
            "schema_version": 1,
            "kind": "domain_classifier_input_parity",
            "runs": runs,
            "latest": run,
            "command_log": str(command_log),
        }
    else:
        payload = {
            "schema_version": 1,
            "kind": "domain_classifier_input_parity",
            "run": run,
            "command_log": str(command_log),
        }
    output.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-cohort", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/diagnostics/domain_classifier/parity.json")
    parser.add_argument("--command-log", type=Path, default=REPO_ROOT / "outputs/diagnostics/domain_classifier/parity_commands.log")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)
    if args.pairs_per_cohort < 1:
        parser.error("--pairs-per-cohort must be >= 1")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    runs = []
    for cohort, comparison_path in COMPARISON_PATHS.items():
        if comparison_path.is_file():
            runs.append(_run(cohort, comparison_path, args.pairs_per_cohort, threads=args.threads))
        else:
            runs.append({"cohort": cohort, "comparison_path": str(comparison_path), "status": "INCOMPLETE", "reason": f"FileNotFoundError: {comparison_path}", "requested_pairs": args.pairs_per_cohort})
    overall = "PASS" if runs and all(item.get("status") == "PASS" for item in runs) else (
        "FAIL" if any(item.get("status") == "FAIL" for item in runs) else "INCOMPLETE"
    )
    run = {
        "status": overall,
        "pairs_per_cohort": int(args.pairs_per_cohort),
        "threads": int(args.threads),
        "cohorts": runs,
        "command": command,
        "started_at_utc": min((item.get("started_at_utc", "") for item in runs), default=""),
        "finished_at_utc": max((item.get("finished_at_utc", "") for item in runs), default=""),
    }
    _write_result(args.output, args.command_log, run, command, bool(args.append))
    print(json.dumps(_safe_json(run), indent=2, ensure_ascii=False))
    return 0 if overall == "PASS" else 2 if overall == "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
