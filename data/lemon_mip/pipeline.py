"""Fail-closed, resumable stage orchestration for the LEMON MIP dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .geometry import FixedSquareROI, derive_fixed_square_roi
from .manifest import (
    MODALITY_ORDER,
    SessionRecord,
    deterministic_session_split,
    validate_session_csv,
    write_manifest_products,
)
from .processing import process_session, save_transform_artifacts
from .qc import (
    apply_dataset_outliers,
    save_edge_mask_overlay,
    save_fixed_roi_slices,
    save_flair_montage,
    save_mask_qc_figure,
    save_registration_qc_figure,
)
from .registration import RegistrationSettings
from .resources import (
    brain_mask_path,
    disk_free_gib,
    fetch_mni2009c_template,
    inspect_hdbet_runtime,
    load_mni2009c_template,
    run_hdbet,
    sha256_file,
)
from .store import (
    MAP_SIZE_BYTES,
    audit_published_lmdb,
    audit_staging_lmdb,
    build_staging_lmdb,
    publish_lmdb,
    save_session_bundle,
    session_artifact_dir,
    validate_session_bundle,
)


PIPELINE_SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_hash(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _output_root(config: dict) -> Path:
    path = Path(config["output_root"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def registration_settings(config: dict) -> RegistrationSettings:
    value = config.get("registration", {})
    return RegistrationSettings(
        nbins=int(value.get("mi_bins", 32)),
        sampling_proportion=value.get("sampling_proportion"),
        level_iters=tuple(int(item) for item in value.get("iterations", [1000, 100, 10])),
        sigmas=tuple(float(item) for item in value.get("sigmas", [3, 1, 0])),
        factors=tuple(int(item) for item in value.get("factors", [4, 2, 1])),
    )


def load_validated_records(config: dict) -> tuple[list[SessionRecord], dict]:
    records = validate_session_csv(
        config["input_csv"],
        expected_sessions=int(config.get("expected_sessions", 115)),
        validate_voxels=True,
    )
    train, validation, split_metadata = deterministic_session_split(
        records,
        validation_count=int(config.get("validation_count", 5)),
        seed=int(config.get("seed", 73)),
    )
    by_id = {record.session_id: record for record in [*train, *validation]}
    ordered = [by_id[record.session_id] for record in records]
    return ordered, split_metadata


def explicit_exclusion_ids(config: dict) -> tuple[str, ...]:
    """Return the ordered, explicitly authorized release exclusions."""

    value = config.get("explicit_exclusions")
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise TypeError("explicit_exclusions must be a mapping.")
    if value.get("authorized") is not True:
        raise ValueError("explicit_exclusions.authorized must be true.")
    if value.get("preserve_original_split") is not True:
        raise ValueError("Explicit exclusion must preserve the original seeded split.")
    raw_ids = value.get("session_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("explicit_exclusions.session_ids must be a non-empty list.")
    session_ids: list[str] = []
    for index, item in enumerate(raw_ids):
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(
                f"explicit_exclusions.session_ids[{index}] must be a trimmed non-empty string."
            )
        session_ids.append(item)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("explicit_exclusions.session_ids must be unique.")
    expected = value.get("expected_excluded_sessions")
    if expected is not None and int(expected) != len(session_ids):
        raise ValueError(
            "Explicit exclusion count mismatch: "
            f"expected={expected}, configured={len(session_ids)}."
        )
    return tuple(session_ids)


def select_release_records(
    config: dict,
    records: Iterable[SessionRecord],
    review_rows: Iterable[dict[str, str]],
) -> tuple[list[SessionRecord], dict[str, Any]]:
    """Filter only explicitly adjudicated exclusions while preserving the seeded split."""

    records = list(records)
    review_rows = list(review_rows)
    excluded = explicit_exclusion_ids(config)
    excluded_set = set(excluded)
    known = {record.session_id for record in records}
    if excluded_set - known:
        raise ValueError(f"Configured exclusions are absent from the source data: {sorted(excluded_set - known)}")
    review_by_id = {row["session_id"]: row for row in review_rows}
    observed_excluded = {
        session_id
        for session_id, row in review_by_id.items()
        if row.get("manual_status") == "EXCLUDE"
    }
    if observed_excluded != excluded_set:
        raise RuntimeError(
            "Release exclusions do not exactly match manual EXCLUDE adjudications: "
            f"configured_only={sorted(excluded_set - observed_excluded)}, "
            f"adjudicated_only={sorted(observed_excluded - excluded_set)}."
        )
    missing_review = excluded_set - set(review_by_id)
    if missing_review:
        raise RuntimeError(f"Configured exclusions are absent from the manual-review queue: {sorted(missing_review)}")
    selected = [record for record in records if record.session_id not in excluded_set]
    settings = config.get("explicit_exclusions", {})
    expected_remaining = settings.get("expected_remaining_sessions")
    if expected_remaining is not None and int(expected_remaining) != len(selected):
        raise RuntimeError(
            f"Release session count mismatch: expected={expected_remaining}, actual={len(selected)}."
        )
    split_counts = {
        split: sum(record.split == split for record in selected) for split in ("train", "val")
    }
    expected_split_counts = settings.get("expected_remaining_split_counts")
    if expected_split_counts is not None:
        normalized = {str(key): int(value) for key, value in expected_split_counts.items()}
        if normalized != split_counts:
            raise RuntimeError(
                f"Release split counts mismatch: expected={normalized}, actual={split_counts}."
            )
    if set(split_counts) != {"train", "val"} or min(split_counts.values()) <= 0:
        raise RuntimeError(f"Release must retain non-empty train and val splits: {split_counts}.")
    metadata = {
        "status": "PASS",
        "selection_policy": "explicit user exclusions; preserve original seed-73 split; no redraw",
        "source_session_count": len(records),
        "selected_session_count": len(selected),
        "excluded_session_count": len(excluded),
        "source_split_counts": {
            split: sum(record.split == split for record in records) for split in ("train", "val")
        },
        "selected_split_counts": split_counts,
        "excluded_session_ids": list(excluded),
        "selected_session_ids_csv_order": [record.session_id for record in selected],
        "authorized_by": settings.get("authorized_by", "user"),
        "decision_reason": settings.get("reason", "explicit user decision"),
    }
    return selected, metadata


def _manifest_is_compatible(config: dict, records: list[SessionRecord]) -> bool:
    path = _output_root(config) / "metadata" / "manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = Path(config["input_csv"]).resolve()
    return bool(
        payload.get("input_csv") == str(source)
        and payload.get("input_csv_sha256") == sha256_file(source)
        and payload.get("session_count") == len(records)
        and payload.get("expected_channel_order") == list(MODALITY_ORDER)
    )


def validate_stage(config: dict, *, config_path: str | Path | None = None) -> dict:
    root = _output_root(config)
    before = disk_free_gib(root)
    records, split_metadata = load_validated_records(config)
    metadata_dir = root / "metadata"
    if metadata_dir.exists() and any(metadata_dir.iterdir()):
        if not _manifest_is_compatible(config, records):
            raise FileExistsError(
                f"Metadata directory is non-empty and incompatible: {metadata_dir}"
            )
    else:
        write_manifest_products(metadata_dir, config["input_csv"], records, split_metadata)
        if config_path is not None:
            shutil.copy2(Path(config_path).resolve(), metadata_dir / "preprocess_config_snapshot.yaml")
    expected_validation = list(config.get("expected_validation_session_ids", []))
    actual_validation = [record.session_id for record in records if record.split == "val"]
    if expected_validation and set(actual_validation) != set(expected_validation):
        raise ValueError(
            f"Seeded validation split mismatch: actual={actual_validation}, expected={expected_validation}."
        )
    report = {
        "stage": "validate",
        "status": "PASS",
        "validated_at": _now(),
        "python_executable": sys.executable,
        "input_csv": str(Path(config["input_csv"]).resolve()),
        "input_csv_sha256": sha256_file(Path(config["input_csv"]).resolve()),
        "sessions": len(records),
        "nifti_files_fully_validated": len(records) * 3,
        "train_sessions": sum(record.split == "train" for record in records),
        "validation_sessions_csv_order": actual_validation,
        "validation_sessions_permutation_order": split_metadata[
            "validation_session_ids_permutation_order"
        ],
        "channel_order": list(MODALITY_ORDER),
        "disk_free_gib_before": before,
        "disk_free_gib_after": disk_free_gib(root),
    }
    (metadata_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def template_stage(config: dict) -> dict:
    root = _output_root(config)
    metadata = fetch_mni2009c_template(root)
    t1, mask, affine = load_mni2009c_template(root)
    roi = derive_fixed_square_roi(
        mask,
        affine,
        margin_mm=float(config.get("roi", {}).get("margin_mm", 8.0)),
        output_size=int(config.get("roi", {}).get("output_size", 128)),
    )
    roi_path = root / "templates" / "mni2009c" / "fixed_roi.json"
    payload = {
        "status": "PASS",
        "template": metadata,
        "roi": roi.as_dict(),
        "roi_policy": "all-z template-mask x/y bounding box; physical margin; centered physical square",
    }
    if roi_path.exists():
        existing = json.loads(roi_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Existing ROI definition is incompatible: {roi_path}")
    else:
        roi_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_fixed_roi(config: dict) -> FixedSquareROI:
    path = _output_root(config) / "templates" / "mni2009c" / "fixed_roi.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["roi"]
    return FixedSquareROI(
        x_start=int(payload["x_start"]),
        x_stop=int(payload["x_stop"]),
        y_start=int(payload["y_start"]),
        y_stop=int(payload["y_stop"]),
        margin_mm=float(payload["margin_mm"]),
        side_mm=float(payload["side_mm"]),
        spacing_x_mm=float(payload["spacing_x_mm"]),
        spacing_y_mm=float(payload["spacing_y_mm"]),
        template_shape=tuple(int(value) for value in payload["template_shape"]),
        output_size=int(payload["output_size"]),
    )


def pilot_records(records: list[SessionRecord], config: dict) -> list[SessionRecord]:
    indices = [int(value) for value in config.get("pilot", {}).get("csv_indices", [0, 28, 57, 86, 114])]
    by_index = {record.csv_index: record for record in records}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"Pilot CSV indices not found: {missing}")
    selected = [by_index[index] for index in indices]
    expected = list(config.get("pilot", {}).get("expected_case_ids", []))
    actual = [record.case_id for record in selected]
    if expected and actual != expected:
        raise ValueError(f"Pilot case IDs mismatch: actual={actual}, expected={expected}.")
    return selected


def hdbet_stage(config: dict, *, scope: str) -> dict:
    root = _output_root(config)
    records, _split = load_validated_records(config)
    selected = pilot_records(records, config) if scope == "pilot" else records
    before = disk_free_gib(root)
    minimum = float(config.get("disk", {}).get("minimum_before_full_gib", 20.0))
    if scope == "full" and before < minimum:
        raise RuntimeError(
            f"Disk gate failed before full preprocessing: {before:.3f} GiB < {minimum:.3f} GiB."
        )
    runtime = inspect_hdbet_runtime(config["hdbet_executable"])
    environment_report = {
        **runtime,
        "disk_free_gib_before_environment_install": config.get("disk", {}).get(
            "hdbet_install_before_gib"
        ),
        "disk_free_gib_after_environment_install_before_weights": before,
        "verified_at": _now(),
    }
    environment_report_path = root / "metadata" / "hdbet_environment_report.json"
    environment_report_path.write_text(
        json.dumps(environment_report, indent=2), encoding="utf-8"
    )
    paths = run_hdbet(selected, root, config["hdbet_executable"], scope=scope)
    report = {
        "stage": "hdbet",
        "scope": scope,
        "status": "PASS",
        "hd_bet_version": "2.0.1",
        "input": "native T1",
        "device": "cuda",
        "tta": "default enabled",
        "flags": ["--save_bet_mask", "--no_bet_image"],
        "mask_count": len(paths),
        "disk_free_gib_before": before,
        "disk_free_gib_after": disk_free_gib(root),
        "completed_at": _now(),
        "environment_report": str(environment_report_path.resolve()),
    }
    report_path = root / "metadata" / f"hdbet_{scope}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def processing_fingerprint(config: dict) -> str:
    root = _output_root(config)
    template_metadata_path = root / "templates" / "mni2009c" / "template_metadata.json"
    roi_path = root / "templates" / "mni2009c" / "fixed_roi.json"
    payload = {
        "schema": PIPELINE_SCHEMA_VERSION,
        "input_csv_sha256": sha256_file(Path(config["input_csv"]).resolve()),
        "template_metadata_sha256": sha256_file(template_metadata_path),
        "roi_sha256": sha256_file(roi_path),
        "registration": registration_settings(config).as_dict(),
        "guard_dilation_mm": float(config.get("mask", {}).get("guard_dilation_mm", 5.0)),
        "channel_order": list(MODALITY_ORDER),
        "normalization": "existing normalize_volume positive-foreground p99",
        "final_interpolation": {"image": "linear", "mask": "nearest"},
    }
    return _json_hash(payload)


def _save_visuals(result, target: Path) -> None:
    visual_dir = target / "qc"
    metrics = result.metrics
    save_registration_qc_figure(
        visual_dir / "flair_to_t1_checker_overlay.png",
        result.registration_images["native_t1_brain"],
        result.registration_images["flair_on_t1_qc"],
        result.masks["native_hdbet"],
        title=f"{result.record.session_id}: FLAIR to T1",
        metrics_text=(
            f"MI {metrics['FLAIR_T1_MI_before']:.4f} -> {metrics['FLAIR_T1_MI_after']:.4f}"
        ),
    )
    save_registration_qc_figure(
        visual_dir / "t2_to_t1_checker_overlay.png",
        result.registration_images["native_t1_brain"],
        result.registration_images["t2_on_t1_qc"],
        result.masks["native_hdbet"],
        title=f"{result.record.session_id}: T2 to T1",
        metrics_text=f"MI {metrics['T2_T1_MI_before']:.4f} -> {metrics['T2_T1_MI_after']:.4f}",
    )
    save_registration_qc_figure(
        visual_dir / "t1_to_mni_checker_overlay.png",
        result.registration_images["mni_t1_brain"],
        result.registration_images["t1_on_mni"],
        result.masks["final"],
        title=f"{result.record.session_id}: T1 to MNI",
        metrics_text=(
            f"MI {metrics['T1_MNI_MI_before']:.4f} -> {metrics['T1_MNI_MI_after']:.4f}; "
            f"Dice={metrics['mask_dice']:.4f}"
        ),
    )
    save_mask_qc_figure(
        visual_dir / "four_masks.png", result.masks, title=f"{result.record.session_id}: masks"
    )
    save_edge_mask_overlay(
        visual_dir / "flair_final_mask_edge.png",
        result.registration_images["flair_on_mni"],
        result.masks["final"],
        title=f"{result.record.session_id}: FLAIR/final-mask edge",
    )
    save_fixed_roi_slices(
        visual_dir / "fixed_roi_slices.png",
        result.slices,
        title=f"{result.record.session_id}: fixed 128x128 ROI",
    )
    save_flair_montage(
        visual_dir / "flair_montage.png",
        result.registration_images["flair_on_mni"],
        result.masks["final"],
        title=f"{result.record.session_id}: registered FLAIR",
    )


def _process_records(config: dict, records: Iterable[SessionRecord]) -> list[dict[str, Any]]:
    root = _output_root(config)
    records = list(records)
    mni_t1, mni_mask, mni_affine = load_mni2009c_template(root)
    roi = load_fixed_roi(config)
    settings = registration_settings(config)
    fingerprint = processing_fingerprint(config)
    metrics_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        target = session_artifact_dir(root, record)
        completion = target / "complete.json"
        if completion.is_file():
            validated = validate_session_bundle(root, record, processing_fingerprint=fingerprint)
            metrics_rows.append(validated["metrics"])
            print(f"[{index}/{len(records)}] reuse COMPLETE {record.session_id}", flush=True)
            continue
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"Incomplete/non-empty session product blocks resume: {target}. "
                "Use --overwrite-product-dir with exactly this directory after inspection."
            )
        print(f"[{index}/{len(records)}] process {record.session_id}", flush=True)
        mask_path = brain_mask_path(root, record)
        if not mask_path.is_file():
            raise FileNotFoundError(f"HD-BET mask missing for {record.session_id}: {mask_path}")
        result = process_session(
            record,
            mask_path,
            mni_t1,
            mni_mask,
            mni_affine,
            roi,
            registration_settings=settings,
            guard_dilation_mm=float(config.get("mask", {}).get("guard_dilation_mm", 5.0)),
        )
        save_transform_artifacts(result, target, settings)
        _save_visuals(result, target)
        save_session_bundle(result, root, processing_fingerprint=fingerprint)
        metrics_rows.append(result.metrics)
    return metrics_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sync_manual_review(
    path: Path,
    required: list[tuple[dict[str, Any], str]],
) -> list[dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["session_id"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, str]] = []
    for metrics, reason in required:
        old = existing.get(str(metrics["session_id"]), {})
        rows.append(
            {
                "case_id": str(metrics["case_id"]),
                "session": str(metrics["session"]),
                "session_id": str(metrics["session_id"]),
                "automatic_status": str(metrics["QC_status"]),
                "required_reason": reason,
                "visual_directory": str(
                    (path.parents[1] / "processed_sessions" / str(metrics["case_id"]) / str(metrics["session"]) / "qc").resolve()
                ),
                "manual_status": old.get("manual_status", "PENDING"),
                "reviewer": old.get("reviewer", ""),
                "reviewed_at": old.get("reviewed_at", ""),
                "notes": old.get("notes", ""),
            }
        )
    _write_csv(path, rows)
    return rows


def _gate_status(
    metrics_rows: list[dict[str, Any]],
    review_rows: list[dict[str, str]],
    *,
    allowed_exclusions: set[str] | None = None,
) -> str:
    allowed_exclusions = set(allowed_exclusions or ())
    if any(row.get("QC_status") == "FAIL" for row in metrics_rows):
        return "FAIL"
    decisions = [row.get("manual_status", "PENDING") for row in review_rows]
    if any(value == "FAIL" for value in decisions):
        return "FAIL"
    observed_exclusions = {
        row["session_id"] for row in review_rows if row.get("manual_status") == "EXCLUDE"
    }
    if observed_exclusions - allowed_exclusions:
        return "FAIL"
    if observed_exclusions != allowed_exclusions:
        return "PENDING_MANUAL_REVIEW"
    if any(value not in {"PASS", "EXCLUDE"} for value in decisions):
        return "PENDING_MANUAL_REVIEW"
    return "PASS"


def _write_qc_report(
    root: Path,
    scope: str,
    metrics_rows: list[dict[str, Any]],
    review_rows: list[dict[str, str]],
    suspicious: list[dict[str, Any]],
    *,
    config: dict,
) -> dict:
    allowed_exclusions = set(explicit_exclusion_ids(config)) if scope == "full" else set()
    status = _gate_status(
        metrics_rows,
        review_rows,
        allowed_exclusions=allowed_exclusions,
    )
    report = {
        "scope": scope,
        "status": status,
        "created_at": _now(),
        "session_count": len(metrics_rows),
        "automatic_status_counts": {
            value: sum(row.get("QC_status") == value for row in metrics_rows)
            for value in ("PASS", "REVIEW", "FAIL")
        },
        "manual_review_required": len(review_rows),
        "manual_status_counts": {
            value: sum(row.get("manual_status") == value for row in review_rows)
            for value in ("PASS", "PENDING", "FAIL", "EXCLUDE")
        },
        "explicit_exclusion_ids": sorted(allowed_exclusions),
        "suspicious_top15": [row["session_id"] for row in suspicious],
        "fail_closed": True,
        "automatic_fail_not_overridable": True,
    }
    json_path = root / "qc" / f"{scope}_qc_report.json"
    md_path = root / "qc" / f"{scope}_qc_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# {scope.upper()} QC report",
        "",
        f"- Gate status: **{status}**",
        f"- Sessions: {len(metrics_rows)}",
        f"- Automatic PASS/REVIEW/FAIL: "
        f"{report['automatic_status_counts']['PASS']}/"
        f"{report['automatic_status_counts']['REVIEW']}/"
        f"{report['automatic_status_counts']['FAIL']}",
        f"- Manual reviews required: {len(review_rows)}",
        "",
        "No session is automatically excluded. Any automatic FAIL stops publication. "
        "EXCLUDE requires an exact config authorization and matching manual adjudication.",
        "",
        "## Manual review queue",
        "",
    ]
    for row in review_rows:
        lines.append(
            f"- {row['session_id']}: {row['manual_status']} — {row['required_reason']} — "
            f"`{row['visual_directory']}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _summarize_intensities(root: Path, records: Iterable[SessionRecord], scope: str) -> dict:
    values: dict[str, dict[str, list[float]]] = {
        modality: {name: [] for name in ("roi_fraction_lt_0", "roi_fraction_gt_1", "roi_fraction_eq_0")}
        for modality in MODALITY_ORDER
    }
    for record in records:
        path = session_artifact_dir(root, record) / "intensity_statistics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for modality in MODALITY_ORDER:
            for name in values[modality]:
                values[modality][name].append(float(payload[modality][name]))
    summary = {
        "scope": scope,
        "channel_order": list(MODALITY_ORDER),
        "fractions": {
            modality: {
                name: {
                    "mean": float(np.mean(items)),
                    "min": float(np.min(items)),
                    "max": float(np.max(items)),
                }
                for name, items in fields.items()
            }
            for modality, fields in values.items()
        },
    }
    (root / "qc" / f"{scope}_intensity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def processing_stage(config: dict, *, scope: str) -> dict:
    root = _output_root(config)
    records, _split = load_validated_records(config)
    selected = pilot_records(records, config) if scope == "pilot" else records
    if scope == "full":
        pilot_report_path = root / "qc" / "pilot_qc_report.json"
        if not pilot_report_path.is_file() or json.loads(pilot_report_path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("Pilot QC gate is not PASS; full preprocessing is refused.")
        free = disk_free_gib(root)
        minimum = float(config.get("disk", {}).get("minimum_before_full_gib", 20.0))
        if free < minimum:
            raise RuntimeError(f"Full preprocessing disk gate failed: {free:.3f} GiB < {minimum:.3f} GiB.")

    metrics_rows = _process_records(config, selected)
    suspicious: list[dict[str, Any]] = []
    if scope == "full":
        suspicious = apply_dataset_outliers(
            metrics_rows,
            threshold=float(config.get("qc", {}).get("modified_z_threshold", 3.5)),
            suspicious_count=int(config.get("qc", {}).get("suspicious_count", 15)),
        )
    metrics_path = root / "qc" / ("registration_qc.csv" if scope == "full" else "pilot_registration_qc.csv")
    _write_csv(metrics_path, metrics_rows)
    if scope == "pilot":
        required = [(row, "pilot: all five cases require visual review") for row in metrics_rows]
    else:
        suspicious_ids = {str(row["session_id"]) for row in suspicious}
        required = []
        for row in metrics_rows:
            reasons = []
            if row.get("QC_status") == "REVIEW":
                reasons.append("automatic REVIEW")
            if str(row["session_id"]) in suspicious_ids:
                reasons.append(f"top-15 suspicious rank {row.get('suspicious_rank')}")
            if reasons:
                required.append((row, "; ".join(reasons)))
    review_path = root / "qc" / f"manual_review_{scope}.csv"
    review_rows = _sync_manual_review(review_path, required)
    report = _write_qc_report(
        root,
        scope,
        metrics_rows,
        review_rows,
        suspicious,
        config=config,
    )
    _summarize_intensities(root, selected, scope)

    if scope == "full" and not any(row.get("QC_status") == "FAIL" for row in metrics_rows):
        report["lmdb_build"] = build_staging_lmdb(
            records,
            root,
            processing_fingerprint=processing_fingerprint(config),
            map_size=int(config.get("lmdb", {}).get("map_size_bytes", MAP_SIZE_BYTES)),
        )
        (root / "qc" / "full_qc_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def adjudicate_stage(
    config: dict,
    *,
    scope: str,
    decision: str,
    reviewer: str,
    session_ids: Iterable[str] | None = None,
    all_pending: bool = False,
    notes: str = "",
) -> dict:
    decision = decision.upper()
    if decision not in {"PASS", "FAIL", "EXCLUDE"}:
        raise ValueError("Manual decision must be PASS, FAIL, or EXCLUDE.")
    root = _output_root(config)
    review_path = root / "qc" / f"manual_review_{scope}.csv"
    with review_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(session_ids or [])
    known = {row["session_id"] for row in rows}
    if requested - known:
        raise ValueError(f"Unknown/non-required session IDs: {sorted(requested - known)}")
    if decision == "EXCLUDE":
        if scope != "full":
            raise ValueError("EXCLUDE is only allowed for the full release review.")
        selected_for_exclusion = {
            row["session_id"]
            for row in rows
            if row["session_id"] in requested
            or (all_pending and row["manual_status"] == "PENDING")
        }
        allowed = set(explicit_exclusion_ids(config))
        if selected_for_exclusion - allowed:
            raise ValueError(
                "EXCLUDE selection is not explicitly authorized by config: "
                f"{sorted(selected_for_exclusion - allowed)}"
            )
    changed = 0
    for row in rows:
        if row["session_id"] in requested or (all_pending and row["manual_status"] == "PENDING"):
            row["manual_status"] = decision
            row["reviewer"] = reviewer
            row["reviewed_at"] = _now()
            row["notes"] = notes
            changed += 1
    if changed == 0:
        raise ValueError("No manual-review rows selected.")
    _write_csv(review_path, rows)
    metrics_name = "registration_qc.csv" if scope == "full" else "pilot_registration_qc.csv"
    with (root / "qc" / metrics_name).open(newline="", encoding="utf-8") as handle:
        metrics_rows = list(csv.DictReader(handle))
    suspicious = [row for row in metrics_rows if str(row.get("suspicious_rank", "")).strip()]
    report = _write_qc_report(
        root,
        scope,
        metrics_rows,
        rows,
        suspicious,
        config=config,
    )
    return {"changed": changed, "report": report}


def finalize_stage(config: dict) -> dict:
    root = _output_root(config)
    full_report_path = root / "qc" / "full_qc_report.json"
    if not full_report_path.is_file():
        raise FileNotFoundError("Full QC report is missing.")
    full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
    if full_report.get("status") != "PASS":
        raise RuntimeError(f"Full QC gate is {full_report.get('status')}; publication is refused.")
    records, _split = load_validated_records(config)
    review_path = root / "qc" / "manual_review_full.csv"
    with review_path.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))
    release_records, release_selection = select_release_records(config, records, review_rows)
    release_selection = {**release_selection, "created_at": _now()}
    selection_path = root / "metadata" / "release_selection.json"
    selection_path.write_text(json.dumps(release_selection, indent=2), encoding="utf-8")
    fingerprint = processing_fingerprint(config)
    build_report = build_staging_lmdb(
        release_records,
        root,
        processing_fingerprint=fingerprint,
        map_size=int(config.get("lmdb", {}).get("map_size_bytes", MAP_SIZE_BYTES)),
    )
    building_audit = audit_staging_lmdb(
        release_records,
        root,
        processing_fingerprint=fingerprint,
    )
    publication = publish_lmdb(root)
    audit = audit_published_lmdb(
        release_records,
        root,
        processing_fingerprint=fingerprint,
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = None
    report = {
        "status": "PASS",
        "completed_at": _now(),
        "input_validation": str((root / "metadata" / "validation_report.json").resolve()),
        "pilot_qc": str((root / "qc" / "pilot_qc_report.md").resolve()),
        "full_qc": str((root / "qc" / "full_qc_report.md").resolve()),
        "registration_qc": str((root / "qc" / "registration_qc.csv").resolve()),
        "release_selection": release_selection,
        "release_selection_path": str(selection_path.resolve()),
        "lmdb_build": build_report,
        "lmdb_building_audit": building_audit,
        "lmdb_audit": audit,
        "publication": publication,
        "channel_order": list(MODALITY_ORDER),
        "python_executable": sys.executable,
        "git_commit": commit,
        "disk_free_gib": disk_free_gib(root),
        "automatic_exclusions": 0,
        "explicit_user_exclusions": len(release_selection["excluded_session_ids"]),
    }
    (root / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def clear_explicit_product_dir(config: dict, requested: str | Path) -> Path:
    root = _output_root(config).resolve()
    target = Path(requested).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Overwrite target must be a specific directory below {root}: {target}")
    if not target.is_dir() or target.is_symlink():
        raise ValueError(f"Overwrite target must be an existing non-symlink directory: {target}")
    candidates = [target, *target.rglob("*")]
    reparse_points = []
    for candidate in candidates:
        attributes = getattr(os.lstat(candidate), "st_file_attributes", 0)
        if candidate.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            reparse_points.append(str(candidate))
    if reparse_points:
        raise ValueError(f"Overwrite target contains symlink/reparse points: {reparse_points}")
    shutil.rmtree(target)
    return target


__all__ = [
    "adjudicate_stage",
    "clear_explicit_product_dir",
    "finalize_stage",
    "hdbet_stage",
    "load_validated_records",
    "explicit_exclusion_ids",
    "pilot_records",
    "processing_fingerprint",
    "select_release_records",
    "processing_stage",
    "registration_settings",
    "template_stage",
    "validate_stage",
]
