"""Offline, resumable cache pipeline for BraTS-to-MPI preprocessing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .manifest import BraTSMPIRecord, discover_records, validate_record_geometry
from .processing import (
    CACHE_SCHEMA_VERSION,
    MODES,
    PROCESSING_IMPLEMENTATION_VERSION,
    process_subject,
    resolve_mni_resources,
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    payload["_config_path"] = str(path.resolve())
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _stable_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    # Size/mtime are part of the cache key so a changed resource is re-read.
    del size, mtime_ns
    return _sha256(Path(path_text))


def _file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        identity["sha256"] = _sha256_cached(
            identity["path"], identity["size"], identity["mtime_ns"]
        )
    return identity


def subject_fingerprint(record: BraTSMPIRecord, mode: str, config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    relevant_config = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_") and key not in {"pilot_subjects", "selections"}
    }
    sources = {
        name: _file_identity(Path(path))
        for name, path in {
            "flair": record.flair_path,
            "t1": record.t1_path,
            "t2": record.t2_path,
            "seg": record.segmentation_path,
        }.items()
    }
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "implementation_version": PROCESSING_IMPLEMENTATION_VERSION if mode == "mni_affine" else "brats-mpi-v1-fixed-fov",
        "mode": mode,
        "config": relevant_config,
        "sources": sources,
    }
    if mode == "mni_affine":
        resources = resolve_mni_resources(config)
        payload["mni_resources"] = {
            "t1": _file_identity(resources["t1"], content_hash=True),
            "mask": _file_identity(resources["mask"], content_hash=True),
            "roi": _file_identity(resources["roi"], content_hash=True),
        }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest(), payload


def _subject_dir(config: Mapping[str, Any], mode: str, subject_id: str) -> Path:
    return Path(config["output_root"]) / mode / "subjects" / subject_id


def _cache_is_current(path: Path, fingerprint: str) -> bool:
    metadata_path = path / "spatial_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("fingerprint") != fingerprint:
        return False
    if metadata.get("qc", {}).get("status") == "PASS":
        volume_path = path / "volume.npz"
        if not volume_path.is_file():
            return False
        try:
            with np.load(volume_path, allow_pickle=False) as cached:
                image = cached["image"]
                segmentation = cached["segmentation"]
                brain_mask = cached["brain_mask"]
                return (
                    image.ndim == 4
                    and image.shape[0] == 3
                    and segmentation.shape == image.shape[1:]
                    and brain_mask.shape == image.shape[1:]
                    and np.isfinite(image).all()
                )
        except (OSError, ValueError, KeyError):
            return False
    return True


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def process_and_cache_subject(
    record: BraTSMPIRecord,
    mode: str,
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    fingerprint, fingerprint_payload = subject_fingerprint(record, mode, config)
    target = _subject_dir(config, mode, record.subject_id)
    subjects_root = target.parent
    subjects_root.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        if _cache_is_current(target, fingerprint):
            metadata = json.loads((target / "spatial_metadata.json").read_text(encoding="utf-8"))
            return {"action": "resumed", **metadata}
        raise RuntimeError(
            f"cache exists but fingerprint/contents do not match for {record.subject_id}; "
            "rerun with --overwrite-subject for this exact subject"
        )
    if overwrite and target.exists():
        if not _within(target, subjects_root) or target == subjects_root:
            raise RuntimeError(f"refusing to replace unsafe cache path: {target}")
        shutil.rmtree(target)

    try:
        processed = process_subject(record, mode, config)
        metadata = {
            **processed.metadata,
            "fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_payload,
            "qc": processed.qc,
        }
        error: Exception | None = None
    except Exception as exc:  # Per-subject failures must not abort a 251-case run.
        error = exc
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "subject_id": record.subject_id,
            "mode": mode,
            "fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_payload,
            "qc": {"status": "EXCLUDED", "failures": [f"processing_error:{type(exc).__name__}"], "error": str(exc)},
        }
        processed = None

    staging = Path(tempfile.mkdtemp(prefix=f".{record.subject_id}.", dir=str(subjects_root)))
    try:
        if processed is not None and processed.qc["status"] == "PASS":
            np.savez_compressed(
                staging / "volume.npz",
                image=processed.image.astype(np.float32, copy=False),
                segmentation=processed.segmentation.astype(np.uint8, copy=False),
                brain_mask=processed.brain_mask.astype(np.uint8, copy=False),
            )
        (staging / "spatial_metadata.json").write_text(
            json.dumps(metadata, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    result = {"action": "processed", **metadata}
    if error is not None:
        result["exception_type"] = type(error).__name__
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["subject_id", "mode", "status", "action", "failures", "error"]
    keys = {str(key) for row in rows for key in row}
    fieldnames = [key for key in preferred if key in keys] + sorted(keys.difference(preferred))
    if not fieldnames:
        fieldnames = ["subject_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stable_json(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _flatten_result(result: Mapping[str, Any]) -> dict[str, Any]:
    qc = dict(result.get("qc", {}))
    return {
        "subject_id": result.get("subject_id"),
        "mode": result.get("mode"),
        "status": qc.pop("status", "EXCLUDED"),
        "action": result.get("action", "unknown"),
        **qc,
    }


def validate_dataset(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in discover_records(config):
        try:
            result = validate_record_geometry(record, check_voxels=True)
            rows.append({"subject_id": record.subject_id, "status": "PASS", **result})
        except Exception as exc:
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "status": "EXCLUDED",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            )
    report_dir = Path(config["output_root"]) / "reports"
    _write_csv(report_dir / "validation.csv", rows)
    snapshot = {key: value for key, value in config.items() if not str(key).startswith("_")}
    (report_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return rows


def _selected_records(records: Sequence[BraTSMPIRecord], subject_ids: Iterable[str] | None) -> list[BraTSMPIRecord]:
    if subject_ids is None:
        return list(records)
    requested = [str(value) for value in subject_ids]
    by_id = {record.subject_id: record for record in records}
    missing = [subject_id for subject_id in requested if subject_id not in by_id]
    if missing:
        raise ValueError(f"subjects are absent from the dataset manifest: {missing}")
    return [by_id[subject_id] for subject_id in requested]


def run_processing(
    config: Mapping[str, Any],
    *,
    modes: Sequence[str],
    subject_ids: Iterable[str] | None = None,
    label: str = "full",
    overwrite_subject: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    unknown = set(modes).difference(MODES)
    if unknown:
        raise ValueError(f"unsupported modes: {sorted(unknown)}")
    records = _selected_records(discover_records(config), subject_ids)
    overwrite_subjects = (
        {overwrite_subject}
        if isinstance(overwrite_subject, str)
        else set(overwrite_subject or [])
    )
    unknown_overwrites = overwrite_subjects.difference(record.subject_id for record in records)
    if unknown_overwrites:
        raise ValueError(f"--overwrite-subject is outside this run's subject set: {sorted(unknown_overwrites)}")
    all_results: list[dict[str, Any]] = []
    for mode in modes:
        rows: list[dict[str, Any]] = []
        for record in records:
            result = process_and_cache_subject(
                record,
                mode,
                config,
                overwrite=record.subject_id in overwrite_subjects,
            )
            flattened = _flatten_result(result)
            rows.append(flattened)
            all_results.append(flattened)
        report_dir = Path(config["output_root"]) / mode / "reports"
        _write_csv(report_dir / f"{label}_qc.csv", rows)
        _write_csv(report_dir / f"{label}_pass.csv", [row for row in rows if row["status"] == "PASS"])
        _write_csv(report_dir / f"{label}_excluded.csv", [row for row in rows if row["status"] != "PASS"])
    return all_results


def collect_status(config: Mapping[str, Any], modes: Sequence[str]) -> list[dict[str, Any]]:
    records = discover_records(config)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for record in records:
            subject_dir = _subject_dir(config, mode, record.subject_id)
            path = subject_dir / "spatial_metadata.json"
            if not path.is_file():
                rows.append({"subject_id": record.subject_id, "mode": mode, "status": "MISSING"})
                continue
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                expected_fingerprint, _ = subject_fingerprint(record, mode, config)
                if metadata.get("fingerprint") != expected_fingerprint:
                    rows.append(
                        {
                            "subject_id": record.subject_id,
                            "mode": mode,
                            "status": "STALE",
                            "failures": ["fingerprint_mismatch"],
                        }
                    )
                    continue
                if not _cache_is_current(subject_dir, expected_fingerprint):
                    rows.append(
                        {
                            "subject_id": record.subject_id,
                            "mode": mode,
                            "status": "INVALID",
                            "failures": ["cache_contents_invalid"],
                        }
                    )
                    continue
                qc = metadata.get("qc", {})
                rows.append(
                    {
                        "subject_id": record.subject_id,
                        "mode": mode,
                        "status": qc.get("status", "INVALID"),
                        "failures": qc.get("failures", []),
                    }
                )
            except (OSError, json.JSONDecodeError) as exc:
                rows.append({"subject_id": record.subject_id, "mode": mode, "status": "INVALID", "error": str(exc)})
    _write_csv(Path(config["output_root"]) / "reports" / "cache_status.csv", rows)
    return rows


def _read_first_column(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    values = [row[0].strip() for row in rows if row and row[0].strip()]
    first = values[0].lower() if values else ""
    if first in {"subject_id", "subject", "id", "scan"}:
        values = values[1:]
    return values


def write_manifests(config: Mapping[str, Any], modes: Sequence[str]) -> dict[str, Path]:
    statuses = collect_status(config, modes)
    pass_sets = {
        mode: {row["subject_id"] for row in statuses if row["mode"] == mode and row["status"] == "PASS"}
        for mode in modes
    }
    common = set.intersection(*(pass_sets[mode] for mode in modes)) if modes else set()
    source_order = [record.subject_id for record in discover_records(config)]
    ordered_common = [subject_id for subject_id in source_order if subject_id in common]
    output_dir = Path(config["output_root"]) / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    def write_subjects(name: str, subjects: Sequence[str]) -> None:
        path = output_dir / f"{name}.csv"
        _write_csv(path, [{"subject_id": subject_id} for subject_id in subjects])
        outputs[name] = path

    write_subjects("common_pass", ordered_common)
    for mode in modes:
        write_subjects(f"{mode}_pass", [subject_id for subject_id in source_order if subject_id in pass_sets[mode]])

    for selection in config.get("selections", []):
        name = str(selection["name"])
        source_csv = Path(selection["source_csv"])
        selected = _read_first_column(source_csv)
        selected_common = [subject_id for subject_id in selected if subject_id in common]
        write_subjects(f"{name}_common_pass", selected_common)
        for mode in modes:
            write_subjects(f"{name}_{mode}_pass", [subject_id for subject_id in selected if subject_id in pass_sets[mode]])
    return outputs
