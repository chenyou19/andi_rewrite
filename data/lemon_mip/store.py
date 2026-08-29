"""Resumable session bundles and fail-closed LEMON LMDB publication."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .manifest import MODALITY_ORDER, SessionRecord
from .processing import ProcessedSession


MAP_SIZE_BYTES = 16 * 1024**3


def _record_selection_signature(records: Iterable[SessionRecord]) -> dict[str, object]:
    records = list(records)
    session_ids = [record.session_id for record in records]
    digest = hashlib.sha256("\n".join(session_ids).encode("utf-8")).hexdigest()
    return {
        "session_count": len(records),
        "session_ids_sha256": digest,
        "session_split_counts": {
            split: sum(record.split == split for record in records) for split in ("train", "val")
        },
    }


def session_artifact_dir(output_root: str | Path, record: SessionRecord) -> Path:
    return Path(output_root) / "processed_sessions" / record.case_id / record.session


def save_session_bundle(
    result: ProcessedSession,
    output_root: str | Path,
    *,
    processing_fingerprint: str,
) -> Path:
    target = session_artifact_dir(output_root, result.record)
    completion = target / "complete.json"
    target.mkdir(parents=True, exist_ok=True)
    owned_products = (
        completion,
        target / "slice_bundle.npz",
        target / "metrics.json",
        target / "intensity_statistics.json",
    )
    if any(path.exists() for path in owned_products):
        raise FileExistsError(f"Session bundle products already exist: {target}")
    values = np.stack([value for _z, value in result.slices], axis=0).astype(np.float32)
    z_indices = np.asarray([z for z, _value in result.slices], dtype=np.int32)
    np.savez_compressed(target / "slice_bundle.npz", slices=values, z_indices=z_indices)
    (target / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    (target / "intensity_statistics.json").write_text(
        json.dumps(result.intensity_statistics, indent=2, allow_nan=False), encoding="utf-8"
    )
    completion.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "case_id": result.record.case_id,
                "session": result.record.session,
                "split": result.record.split,
                "slice_count": len(result.slices),
                "channel_order": list(MODALITY_ORDER),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "processing_fingerprint": processing_fingerprint,
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def validate_session_bundle(
    output_root: str | Path,
    record: SessionRecord,
    *,
    processing_fingerprint: str,
) -> dict:
    target = session_artifact_dir(output_root, record)
    completion_path = target / "complete.json"
    bundle_path = target / "slice_bundle.npz"
    metrics_path = target / "metrics.json"
    if not completion_path.is_file() or not bundle_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete session bundle for {record.session_id}: {target}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE":
        raise ValueError(f"Session completion status is not COMPLETE: {target}")
    if completion.get("processing_fingerprint") != processing_fingerprint:
        raise ValueError(
            f"Processing fingerprint mismatch for {record.session_id}: "
            f"{completion.get('processing_fingerprint')} != {processing_fingerprint}."
        )
    if completion.get("channel_order") != list(MODALITY_ORDER):
        raise ValueError(f"Session channel order mismatch: {record.session_id}")
    with np.load(bundle_path, allow_pickle=False) as archive:
        values = archive["slices"]
        z_indices = archive["z_indices"]
    if values.ndim != 4 or values.shape[1:] != (3, 128, 128) or values.dtype != np.float32:
        raise ValueError(f"Invalid bundle values for {record.session_id}: {values.shape}/{values.dtype}")
    if z_indices.ndim != 1 or len(z_indices) != len(values):
        raise ValueError(f"Invalid z index array for {record.session_id}.")
    if len(z_indices) == 0 or not np.all(np.diff(z_indices.astype(np.int64)) > 0):
        raise ValueError(f"Z indices are empty or not strictly ascending for {record.session_id}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Bundle contains NaN/Inf for {record.session_id}.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("slice_count", -1)) != len(values):
        raise ValueError(f"Bundle/metrics slice count mismatch for {record.session_id}.")
    return {"completion": completion, "metrics": metrics, "slice_count": len(values)}


def load_session_slices(output_root: str | Path, record: SessionRecord):
    bundle_path = session_artifact_dir(output_root, record) / "slice_bundle.npz"
    with np.load(bundle_path, allow_pickle=False) as archive:
        values = archive["slices"].astype(np.float32, copy=False)
        z_indices = archive["z_indices"].astype(np.int64, copy=False)
    return z_indices, values


def _manifest_row(key_index: int, record: SessionRecord, z: int) -> dict[str, object]:
    return {
        "key": f"{key_index:08}",
        "case_id": record.case_id,
        "session": record.session,
        "session_id": record.session_id,
        "z": int(z),
        "split": record.split,
        "channel_0": "FLAIR",
        "channel_1": "T1",
        "channel_2": "T2",
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_staging_lmdb(
    records: Iterable[SessionRecord],
    output_root: str | Path,
    *,
    processing_fingerprint: str,
    map_size: int = MAP_SIZE_BYTES,
) -> dict:
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LEMON LMDB creation requires lmdb.") from exc

    records = list(records)
    output_root = Path(output_root)
    lmdb_root = output_root / "MIP_lmdb"
    lmdb_root.mkdir(parents=True, exist_ok=True)
    report_path = lmdb_root / "build_report.json"
    manifest_path = lmdb_root / "manifest.building.csv"
    staging_paths = {split: lmdb_root / f"{split}.building" for split in ("train", "val")}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_selection = _record_selection_signature(records)
        actual_selection = {
            key: report.get(key) for key in expected_selection
        }
        if actual_selection != expected_selection:
            raise ValueError(
                "Existing LMDB build uses a different release selection; remove only the "
                f"explicit MIP_lmdb product directory before rebuilding: "
                f"actual={actual_selection}, expected={expected_selection}."
            )
        audit_staging_lmdb(records, output_root, processing_fingerprint=processing_fingerprint)
        return report
    occupied = [path for path in [manifest_path, *staging_paths.values()] if path.exists()]
    if occupied:
        raise FileExistsError(
            "Incomplete LMDB building artifacts already exist; remove only the explicit "
            f"MIP_lmdb build products before retrying: {occupied}"
        )

    for record in records:
        validate_session_bundle(output_root, record, processing_fingerprint=processing_fingerprint)

    manifest_rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    environments = {
        split: lmdb.open(str(path), map_size=int(map_size), subdir=True)
        for split, path in staging_paths.items()
    }
    transactions = {split: environment.begin(write=True) for split, environment in environments.items()}
    try:
        for split in ("train", "val"):
            key_index = 0
            for record in records:
                if record.split != split:
                    continue
                z_indices, values = load_session_slices(output_root, record)
                for z, value in zip(z_indices, values):
                    key = f"{key_index:08}"
                    transactions[split].put(
                        key.encode("ascii"),
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                        overwrite=False,
                    )
                    manifest_rows.append(_manifest_row(key_index, record, int(z)))
                    key_index += 1
            split_counts[split] = key_index
        for transaction in transactions.values():
            transaction.commit()
        transactions.clear()
    except Exception:
        for transaction in transactions.values():
            transaction.abort()
        raise
    finally:
        for environment in environments.values():
            environment.sync()
            environment.close()

    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "status": "BUILT_NOT_PUBLISHED",
        "map_size_bytes_per_split": int(map_size),
        "channel_order": list(MODALITY_ORDER),
        "split_counts": split_counts,
        "manifest": str(manifest_path.resolve()),
        "processing_fingerprint": processing_fingerprint,
        **_record_selection_signature(records),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit_staging_lmdb(records, output_root, processing_fingerprint=processing_fingerprint)
    return report


def audit_staging_lmdb(
    records: Iterable[SessionRecord],
    output_root: str | Path,
    *,
    processing_fingerprint: str,
) -> dict:
    return _audit_lmdb(
        records,
        output_root,
        processing_fingerprint=processing_fingerprint,
        published=False,
    )


def audit_published_lmdb(
    records: Iterable[SessionRecord],
    output_root: str | Path,
    *,
    processing_fingerprint: str,
) -> dict:
    return _audit_lmdb(
        records,
        output_root,
        processing_fingerprint=processing_fingerprint,
        published=True,
    )


def _audit_lmdb(
    records: Iterable[SessionRecord],
    output_root: str | Path,
    *,
    processing_fingerprint: str,
    published: bool,
) -> dict:
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LEMON LMDB audit requires lmdb.") from exc

    records = list(records)
    output_root = Path(output_root)
    lmdb_root = output_root / "MIP_lmdb"
    manifest_path = lmdb_root / ("manifest.csv" if published else "manifest.building.csv")
    rows = _read_manifest(manifest_path)
    session_splits = {record.session_id: record.split for record in records}
    if set(session_splits.values()) != {"train", "val"}:
        raise ValueError("Expected both train and val sessions before LMDB audit.")
    manifest_session_ids = {row["session_id"] for row in rows}
    if manifest_session_ids != set(session_splits):
        raise ValueError(
            "LMDB manifest session selection mismatch: "
            f"manifest_only={sorted(manifest_session_ids - set(session_splits))}, "
            f"records_only={sorted(set(session_splits) - manifest_session_ids)}."
        )
    if set(row["session_id"] for row in rows if row["split"] == "train") & set(
        row["session_id"] for row in rows if row["split"] == "val"
    ):
        raise ValueError("Train and validation session IDs overlap.")

    total = 0
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        expected_rows = [row for row in rows if row["split"] == split]
        expected_keys = [f"{index:08}" for index in range(len(expected_rows))]
        if [row["key"] for row in expected_rows] != expected_keys:
            raise ValueError(f"Manifest keys are not contiguous for split={split}.")
        pairs = [(int(record.csv_index), record.session_id) for record in records if record.split == split]
        csv_order = {session_id: index for index, session_id in pairs}
        order_values = [(csv_order[row["session_id"]], int(row["z"])) for row in expected_rows]
        if order_values != sorted(order_values):
            raise ValueError(f"Manifest is not in CSV-order then ascending-z for split={split}.")
        environment = lmdb.open(
            str(lmdb_root / (split if published else f"{split}.building")),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
        )
        try:
            with environment.begin(write=False) as transaction:
                if transaction.stat()["entries"] != len(expected_rows):
                    raise ValueError(f"LMDB/manifest entry count mismatch for split={split}.")
                for index, row in enumerate(expected_rows):
                    payload = transaction.get(row["key"].encode("ascii"))
                    if payload is None:
                        raise ValueError(f"Missing LMDB key {row['key']} in split={split}.")
                    value = pickle.loads(payload)
                    if not isinstance(value, np.ndarray) or value.shape != (3, 128, 128):
                        raise ValueError(f"Invalid LMDB shape at split={split}, key={row['key']}.")
                    if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                        raise ValueError(f"Invalid dtype/finite state at split={split}, key={row['key']}.")
                    if [row[f"channel_{channel}"] for channel in range(3)] != list(MODALITY_ORDER):
                        raise ValueError(f"Invalid channel order at split={split}, key={row['key']}.")
        finally:
            environment.close()
        split_counts[split] = len(expected_rows)
        total += len(expected_rows)
    audit = {
        "status": "PASS",
        "product_state": "PUBLISHED" if published else "BUILDING",
        "total_entries": total,
        "split_counts": split_counts,
        "train_val_disjoint": True,
        "keys_contiguous": True,
        "shape": [3, 128, 128],
        "dtype": "float32",
        "finite": True,
        "order": "source CSV order, then z ascending",
        "channel_order": list(MODALITY_ORDER),
        "processing_fingerprint": processing_fingerprint,
        **_record_selection_signature(records),
    }
    audit_path = lmdb_root / ("audit.json" if published else "audit.building.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def publish_lmdb(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    lmdb_root = output_root / "MIP_lmdb"
    audit_path = lmdb_root / "audit.building.json"
    if not audit_path.is_file():
        raise FileNotFoundError("LMDB building audit is missing; publication is refused.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("LMDB building audit did not PASS; publication is refused.")
    targets = [lmdb_root / "train", lmdb_root / "val", lmdb_root / "manifest.csv"]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"A published LMDB product already exists: {targets}")
    (lmdb_root / "train.building").replace(lmdb_root / "train")
    (lmdb_root / "val.building").replace(lmdb_root / "val")
    (lmdb_root / "manifest.building.csv").replace(lmdb_root / "manifest.csv")
    publication = {
        **audit,
        "status": "PUBLISHED",
        "product_state": "PUBLISHED",
        "published_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_path": str((lmdb_root / "train").resolve()),
        "val_path": str((lmdb_root / "val").resolve()),
        "manifest": str((lmdb_root / "manifest.csv").resolve()),
    }
    (lmdb_root / "publication.json").write_text(json.dumps(publication, indent=2), encoding="utf-8")
    return publication


__all__ = [
    "MAP_SIZE_BYTES",
    "audit_published_lmdb",
    "audit_staging_lmdb",
    "build_staging_lmdb",
    "load_session_slices",
    "publish_lmdb",
    "save_session_bundle",
    "session_artifact_dir",
    "validate_session_bundle",
]
