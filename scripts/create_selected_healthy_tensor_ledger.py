"""Build a read-only tensor hash ledger for selected healthy model-grid rows.

The ledger binds the source tensors used by a subsequent observed fit to the
immutable v3 manifests.  It streams each selected healthy row from its
canonical LMDB, hashes the exact float32 ``[3,128,128]`` tensor, and checks
duplicate underlying rows across the standalone and Mixed comparisons.  No
raw data or tensor copies are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.data.datasets.lmdb import LMDBSliceDataset  # noqa: E402


COMPARISONS = ("fomo45k", "mpi", "oasis3", "mixed")
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_source(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    provenance = row.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_dataset = str(metadata.get("underlying_source_dataset") or row.get("source_dataset") or "")
    source_split = str(metadata.get("underlying_source_split") or row.get("source_split") or "")
    source_key = str(metadata.get("underlying_source_key") or row.get("source_key") or "")
    lmdb_path = str(
        metadata.get("lmdb_path")
        or metadata.get("source_lmdb_path")
        or provenance.get("source_lmdb_path")
        or ""
    )
    local_key = str(row.get("source_key") or "")
    if not source_dataset or not source_split or not source_key or not lmdb_path or not local_key:
        raise ValueError(
            "Healthy manifest row lacks complete source identity: "
            f"dataset={source_dataset!r}, split={source_split!r}, key={source_key!r}, "
            f"lmdb={lmdb_path!r}, local_key={local_key!r}"
        )
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_key": source_key,
        "lmdb_path": lmdb_path,
        "local_source_key": local_key,
    }


def _close_dataset(dataset: Any) -> None:
    transaction = getattr(dataset, "txn", None)
    if transaction is not None:
        transaction.abort()
        dataset.txn = None
    environment = getattr(dataset, "env", None)
    if environment is not None:
        environment.close()
        dataset.env = None


def _load_rows(manifest_root: Path) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]]]:
    manifest_paths: dict[tuple[str, str], Path] = {}
    healthy_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        for split in SPLITS:
            path = manifest_root / comparison / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest_paths[(comparison, split)] = path
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                    if not isinstance(row, Mapping):
                        raise ValueError(f"Manifest row is not an object at {path}:{line_number}")
                    if int(row.get("label", -1)) != 0:
                        continue
                    source = _record_source(row)
                    healthy_rows.append(
                        {
                            "comparison": comparison,
                            "manifest_split": split,
                            "manifest_path": str(path.resolve()),
                            "manifest_row_index": len(healthy_rows),
                            "row": dict(row),
                            "source": source,
                        }
                    )
    return manifest_paths, healthy_rows


def _source_fingerprint(path: Path) -> dict[str, Any]:
    data = path / "data.mdb"
    normalization = path / "normalization.json"
    if not data.is_file():
        raise FileNotFoundError(data)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "data_mdb": {
            "path": str(data.resolve()),
            "size_bytes": int(data.stat().st_size),
            "mtime_ns": int(data.stat().st_mtime_ns),
        },
        "normalization": {
            "path": str(normalization.resolve()),
            "exists": normalization.is_file(),
        },
    }
    if normalization.is_file():
        result["normalization"]["sha256"] = _sha256(normalization)
        value = json.loads(normalization.read_text(encoding="utf-8"))
        result["normalization"]["type"] = value.get("type") if isinstance(value, Mapping) else None
    return result


def _read_tensor(dataset: LMDBSliceDataset, local_key: str) -> tuple[str, list[int], str]:
    try:
        index = int(local_key)
    except ValueError as exc:
        raise ValueError(f"LMDB source key is not an integer: {local_key!r}") from exc
    tensor = torch.as_tensor(dataset[index], dtype=torch.float32).contiguous()
    if tuple(tensor.shape) != (3, 128, 128):
        raise ValueError(f"Expected [3,128,128], found {tuple(tensor.shape)} for key {local_key}")
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Non-finite/non-float32 tensor for key {local_key}")
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return digest, list(tensor.shape), str(tensor.dtype).replace("torch.", "")


def build_ledger(*, manifest_root: Path, output_root: Path, progress_every: int = 250) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_paths, rows = _load_rows(manifest_root)
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        row = item["row"]
        source = item["source"]
        grouped[(source["source_dataset"], source["source_split"], source["source_key"], int(row["z"]))].append(item)

    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = output_root / "selected_healthy_source_tensor_ledger.partial.jsonl"
    progress_path = output_root / "selected_healthy_source_tensor_ledger_progress.json"
    if partial_path.exists():
        raise FileExistsError(f"Refusing to append to existing partial ledger: {partial_path}")

    source_paths = sorted({Path(item["source"]["lmdb_path"]).resolve() for item in rows})
    source_fingerprints = {str(path): _source_fingerprint(path) for path in source_paths}
    datasets: dict[str, LMDBSliceDataset] = {}
    ledger_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as handle:
            for canonical_key in sorted(grouped):
                memberships: list[dict[str, Any]] = []
                expected_digest: str | None = None
                expected_shape: list[int] | None = None
                expected_dtype: str | None = None
                for item in grouped[canonical_key]:
                    source = item["source"]
                    lmdb_text = str(Path(source["lmdb_path"]).resolve())
                    dataset = datasets.get(lmdb_text)
                    if dataset is None:
                        dataset = LMDBSliceDataset(lmdb_text, image_size=None)
                        datasets[lmdb_text] = dataset
                    try:
                        digest, shape, dtype = _read_tensor(dataset, source["local_source_key"])
                    except Exception as exc:
                        errors.append(
                            {
                                "comparison": item["comparison"],
                                "manifest_split": item["manifest_split"],
                                "source": source,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    if expected_digest is None:
                        expected_digest, expected_shape, expected_dtype = digest, shape, dtype
                    elif digest != expected_digest:
                        errors.append(
                            {
                                "canonical_key": list(canonical_key),
                                "comparison": item["comparison"],
                                "source": source,
                                "error": "duplicate underlying source rows have different tensor hashes",
                                "expected_sha256": expected_digest,
                                "actual_sha256": digest,
                            }
                        )
                    row = item["row"]
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
                    memberships.append(
                        OrderedDict(
                            (
                                ("comparison", item["comparison"]),
                                ("manifest_split", item["manifest_split"]),
                                ("target_split", str(row.get("split", ""))),
                                ("pair_id", str(row.get("pair_id", ""))),
                                ("participant_id", str(row.get("participant_id", ""))),
                                ("case_id", str(row.get("case_id", ""))),
                                ("z", int(row["z"])),
                                ("z_bin", int(row["z_bin"])),
                                ("local_source_dataset", str(row.get("source_dataset", ""))),
                                ("local_source_split", str(row.get("source_split", ""))),
                                ("local_source_key", str(row.get("source_key", ""))),
                                ("underlying_source_dataset", source["source_dataset"]),
                                ("underlying_source_split", source["source_split"]),
                                ("underlying_source_key", source["source_key"]),
                                ("lmdb_path", lmdb_text),
                                ("mixed_local_key", str(metadata.get("mixed_local_key", ""))),
                            )
                        )
                    )
                if expected_digest is None:
                    continue
                output_row = OrderedDict(
                    (
                        ("canonical_source_dataset", canonical_key[0]),
                        ("canonical_source_split", canonical_key[1]),
                        ("canonical_source_key", canonical_key[2]),
                        ("z", canonical_key[3]),
                        ("tensor_sha256", expected_digest),
                        ("shape", expected_shape),
                        ("dtype", expected_dtype),
                        ("memberships", memberships),
                    )
                )
                handle.write(json.dumps(output_row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                ledger_rows.append(output_row)
                completed += 1
                if progress_every > 0 and completed % int(progress_every) == 0:
                    _atomic_json(
                        progress_path,
                        {
                            "status": "RUNNING",
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "completed_unique_rows": completed,
                            "expected_unique_rows": len(grouped),
                            "healthy_manifest_rows": len(rows),
                            "error_count": len(errors),
                        },
                    )
    finally:
        for dataset in datasets.values():
            _close_dataset(dataset)

    if errors:
        _atomic_json(
            output_root / "selected_healthy_source_tensor_ledger_errors.json",
            {"status": "FAIL", "errors": errors, "error_count": len(errors)},
        )
        _atomic_json(
            progress_path,
            {
                "status": "FAIL",
                "completed_unique_rows": completed,
                "expected_unique_rows": len(grouped),
                "healthy_manifest_rows": len(rows),
                "error_count": len(errors),
            },
        )
        raise RuntimeError(f"Healthy source tensor ledger failed with {len(errors)} error(s).")

    final_path = output_root / "selected_healthy_source_tensor_ledger.jsonl"
    partial_path.replace(final_path)
    manifest_identity = {
        f"{comparison}/{split}": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "rows": sum(1 for item in rows if item["comparison"] == comparison and item["manifest_split"] == split),
        }
        for (comparison, split), path in manifest_paths.items()
    }
    summary = OrderedDict(
        (
            ("schema_version", 1),
            ("status", "PASS"),
            ("created_at_utc", datetime.now(timezone.utc).isoformat()),
            ("manifest_root", str(manifest_root.resolve())),
            ("manifest_identity", manifest_identity),
            ("comparisons", list(COMPARISONS)),
            ("healthy_manifest_rows", len(rows)),
            ("unique_underlying_source_rows", len(grouped)),
            ("ledger_rows", completed),
            ("duplicate_underlying_memberships", len(rows) - len(grouped)),
            ("source_lmdb_fingerprints", source_fingerprints),
            ("ledger_path", str(final_path.resolve())),
            ("ledger_sha256", _sha256(final_path)),
            ("source_tensor_contract", {"shape": [3, 128, 128], "dtype": "float32", "finite": True}),
            ("raw_dataset_bytes_copied", False),
            ("tensor_cache_bytes_copied", False),
            ("elapsed_seconds", float(time.perf_counter() - started)),
            ("training_started", False),
        )
    )
    _atomic_json(output_root / "selected_healthy_source_tensor_ledger_summary.json", summary)
    _atomic_json(
        progress_path,
        {
            "status": "COMPLETE",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_unique_rows": completed,
            "expected_unique_rows": len(grouped),
            "healthy_manifest_rows": len(rows),
            "error_count": 0,
            "ledger_sha256": summary["ledger_sha256"],
        },
    )
    return dict(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    summary = build_ledger(
        manifest_root=manifest_root.resolve(),
        output_root=output_root.resolve(),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
