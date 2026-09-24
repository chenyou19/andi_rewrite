"""Audit that a completed observed fit used the frozen selected input tensors.

This validator runs after the fit and therefore does not pretend that the
ledger was consulted by the historical process.  It re-reads the same v3
manifests, hashes every healthy FOMO row against the prelaunch ledger, and
recomputes the calibration runner's aggregate source-cache digest over all
5,896 rows.  Equality with the digest saved by the observed process binds the
frozen source rows to the actual model-input bytes used by that process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    _record_mapping,
    dataset_from_manifest,
)
from scripts.run_domain_classifier_calibration import _row_key  # noqa: E402


SPLITS = ("train", "val", "test")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return (
        str(metadata.get("underlying_source_dataset") or row.get("source_dataset") or ""),
        str(metadata.get("underlying_source_split") or row.get("source_split") or ""),
        str(metadata.get("underlying_source_key") or row.get("source_key") or ""),
        int(row.get("z", 0)),
    )


def _ledger_key(item: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(item.get("canonical_source_dataset", "")),
        str(item.get("canonical_source_split", "")),
        str(item.get("canonical_source_key", "")),
        int(item.get("z", 0)),
    )


def _read_ledger(path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Ledger row {path}:{line_number} is not an object")
            key = _ledger_key(value)
            if key in result:
                raise ValueError(f"Duplicate ledger key: {key!r}")
            result[key] = dict(value)
    return result


def audit_binding(
    *,
    observed_root: Path,
    manifest_root: Path,
    ledger_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    observed_digest_path = observed_root / "source_cache_digest.json"
    observed_result_path = observed_root / "observed" / "result.json"
    ledger_path = ledger_root / "selected_healthy_source_tensor_ledger.jsonl"
    ledger_summary_path = ledger_root / "selected_healthy_source_tensor_ledger_summary.json"
    required = (observed_digest_path, observed_result_path, ledger_path, ledger_summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input-binding audit input(s): {missing!r}")

    observed_digest = json.loads(observed_digest_path.read_text(encoding="utf-8"))
    observed_result = json.loads(observed_result_path.read_text(encoding="utf-8"))
    ledger_summary = json.loads(ledger_summary_path.read_text(encoding="utf-8"))
    ledger = _read_ledger(ledger_path)
    expected_ledger_sha = str(ledger_summary.get("ledger_sha256", ""))
    actual_ledger_sha = _sha256(ledger_path)
    if expected_ledger_sha != actual_ledger_sha:
        raise ValueError("Selected healthy ledger SHA does not match its summary.")

    aggregate = hashlib.sha256()
    aggregate_rows = 0
    healthy_rows = 0
    healthy_keys: set[tuple[str, str, str, int]] = set()
    label_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    fomo_manifest_identity: dict[str, Any] = {}

    for split in SPLITS:
        manifest_path = manifest_root / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        fomo_manifest_identity[split] = {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        }
        dataset = dataset_from_manifest(
            manifest_path,
            stage="final",
            modalities=("flair", "t1", "t2"),
            shared_train_scalar=None,
        )
        records = getattr(dataset, "records", None)
        if records is None:
            raise TypeError(f"Dataset for {manifest_path} has no records property")
        for index, record in enumerate(records):
            row = _record_mapping(record)
            sample = dataset[index]
            if not isinstance(sample, Mapping) or "image" not in sample:
                raise TypeError(f"Dataset sample {split}[{index}] lacks image mapping")
            image = sample["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            image = image.detach().cpu().contiguous().float()
            if tuple(image.shape) != (3, 128, 128) or not bool(torch.isfinite(image).all()):
                raise ValueError(f"Invalid observed input tensor at {split}[{index}]: {tuple(image.shape)}")
            aggregate.update(split.encode("utf-8"))
            aggregate.update(repr(_row_key(record)).encode("utf-8"))
            aggregate.update(image.numpy().tobytes())
            aggregate_rows += 1
            label = int(row.get("label", -1))
            label_counts[label] += 1
            split_counts[split] += 1
            shape_counts[str(list(image.shape))] += 1
            dtype_counts[str(image.dtype)] += 1
            if label == 0:
                healthy_rows += 1
                canonical_key = _canonical_source(row)
                if canonical_key in healthy_keys:
                    errors.append({"split": split, "index": index, "error": "duplicate healthy canonical key", "key": list(canonical_key)})
                healthy_keys.add(canonical_key)
                ledger_item = ledger.get(canonical_key)
                if ledger_item is None:
                    errors.append({"split": split, "index": index, "error": "healthy key absent from frozen ledger", "key": list(canonical_key)})
                else:
                    digest = hashlib.sha256(image.numpy().tobytes()).hexdigest()
                    if digest != str(ledger_item.get("tensor_sha256", "")):
                        errors.append({"split": split, "index": index, "error": "healthy tensor SHA mismatch", "key": list(canonical_key), "expected": ledger_item.get("tensor_sha256"), "actual": digest})
                    memberships = ledger_item.get("memberships", [])
                    matching = [item for item in memberships if isinstance(item, Mapping) and item.get("comparison") == "fomo45k" and item.get("manifest_split") == split]
                    if len(matching) != 1:
                        errors.append({"split": split, "index": index, "error": "ledger membership join is not unique", "key": list(canonical_key), "matching_memberships": len(matching)})

    aggregate_sha = aggregate.hexdigest()
    expected_aggregate_sha = str(observed_digest.get("sha256", ""))
    expected_rows = int(observed_digest.get("rows", -1))
    if aggregate_rows != expected_rows:
        errors.append({"error": "aggregate row count mismatch", "expected": expected_rows, "actual": aggregate_rows})
    if aggregate_sha != expected_aggregate_sha:
        errors.append({"error": "observed source cache digest mismatch", "expected": expected_aggregate_sha, "actual": aggregate_sha})
    expected_healthy_fomo = sum(
        1
        for split in SPLITS
        for item in ledger.values()
        if any(
            isinstance(membership, Mapping)
            and membership.get("comparison") == "fomo45k"
            and membership.get("manifest_split") == split
            for membership in item.get("memberships", [])
        )
    )
    if healthy_rows != expected_healthy_fomo:
        errors.append({"error": "FOMO healthy ledger membership count mismatch", "expected": expected_healthy_fomo, "actual": healthy_rows})

    result = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "post-fit external input binding validation",
        "training_started_before_audit": True,
        "observed_root": str(observed_root.resolve()),
        "observed_result_fingerprint": observed_result.get("run_fingerprint"),
        "observed_source_cache_digest": observed_digest,
        "recomputed_source_cache_digest": {
            "rows": aggregate_rows,
            "sha256": aggregate_sha,
            "stage": "final",
            "materialized_once_equivalent": True,
        },
        "digest_equal": aggregate_rows == expected_rows and aggregate_sha == expected_aggregate_sha,
        "fomo_manifest_identity": fomo_manifest_identity,
        "frozen_healthy_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": actual_ledger_sha,
            "summary_path": str(ledger_summary_path.resolve()),
            "healthy_ledger_rows": len(ledger),
            "fomo_healthy_rows_checked": healthy_rows,
            "fomo_healthy_keys_checked": len(healthy_keys),
        },
        "observed_rows_checked": aggregate_rows,
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "split_counts": {key: int(value) for key, value in sorted(split_counts.items())},
        "shape_counts": dict(shape_counts),
        "dtype_counts": dict(dtype_counts),
        "braTS_rows_checked_through_canonical_reader": int(label_counts.get(1, 0)),
        "errors": errors,
        "source_cache_code_contract": "same split order, _row_key, contiguous float32 image bytes as calibration _cache_digest",
        "raw_dataset_bytes_copied": False,
        "tensor_cache_bytes_copied": False,
    }
    _atomic_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    values = {
        "observed_root": args.observed_root,
        "manifest_root": args.manifest_root,
        "ledger_root": args.ledger_root,
        "output_path": args.output,
    }
    values = {
        key: value if value.is_absolute() else REPO_ROOT / value
        for key, value in values.items()
    }
    result = audit_binding(**{key: value.resolve() for key, value in values.items()})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
