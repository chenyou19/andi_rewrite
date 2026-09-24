"""Read and audit every selected FOMO registered slice without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    atomic_write_json,
    dataset_from_manifest,
    fit_shared_train_scalar,
    materialize_registered_dataset,
    read_jsonl_manifest,
)


SPLITS = ("train", "val", "test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_registered_materialization(*, manifest_root: Path, output_root: Path) -> dict[str, Any]:
    paths = {split: manifest_root / f"{split}.jsonl" for split in SPLITS}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Missing registered manifest under {manifest_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    split_audits: dict[str, Any] = {}
    train_images: list[torch.Tensor] = []
    train_keys: set[tuple[str, str, str, str, str, int]] = set()
    all_keys: set[tuple[str, str, str, str, str, int]] = set()
    for split in SPLITS:
        rows = read_jsonl_manifest(paths[split])
        dataset = dataset_from_manifest(
            paths[split],
            stage="registered",
            modalities=("flair", "t1", "t2"),
        )
        cached = materialize_registered_dataset(dataset)
        digest = hashlib.sha256()
        failures: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            sample = cached[index]
            image = sample.get("image") if isinstance(sample, dict) else None
            key = (
                str(row.get("source_dataset", "")),
                str(row.get("source_split", "")),
                str(row.get("source_key", "")),
                str(row.get("participant_id", "")),
                str(row.get("case_id", "")),
                int(row.get("z", 0)),
            )
            all_keys.add(key)
            if split == "train":
                train_keys.add(key)
            if not isinstance(image, torch.Tensor) or tuple(image.shape) != (3, 128, 128) or image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
                failures.append({"index": index, "key": list(key), "reason": "shape_dtype_or_finite_failure"})
                continue
            for field in ("participant_id", "pair_id", "case_id"):
                if str(sample.get(field, "")) != str(row.get(field, "")):
                    failures.append({"index": index, "key": list(key), "reason": f"sample_{field}_mismatch"})
            digest.update(repr(key).encode("utf-8"))
            digest.update(image.detach().cpu().contiguous().numpy().tobytes())
            if split == "train":
                train_images.append(image.detach().cpu().contiguous())
        split_audits[split] = {
            "records": len(rows),
            "unique_keys": len({(
                str(row.get("source_dataset", "")), str(row.get("source_split", "")), str(row.get("source_key", "")),
                str(row.get("participant_id", "")), str(row.get("case_id", "")), int(row.get("z", 0)),
            ) for row in rows}),
            "tensor_shape": [3, 128, 128],
            "tensor_dtype": "float32",
            "finite": not failures,
            "tensor_sha256": digest.hexdigest(),
            "failures": failures[:50],
            "failure_count": len(failures),
        }
    if not train_images:
        raise RuntimeError("No train images were materialized for shared scalar fitting.")
    train_stack = torch.stack(train_images, dim=0)
    shared_scalar = float(fit_shared_train_scalar(train_stack))
    expected_counts = {"train": len(read_jsonl_manifest(paths["train"])), "val": len(read_jsonl_manifest(paths["val"])), "test": len(read_jsonl_manifest(paths["test"]))}
    status = "PASS" if all(int(split_audits[split]["failure_count"]) == 0 and int(split_audits[split]["records"]) == expected_counts[split] for split in SPLITS) else "FAIL"
    audit = {
        "schema_version": 1,
        "status": status,
        "protocol": "fomo45k_v3_registered_reader_materialization_v1",
        "manifest_root": str(manifest_root.resolve()),
        "manifest_sha256": {split: _sha256_file(paths[split]) for split in SPLITS},
        "split_audits": split_audits,
        "total_records": sum(expected_counts.values()),
        "total_unique_keys": len(all_keys),
        "train_unique_keys": len(train_keys),
        "all_three_modalities": True,
        "z_range_checked_by_registered_reader": True,
        "shared_train_scalar": shared_scalar,
        "shared_train_scalar_fit_source": "train split only",
        "shared_train_scalar_applied_to": ["train", "val", "test"],
        "training_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_root / "registered_materialization_audit.json", audit, overwrite=False)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = (args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root).resolve()
    output_root = (args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root).resolve()
    result = audit_registered_materialization(manifest_root=manifest_root, output_root=output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
