"""Freeze the source and metadata bytes used by the next Stage-A fit.

This is a small, append-only provenance helper.  It copies only source code,
configuration, manifests, and audit metadata; it never copies raw scans or
selected tensor caches and never changes an existing run artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path, relative: str) -> Iterable[Path]:
    base = root / relative
    if base.is_file():
        yield base
    elif base.is_dir():
        yield from sorted(
            path
            for path in base.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )


def freeze(output_root: Path, build_root: Path, input_binding_root: Path | None = None) -> dict:
    freeze_root = output_root / "provenance_freeze_current_v1"
    source_root = freeze_root / "source"
    if freeze_root.exists():
        raise FileExistsError(f"Refusing to overwrite provenance freeze: {freeze_root}")
    source_root.mkdir(parents=True)

    relative_files: list[str] = [
        "configs/domain_classifier.yaml",
        "domain_classifier",
        "data/domain_classifier",
        "scripts/run_domain_classifier.py",
        "scripts/run_domain_classifier_controls.py",
        "scripts/run_domain_classifier_controls_threaded.py",
        "scripts/run_domain_classifier_calibration.py",
        "scripts/run_domain_classifier_calibration_threaded.py",
        "scripts/audit_domain_classifier_calibration.py",
        "scripts/evaluate_immutable_test_label_sanity.py",
        "scripts/enrich_fomo_registered_paths_v3.py",
        "scripts/audit_registered_fomo_materialization.py",
        "scripts/audit_model_grid_v3_final.py",
        "scripts/freeze_stage_a_provenance.py",
    ]
    artifact_files = [
        "training_protocol.json",
        "training_protocol_manifest.json",
        "training_protocol_amendment_20260917.json",
        "training_protocol_amendment_manifest.json",
        "protocol.json",
        "source_fingerprints.json",
        "build_summary.json",
        "brats_candidate_inventory.json",
        "final_external_audit_v2.json",
        "audits/fomo45k.json",
        "audits/mixed.json",
        "fomo45k_registered_paths_v1/registered_paths_audit.json",
        "fomo45k_registered_paths_v1/registered_paths_access_audit.json",
        "fomo45k_registered_paths_v1/registered_materialization_audit.json",
    ]
    for split in ("train", "val", "test"):
        artifact_files.append(f"manifests/fomo45k/{split}.jsonl")
        artifact_files.append(f"fomo45k_registered_paths_v1/manifests/fomo45k/{split}.jsonl")

    records: list[dict] = []
    missing: list[str] = []

    def copy_one(path: Path, relative_destination: str) -> None:
        destination = source_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append(
            {
                "path": relative_destination.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    for relative in relative_files:
        paths = list(_files(ROOT, relative))
        if not paths:
            missing.append(relative)
            continue
        for path in paths:
            copy_one(path, path.relative_to(ROOT).as_posix())
    for relative in artifact_files:
        path = build_root / relative
        if not path.is_file():
            missing.append(str(path))
            continue
        copy_one(path, (Path("build_artifacts") / relative).as_posix())
    if input_binding_root is not None:
        binding_files = (
            "selected_healthy_source_tensor_ledger.jsonl",
            "selected_healthy_source_tensor_ledger_summary.json",
            "selected_healthy_source_tensor_ledger_progress.json",
        )
        for relative in binding_files:
            path = input_binding_root / relative
            if not path.is_file():
                missing.append(str(path))
                continue
            copy_one(path, (Path("input_binding") / relative).as_posix())

    dependency: dict = {
        "interpreter_requested": r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe",
        "executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "pytest", "nibabel", "lmdb"):
        try:
            module = __import__(name)
            dependency["packages"][name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment dependent
            dependency["packages"][name] = f"ERROR: {type(exc).__name__}: {exc}"

    manifest = {
        "schema_version": 1,
        "protocol": "stage_a_fomo45k_v3_source_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT),
        "output_root": str(output_root.resolve()),
        "build_root": str(build_root.resolve()),
        "input_binding_root": str(input_binding_root.resolve()) if input_binding_root is not None else None,
        "raw_dataset_bytes_copied": False,
        "tensor_cache_bytes_copied": False,
        "records": sorted(records, key=lambda item: item["path"]),
        "missing": missing,
        "status": "PASS" if not missing else "INCOMPLETE",
        "training_started_by_this_helper": False,
    }
    (freeze_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (freeze_root / "dependency_inventory.json").write_text(json.dumps(dependency, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--input-binding-root", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    build_root = args.build_root if args.build_root.is_absolute() else ROOT / args.build_root
    input_binding_root = args.input_binding_root
    if input_binding_root is not None and not input_binding_root.is_absolute():
        input_binding_root = ROOT / input_binding_root
    result = freeze(
        output_root.resolve(),
        build_root.resolve(),
        input_binding_root.resolve() if input_binding_root is not None else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
