"""Freeze provenance for the pre-v3 domain-classifier substudy.

This is intentionally metadata-only after the initial file copies.  It records
the local source/dependency state and hashes of the historical calibration
artifacts without opening or rewriting any dataset files.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs" / "diagnostics" / "domain_classifier" / "model_grid_v3" / "provenance_freeze"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files_under(path: Path):
    if path.is_file():
        yield path
    elif path.exists():
        yield from (p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def file_record(path: Path, root: Path) -> dict:
    st = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": st.st_size,
        "sha256": sha256(path),
        "mtime_ns": st.st_mtime_ns,
    }


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"ERROR: {type(exc).__name__}: {exc}"


def main() -> None:
    FREEZE.mkdir(parents=True, exist_ok=True)
    source_files: list[dict] = []
    for directory in ("source", "calibration_replay"):
        base = FREEZE / directory
        for path in files_under(base):
            source_files.append(file_record(path, FREEZE))
    source_files.sort(key=lambda x: x["path"])

    dependency = {
        "interpreter_requested": r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe",
        "executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "pytest", "nibabel", "lmdb"):
        try:
            mod = __import__(name)
            dependency["packages"][name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            dependency["packages"][name] = f"ERROR: {type(exc).__name__}: {exc}"
    dependency["pip_freeze"] = command_output([sys.executable, "-m", "pip", "freeze"])
    (FREEZE / "dependency_inventory.json").write_text(json.dumps(dependency, indent=2) + "\n", encoding="utf-8")

    git = {
        "status_porcelain": command_output(["git", "status", "--porcelain=v1"]),
        "tracked_files": command_output(["git", "ls-files"]),
        "head": command_output(["git", "rev-parse", "HEAD"]),
    }
    (FREEZE / "git_state.json").write_text(json.dumps(git, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "pre-v3 provenance freeze; no source dataset bytes copied or modified",
        "workspace": str(ROOT),
        "freeze_root": str(FREEZE),
        "source_files": source_files,
        "source_file_count": len(source_files),
        "historical_calibration": {
            "source_root": str(ROOT / "outputs" / "diagnostics" / "domain_classifier" / "fomo45k_calibration_smallcnn"),
            "replay_snapshot": str(FREEZE / "calibration_replay"),
            "full_source_remains_immutable_at": str(ROOT / "outputs" / "diagnostics" / "domain_classifier" / "fomo45k_calibration_smallcnn"),
        },
        "dataset_policy": {
            "dataset_bytes_copied": False,
            "healthy_raw_volumes_opened": False,
            "source_dataset_files_modified": False,
        },
    }
    (FREEZE / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_root": str(FREEZE), "file_count": len(source_files), "manifest": str(FREEZE / "snapshot_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
