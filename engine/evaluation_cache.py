from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


CACHE_SCHEMA_VERSION = 1


def stable_fingerprint(payload: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(value: Any) -> dict[str, Any] | None:
    """Describe a configured file without reading a potentially huge payload."""

    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except (FileNotFoundError, OSError):
        return {"path": str(path), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class DiskEvaluationCache:
    """Atomic per-subject anomaly-map cache with a resumable manifest."""

    def __init__(
        self,
        directory: str | Path,
        *,
        raw_fingerprint: str,
        score_fingerprint: str,
        resume: bool = True,
        keep_on_success: bool = True,
    ) -> None:
        self.root = Path(directory).expanduser().resolve()
        current = Path.cwd().resolve()
        if self.root == current or self.root.parent == self.root:
            raise ValueError(
                f"Refusing to use a broad directory as an evaluation cache: {self.root}"
            )
        self.manifest_path = self.root / "manifest.json"
        self.raw_directory = self.root / "raw"
        self.label_directory = self.root / "labels"
        self.mf_directory = self.root / "mf"
        self.staging_directory = self.root / "staging"
        self.sort_directory = self.root / "external_sort"
        self.resume = bool(resume)
        self.keep_on_success = bool(keep_on_success)
        self.raw_fingerprint = str(raw_fingerprint)
        self.score_fingerprint = str(score_fingerprint)
        self.manifest = self._open_or_create_manifest()

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self.manifest["entries"]

    @property
    def labels_available(self) -> bool:
        return bool(self.entries) and all(bool(entry.get("has_label")) for entry in self.entries)

    @property
    def total_voxels(self) -> int:
        return sum(int(entry.get("numel", 0)) for entry in self.entries)

    def _open_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Evaluation cache schema mismatch. Use a new cache.directory; "
                    f"found {payload.get('schema_version')!r}, expected {CACHE_SCHEMA_VERSION}."
                )
            if payload.get("raw_fingerprint") != self.raw_fingerprint:
                raise RuntimeError(
                    "Evaluation cache raw-score fingerprint does not match this run. "
                    "Use a new cache.directory; the existing cache was not modified."
                )
            if payload.get("score_fingerprint") != self.score_fingerprint:
                raise RuntimeError(
                    "Evaluation cache postprocess fingerprint does not match this run. "
                    "Use a new cache.directory; the existing cache was not modified."
                )
            if not self.resume and payload.get("entries"):
                raise RuntimeError(
                    "evaluation.cache.resume is false but the cache already contains subjects. "
                    "Use an empty cache.directory or enable resume."
                )
            self._ensure_directories()
            return payload

        if self.root.exists():
            existing = list(self.root.iterdir())
            if existing:
                raise RuntimeError(
                    "Evaluation cache directory is non-empty but has no manifest.json: "
                    f"{self.root}. Use a new directory; existing files were not modified."
                )
        self._ensure_directories()
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "raw_fingerprint": self.raw_fingerprint,
            "score_fingerprint": self.score_fingerprint,
            "entries": [],
            "collection_complete": False,
            "raw_score_bounds": None,
            "mf_score_bounds": None,
        }
        self.manifest = payload
        self._write_manifest()
        return payload

    def _ensure_directories(self) -> None:
        for path in (
            self.root,
            self.raw_directory,
            self.label_directory,
            self.mf_directory,
            self.staging_directory,
            self.sort_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        temporary.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _atomic_save_array(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _validate_array(path: Path, shape: list[int], dtype: str) -> None:
        if not path.exists():
            raise RuntimeError(f"Evaluation cache file is missing: {path}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(f"Evaluation cache file cannot be read: {path}") from exc
        if list(array.shape) != [int(value) for value in shape]:
            raise RuntimeError(
                f"Evaluation cache shape mismatch for {path}: "
                f"found {list(array.shape)}, expected {shape}."
            )
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise RuntimeError(
                f"Evaluation cache dtype mismatch for {path}: "
                f"found {array.dtype}, expected {dtype}."
            )

    def cached_entry(
        self,
        index: int,
        *,
        subject_id: str,
        has_label: bool,
    ) -> dict[str, Any] | None:
        if index >= len(self.entries):
            return None
        entry = self.entries[index]
        if int(entry.get("index", -1)) != index or entry.get("subject_id") != subject_id:
            raise RuntimeError(
                "Evaluation cache subject order does not match the current dataset at "
                f"index {index}: cached={entry.get('subject_id')!r}, current={subject_id!r}."
            )
        if bool(entry.get("has_label")) != bool(has_label):
            raise RuntimeError(
                f"Evaluation cache label availability changed for subject {subject_id!r}."
            )
        self._validate_array(
            self.root / entry["raw_file"],
            entry["shape"],
            entry["raw_dtype"],
        )
        if has_label:
            self._validate_array(
                self.root / entry["label_file"],
                entry["label_shape"],
                entry["label_dtype"],
            )
        return entry

    def store_raw_entry(
        self,
        index: int,
        *,
        subject_id: str,
        raw: torch.Tensor | np.ndarray,
        label: torch.Tensor | np.ndarray | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if index != len(self.entries):
            raise RuntimeError(
                "New cache entries must be written in dataset order: "
                f"received index {index}, next index is {len(self.entries)}."
            )
        raw_array = np.nan_to_num(
            np.asarray(raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else raw),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        raw_name = f"{index:06d}_{subject_id}.npy"
        raw_path = self.raw_directory / raw_name
        self._atomic_save_array(raw_path, raw_array)

        entry: dict[str, Any] = {
            "index": int(index),
            "subject_id": subject_id,
            "metadata": metadata,
            "raw_file": str(raw_path.relative_to(self.root)),
            "shape": [int(value) for value in raw_array.shape],
            "raw_dtype": str(raw_array.dtype),
            "numel": int(raw_array.size),
            "raw_min": float(raw_array.min()) if raw_array.size else 0.0,
            "raw_max": float(raw_array.max()) if raw_array.size else 0.0,
            "has_label": label is not None,
        }
        if label is not None:
            label_array = np.asarray(
                label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label,
                dtype=np.bool_,
            )
            if label_array.shape != raw_array.shape:
                raise ValueError(
                    f"Label shape {label_array.shape} does not match anomaly-map shape "
                    f"{raw_array.shape} for subject {subject_id!r}."
                )
            label_path = self.label_directory / raw_name
            self._atomic_save_array(label_path, label_array)
            entry.update(
                {
                    "label_file": str(label_path.relative_to(self.root)),
                    "label_shape": [int(value) for value in label_array.shape],
                    "label_dtype": str(label_array.dtype),
                }
            )
        self.entries.append(entry)
        self._write_manifest()
        return entry

    def finish_collection(self, subject_count: int) -> None:
        if len(self.entries) != int(subject_count):
            raise RuntimeError(
                "Evaluation cache contains a different number of subjects than the current "
                f"dataset: cached={len(self.entries)}, current={subject_count}."
            )
        self.manifest["collection_complete"] = True
        self.manifest["subjects"] = int(subject_count)
        self.manifest["total_voxels"] = int(self.total_voxels)
        self._write_manifest()

    def load_raw(self, entry: dict[str, Any]) -> torch.Tensor:
        array = np.array(
            np.load(self.root / entry["raw_file"], mmap_mode="r", allow_pickle=False),
            copy=True,
        )
        return torch.from_numpy(array).float()

    def load_label(self, entry: dict[str, Any]) -> torch.Tensor | None:
        if not entry.get("has_label"):
            return None
        array = np.array(
            np.load(self.root / entry["label_file"], mmap_mode="r", allow_pickle=False),
            copy=True,
        )
        return torch.from_numpy(array).bool()

    def _valid_product(self, entry: dict[str, Any], key: str) -> bool:
        product = entry.get(key)
        if not isinstance(product, dict):
            return False
        self._validate_array(
            self.root / product["file"],
            product["shape"],
            product["dtype"],
        )
        return True

    def has_mf_pre(self, entry: dict[str, Any]) -> bool:
        return self._valid_product(entry, "mf_pre")

    def has_mf(self, entry: dict[str, Any]) -> bool:
        return self._valid_product(entry, "mf")

    def store_product(
        self,
        entry: dict[str, Any],
        *,
        key: str,
        tensor: torch.Tensor,
    ) -> None:
        if key not in {"mf_pre", "mf"}:
            raise ValueError(f"Unknown evaluation cache product: {key}")
        array = np.nan_to_num(
            tensor.detach().cpu().numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        directory = self.staging_directory if key == "mf_pre" else self.mf_directory
        path = directory / f"{int(entry['index']):06d}_{entry['subject_id']}.npy"
        self._atomic_save_array(path, array)
        entry[key] = {
            "file": str(path.relative_to(self.root)),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "min": float(array.min()) if array.size else 0.0,
            "max": float(array.max()) if array.size else 0.0,
        }
        self._write_manifest()

    def load_product(self, entry: dict[str, Any], key: str) -> torch.Tensor:
        if not self._valid_product(entry, key):
            raise RuntimeError(
                f"Evaluation cache product {key!r} is unavailable for {entry['subject_id']!r}."
            )
        array = np.array(
            np.load(self.root / entry[key]["file"], mmap_mode="r", allow_pickle=False),
            copy=True,
        )
        return torch.from_numpy(array).float()

    def remove_mf_pre(self, entry: dict[str, Any]) -> None:
        product = entry.get("mf_pre")
        if not isinstance(product, dict):
            return
        path = self.root / product["file"]
        if path.exists():
            path.unlink()
        entry.pop("mf_pre", None)
        self._write_manifest()

    def get_bounds(self, key: str) -> tuple[float, float] | None:
        value = self.manifest.get(key)
        if not isinstance(value, dict):
            return None
        return float(value["min"]), float(value["max"])

    def set_bounds(self, key: str, minimum: float, maximum: float) -> None:
        self.manifest[key] = {"min": float(minimum), "max": float(maximum)}
        self._write_manifest()

    def cleanup_after_success(self) -> None:
        if self.keep_on_success:
            return
        resolved = self.root.resolve()
        if resolved == Path.cwd().resolve() or resolved.parent == resolved:
            raise RuntimeError(f"Refusing to remove broad cache path: {resolved}")
        shutil.rmtree(resolved)
