"""Small, semantically shared helpers for dataset adapters.

Adapter-specific discovery, geometry, loading, and resize behaviour intentionally
remain with the owning adapter modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _subject_file_path(subject_dir: Path, file_stem: str, suffix: str, separator: str = "_") -> Path:
    return subject_dir / f"{file_stem}{separator}{suffix}.nii.gz"


def _as_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _shape_text(shape: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(int(item)) for item in shape)
