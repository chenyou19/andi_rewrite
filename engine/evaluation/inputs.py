"""Evaluation batch and metadata normalization helpers.

These functions are intentionally independent of ``VolumeEvaluator`` so both
in-memory and streaming collection use the same interpretation of a batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def split_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    """Return image, optional label, and optional metadata from a batch."""

    if isinstance(batch, dict):
        image = batch.get("image")
        label = batch.get("label", batch.get("mask"))
        metadata = batch.get("metadata", batch.get("meta"))
        if image is None:
            raise ValueError("Dictionary batches must contain an 'image' key.")
        return image, label, metadata
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Empty evaluation batch.")
        image = batch[0]
        label = batch[1] if len(batch) > 1 else None
        metadata = batch[2] if len(batch) > 2 else None
        return image, label, metadata
    raise TypeError(f"Unsupported evaluation batch type: {type(batch)!r}")


def metadata_items(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    """Uncollate metadata into its corresponding per-subject dictionaries."""

    if metadata is None:
        return [{} for _ in range(batch_size)]
    return [metadata_item(metadata, index, batch_size) for index in range(batch_size)]


def metadata_item(value: Any, index: int, batch_size: int) -> Any:
    """Extract one value from recursively collated DataLoader metadata."""

    if isinstance(value, dict):
        return {key: metadata_item(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        item = value
        if item.ndim > 0 and item.shape[0] == batch_size:
            item = item[index]
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        item = value
        if item.ndim > 0 and item.shape[0] == batch_size:
            item = item[index]
        return item.item() if item.ndim == 0 else item.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return metadata_item(value[index], 0, 1)
        return [metadata_item(item, index, batch_size) for item in value]
    return value


def truthy(value: Any, default: bool = True) -> bool:
    """Interpret legacy metadata label flags without changing their semantics."""

    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def safe_subject_id(value: Any) -> str:
    """Make a stable filesystem-safe subject identifier."""

    text = str(value or "subject").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe or "subject"


def shape_from_text(value: Any) -> tuple[int, ...] | None:
    """Parse legacy comma-separated shape metadata when it is present."""

    if isinstance(value, str) and value:
        try:
            return tuple(int(part) for part in value.split(",") if part)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return None
    return None


def has_label(label: torch.Tensor | None, metadata: dict[str, Any]) -> bool:
    """Apply the existing tensor-and-metadata label availability policy."""

    if label is None:
        return False
    return truthy(metadata.get("has_label"), default=True)
