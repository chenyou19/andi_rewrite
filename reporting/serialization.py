from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_get(mapping: Any, path: str | list[str], default: Any = None) -> Any:
    """Read a nested config value without raising when a key is missing."""

    parts = path.split(".") if isinstance(path, str) else path
    current = mapping
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def snapshot_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of the full run config."""

    return _json_safe(config)


def _format_time(value: datetime | None) -> str | None:
    return value.astimezone().isoformat(timespec="seconds") if isinstance(value, datetime) else None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return round((end - start).total_seconds(), 3)


def _format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return "`" + json.dumps(_json_safe(value), ensure_ascii=False) + "`"
    return str(value).replace("|", "\\|")


def _dict_table(values: dict[str, Any]) -> str:
    rows = ["| key | value |", "|---|---|"]
    for key, value in values.items():
        rows.append(f"| {key} | {_format_value(value)} |")
    return "\n".join(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _format_time(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
