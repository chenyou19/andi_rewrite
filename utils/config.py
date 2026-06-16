"""scripts 共用的 YAML config 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config_schema import validate_config


def load_config(path: str | Path, *, validate: bool = True, mode: str | None = None) -> dict[str, Any]:
    """載入 YAML，並記錄來源路徑以利重現實驗。

    預設會做輕量 schema validation（validate=True）；回傳仍是 dict，現有
    builder / trainer / evaluator 的使用方式完全不變。需要跳過驗證時可傳
    validate=False。
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")
    data["_config_path"] = str(config_path)
    if validate:
        data = validate_config(data, mode=mode)
    return data


def print_config(config: dict[str, Any]) -> None:
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def summarize_components(**components: Any) -> dict[str, Any]:
    """若 component 提供 describe()，就用它產生精簡摘要。"""

    summary = {}
    for name, component in components.items():
        if hasattr(component, "describe"):
            summary[name] = component.describe()
        else:
            summary[name] = str(component)
    return summary
