"""scripts 共用的 YAML config 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """載入 YAML，並記錄來源路徑以利重現實驗。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(config_path)
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
