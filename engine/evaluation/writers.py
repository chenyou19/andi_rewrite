"""Evaluation 結果的 CSV 寫出。

把原本 VolumeEvaluator.write_original_style_csv 的輸出邏輯獨立出來，方便之後
新增其他輸出格式而不必動到 evaluator。欄位順序與 row 結構與原版逐字一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in scores.items():
        if isinstance(value, dict):
            rows.append(
                {
                    "thr": key,
                    "value": value.get("value"),
                    "dice": value.get("dice"),
                    "sensitivity": value.get("sensitivity"),
                    "precision": value.get("precision"),
                }
            )
        else:
            rows.append({"thr": key, "value": value, "dice": None, "sensitivity": None, "precision": None})
    frame = pd.DataFrame(rows, columns=["thr", "value", "dice", "sensitivity", "precision"])
    frame.to_csv(path, index=False)
