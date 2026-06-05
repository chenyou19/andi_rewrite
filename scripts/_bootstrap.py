"""讓 scripts 可同時以檔案路徑或 python -m module 形式執行。"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
