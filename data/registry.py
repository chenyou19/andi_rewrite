"""Dataset registry：用註冊取代 build_dataset 的 if 判斷鏈。

把「name -> builder」對照表與註冊機制放在這裡，datasets 模組只負責登錄各個
dataset builder 並提供 build_dataset 入口，彼此不會 circular import。
"""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset

DatasetBuilder = Callable[[dict[str, Any]], Dataset]

DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(*names: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    """把 builder function 註冊到一個或多個 dataset type alias 底下。"""

    def decorator(builder: DatasetBuilder) -> DatasetBuilder:
        for name in names:
            DATASET_REGISTRY[name.lower()] = builder
        return builder

    return decorator
