"""Model registry：用註冊取代大型 if/elif，新增 model 只需註冊 builder。

把「name -> builder」的對照表與註冊機制獨立放在這裡，factory 只負責登錄各個
model builder 並提供 build_model 入口，彼此不會 circular import。
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

ModelBuilder = Callable[[dict[str, Any]], nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(*names: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """把 builder function 註冊到一個或多個 model type alias 底下。"""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        for name in names:
            MODEL_REGISTRY[name.lower()] = builder
        return builder

    return decorator
