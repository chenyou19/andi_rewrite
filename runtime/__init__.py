"""Runtime/builder helpers shared by the training 與 evaluation CLI scripts。

把 config -> components 的建構邏輯集中在這裡，讓 scripts/train.py 與
scripts/eval.py 只保留 CLI 解析與高階流程，避免兩個 script 各自演化出
不同步的 device / accelerator / checkpoint 邏輯。
"""

from .builders import (
    build_accelerator,
    build_detector_from_config,
    build_evaluator_from_config,
    build_trainer_from_config,
    configure_training_backend,
    load_model_if_configured,
    resolve_device,
)

__all__ = [
    "build_accelerator",
    "build_detector_from_config",
    "build_evaluator_from_config",
    "build_trainer_from_config",
    "configure_training_backend",
    "load_model_if_configured",
    "resolve_device",
]
