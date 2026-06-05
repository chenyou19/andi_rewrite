"""local 與 accelerate training 共用的 checkpoint 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """當模型被 DDP/Accelerate wrap 時，取回底層原始 model。"""

    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Any = None,
    ema_model: nn.Module | None = None,
    ema_state: dict | None = None,
    config: dict | None = None,
    accelerator: Any = None,
) -> None:
    """儲存 resume training 或 EMA checkpoint evaluation 所需的完整狀態。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config or {},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema_model is not None:
        payload["ema_model"] = unwrap_model(ema_model).state_dict()
    if ema_state is not None:
        payload["ema"] = ema_state

    if accelerator is not None:
        # accelerator.save 會處理 distributed rank 協調與 device-safe serialization；
        # local single-process 則使用 torch.save 即可。
        accelerator.save(payload, str(path))
    else:
        torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema_model: nn.Module | None = None,
    ema: Any = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """載入 checkpoint 狀態，並回傳下一個要訓練的 epoch。"""

    payload = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if ema_model is not None and "ema_model" in payload:
        unwrap_model(ema_model).load_state_dict(payload["ema_model"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    return int(payload.get("epoch", -1)) + 1
