from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LRWarmupCosineDecay(LambdaLR):
    """原版 ANDi 的 linear warmup 加 cosine decay scheduler。"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        steps_total: int,
        start_lr: float,
        target_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(int(warmup_steps), 1)
        self.steps_total = max(int(steps_total), self.warmup_steps + 1)
        self.start_lr = float(start_lr)
        self.target_lr = float(target_lr)
        self.increase = (self.target_lr - self.start_lr) / self.warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start_lr + (step * self.increase)
        progress = (step - self.warmup_steps) / float(self.steps_total - self.warmup_steps)
        return self.start_lr + (self.target_lr - self.start_lr) * ((1.0 + math.cos(math.pi * progress)) * 0.5)


def build_scheduler(config: dict, optimizer: Optimizer, steps_per_epoch: int, epochs: int):
    """從 config 建立 training LR scheduler。

    原版 ANDi optimizer 使用 lr=1，實際 learning rate multiplier 由此 scheduler 輸出。
    """

    scheduler_config = config.get("scheduler", {})
    scheduler_type = str(scheduler_config.get("type", "warmup_cosine")).lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None
    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    steps_total = max(int(steps_per_epoch) * int(epochs), 1)
    warmup_value = scheduler_config.get("warmup_steps", config.get("warmup_steps", 0.05))
    if isinstance(warmup_value, float) and warmup_value < 1:
        warmup_steps = int(warmup_value * steps_total)
    else:
        warmup_steps = int(warmup_value)
    return LRWarmupCosineDecay(
        optimizer=optimizer,
        warmup_steps=max(warmup_steps, 1),
        steps_total=steps_total,
        start_lr=float(scheduler_config.get("start_lr", config.get("start_lr", 2.0e-5))),
        target_lr=float(scheduler_config.get("target_lr", config.get("target_lr", config.get("learning_rate", 1.0e-4)))),
    )
