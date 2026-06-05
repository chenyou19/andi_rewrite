from __future__ import annotations

import torch.nn as nn


class EMA:
    """相容 wrapped model 的 Exponential Moving Average 輔助工具。

    EMA 維護一份緩慢更新的 denoising model 副本；sampling 時通常比最新訓練權重穩定。
    """

    def __init__(self, decay: float, step_start: int = 2000):
        self.decay = float(decay)
        self.step_start = int(step_start)
        self.step = 0

    @staticmethod
    def _state_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module) -> None:
        self._state_model(ema_model).load_state_dict(self._state_model(model).state_dict())

    def update_model_average(self, ema_model: nn.Module, model: nn.Module) -> None:
        ema_model = self._state_model(ema_model)
        model = self._state_model(model)
        for ema_params, current_params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data = ema_params.data * self.decay + current_params.data * (1.0 - self.decay)

    def step_ema(self, ema_model: nn.Module, model: nn.Module) -> None:
        if self.step < self.step_start:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step_start": self.step_start, "step": self.step}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.step_start = int(state.get("step_start", self.step_start))
        self.step = int(state.get("step", self.step))
