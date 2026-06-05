"""依 config 建立 noise sampler 與 epoch schedule。

trainer 與 detector 不應判斷「gaussian vs pyramid vs ...」。
它們只接收 NoisePlan 並向它要求 noise。因此新增 noise family 時，只需要
在 noise 模組新增 class，並在這個 factory 註冊即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .base import BaseNoise, NoisePlan
from .empirical_spectrum import EmpiricalSpectrumNoise
from .gaussian import GaussianNoise
from .hybrid import HybridNoise
from .pyramid import PyramidNoise
from .spectrum import SpectrumNoise


@dataclass
class StaticNoisePlan(NoisePlan):
    """所有 epoch 與 inference call 都使用同一個 sampler。"""

    sampler: BaseNoise

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> torch.Tensor:
        del epoch, total_epochs
        return self.sampler.sample(shape, device=device, dtype=dtype)

    def describe(self) -> dict:
        return {"schedule": "static", "sampler": self.sampler.describe()}


@dataclass
class EpochSwitchNoisePlan(NoisePlan):
    """在 config 指定的 epoch 邊界由一個 sampler 切換到另一個 sampler。"""

    before: BaseNoise
    after: BaseNoise
    switch_epoch: int | None = None
    switch_fraction: float | None = 0.5

    def _use_after(self, epoch: int | None, total_epochs: int | None) -> bool:
        if epoch is None:
            return False
        if self.switch_epoch is not None:
            return epoch >= self.switch_epoch
        if total_epochs is None:
            return False
        return epoch >= int(total_epochs * float(self.switch_fraction))

    def sample(
        self,
        shape: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> torch.Tensor:
        sampler = self.after if self._use_after(epoch, total_epochs) else self.before
        return sampler.sample(shape, device=device, dtype=dtype)

    def describe(self) -> dict:
        return {
            "schedule": "epoch_switch",
            "switch_epoch": self.switch_epoch,
            "switch_fraction": self.switch_fraction,
            "before": self.before.describe(),
            "after": self.after.describe(),
        }


def build_noise_sampler(config: dict[str, Any]) -> BaseNoise:
    """從 config 建立 sampler；呼叫端不需要也不應該判斷 sampler type。"""

    noise_type = str(config.get("type", "gaussian")).lower()
    if noise_type == "gaussian":
        return GaussianNoise()
    if noise_type == "pyramid":
        return PyramidNoise(
            discount=float(config.get("discount", 0.8)),
            levels=int(config.get("levels", 10)),
            normalize=bool(config.get("normalize", True)),
        )
    if noise_type == "spectrum":
        return SpectrumNoise(
            exponent=float(config.get("exponent", 1.0)),
            low_frequency_bias=bool(config.get("low_frequency_bias", True)),
            normalize=bool(config.get("normalize", True)),
        )
    if noise_type == "empirical_spectrum":
        if "stats_path" not in config:
            raise ValueError("noise sampler type 'empirical_spectrum' requires stats_path.")
        return EmpiricalSpectrumNoise(
            stats_path=config["stats_path"],
            mode=str(config.get("mode", "radial")),
            spectrum_key=str(config.get("spectrum_key", "mean_amplitude")),
            radial_key=str(config.get("radial_key", "radial_amplitude")),
            per_channel=bool(config.get("per_channel", True)),
            strength=float(config.get("strength", 1.0)),
            normalize=bool(config.get("normalize", True)),
            eps=float(config.get("eps", 1.0e-8)),
        )
    if noise_type == "hybrid":
        components = []
        for item in config.get("components", []):
            weight = float(item.get("weight", 1.0))
            # component config 本身也是一般 noise config，因此 hybrid 可以巢狀組合，
            # training loop 不需要新增任何特殊分支。
            component_config = {key: value for key, value in item.items() if key != "weight"}
            components.append((build_noise_sampler(component_config), weight))
        return HybridNoise(components, normalize=bool(config.get("normalize", True)))

    raise ValueError(f"Unknown noise type: {noise_type}")


def build_noise_plan(config: dict[str, Any]) -> NoisePlan:
    """從 config section 建立 epoch-aware noise plan。"""

    schedule_config = config.get("schedule", config)
    schedule_type = str(schedule_config.get("type", "static")).lower()
    if schedule_type == "static":
        sampler_config = schedule_config.get("sampler", schedule_config)
        return StaticNoisePlan(build_noise_sampler(sampler_config))
    if schedule_type == "epoch_switch":
        return EpochSwitchNoisePlan(
            before=build_noise_sampler(schedule_config["before"]),
            after=build_noise_sampler(schedule_config["after"]),
            switch_epoch=schedule_config.get("switch_epoch"),
            switch_fraction=schedule_config.get("switch_epoch_fraction", 0.5),
        )

    raise ValueError(f"Unknown noise schedule: {schedule_type}")
