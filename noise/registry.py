"""Noise registry：用註冊取代 sampler 與 schedule 的 if 判斷鏈。

把「name -> builder」對照表與註冊機制放在這裡，factory 只負責登錄各個 noise
sampler / schedule builder 並提供 build_noise_sampler / build_noise_plan 入口，
彼此不會 circular import（registry 只依賴 base 的型別）。
"""

from __future__ import annotations

from typing import Any, Callable

from .base import BaseNoise, NoisePlan

NoiseBuilder = Callable[[dict[str, Any]], BaseNoise]
ScheduleBuilder = Callable[[dict[str, Any]], NoisePlan]

NOISE_REGISTRY: dict[str, NoiseBuilder] = {}
SCHEDULE_REGISTRY: dict[str, ScheduleBuilder] = {}


def register_noise(*names: str) -> Callable[[NoiseBuilder], NoiseBuilder]:
    """把 sampler builder 註冊到一個或多個 noise type alias 底下。"""

    def decorator(builder: NoiseBuilder) -> NoiseBuilder:
        for name in names:
            NOISE_REGISTRY[name.lower()] = builder
        return builder

    return decorator


def register_schedule(*names: str) -> Callable[[ScheduleBuilder], ScheduleBuilder]:
    """把 schedule builder 註冊到一個或多個 schedule type alias 底下。"""

    def decorator(builder: ScheduleBuilder) -> ScheduleBuilder:
        for name in names:
            SCHEDULE_REGISTRY[name.lower()] = builder
        return builder

    return decorator
