"""Anomaly scorer 的抽象介面。

evaluation collector 只需要「把一批 2D slice 變成 raw anomaly map」這個能力，
不需要知道背後是 DDPM transition deviation、autoencoder reconstruction 或其他
scorer。用 typing.Protocol 定義結構型介面，對既有的 ANDiDetector 完全不侵入
（只要它提供對應方法即可，不需要繼承）。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SliceAnomalyScorer(Protocol):
    """將 2D slice batch 轉成 raw anomaly score map 的共同介面。

    Input:
        images: [N, C, H, W]

    Output:
        raw anomaly map: [N, H, W]
    """

    model: Any
    device: torch.device

    def score_slices(
        self,
        images: torch.Tensor,
        progress: bool = False,
        progress_description: str = "Anomaly scoring",
        progress_leave: bool = False,
    ) -> torch.Tensor:
        ...

    def describe(self) -> dict[str, Any]:
        ...
