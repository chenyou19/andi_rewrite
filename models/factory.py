"""依 experiment config 建立 model 的 factory。

把 model 建立集中在這裡，讓 framework 其他部分只依賴共同的 noise-prediction
contract：model(x_t, t) -> predicted_noise。新增 model family 時只需以
@register_model 註冊一個 builder，不需要再修改 build_model 的判斷分支。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .convnext_unet import ConvNeXtUNet
from .registry import MODEL_REGISTRY, register_model
from .unet import ANDiUNet


def _parse_channel_mults(value: Any) -> tuple[int, ...]:
    if value is None:
        return (1, 2, 4, 8)
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(part) for part in value)


@register_model("andi_unet", "original_unet", "unet")
def _build_andi_unet(config: dict[str, Any]) -> nn.Module:
    return ANDiUNet(
        in_channels=int(config.get("in_channels", config.get("channels", 4))),
        out_channels=int(config.get("out_channels", config.get("channels", 4))),
        image_size=int(config.get("image_size", 128)),
        time_dim=int(config.get("time_dim", 256)),
    )


@register_model("convnext_unet", "convnext-unet")
def _build_convnext_unet(config: dict[str, Any]) -> nn.Module:
    in_channels = int(config.get("in_channels", config.get("channels", 4)))
    out_channels = int(config.get("out_channels", config.get("channels", in_channels)))
    return ConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=int(config.get("base_channels", 64)),
        channel_mults=_parse_channel_mults(config.get("channel_mults")),
        num_blocks=int(config.get("num_blocks", 2)),
        time_emb_dim=int(config.get("time_emb_dim", config.get("time_dim", 256))),
        dropout=float(config.get("dropout", 0.0)),
        image_size=int(config.get("image_size", config.get("size", 128))),
    )


def build_model(config: dict[str, Any], device: torch.device | str) -> nn.Module:
    """建立 model 並移到指定 device。"""

    model_type = str(config.get("type", "andi_unet")).lower()
    builder = MODEL_REGISTRY.get(model_type)
    if builder is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return builder(config).to(device)
