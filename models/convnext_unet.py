from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.reshape(-1).float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps[:, None]
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=timesteps.dtype,
        ) / max(half_dim - 1, 1)
        emb = timesteps[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_channels = int(channels) * int(expansion)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels))
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = x + self.time_proj(time_emb).to(dtype=x.dtype)[:, :, None, None]
        x = self.pointwise(x)
        return x + residual


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        time_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels=channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_emb)
        return x


class ConvNeXtUNet(nn.Module):
    """ConvNeXt-style U-Net for DDPM noise prediction."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
        image_size: int | None = None,
    ):
        super().__init__()
        del image_size
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels if out_channels is not None else in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(mult) for mult in channel_mults)
        self.num_blocks = int(num_blocks)
        self.time_emb_dim = int(time_emb_dim)

        channels = [self.base_channels * mult for mult in self.channel_mults]
        if not channels:
            raise ValueError("channel_mults must contain at least one value.")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
        )

        self.stem = nn.Conv2d(self.in_channels, channels[0], kernel_size=3, padding=1)
        self.encoder_stages = nn.ModuleList(
            [
                ConvNeXtStage(
                    channels=stage_channels,
                    num_blocks=self.num_blocks,
                    time_emb_dim=self.time_emb_dim,
                    dropout=dropout,
                )
                for stage_channels in channels
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(
                    channels[index],
                    channels[index + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                for index in range(len(channels) - 1)
            ]
        )

        bottleneck_channels = channels[-1]
        self.bottleneck = ConvNeXtStage(
            channels=bottleneck_channels,
            num_blocks=self.num_blocks,
            time_emb_dim=self.time_emb_dim,
            dropout=dropout,
        )

        decoder_pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1])))
        self.decoder_projections = nn.ModuleList(
            [
                nn.Conv2d(current_channels + skip_channels, skip_channels, kernel_size=1)
                for current_channels, skip_channels in decoder_pairs
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [
                ConvNeXtStage(
                    channels=skip_channels,
                    num_blocks=self.num_blocks,
                    time_emb_dim=self.time_emb_dim,
                    dropout=dropout,
                )
                for _, skip_channels in decoder_pairs
            ]
        )

        self.final_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.final_act = nn.SiLU()
        self.out = nn.Conv2d(channels[0], self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        time_emb = self.time_embedding(timesteps).to(dtype=x.dtype)

        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.encoder_stages):
            x = stage(x, time_emb)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.bottleneck(x, time_emb)
        for projection, stage, skip in zip(
            self.decoder_projections,
            self.decoder_stages,
            reversed(skips[:-1]),
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = projection(x)
            x = stage(x, time_emb)

        x = self.out(self.final_act(self.final_norm(x)))
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
