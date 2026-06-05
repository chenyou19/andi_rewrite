from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """原版 ANDi attention block，保留給 full-size UNet 重用。"""

    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = int(channels)
        self.size = int(size)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2).contiguous()
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size).contiguous()


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        residual: bool = False,
    ):
        super().__init__()
        del mid_channels
        self.residual = bool(residual)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip_x.shape[-2:]:
            x = F.interpolate(x, size=skip_x.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ANDiUNet(nn.Module):
    """相容原版 ANDi model shape 的 self-attention UNet。"""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        image_size: int = 128,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_dim = int(time_dim)
        self.inc = DoubleConv(in_channels, 32)
        self.inc2 = DoubleConv(32, 32, residual=True)
        self.down1 = Down(32, 64, self.time_dim)
        self.down2 = Down(64, 128, self.time_dim)
        self.sa1 = SelfAttention(128, int(image_size / 4))
        self.down3 = Down(128, 256, self.time_dim)
        self.sa2 = SelfAttention(256, int(image_size / 8))
        self.down4 = Down(256, 256, self.time_dim)
        self.sa3 = SelfAttention(256, int(image_size / 16))
        self.bot1 = DoubleConv(256, 512)
        self.bot3 = DoubleConv(512, 256)
        self.sa4 = SelfAttention(256, int(image_size / 16))
        self.up1 = Up(512, 128, self.time_dim)
        self.sa5 = SelfAttention(128, int(image_size / 8))
        self.up2 = Up(256, 64, self.time_dim)
        self.sa6 = SelfAttention(64, int(image_size / 4))
        self.up3 = Up(128, 32, self.time_dim)
        self.up4 = Up(64, 32, self.time_dim)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def pos_encoding(self, timesteps: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=timesteps.device).float() / channels)
        )
        pos_enc_a = torch.sin(timesteps.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(timesteps.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def unet_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x1 = self.inc2(x1)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa3(x5)
        x5 = self.bot1(x5)
        x5 = self.bot3(x5)
        x = self.sa4(x5)
        x = self.up1(x, x4, t)
        x = self.sa5(x)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        return self.outc(x)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.unsqueeze(-1).type(x.dtype)
        return self.unet_forward(x, self.pos_encoding(timesteps, self.time_dim).type(x.dtype))
