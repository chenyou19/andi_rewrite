from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDPMScheduler:
    """Linear DDPM schedule 與係數抽取工具。"""

    steps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.steps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        self.device = torch.device(device)
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alpha_hat = self.alpha_hat.to(self.device)
        return self

    @staticmethod
    def extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        """依 timestep 取出係數，並 reshape 成可 broadcast 的形狀。"""

        shape = (timesteps.shape[0],) + (1,) * (ndim - 1)
        return values[timesteps].reshape(shape)

    def describe(self) -> dict:
        return {
            "steps": self.steps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "device": str(self.device),
        }
