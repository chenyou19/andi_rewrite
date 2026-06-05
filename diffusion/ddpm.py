from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from andi_rewrite.utils.progress import ProgressReporter

from .scheduler import DDPMScheduler


class DDPMDiffusion:
    """DDPM forward/reverse transition 的數學計算。

    這個 class 刻意不放 anomaly map、metric、plotting 或 data loading。
    ANDi 專屬推論流程放在 andi_rewrite.anomaly。
    """

    def __init__(self, scheduler: DDPMScheduler):
        self.scheduler = scheduler

    @property
    def steps(self) -> int:
        return self.scheduler.steps

    def to(self, device: torch.device | str) -> "DDPMDiffusion":
        self.scheduler.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(1, self.steps, size=(batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM forward process: q(x_t | x_0)。"""

        sqrt_alpha_hat = torch.sqrt(
            self.scheduler.extract(self.scheduler.alpha_hat, timesteps, x0.ndim)
        )
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1.0 - self.scheduler.extract(self.scheduler.alpha_hat, timesteps, x0.ndim)
        )
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def posterior_mean_from_noise(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """p_theta(x_{t-1} | x_t) 的 mean；若輸入真實 noise，則等價於 q 的 mean。"""

        alpha = self.scheduler.extract(self.scheduler.alpha, timesteps, x_t.ndim)
        alpha_hat = self.scheduler.extract(self.scheduler.alpha_hat, timesteps, x_t.ndim)
        return (x_t - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * predicted_noise) / torch.sqrt(alpha)

    def posterior_mean_from_x0(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """q(x_{t-1} | x_t, x_0) 的 mean。"""

        alpha = self.scheduler.extract(self.scheduler.alpha, timesteps, x_t.ndim)
        alpha_hat = self.scheduler.extract(self.scheduler.alpha_hat, timesteps, x_t.ndim)
        beta = self.scheduler.extract(self.scheduler.beta, timesteps, x_t.ndim)
        alpha_hat_prev = self.scheduler.extract(self.scheduler.alpha_hat, timesteps - 1, x_t.ndim)
        w0 = torch.sqrt(alpha_hat_prev) * beta / (1.0 - alpha_hat)
        wt = torch.sqrt(alpha) * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)
        return w0 * x0 + wt * x_t

    def reverse_variance(self, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        beta = self.scheduler.extract(self.scheduler.beta, timesteps, ndim)
        alpha_hat = self.scheduler.extract(self.scheduler.alpha_hat, timesteps, ndim)
        alpha_hat_prev = self.scheduler.extract(self.scheduler.alpha_hat, timesteps - 1, ndim)
        return beta * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Sequence[int],
        noise_plan: Any,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        epoch: int | None = None,
        total_epochs: int | None = None,
        clip_denoised: bool = True,
        progress: bool = False,
        progress_description: str = "Sampling",
    ) -> torch.Tensor:
        """從已學到的 reverse DDPM chain 取樣。

        noise 來源仍由 NoisePlan 提供，因此 sampling 可以使用與 training 相同的
        gaussian/pyramid/spectrum/hybrid 策略。
        """

        device = torch.device(device)
        model_was_training = model.training
        model.eval()
        x = noise_plan.sample(
            shape,
            device=device,
            dtype=dtype,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        batch_size = int(shape[0])
        progress_bar = ProgressReporter(
            max(self.steps - 1, 0),
            progress_description,
            enabled=progress,
            unit="step",
            leave=True,
        )
        with torch.no_grad():
            try:
                for step in reversed(range(1, self.steps)):
                    timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
                    predicted_noise = model(x, timesteps)
                    mean = self.posterior_mean_from_noise(x, predicted_noise, timesteps)
                    if step > 1:
                        noise = noise_plan.sample(
                            shape,
                            device=device,
                            dtype=dtype,
                            epoch=epoch,
                            total_epochs=total_epochs,
                        )
                        variance = self.reverse_variance(timesteps, x.ndim)
                        x = mean + torch.sqrt(variance) * noise
                    else:
                        x = mean
                    if clip_denoised:
                        x = x.clamp(-1.0, 1.0)
                    progress_bar.update(postfix={"t": step})
            finally:
                progress_bar.close()

        if model_was_training:
            model.train()
        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0

    def describe(self) -> dict:
        return self.scheduler.describe()


def build_diffusion(config: dict, device: torch.device | str) -> DDPMDiffusion:
    scheduler = DDPMScheduler(
        steps=int(config.get("steps", config.get("noise_steps", 1000))),
        beta_start=float(config.get("beta_start", 1.0e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        device=device,
    )
    return DDPMDiffusion(scheduler)
