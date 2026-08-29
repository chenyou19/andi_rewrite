from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from andi_rewrite.diffusion import DDPMDiffusion
from andi_rewrite.engine.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from andi_rewrite.engine.ema import EMA
from andi_rewrite.engine.schedulers import build_scheduler
from andi_rewrite.noise import NoisePlan
from andi_rewrite.utils.progress import ProgressReporter
from andi_rewrite.utils.visualization import save_images


class Trainer:
    """DDPM trainer；training loop 不依賴任何具體 noise type。"""

    def __init__(
        self,
        model: nn.Module,
        diffusion: DDPMDiffusion,
        noise_plan: NoisePlan,
        config: dict[str, Any],
        device: torch.device | str,
        steps_per_epoch: int = 1,
        accelerator: Any = None,
        validation_config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.noise_plan = noise_plan
        self.config = config
        self.device = torch.device(device)
        self.accelerator = accelerator
        self.validation_config = dict(validation_config or {})
        self.validation_enabled = bool(self.validation_config.get("enabled", bool(validation_config)))
        self.validation_every_epochs = max(int(self.validation_config.get("every_epochs", 1)), 1)
        self.validation_seed = int(self.validation_config.get("seed", 73))
        self.validation_use_ema = bool(self.validation_config.get("use_ema", False))
        self.validation_dataloader: Iterable | None = None
        self.epochs = int(config.get("epochs", 1))
        self.start_epoch = 0
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.normalize_input = bool(config.get("normalize_input", True))
        self.finite_checks = bool(config.get("finite_checks", True))
        self.loss_fn = nn.MSELoss()
        scheduler_type = str(config.get("scheduler", {}).get("type", "warmup_cosine")).lower()
        # 原版 ANDi scheduler 預期 optimizer lr=1，實際 learning rate 由 LambdaLR 輸出。
        optimizer_lr = 1.0 if scheduler_type == "warmup_cosine" else float(
            config.get("learning_rate", config.get("target_lr", 1.0e-4))
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_lr,
        )
        self.scheduler = build_scheduler(config, self.optimizer, self.steps_per_epoch, self.epochs)

        ema_config = config.get("ema", {})
        if isinstance(ema_config, bool):
            ema_enabled = ema_config
            ema_config = {}
        else:
            ema_enabled = bool(ema_config.get("enabled", True))
        self.ema = EMA(
            decay=float(ema_config.get("decay", config.get("ema_decay", 0.995))),
            step_start=int(ema_config.get("step_start", 2000)),
        ) if ema_enabled else None
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False) if self.ema else None

        checkpoint_config = config.get("checkpoint", {})
        self.checkpoint_dir = Path(checkpoint_config.get("dir", config.get("checkpoint_dir", "outputs/checkpoints")))
        self.save_every_epochs = int(checkpoint_config.get("save_every_epochs", config.get("save_ckpt", 0)))
        self.save_start_epoch = int(checkpoint_config.get("start_epoch", config.get("start_ckpt", 0)))
        self.save_last_checkpoint = bool(checkpoint_config.get("save_last", False))
        sample_config = config.get("samples", config.get("sample", {}))
        self.sample_enabled = bool(sample_config.get("enabled", False))
        self.sample_every_epochs = int(sample_config.get("every_epochs", self.save_every_epochs or 1))
        self.sample_start_epoch = int(sample_config.get("start_epoch", self.save_start_epoch))
        self.sample_num_images = int(sample_config.get("num_images", config.get("num_images", 3)))
        self.sample_channels = int(sample_config.get("channels", 4))
        self.sample_image_size = int(sample_config.get("image_size", 128))
        self.sample_mode = str(sample_config.get("mode", "L"))
        self.sample_nrow = sample_config.get("nrow")
        self.sample_output_dir = Path(sample_config.get("output_dir", "outputs/samples"))
        self.sample_use_ema = bool(sample_config.get("use_ema", True))
        self.sample_dtype = str(sample_config.get("dtype", "float32"))
        self.sample_clip_denoised = bool(sample_config.get("clip_denoised", True))
        self.last_loss: float | None = None
        self.best_loss: float | None = None
        self.last_validation_loss: float | None = None
        self.best_validation_loss: float | None = None
        self.last_epoch: int | None = None
        self.last_checkpoint_path: Path | None = None
        self.last_sample_path: Path | None = None
        self.last_epoch_metrics: dict[str, Any] | None = None
        self._last_step_diagnostics: dict[str, Any] = {}
        run_name = str(self.config.get("run_name", "andi_rewrite"))
        self.metrics_path = self.checkpoint_dir / run_name / "training_metrics.csv"
        resume_path = checkpoint_config.get("resume", config.get("resume"))
        if resume_path:
            self.start_epoch = load_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                ema_model=self.ema_model,
                ema=self.ema,
                map_location=self.device,
            )

    def _progress_enabled(self) -> bool:
        progress_config = self.config.get("progress", True)
        if isinstance(progress_config, dict):
            return bool(progress_config.get("enabled", True)) and self.is_main_process
        return bool(progress_config) and self.is_main_process

    @property
    def is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def prepare(
        self,
        dataloader: Iterable,
        validation_dataloader: Iterable | None = None,
    ) -> Iterable | tuple[Iterable, Iterable]:
        """當啟用 distributed execution 時，交由 Accelerate wrap 相關物件。"""

        if self.accelerator is None:
            if validation_dataloader is None:
                return dataloader
            return dataloader, validation_dataloader
        objects = [self.model, self.optimizer, dataloader]
        if validation_dataloader is not None:
            objects.append(validation_dataloader)
        if self.scheduler is not None:
            objects.append(self.scheduler)
        if self.ema_model is not None:
            objects.append(self.ema_model)
        prepared = self.accelerator.prepare(*objects)
        self.model = prepared[0]
        self.optimizer = prepared[1]
        dataloader = prepared[2]
        offset = 3
        prepared_validation = None
        if validation_dataloader is not None:
            prepared_validation = prepared[offset]
            offset += 1
        if self.scheduler is not None:
            self.scheduler = prepared[offset]
            offset += 1
        if self.ema_model is not None:
            self.ema_model = prepared[offset]
        if prepared_validation is None:
            return dataloader
        return dataloader, prepared_validation

    def _prepare_batch(self, batch: torch.Tensor | list | tuple) -> torch.Tensor:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(self.device)
        if self.normalize_input:
            images = images * 2.0 - 1.0
        return images

    def _require_finite(self, name: str, value: torch.Tensor) -> None:
        if self.finite_checks and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite tensor detected during training: {name}.")

    def _forward_loss(
        self,
        batch: torch.Tensor | list | tuple,
        epoch: int,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        images = self._prepare_batch(batch)
        self._require_finite("input", images)
        timesteps = self.diffusion.sample_timesteps(images.shape[0], self.device)
        noise = self.noise_plan.sample(
            images.shape,
            device=self.device,
            dtype=images.dtype,
            epoch=epoch,
            total_epochs=self.epochs,
        )
        self._require_finite("noise", noise)
        x_t = self.diffusion.q_sample(images, timesteps, noise)
        self._require_finite("noisy_input", x_t)
        predicted_noise = (model or self.model)(x_t, timesteps)
        self._require_finite("predicted_noise", predicted_noise)
        loss = self.loss_fn(predicted_noise, noise)
        self._require_finite("loss", loss)
        diagnostics = {
            "input_shape": list(images.shape),
            "noise_shape": list(noise.shape),
            "model_output_shape": list(predicted_noise.shape),
            "input_finite": True,
            "noise_finite": True,
            "model_output_finite": True,
            "loss_finite": bool(math.isfinite(float(loss.detach().cpu()))),
            "batch_size": int(images.shape[0]),
        }
        return loss, diagnostics

    def _require_finite_gradients(self) -> None:
        if not self.finite_checks:
            return
        for name, parameter in unwrap_model(self.model).named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Non-finite gradient detected during training: {name}.")

    def train_step(self, batch: torch.Tensor | list | tuple, epoch: int = 0) -> dict[str, float]:
        loss, diagnostics = self._forward_loss(batch, epoch=epoch)

        self.optimizer.zero_grad(set_to_none=True)
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        self._require_finite_gradients()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None and self.ema_model is not None:
            self.ema.step_ema(self.ema_model, self.model)
        diagnostics.update(
            {
                "backward_successful": True,
                "gradients_finite": True,
                "optimizer_step_successful": True,
                "ema_step": self.ema.step if self.ema is not None else None,
            }
        )
        self._last_step_diagnostics = diagnostics
        return {"loss": float(loss.detach().cpu())}

    def run_one_step_diagnostics(
        self,
        batch: torch.Tensor | list | tuple,
        epoch: int = 0,
    ) -> dict[str, Any]:
        parameter = next((item for item in unwrap_model(self.model).parameters() if item.requires_grad), None)
        before = parameter.detach().clone() if parameter is not None else None
        ema_step_before = self.ema.step if self.ema is not None else None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        result = self.train_step(batch, epoch=epoch)
        changed = True if parameter is None else not torch.equal(before, parameter.detach())
        diagnostics = dict(self._last_step_diagnostics)
        diagnostics.update(
            {
                "loss": result["loss"],
                "optimizer_step_successful": bool(changed),
                "ema_successful": (
                    True
                    if self.ema is None
                    else self.ema.step == int(ema_step_before or 0) + 1
                ),
                "gpu_peak_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
            }
        )
        if not diagnostics["optimizer_step_successful"]:
            raise RuntimeError("One-step smoke test did not change the first trainable model parameter.")
        return diagnostics

    def should_validate(self, epoch: int, validation_dataloader: Iterable | None) -> bool:
        return bool(
            self.validation_enabled
            and validation_dataloader is not None
            and (epoch + 1) % self.validation_every_epochs == 0
        )

    def validate(self, validation_dataloader: Iterable, epoch: int) -> float:
        model = self.ema_model if self.validation_use_ema and self.ema_model is not None else self.model
        was_training = bool(model.training)
        model.eval()
        devices: list[int] = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        loss_sum = 0.0
        sample_count = 0
        try:
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(self.validation_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.validation_seed)
                with torch.no_grad():
                    for batch in validation_dataloader:
                        loss, diagnostics = self._forward_loss(batch, epoch=epoch, model=model)
                        batch_size = int(diagnostics["batch_size"])
                        loss_sum += float(loss.detach().cpu()) * batch_size
                        sample_count += batch_size
        finally:
            if was_training:
                model.train()
        if sample_count <= 0:
            raise ValueError("Validation dataloader yielded zero samples.")
        value = loss_sum / sample_count
        if not math.isfinite(value):
            raise FloatingPointError("Validation loss is NaN or Inf.")
        return float(value)

    def _write_epoch_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "epoch_index",
            "completed_epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
            "ema_step",
            "finite_status",
            "elapsed_seconds",
            "gpu_peak_bytes",
            "checkpoint_path",
        ]
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({name: metrics.get(name) for name in fieldnames})

    def fit(
        self,
        dataloader: Iterable,
        validation_dataloader: Iterable | None = None,
    ) -> None:
        """執行 config 指定的 training loop，並依設定做 checkpoint 與 sample 輸出。"""

        self.model.train()
        progress_enabled = self._progress_enabled()
        epoch_bar = ProgressReporter(
            max(self.epochs - self.start_epoch, 0),
            "Training epochs",
            enabled=progress_enabled,
            unit="epoch",
        )
        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_started = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                train_loss_sum = 0.0
                train_sample_count = 0
                batch_bar = ProgressReporter(
                    self.steps_per_epoch,
                    f"Epoch {epoch + 1}/{self.epochs}",
                    enabled=progress_enabled,
                    unit="batch",
                    leave=False,
                )
                try:
                    for batch in dataloader:
                        result = self.train_step(batch, epoch=epoch)
                        batch_size = int(self._last_step_diagnostics.get("batch_size", 1))
                        train_loss_sum += result["loss"] * batch_size
                        train_sample_count += batch_size
                        batch_bar.update(postfix={"loss": f"{result['loss']:.6f}"})
                finally:
                    batch_bar.close()
                if train_sample_count <= 0:
                    raise ValueError("Training dataloader yielded zero samples.")
                train_loss = train_loss_sum / train_sample_count
                if not math.isfinite(train_loss):
                    raise FloatingPointError("Mean training loss is NaN or Inf.")
                self.last_loss = float(train_loss)
                if self.best_loss is None or self.last_loss < self.best_loss:
                    self.best_loss = self.last_loss

                validation_loss = None
                if self.should_validate(epoch, validation_dataloader):
                    validation_loss = self.validate(validation_dataloader, epoch=epoch)
                    self.last_validation_loss = validation_loss
                    if self.best_validation_loss is None or validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                    self.model.train()
                self.last_epoch = epoch
                checkpoint_path = None
                if self.should_save(epoch):
                    checkpoint_path = self.save(epoch)
                if self.should_sample(epoch):
                    path = self.save_samples(epoch)
                    if self.is_main_process:
                        print(f"Saved samples: {path}")
                elapsed = time.perf_counter() - epoch_started
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                )
                epoch_metrics = {
                    "epoch_index": epoch,
                    "completed_epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": learning_rate,
                    "ema_step": self.ema.step if self.ema is not None else None,
                    "finite_status": "PASS",
                    "elapsed_seconds": elapsed,
                    "gpu_peak_bytes": peak_memory,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                }
                self.last_epoch_metrics = epoch_metrics
                self._write_epoch_metrics(epoch_metrics)
                if self.is_main_process:
                    validation_text = "n/a" if validation_loss is None else f"{validation_loss:.6f}"
                    print(
                        "Epoch "
                        f"index={epoch} completed={epoch + 1} "
                        f"loss={train_loss:.6f} validation_loss={validation_text} "
                        f"lr={learning_rate:.10g} "
                        f"elapsed_seconds={elapsed:.3f} "
                        f"gpu_peak_bytes={peak_memory} checkpoint={checkpoint_path}"
                    )
                epoch_bar.update(postfix={"loss": f"{train_loss:.6f}"})
        finally:
            epoch_bar.close()

    def should_save(self, epoch: int) -> bool:
        if not self.is_main_process or self.save_every_epochs <= 0:
            return False
        if self.save_last_checkpoint and epoch == self.epochs - 1:
            return True
        return (
            epoch >= self.save_start_epoch
            and (epoch - self.save_start_epoch) % self.save_every_epochs == 0
        )

    def save(self, epoch: int) -> Path:
        run_name = self.config.get("run_name", "andi_rewrite")
        path = self.checkpoint_dir / str(run_name) / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_model=self.ema_model,
            ema_state=self.ema.state_dict() if self.ema is not None else None,
            epoch=epoch,
            config=self.config,
            accelerator=self.accelerator,
        )
        self.last_checkpoint_path = path
        return path

    def should_sample(self, epoch: int) -> bool:
        return (
            self.sample_enabled
            and self.sample_every_epochs > 0
            and epoch >= self.sample_start_epoch
            and (epoch - self.sample_start_epoch) % self.sample_every_epochs == 0
            and self.is_main_process
        )

    def _sample_dtype(self) -> torch.dtype:
        if self.sample_dtype == "float16":
            return torch.float16
        if self.sample_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def sample_images(self, epoch: int | None = None) -> torch.Tensor:
        """在設定允許且 EMA 可用時，使用 EMA 權重產生 DDPM samples。"""

        model = self.ema_model if self.sample_use_ema and self.ema_model is not None else self.model
        model = unwrap_model(model)
        return self.diffusion.p_sample_loop(
            model=model,
            shape=(
                self.sample_num_images,
                self.sample_channels,
                self.sample_image_size,
                self.sample_image_size,
            ),
            noise_plan=self.noise_plan,
            device=self.device,
            dtype=self._sample_dtype(),
            epoch=epoch,
            total_epochs=self.epochs,
            clip_denoised=self.sample_clip_denoised,
            progress=self._progress_enabled(),
            progress_description=f"Sampling epoch {epoch}" if epoch is not None else "Sampling",
        )

    def save_samples(self, epoch: int, prefix: str = "epoch") -> Path:
        run_name = self.config.get("run_name", "andi_rewrite")
        suffix = f"{prefix}_{epoch:04d}.png" if isinstance(epoch, int) else f"{prefix}.png"
        path = self.sample_output_dir / str(run_name) / suffix
        images = self.sample_images(epoch=epoch)
        save_images(images, path, mode=self.sample_mode, nrow=self.sample_nrow)
        self.last_sample_path = path
        return path

    def describe(self) -> dict:
        return {
            "start_epoch": self.start_epoch,
            "epochs": self.epochs,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "scheduler": type(self.scheduler).__name__ if self.scheduler is not None else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "checkpoint_dir": str(self.checkpoint_dir),
            "metrics_path": str(self.metrics_path),
            "validation": {
                "enabled": self.validation_enabled,
                "every_epochs": self.validation_every_epochs,
                "seed": self.validation_seed,
                "use_ema": self.validation_use_ema,
                "last_loss": self.last_validation_loss,
                "best_loss": self.best_validation_loss,
            },
            "samples": {
                "enabled": self.sample_enabled,
                "every_epochs": self.sample_every_epochs,
                "num_images": self.sample_num_images,
                "output_dir": str(self.sample_output_dir),
                "use_ema": self.sample_use_ema,
            },
            "normalize_input": self.normalize_input,
            "noise_plan": self.noise_plan.describe(),
            "model": type(unwrap_model(self.model)).__name__,
            "accelerate": self.accelerator is not None,
        }
