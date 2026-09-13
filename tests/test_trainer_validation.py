from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.engine.trainer import Trainer  # noqa: E402


class _Diffusion:
    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.arange(batch, device=device, dtype=torch.long) % 4

    def q_sample(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return images + 0.1 * noise


class _NoisePlan:
    def sample(self, shape, device, dtype, epoch=None, total_epochs=None):
        del epoch, total_epochs
        return torch.randn(tuple(shape), device=device, dtype=dtype)

    def describe(self) -> dict:
        return {"type": "test"}


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return self.conv(x)


class TrainerValidationTest(unittest.TestCase):
    def _trainer(
        self,
        root: Path,
        *,
        epochs: int = 2,
        checkpoint: dict | None = None,
    ) -> Trainer:
        return Trainer(
            model=_Model(),
            diffusion=_Diffusion(),
            noise_plan=_NoisePlan(),
            config={
                "epochs": epochs,
                "scheduler": {"type": "none"},
                "learning_rate": 1.0e-3,
                "normalize_input": False,
                "progress": {"enabled": False},
                "run_name": "validation-contract",
                "checkpoint": {
                    "dir": str(root / "checkpoints"),
                    **(checkpoint or {}),
                },
                "samples": {"enabled": False},
                "ema": {"enabled": True, "decay": 0.9, "step_start": 0},
            },
            validation_config={
                "enabled": True,
                "every_epochs": 1,
                "seed": 73,
                "use_ema": False,
            },
            device="cpu",
            steps_per_epoch=2,
        )

    @staticmethod
    def _loader(count: int = 4, batch_size: int = 2) -> DataLoader:
        values = torch.linspace(0.0, 1.0, count * 3 * 4 * 4).reshape(count, 3, 4, 4)
        return DataLoader(TensorDataset(values), batch_size=batch_size, shuffle=False)

    def test_validation_is_deterministic_and_does_not_mutate_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer(Path(directory))
            loader = self._loader()
            trainer.train_step(next(iter(loader)), epoch=0)
            parameters = [item.detach().clone() for item in trainer.model.parameters()]
            optimizer_state = deepcopy(trainer.optimizer.state_dict())
            ema_step = trainer.ema.step

            first = trainer.validate(loader, epoch=0)
            second = trainer.validate(loader, epoch=0)

        self.assertEqual(first, second)
        self.assertEqual(trainer.ema.step, ema_step)
        self.assertEqual(trainer.optimizer.state_dict()["param_groups"], optimizer_state["param_groups"])
        for before, after in zip(parameters, trainer.model.parameters()):
            torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
        self.assertTrue(trainer.model.training)

    def test_fit_logs_epoch_mean_validation_and_finite_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer(Path(directory), epochs=2)
            loader = self._loader()
            trainer.fit(loader, loader)
            with trainer.metrics_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["completed_epoch"] for row in rows], ["1", "2"])
        self.assertTrue(all(row["finite_status"] == "PASS" for row in rows))
        self.assertTrue(all(float(row["train_loss"]) >= 0.0 for row in rows))
        self.assertTrue(all(float(row["validation_loss"]) >= 0.0 for row in rows))
        self.assertIsNotNone(trainer.last_validation_loss)
        self.assertIsNotNone(trainer.best_validation_loss)

    def test_one_step_diagnostics_cover_shapes_optimizer_and_ema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer(Path(directory))
            result = trainer.run_one_step_diagnostics(next(iter(self._loader())), epoch=0)

        self.assertEqual(result["input_shape"], [2, 3, 4, 4])
        self.assertEqual(result["noise_shape"], [2, 3, 4, 4])
        self.assertEqual(result["model_output_shape"], [2, 3, 4, 4])
        self.assertTrue(result["loss_finite"])
        self.assertTrue(result["backward_successful"])
        self.assertTrue(result["optimizer_step_successful"])
        self.assertTrue(result["ema_successful"])

    def test_nonfinite_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer(Path(directory))
            invalid = torch.zeros(1, 3, 4, 4)
            invalid[0, 0, 0, 0] = float("nan")
            with self.assertRaisesRegex(FloatingPointError, "input"):
                trainer.train_step(invalid)

    def test_checkpoint_schedule_uses_zero_based_epoch_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer(
                Path(directory),
                epochs=233,
                checkpoint={"start_epoch": 19, "save_every_epochs": 20, "save_last": True},
            )
            actual = [epoch for epoch in range(233) if trainer.should_save(epoch)]

        self.assertEqual(actual, [19, 39, 59, 79, 99, 119, 139, 159, 179, 199, 219, 232])

    def test_lemon_and_brats_233_configs_are_matched_and_three_channel(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "train_lemon_t1_t2_flair_empirical_spectrum233.yaml").open(
            encoding="utf-8"
        ) as handle:
            lemon = yaml.safe_load(handle)
        with (root / "configs" / "train_brats_flair_t1_t2_empirical_spectrum233.yaml").open(
            encoding="utf-8"
        ) as handle:
            brats = yaml.safe_load(handle)

        self.assertEqual(lemon["runtime"]["seed"], 73)
        self.assertEqual(lemon["data"]["batch_size"], 52)
        self.assertEqual(lemon["model"]["in_channels"], 3)
        self.assertEqual(lemon["model"]["out_channels"], 3)
        self.assertEqual(lemon["training"]["epochs"], 233)
        self.assertFalse(lemon["training"]["samples"]["enabled"])
        self.assertFalse(lemon["training"]["eval_after_fit"]["enabled"])
        self.assertEqual(lemon["validation"]["every_epochs"], 1)
        self.assertEqual(lemon["validation"]["seed"], 73)
        self.assertFalse(lemon["validation"]["use_ema"])
        self.assertNotIn("channel_indices", lemon["noise"]["schedule"]["sampler"])
        self.assertEqual(brats["data"]["channel_indices"], [0, 1, 3])
        self.assertEqual(brats["validation"]["data"]["channel_indices"], [0, 1, 3])
        self.assertEqual(brats["noise"]["schedule"]["sampler"]["channel_indices"], [0, 1, 3])

        for section in ("runtime", "model", "diffusion", "training"):
            lemon_section = deepcopy(lemon[section])
            brats_section = deepcopy(brats[section])
            if section == "training":
                lemon_section.pop("run_name")
                brats_section.pop("run_name")
            self.assertEqual(lemon_section, brats_section, section)


if __name__ == "__main__":
    unittest.main()
