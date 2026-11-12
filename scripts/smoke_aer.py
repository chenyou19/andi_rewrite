"""Run AER training, inference diagnostics and checkpoint reload on synthetic tensors."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import torch

from andi_rewrite.diffusion import DDPMDiffusion, DDPMScheduler
from andi_rewrite.engine import Trainer
from andi_rewrite.models import build_model
from andi_rewrite.noise import build_noise_plan


def smoke_architecture(kind: str, updates: int, root: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    spatial = (16, 16) if kind == "andi_unet" else (13, 17)
    model_config = {
        "type": kind,
        "in_channels": 1,
        "out_channels": 1,
        "image_size": 16,
        "time_dim": 16,
        "base_channels": 8,
        "channel_mults": [1, 2],
        "num_blocks": 1,
        "aer": {
            "enabled": True,
            "steps": 3,
            "latent_channels": 8,
            "num_tokens": 3,
            "mode": "guarded",
            "aux_loss_weight": 0.1,
            "residual_scale": 0.05,
        },
    }
    training_config = {
        "epochs": 1,
        "learning_rate": 0.001,
        "scheduler": {"type": "none"},
        "ema": {"enabled": True, "decay": 0.9, "step_start": 0},
        "checkpoint": {"dir": str(root), "save_every_epochs": 1},
        "progress": False,
        "run_name": kind,
    }
    diffusion = DDPMDiffusion(DDPMScheduler(steps=32))
    noise_plan = build_noise_plan({"schedule": {"type": "static", "sampler": {"type": "gaussian"}}})
    model = build_model(model_config, "cpu")
    trainer = Trainer(model, diffusion, noise_plan, training_config, "cpu")
    history = []
    for _ in range(updates):
        batch = torch.rand((2, 1, *spatial))
        result = trainer.train_step(batch)
        gradient_squared = sum(
            parameter.grad.square().sum().item()
            for parameter in model.aer.proposer.parameters()
            if parameter.grad is not None
        )
        result["proposal_gradient_l2"] = gradient_squared**0.5
        history.append(result)
    if not all(step["proposal_gradient_l2"] > 0 for step in history):
        raise AssertionError("the shared proposer did not receive a nonzero training gradient")

    images = torch.rand((2, 1, *spatial))
    validation_noise_mse = trainer.validate([images], epoch=0)
    timesteps = torch.tensor([3, 15])
    x_t = diffusion.q_sample(images, timesteps, torch.randn_like(images))
    traces = []

    def request_trace(_, args, kwargs):
        return args, {**kwargs, "return_trace": True}

    def record_trace(_, args, output):
        traces.append(output.trace)

    with model.aer.register_forward_pre_hook(request_trace, with_kwargs=True), model.aer.register_forward_hook(record_trace):
        with torch.inference_mode():
            prediction = model.eval()(x_t, timesteps)
            ema_prediction = trainer.ema_model(x_t, timesteps)
    if prediction.shape != images.shape or not torch.isfinite(prediction).all():
        raise AssertionError("model returned invalid noise predictions")
    trace, = traces
    if not (trace.energy[1:] <= trace.reference_energy).all():
        raise AssertionError("guarded iterate exceeds the reference energy")

    path = trainer.save(epoch=0)
    resume = {**training_config, "checkpoint": {**training_config["checkpoint"], "resume": str(path)}}
    restored = Trainer(build_model(model_config, "cpu"), diffusion, noise_plan, resume, "cpu")
    with torch.inference_mode():
        reloaded = restored.model.eval()(x_t, timesteps)
        reloaded_ema = restored.ema_model(x_t, timesteps)
    torch.testing.assert_close(reloaded, prediction, rtol=0, atol=0)
    torch.testing.assert_close(reloaded_ema, ema_prediction, rtol=0, atol=0)
    if restored.start_epoch != 1 or restored.ema.step != updates:
        raise AssertionError("trainer did not restore the expected epoch and EMA step")
    return {
        "seed": seed,
        "model": model_config,
        "training": training_config,
        "training_history": history,
        "validation_noise_mse": validation_noise_mse,
        "input_shape": list(images.shape),
        "energy": trace.energy.tolist(),
        "gradient_l2": trace.residual.tolist(),
        "accepted": trace.accepted.tolist(),
        "iterations": trace.iterations.tolist(),
        "checkpoint": str(path),
        "checkpoint_max_logit_difference": (reloaded - prediction).abs().max().item(),
        "ema_checkpoint_max_logit_difference": (reloaded_ema - ema_prediction).abs().max().item(),
        "ema_steps": restored.ema.step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "aer_smoke")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "intraop_threads": torch.get_num_threads(),
        "architectures": {
            kind: smoke_architecture(kind, args.updates, args.output_dir, args.seed + index)
            for index, kind in enumerate(("andi_unet", "convnext_unet"))
        },
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"AER smoke passed for ANDiUNet and ConvNeXtUNet. Summary at {path}")


if __name__ == "__main__":
    main()
