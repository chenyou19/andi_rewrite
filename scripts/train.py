"""新版 ANDi framework 的 DDPM training CLI 入口。

這個 script 只負責把 config -> components -> Trainer 接起來。
model、noise、data 與 training policy 都留在各自模組，方便實驗乾淨替換。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import torch

from andi_rewrite.data import build_dataloader
from andi_rewrite.diffusion.ddpm import build_diffusion
from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.engine import Trainer, VolumeEvaluator
from andi_rewrite.engine.checkpoint import unwrap_model
from andi_rewrite.models import build_model
from andi_rewrite.noise import build_noise_plan
from andi_rewrite.utils.reporting import save_inference_report, save_training_report
from andi_rewrite.utils import load_config, print_config, set_seed, summarize_components


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_training_backend(runtime: dict) -> None:
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))


def build_accelerator(config: dict):
    """只有 YAML 要求 distributed execution 時才建立 Accelerator。"""

    runtime = config.get("runtime", {})
    if not bool(runtime.get("accelerate", runtime.get("distributed", False))):
        return None
    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
    except ImportError as exc:
        raise ImportError("runtime.accelerate requires the optional 'accelerate' package.") from exc

    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(runtime.get("find_unused_parameters", True))
    )
    return Accelerator(kwargs_handlers=[kwargs])


def build_trainer_from_config(config: dict) -> tuple[Trainer, object]:
    """從 YAML 建立所有 training components，不在 script 寫死 experiment 邏輯。"""

    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 73))
    set_seed(seed)
    configure_training_backend(runtime)
    accelerator = build_accelerator(config)
    device = accelerator.device if accelerator is not None else resolve_device(str(runtime.get("device", "auto")))
    dataloader = build_dataloader(config.get("data", {}))
    model = build_model(config.get("model", {}), device=device)
    diffusion = build_diffusion(config.get("diffusion", {}), device=device)
    noise_plan = build_noise_plan(config.get("noise", {}))
    # Trainer 負責 optimizer/scheduler/EMA/checkpointing；
    # script 層只決定要從 config 建立哪些 component implementation。
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        noise_plan=noise_plan,
        config=config.get("training", {}),
        device=device,
        steps_per_epoch=len(dataloader),
        accelerator=accelerator,
    )
    dataloader = trainer.prepare(dataloader)
    return trainer, dataloader


def load_model_if_configured(model: torch.nn.Module, config: dict, device: torch.device) -> None:
    checkpoint_path = config.get("model", {}).get("checkpoint")
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location=device)
    key = "ema_model" if config.get("model", {}).get("use_ema", False) and "ema_model" in payload else "model"
    state = payload[key] if isinstance(payload, dict) and key in payload else payload
    unwrap_model(model).load_state_dict(state)


def build_detector_from_config(config: dict) -> tuple[ANDiDetector, object | None]:
    seed = int(config.get("runtime", {}).get("seed", 73))
    set_seed(seed)
    accelerator = build_accelerator(config)
    device = accelerator.device if accelerator is not None else resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    model = build_model(config.get("model", {}), device=device)
    load_model_if_configured(model, config, device)
    diffusion = build_diffusion(config.get("diffusion", {}), device=device)
    noise_plan = build_noise_plan(config.get("noise", {}))
    detector = ANDiDetector(
        model=model,
        diffusion=diffusion,
        noise_plan=noise_plan,
        config=config.get("anomaly", {}),
        device=device,
    )
    return detector, accelerator


def run_evaluation_from_config(eval_config_path: str | Path) -> dict:
    eval_config = load_config(eval_config_path)
    detector, accelerator = build_detector_from_config(eval_config)
    print("Loaded eval config:")
    print_config(eval_config)
    print("Built eval components:")
    print_config(summarize_components(detector=detector, diffusion=detector.diffusion, noise=detector.noise_plan))
    dataloader = build_dataloader(eval_config.get("data", {}))
    evaluator = VolumeEvaluator(
        detector=detector,
        config={**eval_config.get("data", {}), **eval_config.get("metrics", {}), **eval_config.get("evaluation", {})},
        accelerator=accelerator,
    )
    dataloader = evaluator.prepare(dataloader)
    eval_start = datetime.now().astimezone()
    result = evaluator.evaluate(dataloader)
    eval_end = datetime.now().astimezone()
    if result:
        print("Evaluation result:")
        print_config(result)
        if evaluator.is_main_process:
            save_inference_report(
                config=eval_config,
                evaluator=evaluator,
                result=result,
                dataloader=dataloader,
                start_time=eval_start,
                end_time=eval_end,
                config_path=eval_config_path,
                cli_args={"config": str(eval_config_path), "trigger": "eval_after_fit"},
            )
    return result


def resolve_eval_after_fit(config: dict, cli_eval_config: str | None) -> str | None:
    if cli_eval_config:
        return cli_eval_config
    eval_after_fit = config.get("training", {}).get("eval_after_fit")
    if isinstance(eval_after_fit, dict):
        if not bool(eval_after_fit.get("enabled", True)):
            return None
        path = eval_after_fit.get("config")
        return str(path) if path else None
    if eval_after_fit:
        return str(eval_after_fit)
    return None


def main() -> None:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    parser = argparse.ArgumentParser(description="Build the rewritten ANDi training pipeline.")
    parser.add_argument("--config", default=str(default_config), help="Path to a YAML training config.")
    parser.add_argument("--run-one-step", action="store_true", help="Run one dataloader training step.")
    parser.add_argument("--fit", action="store_true", help="Run the configured training loop.")
    parser.add_argument("--sample-once", action="store_true", help="Generate and save one sample grid without training.")
    parser.add_argument("--eval-config", help="Run this eval config automatically after --fit completes.")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer, dataloader = build_trainer_from_config(config)
    print("Loaded training config:")
    print_config(config)
    print("Built components:")
    print_config(summarize_components(trainer=trainer, diffusion=trainer.diffusion, noise=trainer.noise_plan))

    if args.run_one_step:
        batch = next(iter(dataloader))
        result = trainer.train_step(batch, epoch=0)
        print("One-step training result:")
        print_config(result)
    elif args.sample_once:
        path = trainer.save_samples(epoch=0, prefix="sample")
        print("Sample grid saved:")
        print_config({"path": str(path)})
    elif args.fit:
        train_start = datetime.now().astimezone()
        trainer.fit(dataloader)
        train_end = datetime.now().astimezone()
        if trainer.is_main_process:
            save_training_report(
                config=config,
                trainer=trainer,
                dataloader=dataloader,
                start_time=train_start,
                end_time=train_end,
                config_path=args.config,
                cli_args=vars(args),
            )
        eval_config = resolve_eval_after_fit(config, args.eval_config)
        if eval_config:
            print("Training complete. Running evaluation:")
            print_config({"eval_config": eval_config})
            run_evaluation_from_config(eval_config)
    else:
        print("Dry run complete. Use --run-one-step, --sample-once, or --fit.")


if __name__ == "__main__":
    main()
