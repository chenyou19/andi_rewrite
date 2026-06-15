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

from andi_rewrite.runtime.builders import (
    build_detector_from_config,
    build_evaluator_from_config,
    build_trainer_from_config,
)
from andi_rewrite.utils.reporting import save_inference_report, save_training_report
from andi_rewrite.utils import load_config, print_config, summarize_components


def run_evaluation_from_config(eval_config_path: str | Path) -> dict:
    eval_config = load_config(eval_config_path)
    detector, accelerator = build_detector_from_config(eval_config)
    print("Loaded eval config:")
    print_config(eval_config)
    print("Built eval components:")
    print_config(summarize_components(detector=detector, diffusion=detector.diffusion, noise=detector.noise_plan))
    evaluator, dataloader = build_evaluator_from_config(eval_config, detector, accelerator)
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
