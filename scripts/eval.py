"""ANDi evaluation 的 CLI 入口。

此 script 負責建立 detector 並執行會寫出 CSV 的完整 volume evaluation。
metric 與 postprocess 邏輯刻意不放在這裡，而是放在 engine/evaluator.py
與 anomaly/postprocess.py。
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

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.data import build_dataloader
from andi_rewrite.diffusion.ddpm import build_diffusion
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.engine.checkpoint import unwrap_model
from andi_rewrite.models import build_model
from andi_rewrite.noise import build_noise_plan
from andi_rewrite.utils.reporting import save_inference_report
from andi_rewrite.utils import load_config, print_config, set_seed, summarize_components


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_accelerator(config: dict):
    """只有 runtime.accelerate 要求時才建立 Accelerator。"""

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


def load_model_if_configured(model: torch.nn.Module, config: dict, device: torch.device) -> None:
    """依 config 載入 raw/framework checkpoint；若指定 use_ema 則優先使用 EMA 權重。"""

    checkpoint_path = config.get("model", {}).get("checkpoint")
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location=device)
    key = "ema_model" if config.get("model", {}).get("use_ema", False) and "ema_model" in payload else "model"
    state = payload[key] if isinstance(payload, dict) and key in payload else payload
    try:
        unwrap_model(model).load_state_dict(state)
    except RuntimeError as exc:
        model_config = config.get("model", {})
        raise RuntimeError(
            "Checkpoint is not compatible with the configured model. "
            f"checkpoint={checkpoint_path}, state_key={key}, "
            f"in_channels={model_config.get('in_channels', model_config.get('channels'))}, "
            f"out_channels={model_config.get('out_channels', model_config.get('channels'))}."
        ) from exc


def build_detector_from_config(config: dict) -> tuple[ANDiDetector, object | None]:
    """從 YAML 建立 detector 與選擇性的 distributed runtime。"""

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


def main() -> None:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"
    parser = argparse.ArgumentParser(description="Build the rewritten ANDi evaluation pipeline.")
    parser.add_argument("--config", default=str(default_config), help="Path to a YAML eval config.")
    parser.add_argument("--run-eval", action="store_true", help="Run full volume evaluation and write CSV files.")
    args = parser.parse_args()

    config = load_config(args.config)
    detector, accelerator = build_detector_from_config(config)
    print("Loaded eval config:")
    print_config(config)
    print("Built components:")
    print_config(summarize_components(detector=detector, diffusion=detector.diffusion, noise=detector.noise_plan))

    if args.run_eval:
        dataloader = build_dataloader(config.get("data", {}))
        evaluator = VolumeEvaluator(
            detector=detector,
            config={
                **config.get("data", {}),
                **config.get("metrics", {}),
                **config.get("evaluation", {}),
                "prediction_output": config.get("prediction_output", {}),
                "model": config.get("model", {}),
                "anomaly": config.get("anomaly", {}),
            },
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
                    config=config,
                    evaluator=evaluator,
                    result=result,
                    dataloader=dataloader,
                    start_time=eval_start,
                    end_time=eval_end,
                    config_path=args.config,
                    cli_args=vars(args),
                )
    else:
        print("Configuration loaded. Use --run-eval to run full volume evaluation.")


if __name__ == "__main__":
    main()
