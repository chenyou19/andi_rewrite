"""ANDi evaluation 的 CLI 入口。

此 script 負責建立 detector 並執行會寫出 CSV 的完整 volume evaluation。
metric 與 postprocess 邏輯刻意不放在這裡，而是由 engine/evaluation/
與 anomaly/postprocess/ 的責任模組擁有；engine/evaluator.py 保留為 facade。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import torch

from andi_rewrite.anomaly import (
    ANDiDetector,
    build_postprocess_policy,
    supported_threshold_methods,
)
from andi_rewrite.anomaly.inference_profiles import INFERENCE_PROFILES, resolve_inference_profile
from andi_rewrite.data import build_dataloader
from andi_rewrite.diffusion.ddpm import build_diffusion
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.engine.checkpoint import unwrap_model
from andi_rewrite.engine.evaluation.timing import write_timing
from andi_rewrite.models import build_model
from andi_rewrite.noise import build_noise_plan
from andi_rewrite.utils.reporting import save_inference_report
from andi_rewrite.utils import load_config, print_config, set_seed, summarize_components


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_evaluation_backend(runtime: dict) -> None:
    """Apply the same explicit cuDNN policy used by controlled training runs."""

    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(runtime.get("deterministic", False))


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

    profile_selection = config.get("evaluation", {}).get("inference_profile")
    profile = resolve_inference_profile(profile_selection)
    runtime = config.get("runtime", {})
    anomaly = config.get("anomaly", {})
    model_config = config.get("model", {})
    diffusion_config = config.get("diffusion", {})
    amp_dtype = runtime.get("amp_dtype", "none")
    if profile_selection == "original":
        anomaly = {**anomaly, "num_timesteps": None, "streaming": False}
        model_config = {**model_config, "attention_backend": "legacy"}
        amp_dtype = "none"
    elif profile != "original":
        steps = int(diffusion_config.get("steps", diffusion_config.get("noise_steps", 1000)))
        if steps < 200:
            raise ValueError(f"evaluation.inference_profile={profile!r} requires diffusion.steps >= 200.")
        runtime = {**runtime, "cudnn_benchmark": False}
        anomaly = {
            **anomaly,
            "t_lower": 75,
            "t_upper": 200,
            "num_timesteps": 12 if profile == "fast12" else None,
            "streaming": True,
        }
        model_config = {**model_config, "attention_backend": "no_weights"}

    seed = int(runtime.get("seed", 73))
    set_seed(seed)
    configure_evaluation_backend(runtime)
    accelerator = build_accelerator(config)
    device = accelerator.device if accelerator is not None else resolve_device(str(runtime.get("device", "auto")))
    if profile != "original":
        amp_dtype = "float16" if device.type == "cuda" else "none"
    model = build_model(model_config, device=device)
    load_model_if_configured(model, config, device)
    diffusion = build_diffusion(diffusion_config, device=device)
    noise_plan = build_noise_plan(config.get("noise", {}))
    postprocess_policy = build_postprocess_policy(
        config.get("metrics", {}),
        anomaly_config=anomaly,
        median_cuda_device=device if profile != "original" and device.type == "cuda" else None,
    )
    detector = ANDiDetector(
        model=model,
        diffusion=diffusion,
        noise_plan=noise_plan,
        config=anomaly,
        device=device,
        postprocess_policy=postprocess_policy,
        inference_profile=profile,
        amp_dtype=amp_dtype,
    )
    return detector, accelerator


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"
    parser = argparse.ArgumentParser(description="Build the rewritten ANDi evaluation pipeline.")
    parser.add_argument("--config", default=str(default_config), help="Path to a YAML eval config.")
    parser.add_argument("--run-eval", action="store_true", help="Run full volume evaluation and write CSV files.")
    parser.add_argument("--timing", action="store_true", help="Measure evaluation stages and write timing artifacts.")
    parser.add_argument(
        "--inference-profile",
        choices=INFERENCE_PROFILES,
        help="Override evaluation.inference_profile (default: original).",
    )
    parser.add_argument(
        "--threshold-method",
        choices=supported_threshold_methods(),
        default=None,
        help=(
            "Override metrics.threshold_method "
            f"(registered: {', '.join(supported_threshold_methods())}; built-in default: yen)."
        ),
    )
    return parser


def apply_threshold_method_override(config: dict, method: str | None) -> dict:
    """Apply the CLI threshold choice through the existing config structure."""

    if method is None:
        return config
    normalized = str(method).strip().lower()
    methods = supported_threshold_methods()
    if normalized not in methods:
        supported = ", ".join(methods)
        raise ValueError(f"Unknown threshold method: {method!r}. Supported methods: {supported}.")
    metrics = config.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a mapping before applying --threshold-method.")
    metrics["threshold_method"] = normalized

    # Keep the legacy detector-facing adaptive setting coherent in config
    # snapshots. Numeric fixed thresholds remain untouched.
    anomaly = config.get("anomaly")
    if isinstance(anomaly, dict):
        legacy_threshold = str(anomaly.get("threshold", "yen")).strip().lower()
        if legacy_threshold in methods:
            anomaly["threshold"] = normalized
    return config


def main() -> None:
    args = build_parser().parse_args()
    cli_started = perf_counter()

    config = load_config(args.config)
    if args.inference_profile is not None:
        config.setdefault("evaluation", {})["inference_profile"] = args.inference_profile
    if args.timing:
        config.setdefault("evaluation", {}).setdefault("timing", {})["enabled"] = True
    apply_threshold_method_override(config, args.threshold_method)
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
                "_run_config": config,
            },
            accelerator=accelerator,
        )
        dataloader = evaluator.prepare(dataloader)
        eval_start = datetime.now().astimezone()
        cli_setup_seconds = perf_counter() - cli_started if evaluator.timing_enabled else 0.0
        result = evaluator.evaluate(dataloader)
        eval_end = datetime.now().astimezone()
        if result:
            print("Evaluation result:")
            print_config(result)
            if evaluator.is_main_process:
                report_started = perf_counter() if evaluator.timing_enabled else 0.0
                report_dir = save_inference_report(
                    config=config,
                    evaluator=evaluator,
                    result=result,
                    dataloader=dataloader,
                    start_time=eval_start,
                    end_time=eval_end,
                    config_path=args.config,
                    cli_args=vars(args),
                )
                if evaluator.timing_enabled:
                    report_finished = perf_counter()
                    write_timing(
                        report_dir or evaluator.output_csv.parent,
                        result["timing"],
                        evaluator.last_timing_subjects,
                        cli_wall_seconds=report_finished - cli_started,
                        cli_setup_seconds=cli_setup_seconds,
                        cli_report_seconds=report_finished - report_started,
                    )
    else:
        print("Configuration loaded. Use --run-eval to run full volume evaluation.")


if __name__ == "__main__":
    main()
