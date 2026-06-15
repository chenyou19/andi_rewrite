"""共用的 config -> components builder。

scripts/train.py 與 scripts/eval.py 原本各自重複了 device 解析、accelerator
建立、checkpoint 載入與 detector/evaluator 建構等邏輯。這個模組把那些邏輯收斂到
單一來源，行為與原本兩個 script 完全相同，只是不再重複實作。
"""

from __future__ import annotations

import torch

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.data import build_dataloader
from andi_rewrite.diffusion.ddpm import build_diffusion
from andi_rewrite.engine import Trainer, VolumeEvaluator
from andi_rewrite.engine.checkpoint import unwrap_model
from andi_rewrite.models import build_model
from andi_rewrite.noise import build_noise_plan
from andi_rewrite.utils import set_seed


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


def load_model_if_configured(model: torch.nn.Module, config: dict, device: torch.device) -> None:
    """依 config 載入 raw/framework checkpoint；若指定 use_ema 則優先使用 EMA 權重。"""

    checkpoint_path = config.get("model", {}).get("checkpoint")
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location=device)
    key = "ema_model" if config.get("model", {}).get("use_ema", False) and "ema_model" in payload else "model"
    state = payload[key] if isinstance(payload, dict) and key in payload else payload
    unwrap_model(model).load_state_dict(state)


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


def build_evaluator_from_config(
    config: dict,
    detector: ANDiDetector,
    accelerator: object | None = None,
) -> tuple[VolumeEvaluator, object]:
    """建立 volume evaluation 所需的 dataloader 與 VolumeEvaluator。

    config dict 的合併方式與輸出行為和原本兩個 script 完全一致：data/metrics/
    evaluation 三段 config 合併後交給 VolumeEvaluator，dataloader 再經過
    evaluator.prepare() 處理 distributed sharding。
    """

    dataloader = build_dataloader(config.get("data", {}))
    evaluator = VolumeEvaluator(
        detector=detector,
        config={**config.get("data", {}), **config.get("metrics", {}), **config.get("evaluation", {})},
        accelerator=accelerator,
    )
    dataloader = evaluator.prepare(dataloader)
    return evaluator, dataloader
