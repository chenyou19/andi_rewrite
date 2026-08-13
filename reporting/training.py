from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .metadata import (
    _dataset_len,
    _diffusion_noise_settings,
    _first_present,
    _git_commit_hash,
    _get_attr,
    _infer_healthy_only,
    _infer_z_balanced,
    _parameter_count,
    _run_name,
)
from .serialization import (
    _dict_table,
    _duration_seconds,
    _format_time,
    _json_safe,
    safe_get,
    snapshot_config,
)


def _build_training_report(
    *,
    config: dict[str, Any],
    trainer: Any,
    dataloader: Any,
    start_time: datetime | None,
    end_time: datetime | None,
    config_path: str | Path | None,
    cli_args: dict[str, Any] | None,
    report_dir: Path,
) -> dict[str, Any]:
    training = config.get("training", {})
    data = config.get("data", {})
    model = config.get("model", {})
    diffusion = config.get("diffusion", {})
    noise = config.get("noise", {})
    scheduler = training.get("scheduler", {})
    optimizer = _get_attr(trainer, "optimizer")
    optimizer_group = (optimizer.param_groups[0] if getattr(optimizer, "param_groups", None) else {})
    run_name = _run_name(config, training)
    checkpoint_run_dir = Path(_get_attr(trainer, "checkpoint_dir") or training.get("checkpoint", {}).get("dir", "outputs/checkpoints")) / str(run_name)
    sample_path = _get_attr(trainer, "last_sample_path")
    checkpoint_path = _get_attr(trainer, "last_checkpoint_path")

    return {
        "basic_info": {
            "experiment_name": safe_get(config, "experiment.name", run_name),
            "start_time": _format_time(start_time),
            "end_time": _format_time(end_time),
            "total_training_time_seconds": _duration_seconds(start_time, end_time),
            "device": str(_get_attr(trainer, "device", safe_get(config, "runtime.device"))),
            "seed": safe_get(config, "runtime.seed"),
            "config_path": str(config_path or config.get("_config_path") or ""),
            "cli_args": _json_safe(cli_args or {}),
            "output_directory": str(report_dir),
            "checkpoint_output_directory": str(checkpoint_run_dir),
            "git_commit_hash": _git_commit_hash(),
        },
        "dataset_settings": {
            "data_type": data.get("type"),
            "data_path": _first_present(data.get("path"), data.get("dataset_path")),
            "csv_path": data.get("path_to_csv"),
            "image_size": data.get("image_size"),
            "channels": data.get("channels", model.get("in_channels")),
            "batch_size": data.get("batch_size"),
            "workers": data.get("workers"),
            "shuffle": data.get("shuffle"),
            "number_of_training_samples": _dataset_len(dataloader),
            "slice_selection_method": _first_present(data.get("slice_selection_method"), data.get("sampling_mode")),
            "healthy_slices_only": _infer_healthy_only(data),
            "z_balanced_sampling": _infer_z_balanced(data),
            "samples_per_z": _first_present(data.get("samples_per_z"), data.get("per_z_count")),
        },
        "model_settings": {
            "model_type": model.get("type"),
            "image_size": model.get("image_size"),
            "in_channels": _first_present(model.get("in_channels"), model.get("channels")),
            "base_channels": model.get("base_channels"),
            "hidden_channels": model.get("hidden_channels"),
            "channel_multipliers": _first_present(model.get("channel_mults"), model.get("channel_multipliers")),
            "parameter_count": _parameter_count(_get_attr(trainer, "model")),
            "checkpoint_path": _first_present(model.get("checkpoint"), safe_get(training, "checkpoint.resume"), training.get("resume")),
        },
        "diffusion_noise_settings": _diffusion_noise_settings(diffusion, noise, _get_attr(trainer, "noise_plan")),
        "optimizer_scheduler_settings": {
            "epochs": training.get("epochs"),
            "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
            "learning_rate": optimizer_group.get("lr"),
            "start_lr": _first_present(scheduler.get("start_lr"), training.get("start_lr")),
            "target_lr": _first_present(scheduler.get("target_lr"), training.get("target_lr"), training.get("learning_rate")),
            "warmup_steps": _first_present(scheduler.get("warmup_steps"), training.get("warmup_steps")),
            "weight_decay": optimizer_group.get("weight_decay"),
            "ema_decay": _first_present(safe_get(training, "ema.decay"), training.get("ema_decay")),
            "mixed_precision": _first_present(safe_get(config, "runtime.mixed_precision"), training.get("mixed_precision")),
            "gradient_clipping": _first_present(training.get("gradient_clipping"), training.get("grad_clip")),
            "gradient_accumulation": _first_present(training.get("gradient_accumulation"), training.get("accumulation_steps")),
            "scheduler_type": scheduler.get("type"),
        },
        "training_result": {
            "final_loss": _get_attr(trainer, "last_loss"),
            "best_loss": _get_attr(trainer, "best_loss"),
            "best_checkpoint_path": None,
            "final_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "sample_output_path": str(sample_path) if sample_path else None,
            "loss_curve_path": _first_present(training.get("loss_curve_path"), safe_get(training, "logging.loss_curve_path")),
            "last_epoch": _get_attr(trainer, "last_epoch"),
        },
        "full_config_snapshot": snapshot_config(config),
    }


def _training_markdown(report: dict[str, Any]) -> str:
    sections = [
        "# Training Report",
        "## Basic Info",
        _dict_table(report["basic_info"]),
        "## Dataset Settings",
        _dict_table(report["dataset_settings"]),
        "## Model Settings",
        _dict_table(report["model_settings"]),
        "## Diffusion and Noise Settings",
        _dict_table(report["diffusion_noise_settings"]),
        "## Optimizer and Scheduler Settings",
        _dict_table(report["optimizer_scheduler_settings"]),
        "## Training Result",
        _dict_table(report["training_result"]),
        "## Full Config Snapshot",
        "```json",
        json.dumps(report["full_config_snapshot"], indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n\n".join(sections) + "\n"
