from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .serialization import _json_safe, safe_get


def _describe(component: Any) -> Any:
    if component is None:
        return None
    if hasattr(component, "describe"):
        try:
            return _json_safe(component.describe())
        except Exception:
            return None
    return str(component)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _diffusion_noise_settings(diffusion: dict[str, Any], noise: dict[str, Any], noise_plan: Any) -> dict[str, Any]:
    sampler = _noise_sampler_config(noise)
    return {
        "noise_steps": diffusion.get("steps"),
        "beta_start": diffusion.get("beta_start"),
        "beta_end": diffusion.get("beta_end"),
        "noise_type": sampler.get("type"),
        "pyramid_noise_settings": sampler if sampler.get("type") == "pyramid" else None,
        "simplex_noise_settings": sampler if sampler.get("type") == "simplex" else None,
        "spectrum_noise_settings": {
            "spectrum_stats_path": sampler.get("stats_path"),
            "spectrum_mode": sampler.get("mode"),
            "spectrum_strength": sampler.get("strength"),
            "spectrum_eps": sampler.get("eps"),
            "spectrum_per_channel": sampler.get("per_channel"),
            "radial_or_full_2d_spectrum": _first_present(sampler.get("spectrum_key"), sampler.get("radial_key")),
            "raw_settings": sampler,
        } if sampler.get("type") in {"spectrum", "empirical_spectrum"} else None,
        "noise_schedule": safe_get(noise, "schedule.type", noise.get("type")),
        "noise_plan": _describe(noise_plan),
    }


def _noise_sampler_config(noise: dict[str, Any]) -> dict[str, Any]:
    schedule = noise.get("schedule", noise)
    if isinstance(schedule, dict) and "sampler" in schedule and isinstance(schedule["sampler"], dict):
        return schedule["sampler"]
    if isinstance(schedule, dict):
        return schedule
    return {}


def _noise_type(noise: dict[str, Any]) -> Any:
    return _noise_sampler_config(noise).get("type")


def _run_name(config: dict[str, Any], training: dict[str, Any]) -> Any:
    return _first_present(training.get("run_name"), safe_get(config, "experiment.name"), "andi_rewrite")


def _dataset_len(dataloader: Any) -> int | None:
    dataset = _get_attr(dataloader, "dataset")
    if dataset is not None:
        try:
            return len(dataset)
        except Exception:
            pass
    try:
        return len(dataloader)
    except Exception:
        return None


def _parameter_count(model: Any) -> int | None:
    if model is None:
        return None
    if hasattr(model, "module"):
        model = model.module
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def _infer_healthy_only(data: dict[str, Any]) -> bool | None:
    data_type = str(data.get("type", "")).lower()
    if "healthy" in data_type:
        return True
    joined = " ".join(str(data.get(key, "")) for key in ["path", "path_to_csv", "dataset_path"]).lower()
    if "healthy" in joined:
        return True
    return data.get("healthy_slices_only")


def _infer_z_balanced(data: dict[str, Any]) -> bool | None:
    if data.get("z_balanced") is not None:
        return bool(data.get("z_balanced"))
    joined = " ".join(str(data.get(key, "")) for key in ["path", "path_to_csv", "sampling_mode"]).lower()
    if "zbalanced" in joined or "z_balanced" in joined or "z-balanced" in joined:
        return True
    return None


def _yen_enabled(config: dict[str, Any]) -> bool:
    threshold = str(safe_get(config, "anomaly.threshold", "")).lower()
    if threshold == "yen":
        return True
    return bool(safe_get(config, "metrics.postprocess.yen_mask"))


def _find_pipeline_step(config: Any, step_type: str) -> dict[str, Any]:
    if isinstance(config, dict):
        if config.get("type") == step_type:
            return config
        for item in config.get("pipeline", []):
            if isinstance(item, dict) and item.get("type") == step_type:
                return item
        nested = config.get(step_type)
        if isinstance(nested, dict):
            return nested
    return {}


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
