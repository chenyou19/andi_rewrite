"""Stable cache-fingerprint payload construction and JSON-safe conversion."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from andi_rewrite.anomaly.postprocess import PostprocessPolicy
from ..evaluation_cache import file_identity, stable_fingerprint


def json_safe(value: Any) -> Any:
    """Convert metadata/configuration values to the existing JSON-safe form."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def nested_value(mapping: dict[str, Any], *parts: str) -> Any:
    """Read a nested mapping value without coercing malformed config sections."""

    value: Any = mapping
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def cache_fingerprints(
    config: dict[str, Any],
    *,
    model_config: dict[str, Any],
    anomaly_config: dict[str, Any],
    postprocess_policy: PostprocessPolicy,
) -> tuple[str, str]:
    """Construct the byte-compatible raw and score cache fingerprints."""

    run_config = config.get("_run_config")
    if not isinstance(run_config, dict):
        run_config = {
            "data": {
                key: value
                for key, value in config.items()
                if key
                in {
                    "type",
                    "dataset_path",
                    "path",
                    "path_to_csv",
                    "image_size",
                    "channels",
                    "modalities",
                    "histogram_normalization",
                    "shift_naming",
                    "normalize_input",
                    "size_splits",
                }
            },
            "model": model_config,
            "anomaly": anomaly_config,
        }

    fingerprint_anomaly_config = copy.deepcopy(run_config.get("anomaly", {}))
    if isinstance(fingerprint_anomaly_config, dict):
        for key in ("threshold", "median_filter", "postprocess"):
            fingerprint_anomaly_config.pop(key, None)
    runtime_config = run_config.get("runtime", {})
    raw_payload = {
        "implementation": "disk_streaming_v1",
        "data": run_config.get("data", {}),
        "model": run_config.get("model", model_config),
        "diffusion": run_config.get("diffusion", {}),
        "noise": run_config.get("noise", {}),
        "anomaly": fingerprint_anomaly_config,
        "runtime": {
            key: runtime_config.get(key)
            for key in ("seed", "deterministic", "cudnn_benchmark")
            if isinstance(runtime_config, dict) and key in runtime_config
        },
        "file_identities": {
            "checkpoint": file_identity(nested_value(run_config, "model", "checkpoint")),
            "split_csv": file_identity(nested_value(run_config, "data", "path_to_csv")),
            "noise_statistics": file_identity(
                nested_value(run_config, "noise", "schedule", "sampler", "stats_path")
            ),
        },
    }
    raw_fingerprint = stable_fingerprint(json_safe(raw_payload))

    description = postprocess_policy.describe()
    score_payload = {
        "implementation": "disk_streaming_scores_v1",
        "raw_fingerprint": raw_fingerprint,
        "postprocess_mode": description.get("postprocess_mode"),
        "normalization_scope": description.get("normalization_scope"),
        "raw_score_pipeline": description.get("raw_score_pipeline"),
        "mf_score_pipeline": description.get("mf_score_pipeline"),
        "median_filter_settings": description.get("median_filter_settings"),
        "legacy_compatibility": description.get("legacy_compatibility"),
        "legacy_profile": description.get("legacy_profile"),
        "numerical_safety": description.get("numerical_safety"),
    }
    return raw_fingerprint, stable_fingerprint(json_safe(score_payload))
