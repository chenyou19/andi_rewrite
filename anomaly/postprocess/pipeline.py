"""Pipeline compilation and execution for postprocessing steps."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import torch

from .base import BasePostprocessor, MASK_POSTPROCESSORS, SCORE_POSTPROCESSORS
from .transforms import NormalizePostprocessor


def _legacy_score_pipeline(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the legacy nested score settings to their ordered pipeline."""

    pipeline = []
    if config.get("gray_dilation", {}).get("enabled", False):
        pipeline.append({"type": "gray_dilation", **config["gray_dilation"]})
    if config.get("median_filter", {}).get("enabled", False):
        pipeline.append({"type": "median_filter", **config["median_filter"]})
    if config.get("normalize", False):
        normalize_config = config.get("normalize")
        if isinstance(normalize_config, dict):
            pipeline.append({"type": "normalize", **normalize_config})
        else:
            pipeline.append({"type": "normalize"})
    return pipeline


def _legacy_mask_pipeline(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the legacy nested binary-mask settings to an ordered pipeline."""

    pipeline = []
    if config.get("binary_dilation", {}).get("enabled", False):
        pipeline.append({"type": "binary_dilation", **config["binary_dilation"]})
    if config.get("connected_components", {}).get("enabled", False):
        pipeline.append({"type": "connected_components", **config["connected_components"]})
    return pipeline


def build_postprocess_pipeline(
    config: dict[str, Any] | None,
    registry: dict[str, type[BasePostprocessor]],
    legacy_builder: Callable[[dict[str, Any]], list[dict[str, Any]]],
    step_kind: str | None = None,
) -> list[BasePostprocessor]:
    if not config:
        return []
    step_configs = config.get("pipeline")
    if step_configs is None:
        warnings.warn(
            "Nested postprocess settings without an explicit 'pipeline' are deprecated; "
            "they were translated to an ordered pipeline for backward compatibility.",
            FutureWarning,
            stacklevel=2,
        )
        step_configs = legacy_builder(config)
    if isinstance(step_configs, dict):
        step_configs = [step_configs]

    steps = []
    for step_config in step_configs:
        if not step_config or step_config.get("enabled", True) is False:
            continue
        step_type = str(step_config.get("type")).lower()
        if step_type not in registry:
            supported = ", ".join(registry.keys())
            label = f"{step_kind} postprocess" if step_kind else "postprocess"
            message = (
                f"Unknown {label} step: {step_type}.\n"
                f"Supported {label} steps: {supported}."
            )
            if step_kind == "mask" and step_type in {"yen_threshold", "otsu_threshold"}:
                message += (
                    f"\nNote: {step_type} is a score-to-mask thresholding stage "
                    "inside PostprocessPolicy and should not be configured as a "
                    "mask postprocess step."
                )
            raise ValueError(message)
        kwargs = {
            key: value
            for key, value in step_config.items()
            if key not in {"type", "enabled"}
        }
        steps.append(registry[step_type](**kwargs))
    return steps


def apply_postprocess_pipeline(
    tensor: torch.Tensor,
    steps: list[BasePostprocessor],
    normalization_scope: str | None = None,
) -> torch.Tensor:
    """Apply configured score-map or mask steps in their declared order."""

    output = tensor
    for step in steps:
        if isinstance(step, NormalizePostprocessor):
            output = step(output, scope=normalization_scope)
        else:
            output = step(output)
    return output


def apply_score_postprocess(
    tensor: torch.Tensor,
    config: dict[str, Any] | None,
    normalization_scope: str | None = None,
) -> torch.Tensor:
    """Apply configured score-map postprocessing."""

    return apply_postprocess_pipeline(
        tensor,
        build_postprocess_pipeline(config, SCORE_POSTPROCESSORS, _legacy_score_pipeline, "score-map"),
        normalization_scope=normalization_scope,
    )


def apply_mask_postprocess(tensor: torch.Tensor, config: dict[str, Any] | None) -> torch.Tensor:
    """Apply configured binary-mask postprocessing."""

    output = apply_postprocess_pipeline(
        tensor.bool(),
        build_postprocess_pipeline(config, MASK_POSTPROCESSORS, _legacy_mask_pipeline, "mask"),
    )
    return output.bool()
