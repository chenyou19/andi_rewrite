"""Configuration migration and postprocessing-policy construction."""

from __future__ import annotations

import warnings
from typing import Any

from .base import SUPPORTED_THRESHOLD_METHODS
from .policies import OriginalANDiPostprocessPolicy, PostprocessPolicy, RewritePostprocessPolicy
from .threshold import _validate_threshold_method


def build_postprocess_policy(
    config: dict[str, Any] | None,
    anomaly_config: dict[str, Any] | None = None,
    *,
    warn_legacy: bool = True,
    legacy_profile: str = "evaluator",
) -> PostprocessPolicy:
    """Build an explicit policy while retaining legacy rewrite configuration."""

    config = config or {}
    anomaly_config = anomaly_config or {}
    legacy_anomaly_threshold = str(anomaly_config.get("threshold", "")).strip().lower()
    default_threshold_method = (
        legacy_anomaly_threshold
        if legacy_anomaly_threshold in SUPPORTED_THRESHOLD_METHODS
        else "yen"
    )
    threshold_method = _validate_threshold_method(
        config.get("threshold_method", anomaly_config.get("threshold_method", default_threshold_method))
    )
    configured_mode = config.get("postprocess_mode")
    legacy_compatibility = configured_mode in (None, "")
    mode = "rewrite" if legacy_compatibility else str(configured_mode).strip().lower()
    if legacy_compatibility and warn_legacy:
        warnings.warn(
            "metrics.postprocess_mode is not set; retaining the legacy-compatible "
            "rewrite postprocessing path. Set postprocess_mode: rewrite explicitly "
            "for new configs, or postprocess_mode: original_andi for reference behavior.",
            FutureWarning,
            stacklevel=2,
        )
    if mode == "original_andi":
        original_config = config.get("original_andi", {})
        if not isinstance(original_config, dict):
            raise TypeError("metrics.original_andi must be a mapping.")
        if "eps" not in original_config:
            original_config = {
                **original_config,
                "eps": config.get("eps", anomaly_config.get("eps", 1.0e-8)),
            }
        return OriginalANDiPostprocessPolicy(
            original_config,
            threshold_method=threshold_method,
        )
    if mode == "rewrite":
        return RewritePostprocessPolicy(
            config,
            anomaly_config,
            threshold_method=threshold_method,
            legacy_compatibility=legacy_compatibility,
            legacy_profile=legacy_profile,
        )
    raise ValueError(
        f"Unknown metrics.postprocess_mode: {configured_mode!r}. "
        "Supported modes: original_andi, rewrite."
    )
