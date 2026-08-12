"""Postprocessing policies and their frozen execution descriptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ._runtime import apply_median_filter_tensor, apply_normalize_minmax
from .base import BasePostprocessor, MASK_POSTPROCESSORS, SCORE_POSTPROCESSORS
from .numerics import _validate_normalization_scope, normalize_minmax, sanitize_scores
from .pipeline import (
    _legacy_mask_pipeline,
    _legacy_score_pipeline,
    apply_postprocess_pipeline,
    build_postprocess_pipeline,
)
from .result import PostprocessResult
from .threshold import _validate_threshold_method, threshold_anomaly_map
from .transforms import (
    BinaryDilationPostprocessor,
    ConnectedComponentsPostprocessor,
    GrayDilationPostprocessor,
    MedianFilterPostprocessor,
    NormalizePostprocessor,
    median_filter_tensor,
)


def _step_label(step: BasePostprocessor, normalization_scope: str) -> str:
    if isinstance(step, NormalizePostprocessor):
        return f"{normalization_scope}_minmax"
    if isinstance(step, MedianFilterPostprocessor):
        return f"median_filter_{step.mode.lower()}(kernel={step.kernel_size})"
    if isinstance(step, GrayDilationPostprocessor):
        return f"gray_dilation(kernel={step.kernel_size})"
    if isinstance(step, BinaryDilationPostprocessor):
        return (
            "binary_dilation("
            f"rank={step.rank}, connectivity={step.connectivity}, iterations={step.iterations}"
            ")"
        )
    if isinstance(step, ConnectedComponentsPostprocessor):
        return f"connected_components(min_size={step.min_size}, connectivity={step.connectivity})"
    return step.name


def _pipeline_labels(steps: list[BasePostprocessor], normalization_scope: str) -> list[str]:
    return [_step_label(step, normalization_scope) for step in steps]


@dataclass(frozen=True)
class ScorePipelineSpec:
    """Engine-neutral description of dataset-normalized score preparation.

    Policies that can split their score transforms around dataset-wide
    normalization expose this small value object.  Evaluation backends remain
    responsible for storage and range scans; the policy remains the sole owner
    of transform order and of whether the MF branch consumes the raw map or the
    processed raw score.
    """

    raw_steps: tuple[BasePostprocessor, ...]
    mf_steps: tuple[BasePostprocessor, ...]
    mf_source: str

    def __post_init__(self) -> None:
        if self.mf_source not in {"raw", "score_raw"}:
            raise ValueError("score pipeline mf_source must be 'raw' or 'score_raw'.")


class PostprocessPolicy(ABC):
    """Single source of truth for score, threshold, and mask postprocessing."""

    mode = "base"

    def __init__(
        self,
        *,
        normalization_scope: str,
        threshold_method: str = "yen",
        threshold_mask_config: dict[str, Any] | None = None,
        binary_mask_config: dict[str, Any] | None = None,
        yen_mask_config: dict[str, Any] | None = None,
    ):
        self.normalization_scope = _validate_normalization_scope(normalization_scope)
        self.threshold_method = _validate_threshold_method(threshold_method)
        self.threshold_mask_steps = build_postprocess_pipeline(
            threshold_mask_config,
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )
        selected_mask_config = binary_mask_config if binary_mask_config is not None else yen_mask_config
        self.binary_mask_steps = build_postprocess_pipeline(
            selected_mask_config,
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )
        # Attribute retained for callers that inspect the old Yen-specific name.
        self.yen_mask_steps = self.binary_mask_steps

    @abstractmethod
    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        """Derive both score branches and their selected-method masks."""

    def _complete(
        self,
        score_raw: torch.Tensor,
        score_mf: torch.Tensor,
        normalization_scope: str,
    ) -> PostprocessResult:
        score_raw = sanitize_scores(score_raw)
        score_mf = sanitize_scores(score_mf)
        binary_mask_raw, thresholds_raw = threshold_anomaly_map(
            score_raw,
            method=self.threshold_method,
        )
        binary_mask_mf, thresholds_mf = threshold_anomaly_map(
            score_mf,
            method=self.threshold_method,
        )
        binary_mask_raw_postprocessed = apply_postprocess_pipeline(
            binary_mask_raw,
            self.binary_mask_steps,
        ).bool()
        binary_mask_mf_postprocessed = apply_postprocess_pipeline(
            binary_mask_mf,
            self.binary_mask_steps,
        ).bool()
        return PostprocessResult(
            score_raw=score_raw,
            score_mf=score_mf,
            thresholds_raw=thresholds_raw,
            thresholds_mf=thresholds_mf,
            binary_mask_raw=binary_mask_raw,
            binary_mask_mf=binary_mask_mf,
            binary_mask_raw_postprocessed=binary_mask_raw_postprocessed,
            binary_mask_mf_postprocessed=binary_mask_mf_postprocessed,
            threshold_method=self.threshold_method,
            normalization_scope=normalization_scope,
        )

    def complete_scores(
        self,
        score_raw: torch.Tensor,
        score_mf: torch.Tensor,
        normalization_scope: str,
    ) -> PostprocessResult:
        """Complete prepared score branches through the public policy contract.

        Delegating to ``_complete`` intentionally preserves the historical
        subclass override seam while keeping evaluation backends out of the
        policy's protected implementation.
        """

        return self._complete(score_raw, score_mf, normalization_scope)

    def score_pipeline_spec(self) -> ScorePipelineSpec | None:
        """Describe dataset-streamable score transforms, when supported.

        Subject-normalized execution only needs :meth:`process`.  A custom
        policy opts into dataset-normalized streaming by overriding this hook;
        returning ``None`` keeps existing custom policies source-compatible.
        """

        return None

    def fixed_threshold_mask(self, score: torch.Tensor, threshold: float) -> torch.Tensor:
        mask = sanitize_scores(score) > float(threshold)
        return apply_postprocess_pipeline(mask, self.threshold_mask_steps).bool()

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Describe the exact executable order for reports and debugging."""


class OriginalANDiPostprocessPolicy(PostprocessPolicy):
    """Reference-compatible postprocessing from AlexanderFrotscher/ANDi."""

    mode = "original_andi"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        eps: float = 1.0e-8,
        *,
        threshold_method: str = "yen",
    ):
        try:
            from scipy import ndimage as _scipy_ndimage  # noqa: F401
            from skimage import filters as _skimage_filters  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "original_andi postprocessing requires scipy and scikit-image; "
                "the rewrite fallbacks are not reference-equivalent."
            ) from exc
        config = config or {}
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", True))
        self.median_kernel_size = int(median_config.get("kernel_size", 5))
        self.median_mode = str(median_config.get("mode", "3d")).lower()
        if self.median_mode != "3d":
            raise ValueError("original_andi median_filter.mode must be '3d'.")
        self.eps = float(config.get("eps", eps))

        binary_mask_config = config.get("binary_mask")
        if not isinstance(binary_mask_config, dict):
            binary_mask_config = config.get("yen", {})
        dilation_config = binary_mask_config.get("binary_dilation", {})
        dilation_enabled = bool(dilation_config.get("enabled", True))
        self.dilation_settings = {
            "enabled": dilation_enabled,
            "rank": int(dilation_config.get("rank", 3)),
            "connectivity": int(dilation_config.get("connectivity", 1)),
            "iterations": int(dilation_config.get("iterations", 1)),
        }
        binary_mask_pipeline: list[dict[str, Any]] = []
        if dilation_enabled:
            binary_mask_pipeline.append({"type": "binary_dilation", **self.dilation_settings})
        normalization_scope = _validate_normalization_scope(
            str(config.get("normalization_scope", "dataset"))
        )
        if normalization_scope != "dataset":
            raise ValueError(
                "metrics.original_andi.normalization_scope must be 'dataset' to reproduce "
                "the reference evaluation. Use prediction_output.normalization_scope for "
                "an explicitly different export scope."
            )
        super().__init__(
            normalization_scope=normalization_scope,
            threshold_method=threshold_method,
            threshold_mask_config={"pipeline": []},
            binary_mask_config={"pipeline": binary_mask_pipeline},
        )

    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        scope = _validate_normalization_scope(normalization_scope or self.normalization_scope)
        raw_finite = sanitize_scores(raw_maps)
        raw_mf = (
            apply_median_filter_tensor(
                raw_finite,
                kernel_size=self.median_kernel_size,
                mode=self.median_mode,
            )
            if self.median_enabled
            else raw_finite.clone()
        )
        # The branches are intentionally normalized independently and only
        # after the MF branch has filtered the unnormalized raw anomaly map.
        score_raw = apply_normalize_minmax(raw_finite, eps=self.eps, scope=scope)
        score_mf = apply_normalize_minmax(raw_mf, eps=self.eps, scope=scope)
        return self._complete(score_raw, score_mf, scope)

    def score_pipeline_spec(self) -> ScorePipelineSpec:
        mf_steps: list[BasePostprocessor] = []
        if self.median_enabled:
            mf_steps.append(
                MedianFilterPostprocessor(
                    kernel_size=self.median_kernel_size,
                    mode=self.median_mode,
                )
            )
        mf_steps.append(NormalizePostprocessor(eps=self.eps))
        return ScorePipelineSpec(
            raw_steps=(NormalizePostprocessor(eps=self.eps),),
            mf_steps=tuple(mf_steps),
            mf_source="raw",
        )

    def describe(self) -> dict[str, Any]:
        scope = self.normalization_scope
        mf_pipeline = ["nan_to_num"]
        if self.median_enabled:
            mf_pipeline.append(f"median_filter_3d(kernel={self.median_kernel_size})")
        mf_pipeline.append(f"{scope}_minmax")
        binary_mask_pipeline = [
            f"subject_{self.threshold_method}_threshold",
            *_pipeline_labels(self.binary_mask_steps, scope),
        ]
        description = {
            "postprocess_mode": self.mode,
            "normalization_scope": scope,
            "raw_score_pipeline": ["nan_to_num", f"{scope}_minmax"],
            "mf_score_pipeline": mf_pipeline,
            "threshold_method": self.threshold_method,
            "threshold_strategy": "per_subject_3d_volume",
            "threshold_comparator": ">",
            "binary_mask_pipeline": binary_mask_pipeline,
            "fixed_threshold_mask_pipeline": ["score > threshold"],
            "median_filter_settings": {
                "enabled": self.median_enabled,
                "kernel_size": self.median_kernel_size,
                "mode": self.median_mode,
            },
            "dilation_settings": dict(self.dilation_settings),
            "numerical_safety": {
                "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                "eps": self.eps,
                "constant_tensor": "zeros",
                "constant_volume_threshold": "constant value; empty mask with RuntimeWarning",
            },
        }
        if self.threshold_method == "yen":
            description.update(
                {
                    "yen_threshold_strategy": description["threshold_strategy"],
                    "yen_mask_pipeline": binary_mask_pipeline,
                }
            )
        return description


def _compile_legacy_normalization_steps(
    steps: list[BasePostprocessor],
    *,
    prepend_normalize: bool,
    input_is_normalized: bool,
    ensure_final_normalize: bool,
    eps: float,
) -> list[BasePostprocessor]:
    """Compile old implicit normalization into one explicit, traceable path."""

    compiled: list[BasePostprocessor] = []
    normalized = input_is_normalized
    if prepend_normalize:
        compiled.append(NormalizePostprocessor(eps=eps))
        normalized = True
    for step in steps:
        if isinstance(step, NormalizePostprocessor):
            if not normalized:
                compiled.append(step)
            normalized = True
        else:
            compiled.append(step)
            normalized = False
    if ensure_final_normalize and not normalized:
        compiled.append(NormalizePostprocessor(eps=eps))
    return compiled


class RewritePostprocessPolicy(PostprocessPolicy):
    """Configurable rewrite pipelines, including legacy-compatible defaults."""

    mode = "rewrite"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        anomaly_config: dict[str, Any] | None = None,
        *,
        threshold_method: str = "yen",
        legacy_compatibility: bool = False,
        legacy_profile: str = "evaluator",
    ):
        config = config or {}
        anomaly_config = anomaly_config or {}
        rewrite_config = config.get("rewrite", {}) if isinstance(config.get("rewrite"), dict) else {}
        postprocess_config = config.get("postprocess")
        if not isinstance(postprocess_config, dict):
            postprocess_config = rewrite_config.get("postprocess")
        if not isinstance(postprocess_config, dict):
            postprocess_config = anomaly_config.get("postprocess", {})

        median_config = config.get("median_filter")
        if not isinstance(median_config, dict):
            median_config = anomaly_config.get("median_filter", {})
        median_enabled = bool(median_config.get("enabled", True))
        median_kernel = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        median_mode = str(median_config.get("mode", "3d"))
        self.eps = float(config.get("eps", anomaly_config.get("eps", 1.0e-8)))
        self.legacy_compatibility = bool(legacy_compatibility)
        self.legacy_profile = str(legacy_profile)

        score_config = postprocess_config.get("score")
        score_mf_config = postprocess_config.get("score_mf")
        if score_config is None:
            score_config = {} if legacy_compatibility else {"pipeline": [{"type": "normalize"}]}
        if score_mf_config is None:
            default_mf_pipeline: list[dict[str, Any]] = []
            if median_enabled:
                default_mf_pipeline.append(
                    {"type": "median_filter", "kernel_size": median_kernel, "mode": median_mode}
                )
            if not legacy_compatibility:
                default_mf_pipeline.append({"type": "normalize"})
            score_mf_config = {"pipeline": default_mf_pipeline}

        raw_steps = build_postprocess_pipeline(
            score_config,
            SCORE_POSTPROCESSORS,
            _legacy_score_pipeline,
            "score-map",
        )
        mf_steps = build_postprocess_pipeline(
            score_mf_config,
            SCORE_POSTPROCESSORS,
            _legacy_score_pipeline,
            "score-map",
        )
        if legacy_compatibility and self.legacy_profile == "evaluator":
            self.raw_score_steps = _compile_legacy_normalization_steps(
                raw_steps,
                prepend_normalize=True,
                input_is_normalized=False,
                ensure_final_normalize=True,
                eps=self.eps,
            )
            self.mf_score_steps = _compile_legacy_normalization_steps(
                mf_steps,
                prepend_normalize=False,
                input_is_normalized=True,
                ensure_final_normalize=True,
                eps=self.eps,
            )
        elif legacy_compatibility and self.legacy_profile == "detector":
            self.raw_score_steps = _compile_legacy_normalization_steps(
                raw_steps,
                prepend_normalize=True,
                input_is_normalized=False,
                ensure_final_normalize=False,
                eps=self.eps,
            )
            self.mf_score_steps = mf_steps
        else:
            self.raw_score_steps = raw_steps
            self.mf_score_steps = mf_steps

        rank = int(config.get("rank", 3))
        connectivity = int(config.get("connectivity", 1))
        detector_mask_config = postprocess_config.get("mask", {})
        threshold_mask_config = postprocess_config.get("threshold_mask", detector_mask_config)
        binary_mask_config = postprocess_config.get("binary_mask")
        if binary_mask_config is None:
            binary_mask_config = postprocess_config.get(f"{threshold_method}_mask")
        if binary_mask_config is None and threshold_method != "yen":
            # Existing configs named this shared stage ``yen_mask``. Reusing it
            # keeps Otsu isolated to the threshold algorithm itself.
            binary_mask_config = postprocess_config.get("yen_mask")
        if binary_mask_config is None:
            if self.legacy_profile == "detector" and detector_mask_config:
                binary_mask_config = detector_mask_config
            else:
                binary_mask_config = {
                    "pipeline": [
                        {
                            "type": "binary_dilation",
                            "rank": rank,
                            "connectivity": connectivity,
                            "iterations": 1,
                            "enabled": bool(config.get("yen_binary_dilation", True)),
                        }
                    ]
                }
        normalization_scope = str(
            rewrite_config.get("normalization_scope", config.get("normalization_scope", "dataset"))
        )
        super().__init__(
            normalization_scope=normalization_scope,
            threshold_method=threshold_method,
            threshold_mask_config=threshold_mask_config,
            binary_mask_config=binary_mask_config,
        )

    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        scope = _validate_normalization_scope(normalization_scope or self.normalization_scope)
        raw_finite = sanitize_scores(raw_maps)
        score_raw = apply_postprocess_pipeline(
            raw_finite,
            self.raw_score_steps,
            normalization_scope=scope,
        )
        score_mf = apply_postprocess_pipeline(
            sanitize_scores(score_raw),
            self.mf_score_steps,
            normalization_scope=scope,
        )
        return self._complete(score_raw, score_mf, scope)

    def score_pipeline_spec(self) -> ScorePipelineSpec:
        return ScorePipelineSpec(
            raw_steps=tuple(self.raw_score_steps),
            mf_steps=tuple(self.mf_score_steps),
            mf_source="score_raw",
        )

    def describe(self) -> dict[str, Any]:
        scope = self.normalization_scope
        median_steps = [
            step.describe()
            for step in self.mf_score_steps
            if isinstance(step, MedianFilterPostprocessor)
        ]
        dilation_steps = [
            step.describe()
            for step in self.binary_mask_steps
            if isinstance(step, BinaryDilationPostprocessor)
        ]
        binary_mask_pipeline = [
            f"subject_{self.threshold_method}_threshold",
            *_pipeline_labels(self.binary_mask_steps, scope),
        ]
        description = {
            "postprocess_mode": self.mode,
            "normalization_scope": scope,
            "raw_score_pipeline": ["nan_to_num", *_pipeline_labels(self.raw_score_steps, scope)],
            "mf_score_pipeline": _pipeline_labels(self.mf_score_steps, scope),
            "threshold_method": self.threshold_method,
            "threshold_strategy": "per_subject_3d_volume",
            "threshold_comparator": ">",
            "binary_mask_pipeline": binary_mask_pipeline,
            "fixed_threshold_mask_pipeline": [
                "score > threshold",
                *_pipeline_labels(self.threshold_mask_steps, scope),
            ],
            "median_filter_settings": (
                {"enabled": True, **median_steps[0]} if median_steps else {"enabled": False}
            ),
            "dilation_settings": (
                {"enabled": True, **dilation_steps[0]} if dilation_steps else {"enabled": False}
            ),
            "legacy_compatibility": self.legacy_compatibility,
            "legacy_profile": self.legacy_profile if self.legacy_compatibility else None,
            "numerical_safety": {
                "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                "eps": self.eps,
                "constant_tensor": "zeros",
                "constant_volume_threshold": "constant value; empty mask with RuntimeWarning",
            },
        }
        if self.threshold_method == "yen":
            description.update(
                {
                    "yen_threshold_strategy": description["threshold_strategy"],
                    "yen_mask_pipeline": binary_mask_pipeline,
                }
            )
        return description
