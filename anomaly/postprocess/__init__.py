"""Compatibility facade for anomaly-map postprocessing.

The implementation is split by responsibility, while this package preserves the
former ``anomaly.postprocess`` import surface and object identities for callers.
"""

from .base import (
    MASK_POSTPROCESSORS,
    NORMALIZATION_SCOPES,
    SCORE_POSTPROCESSORS,
    SUPPORTED_THRESHOLD_METHODS,
    BasePostprocessor,
    register_mask_postprocessor,
    register_score_postprocessor,
)
from .factory import build_postprocess_policy
from .numerics import _validate_normalization_scope, normalize_minmax, sanitize_scores
from .pipeline import (
    _legacy_mask_pipeline,
    _legacy_score_pipeline,
    apply_mask_postprocess,
    apply_postprocess_pipeline,
    apply_score_postprocess,
    build_postprocess_pipeline,
)
from .policies import (
    OriginalANDiPostprocessPolicy,
    PostprocessPolicy,
    RewritePostprocessPolicy,
    ScorePipelineSpec,
    _compile_legacy_normalization_steps,
    _pipeline_labels,
    _step_label,
)
from .result import PostprocessResult
from .threshold import (
    THRESHOLD_FUNCTION_LOADERS,
    _validate_threshold_method,
    otsu_threshold,
    register_threshold_method,
    supported_threshold_methods,
    threshold_anomaly_map,
    yen_threshold,
)
from .transforms import (
    BinaryDilationPostprocessor,
    ConnectedComponentsPostprocessor,
    GrayDilationPostprocessor,
    MedianFilterPostprocessor,
    NormalizePostprocessor,
    _as_kernel_size,
    _kernel_enabled,
    binary_dilation_tensor,
    connected_components_tensor,
    gray_dilation_tensor,
    median_filter_tensor,
    remove_small_components_tensor,
)
from ._runtime import configure_runtime_functions as _configure_runtime_functions


def _dispatch_median_filter_tensor(
    tensor,
    kernel_size,
    mode="3d",
):
    return median_filter_tensor(tensor, kernel_size, mode)


def _dispatch_normalize_minmax(
    tensor,
    eps=1.0e-8,
    scope="dataset",
):
    return normalize_minmax(tensor, eps, scope)


_configure_runtime_functions(
    median_filter_tensor=_dispatch_median_filter_tensor,
    normalize_minmax=_dispatch_normalize_minmax,
)


__all__ = [
    "BasePostprocessor",
    "BinaryDilationPostprocessor",
    "ConnectedComponentsPostprocessor",
    "GrayDilationPostprocessor",
    "MASK_POSTPROCESSORS",
    "MedianFilterPostprocessor",
    "NORMALIZATION_SCOPES",
    "NormalizePostprocessor",
    "OriginalANDiPostprocessPolicy",
    "PostprocessPolicy",
    "PostprocessResult",
    "RewritePostprocessPolicy",
    "ScorePipelineSpec",
    "SCORE_POSTPROCESSORS",
    "SUPPORTED_THRESHOLD_METHODS",
    "THRESHOLD_FUNCTION_LOADERS",
    "_as_kernel_size",
    "_compile_legacy_normalization_steps",
    "_kernel_enabled",
    "_legacy_mask_pipeline",
    "_legacy_score_pipeline",
    "_pipeline_labels",
    "_step_label",
    "_validate_normalization_scope",
    "_validate_threshold_method",
    "apply_mask_postprocess",
    "apply_postprocess_pipeline",
    "apply_score_postprocess",
    "binary_dilation_tensor",
    "build_postprocess_pipeline",
    "build_postprocess_policy",
    "connected_components_tensor",
    "gray_dilation_tensor",
    "median_filter_tensor",
    "normalize_minmax",
    "otsu_threshold",
    "register_mask_postprocessor",
    "register_score_postprocessor",
    "register_threshold_method",
    "remove_small_components_tensor",
    "sanitize_scores",
    "supported_threshold_methods",
    "threshold_anomaly_map",
    "yen_threshold",
]
