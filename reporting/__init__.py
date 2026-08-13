"""Engine-agnostic training and inference report helpers."""

from .inference import _build_inference_report, _inference_markdown, _metrics_markdown_table
from .io import (
    _inference_report_dir,
    _training_report_dir,
    _write_json,
    _write_text,
    save_inference_report,
    save_training_report,
)
from .metadata import (
    _dataset_len,
    _describe,
    _diffusion_noise_settings,
    _find_pipeline_step,
    _first_dict,
    _first_present,
    _get_attr,
    _git_commit_hash,
    _infer_healthy_only,
    _infer_z_balanced,
    _noise_sampler_config,
    _noise_type,
    _parameter_count,
    _run_name,
    _yen_enabled,
)
from .metrics_csv import (
    SUMMARY_COLUMNS,
    _matching_field,
    _metric_alias,
    _parse_metric_row,
    _summarize_one_metrics_csv,
    _to_float,
    _write_metrics_summary_csv,
    summarize_eval_metrics,
)
from .serialization import (
    _dict_table,
    _duration_seconds,
    _format_time,
    _format_value,
    _json_safe,
    safe_get,
    snapshot_config,
)
from .training import _build_training_report, _training_markdown

__all__ = [
    "SUMMARY_COLUMNS",
    "safe_get",
    "save_inference_report",
    "save_training_report",
    "snapshot_config",
    "summarize_eval_metrics",
]
