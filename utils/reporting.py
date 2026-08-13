"""Compatibility facade for the engine-agnostic :mod:`andi_rewrite.reporting` package."""

from __future__ import annotations

# Keep direct imports from the former module path working, including its
# de-facto private helpers and standard-library names.
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ == "utils":
    import reporting as _reporting
else:
    from .. import reporting as _reporting


SUMMARY_COLUMNS = _reporting.SUMMARY_COLUMNS
safe_get = _reporting.safe_get
snapshot_config = _reporting.snapshot_config
summarize_eval_metrics = _reporting.summarize_eval_metrics
save_training_report = _reporting.save_training_report
save_inference_report = _reporting.save_inference_report
_build_training_report = _reporting._build_training_report
_build_inference_report = _reporting._build_inference_report
_summarize_one_metrics_csv = _reporting._summarize_one_metrics_csv
_parse_metric_row = _reporting._parse_metric_row
_metric_alias = _reporting._metric_alias
_training_markdown = _reporting._training_markdown
_inference_markdown = _reporting._inference_markdown
_metrics_markdown_table = _reporting._metrics_markdown_table
_dict_table = _reporting._dict_table
_write_metrics_summary_csv = _reporting._write_metrics_summary_csv
_diffusion_noise_settings = _reporting._diffusion_noise_settings
_noise_sampler_config = _reporting._noise_sampler_config
_noise_type = _reporting._noise_type
_training_report_dir = _reporting._training_report_dir
_inference_report_dir = _reporting._inference_report_dir
_run_name = _reporting._run_name
_dataset_len = _reporting._dataset_len
_parameter_count = _reporting._parameter_count
_infer_healthy_only = _reporting._infer_healthy_only
_infer_z_balanced = _reporting._infer_z_balanced
_yen_enabled = _reporting._yen_enabled
_find_pipeline_step = _reporting._find_pipeline_step
_matching_field = _reporting._matching_field
_to_float = _reporting._to_float
_format_time = _reporting._format_time
_duration_seconds = _reporting._duration_seconds
_format_value = _reporting._format_value
_json_safe = _reporting._json_safe
_describe = _reporting._describe
_get_attr = _reporting._get_attr
_first_present = _reporting._first_present
_first_dict = _reporting._first_dict
_git_commit_hash = _reporting._git_commit_hash
_write_json = _reporting._write_json
_write_text = _reporting._write_text
del _reporting
