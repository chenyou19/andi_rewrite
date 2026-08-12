from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .metadata import (
    _dataset_len,
    _describe,
    _find_pipeline_step,
    _first_dict,
    _first_present,
    _git_commit_hash,
    _get_attr,
    _noise_type,
    _yen_enabled,
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


def _build_inference_report(
    *,
    config: dict[str, Any],
    evaluator: Any,
    result: dict[str, Any],
    dataloader: Any,
    metrics_summary: dict[str, dict[str, Any]],
    start_time: datetime | None,
    end_time: datetime | None,
    config_path: str | Path | None,
    cli_args: dict[str, Any] | None,
    report_dir: Path,
    raw_csv: str | Path | None,
    mf_csv: str | Path | None,
) -> dict[str, Any]:
    data = config.get("data", {})
    model = config.get("model", {})
    diffusion = config.get("diffusion", {})
    noise = config.get("noise", {})
    anomaly = config.get("anomaly", {})
    metrics = config.get("metrics", {})
    median_filter = _first_dict(metrics.get("median_filter"), anomaly.get("median_filter"), {})
    binary_mask = _first_dict(
        safe_get(metrics, "postprocess.binary_mask"),
        safe_get(metrics, "postprocess.yen_mask"),
        {},
    )
    dilation = _find_pipeline_step(binary_mask, "binary_dilation")
    described_policy = _describe(_get_attr(evaluator, "postprocess_policy"))
    policy_description = (
        described_policy
        if isinstance(described_policy, dict)
        else _first_dict(result.get("postprocessing"), {})
    )
    policy_median = _first_dict(policy_description.get("median_filter_settings"), {})
    policy_dilation = _first_dict(policy_description.get("dilation_settings"), {})
    export_normalization_scope = _first_present(
        _get_attr(evaluator, "prediction_normalization_scope"),
        safe_get(config, "prediction_output.normalization_scope"),
    )
    threshold_method = _first_present(
        policy_description.get("threshold_method"),
        result.get("threshold_method"),
        metrics.get("threshold_method"),
        anomaly.get("threshold_method"),
        anomaly.get("threshold"),
        "yen",
    )

    return {
        "basic_info": {
            "experiment_name": safe_get(config, "experiment.name"),
            "start_time": _format_time(start_time),
            "end_time": _format_time(end_time),
            "total_inference_time_seconds": _duration_seconds(start_time, end_time),
            "device": str(_get_attr(_get_attr(evaluator, "detector"), "device", safe_get(config, "runtime.device"))),
            "seed": safe_get(config, "runtime.seed"),
            "eval_config_path": str(config_path or config.get("_config_path") or ""),
            "cli_args": _json_safe(cli_args or {}),
            "output_directory": str(report_dir),
            "git_commit_hash": _git_commit_hash(),
        },
        "inference_settings": {
            "checkpoint_path": model.get("checkpoint"),
            "dataset_path": data.get("dataset_path"),
            "scan_csv_or_test_csv_path": data.get("path_to_csv"),
            "number_of_cases": _dataset_len(dataloader),
            "image_size": _first_present(data.get("image_size"), model.get("image_size")),
            "channels": _first_present(data.get("channels"), model.get("in_channels")),
            "batch_size": data.get("batch_size"),
            "noise_type": _noise_type(noise),
            "noise_steps": diffusion.get("steps"),
            "aggregation_method": safe_get(anomaly, "aggregation.type"),
            "anomaly_map_method": _first_present(safe_get(anomaly, "modality_pool.type"), anomaly.get("map_method")),
            "threshold_method": threshold_method,
            "threshold_search_start": _first_present(metrics.get("thr_start"), metrics.get("threshold_start")),
            "threshold_search_end": _first_present(metrics.get("thr_end"), metrics.get("threshold_end")),
            "threshold_search_step": _first_present(metrics.get("thr_step"), metrics.get("threshold_step")),
            "adaptive_threshold_enabled": True,
            "yen_threshold_enabled": threshold_method == "yen" and _yen_enabled(config),
            "raw_metrics_csv": str(raw_csv) if raw_csv else None,
            "median_filter_metrics_csv": str(mf_csv) if mf_csv else None,
        },
        "post_processing_settings": {
            "postprocess_mode": _first_present(
                policy_description.get("postprocess_mode"),
                metrics.get("postprocess_mode"),
                "rewrite",
            ),
            "normalization_scope": _first_present(
                policy_description.get("normalization_scope"),
                result.get("normalization_scope"),
            ),
            "raw_score_pipeline": policy_description.get("raw_score_pipeline"),
            "mf_score_pipeline": policy_description.get("mf_score_pipeline"),
            "threshold_method": threshold_method,
            "threshold_strategy": _first_present(
                policy_description.get("threshold_strategy"),
                policy_description.get("yen_threshold_strategy"),
            ),
            "threshold_comparator": policy_description.get("threshold_comparator"),
            "binary_mask_pipeline": _first_present(
                policy_description.get("binary_mask_pipeline"),
                policy_description.get("yen_mask_pipeline"),
            ),
            "yen_threshold_strategy": policy_description.get("yen_threshold_strategy"),
            "yen_mask_pipeline": policy_description.get("yen_mask_pipeline"),
            "dilation_settings": policy_dilation,
            "export_normalization_scope": export_normalization_scope,
            "median_filter_enabled": _first_present(
                policy_median.get("enabled"),
                median_filter.get("enabled"),
            ),
            "median_filter_kernel_size": _first_present(
                policy_median.get("kernel_size"),
                median_filter.get("kernel_size"),
                metrics.get("kernel_size"),
            ),
            "median_filter_mode": _first_present(
                policy_median.get("mode"),
                median_filter.get("mode"),
            ),
            "dilation_enabled": _first_present(
                policy_dilation.get("enabled"),
                dilation.get("enabled") if dilation else None,
                bool(dilation) if dilation else None,
            ),
            "dilation_rank": _first_present(
                policy_dilation.get("rank"),
                dilation.get("rank") if dilation else None,
                metrics.get("rank"),
            ),
            "connectivity": _first_present(
                policy_dilation.get("connectivity"),
                dilation.get("connectivity") if dilation else None,
                metrics.get("connectivity"),
            ),
            "threshold_mask_postprocess": safe_get(metrics, "postprocess.threshold_mask"),
            "binary_mask_postprocess": binary_mask,
            "yen_mask_postprocess": (
                binary_mask if threshold_method == "yen" else safe_get(metrics, "postprocess.yen_mask")
            ),
            "score_postprocess": safe_get(metrics, "postprocess.score"),
            "score_mf_postprocess": safe_get(metrics, "postprocess.score_mf"),
        },
        "metrics_summary": metrics_summary,
        "raw_metrics": metrics_summary.get("raw", {}),
        "median_filter_metrics": metrics_summary.get("median_filter", {}),
        "evaluation_result": _json_safe(result),
        "full_config_snapshot": snapshot_config(config),
    }


def _inference_markdown(report: dict[str, Any]) -> str:
    sections = [
        "# Inference Report",
        "## Basic Info",
        _dict_table(report["basic_info"]),
        "## Inference Settings",
        _dict_table(report["inference_settings"]),
        "## Post-processing Settings",
        _dict_table(report["post_processing_settings"]),
        "## Metrics Summary",
        _metrics_markdown_table(report["metrics_summary"]),
        "## Raw Metrics",
        _dict_table(report["raw_metrics"]),
        "## Median Filter Metrics",
        _dict_table(report["median_filter_metrics"]),
        "## Full Config Snapshot",
        "```json",
        json.dumps(report["full_config_snapshot"], indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n\n".join(sections) + "\n"


def _metrics_markdown_table(summary: dict[str, dict[str, Any]]) -> str:
    headers = [
        "version",
        "AUPRC",
        "bestdice",
        "bestthr",
        "Threshold method",
        "Adaptive Dice",
        "Adaptive Thr",
        "bestsen",
        "bestpre",
        "Adaptive Sensitivity",
        "Adaptive Precision",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for version in ["raw", "median_filter"]:
        metrics = summary.get(version, {})
        values = [
            version,
            _format_value(metrics.get("AUPRC")),
            _format_value(metrics.get("bestdice")),
            _format_value(metrics.get("bestthr")),
            _format_value(metrics.get("adaptive_method")),
            _format_value(metrics.get("adaptive_dice")),
            _format_value(metrics.get("adaptive_threshold")),
            _format_value(metrics.get("bestsen")),
            _format_value(metrics.get("bestpre")),
            _format_value(metrics.get("adaptive_sensitivity")),
            _format_value(metrics.get("adaptive_precision")),
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
