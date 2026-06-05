from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "AUPRC",
    "bestdice",
    "bestthr",
    "yendice",
    "yenthr",
    "bestsen",
    "bestpre",
    "yensen",
    "yenpre",
]


def safe_get(mapping: Any, path: str | list[str], default: Any = None) -> Any:
    """Read a nested config value without raising when a key is missing."""

    parts = path.split(".") if isinstance(path, str) else path
    current = mapping
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def snapshot_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of the full run config."""

    return _json_safe(config)


def summarize_eval_metrics(
    raw_csv: str | Path | None,
    median_filter_csv: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Parse existing eval CSV files into raw and median-filter metric summaries."""

    return {
        "raw": _summarize_one_metrics_csv(raw_csv),
        "median_filter": _summarize_one_metrics_csv(median_filter_csv),
    }


def save_training_report(
    *,
    config: dict[str, Any],
    trainer: Any,
    dataloader: Any = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    config_path: str | Path | None = None,
    cli_args: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Write training_report.md and training_report.json for a completed training run."""

    try:
        report_dir = _training_report_dir(config, trainer, output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report = _build_training_report(
            config=config,
            trainer=trainer,
            dataloader=dataloader,
            start_time=start_time,
            end_time=end_time,
            config_path=config_path,
            cli_args=cli_args,
            report_dir=report_dir,
        )
        _write_json(report_dir / "training_report.json", report)
        _write_text(report_dir / "training_report.md", _training_markdown(report))
        return report_dir
    except Exception as exc:  # pragma: no cover - report failures must not break training.
        print(f"Warning: failed to write training report: {exc}")
        return None


def save_inference_report(
    *,
    config: dict[str, Any],
    evaluator: Any = None,
    result: dict[str, Any] | None = None,
    dataloader: Any = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    config_path: str | Path | None = None,
    cli_args: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Write inference reports and a compact metrics summary from existing CSV outputs."""

    try:
        raw_csv = _first_present(
            _get_attr(evaluator, "output_csv"),
            safe_get(result or {}, "output"),
            safe_get(config, "metrics.output_csv"),
            safe_get(config, "metrics.output"),
        )
        mf_csv = _first_present(
            _get_attr(evaluator, "output_mf_csv"),
            safe_get(result or {}, "output_mf"),
            safe_get(config, "metrics.output_mf_csv"),
            safe_get(config, "metrics.output_mf"),
        )
        report_dir = _inference_report_dir(raw_csv, mf_csv, output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary = summarize_eval_metrics(raw_csv, mf_csv)
        _write_metrics_summary_csv(report_dir / "inference_metrics_summary.csv", metrics_summary)
        report = _build_inference_report(
            config=config,
            evaluator=evaluator,
            result=result or {},
            dataloader=dataloader,
            metrics_summary=metrics_summary,
            start_time=start_time,
            end_time=end_time,
            config_path=config_path,
            cli_args=cli_args,
            report_dir=report_dir,
            raw_csv=raw_csv,
            mf_csv=mf_csv,
        )
        _write_json(report_dir / "inference_report.json", report)
        _write_text(report_dir / "inference_report.md", _inference_markdown(report))
        return report_dir
    except Exception as exc:  # pragma: no cover - report failures must not break inference.
        print(f"Warning: failed to write inference report: {exc}")
        return None


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
    yen_mask = safe_get(metrics, "postprocess.yen_mask", {})
    dilation = _find_pipeline_step(yen_mask, "binary_dilation")

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
            "threshold_method": anomaly.get("threshold"),
            "threshold_search_start": _first_present(metrics.get("thr_start"), metrics.get("threshold_start")),
            "threshold_search_end": _first_present(metrics.get("thr_end"), metrics.get("threshold_end")),
            "threshold_search_step": _first_present(metrics.get("thr_step"), metrics.get("threshold_step")),
            "yen_threshold_enabled": _yen_enabled(config),
            "raw_metrics_csv": str(raw_csv) if raw_csv else None,
            "median_filter_metrics_csv": str(mf_csv) if mf_csv else None,
        },
        "post_processing_settings": {
            "median_filter_enabled": median_filter.get("enabled"),
            "median_filter_kernel_size": median_filter.get("kernel_size", metrics.get("kernel_size")),
            "median_filter_mode": median_filter.get("mode"),
            "dilation_enabled": _first_present(dilation.get("enabled") if dilation else None, bool(dilation) if dilation else None),
            "dilation_rank": _first_present(dilation.get("rank") if dilation else None, metrics.get("rank")),
            "connectivity": _first_present(dilation.get("connectivity") if dilation else None, metrics.get("connectivity")),
            "threshold_mask_postprocess": safe_get(metrics, "postprocess.threshold_mask"),
            "yen_mask_postprocess": yen_mask,
            "score_postprocess": safe_get(metrics, "postprocess.score"),
            "score_mf_postprocess": safe_get(metrics, "postprocess.score_mf"),
        },
        "metrics_summary": metrics_summary,
        "raw_metrics": metrics_summary.get("raw", {}),
        "median_filter_metrics": metrics_summary.get("median_filter", {}),
        "evaluation_result": _json_safe(result),
        "full_config_snapshot": snapshot_config(config),
    }


def _summarize_one_metrics_csv(path: str | Path | None) -> dict[str, Any]:
    summary = {column: None for column in SUMMARY_COLUMNS}
    summary["source_csv"] = str(path) if path else None
    if not path:
        summary["warning"] = "metrics csv not available"
        return summary
    csv_path = Path(path)
    if not csv_path.exists():
        summary["warning"] = f"metrics csv not found: {csv_path}"
        return summary

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        summary["warning"] = f"failed to parse metrics csv: {exc}"
        return summary

    threshold_rows: list[dict[str, float | None]] = []
    for row in rows:
        _parse_metric_row(row, fieldnames, summary, threshold_rows)

    if threshold_rows:
        best = max(threshold_rows, key=lambda item: float(item.get("dice") or float("-inf")))
        summary["bestdice"] = _first_present(summary.get("bestdice"), best.get("dice"))
        summary["bestthr"] = _first_present(summary.get("bestthr"), best.get("threshold"))
        if best.get("sensitivity") is not None:
            summary["bestsen"] = best.get("sensitivity")
        if best.get("precision") is not None:
            summary["bestpre"] = best.get("precision")

    return summary


def _parse_metric_row(
    row: dict[str, str],
    fieldnames: list[str],
    summary: dict[str, Any],
    threshold_rows: list[dict[str, float | None]],
) -> None:
    key_field = _first_present(
        _matching_field(fieldnames, ["metric", "name", "key", "thr", "threshold"]),
        fieldnames[0] if fieldnames else None,
    )
    value_field = _matching_field(fieldnames, ["value", "score", "metric_value", "dice"])
    key = str(row.get(key_field, "")).strip() if key_field else ""
    value = _to_float(row.get(value_field, "")) if value_field else None

    threshold = _to_float(key)
    if threshold is None:
        threshold = _to_float(row.get(_matching_field(fieldnames, ["threshold", "thr"]) or "", ""))
    if threshold is not None:
        dice_value = _first_present(
            _to_float(row.get(_matching_field(fieldnames, ["dice", "bestdice"]) or "", "")),
            value,
        )
        if dice_value is None:
            return
        threshold_rows.append(
            {
                "threshold": threshold,
                "dice": dice_value,
                "sensitivity": _to_float(row.get(_matching_field(fieldnames, ["sensitivity", "sen", "bestsen"]) or "", "")),
                "precision": _to_float(row.get(_matching_field(fieldnames, ["precision", "pre", "bestpre"]) or "", "")),
            }
        )
        return

    target = _metric_alias(key)
    if target and value is not None:
        summary[target] = _first_present(summary.get(target), value)
        return

    for field, field_value in row.items():
        target = _metric_alias(field)
        parsed = _to_float(field_value)
        if target and parsed is not None:
            summary[target] = _first_present(summary.get(target), parsed)


def _metric_alias(name: Any) -> str | None:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "auprc": "AUPRC",
        "averageprecision": "AUPRC",
        "ap": "AUPRC",
        "bestdice": "bestdice",
        "maxdice": "bestdice",
        "dicebest": "bestdice",
        "bestthr": "bestthr",
        "bestthreshold": "bestthr",
        "yendice": "yendice",
        "yundice": "yendice",
        "diceyen": "yendice",
        "diceyun": "yendice",
        "yen": "yendice",
        "yun": "yendice",
        "yenthr": "yenthr",
        "yunthr": "yenthr",
        "yenthreshold": "yenthr",
        "yunthreshold": "yenthr",
        "bestsen": "bestsen",
        "bestsensitivity": "bestsen",
        "sensitivity": "bestsen",
        "sen": "bestsen",
        "bestpre": "bestpre",
        "bestprecision": "bestpre",
        "precision": "bestpre",
        "pre": "bestpre",
        "yensen": "yensen",
        "yunsen": "yensen",
        "yensensitivity": "yensen",
        "yunsensitivity": "yensen",
        "yenpre": "yenpre",
        "yunpre": "yenpre",
        "yenprecision": "yenpre",
        "yunprecision": "yenpre",
    }
    return aliases.get(key)


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
        "Yen Dice",
        "Yen Thr",
        "bestsen",
        "bestpre",
        "Yen Sensitivity",
        "Yen Precision",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for version in ["raw", "median_filter"]:
        metrics = summary.get(version, {})
        values = [
            version,
            _format_value(metrics.get("AUPRC")),
            _format_value(metrics.get("bestdice")),
            _format_value(metrics.get("bestthr")),
            _format_value(metrics.get("yendice")),
            _format_value(metrics.get("yenthr")),
            _format_value(metrics.get("bestsen")),
            _format_value(metrics.get("bestpre")),
            _format_value(metrics.get("yensen")),
            _format_value(metrics.get("yenpre")),
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _dict_table(values: dict[str, Any]) -> str:
    rows = ["| key | value |", "|---|---|"]
    for key, value in values.items():
        rows.append(f"| {key} | {_format_value(value)} |")
    return "\n".join(rows)


def _write_metrics_summary_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version", *SUMMARY_COLUMNS])
        for version in ["raw", "median_filter"]:
            metrics = summary.get(version, {})
            writer.writerow([version, *[metrics.get(column) for column in SUMMARY_COLUMNS]])


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


def _training_report_dir(config: dict[str, Any], trainer: Any, output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    training = config.get("training", {})
    run_name = _run_name(config, training)
    checkpoint_dir = Path(_get_attr(trainer, "checkpoint_dir") or safe_get(training, "checkpoint.dir", "outputs/checkpoints"))
    return checkpoint_dir / str(run_name)


def _inference_report_dir(raw_csv: Any, mf_csv: Any, output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    paths = [Path(item) for item in [raw_csv, mf_csv] if item]
    if paths:
        parents = {path.parent for path in paths}
        if len(parents) == 1:
            return paths[0].parent
        return paths[0].parent
    return Path("outputs/metrics")


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


def _matching_field(fieldnames: list[str], candidates: list[str]) -> str | None:
    canonical = {"".join(ch for ch in field.lower() if ch.isalnum()): field for field in fieldnames}
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in canonical:
            return canonical[key]
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_time(value: datetime | None) -> str | None:
    return value.astimezone().isoformat(timespec="seconds") if isinstance(value, datetime) else None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return round((end - start).total_seconds(), 3)


def _format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return "`" + json.dumps(_json_safe(value), ensure_ascii=False) + "`"
    return str(value).replace("|", "\\|")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _format_time(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
