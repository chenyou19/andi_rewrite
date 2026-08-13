from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .inference import _build_inference_report, _inference_markdown
from .metadata import _first_present, _get_attr, _run_name
from .metrics_csv import _write_metrics_summary_csv, summarize_eval_metrics
from .serialization import _json_safe, safe_get
from .training import _build_training_report, _training_markdown


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
