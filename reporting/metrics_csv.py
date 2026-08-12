from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .metadata import _first_present


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
    "otsudice",
    "otsuthr",
    "otsusen",
    "otsupre",
    "adaptive_method",
    "adaptive_dice",
    "adaptive_threshold",
    "adaptive_sensitivity",
    "adaptive_precision",
]


def summarize_eval_metrics(
    raw_csv: str | Path | None,
    median_filter_csv: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Parse existing eval CSV files into raw and median-filter metric summaries."""

    return {
        "raw": _summarize_one_metrics_csv(raw_csv),
        "median_filter": _summarize_one_metrics_csv(median_filter_csv),
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

    adaptive_method = None
    adaptive_prefix = None
    if any(summary.get(key) is not None for key in ("otsudice", "otsuthr", "otsusen", "otsupre")):
        adaptive_method = "otsu"
        adaptive_prefix = "otsu"
    elif any(summary.get(key) is not None for key in ("yendice", "yenthr", "yensen", "yenpre")):
        adaptive_method = "yen"
        adaptive_prefix = "yen"
    if adaptive_prefix is not None:
        summary["adaptive_method"] = adaptive_method
        summary["adaptive_dice"] = summary.get(f"{adaptive_prefix}dice")
        summary["adaptive_threshold"] = summary.get(f"{adaptive_prefix}thr")
        summary["adaptive_sensitivity"] = summary.get(f"{adaptive_prefix}sen")
        summary["adaptive_precision"] = summary.get(f"{adaptive_prefix}pre")

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
        "otsudice": "otsudice",
        "diceotsu": "otsudice",
        "otsu": "otsudice",
        "otsuthr": "otsuthr",
        "otsuthreshold": "otsuthr",
        "otsusen": "otsusen",
        "otsusensitivity": "otsusen",
        "otsupre": "otsupre",
        "otsuprecision": "otsupre",
    }
    return aliases.get(key)


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


def _write_metrics_summary_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version", *SUMMARY_COLUMNS])
        for version in ["raw", "median_filter"]:
            metrics = summary.get(version, {})
            writer.writerow([version, *[metrics.get(column) for column in SUMMARY_COLUMNS]])
