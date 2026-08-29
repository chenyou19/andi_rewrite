"""Compare matched BraTS21-251 inference summaries for two LEMON models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


PERFORMANCE_METRICS = {
    "AUPRC",
    "bestdice",
    "yendice",
    "bestsen",
    "bestpre",
    "yensen",
    "yenpre",
    "otsudice",
    "otsusen",
    "otsupre",
    "adaptive_dice",
    "adaptive_sensitivity",
    "adaptive_precision",
}
EXPECTED_VERSIONS = ("raw", "median_filter")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_summary(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Inference summary is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or reader.fieldnames[0] != "version":
            raise ValueError(f"Summary must start with a version column: {path}")
        candidate_metrics = [name for name in reader.fieldnames if name != "version"]
        raw_rows: dict[str, dict[str, str]] = {}
        for row in reader:
            version = str(row["version"]).strip()
            if not version or version in raw_rows:
                raise ValueError(f"Invalid or duplicate version {version!r}: {path}")
            raw_rows[version] = row
    if tuple(raw_rows) != EXPECTED_VERSIONS:
        raise ValueError(f"Expected versions {EXPECTED_VERSIONS}, got {tuple(raw_rows)} in {path}")
    metrics: list[str] = []
    rows: dict[str, dict[str, float]] = {version: {} for version in EXPECTED_VERSIONS}
    for metric in candidate_metrics:
        raw_values = [str(raw_rows[version].get(metric, "")).strip() for version in EXPECTED_VERSIONS]
        if not all(raw_values):
            continue
        try:
            values = [float(value) for value in raw_values]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"Non-finite metric {metric!r} in {path}")
        metrics.append(metric)
        for version, value in zip(EXPECTED_VERSIONS, values, strict=True):
            rows[version][metric] = value
    if not metrics:
        raise ValueError(f"Summary has no complete finite numeric metrics: {path}")
    return metrics, rows


def build_comparison(native_path: Path, cross_path: Path) -> dict[str, Any]:
    native_metrics, native = _read_summary(native_path)
    cross_metrics, cross = _read_summary(cross_path)
    if native_metrics != cross_metrics:
        raise ValueError(
            "Inference summaries have different metric columns: "
            f"native={native_metrics}, cross={cross_metrics}"
        )
    rows: list[dict[str, Any]] = []
    for version in EXPECTED_VERSIONS:
        for metric in native_metrics:
            native_value = native[version][metric]
            cross_value = cross[version][metric]
            difference = cross_value - native_value
            relative_percent = (
                difference / abs(native_value) * 100.0 if native_value != 0 else None
            )
            if metric in PERFORMANCE_METRICS:
                if math.isclose(native_value, cross_value, rel_tol=0.0, abs_tol=1e-12):
                    winner = "tie"
                else:
                    winner = "lemon_brats21_noise" if difference > 0 else "native_lemon"
            else:
                winner = "not_applicable"
            rows.append(
                {
                    "version": version,
                    "metric": metric,
                    "native_lemon": native_value,
                    "lemon_brats21_noise": cross_value,
                    "cross_minus_native": difference,
                    "relative_percent": relative_percent,
                    "winner": winner,
                }
            )
    return {
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "comparison_design": {
            "dataset": "BraTS21 scans_test.csv (251 subjects)",
            "modalities": ["FLAIR", "T1", "T2"],
            "inference_noise_by_model": {
                "native_lemon": "LEMON empirical spectrum [FLAIR,T1,T2]",
                "lemon_brats21_noise": "BraTS21 empirical spectrum channels [0,1,3]",
            },
            "seed": 73,
            "comparison_type": "matched training/inference spectrum per model",
        },
        "inputs": {
            "native_lemon_summary": str(native_path.resolve()),
            "native_lemon_summary_sha256": _sha256(native_path),
            "lemon_brats21_noise_summary": str(cross_path.resolve()),
            "lemon_brats21_noise_summary_sha256": _sha256(cross_path),
        },
        "rows": rows,
    }


def write_comparison(report: dict[str, Any], output_dir: Path, *, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"
    markdown_path = output_dir / "comparison_report.md"
    targets = (csv_path, json_path, markdown_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Comparison outputs already exist: {existing}")

    fieldnames = [
        "version",
        "metric",
        "native_lemon",
        "lemon_brats21_noise",
        "cross_minus_native",
        "relative_percent",
        "winner",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# BraTS21 251-subject LEMON Model Comparison",
        "",
        "Both models use the same subjects, `[FLAIR,T1,T2]` inputs, seed 73, EMA policy, "
        "and postprocessing. Native LEMON uses LEMON inference noise; the cross model uses "
        "BraTS21 `[0,1,3]` inference noise, matching each model's training spectrum.",
        "",
        "| version | metric | native LEMON | LEMON + BraTS21 noise | difference | relative % | winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        relative = row["relative_percent"]
        relative_text = "n/a" if relative is None else f"{relative:.4f}"
        lines.append(
            f"| {row['version']} | {row['metric']} | {row['native_lemon']:.8g} | "
            f"{row['lemon_brats21_noise']:.8g} | {row['cross_minus_native']:+.8g} | "
            f"{relative_text} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "Positive differences mean the LEMON-data/BraTS21-spectrum-trained model is higher. "
            "Threshold metrics (`bestthr`, `yenthr`) are descriptive and have no winner.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/comparisons/brats21_251_lemon_models_matched_noise")
    parser.add_argument(
        "--native-summary",
        type=Path,
        default=root / "native_lemon" / "inference_metrics_summary.csv",
    )
    parser.add_argument(
        "--cross-summary",
        type=Path,
        default=root / "lemon_brats21_noise" / "inference_metrics_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_comparison(args.native_summary, args.cross_summary)
    write_comparison(report, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
