"""Wait for cross training, run matched BraTS21-251 evals, and compare them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

try:
    from compare_brats21_251_lemon_models import build_comparison, write_comparison
except ImportError:  # Package-style import used by tests and read-only validation.
    from scripts.compare_brats21_251_lemon_models import build_comparison, write_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "comparisons" / "brats21_251_lemon_models_matched_noise"
NATIVE_RUN = REPO_ROOT / "outputs" / "runs" / "lemon_t1_t2_flair_empirical_spectrum233"
CROSS_RUN = REPO_ROOT / "outputs" / "runs" / "lemon_t1_t2_flair_brats21_empirical_spectrum233"
NATIVE_CONFIG = REPO_ROOT / "configs" / "eval_brats21_251_native_lemon_model_matched_noise.yaml"
CROSS_CONFIG = REPO_ROOT / "configs" / "eval_brats21_251_lemon_brats21_noise_model.yaml"
BAD_LOG_PATTERN = re.compile(
    r"\boom\b|out of memory|Traceback \(most recent call last\)|"
    r"(?:loss|metric|tensor|output|gradient|value)\s*(?:is|=|:)\s*(?:nan|[+-]?inf)\b|"
    r"\b(?:nan|[+-]?inf)\b.{0,40}\b(?:detected|encountered)\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class StatusWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "status": "STARTING",
            "orchestrator_pid": os.getpid(),
            "started_at": _now(),
            "updated_at": _now(),
            "active_child_pid": None,
        }

    def update(self, status: str, **values: Any) -> None:
        self.payload.update(values)
        self.payload["status"] = status
        self.payload["updated_at"] = _now()
        _atomic_json(self.path, self.payload)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _bad_log_findings(run_dir: Path) -> list[str]:
    findings: list[str] = []
    for name in ("stdout.log", "stderr.log"):
        path = run_dir / name
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if BAD_LOG_PATTERN.search(text):
            findings.append(str(path))
    return findings


def validate_completed_training(run_dir: Path, *, expected_epochs: int = 233) -> dict[str, Any]:
    rows = _read_metrics(run_dir / "training_metrics.csv")
    if len(rows) != expected_epochs:
        raise RuntimeError(f"Expected {expected_epochs} completed epochs in {run_dir}, got {len(rows)}")
    completed = [int(row["completed_epoch"]) for row in rows]
    if completed != list(range(1, expected_epochs + 1)):
        raise RuntimeError(f"Training epochs are not contiguous in {run_dir}")
    for row in rows:
        if row.get("finite_status") != "PASS":
            raise FloatingPointError(f"Non-PASS finite status in {run_dir}: {row}")
        if not math.isfinite(float(row["train_loss"])) or not math.isfinite(float(row["validation_loss"])):
            raise FloatingPointError(f"NaN/Inf training metric in {run_dir}: {row}")
    report_path = run_dir / "training_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Training report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if int(report["training_result"]["last_epoch"]) != expected_epochs - 1:
        raise RuntimeError(f"Training report has the wrong last_epoch: {report_path}")
    checkpoint = run_dir / f"epoch_{expected_epochs - 1:04d}.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Final checkpoint is missing or empty: {checkpoint}")
    findings = _bad_log_findings(run_dir)
    if findings:
        raise RuntimeError(f"OOM/NaN evidence in training logs: {findings}")
    return {
        "run_dir": str(run_dir),
        "completed_epochs": len(rows),
        "final_train_loss": float(rows[-1]["train_loss"]),
        "final_validation_loss": float(rows[-1]["validation_loss"]),
        "final_checkpoint": str(checkpoint),
        "log_scan": "PASS",
    }


def _matching_training_process(run_dir: Path) -> tuple[int, bool]:
    metadata_path = run_dir / "launch_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Launch metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    pid = int(metadata["pid"])
    if not psutil.pid_exists(pid):
        return pid, False
    process = psutil.Process(pid)
    expected = datetime.fromisoformat(str(metadata["start_time"])).timestamp()
    if abs(process.create_time() - expected) > 5:
        raise RuntimeError(f"PID {pid} was reused by a different process")
    executable = Path(process.exe()).resolve()
    expected_executable = Path(str(metadata["python_executable"])).resolve()
    if executable != expected_executable:
        raise RuntimeError(f"PID {pid} executable mismatch: {executable} != {expected_executable}")
    return pid, process.is_running()


def wait_for_cross_training(status: StatusWriter, poll_seconds: int) -> dict[str, Any]:
    while True:
        pid, alive = _matching_training_process(CROSS_RUN)
        rows = _read_metrics(CROSS_RUN / "training_metrics.csv")
        completed = int(rows[-1]["completed_epoch"]) if rows else 0
        status.update(
            "WAITING_FOR_CROSS_TRAINING",
            cross_training_pid=pid,
            cross_training_alive=alive,
            cross_completed_epoch=completed,
            cross_expected_epochs=233,
        )
        if not alive:
            return validate_completed_training(CROSS_RUN)
        time.sleep(poll_seconds)


def validate_eval_design() -> dict[str, Any]:
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in (NATIVE_CONFIG, CROSS_CONFIG)]
    left = json.loads(json.dumps(configs[0]))
    right = json.loads(json.dumps(configs[1]))
    left["experiment"]["name"] = right["experiment"]["name"]
    left["model"]["checkpoint"] = right["model"]["checkpoint"]
    left["metrics"]["output_csv"] = right["metrics"]["output_csv"]
    left["metrics"]["output_mf_csv"] = right["metrics"]["output_mf_csv"]
    left_sampler = left["noise"]["schedule"]["sampler"]
    right_sampler = right["noise"]["schedule"]["sampler"]
    left_sampler["stats_path"] = right_sampler["stats_path"]
    left_sampler["channel_indices"] = right_sampler["channel_indices"]
    if left != right:
        raise ValueError(
            "Eval configs differ beyond experiment/checkpoint/output and matched-noise fields"
        )
    config = configs[0]
    native_sampler = configs[0]["noise"]["schedule"]["sampler"]
    cross_sampler = configs[1]["noise"]["schedule"]["sampler"]
    expected_native_stats = str(
        Path("C:/ML/data/MPI/LEMON_ANDi_MIP/spectrum/lemon_t1_t2_flair_empirical_spectrum.npz")
    )
    expected_cross_stats = str(Path("C:/ML/data/spectrum/brats21_healthy_empirical_spectrum.npz"))
    if str(Path(native_sampler["stats_path"])) != expected_native_stats:
        raise ValueError("Native LEMON eval must use the LEMON empirical spectrum")
    if "channel_indices" in native_sampler:
        raise ValueError("Native three-channel LEMON spectrum must not be channel-subselected")
    if str(Path(cross_sampler["stats_path"])) != expected_cross_stats:
        raise ValueError("Cross-model eval must use the BraTS21 empirical spectrum")
    if cross_sampler.get("channel_indices") != [0, 1, 3]:
        raise ValueError("Cross-model BraTS21 spectrum must use channel indices [0,1,3]")
    csv_path = Path(config["data"]["path_to_csv"])
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        subjects = [row["BraTS21ID"] for row in csv.DictReader(handle)]
    if len(subjects) != 251 or len(set(subjects)) != 251:
        raise ValueError(f"Expected 251 unique BraTS21 subjects, got {len(subjects)}")
    return {
        "subjects": len(subjects),
        "modalities": config["data"]["modalities"],
        "inference_noise_by_model": {
            "native_lemon": {
                "stats_path": native_sampler["stats_path"],
                "channel_indices": None,
            },
            "lemon_brats21_noise": {
                "stats_path": cross_sampler["stats_path"],
                "channel_indices": cross_sampler["channel_indices"],
            },
        },
        "seed": config["runtime"]["seed"],
        "configs_matched_except_model_and_matched_noise_fields": True,
    }


def _validate_eval_outputs(output_dir: Path) -> dict[str, Any]:
    required = [
        output_dir / "ANDi.csv",
        output_dir / "ANDi_mf.csv",
        output_dir / "inference_metrics_summary.csv",
        output_dir / "inference_report.json",
        output_dir / "inference_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Evaluation outputs are missing or empty: {missing}")
    with (output_dir / "inference_metrics_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if [row.get("version") for row in rows] != ["raw", "median_filter"]:
        raise ValueError(f"Unexpected inference summary versions in {output_dir}")
    required_metrics = ("AUPRC", "bestdice", "yendice", "bestsen", "bestpre", "yensen", "yenpre")
    for row in rows:
        missing_metrics = [metric for metric in required_metrics if not str(row.get(metric, "")).strip()]
        if missing_metrics:
            raise ValueError(f"Required metrics are missing in {output_dir}: {missing_metrics}")
        values = [float(row[metric]) for metric in required_metrics]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"Non-finite inference summary in {output_dir}")
    return {"output_dir": str(output_dir), "summary_rows": len(rows), "status": "PASS"}


def validate_existing_eval(*, label: str, output_dir: Path) -> dict[str, Any]:
    result = _validate_eval_outputs(output_dir)
    log_paths = [OUTPUT_ROOT / f"{label}_eval_stdout.log", OUTPUT_ROOT / f"{label}_eval_stderr.log"]
    missing_logs = [str(path) for path in log_paths if not path.is_file()]
    if missing_logs:
        raise FileNotFoundError(f"Existing {label} evaluation logs are missing: {missing_logs}")
    findings = [
        str(path)
        for path in log_paths
        if BAD_LOG_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    if findings:
        raise RuntimeError(f"OOM/NaN/traceback evidence in existing {label} logs: {findings}")
    return {**result, "reused_completed_output": True, "log_scan": "PASS"}


def run_eval(status: StatusWriter, *, label: str, config: Path, output_dir: Path) -> dict[str, Any]:
    expected_outputs = [
        output_dir / "ANDi.csv",
        output_dir / "ANDi_mf.csv",
        output_dir / "inference_metrics_summary.csv",
        output_dir / "inference_report.json",
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing {label} evaluation outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_ROOT / f"{label}_eval_stdout.log"
    stderr_path = OUTPUT_ROOT / f"{label}_eval_stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"Evaluation logs already exist for {label}")
    command = [sys.executable, str(REPO_ROOT / "scripts" / "eval.py"), "--config", str(config), "--run-eval"]
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)
        status.update(
            f"EVALUATING_{label.upper()}",
            active_child_pid=process.pid,
            active_command=command,
            active_stdout=str(stdout_path),
            active_stderr=str(stderr_path),
        )
        return_code = process.wait()
    status.update(f"VALIDATING_{label.upper()}", active_child_pid=None, active_return_code=return_code)
    if return_code != 0:
        raise RuntimeError(f"{label} evaluation failed with exit code {return_code}: {stderr_path}")
    findings = []
    for path in (stdout_path, stderr_path):
        if BAD_LOG_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")):
            findings.append(str(path))
    if findings:
        raise RuntimeError(f"OOM/NaN evidence in {label} evaluation logs: {findings}")
    return _validate_eval_outputs(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--resume-completed-evals",
        action="store_true",
        help="Validate and reuse a completed first eval after a fail-closed orchestration stop.",
    )
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(OUTPUT_ROOT / "orchestration_status.json")
    try:
        design = validate_eval_design()
        native_training = validate_completed_training(NATIVE_RUN)
        status.update(
            "WAITING_FOR_CROSS_TRAINING",
            eval_design=design,
            native_training=native_training,
            resume_completed_evals=args.resume_completed_evals,
        )
        cross_training = wait_for_cross_training(status, args.poll_seconds)
        native_output_dir = OUTPUT_ROOT / "native_lemon"
        if args.resume_completed_evals:
            status.update("VALIDATING_REUSED_NATIVE_LEMON", active_child_pid=None)
            native_eval = validate_existing_eval(label="native_lemon", output_dir=native_output_dir)
        else:
            native_eval = run_eval(
                status,
                label="native_lemon",
                config=NATIVE_CONFIG,
                output_dir=native_output_dir,
            )
        cross_eval = run_eval(
            status,
            label="lemon_brats21_noise",
            config=CROSS_CONFIG,
            output_dir=OUTPUT_ROOT / "lemon_brats21_noise",
        )
        comparison = build_comparison(
            OUTPUT_ROOT / "native_lemon" / "inference_metrics_summary.csv",
            OUTPUT_ROOT / "lemon_brats21_noise" / "inference_metrics_summary.csv",
        )
        write_comparison(comparison, OUTPUT_ROOT)
        status.update(
            "COMPLETE",
            active_child_pid=None,
            active_command=None,
            cross_training=cross_training,
            native_eval=native_eval,
            cross_eval=cross_eval,
            comparison_report=str(OUTPUT_ROOT / "comparison_report.md"),
            completed_at=_now(),
        )
    except Exception as exc:
        status.update("BLOCKED", active_child_pid=None, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
