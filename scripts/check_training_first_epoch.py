"""Fail-closed first-epoch health gate for a hidden ANDi training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "launch_metadata.json"
    metrics_path = run_dir / "training_metrics.csv"
    if not metadata_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Launch metadata or first-epoch metrics are not available yet.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[0]["completed_epoch"]) != 1:
        raise RuntimeError("The first completed epoch is not present in training_metrics.csv.")
    first = rows[0]
    train_loss = float(first["train_loss"])
    validation_loss = float(first["validation_loss"])
    if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
        raise FloatingPointError("First-epoch train or validation loss is NaN/Inf.")
    if first["finite_status"] != "PASS":
        raise FloatingPointError(f"First-epoch finite status is {first['finite_status']}.")

    pid = int(metadata["pid"])
    try:
        import psutil

        process_alive = bool(psutil.pid_exists(pid) and psutil.Process(pid).is_running())
    except ImportError:  # pragma: no cover
        process_alive = False
    if not process_alive:
        raise RuntimeError(f"Training process {pid} is not alive after the first epoch.")

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    gpu_process_alive = any(line.split(",", 1)[0].strip() == str(pid) for line in gpu_rows)
    if query.returncode != 0 or not gpu_process_alive:
        raise RuntimeError(f"Training PID {pid} is not visible as a live NVIDIA compute process.")

    bad_log_pattern = re.compile(r"\b(?:nan|oom)\b|out of memory", re.IGNORECASE)
    log_findings = []
    for name in ("stdout.log", "stderr.log"):
        path = run_dir / name
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if bad_log_pattern.search(text):
            log_findings.append(name)
    if log_findings:
        raise RuntimeError(f"OOM/NaN evidence found in logs: {log_findings}")

    report = {
        "status": "PASS",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": pid,
        "process_alive": process_alive,
        "gpu_process_alive": gpu_process_alive,
        "first_epoch": {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "finite_status": first["finite_status"],
            "gpu_peak_bytes": int(first["gpu_peak_bytes"]),
        },
        "oom_nan_log_scan": "PASS",
    }
    (run_dir / "first_epoch_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
