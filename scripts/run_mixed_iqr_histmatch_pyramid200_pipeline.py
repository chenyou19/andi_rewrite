"""Run matched-IQR mix training for 200 epochs, then fixed BraTS21 test251 inference."""
from __future__ import annotations

import csv
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import yaml
import psutil

ROOT = Path(__file__).resolve().parents[1]
NAME = "mixed_sri24_robust_iqr_andi_histmatch_mean888_pyramid200_20261005"
RUN = ROOT / "outputs/runs" / NAME
TRAIN = RUN / "config_snapshot.yaml"
EVAL = RUN / "eval_config_snapshot.yaml"
STATE = RUN / "pipeline_status.json"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(attach_training=False):
    if attach_training:
        state = json.loads(STATE.read_text())
        if state.get("active_stage") != "training" or "inference" in state["stages"]:
            raise RuntimeError("Attachment requires a training-stage pipeline with no inference launched")
        state.update(status="RUNNING", controller_pid=os.getpid(), controller_restarted_at=now())
    else:
        state = {"status": "RUNNING", "controller_pid": os.getpid(), "started_at": now(), "stages": {}}
    frozen = {str(p): digest(p) for p in (TRAIN, EVAL)}
    if attach_training and state["config_sha256"][str(TRAIN)] != frozen[str(TRAIN)]:
        raise ValueError("Training configuration changed before attachment")
    state["config_sha256"] = frozen

    def persist():
        state["updated_at"] = now()
        temporary = STATE.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(STATE)

    def execute(stage, script, config, flag):
        if any(digest(p) != value for p, value in frozen.items()):
            raise ValueError("A frozen experiment configuration changed")
        stdout = RUN / ("stdout.log" if stage == "training" else "inference_stdout.log")
        stderr = RUN / ("stderr.log" if stage == "training" else "inference_stderr.log")
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(config), flag]
        attached = stage == "training" and attach_training
        with stdout.open("a" if attached else "w", encoding="utf-8") as out, stderr.open("a" if attached else "w", encoding="utf-8") as err:
            if attached:
                child = psutil.Process(state["stages"][stage]["pid"])
                if child.cmdline() != command:
                    raise RuntimeError("Existing training PID does not match the frozen command")
            else:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            alive = child.is_running if attached else lambda: child.poll() is None
            state["active_stage"] = stage
            prior_stage = state["stages"].get(stage, {}) if attached else {}
            state["stages"][stage] = {**prior_stage, "status": "RUNNING", "pid": child.pid, "command": command,
                                       "started_at": prior_stage.get("started_at", now()), "stdout": str(stdout), "stderr": str(stderr)}
            if attached:
                state["stages"][stage]["attached_at"] = now()
            if stage == "training":
                previous_metadata = json.loads((RUN / "launch_metadata.json").read_text()) if attached else {}
                metadata = {**previous_metadata, "status": "LAUNCHED", "run_name": NAME, "pid": child.pid,
                            "controller_pid": os.getpid(), "start_time": previous_metadata.get("start_time", now()), "hidden_window": True,
                            "python_executable": sys.executable, "config_snapshot": str(TRAIN),
                            "eval_config_snapshot": str(EVAL), "stdout": str(stdout), "stderr": str(stderr)}
                (RUN / "launch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            persist()
            try:
                while alive():
                    metrics = RUN / "training_metrics.csv"
                    if stage == "training" and metrics.exists() and not (RUN / "first_epoch_gate.json").exists():
                        with metrics.open(newline="", encoding="utf-8") as handle:
                            rows = list(csv.DictReader(handle))
                        if rows:
                            gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_training_first_epoch.py"), str(RUN)],
                                                  cwd=ROOT, capture_output=True, text=True)
                            (RUN / "first_epoch_gate_stdout.log").write_text(gate.stdout, encoding="utf-8")
                            (RUN / "first_epoch_gate_stderr.log").write_text(gate.stderr, encoding="utf-8")
                            if gate.returncode:
                                raise RuntimeError("First epoch health gate failed; inspect first_epoch_gate_stderr.log")
                            state["first_epoch_gate"] = json.loads((RUN / "first_epoch_gate.json").read_text())
                            persist()
                    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                        raise RuntimeError("Free disk space below 5 GiB")
                    time.sleep(10)
                if not attached and child.returncode:
                    raise RuntimeError(f"{stage} exited {child.returncode}; inspect {stderr}")
            except BaseException:
                if alive():
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
                        child.kill()
                        child.wait(timeout=10)
                raise
        state["stages"][stage].update(status="PASS", finished_at=now(), exit_code=None if attached else child.returncode)
        persist()

    try:
        persist()
        train = yaml.safe_load(TRAIN.read_text())
        evaluation = yaml.safe_load(EVAL.read_text())
        assert train["training"]["epochs"] == 200 and train["training"]["checkpoint"]["resume"] is None
        assert train["training"]["checkpoint"]["start_epoch"] == 19
        assert train["training"]["checkpoint"]["save_every_epochs"] == 20
        assert train["training"]["run_name"] == NAME
        assert not train["training"]["normalize_input"]
        assert evaluation["noise"] == {"schedule": {"type": "static", "sampler": {"type": "gaussian"}}}
        assert train["noise"]["schedule"]["sampler"] == {"type": "pyramid", "discount": 0.8, "levels": 10, "normalize": True}
        final = RUN / "epoch_0199.pt"
        assert Path(evaluation["model"]["checkpoint"]) == final and evaluation["model"]["use_ema"]
        roster = Path(evaluation["data"]["path_to_csv"])
        ids = [r["BraTS21ID"] for r in csv.DictReader(roster.open())]
        assert len(ids) == len(set(ids)) == 251
        state["roster_sha256"] = digest(roster)
        execute("training", "train.py", TRAIN, "--fit")
        with (RUN / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["epoch_index"]) for row in rows] != list(range(200)):
            raise ValueError("Training did not complete exactly 200 epochs")
        if any(row["finite_status"] != "PASS" or not all(math.isfinite(float(row[key])) for key in ("train_loss", "validation_loss")) for row in rows):
            raise ValueError("Nonfinite training/validation metrics")
        if not (RUN / "first_epoch_gate.json").is_file():
            raise ValueError("Missing first epoch health gate")
        import torch
        checkpoint = torch.load(final, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 199 and checkpoint["config"]["run_name"] == NAME
        ema = checkpoint.get("ema_model")
        assert ema and all(torch.isfinite(v).all() for v in ema.values() if v.is_floating_point())
        del checkpoint, ema
        if digest(roster) != state["roster_sha256"]:
            raise ValueError("Fixed BraTS21 roster changed")
        execute("inference", "eval.py", EVAL, "--run-eval")
        eval_root = RUN / "evaluation/brats21_test251"
        manifest = json.loads((eval_root / "cache/manifest.json").read_text())
        assert manifest["collection_complete"] and manifest["subjects"] == 251
        assert [r["subject_id"] for r in manifest["entries"]] == ids
        for name in ("ANDi.csv", "ANDi_mf.csv", "inference_report.json", "inference_report.md"):
            assert (eval_root / name).is_file() and (eval_root / name).stat().st_size
        state.update(status="COMPLETE", active_stage=None, finished_at=now(), inference_directory=str(eval_root))
        persist()
    except BaseException as exc:
        active = state.get("active_stage")
        if active:
            state["stages"][active].update(status="FAILED", error=str(exc))
        state.update(status="FAILED", error=str(exc), traceback=traceback.format_exc(), finished_at=now())
        persist()
        raise


if __name__ == "__main__":
    import msvcrt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-training", action="store_true", help="Monitor the existing training process after an authorized eval-config change")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "pipeline.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if STATE.exists() and not args.attach_training:
            raise RuntimeError("This fresh-run controller has already been started; inspect pipeline_status.json before recovery")
        main(attach_training=args.attach_training)
