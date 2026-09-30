"""Durable, restartable raw-source training -> fixed EMA test251 pipeline.

Run prepare_mixed_raw_andi_histmatch.py under the source-volume account first.
This controller may start while that builder is active. Run it under the MRI
data account so the eventual BraTS inference can read the external volumes.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAME = "mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh"
DATASET = ROOT / "outputs/datasets/mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888"
REPORT = ROOT / "outputs/reports" / NAME
RUN = ROOT / "outputs/runs" / NAME
TRAIN_CONFIG = ROOT / "configs" / f"train_{NAME}.yaml"
EVAL_CONFIG = ROOT / "configs" / f"eval_{NAME}_brats21_test251.yaml"
EVAL_ROOT = RUN / "evaluation/brats21_test251"
STATE_PATH = REPORT / "status.json"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def verify_checkpoint(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 199 or payload.get("config", {}).get("run_name") != NAME:
        raise ValueError("Checkpoint epoch/run identity does not match this experiment")
    ema = payload.get("ema_model")
    if not isinstance(ema, dict) or not ema or any(not torch.isfinite(v).all() for v in ema.values() if v.is_floating_point()):
        raise ValueError("Missing/nonfinite final EMA checkpoint")
    return {"path": str(path), "epoch": 199, "sha256": sha(path)}


def prepare_resume_metrics(checkpoint_epoch):
    """Retain only committed epochs; archive superseded metrics and old logs."""
    recovery = RUN / "recovery" / str(time.time_ns())
    metrics_path = RUN / "training_metrics.csv"
    if metrics_path.exists():
        recovery.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metrics_path, recovery / metrics_path.name)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            retained = {int(r["epoch_index"]): r for r in reader if int(r["epoch_index"]) <= checkpoint_epoch}
        if retained:
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(retained[k] for k in sorted(retained))
        else:
            metrics_path.unlink()
    for name in ("stdout.log", "stderr.log", *( ["first_epoch_gate.json"] if checkpoint_epoch < 0 else [])):
        path = RUN / name
        if path.exists():
            recovery.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(recovery / name))


def verify_evaluation(ids):
    manifest = json.loads((EVAL_ROOT / "cache/manifest.json").read_text())
    entries = manifest["entries"]
    if not manifest["collection_complete"] or manifest["subjects"] != 251:
        raise ValueError("Incomplete test251 inference")
    if [r["subject_id"] for r in entries] != ids:
        raise ValueError("Inference cache does not match the fixed test251 order")
    for row in entries:
        if not row.get("has_label") or "mf" not in row:
            raise ValueError("Missing test label or median-filtered score cache")
        paths = [(row["raw_file"], row["shape"], "float32"),
                 (row["label_file"], row["label_shape"], "bool"),
                 (row["mf"]["file"], row["mf"]["shape"], "float32")]
        for name, shape, dtype in paths:
            array = np.load(EVAL_ROOT / "cache" / name, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != shape or str(array.dtype) != dtype or not np.isfinite(array).all():
                raise ValueError(f"Invalid inference cache array: {name}")
    for name in ("ANDi.csv", "ANDi_mf.csv", "inference_report.json", "inference_report.md"):
        if not (EVAL_ROOT / name).is_file() or not (EVAL_ROOT / name).stat().st_size:
            raise ValueError(f"Missing inference result: {name}")
    return {"subjects": 251, "cache": str(EVAL_ROOT / "cache"), "report": str(EVAL_ROOT / "inference_report.md"),
            "metrics": [str(EVAL_ROOT / name) for name in ("ANDi.csv", "ANDi_mf.csv")]}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"stages": {}}
    state.update(status="RUNNING", pid=os.getpid(), updated_at=now())
    state.pop("error", None); state.pop("traceback", None)

    def persist():
        state["updated_at"] = now(); save(STATE_PATH, state)

    def stage(name, status="RUNNING", **fields):
        state["active_stage"] = name
        state["stages"].setdefault(name, {}).update(status=status, updated_at=now(), **fields)
        persist()

    def validate_frozen_inputs():
        for path, digest in state["frozen_files"].items():
            if sha(path) != digest:
                raise ValueError(f"Frozen experiment input changed: {path}")

    def launch(stage_name, script, config, flag, train=False):
        prior = state["stages"].get(stage_name, {})
        attached = None
        if prior.get("pid") and psutil.pid_exists(prior["pid"]):
            process = psutil.Process(prior["pid"])
            command = process.cmdline()
            if str(ROOT / "scripts" / script) not in command or not any(str(RUN) in s for s in command):
                raise RuntimeError("Previous child PID was reused; refusing to attach to a different process")
            attached = process
        stdout = RUN / ("stdout.log" if train else f"{stage_name}_stdout.log")
        stderr = RUN / ("stderr.log" if train else f"{stage_name}_stderr.log")
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        handles = []
        child = None
        try:
            if attached is None:
                handles = [stdout.open("a", encoding="utf-8"), stderr.open("a", encoding="utf-8")]
                child = subprocess.Popen([sys.executable, str(ROOT / "scripts" / script), "--config", str(config), flag],
                                         cwd=ROOT, env=env, stdout=handles[0], stderr=handles[1],
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                pid = child.pid
            else:
                pid = attached.pid
            stage(stage_name, pid=pid, config=str(config), stdout=str(stdout), stderr=str(stderr))
            if train:
                save(RUN / "launch_metadata.json", {"status": "LAUNCHED", "pid": pid, "run_name": NAME,
                     "start_time": now(), "python_executable": sys.executable, "config_snapshot": str(RUN / "config_snapshot.yaml"),
                     "eval_config_snapshot": str(RUN / "eval_config_snapshot.yaml"), "active_config": str(config),
                     "stdout": str(stdout), "stderr": str(stderr), "controller_pid": os.getpid(), "hidden_window": True})
            def alive():
                return child.poll() is None if child else attached.is_running()
            while alive():
                if train and not (RUN / "first_epoch_gate.json").exists() and (RUN / "training_metrics.csv").exists():
                    with (RUN / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
                        metrics = list(csv.DictReader(handle))
                    if metrics and metrics[0].get("finite_status") == "PASS" and metrics[0].get("checkpoint_path") is not None:
                        subprocess.run([sys.executable, str(ROOT / "scripts/check_training_first_epoch.py"), str(RUN)],
                                       cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
                        state["first_epoch_gate"] = json.loads((RUN / "first_epoch_gate.json").read_text())
                        persist()
                if shutil.disk_usage(ROOT).free < 5*1024**3:
                    raise RuntimeError("Free disk space fell below 5 GiB")
                time.sleep(10)
            if child and child.returncode:
                raise RuntimeError(f"{stage_name} exited {child.returncode}; see {stderr}")
            stage(stage_name, status="PASS", finished_at=now())
        except BaseException:
            if child and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
            raise
        finally:
            for handle in handles:
                handle.close()

    try:
        stage("build")
        # A separate source-account builder owns the MPI/OASIS private volumes.
        while not DATASET.exists():
            build_path = REPORT / "build_status.json"
            if not build_path.exists():
                raise RuntimeError("Start prepare_mixed_raw_andi_histmatch.py before the controller")
            build_state = json.loads(build_path.read_text())
            if build_state["status"] == "FAILED":
                raise RuntimeError(f"Dataset build failed: {build_state.get('error')}")
            if not psutil.pid_exists(build_state["pid"]):
                raise RuntimeError("Builder stopped before dataset publication; resume the builder")
            state["stages"]["build"]["builder_progress"] = build_state; persist()
            time.sleep(10)
        build_report = json.loads((DATASET / "build_report.json").read_text())
        if build_report["status"] != "PASS" or build_report["entries"] != {"train": 84341, "val": 9569}:
            raise ValueError("Published dataset failed validation")
        stage("build", status="PASS", report=str(DATASET / "build_report.json"))
        train = yaml.safe_load(TRAIN_CONFIG.read_text())
        evaluation = yaml.safe_load(EVAL_CONFIG.read_text())
        spectrum = Path(train["noise"]["schedule"]["sampler"]["stats_path"])
        if spectrum != Path(evaluation["noise"]["schedule"]["sampler"]["stats_path"]) or sha(spectrum) != build_report["spectrum_sha256"]:
            raise ValueError("Train/eval spectrum identity differs from the verified dataset")
        if evaluation["prediction_output"].get("enabled"):
            raise ValueError("This experiment only saves metrics and inference cache")
        ids = [r["BraTS21ID"] for r in csv.DictReader(Path(evaluation["data"]["path_to_csv"]).open())]
        if len(ids) != 251 or len(set(ids)) != 251:
            raise ValueError("Expected the fixed unique 251-subject roster")
        braTS_root = Path(evaluation["data"]["dataset_path"])
        for case in ids:
            for mod in ("flair", "t1", "t2", "seg"):
                if not (braTS_root / case / f"{case}_{mod}.nii.gz").is_file():
                    raise FileNotFoundError(f"Missing BraTS source: {case}:{mod}")
        frozen = {str(p): sha(p) for p in (TRAIN_CONFIG, EVAL_CONFIG, spectrum,
                   DATASET / "build_report.json", Path(evaluation["data"]["path_to_csv"]))}
        if state.get("frozen_files") and state["frozen_files"] != frozen:
            raise ValueError("Experiment input changed on resume")
        state["frozen_files"] = frozen
        RUN.mkdir(parents=True, exist_ok=True)
        for src, dest in ((TRAIN_CONFIG, RUN / "config_snapshot.yaml"), (EVAL_CONFIG, RUN / "eval_config_snapshot.yaml")):
            if dest.exists() and sha(dest) != sha(src):
                raise ValueError("Existing config snapshot differs")
            if not dest.exists(): shutil.copyfile(src, dest)
        persist()
        if state["stages"].get("preflight", {}).get("status") != "PASS":
            if shutil.disk_usage(ROOT).free < 15*1024**3:
                raise RuntimeError("Need 15 GiB after dataset build for checkpoint/cache/workspace reserve")
            launch("preflight", "train.py", RUN / "config_snapshot.yaml", "--run-one-step")
        final = RUN / "epoch_0199.pt"
        if not final.exists():
            validate_frozen_inputs()
            active_config = RUN / "config_snapshot.yaml"
            prior = state["stages"].get("training", {})
            if prior.get("pid") and psutil.pid_exists(prior["pid"]):
                active_config = Path(prior["config"])
            else:
                checkpoints = sorted(RUN.glob("epoch_*.pt"))
                resume_epoch = -1
                if checkpoints:
                    import torch
                    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
                    if payload.get("config", {}).get("run_name") != NAME:
                        raise ValueError("Refusing a checkpoint from another run")
                    for key in ("optimizer", "scheduler", "ema_model", "ema"):
                        if key not in payload: raise ValueError(f"Resume checkpoint missing {key}")
                    resume_epoch = int(payload["epoch"])
                    resume = yaml.safe_load((RUN / "config_snapshot.yaml").read_text())
                    resume["training"]["checkpoint"]["resume"] = str(checkpoints[-1])
                    active_config = RUN / f"recovery_from_{checkpoints[-1].stem}.yaml"
                    active_config.write_text(yaml.safe_dump(resume, sort_keys=False), encoding="utf-8")
                prepare_resume_metrics(resume_epoch)
            launch("training", "train.py", active_config, "--fit", train=True)
        checkpoint = verify_checkpoint(final)
        if not (RUN / "first_epoch_gate.json").exists():
            raise RuntimeError("Missing first-epoch health gate; inspect training before inference")
        stage("training", status="PASS", checkpoint=checkpoint)
        if state["stages"].get("inference", {}).get("status") != "PASS":
            validate_frozen_inputs()
            launch("inference", "eval.py", RUN / "eval_config_snapshot.yaml", "--run-eval")
        outputs = verify_evaluation(ids)
        stage("inference", status="PASS", outputs=outputs)
        state.update(status="COMPLETE", active_stage=None, finished_at=now()); persist()
    except BaseException as exc:
        state.update(status="FAILED", error=str(exc), traceback=traceback.format_exc(), finished_at=now()); persist()
        raise


if __name__ == "__main__":
    import msvcrt
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "pipeline.lock").open("a+b") as lock:
        if lock.tell() == 0: lock.write(b"0"); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
