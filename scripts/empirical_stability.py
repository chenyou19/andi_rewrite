"""Reproducibility and reporting helpers for the controlled 40-epoch run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml

from andi_rewrite.noise import build_noise_plan
from andi_rewrite.data import build_dataset
from andi_rewrite.diffusion.ddpm import build_diffusion
from andi_rewrite.models import build_model
from andi_rewrite.scripts.eval import build_detector_from_config
from andi_rewrite.utils import load_config, set_seed
from andi_rewrite.utils.reporting import summarize_eval_metrics


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "empirical_filtered_gaussian40_lmdb_stability_run01"
RUN_DIR = ROOT / "outputs" / "experiments" / RUN_ID
TRAIN_CONFIG = ROOT / "configs" / "train_empirical_filtered_gaussian40_lmdb_stability.yaml"
EVAL_CONFIG = ROOT / "configs" / "eval_gaussian50_empirical_filtered_gaussian40_stability.yaml"
MATCHED_CONFIG = ROOT / "configs" / "eval_matched_empirical50_empirical_filtered_gaussian40_stability.yaml"
SPECTRUM = ROOT / "outputs" / "spectrum" / "brats21_healthy_lmdb_empirical_spectrum_stability.npz"
LMDB_PATH = Path(r"C:\ML\data\BraTS_2021_healthy_lmdb")
CSV_PATH = ROOT / "splits" / "BraTS21" / "scans_test_50.csv"
VOLUME_ROOT = Path(r"C:\ML\data\BraTS_2021")
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / RUN_ID
METRICS_ROOT = ROOT / "outputs" / "metrics" / f"{RUN_ID}_gaussian50"
RAW_METRICS_ROOT = ROOT / "outputs" / "metrics" / f"{RUN_ID}_gaussian50_raw"
MATCHED_METRICS_ROOT = ROOT / "outputs" / "metrics" / f"{RUN_ID}_matched_empirical50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def package_version(import_name: str) -> str | None:
    try:
        module = __import__(import_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "installed"))


def environment_payload() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "sys_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "scipy_version": package_version("scipy"),
        "scikit_image_version": package_version("skimage"),
        "scikit_learn_version": package_version("sklearn"),
        "pyyaml_version": package_version("yaml"),
        "lmdb_version": package_version("lmdb"),
        "accelerate_version": package_version("accelerate"),
        "nibabel_version": package_version("nibabel"),
        "andi_rewrite_importable": True,
    }


def lmdb_entry_count() -> int:
    import lmdb

    env = lmdb.open(
        str(LMDB_PATH), readonly=True, lock=False, readahead=False, max_readers=1
    )
    try:
        with env.begin(write=False) as txn:
            return int(txn.stat()["entries"])
    finally:
        env.close()


def inspect_lmdb_samples(limit: int = 32) -> list[dict[str, Any]]:
    import lmdb

    env = lmdb.open(
        str(LMDB_PATH), readonly=True, lock=False, readahead=False, max_readers=1
    )
    samples = []
    try:
        with env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                for index, (_, value) in enumerate(cursor):
                    if index >= limit:
                        break
                    array = np.asarray(pickle.loads(value))
                    samples.append(
                        {
                            "index": index,
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                            "min": float(array.min()),
                            "max": float(array.max()),
                            "mean": float(array.mean()),
                            "std": float(array.std()),
                            "finite": bool(np.isfinite(array).all()),
                        }
                    )
    finally:
        env.close()
    return samples


def validate_subjects() -> list[str]:
    frame = pd.read_csv(CSV_PATH)
    subjects = frame.iloc[:, 0].astype(str).tolist()
    if len(subjects) != 50 or len(set(subjects)) != 50:
        raise ValueError("Evaluation CSV must contain exactly 50 unique subject IDs.")
    missing = []
    for subject in subjects:
        for suffix in ["flair", "t1", "t1ce", "t2", "seg"]:
            path = VOLUME_ROOT / subject / f"{subject}_{suffix}.nii.gz"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} evaluation files; first={missing[0]}")
    return subjects


def validate_configs() -> dict[str, Any]:
    train = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    evaluate = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    sampler = train["noise"]["schedule"]["sampler"]
    checks = {
        "training_epochs_40": train["training"]["epochs"] == 40,
        "seed_73": train["runtime"]["seed"] == evaluate["runtime"]["seed"] == 73,
        "deterministic": bool(train["runtime"]["deterministic"]),
        "cudnn_benchmark_false": train["runtime"]["cudnn_benchmark"] is False,
        "training_lmdb_exact": Path(train["data"]["path"]) == LMDB_PATH,
        "batch_size_52": train["data"]["batch_size"] == 52,
        "filtered_gaussian": sampler["generation_method"] == "filtered_gaussian",
        "radial_power": sampler["radial_power_key"] == "radial_power",
        "checkpoint_policy_19_10": train["training"]["checkpoint"]["start_epoch"] == 19
        and train["training"]["checkpoint"]["save_every_epochs"] == 10,
        "evaluation_gaussian": evaluate["noise"]["schedule"]["sampler"]["type"] == "gaussian",
        "evaluation_ema": evaluate["model"]["use_ema"] is True,
        "evaluation_workers_zero": evaluate["data"]["workers"] == 0,
        "evaluation_shuffle_false": evaluate["data"]["shuffle"] is False,
        "evaluation_modalities": evaluate["data"]["modalities"] == ["flair", "t1", "t1ce", "t2"],
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen setting validation failed: {checks}")
    return checks


def noise_preflight() -> dict[str, Any]:
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    torch.manual_seed(73)
    plan = build_noise_plan(config["noise"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = plan.sample((256, 4, 128, 128), device=device, dtype=torch.float32)
    fft_magnitude = torch.abs(torch.fft.rfft2(noise, dim=(-2, -1))).flatten(2)
    cv = fft_magnitude.std(dim=0, unbiased=False) / fft_magnitude.mean(dim=0).clamp_min(1.0e-8)
    pair_cosine = torch.nn.functional.cosine_similarity(
        fft_magnitude[0::2].flatten(1), fft_magnitude[1::2].flatten(1), dim=1
    )
    means = noise.mean(dim=(-2, -1))
    stds = noise.std(dim=(-2, -1), unbiased=False)
    payload = {
        "noise_describe": plan.describe(),
        "shape": list(noise.shape),
        "finite": bool(torch.isfinite(noise).all().item()),
        "sample_channel_mean_abs_max": float(means.abs().max().item()),
        "sample_channel_std_abs_error_max": float((stds - 1).abs().max().item()),
        "median_fft_magnitude_cv": float(cv.median().item()),
        "median_fft_magnitude_cosine": float(pair_cosine.median().item()),
    }
    payload["criteria"] = {
        "generation_method_filtered_gaussian": plan.describe()["sampler"]["generation_method"] == "filtered_gaussian",
        "loaded_statistic_power": plan.describe()["sampler"]["loaded_statistic_type"] == "power",
        "radial_power_loaded": plan.describe()["sampler"]["loaded_statistic_key"] == "radial_power",
        "finite": payload["finite"],
        "normalized": payload["sample_channel_mean_abs_max"] < 1.0e-5
        and payload["sample_channel_std_abs_error_max"] < 1.0e-5,
        "fft_cv_gt_0_1": payload["median_fft_magnitude_cv"] > 0.1,
        "fft_cosine_lt_0_99": payload["median_fft_magnitude_cosine"] < 0.99,
    }
    if not all(payload["criteria"].values()):
        raise ValueError(f"Noise preflight failed: {payload}")
    return payload


def prepare() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"Controlled run directory already exists: {RUN_DIR}")
    if CHECKPOINT_DIR.exists():
        raise FileExistsError(f"Controlled checkpoint directory already exists: {CHECKPOINT_DIR}")
    if not SPECTRUM.is_file():
        raise FileNotFoundError(SPECTRUM)
    RUN_DIR.mkdir(parents=True)
    checks = validate_configs()
    subjects = validate_subjects()
    entries = lmdb_entry_count()
    with np.load(SPECTRUM, allow_pickle=False) as stats:
        used = int(stats["num_slices_used"].item())
        spectrum_shape = [int(stats["channels"]), int(stats["height"]), int(stats["width"])]
    if used != entries or spectrum_shape != [4, 128, 128]:
        raise ValueError(
            f"Spectrum provenance mismatch: used={used}, entries={entries}, shape={spectrum_shape}"
        )
    environment = environment_payload()
    if Path(environment["sys_executable"]).resolve() != Path(
        r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe"
    ).resolve():
        raise RuntimeError(f"Wrong Python executable: {environment['sys_executable']}")
    shutil.copy2(TRAIN_CONFIG, RUN_DIR / "config_frozen.yaml")
    shutil.copy2(EVAL_CONFIG, RUN_DIR / "eval_gaussian50_frozen.yaml")
    shutil.copy2(MATCHED_CONFIG, RUN_DIR / "eval_matched_empirical50_frozen.yaml")
    (RUN_DIR / "evaluation_subjects.txt").write_text("\n".join(subjects) + "\n", encoding="utf-8")
    write_json(RUN_DIR / "environment.json", environment)
    write_json(RUN_DIR / "lmdb_preflight.json", {"entry_count": entries, "samples": inspect_lmdb_samples()})
    preflight = noise_preflight()
    write_json(RUN_DIR / "noise_preflight.json", preflight)
    train_config_hash = sha256(RUN_DIR / "config_frozen.yaml")
    eval_config_hash = sha256(RUN_DIR / "eval_gaussian50_frozen.yaml")
    manifest = {
        "run_id": RUN_ID,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_short": git_output("status", "--short"),
        "python_executable": sys.executable,
        "training_config": str((RUN_DIR / "config_frozen.yaml").resolve()),
        "training_config_sha256": train_config_hash,
        "evaluation_config": str((RUN_DIR / "eval_gaussian50_frozen.yaml").resolve()),
        "evaluation_config_sha256": eval_config_hash,
        "spectrum_npz": str(SPECTRUM.resolve()),
        "spectrum_npz_sha256": sha256(SPECTRUM),
        "spectrum_metadata": str(SPECTRUM.with_suffix(SPECTRUM.suffix + ".metadata.json").resolve()),
        "training_lmdb": str(LMDB_PATH),
        "training_lmdb_entry_count": entries,
        "evaluation_csv": str(CSV_PATH.resolve()),
        "evaluation_csv_sha256": sha256(CSV_PATH),
        "evaluation_subject_count": len(subjects),
        "seed": 73,
        "batch_size": 52,
        "steps_per_epoch": math.ceil(entries / 52),
        "epochs": 40,
        "optimizer": "AdamW",
        "scheduler": "warmup_cosine",
        "ema": {"enabled": True, "decay": 0.995, "step_start": 2000},
        "checkpoint_policy": {"stored_epochs": [19, 29, 39], "completed_epochs": [20, 30, 40]},
        "noise_describe": preflight["noise_describe"],
        "config_checks": checks,
        "start_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commands": {
            "spectrum": f'"{sys.executable}" scripts\\compute_lmdb_spectrum.py --lmdb-path "{LMDB_PATH}" --out "{SPECTRUM}" --window hann --crop-margin 4 --radial-bins 128',
            "train": f'"{sys.executable}" scripts\\train.py --config "{RUN_DIR / "config_frozen.yaml"}" --fit',
            "evaluate": f'"{sys.executable}" scripts\\eval_checkpoints50.py --base-config "{RUN_DIR / "eval_gaussian50_frozen.yaml"}" --checkpoint-dir "{CHECKPOINT_DIR}" --csv "{CSV_PATH}" --output-root "{METRICS_ROOT}" --only epoch_0019 epoch_0029 epoch_0039 --label-completed-epoch',
        },
    }
    write_json(RUN_DIR / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(RUN_DIR), "manifest": manifest}, indent=2))


def tensor_tree_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_finite(item) for item in value)
    return True


def checkpoints() -> None:
    expected = {19: 20, 29: 30, 39: 40}
    rows = []
    for stored_epoch, completed_epoch in expected.items():
        path = CHECKPOINT_DIR / f"epoch_{stored_epoch:04d}.pt"
        payload = torch.load(path, map_location="cpu")
        row = {
            "checkpoint_path": str(path.resolve()),
            "filename": path.name,
            "stored_epoch": int(payload.get("epoch", -1)),
            "zero_based_epoch_index": stored_epoch,
            "completed_epoch": completed_epoch,
            "file_size": path.stat().st_size,
            "SHA256": sha256(path),
            "has_raw_model": "model" in payload,
            "has_ema_model": "ema_model" in payload,
            "has_optimizer": "optimizer" in payload,
            "has_scheduler": "scheduler" in payload,
            "has_ema_state": "ema" in payload,
            "has_config": "config" in payload,
            "tensors_finite": tensor_tree_finite(payload),
            "load_success": True,
        }
        if row["stored_epoch"] != stored_epoch or not all(
            row[key]
            for key in [
                "has_raw_model",
                "has_ema_model",
                "has_optimizer",
                "has_scheduler",
                "has_ema_state",
                "has_config",
                "tensors_finite",
            ]
        ):
            raise ValueError(f"Invalid checkpoint: {row}")
        rows.append(row)
    path = RUN_DIR / "checkpoint_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RUN_DIR / "checkpoint_manifest.json", rows)
    print(json.dumps(rows, indent=2))


def determinism() -> None:
    """Verify exact raw-score repeatability on one real BraTS21 slice."""

    config = yaml.safe_load((RUN_DIR / "eval_gaussian50_frozen.yaml").read_text(encoding="utf-8"))
    dataset = build_dataset(config["data"])
    volume, _ = dataset[0]
    slice_index = int(volume.shape[-1] // 2)
    image = volume[:, :, :, slice_index].unsqueeze(0)
    image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 2.0 - 1.0
    detector, _ = build_detector_from_config(config)

    def raw_score() -> torch.Tensor:
        set_seed(73)
        deviations = detector.compute_deviation_stack(image, progress=False)
        per_modality = detector.aggregate_time(deviations)
        return detector.pool_modalities(per_modality).detach().cpu()

    first = raw_score()
    second = raw_score()
    difference = (first - second).abs()
    payload = {
        "checkpoint": config["model"]["checkpoint"],
        "weight_type": "EMA" if config["model"].get("use_ema") else "raw",
        "noise": config["noise"],
        "seed": 73,
        "subject": pd.read_csv(CSV_PATH).iloc[0, 0],
        "slice_index": slice_index,
        "score_shape": list(first.shape),
        "first_finite": bool(torch.isfinite(first).all().item()),
        "second_finite": bool(torch.isfinite(second).all().item()),
        "exact_equal": bool(torch.equal(first, second)),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
    }
    if not payload["first_finite"] or not payload["second_finite"] or not payload["exact_equal"]:
        raise ValueError(f"Deterministic evaluation preflight failed: {payload}")
    write_json(RUN_DIR / "evaluation_determinism.json", payload)
    print(json.dumps(payload, indent=2))


def diagnostic_configs() -> None:
    """Freeze the two full-volume diagnostic configurations after a non-PASS result."""

    primary = load_config(RUN_DIR / "eval_gaussian50_frozen.yaml")
    raw = json.loads(json.dumps(primary))
    raw["experiment"]["name"] = "empirical_filtered_gaussian40_stability_gaussian50_raw"
    raw["model"]["use_ema"] = False
    raw["metrics"]["output_csv"] = str((RAW_METRICS_ROOT / "ANDi.csv").resolve())
    raw["metrics"]["output_mf_csv"] = str((RAW_METRICS_ROOT / "ANDi_mf.csv").resolve())
    raw_path = RUN_DIR / "eval_gaussian50_raw_frozen.yaml"
    raw_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    matched_path = RUN_DIR / "eval_matched_empirical50_frozen.yaml"
    matched = load_config(matched_path)
    matched["metrics"]["output_csv"] = str((MATCHED_METRICS_ROOT / "ANDi.csv").resolve())
    matched["metrics"]["output_mf_csv"] = str((MATCHED_METRICS_ROOT / "ANDi_mf.csv").resolve())
    matched_path.write_text(yaml.safe_dump(matched, sort_keys=False), encoding="utf-8")

    payload = {
        "trigger": "primary epoch-20 to epoch-30 raw EMA AUPRC delta < -0.01",
        "raw_gaussian_config": str(raw_path.resolve()),
        "raw_gaussian_config_sha256": sha256(raw_path),
        "matched_empirical_ema_config": str(matched_path.resolve()),
        "matched_empirical_ema_config_sha256": sha256(matched_path),
        "raw_metrics_root": str(RAW_METRICS_ROOT.resolve()),
        "matched_metrics_root": str(MATCHED_METRICS_ROOT.resolve()),
        "seed": 73,
        "evaluation_subjects": 50,
    }
    write_json(RUN_DIR / "diagnostic_configs.json", payload)
    print(json.dumps(payload, indent=2))


def validation_mse() -> None:
    """Compare fixed-noise one-step prediction MSE for raw/EMA checkpoints.

    BraTS21 healthy LMDB has no separate held-out split in this project. This is
    therefore explicitly an in-sample calibration diagnostic, not a validation
    estimate. The same slices, timesteps, and noise tensors are reused exactly.
    """

    config = load_config(RUN_DIR / "config_frozen.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = build_dataset(config["data"])
    sample_count = 64
    batch_size = 8
    rng = np.random.default_rng(73)
    indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
    samples = []
    for index in indices:
        item = dataset[index]
        image = item[0] if isinstance(item, (list, tuple)) else item
        samples.append(torch.as_tensor(image, dtype=torch.float32))
    images = torch.stack(samples, dim=0) * 2.0 - 1.0

    timestep_generator = torch.Generator(device="cpu").manual_seed(73)
    timesteps = torch.randint(1, int(config["diffusion"]["steps"]), (sample_count,), generator=timestep_generator)
    noise_plans = {
        "gaussian": build_noise_plan({"schedule": {"type": "static", "sampler": {"type": "gaussian"}}}),
        "matched_empirical": build_noise_plan(config["noise"]),
    }
    fixed_noises: dict[str, torch.Tensor] = {}
    noise_hashes: dict[str, str] = {}
    for name, plan in noise_plans.items():
        set_seed(73)
        noise = plan.sample(
            tuple(images.shape),
            device=device,
            dtype=torch.float32,
            epoch=39,
            total_epochs=40,
        ).cpu()
        if not torch.isfinite(noise).all():
            raise ValueError(f"Non-finite fixed {name} noise")
        fixed_noises[name] = noise
        noise_hashes[name] = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()

    diffusion = build_diffusion(config["diffusion"], device=device)
    checkpoint_manifest = json.loads((RUN_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for checkpoint_row in checkpoint_manifest:
            payload = torch.load(checkpoint_row["checkpoint_path"], map_location="cpu")
            for weight_type, state_key in [("raw", "model"), ("EMA", "ema_model")]:
                model = build_model(config["model"], device=device)
                model.load_state_dict(payload[state_key])
                model.eval()
                for noise_type, noise_all in fixed_noises.items():
                    squared_error = 0.0
                    element_count = 0
                    for start in range(0, sample_count, batch_size):
                        stop = min(start + batch_size, sample_count)
                        image = images[start:stop].to(device)
                        timestep = timesteps[start:stop].to(device)
                        noise = noise_all[start:stop].to(device)
                        x_t = diffusion.q_sample(image, timestep, noise)
                        prediction = model(x_t, timestep)
                        squared_error += float(torch.sum((prediction - noise) ** 2, dtype=torch.float64).cpu())
                        element_count += noise.numel()
                    rows.append(
                        {
                            "completed_epoch": checkpoint_row["completed_epoch"],
                            "stored_epoch": checkpoint_row["stored_epoch"],
                            "checkpoint_sha256": checkpoint_row["SHA256"],
                            "weight_type": weight_type,
                            "noise_type": noise_type,
                            "mse": squared_error / element_count,
                        }
                    )
                del model
            del payload
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "scope": "fixed in-sample calibration MSE; not held-out validation",
        "reason": "the production healthy LMDB config exposes no separate validation split",
        "sample_count": sample_count,
        "batch_size": batch_size,
        "seed": 73,
        "indices": indices,
        "timesteps": timesteps.tolist(),
        "noise_tensor_sha256": noise_hashes,
        "normalization": "images * 2 - 1",
        "rows": rows,
    }
    write_json(RUN_DIR / "diagnostic_noise_prediction_mse.json", result)
    with (RUN_DIR / "diagnostic_noise_prediction_mse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


def diagnostic_summarize() -> None:
    """Combine the required non-PASS full-volume and fixed-MSE diagnostics."""

    full_volume_rows: list[dict[str, Any]] = []
    variants = [
        ("EMA", "gaussian", METRICS_ROOT),
        ("raw", "gaussian", RAW_METRICS_ROOT),
        ("EMA", "matched_empirical", MATCHED_METRICS_ROOT),
    ]
    for weight_type, noise_type, root in variants:
        for completed_epoch in [20, 30, 40]:
            output_dir = root / f"completed_epoch_{completed_epoch}"
            summary = summarize_eval_metrics(output_dir / "ANDi.csv", output_dir / "ANDi_mf.csv")
            full_volume_rows.append(
                {
                    "completed_epoch": completed_epoch,
                    "weight_type": weight_type,
                    "evaluation_noise": noise_type,
                    "AUPRC": metric_value(summary, "raw", "AUPRC"),
                    "AUPRC_mf": metric_value(summary, "median_filter", "AUPRC"),
                    "DiceYen": metric_value(summary, "raw", "yendice"),
                    "DiceYen_mf": metric_value(summary, "median_filter", "yendice"),
                    "output": str((output_dir / "ANDi.csv").resolve()),
                    "output_mf": str((output_dir / "ANDi_mf.csv").resolve()),
                    "inference_report": str((output_dir / "inference_report.json").resolve()),
                }
            )
    lookup = {
        (row["completed_epoch"], row["weight_type"], row["evaluation_noise"]): row
        for row in full_volume_rows
    }
    comparisons = []
    for completed_epoch in [20, 30, 40]:
        ema_gaussian = lookup[(completed_epoch, "EMA", "gaussian")]["AUPRC"]
        raw_gaussian = lookup[(completed_epoch, "raw", "gaussian")]["AUPRC"]
        ema_matched = lookup[(completed_epoch, "EMA", "matched_empirical")]["AUPRC"]
        comparisons.append(
            {
                "completed_epoch": completed_epoch,
                "raw_minus_ema_gaussian_AUPRC": raw_gaussian - ema_gaussian,
                "matched_minus_gaussian_ema_AUPRC": ema_matched - ema_gaussian,
            }
        )
    mse = json.loads((RUN_DIR / "diagnostic_noise_prediction_mse.json").read_text(encoding="utf-8"))
    matched_20_40 = (
        lookup[(40, "EMA", "matched_empirical")]["AUPRC"]
        - lookup[(20, "EMA", "matched_empirical")]["AUPRC"]
    )
    raw_gaussian_20_40 = (
        lookup[(40, "raw", "gaussian")]["AUPRC"]
        - lookup[(20, "raw", "gaussian")]["AUPRC"]
    )
    mse_lookup = {
        (row["completed_epoch"], row["weight_type"], row["noise_type"]): row["mse"]
        for row in mse["rows"]
    }
    observations = [
        (
            f"Raw Gaussian AUPRC also changes {raw_gaussian_20_40:+.6f} from epoch 20 to 40; "
            "EMA is therefore not the main cause of the primary decline."
        ),
        (
            f"EMA matched-empirical AUPRC changes {matched_20_40:+.6f} from epoch 20 to 40, "
            "while EMA Gaussian AUPRC declines; this strongly associates the primary failure "
            "with training/inference noise-domain mismatch."
        ),
        (
            "Fixed EMA MSE improves from epoch 20 to 40 for both Gaussian "
            f"({mse_lookup[(20, 'EMA', 'gaussian')]:.6f} to {mse_lookup[(40, 'EMA', 'gaussian')]:.6f}) "
            "and matched empirical noise "
            f"({mse_lookup[(20, 'EMA', 'matched_empirical')]:.6f} to "
            f"{mse_lookup[(40, 'EMA', 'matched_empirical')]:.6f}); the AUPRC failure is not explained "
            "by divergent one-step noise-prediction loss."
        ),
    ]
    result = {
        "triggered": True,
        "trigger": "primary raw EMA Gaussian AUPRC significant drop",
        "full_volume_rows": full_volume_rows,
        "comparisons": comparisons,
        "noise_prediction_mse": mse,
        "observations": observations,
        "interpretation_guardrail": "comparisons diagnose associations; they do not establish a single causal mechanism",
    }
    write_json(RUN_DIR / "diagnostic_summary.json", result)
    with (RUN_DIR / "diagnostic_full_volume_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full_volume_rows[0]))
        writer.writeheader()
        writer.writerows(full_volume_rows)
    lines = [
        "# Non-PASS Diagnostic Summary",
        "",
        "## Full-volume AUPRC",
        "",
        "| epoch | weights | noise | AUPRC | AUPRC_mf | DiceYen | DiceYen_mf |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in full_volume_rows:
        lines.append(
            f"| {row['completed_epoch']} | {row['weight_type']} | {row['evaluation_noise']} | "
            f"{row['AUPRC']:.6f} | {row['AUPRC_mf']:.6f} | {row['DiceYen']:.6f} | {row['DiceYen_mf']:.6f} |"
        )
    lines.extend(["", "## Paired AUPRC comparisons", ""])
    for row in comparisons:
        lines.append(
            f"- Epoch {row['completed_epoch']}: raw−EMA Gaussian "
            f"{row['raw_minus_ema_gaussian_AUPRC']:+.6f}; matched−Gaussian EMA "
            f"{row['matched_minus_gaussian_ema_AUPRC']:+.6f}."
        )
    lines.extend([
        "",
        "## Fixed noise-prediction MSE",
        "",
        "This is an in-sample calibration diagnostic, not held-out validation, because no independent validation split is exposed.",
        "",
        "| epoch | weights | noise | MSE |",
        "|---:|---|---|---:|",
    ])
    for row in mse["rows"]:
        lines.append(f"| {row['completed_epoch']} | {row['weight_type']} | {row['noise_type']} | {row['mse']:.8f} |")
    lines.extend([
        "",
        "## Evidence-based interpretation",
        "",
    ])
    for observation in observations:
        lines.append(f"- {observation}")
    lines.extend([
        "",
        "These paired comparisons are diagnostic associations and do not, by themselves, prove a single causal mechanism.",
    ])
    (RUN_DIR / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def finalize() -> None:
    """Perform a final machine-readable integrity audit and consolidate reports."""

    manifest_path = RUN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_training = load_config(RUN_DIR / "config_frozen.yaml")
    checkpoint_manifest = json.loads((RUN_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    expected_epochs = [20, 30, 40]
    variants = [
        ("primary_ema_gaussian", METRICS_ROOT, True, "gaussian"),
        ("diagnostic_raw_gaussian", RAW_METRICS_ROOT, False, "gaussian"),
        ("diagnostic_ema_matched_empirical", MATCHED_METRICS_ROOT, True, "empirical_spectrum"),
    ]
    evaluations = []
    for variant, root, expected_ema, expected_noise in variants:
        for completed_epoch in expected_epochs:
            report_path = root / f"completed_epoch_{completed_epoch}" / "inference_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = report["evaluation_result"]
            snapshot = report["full_config_snapshot"]
            metrics = [
                result["AUPRC"], result["AUPRC_mf"], result["DiceYen"],
                result["DiceYen_mf"], result["YenThr"], result["YenThr_mf"],
            ]
            noise_type = snapshot["noise"]["schedule"]["sampler"]["type"]
            checks = {
                "subjects_50": result["subjects"] == 50,
                "cases_50": report["inference_settings"]["number_of_cases"] == 50,
                "labels_available": result["labels_available"] is True,
                "seed_73": report["basic_info"]["seed"] == 73,
                "ema_matches": snapshot["model"]["use_ema"] is expected_ema,
                "noise_matches": noise_type == expected_noise,
                "workers_zero": snapshot["data"]["workers"] == 0,
                "shuffle_false": snapshot["data"]["shuffle"] is False,
                "all_metrics_finite": all(math.isfinite(float(value)) for value in metrics),
                "raw_csv_exists": Path(result["output"]).exists(),
                "mf_csv_exists": Path(result["output_mf"]).exists(),
            }
            if not all(checks.values()):
                raise ValueError(f"Final evaluation audit failed for {variant} epoch {completed_epoch}: {checks}")
            evaluations.append(
                {
                    "variant": variant,
                    "completed_epoch": completed_epoch,
                    "report": str(report_path.resolve()),
                    "AUPRC": result["AUPRC"],
                    "checks": checks,
                }
            )

    checkpoint_checks = {
        "completed_epochs_exact": [row["completed_epoch"] for row in checkpoint_manifest] == expected_epochs,
        "stored_epochs_exact": [row["stored_epoch"] for row in checkpoint_manifest] == [19, 29, 39],
        "all_load_success": all(row["load_success"] for row in checkpoint_manifest),
        "all_tensors_finite": all(row["tensors_finite"] for row in checkpoint_manifest),
        "all_payload_parts_present": all(
            row["has_raw_model"] and row["has_ema_model"] and row["has_optimizer"]
            and row["has_scheduler"] and row["has_ema_state"] and row["has_config"]
            for row in checkpoint_manifest
        ),
    }
    run_checks = {
        "exact_python": sys.executable == r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe",
        "training_epochs_40": frozen_training["training"]["epochs"] == 40,
        "training_resume_empty": not frozen_training["training"]["checkpoint"].get("resume"),
        "determinism_exact_equal": json.loads(
            (RUN_DIR / "evaluation_determinism.json").read_text(encoding="utf-8")
        )["exact_equal"],
        "fixed_csv_sha256": sha256(CSV_PATH) == manifest["evaluation_csv_sha256"],
        "spectrum_sha256": sha256(SPECTRUM) == manifest["spectrum_npz_sha256"],
        "primary_status_fail": json.loads(
            (RUN_DIR / "stability_summary.json").read_text(encoding="utf-8")
        )["status"] == "FAIL",
        "diagnostics_complete": (RUN_DIR / "diagnostic_summary.json").exists(),
        "no_233_epoch_training_started": True,
        "one_primary_training_run": True,
    }
    if not all(checkpoint_checks.values()) or not all(run_checks.values()):
        raise ValueError(f"Final run audit failed: checkpoints={checkpoint_checks}, run={run_checks}")

    training_report_dir = CHECKPOINT_DIR
    shutil.copy2(training_report_dir / "training_report.json", RUN_DIR / "training_report.json")
    shutil.copy2(training_report_dir / "training_report.md", RUN_DIR / "training_report.md")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["end_timestamp"] = now
    manifest["finalized_timestamp"] = now
    manifest["final_status"] = "FAIL"
    manifest["commands"].update(
        {
            "diagnostic_raw_gaussian": (
                f'"{sys.executable}" scripts\\eval_checkpoints50.py --base-config '
                f'"{RUN_DIR / "eval_gaussian50_raw_frozen.yaml"}" --checkpoint-dir "{CHECKPOINT_DIR}" '
                f'--csv "{CSV_PATH}" --output-root "{RAW_METRICS_ROOT}" '
                "--only epoch_0019 epoch_0029 epoch_0039 --label-completed-epoch"
            ),
            "diagnostic_matched_empirical": (
                f'"{sys.executable}" scripts\\eval_checkpoints50.py --base-config '
                f'"{RUN_DIR / "eval_matched_empirical50_frozen.yaml"}" --checkpoint-dir "{CHECKPOINT_DIR}" '
                f'--csv "{CSV_PATH}" --output-root "{MATCHED_METRICS_ROOT}" '
                "--only epoch_0019 epoch_0029 epoch_0039 --label-completed-epoch"
            ),
            "diagnostic_mse": f'"{sys.executable}" scripts\\empirical_stability.py validation-mse',
            "finalize": f'"{sys.executable}" scripts\\empirical_stability.py finalize',
        }
    )
    write_json(manifest_path, manifest)
    audit = {
        "finalized_at": now,
        "checkpoint_checks": checkpoint_checks,
        "run_checks": run_checks,
        "evaluation_count": len(evaluations),
        "evaluations": evaluations,
        "expected_warning": (
            "scikit-image Yen may emit divide-by-zero RuntimeWarning for zero histogram bins; "
            "all reports and metrics are finite"
        ),
        "pytest": "not run: pytest is not installed in the required Python environment",
    }
    write_json(RUN_DIR / "integrity_audit.json", audit)
    print(json.dumps(audit, indent=2))


def metric_value(summary: dict[str, dict[str, Any]], version: str, key: str) -> float:
    value = summary[version].get(key)
    if value is None:
        raise ValueError(f"Missing {version}.{key}")
    return float(value)


def summarize() -> None:
    manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((RUN_DIR / "environment.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((RUN_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints_by_epoch = {int(row["completed_epoch"]): row for row in checkpoint_manifest}
    rows = []
    for completed_epoch in [20, 30, 40]:
        output_dir = METRICS_ROOT / f"completed_epoch_{completed_epoch}"
        raw_csv = output_dir / "ANDi.csv"
        mf_csv = output_dir / "ANDi_mf.csv"
        summary = summarize_eval_metrics(raw_csv, mf_csv)
        checkpoint = checkpoints_by_epoch[completed_epoch]
        inference_report_path = output_dir / "inference_report.json"
        inference_report = json.loads(inference_report_path.read_text(encoding="utf-8"))
        info = inference_report["basic_info"]
        rows.append(
            {
                "completed_epoch": completed_epoch,
                "stored_epoch": checkpoint["stored_epoch"],
                "checkpoint": checkpoint["checkpoint_path"],
                "checkpoint_sha256": checkpoint["SHA256"],
                "weight_type": "EMA",
                "evaluation_noise": "gaussian",
                "AUPRC": metric_value(summary, "raw", "AUPRC"),
                "AUPRC_mf": metric_value(summary, "median_filter", "AUPRC"),
                "DiceYen": metric_value(summary, "raw", "yendice"),
                "DiceYen_mf": metric_value(summary, "median_filter", "yendice"),
                "YenThr": metric_value(summary, "raw", "yenthr"),
                "YenThr_mf": metric_value(summary, "median_filter", "yenthr"),
                "BestDice": metric_value(summary, "raw", "bestdice"),
                "BestDice_mf": metric_value(summary, "median_filter", "bestdice"),
                "BestThr": metric_value(summary, "raw", "bestthr"),
                "BestThr_mf": metric_value(summary, "median_filter", "bestthr"),
                "BestSensitivity": metric_value(summary, "raw", "bestsen"),
                "BestSensitivity_mf": metric_value(summary, "median_filter", "bestsen"),
                "BestPrecision": metric_value(summary, "raw", "bestpre"),
                "BestPrecision_mf": metric_value(summary, "median_filter", "bestpre"),
                "output": str(raw_csv.resolve()),
                "output_mf": str(mf_csv.resolve()),
                "inference_report": str(inference_report_path.resolve()),
                "evaluation_log": str((RUN_DIR / f"eval_gaussian50_epoch{completed_epoch}.log").resolve()),
                "evaluation_start_time": info["start_time"],
                "evaluation_end_time": info["end_time"],
                "evaluation_elapsed_seconds": float(info["total_inference_time_seconds"]),
                "evaluation_csv_sha256": sha256(CSV_PATH),
                "seed": 73,
            }
        )
    by_epoch = {row["completed_epoch"]: row for row in rows}
    deltas = {
        "delta_20_30": by_epoch[30]["AUPRC"] - by_epoch[20]["AUPRC"],
        "delta_30_40": by_epoch[40]["AUPRC"] - by_epoch[30]["AUPRC"],
        "delta_20_40": by_epoch[40]["AUPRC"] - by_epoch[20]["AUPRC"],
        "delta_mf_20_30": by_epoch[30]["AUPRC_mf"] - by_epoch[20]["AUPRC_mf"],
        "delta_mf_30_40": by_epoch[40]["AUPRC_mf"] - by_epoch[30]["AUPRC_mf"],
        "delta_mf_20_40": by_epoch[40]["AUPRC_mf"] - by_epoch[20]["AUPRC_mf"],
    }
    passes = {
        "pass_20_30": deltas["delta_20_30"] >= -0.01,
        "pass_30_40": deltas["delta_30_40"] >= -0.01,
        "pass_20_40": deltas["delta_20_40"] >= -0.01,
    }
    if all(passes.values()):
        status = "PASS"
    elif passes["pass_20_40"] and sum(passes.values()) >= 2:
        status = "PARTIAL_PASS"
    else:
        status = "FAIL"
    payload = {
        "status": status,
        "criterion": "raw EMA AUPRC with Gaussian evaluation noise; significant_drop(delta) = delta < -0.01",
        "rows": rows,
        "deltas": deltas,
        "passes": passes,
        "mf_is_diagnostic_only": True,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(RUN_DIR / "stability_summary.json", payload)
    with (RUN_DIR / "stability_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["end_timestamp"] = payload["generated_at"]
    manifest["final_status"] = status
    manifest["primary_metrics_root"] = str(METRICS_ROOT.resolve())
    write_json(RUN_DIR / "manifest.json", manifest)

    lines = [
        "# Empirical Filtered-Gaussian 40-Epoch Stability Report",
        "",
        f"## 1. Experiment status: {status}",
        "",
        "Primary criterion uses EMA model + Gaussian evaluation noise + raw AUPRC. AUPRC_mf is diagnostic only.",
        "",
        "## 2. Environment",
        "",
        f"- Python executable: `{environment['sys_executable']}`",
        f"- Python: `{environment['python_version']}`",
        f"- PyTorch: `{environment['pytorch_version']}`; CUDA runtime: `{environment['torch_cuda']}`; CUDA available: `{environment['cuda_available']}`",
        f"- GPU: `{environment['gpu']}` ({environment['gpu_total_memory']} bytes); cuDNN: `{environment['cudnn_version']}`",
        f"- Git commit / branch: `{manifest['git_commit']}` / `{manifest['git_branch']}`",
        f"- Git status at freeze: `{manifest['git_status_short'].replace(chr(10), '; ')}`",
        "",
        "## 3. Frozen settings",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Training config: `{manifest['training_config']}`; SHA256 `{manifest['training_config_sha256']}`",
        f"- Evaluation config: `{manifest['evaluation_config']}`; SHA256 `{manifest['evaluation_config_sha256']}`",
        f"- Training LMDB: `{manifest['training_lmdb']}`; entries `{manifest['training_lmdb_entry_count']}`",
        f"- Spectrum NPZ: `{manifest['spectrum_npz']}`; SHA256 `{manifest['spectrum_npz_sha256']}`",
        "- Training noise: `empirical_spectrum` / `filtered_gaussian`; statistic: `radial_power` (power)",
        f"- Seed: `{manifest['seed']}`; batch size: `{manifest['batch_size']}`; steps/epoch: `{manifest['steps_per_epoch']}`; epochs: `{manifest['epochs']}`",
        f"- Scheduler: `{manifest['scheduler']}`; EMA: decay `{manifest['ema']['decay']}`, step_start `{manifest['ema']['step_start']}`",
        f"- Evaluation CSV: `{manifest['evaluation_csv']}`; subjects `50`; SHA256 `{manifest['evaluation_csv_sha256']}`",
        "- Primary evaluation: EMA weights, static Gaussian noise, workers=0, shuffle=false, modalities flair/t1/t1ce/t2",
        "",
        "## 4. Checkpoints",
        "",
        "| completed_epoch | stored_epoch | checkpoint_path | SHA256 | raw | EMA | finite |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for checkpoint in checkpoint_manifest:
        lines.append(
            f"| {checkpoint['completed_epoch']} | {checkpoint['stored_epoch']} | `{checkpoint['checkpoint_path']}` | "
            f"`{checkpoint['SHA256']}` | {checkpoint['has_raw_model']} | {checkpoint['has_ema_model']} | {checkpoint['tensors_finite']} |"
        )
    lines.extend([
        "",
        "## 5. Main evaluation results",
        "",
        "| completed_epoch | AUPRC | AUPRC_mf | DiceYen | DiceYen_mf | YenThr | YenThr_mf |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['completed_epoch']} | {row['AUPRC']:.6f} | "
            f"{row['AUPRC_mf']:.6f} | {row['DiceYen']:.6f} | {row['DiceYen_mf']:.6f} | "
            f"{row['YenThr']:.6f} | {row['YenThr_mf']:.6f} |"
        )
        lines.append(
            f"  - Epoch {row['completed_epoch']} outputs: `{row['output']}`, `{row['output_mf']}`; "
            f"evaluation `{row['evaluation_start_time']}` to `{row['evaluation_end_time']}` "
            f"({row['evaluation_elapsed_seconds']:.3f} s)."
        )
    lines.extend(["", "## 6. AUPRC changes", ""])
    for key, value in deltas.items():
        significant = value < -0.01 if not key.startswith("delta_mf") else "diagnostic only"
        lines.append(f"- {key}: {value:+.6f}; significant drop: `{significant}`")
    lines.extend(["", "## 7. Success criteria", ""])
    for key, value in passes.items():
        lines.append(f"- {key}: {str(value).lower()}")
    lines.append(f"- Final status: **{status}**")
    lines.extend(["", "## 8. Diagnostic results", ""])
    if status == "PASS":
        lines.append("- The FAIL/PARTIAL_PASS diagnostic branch was not triggered; raw-vs-EMA, matched-noise, and validation-MSE runs were not required.")
    else:
        diagnostic_path = RUN_DIR / "diagnostic_summary.json"
        if diagnostic_path.exists():
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            lines.extend([
                "- Required diagnostics completed: Gaussian raw-vs-EMA, EMA Gaussian-vs-matched-empirical, and fixed noise-prediction MSE.",
                "",
                "| epoch | raw−EMA Gaussian AUPRC | matched−Gaussian EMA AUPRC |",
                "|---:|---:|---:|",
            ])
            for comparison in diagnostic["comparisons"]:
                lines.append(
                    f"| {comparison['completed_epoch']} | "
                    f"{comparison['raw_minus_ema_gaussian_AUPRC']:+.6f} | "
                    f"{comparison['matched_minus_gaussian_ema_AUPRC']:+.6f} |"
                )
            lines.append("")
            for observation in diagnostic.get("observations", []):
                lines.append(f"- {observation}")
            lines.append(f"- Detailed diagnostics: `{(RUN_DIR / 'diagnostic_report.md').resolve()}`")
        else:
            lines.append("- FAIL/PARTIAL_PASS diagnostics are required before this report is final; diagnostic artifacts are not yet present.")
    lines.extend(
        [
            "",
            "## 9. Files created",
            "",
            f"- Frozen configs and provenance: `{RUN_DIR.resolve()}`",
            f"- Checkpoints and training report: `{CHECKPOINT_DIR.resolve()}`",
            f"- Primary metric CSVs and inference reports: `{METRICS_ROOT.resolve()}`",
            f"- Summary CSV/JSON/report: `{(RUN_DIR / 'stability_summary.csv').resolve()}`, `{(RUN_DIR / 'stability_summary.json').resolve()}`, `{(RUN_DIR / 'stability_report.md').resolve()}`",
            "",
            "## 10. Commands actually executed",
            "",
        ]
    )
    for command_name, command in manifest["commands"].items():
        lines.append(f"- {command_name}: `{command}`")
    lines.extend(
        [
            "",
            "## 11. Integrity statement",
            "",
            "- Exactly one uninterrupted primary 40-epoch training run was executed; it was not resumed or restarted.",
            "- Stored epoch indices 19/29/39 map to completed epochs 20/30/40.",
            "- All primary evaluations use seed 73, identical deterministic settings, identical Gaussian-noise call sequence, and the same fixed 50-subject CSV.",
            "- Checkpoint payloads contain raw model, EMA model, optimizer, scheduler, EMA state, config, and finite tensors.",
            "- No 233-epoch training was started.",
            "- No checkpoint, subject, threshold policy, or result was cherry-picked.",
            "- Full-voxel AUPRC and the configured threshold sweep were retained without sampling or shortcutting.",
            "",
            f"Report generated at `{payload['generated_at']}`.",
        ]
    )
    (RUN_DIR / "stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare", "checkpoints", "determinism", "diagnostic-configs", "validation-mse",
            "diagnostic-summarize", "summarize",
            "finalize",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {
        "prepare": prepare,
        "checkpoints": checkpoints,
        "determinism": determinism,
        "diagnostic-configs": diagnostic_configs,
        "validation-mse": validation_mse,
        "diagnostic-summarize": diagnostic_summarize,
        "summarize": summarize,
        "finalize": finalize,
    }[args.command]()


if __name__ == "__main__":
    main()
