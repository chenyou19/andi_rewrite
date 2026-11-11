"""Compare committed raw-source/IQR checkpoints and, when ready, test251 results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "iqr": "mixed_sri24_robust_iqr_andi_histmatch_mean888_200ep_fresh",
    "raw": "mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh",
}
OUT = ROOT / "outputs/reports/mixed_raw_vs_iqr_mean888"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def snapshot_metrics(path):
    contents = path.read_bytes()
    rows = {}
    for row in csv.DictReader(io.StringIO(contents.decode("utf-8"))):
        if not row.get("finite_status"):
            continue  # A concurrently appended incomplete row is not committed.
        epoch = int(row["completed_epoch"])
        if epoch in rows:
            raise ValueError(f"Duplicate epoch {epoch}: {path}")
        values = {key: float(row[key]) for key in
                  ("train_loss", "validation_loss", "learning_rate", "elapsed_seconds")}
        if row["finite_status"] != "PASS" or not all(np.isfinite(list(values.values()))):
            raise ValueError(f"Unhealthy epoch {epoch}: {path}")
        rows[epoch] = {**values, "ema_step": int(row["ema_step"])}
    if sorted(rows) != list(range(1, max(rows) + 1)):
        raise ValueError(f"Noncontiguous metrics: {path}")
    return rows, {"path": str(path), "sha256_at_snapshot": hashlib.sha256(contents).hexdigest(),
                  "completed_epochs": max(rows)}


def audit_checkpoint(path, name, metrics):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(payload["epoch"]) + 1
    if path.name != f"epoch_{epoch-1:04d}.pt" or payload["config"]["run_name"] != name:
        raise ValueError(f"Wrong checkpoint identity: {path}")
    for field in ("optimizer", "scheduler", "ema_model", "ema", "model"):
        if field not in payload:
            raise ValueError(f"Missing checkpoint field {field}: {path}")
    shapes = {}
    counts = {}
    for field in ("model", "ema_model"):
        weights = payload[field]
        if not weights or any(not bool(torch.isfinite(v).all()) for v in weights.values()
                              if v.is_floating_point()):
            raise ValueError(f"Nonfinite weights: {path}:{field}")
        shapes[field] = {key: list(v.shape) for key, v in weights.items()}
        counts[field] = sum(v.numel() for v in weights.values())
    if shapes["model"] != shapes["ema_model"]:
        raise ValueError(f"Model/EMA architecture mismatch: {path}")
    step = int(payload["ema"]["step"])
    if step != metrics["ema_step"] or int(payload["scheduler"]["last_epoch"]) != step:
        raise ValueError(f"Checkpoint progress disagrees with metrics: {path}")
    audit = {"path": str(path), "sha256": digest(path), "completed_epoch": epoch,
             "ema_step": step, "scheduler_step": step, "finite_model_and_ema": True,
             "tensor_elements": counts, "resume_state_present": True}
    return audit, shapes["model"]


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def compare_evaluation():
    reports = {}
    roster = [r["BraTS21ID"] for r in csv.DictReader(
        (ROOT / "splits/BraTS21/scans_test.csv").open())]
    for key, name in NAMES.items():
        directory = ROOT / "outputs/runs" / name / "evaluation/brats21_test251"
        path = directory / "inference_report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        info = report["inference_settings"]
        if info["number_of_cases"] != 251 or Path(info["checkpoint_path"]).name != "epoch_0199.pt":
            raise ValueError(f"Expected epoch 199/test251: {path}")
        manifest = json.loads((directory / "cache/manifest.json").read_text(encoding="utf-8"))
        if not manifest["collection_complete"] or manifest["subjects"] != 251:
            raise ValueError(f"Incomplete inference cache: {path}")
        if [e["subject_id"] for e in manifest["entries"]] != roster:
            raise ValueError(f"Different case roster/order: {path}")
        reports[key] = report
    output = {"status": "COMPLETE" if len(reports) == 2 else "PENDING_RAW_TEST251",
              "subjects": 251, "checkpoint_epoch": 199, "models": {}}
    for key, report in reports.items():
        output["models"][key] = {"raw": report["raw_metrics"],
                                  "median_filter": report["median_filter_metrics"]}
    if len(reports) == 2:
        compared = ("aggregation_method", "anomaly_map_method", "threshold_method", "noise_steps",
                    "threshold_search_start", "threshold_search_end", "threshold_search_step")
        for field in compared:
            if reports["raw"]["inference_settings"][field] != reports["iqr"]["inference_settings"][field]:
                raise ValueError(f"Incompatible inference setting: {field}")
        for field in ("subjects", "AUPRC_mode", "AUPRC_samples", "AUPRC_seed",
                      "postprocess_mode", "normalization_scope"):
            if reports["iqr"]["evaluation_result"][field] != reports["raw"]["evaluation_result"][field]:
                raise ValueError(f"Incompatible scoring setting: {field}")
        output["numeric_deltas_raw_minus_iqr"] = {}
        for branch in ("raw", "median_filter"):
            a, b = (output["models"][k][branch] for k in ("iqr", "raw"))
            output["numeric_deltas_raw_minus_iqr"][branch] = {
                k: float(b[k] - a[k]) for k in a.keys() & b.keys()
                if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
                and not isinstance(a[k], bool) and not isinstance(b[k], bool)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, provenance, paths, configs = {}, {}, {}, {}
    for key, name in NAMES.items():
        run = ROOT / "outputs/runs" / name
        rows[key], provenance[key] = snapshot_metrics(run / "training_metrics.csv")
        configs[key] = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
        paths[key] = {int(p.stem.split("_")[1]) + 1: p for p in run.glob("epoch_*.pt")}
    common = sorted(paths["iqr"].keys() & paths["raw"].keys())
    comparisons = []
    for epoch in common:
        audits, architecture = {}, {}
        for key in NAMES:
            audits[key], architecture[key] = audit_checkpoint(paths[key][epoch], NAMES[key], rows[key][epoch])
        if architecture["iqr"] != architecture["raw"]:
            raise ValueError(f"Different architecture at epoch {epoch}")
        a, b = rows["iqr"][epoch], rows["raw"][epoch]
        if a["ema_step"] != b["ema_step"] or a["learning_rate"] != b["learning_rate"]:
            raise ValueError(f"Training progress/schedule differs at epoch {epoch}")
        comparisons.append({"completed_epoch": epoch, "iqr": a, "raw": b,
                            "raw_vs_iqr_percent": {k: 100 * (b[k] / a[k] - 1)
                                                   for k in ("train_loss", "validation_loss")},
                            "checkpoint_audits": audits})
    flattened = {key: flatten(config) for key, config in configs.items()}
    differences = {key: {k: flattened[k].get(key) for k in NAMES}
                   for key in flattened["iqr"].keys() | flattened["raw"].keys()
                   if flattened["iqr"].get(key) != flattened["raw"].get(key)}
    evaluation = compare_evaluation()
    if args.require_complete and (200 not in common or evaluation["status"] != "COMPLETE"):
        raise RuntimeError("Both final checkpoints and audited test251 reports are required")
    report = {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "metrics_sources": provenance, "config_differences": differences,
              "checkpoint_comparison": comparisons,
              "latest_common_logged_epoch": min(max(r) for r in rows.values()),
              "loss_interpretation": "Losses use each model's own source validation distribution and empirical "
                                     "noise spectrum. A lower loss does not establish better BraTS21 detection.",
              "test251": evaluation}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "comparison.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    last = report["latest_common_logged_epoch"]
    for axis, field, title in zip(axes, ("train_loss", "validation_loss"), ("Training loss", "EMA validation loss")):
        for key, color, label in (("iqr", "#2563eb", "Source IQR + matching"),
                                  ("raw", "#ea580c", "Raw source + matching")):
            epochs = sorted(e for e in rows[key] if e <= last)
            axis.plot(epochs, [rows[key][e][field] for e in epochs], color=color, label=label, linewidth=1.4)
            checkpoints = [e for e in common if e <= last]
            axis.scatter(checkpoints, [rows[key][e][field] for e in checkpoints], color=color, s=23, zorder=3)
        axis.set(title=title, xlabel="Completed epoch", ylabel="Loss (log scale)", yscale="log")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    fig.suptitle(f"Aligned training history through epoch {last}; dots = paired checkpoints", fontsize=12)
    fig.savefig(OUT / "loss_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"report": str(OUT / "comparison.json"), "paired_checkpoint_epochs": common,
                      "latest_checkpoint": comparisons[-1], "test251_status": evaluation["status"]}))


if __name__ == "__main__":
    main()
