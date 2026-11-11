"""Paired CPU comparison of mean/std versus the published median/IQR reference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from _bootstrap import bootstrap
bootstrap()

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize
from andi_rewrite.data.robust_normalization import robust_normalize_volume
from andi_rewrite.scripts.prepare_mixed_andi_histmatch import (
    BRATS, BRATS_IDS, MODALITIES, OUTPUT, QUANTILES, reference_ids,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/brats_train888_meanstd_vs_iqr"
SELECTED = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite summary")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95)),
            "min": float(values.min()), "max": float(values.max())}


def case_compare(case_id):
    torch.set_num_threads(1)
    folder = BRATS / case_id
    paths = [folder / f"{case_id}_{m}.nii.gz" for m in (*MODALITIES, "seg")]
    signature = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    raw = np.stack([np.asarray(nib.load(str(p)).dataobj, dtype=np.float32) for p in paths[:3]])
    seg = np.asarray(nib.load(str(paths[3])).dataobj, dtype=np.float32)
    if raw.shape[1:] != seg.shape or not np.isfinite(raw).all() or not np.isfinite(seg).all():
        raise ValueError(f"Invalid input: {case_id}")
    tensor = torch.from_numpy(raw)
    robust, robust_stats = robust_normalize_volume(tensor, return_statistics=True)
    zscore = np.full_like(raw, -1)
    rows = []
    for c, modality in enumerate(MODALITIES):
        foreground = raw[c] > 0
        tissue = raw[c][foreground].astype(np.float64)
        mean, std = float(tissue.mean()), float(tissue.std(ddof=0))
        if std <= 1e-8:
            raise ValueError(f"Degenerate std: {case_id} {modality}")
        zscore[c][foreground] = ((tissue - mean) / std).astype(np.float32)
        r = robust_stats[c]
        rows.append({"modality": modality, "foreground_voxels": int(tissue.size),
                     "raw_mean": mean, "raw_std": std, "raw_median": r["median"],
                     "raw_iqr": r["iqr"], "iqr_over_std": r["iqr"] / std,
                     "std_over_iqr": std / r["iqr"],
                     "zscore_offset_from_iqr": (r["median"] - mean) / std})
    resize = Resize(128, antialias=True)
    support = resize(torch.from_numpy((raw > 0).astype(np.float32)).permute(3, 0, 1, 2)).numpy() > 0
    lesion = resize(torch.from_numpy((seg > 0).astype(np.float32)).permute(2, 0, 1).unsqueeze(1)).numpy()[:, 0] > 0
    rv = resize(robust.permute(3, 0, 1, 2)).numpy()
    zv = resize(torch.from_numpy(zscore).permute(3, 0, 1, 2)).numpy()
    del raw, tensor, robust, zscore, seg
    curves = np.empty((2, 3, QUANTILES.size), dtype=np.float64)
    occupancy = []
    for c in range(3):
        valid = support[:, c] & ~lesion
        r, z = rv[:, c][valid], zv[:, c][valid]
        if r.size < 2 or not np.isfinite(r).all() or not np.isfinite(z).all():
            raise ValueError(f"Invalid healthy tissue: {case_id}")
        curves[0, c] = np.quantile(r, QUANTILES)
        curves[1, c] = np.quantile(z, QUANTILES)
        diff = z.astype(np.float64) - r.astype(np.float64)
        rows[c].update({"healthy_resized_voxels": int(r.size),
                       "paired_mae": float(np.abs(diff).mean()),
                       "paired_rmse": float(np.sqrt(np.mean(diff ** 2))),
                       "paired_bias_zscore_minus_iqr": float(diff.mean()),
                       "quantile_mae": float(np.abs(curves[1, c] - curves[0, c]).mean()),
                       "iqr_healthy_mean": float(r.mean(dtype=np.float64)),
                       "iqr_healthy_std": float(r.std(dtype=np.float64)),
                       "zscore_healthy_mean": float(z.mean(dtype=np.float64)),
                       "zscore_healthy_std": float(z.std(dtype=np.float64))})
        occupancy.append(float(valid.mean()))
    if [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths] != signature:
        raise ValueError(f"Input changed while reading: {case_id}")
    return {"case_id": case_id, "input_signature": signature,
            "channels": rows, "occupancy": occupancy}, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ids = reference_ids()
    if args.limit:
        ids = ids[:args.limit]
    out = REPORT if not args.limit else REPORT / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    refpath = OUTPUT / "reference.npz"
    refhash = sha(refpath)
    started = time.time()
    curves_sum = np.zeros((2, 3, QUANTILES.size), dtype=np.float64)
    records = []
    with (out / "per_case.jsonl").open("w", encoding="utf-8") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record, curves in pool.map(case_compare, ids, chunksize=1):
            records.append(record)
            curves_sum += curves / len(ids)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            n = len(records)
            if n % 25 == 0 or n == len(ids):
                elapsed = time.time() - started
                print(f"CASES {n}/{len(ids)} elapsed={elapsed:.1f}s ETA={(len(ids)-n)*elapsed/n:.1f}s", flush=True)
    with np.load(refpath) as ref:
        original = ref["quantiles"].copy()
        if not args.limit and list(ref["case_ids"]) != ids:
            raise ValueError("Reference IDs changed")
        previous_occupancy = ref["occupancy_median"].copy()
    reproduction = float(np.max(np.abs(curves_sum[0] - original)))
    occupancy = np.median(np.asarray([r["occupancy"] for r in records]), axis=0)
    if not args.limit and (reproduction > 1e-6 or not np.allclose(occupancy, previous_occupancy, atol=1e-10, rtol=0)):
        raise ValueError(f"Published IQR reference not reproduced: max error {reproduction}")
    if sha(refpath) != refhash:
        raise ValueError("Published reference changed")
    metrics = [k for k, v in records[0]["channels"][0].items() if isinstance(v, (int, float))]
    channels = {}
    for c, modality in enumerate(MODALITIES):
        rcurve, zcurve = curves_sum[:, c]
        delta = zcurve - rcurve
        central = (QUANTILES >= .01) & (QUANTILES <= .99)
        channels[modality] = {
            "case_statistics": {k: summarize([r["channels"][c][k] for r in records]) for k in metrics},
            "reference_quantile_mae_all": float(np.abs(delta).mean()),
            "reference_quantile_mae_p01_p99": float(np.abs(delta[central]).mean()),
            "reference_quantile_rmse_all": float(np.sqrt(np.mean(delta ** 2))),
            "reference_selected_quantiles": {f"p{p*100:g}": {
                "iqr": float(np.interp(p, QUANTILES, rcurve)),
                "mean_std": float(np.interp(p, QUANTILES, zcurve)),
                "mean_std_minus_iqr": float(np.interp(p, QUANTILES, delta))} for p in SELECTED},
            "reference_iqr_width_ratio_zscore_over_iqr": float(
                (np.interp(.75, QUANTILES, zcurve) - np.interp(.25, QUANTILES, zcurve)) /
                (np.interp(.75, QUANTILES, rcurve) - np.interp(.25, QUANTILES, rcurve))),
        }
    report = {"status": "PILOT" if args.limit else "PASS", "case_count": len(ids),
              "case_ids": ids, "modalities": list(MODALITIES), "roster_sha256": sha(BRATS_IDS),
              "published_iqr_reference": str(refpath), "published_iqr_reference_sha256": refhash,
              "iqr_reproduction_max_abs_error": reproduction,
              "method": {"scope": "per-case, per-modality full 3D positive foreground (including lesion for normalization statistics)",
                         "mean_std": "(x-mean)/population_std; float64 stats, float32 outputs",
                         "iqr": "existing robust_normalize_volume; torch float32 q25/median/q75",
                         "background": -1, "resize": "torchvision Resize(128, antialias=True)",
                         "evaluation_mask": "resized positive support AND NOT resized lesion coverage",
                         "reference": "equal-case mean of 4097 healthy-tissue quantiles",
                         "summary_weighting": "equal case; case-level MAE/RMSE then mean; not pooled voxels"},
              "elapsed_seconds": time.time()-started, "workers": args.workers,
              "occupancy_median": occupancy.tolist(), "channels": channels}
    np.savez_compressed(out / "reference_comparison.npz", probability=QUANTILES,
                        iqr_quantiles=curves_sum[0], mean_std_quantiles=curves_sum[1],
                        case_ids=np.asarray(ids), occupancy_median=occupancy)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    central = (QUANTILES >= .01) & (QUANTILES <= .99)
    for c, modality in enumerate(MODALITIES):
        axes[0, c].plot(QUANTILES[central]*100, curves_sum[0,c,central], label="Median / IQR")
        axes[0, c].plot(QUANTILES[central]*100, curves_sum[1,c,central], label="Mean / std")
        axes[0, c].set(title=modality.upper(), xlabel="Healthy-tissue percentile", ylabel="Normalized intensity")
        axes[0,c].legend()
        axes[1,c].plot(QUANTILES[central]*100, (curves_sum[1,c]-curves_sum[0,c])[central], color="tab:green")
        axes[1,c].axhline(0, color="gray", linewidth=.8)
        axes[1,c].set(xlabel="Healthy-tissue percentile", ylabel="Mean/std minus median/IQR")
        for ax in axes[:,c]:
            ax.grid(alpha=.25)
    fig.suptitle(f"BraTS train {len(ids)}: equal-case mean reference quantiles (P1-P99)")
    fig.savefig(out / "reference_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"status": report["status"], "case_count": len(ids), "elapsed_seconds": report["elapsed_seconds"],
                      "iqr_reproduction_max_abs_error": reproduction, "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
