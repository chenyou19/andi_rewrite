"""Read-only source-data audit; writes only analysis artifacts next to this file.

Run with the existing ANDi Python environment. No training or GPU inference.
"""
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
torch.set_num_threads(2)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


FOMO = Path("C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH")
BRA = Path("C:/ML/data/BraTS_2021_healthy_lmdb")
BV = Path("C:/ML/data/BraTS_2021_healthy_lmdb_val")
frows = read_csv(FOMO / "manifests/train_entries.csv")
brows = read_csv(ROOT / "data/BraTS21/healthy_slices_train.csv")
vrows = read_csv(BV / "healthy_slices.csv")
trows = read_csv(ROOT / "splits/BraTS21/scans_test.csv")
result = {"seed": 73, "source_data_read_only": True}
sets = {
    "brats_train": {r["BraTS21ID"] for r in brows},
    "brats_val": {r["BraTS21ID"] for r in vrows},
    "brats_test": {r["BraTS21ID"] for r in trows},
}
result["split_audit"] = {
    "subjects": {k: len(v) for k, v in sets.items()},
    "train_test_overlap": len(sets["brats_train"] & sets["brats_test"]),
    "train_val_overlap": len(sets["brats_train"] & sets["brats_val"]),
    "val_test_overlap": len(sets["brats_val"] & sets["brats_test"]),
}
result["training"] = {}
for short, name in [("fomo", "fomo45k_sri24_flair_t1_t2_empirical_spectrum233"),
                    ("brats", "brats_flair_t1_t2_empirical_spectrum233")]:
    rows = read_csv(ROOT / "outputs/runs" / name / "training_metrics.csv")
    result["training"][short] = {
        "epochs": len(rows), "steps": int(rows[-1]["ema_step"]),
        "steps_per_epoch": int(rows[0]["ema_step"]),
        "final": rows[-1],
        "best_validation": min(rows, key=lambda r: float(r["validation_loss"])),
        "validation_last20": float(np.mean([float(r["validation_loss"]) for r in rows[-20:]])),
        "validation_previous20": float(np.mean([float(r["validation_loss"]) for r in rows[-40:-20]])),
    }

result["slice_statistics"] = {}
for name, path, rows, zkey, indices in [
    ("fomo", FOMO / "train", frows, "z", [0, 1, 2]),
    ("brats", BRA, brows, "Slice", [0, 1, 3]),
]:
    z = np.array([int(r[zkey]) for r in rows])
    rng = np.random.default_rng(73)
    chosen = rng.choice(len(rows), size=384, replace=False)
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        assert txn.stat()["entries"] == len(rows)
        images = np.stack([pickle.loads(txn.get(f"{i:08d}".encode()))[indices] for i in chosen])
    env.close()
    stats = []
    for c in range(3):
        channel = images[:, c]
        positive = channel[channel > 0]
        stats.append({
            "modality": ["FLAIR", "T1", "T2"][c],
            "all_pixel_mean": float(channel.mean()),
            "positive_fraction": float((channel > 0).mean()),
            "above_002_fraction": float((channel > .02).mean()),
            "positive_mean": float(positive.mean()),
            "positive_p10_p50_p90_p99": np.quantile(positive, [.1, .5, .9, .99]).tolist(),
            "above_one_fraction": float((channel > 1).mean()),
        })
    result["slice_statistics"][name] = {
        "n": len(rows), "sample_n": len(chosen), "sample_indices": chosen.tolist(),
        "z_histogram": np.bincount(z, minlength=155).tolist(),
        "z_50_100_fraction": float(((z >= 50) & (z < 100)).mean()),
        "statistics": stats,
        "limitation": "Random slices, not independent participants or matched tissue/z; exploratory only.",
    }
print("Completed split, learning-curve and LMDB sample audits.", flush=True)

from andi_rewrite.noise.empirical_spectrum import EmpiricalSpectrumNoise
samplers = []
result["spectra"] = {}
for name, path, indices in [
    ("fomo", FOMO / "spectrum/fomo45k_sri24_flair_t1_t2_empirical_spectrum.npz", None),
    ("brats", Path("C:/ML/data/spectrum/brats21_healthy_empirical_spectrum.npz"), [0, 1, 3]),
]:
    sampler = EmpiricalSpectrumNoise(path, mode="radial", generation_method="filtered_gaussian", channel_indices=indices)
    samplers.append(sampler)
    with np.load(path, allow_pickle=False) as data:
        result["spectra"][name] = {
            "path": str(path), "arrays": {k: list(data[k].shape) for k in data.files},
            "metadata": {k: data[k].tolist() for k in data.files if data[k].size <= 12},
            "loaded_key": sampler.loaded_statistic_key,
            "fallback": sampler.used_statistic_fallback,
        }
f = samplers[0].filter_amp_rfft.numpy()
b = samplers[1].filter_amp_rfft.numpy()
for name, a in [("fomo", f), ("brats", b)]:
    w = np.full(a.shape[-1], 2.0); w[[0, -1]] = 1
    power = a.astype(float) ** 2 * w
    yy = np.fft.fftfreq(128)[:, None]
    xx = np.fft.rfftfreq(128)[None, :]
    radius = np.sqrt(xx**2 + yy**2)
    result["spectra"][name]["power_fraction_radius_le_0_05"] = [float(p[radius <= .05].sum() / p.sum()) for p in power]
result["spectra"]["filter_cosine_similarity"] = [
    float(np.sum(a*c) / np.sqrt(np.sum(a*a)*np.sum(c*c))) for a,c in zip(f,b)
]
save("audit.json", result)
print("Completed effective noise-filter audit.", flush=True)

CACHE = ROOT / "outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/cache"
manifest = read_json(CACHE / "manifest.json")
assert [e["subject_id"] for e in manifest["entries"]] == [r["BraTS21ID"] for r in trows]
sample_indices = np.random.default_rng(73).integers(0, manifest["total_voxels"], size=5000000, dtype=np.int64)
sample_indices.sort()
y = np.empty(len(sample_indices), dtype=bool)
raw = np.empty(len(sample_indices), dtype=np.float32)
mf = np.empty(len(sample_indices), dtype=np.float32)
offset = 0
for e in manifest["entries"]:
    lo, hi = np.searchsorted(sample_indices, [offset, offset + e["numel"]])
    local = sample_indices[lo:hi] - offset
    y[lo:hi] = np.load(CACHE / e["label_file"], mmap_mode="r").reshape(-1)[local]
    raw[lo:hi] = np.load(CACHE / e["raw_file"], mmap_mode="r").reshape(-1)[local]
    mf[lo:hi] = np.load(CACHE / e["mf"]["file"], mmap_mode="r").reshape(-1)[local]
    offset += e["numel"]
assert offset == manifest["total_voxels"]
result["cached_ap_recalculation"] = {
    "n": len(y), "positives": int(y.sum()), "positive_fraction": float(y.mean()),
    "raw_AP": float(average_precision_score(y, raw)),
    "mf_AP": float(average_precision_score(y, mf)),
    "method": "Same global voxel sample (with replacement), seed 73; sklearn AP on saved scores; no inference.",
    "limitation": "Global affine score normalization preserves rank, barring floating-point ties; no BraTS score cache available at referenced output.",
}
save("audit.json", result)
print("Cached AP:", result["cached_ap_recalculation"], flush=True)

# Descriptive error localization, evenly spaced cases; never used for tuning.
import nibabel as nib
from scipy.ndimage import binary_erosion
from torchvision.transforms import Resize
case_stats = []
for index in np.linspace(0, len(manifest["entries"])-1, 12, dtype=int):
    e = manifest["entries"][index]
    image = np.asarray(nib.load(e["metadata"]["reference_path"]).dataobj, dtype=np.float32)
    brain = Resize(128, antialias=True)(torch.from_numpy((image > 0).transpose(2,0,1).copy()).float()).numpy().transpose(1,2,0) > .5
    lesion = np.load(CACHE / e["label_file"], mmap_mode="r")
    score = np.load(CACHE / e["raw_file"], mmap_mode="r")
    interior = binary_erosion(brain, iterations=2)
    regions = {"lesion": lesion, "normal_interior": interior & ~lesion,
               "normal_boundary": brain & ~interior & ~lesion, "outside_brain": ~brain & ~lesion}
    cutoff = .055 * (manifest["raw_score_bounds"]["max"] - manifest["raw_score_bounds"]["min"]) + manifest["raw_score_bounds"]["min"]
    item = {"subject_id": e["subject_id"], "regions": {}, "threshold": .055,
            "AP_all": float(average_precision_score(lesion.reshape(-1), score.reshape(-1))),
            "AP_brain": float(average_precision_score(lesion[brain], score[brain]))}
    for name, mask in regions.items():
        values = score[mask]
        item["regions"][name] = {"n": int(mask.sum()), "above_threshold": int((values > cutoff).sum()),
            "raw_score_p50_p90_p99": np.quantile(values, [.5,.9,.99]).tolist() if len(values) else []}
    case_stats.append(item)
    print("Localized errors:", e["subject_id"], flush=True)
result["error_localization"] = {"cases": case_stats,
    "limitation": "12 evenly spaced test subjects; FLAIR>0 resized brain-mask proxy; no mask dilation; descriptive, not a validation set.",
    "region_counts": {name: {key: sum(c["regions"][name][key] for c in case_stats) for key in ["n", "above_threshold"]} for name in regions}}
save("audit.json", result)
print("Audit complete:", OUT / "audit.json", flush=True)
