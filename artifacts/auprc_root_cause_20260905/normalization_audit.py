"""Focused CPU follow-up: matched-z statistics and lesion influence on p99.

Ground-truth exclusion is diagnostic only and MUST NOT become inference input.
"""
import csv
import json
import pickle
from pathlib import Path

import lmdb
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FOMO = Path("C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH")


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


result = {"seed": 73, "matched_z": {}, "brats_training_p99": []}
for name, path, records, zkey, channels in [
    ("fomo", FOMO / "train", rows(FOMO / "manifests/train_entries.csv"), "z", [0,1,2]),
    ("brats", Path("C:/ML/data/BraTS_2021_healthy_lmdb"), rows(ROOT / "data/BraTS21/healthy_slices_train.csv"), "Slice", [0,1,3]),
]:
    rng = np.random.default_rng(73)
    z = np.array([int(r[zkey]) for r in records])
    chosen = np.concatenate([rng.choice(np.flatnonzero(z == level), 12, replace=False) for level in [30,45,60,75,90,105,120]])
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        images = np.stack([pickle.loads(txn.get(f"{i:08d}".encode()))[channels] for i in chosen])
    env.close()
    result["matched_z"][name] = {
        "z_levels": [30,45,60,75,90,105,120], "n_per_level": 12, "sample_indices": chosen.tolist(),
        "positive_mean": [float(c[c > 0].mean()) for c in images.transpose(1,0,2,3)],
        "above_002_mean": [float(c[c > .02].mean()) for c in images.transpose(1,0,2,3)],
        "positive_fraction": [float((c > 0).mean()) for c in images.transpose(1,0,2,3)],
        "limitation": "Matched slice index only; not matched tissue, age, site, or participant.",
    }
train = rows(ROOT / "splits/BraTS21/scans_train.csv")
for i in np.linspace(0, len(train)-1, 12, dtype=int):
    subject = train[i]["BraTS21ID"]
    base = Path("C:/ML/data/BraTS_2021") / subject
    label = np.asarray(nib.load(base / f"{subject}_seg.nii.gz").dataobj) > 0
    item = {"subject_id": subject, "modalities": {}}
    for mod in ["flair","t1","t2"]:
        img = np.asarray(nib.load(base / f"{subject}_{mod}.nii.gz").dataobj, dtype=np.float32)
        pos = img > 0
        p_all = float(np.quantile(img[pos], .99))
        p_normal = float(np.quantile(img[pos & ~label], .99))
        item["modalities"][mod] = {"p99_all": p_all, "p99_seg0": p_normal,
            "p99_ratio_all_over_seg0": p_all/p_normal,
            "normalized_seg0_median_all": float(np.median(img[pos & ~label])/p_all),
            "normalized_seg0_median_excluding_lesion": float(np.median(img[pos & ~label])/p_normal)}
    result["brats_training_p99"].append(item)
    print("p99 diagnostic:", subject, flush=True)
result["p99_ratio_summary"] = {
    mod: {"median": float(np.median([r["modalities"][mod]["p99_ratio_all_over_seg0"] for r in result["brats_training_p99"]])),
          "min": min(r["modalities"][mod]["p99_ratio_all_over_seg0"] for r in result["brats_training_p99"]),
          "max": max(r["modalities"][mod]["p99_ratio_all_over_seg0"] for r in result["brats_training_p99"])}
    for mod in ["flair","t1","t2"]}
result["limitation"] = "12 evenly spaced training subjects; seg==0 is an unlabeled-tissue proxy, not verified healthy tissue. Diagnostic lesion exclusion is not a deployable normalization."
(OUT / "normalization_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(result["p99_ratio_summary"], flush=True)
