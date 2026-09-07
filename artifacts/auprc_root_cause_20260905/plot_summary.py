"""Create a static research figure from audited local measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
audit = json.loads((OUT / "audit.json").read_text())
norm = json.loads((OUT / "normalization_audit.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
colors = ["#b36b32", "#286e98", "#799e70"]
ax = axes[0,0]
names = ["FOMO empirical", "BraTS empirical", "FOMO Gaussian"]
raw = [.44508451815979344, .834115646582013, .2710507669478494]
mf = [.47021173099001035, .8477582103543115, .4880518811699871]
x = np.arange(3)
for delta, values, label, color in [(-.18,raw,"Raw AP","#286e98"),(.18,mf,"Median-filtered AP","#799e70")]:
    bars = ax.bar(x+delta, values, .34, label=label, color=color)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
ax.set(xticks=x, xticklabels=names, ylim=(0,1), title="A. Same 251-subject test set; three trained models")
ax.legend(frameon=False, fontsize=9)
ax = axes[0,1]
for i, (short,name) in enumerate([("fomo","fomo45k_sri24_flair_t1_t2_empirical_spectrum233"),("brats","brats_flair_t1_t2_empirical_spectrum233")]):
    with (ROOT / "outputs/runs" / name / "training_metrics.csv").open() as f:
        rows=list(csv.DictReader(f))
    vals=np.array([float(r["validation_loss"]) for r in rows])
    smooth=np.convolve(vals,np.ones(15)/15,mode="valid")
    ax.plot(np.arange(15,234),smooth/np.min(smooth),label=short.upper(),color=colors[i])
ax.set(xlim=(60,233),ylim=(.95,1.65),xlabel="Completed epochs",ylabel="15-epoch mean / own minimum",title="B. Validation plateau (separate validation domains)")
ax.legend(frameon=False)
ax = axes[1,0]
for i, name in enumerate(["fomo","brats"]):
    vals=norm["matched_z"][name]["positive_mean"]
    bars=ax.bar(x+(i-.5)*.34,vals,.34,label=name.upper(),color=colors[i])
    ax.bar_label(bars,fmt="%.3f",fontsize=9,padding=3)
ax.set(xticks=x,xticklabels=["FLAIR","T1","T2"],ylim=(0,.8),ylabel="Mean positive intensity after p99",title="C. Matched z positions: 84 training slices per domain")
ax.legend(frameon=False)
ax = axes[1,1]
data=[[r["modalities"][m]["p99_ratio_all_over_seg0"] for r in norm["brats_training_p99"]] for m in ["flair","t1","t2"]]
for i, vals in enumerate(data):
    offsets=np.linspace(-.13,.13,len(vals))
    ax.scatter(i+offsets,vals,s=28,alpha=.75,color=colors[i])
    ax.plot([i-.2,i+.2],[np.median(vals)]*2,color="#273444",lw=2)
ax.axhline(1,color="#999999",ls="--",lw=1)
ax.set(xticks=x,xticklabels=["FLAIR","T1","T2"],ylabel="p99(all foreground) / p99(seg == 0)",title="D. Lesion influence on scaling: 12 training subjects")
fig.suptitle("AUPRC gap audit | measured evidence, not a causal ablation", fontsize=15, fontweight="bold")
fig.savefig(OUT / "evidence_summary.png",dpi=180)
fig.savefig(OUT / "evidence_summary.svg")
