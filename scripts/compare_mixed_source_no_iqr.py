"""Compare raw-source and robust-IQR ANDi matching with the frozen mean888 reference.

No dataset is overwritten or trained. Export FOMO inputs with export-fomo, run
compare for the three sources, then report. The export is separate because the
external FOMO files and existing MPI/OASIS files have different Windows owners.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import lmdb
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from torchvision.transforms import Resize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_mixed_andi_histmatch as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diagnostics/mixed_source_no_iqr_mean888"
BASE = original.OUTPUT
RESIZE = Resize(128, antialias=True)
GRID = np.linspace(0, 1, 101)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def token(case):
    return hashlib.sha256(case.encode()).hexdigest()[:20]


def raw_slices(paths, statistics, z):
    """Skip both median centering and IQR division; retain -1 before resize."""
    images = [nib.load(str(p)) for p in paths]
    if any(im.shape != images[0].shape or not np.allclose(im.affine, images[0].affine,
                                                       atol=1e-6, rtol=0) for im in images):
        raise ValueError("Raw modalities have different geometry")
    raw = torch.from_numpy(np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in images]))
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite raw MRI")
    mask = raw > 0
    robust = torch.full_like(raw, -1)
    for c, stats in enumerate(statistics):
        if int(mask[c].sum()) != stats["foreground_voxels"]:
            raise ValueError("Raw foreground count changed since robust-IQR build")
        robust[c][mask[c]] = (raw[c][mask[c]] - stats["median"]) / stats["iqr"]
    unnormalized = torch.where(mask, raw, -1.0)
    arrays = [RESIZE(v.permute(3, 0, 1, 2))[z].numpy() for v in (unnormalized, robust, mask.float())]
    return arrays[0], arrays[1], arrays[2] > 0


def export_fomo(out):
    report = json.loads((ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json").read_text())
    z_by_case = defaultdict(list)
    for split in original.SPLITS:
        for row in original.source_rows("fomo45k", split).values():
            z_by_case[row["case_id"]].append(int(row["z"]))
    cache = out / "fomo_raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    sessions = report["sessions"]
    for index, session in enumerate(sessions, 1):
        case = session["case_id"]
        target = cache / f"{token(case)}.npz"
        if target.exists():
            continue
        paths = [Path(session["input_paths"][m.upper()]) for m in original.MODALITIES]
        z = sorted(z_by_case[case])
        raw, robust, support = raw_slices(paths, session["statistics"], z)
        provenance = [{"path": str(p), "sha256": original.digest(p),
                       "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
        with target.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, raw=raw, robust=robust, support=support,
                                z=np.array(z), case_id=case,
                                provenance=json.dumps(provenance))
        target.with_suffix(".tmp").replace(target)
        if index % 5 == 0 or index == len(sessions):
            print(f"EXPORT_FOMO {index}/{len(sessions)}", flush=True)
    dump(out / "fomo_export_report.json", {"status": "PASS", "cases": len(sessions),
         "source_build_report_sha256": original.digest(ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json")})


def read_case(reader, rows):
    with reader.begin() as txn:
        return np.stack([pickle.loads(txn.get(r["key"].encode("ascii"))) for r in rows])


def montage(path, dataset, split, case, z, baseline, variant, support, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = int(np.argmax(support.sum(axis=(1, 2, 3))))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for c, mod in enumerate(original.MODALITIES):
        lo, hi = np.interp([.01, .99], original.QUANTILES, curves[c])
        a, b = baseline[index, c], variant[index, c]
        delta = np.where(support[index, c], b-a, 0)
        limit = max(float(np.quantile(np.abs(delta[support[index, c]]), .99)), .01)
        axes[c, 0].imshow(a, cmap="gray", vmin=lo, vmax=hi)
        axes[c, 1].imshow(b, cmap="gray", vmin=lo, vmax=hi)
        im = axes[c, 2].imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=axes[c, 2], fraction=.04, pad=.02)
        for j, title in enumerate(("IQR + matching", "Raw + matching", "Raw - IQR output")):
            axes[c, j].set_title(f"{mod.upper()} | {title}", fontsize=10)
            axes[c, j].axis("off")
    fig.suptitle(f"{dataset} / {split} / {case} / z={z[index]}\n"
                "Paired outputs share reference p01-p99 display window; difference uses its own scale", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), baseline=baseline[index],
                        no_iqr=variant[index], support=support[index], z=z[index])


def compare(out, sources):
    groups, manifest_hashes = original.case_groups()
    ref_path = BASE / "reference.npz"
    build = json.loads((BASE / "build_report.json").read_text())
    if original.digest(ref_path) != build["reference_sha256"]:
        raise ValueError("Frozen reference hash changed")
    if manifest_hashes != build["input_signature"]["source_manifests_sha256"]:
        raise ValueError("Original mixed manifests changed")
    for split in original.SPLITS:
        if original.digest(original.manifest_path(BASE, split)) != manifest_hashes[split]:
            raise ValueError("Matched baseline manifests changed")
    with np.load(ref_path) as ref:
        curves, ratios = ref["quantiles"], ref["background_ratio"]
    references = original.synthetic_references(curves, ratios)
    target = np.array([np.interp(GRID, original.QUANTILES, c) for c in curves])
    filt = sitk.HistogramMatchingImageFilter()
    contract = {"reference_sha256": original.digest(ref_path), "source_manifest_sha256": manifest_hashes,
                "source_transform": "foreground raw MRI unchanged; background -1 BEFORE same antialias resize",
                "matching": {"histogram_levels": filt.GetNumberOfHistogramLevels(),
                             "match_points": filt.GetNumberOfMatchPoints(),
                             "threshold_at_mean_intensity": filt.GetThresholdAtMeanIntensity()},
                "SimpleITK": sitk.Version_VersionString(), "scope": "all original mixed cases and slices",
                "comparison_mask": "same resized raw>0 support for both paired outputs",
                "equal_case_weight": True, "train_val_resplit": False}
    dump(out / "contract.json", contract)
    stats = {ds: {s["case_id"]: s for s in json.loads((ROOT / f"outputs/datasets/{ds}_sri24_robust_iqr/build_report.json").read_text())["sessions"]}
             for ds in sources}
    buckets = defaultdict(list)
    for split, ds, case, rows in groups:
        buckets[(split, ds)].append(case)
    rng = np.random.default_rng(73)
    fixed = {(split, ds): str(rng.choice(sorted(cases))) for (split, ds), cases in sorted(buckets.items())}
    readers = {split: lmdb.open(str(original.mixed_path(original.SOURCE, split)), readonly=True,
                               lock=False, readahead=False) for split in original.SPLITS}
    baselines = {split: lmdb.open(str(original.mixed_path(BASE, split)), readonly=True,
                                 lock=False, readahead=False) for split in original.SPLITS}
    try:
        for ds in sources:
            shard = out / f"case_results_{ds}.jsonl"
            if shard.exists():
                raise FileExistsError(f"Existing comparison shard: {shard}")
            selected = [g for g in groups if g[1] == ds]
            worst = {split: (-1.0, None) for split in original.SPLITS}
            with shard.open("w", encoding="utf-8") as handle:
                for index, (split, _, case, rows) in enumerate(selected, 1):
                    z = [r["z"] for r in rows]
                    if ds == "fomo45k":
                        cache = out / "fomo_raw_cache" / f"{token(case)}.npz"
                        with np.load(cache) as item:
                            if str(item["case_id"]) != case or list(item["z"]) != z:
                                raise ValueError("FOMO raw cache case/z mismatch")
                            raw, reconstructed, support = item["raw"], item["robust"], item["support"]
                            provenance = json.loads(str(item["provenance"]))
                    else:
                        paths = original.case_paths(ds, case)
                        paths = [paths[m] for m in original.MODALITIES]
                        raw, reconstructed, support = raw_slices(paths, stats[ds][case]["statistics"], z)
                        provenance = [{"path": str(p), "sha256": original.digest(p),
                                       "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
                    stored = read_case(readers[split], rows)
                    reconstruction_error = float(np.max(np.abs(stored-reconstructed)))
                    if reconstruction_error > 2e-5:
                        raise ValueError(f"Raw source no longer reproduces stored IQR: {ds} {case} {reconstruction_error}")
                    baseline = read_case(baselines[split], rows)
                    variant = np.empty_like(raw)
                    replay_error = None
                    for c in range(3):
                        variant[:, c] = original.match_volume(raw[:, c], support[:, c], references[c])
                    if case == fixed[(split, ds)]:
                        replay = np.stack([original.match_volume(stored[:, c], support[:, c], references[c]) for c in range(3)], axis=1)
                        replay_error = float(np.max(np.abs(replay-baseline)))
                        if replay_error > 2e-5:
                            raise ValueError("Same-parameter baseline replay differs")
                    if not np.all(baseline[~support] == -1) or not np.all(variant[~support] == -1):
                        raise ValueError("Paired background contract changed")
                    metrics = []
                    for c, mod in enumerate(original.MODALITIES):
                        a, b = baseline[:, c][support[:, c]], variant[:, c][support[:, c]]
                        pre = raw[:, c][support[:, c]]
                        delta = b.astype(np.float64)-a
                        qa, qb = np.quantile(a, GRID), np.quantile(b, GRID)
                        metrics.append({"modality": mod, "foreground_voxels": len(a),
                                        "paired_mae": float(np.mean(np.abs(delta))),
                                        "paired_rmse": float(np.sqrt(np.mean(delta**2))),
                                        "paired_absdiff_p95": float(np.quantile(np.abs(delta), .95)),
                                        "correlation": float(np.corrcoef(a, b)[0, 1]),
                                        "baseline_reference_quantile_mae": float(np.mean(np.abs(qa-target[c]))),
                                        "no_iqr_reference_quantile_mae": float(np.mean(np.abs(qb-target[c]))),
                                        "raw_input_p01_p50_p99": np.quantile(pre, [.01, .5, .99]).tolist(),
                                        "baseline_p01_p50_p99": np.quantile(a, [.01, .5, .99]).tolist(),
                                        "no_iqr_p01_p50_p99": np.quantile(b, [.01, .5, .99]).tolist(),
                                        "baseline_curve": qa.tolist(), "no_iqr_curve": qb.tolist()})
                    row = {"source_dataset": ds, "split": split, "case_id": case, "entries": len(rows),
                           "participant_id": rows[0]["participant_id"], "z": z,
                           "mixed_keys": [r["key"] for r in rows], "raw_files": provenance,
                           "raw_reconstruction_max_abs_error": reconstruction_error,
                           "baseline_replay_max_abs_error": replay_error, "metrics": metrics}
                    handle.write(json.dumps(row, allow_nan=False)+"\n"); handle.flush()
                    score = float(np.mean([m["paired_mae"] for m in metrics]))
                    if case == fixed[(split, ds)]:
                        montage(out / "images" / f"{split}_{ds}_fixed.png", ds, split, case, z,
                                baseline, variant, support, curves)
                    if score > worst[split][0]:
                        worst[split] = (score, case)
                        montage(out / "images" / f"{split}_{ds}_largest_difference.png", ds, split, case,
                                z, baseline, variant, support, curves)
                    if index % 5 == 0 or index == len(selected):
                        print(f"COMPARE {ds} {index}/{len(selected)} raw_reconstruction={reconstruction_error:.3g}", flush=True)
            dump(out / f"completion_{ds}.json", {"status": "PASS", "cases": len(selected),
                 "fixed_examples": {split: fixed[(split, ds)] for split in original.SPLITS},
                 "largest_difference_examples": {s: {"mae": v[0], "case_id": v[1]} for s, v in worst.items()}})
    finally:
        for env in [*readers.values(), *baselines.values()]:
            env.close()


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps, ImageDraw
    rows = []
    for ds in original.SOURCES:
        completion = json.loads((out / f"completion_{ds}.json").read_text())
        if completion["status"] != "PASS":
            raise ValueError("Incomplete source comparison")
        rows += [json.loads(line) for line in (out / f"case_results_{ds}.jsonl").read_text().splitlines()]
    counts = {s: sum(r["entries"] for r in rows if r["split"] == s) for s in original.SPLITS}
    if counts != original.EXPECTED or len(rows) != 669:
        raise ValueError(f"Comparison coverage mismatch: {counts}, {len(rows)}")
    for split in original.SPLITS:
        keys = [k for r in rows if r["split"] == split for k in r["mixed_keys"]]
        if sorted(keys) != [f"{i:08d}" for i in range(counts[split])]:
            raise ValueError("Comparison does not cover original keys exactly once")
    participants = {s: {r["participant_id"] for r in rows if r["split"] == s} for s in original.SPLITS}
    if participants["train"] & participants["val"]:
        raise ValueError("Participant leakage")
    summaries = []
    scalar_names = ["paired_mae", "paired_rmse", "paired_absdiff_p95", "correlation",
                    "baseline_reference_quantile_mae", "no_iqr_reference_quantile_mae"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    with np.load(BASE / "reference.npz") as ref:
        targets = np.array([np.interp(GRID, original.QUANTILES, c) for c in ref["quantiles"]])
    for si, split in enumerate(original.SPLITS):
        for c, mod in enumerate(original.MODALITIES):
            ax = axes[si, c]
            for ds in [*original.SOURCES, "all"]:
                chosen = [r for r in rows if r["split"] == split and (ds == "all" or r["source_dataset"] == ds)]
                metrics = [r["metrics"][c] for r in chosen]
                result = {"split": split, "source_dataset": ds, "modality": mod,
                          "cases": len(chosen), "slices": sum(r["entries"] for r in chosen)}
                result.update({name: float(np.mean([m[name] for m in metrics])) for name in scalar_names})
                result["cases_reference_error_improved"] = sum(m["no_iqr_reference_quantile_mae"] < m["baseline_reference_quantile_mae"] for m in metrics)
                for prefix in ["baseline", "no_iqr"]:
                    q = np.mean([m[f"{prefix}_p01_p50_p99"] for m in metrics], axis=0)
                    result.update({f"{prefix}_{name}": float(v) for name, v in zip(["p01", "p50", "p99"], q)})
                summaries.append(result)
                if ds == "all":
                    ax.plot(GRID, np.mean([m["baseline_curve"] for m in metrics], axis=0), label="IQR + matching", color="#4063a6")
                    ax.plot(GRID, np.mean([m["no_iqr_curve"] for m in metrics], axis=0), label="Raw + matching", color="#e2762a")
            ax.plot(GRID, targets[c], label="Frozen BraTS reference", color="#252525", linestyle="--")
            ax.set_title(f"{split} / {mod.upper()} | equal weight per case")
            ax.set_xlabel("Foreground quantile"); ax.set_ylabel("Matched intensity")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    baseline_report = json.loads((BASE / "build_report.json").read_text())
    diagnostic_error = max(abs(s["baseline_reference_quantile_mae"]-
                               baseline_report["mean_case_quantile_mae"][s["split"]][s["modality"]]["after"])
                           for s in summaries if s["source_dataset"] == "all")
    if diagnostic_error > 5e-7:
        raise ValueError(f"Baseline quantile metrics differ from original build: {diagnostic_error}")
    fig.tight_layout(); fig.savefig(out / "quantile_comparison.png", dpi=150); plt.close(fig)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    case_scalars = [{"source_dataset": r["source_dataset"], "split": r["split"], "case_id": r["case_id"],
                     "slices": r["entries"], "modality": m["modality"], **{n: m[n] for n in scalar_names}}
                    for r in rows for m in r["metrics"]]
    with (out / "case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_scalars[0])); writer.writeheader(); writer.writerows(case_scalars)
    # Six randomly fixed examples, not selected by the direction of the result.
    canvas = Image.new("RGB", (1500, 2250), "white")
    draw = ImageDraw.Draw(canvas)
    for si, split in enumerate(original.SPLITS):
        for di, ds in enumerate(original.SOURCES):
            with Image.open(out / "images" / f"{split}_{ds}_fixed.png") as im:
                thumb = ImageOps.contain(im.convert("RGB"), (750, 750))
                x, y = si*750, di*750
                canvas.paste(thumb, (x, y)); draw.text((x+8, y+8), f"{split} {ds}", fill="black")
    canvas.save(out / "fixed_examples_contact_sheet.png")
    report = {"status": "PASS", "cases": len(rows), "entries": counts,
              "reference_sha256": original.digest(BASE / "reference.npz"),
              "max_raw_reconstruction_abs_error": max(r["raw_reconstruction_max_abs_error"] for r in rows),
              "max_baseline_replay_abs_error": max(r["baseline_replay_max_abs_error"] for r in rows if r["baseline_replay_max_abs_error"] is not None),
              "background_preserved": True, "participant_overlap": 0,
              "baseline_build_report_quantile_mae_max_abs_error": diagnostic_error,
              "summary": summaries, "interpretation": "Intensity/image comparison only; no performance claim or model training"}
    affine_path = out / "affine_control_mpi.json"
    if affine_path.exists():
        report["single_case_affine_diagnostic"] = json.loads(affine_path.read_text())
    dump(out / "comparison_report.json", report)
    lines = ["# Source no-IQR comparison", "", "All 669 original cases; 84,341 train and 9,569 val slices.",
             "The source median/IQR transform is omitted completely. Raw positive foreground intensities are retained; background is -1 before the unchanged resize.",
             "The frozen mean888 reference, split, z indices, keys, case grouping and SimpleITK default matching parameters are unchanged.",
             "Raw sources reproduce the original robust-IQR LMDB within 2e-5. Six fixed seed-73 examples replay the baseline matching within 2e-5.",
             "Metrics use the SAME resized raw-positive support. Quantile MAE uses 101 points including min/max and equal case weights; it measures distribution distance, not model quality.",
             "", "| Split | Channel | IQR reference MAE | Raw reference MAE | Paired output MAE |", "|---|---|---:|---:|---:|"]
    for s in summaries:
        if s["source_dataset"] == "all":
            lines.append(f"| {s['split']} | {s['modality']} | {s['baseline_reference_quantile_mae']:.4f} | {s['no_iqr_reference_quantile_mae']:.4f} | {s['paired_mae']:.4f} |")
    lines += ["", "![Quantile curves](quantile_comparison.png)", "", "![Fixed examples](fixed_examples_contact_sheet.png)",
              "", "Images share the reference p01-p99 window within each modality. Residual heatmaps have their own symmetric scales. Largest-difference examples are explicitly labelled and kept separately in images/.",
              "", "The single-case affine diagnostic shows that a global positive affine transform INCLUDING background is almost cancelled by this matching operation. The actual no-IQR branch retains background -1 and resizes foreground/background mixtures, so it is not that global affine transform. The mean-threshold histogram population also changes. See affine_control_mpi.json for measured values.",
              "", "Reproduce: export-fomo (read access to external FOMO MRI files), compare --sources mpi oasis3, compare --sources fomo45k, then report, using scripts/compare_mixed_source_no_iqr.py in the ANDi environment. The export cache is temporary input material, not a published training dataset. Existing datasets are read-only."]
    (out / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(out), "entries": counts,
                      "all_summary": [s for s in summaries if s["source_dataset"] == "all"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export-fomo", "compare", "report"])
    parser.add_argument("--sources", nargs="+", choices=original.SOURCES, default=list(original.SOURCES))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Comparison outputs must remain inside workspace outputs")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    {"export-fomo": lambda: export_fomo(out), "compare": lambda: compare(out, args.sources),
     "report": lambda: report(out)}[args.command]()
