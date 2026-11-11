"""Audit actual pre-matching tissue selection without modifying datasets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

import compare_mixed_source_no_iqr as comparison

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis/matching_mechanism"


def arrays(case, statistics):
    paths = [comparison.original.case_paths("mpi", case["case_id"])[m]
             for m in comparison.original.MODALITIES]
    raw = torch.from_numpy(np.stack([
        np.asarray(nib.load(str(p)).dataobj, dtype=np.float32) for p in paths
    ]))
    support = raw > 0
    robust = torch.full_like(raw, -1)
    for c, st in enumerate(statistics):
        if int(support[c].sum()) != st["foreground_voxels"]:
            raise ValueError(f"Source foreground changed: {case['case_id']}")
        robust[c][support[c]] = (raw[c][support[c]] - st["median"]) / st["iqr"]
    raw = torch.where(support, raw, -1.0)
    z = case["z"]
    result = [comparison.RESIZE(v.permute(3, 0, 1, 2))[z].numpy()
              for v in (raw, robust, support.float())]
    return result


def fraction(values):
    return float(np.mean(values)) if values.size else None


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    stats = {r["case_id"]: r["statistics"] for r in json.loads(
        (ROOT / "outputs/datasets/mpi_sri24_robust_iqr/build_report.json").read_text())["sessions"]}
    old_rows = list(csv.DictReader((DIAG / "thresholds.csv").open(newline="", encoding="utf-8")))
    old = {(r["case"], r["modality"], r["source"]): r for r in old_rows}
    ids = {r["case"] for r in old_rows}
    cases = [json.loads(line) for line in
             (comparison.OUT / "case_results_mpi.jsonl").read_text().splitlines()]
    cases = [r for r in cases if r["split"] == "train" and r["case_id"] in ids]
    if len(cases) != 103 or {r["case_id"] for r in cases} != ids:
        raise ValueError("Expected the frozen 103 MPI training cases")
    rows, pairs = [], []
    started = time.monotonic()
    total_slices = 0
    reproduction_max_error = 0.0
    for index, case in enumerate(cases, 1):
        raw, robust, mask_fraction = arrays(case, stats[case["case_id"]])
        total_slices += len(case["z"])
        for c, modality in enumerate(comparison.original.MODALITIES):
            mask = mask_fraction[:, c] > 0
            core = mask_fraction[:, c] >= (1.0 - 1e-6)
            boundary = mask & ~core
            nfg = int(mask.sum())
            gates = {}
            for name, source in (("raw", raw[:, c]), ("robust", robust[:, c])):
                # SimpleITK receives float64 pixels and uses >= threshold.
                double = source.astype(np.float64)
                mean = float(double.mean())
                gate = double >= mean
                gates[name] = gate
                mean_fg = float(source[mask].mean(dtype=np.float64))
                # Change ONLY pure background pixels after resize to isolate
                # the threshold effect; all foreground/boundary values stay fixed.
                mean_zero_bg = mean + float((~mask).mean())
                zero_gate = double >= mean_zero_bg
                fg_mean_gate = double >= mean_fg
                eq_float32 = int(np.count_nonzero(mask & (source == source.mean())))
                eq_float64 = int(np.count_nonzero(mask & (double == mean)))
                old_rate = float(old[(case["case_id"], modality, name)]["foreground_above_full_mean"])
                error = abs(float((source[mask] > source.mean()).mean()) - old_rate)
                reproduction_max_error = max(reproduction_max_error, error)
                if error > 1e-12:
                    raise ValueError("Rebuilt pre-matching input differs from prior diagnostic")
                row = dict(case=case["case_id"], modality=modality, source=name,
                           slices=len(case["z"]), total_voxels=source.size,
                           foreground_voxels=nfg, core_voxels=int(core.sum()),
                           boundary_voxels=int(boundary.sum()),
                           full_mean=mean, foreground_mean=mean_fg,
                           zero_background_mean=mean_zero_bg,
                           foreground_selected=int(np.count_nonzero(mask & gate)),
                           foreground_excluded=int(np.count_nonzero(mask & ~gate)),
                           foreground_selected_fraction=fraction(gate[mask]),
                           core_selected_fraction=fraction(gate[core]),
                           boundary_selected_fraction=fraction(gate[boundary]),
                           background_selected=int(np.count_nonzero(~mask & gate)),
                           zero_background_foreground_selected=int(np.count_nonzero(mask & zero_gate)),
                           zero_background_changed_foreground=int(np.count_nonzero(mask & (gate != zero_gate))),
                           foreground_mean_selected=int(np.count_nonzero(mask & fg_mean_gate)),
                           foreground_mean_changed_foreground=int(np.count_nonzero(mask & (gate != fg_mean_gate))),
                           equality_at_float32_mean=eq_float32,
                           equality_at_float64_mean=eq_float64)
                rows.append(row)
            xor = gates["raw"] != gates["robust"]
            pair = dict(case=case["case_id"], modality=modality,
                        foreground_voxels=nfg, core_voxels=int(core.sum()),
                        boundary_voxels=int(boundary.sum()),
                        changed_foreground=int(np.count_nonzero(mask & xor)),
                        changed_core=int(np.count_nonzero(core & xor)),
                        changed_boundary=int(np.count_nonzero(boundary & xor)),
                        raw_only=int(np.count_nonzero(mask & gates["raw"] & ~gates["robust"])),
                        robust_only=int(np.count_nonzero(mask & gates["robust"] & ~gates["raw"])),
                        changed_foreground_fraction=fraction(xor[mask]),
                        changed_core_fraction=fraction(xor[core]))
            pairs.append(pair)
        if index % 10 == 0 or index == len(cases):
            print(f"CASES {index}/{len(cases)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    for filename, items in (("per_case_selection.csv", rows), ("per_case_pairs.csv", pairs)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    summary = []
    for modality in comparison.original.MODALITIES:
        paired = [r for r in pairs if r["modality"] == modality]
        nfg = sum(r["foreground_voxels"] for r in paired)
        ncore = sum(r["core_voxels"] for r in paired)
        for name in ("raw", "robust"):
            selected = [r for r in rows if r["modality"] == modality and r["source"] == name]
            summary.append(dict(
                modality=modality, source=name, cases=len(selected), foreground_voxels=nfg,
                selected_voxels=sum(r["foreground_selected"] for r in selected),
                excluded_voxels=sum(r["foreground_excluded"] for r in selected),
                case_mean_selected_fraction=float(np.mean([r["foreground_selected_fraction"] for r in selected])),
                voxel_weighted_selected_fraction=sum(r["foreground_selected"] for r in selected)/nfg,
                background_selected=sum(r["background_selected"] for r in selected),
                zero_background_selected=sum(
                    r["total_voxels"] - r["foreground_voxels"]
                    for r in selected if r["zero_background_mean"] <= 0),
                zero_background_selected_cases=sum(r["zero_background_mean"] <= 0 for r in selected),
                zero_background_changed_fraction=sum(r["zero_background_changed_foreground"] for r in selected)/nfg,
                foreground_mean_changed_fraction=sum(r["foreground_mean_changed_foreground"] for r in selected)/nfg,
                pair_changed_fraction=sum(r["changed_foreground"] for r in paired)/nfg,
                pair_changed_core_fraction=sum(r["changed_core"] for r in paired)/ncore,
                pair_raw_only=sum(r["raw_only"] for r in paired),
                pair_robust_only=sum(r["robust_only"] for r in paired),
                cases_with_pair_change=sum(r["changed_foreground"] > 0 for r in paired)))
    report = dict(status="PASS", cases=len(cases), slices=total_slices,
                  scope="All slices used for per-case matching in all 103 MPI training cases; not just the 1600 classifier sample slices",
                  threshold_rule="source >= mean(source), float64 statistics",
                  foreground_rule="resize(raw > 0, antialias=True) > 0",
                  core_rule="resized support fraction >= 1 - 1e-6 (no material background mixing)",
                  zero_background_control="Change only pure background from -1 to 0 after resize; preserve all foreground intensities",
                  foreground_mean_control="Use foreground-only mean as threshold; NOT the mask-only/all-foreground strategy",
                  mask_only_selected_fraction=1.0,
                  old_threshold_diagnostic_reproduction_max_error=reproduction_max_error,
                  elapsed_seconds=time.monotonic()-started, summary=summary)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
