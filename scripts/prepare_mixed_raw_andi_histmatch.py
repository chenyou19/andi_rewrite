"""Publish the validated raw-source mean888 experiment, without replacing data."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import traceback

import lmdb
import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_mixed_source_no_iqr as comparison

original = comparison.original
ROOT = comparison.ROOT
OUTPUT = ROOT / "outputs/datasets/mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888"
REPORT = ROOT / "outputs/reports/mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh"
SPECTRUM_NAME = "mixed_train_raw_andi_histmatch_mean888_empirical_spectrum.npz"
SPEC = {"type": "andi_histmatch_mean888_raw_source", "version": 1,
        "source_normalization": "none_positive_raw_foreground",
        "reference": "equal_case_mean_BraTS_train888_healthy_tissue_quantiles",
        "operation": "SimpleITK.HistogramMatching_default_parameters_per_case_per_modality",
        "background": -1.0, "model_normalize_input": False}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def status(phase, **kwargs):
    comparison.dump(REPORT / "build_status.json", {"status": "RUNNING", "phase": phase,
                    "pid": os.getpid(), "updated_at": now(), **kwargs})


def load_inputs():
    groups, manifests = original.case_groups()
    contract = json.loads((comparison.OUT / "contract.json").read_text())
    ref_path = comparison.BASE / "reference.npz"
    if original.digest(ref_path) != contract["reference_sha256"] or manifests != contract["source_manifest_sha256"]:
        raise ValueError("Reference or source manifests changed since the accepted comparison")
    f = sitk.HistogramMatchingImageFilter()
    params = {"histogram_levels": f.GetNumberOfHistogramLevels(), "match_points": f.GetNumberOfMatchPoints(),
              "threshold_at_mean_intensity": f.GetThresholdAtMeanIntensity()}
    if params != contract["matching"] or sitk.Version_VersionString() != contract["SimpleITK"]:
        raise ValueError("SimpleITK version or matching defaults differ from the comparison")
    expected = {}
    signature = {"reference_sha256": original.digest(ref_path), "source_manifests_sha256": manifests,
                 "normalization": SPEC, "matching": params, "SimpleITK": sitk.Version_VersionString(),
                 "comparison_reports_sha256": {}, "source_build_reports_sha256": {}, "fomo_cache_sha256": {}}
    statistics = {}
    for ds in original.SOURCES:
        path = comparison.OUT / f"case_results_{ds}.jsonl"
        signature["comparison_reports_sha256"][ds] = original.digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            expected[(row["split"], ds, row["case_id"])] = row
            if ds == "fomo45k":
                cache = comparison.OUT / "fomo_raw_cache" / f"{comparison.token(row['case_id'])}.npz"
                signature["fomo_cache_sha256"][row["case_id"]] = original.digest(cache)
        source_report = ROOT / f"outputs/datasets/{ds}_sri24_robust_iqr/build_report.json"
        signature["source_build_reports_sha256"][ds] = original.digest(source_report)
        statistics[ds] = {r["case_id"]: r["statistics"] for r in json.loads(source_report.read_text())["sessions"]}
    if len(groups) != 669 or len(expected) != len(groups):
        raise ValueError("Expected all 669 original cases")
    return groups, manifests, signature, expected, statistics


def raw_case(ds, case, z, stats, expected):
    if ds == "fomo45k":
        cache = comparison.OUT / "fomo_raw_cache" / f"{comparison.token(case)}.npz"
        with np.load(cache) as item:
            if str(item["case_id"]) != case or list(item["z"]) != z:
                raise ValueError("FOMO cache case/z mismatch")
            provenance = json.loads(str(item["provenance"]))
            if provenance != expected["raw_files"]:
                raise ValueError("FOMO cache provenance changed")
            return item["raw"], item["robust"], item["support"]
    paths = original.case_paths(ds, case)
    paths = [paths[m] for m in original.MODALITIES]
    if [original.digest(p) for p in paths] != [p["sha256"] for p in expected["raw_files"]]:
        raise ValueError(f"Raw MRI changed since comparison: {ds}:{case}")
    return comparison.raw_slices(paths, stats, z)


def build():
    status("VERIFY_INPUTS")
    groups, manifests, signature, expected, statistics = load_inputs()
    if OUTPUT.exists():
        report = json.loads((OUTPUT / "build_report.json").read_text())
        if report["status"] != "PASS" or report["input_signature"] != signature:
            raise ValueError("Existing output does not match the frozen experiment")
        comparison.dump(REPORT / "build_status.json", {"status": "PASS", "phase": "PUBLISHED",
                        "pid": os.getpid(), "updated_at": now(), "output": str(OUTPUT), "reused": True})
        return
    stage = OUTPUT.with_name(OUTPUT.name + ".staging")
    stage.mkdir(parents=True, exist_ok=True)
    signature_path = stage / "input_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Staging input signature changed; inspect staging before retry")
    comparison.dump(signature_path, signature)
    # Allow training checkpoints, score cache and 5 GiB headroom after publication.
    payload_bytes = sum(original.EXPECTED.values()) * 3 * 128 * 128 * 4
    existing = sum(p.stat().st_size for p in stage.rglob("data.mdb"))
    required = max(0, int(payload_bytes*1.2)-existing) + 13*1024**3
    if shutil.disk_usage(stage).free < required:
        raise RuntimeError(f"Insufficient space: need {required/1024**3:.1f} GiB including training/evaluation reserve")
    with np.load(comparison.BASE / "reference.npz") as ref:
        curves, ratios = ref["quantiles"], ref["background_ratio"]
    references = original.synthetic_references(curves, ratios)
    target_curves = np.array([np.interp(comparison.GRID, original.QUANTILES, c) for c in curves])
    readers, writers = {}, {}
    metrics = []
    try:
        for split in original.SPLITS:
            readers[split] = lmdb.open(str(original.mixed_path(original.SOURCE, split)), readonly=True,
                                       lock=False, readahead=False)
            path = original.mixed_path(stage, split)
            path.mkdir(parents=True, exist_ok=True)
            size = int(original.EXPECTED[split]*3*128*128*4*1.3) + 1024**3
            writers[split] = lmdb.open(str(path), map_size=size)
            comparison.dump(path / "normalization.json", {**SPEC, "reference_sha256": signature["reference_sha256"]})
        for index, (split, ds, case, rows) in enumerate(groups, 1):
            prior = expected[(split, ds, case)]
            z, keys = [r["z"] for r in rows], [r["key"] for r in rows]
            if z != prior["z"] or keys != prior["mixed_keys"]:
                raise ValueError("Case membership or ordering changed")
            raw, reconstructed, support = raw_case(ds, case, z, statistics[ds][case], prior)
            stored = comparison.read_case(readers[split], rows)
            error = float(np.max(np.abs(stored-reconstructed)))
            if error > 2e-5:
                raise ValueError(f"Raw MRI does not reproduce source LMDB: {ds}:{case}")
            matched = np.empty_like(raw)
            case_metrics = []
            for c, mod in enumerate(original.MODALITIES):
                matched[:, c] = original.match_volume(raw[:, c], support[:, c], references[c])
                q = np.quantile(matched[:, c][support[:, c]], comparison.GRID)
                distance = float(np.mean(np.abs(q-target_curves[c])))
                comparison_error = float(np.max(np.abs(q-prior["metrics"][c]["no_iqr_curve"])))
                if comparison_error > 2e-5:
                    raise ValueError(f"Output changed from accepted comparison: {ds}:{case}:{mod}")
                case_metrics.append({"modality": mod, "reference_quantile_mae": distance,
                                     "comparison_curve_max_abs_error": comparison_error})
            writer = writers[split]
            encoded_keys = [k.encode("ascii") for k in keys]
            with writer.begin() as txn:
                found = [txn.get(k) is not None for k in encoded_keys]
                if all(found):
                    for k, value in zip(encoded_keys, matched):
                        if not np.array_equal(pickle.loads(txn.get(k)), value):
                            raise ValueError("Existing staging case differs from recomputed output")
            if any(found) and not all(found):
                raise ValueError("Partial case transaction in staging")
            if not all(found):
                with writer.begin(write=True) as txn:
                    for k, value in zip(encoded_keys, matched):
                        if not txn.put(k, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                            raise ValueError("Duplicate output key")
            # Read back every case, including its exact support/background.
            with writer.begin() as txn:
                readback = np.stack([pickle.loads(txn.get(k)) for k in encoded_keys])
            if not np.array_equal(readback, matched) or not np.all(readback[~support] == -1):
                raise ValueError("Output readback/background mismatch")
            metrics.append({"split": split, "source_dataset": ds, "case_id": case, "entries": len(rows),
                            "raw_reconstruction_max_abs_error": error, "metrics": case_metrics})
            if index % 5 == 0 or index == len(groups):
                status("BUILD_LMDB", completed_cases=index, total_cases=len(groups))
                print(f"BUILD {index}/{len(groups)} {split} {ds}", flush=True)
                if shutil.disk_usage(stage).free < 5*1024**3:
                    raise RuntimeError("Free space fell below 5 GiB")
        for writer in writers.values():
            writer.sync()
            used = (writer.info()["last_pgno"]+1)*writer.stat()["psize"]
            writer.set_mapsize(((used+2**30-1)//2**30+1)*2**30)
            writer.sync()
    finally:
        for env in [*writers.values(), *readers.values()]:
            env.close()
    (stage / "case_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in metrics))
    for split in original.SPLITS:
        shutil.copyfile(original.manifest_path(original.SOURCE, split), original.manifest_path(stage, split))
    shutil.copyfile(comparison.BASE / "reference.npz", stage / "reference.npz")
    shutil.copyfile(comparison.BASE / "reference_report.json", stage / "reference_report.json")
    status("FULL_READBACK")
    counts, content_hashes = {}, {}
    import hashlib
    for split in original.SPLITS:
        h = hashlib.sha256()
        with lmdb.open(str(original.mixed_path(stage, split)), readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                if txn.stat()["entries"] != original.EXPECTED[split]:
                    raise ValueError("LMDB entry count mismatch")
                for i in range(original.EXPECTED[split]):
                    value = txn.get(f"{i:08d}".encode("ascii"))
                    array = pickle.loads(value)
                    if array.shape != (3, 128, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError(f"Invalid LMDB entry: {split}:{i}")
                    h.update(value)
        if original.digest(original.manifest_path(stage, split)) != manifests[split]:
            raise ValueError("Output manifest changed")
        counts[split] = original.EXPECTED[split]
        content_hashes[split] = h.hexdigest()
    status("COMPUTE_SPECTRUM")
    spectrum = stage / "spectrum" / SPECTRUM_NAME
    if not spectrum.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/compute_lmdb_spectrum.py"),
                        "--lmdb-path", str(stage / "train"), "--out", str(spectrum),
                        "--mask-mode", "robust_iqr_background", "--window", "hann", "--crop-margin", "4",
                        "--radial-bins", "64", "--channel-order", "FLAIR", "T1", "T2",
                        "--source-manifest", str(stage / "source_entries.jsonl"), "--no-progress"], check=True)
    sidecar = spectrum.with_suffix(spectrum.suffix + ".metadata.json")
    metadata = json.loads(sidecar.read_text())
    if metadata["source_serialized_values_sha256"] != content_hashes["train"]:
        raise ValueError("Spectrum was not computed from this train LMDB")
    if metadata["num_slices_used"]+metadata["num_slices_skipped"] != counts["train"]:
        raise ValueError("Spectrum did not inspect every train slice")
    with np.load(spectrum) as values:
        for name in ("mean_amplitude", "radial_amplitude", "mean_power", "radial_power"):
            if not np.isfinite(values[name]).all():
                raise ValueError("Nonfinite spectrum")
    metadata.update(source_lmdb=str(OUTPUT / "train"), source_manifest=str(OUTPUT / "source_entries.jsonl"),
                    output_npz=str(OUTPUT / "spectrum" / SPECTRUM_NAME), staging_source_lmdb=str(stage / "train"))
    comparison.dump(sidecar, metadata)
    report = {"status": "PASS", "completed_at": now(), "case_count": len(groups), "entries": counts,
              "shape": [3, 128, 128], "dtype": "float32", "channel_order": ["FLAIR", "T1", "T2"],
              "input_signature": signature, "source_serialized_values_sha256": content_hashes,
              "reference_sha256": signature["reference_sha256"], "spectrum_sha256": original.digest(spectrum),
              "spectrum_num_slices_used": metadata["num_slices_used"],
              "spectrum_num_slices_skipped": metadata["num_slices_skipped"],
              "max_raw_reconstruction_abs_error": max(r["raw_reconstruction_max_abs_error"] for r in metrics),
              "max_comparison_curve_abs_error": max(m["comparison_curve_max_abs_error"] for r in metrics for m in r["metrics"]),
              "participant_overlap": 0, "verification": "every case exact output readback, support/background, full contiguous-key readback, manifests, comparison quantiles, train spectrum content hash"}
    comparison.dump(stage / "build_report.json", report)
    stage.rename(OUTPUT)
    comparison.dump(REPORT / "build_status.json", {"status": "PASS", "phase": "PUBLISHED", "pid": os.getpid(),
                    "updated_at": now(), "output": str(OUTPUT), "entries": counts})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    import msvcrt
    torch.set_num_threads(4)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "build.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            build()
        except Exception as exc:
            comparison.dump(REPORT / "build_status.json", {"status": "FAILED", "pid": os.getpid(), "updated_at": now(),
                            "error": str(exc), "traceback": traceback.format_exc()})
            raise
