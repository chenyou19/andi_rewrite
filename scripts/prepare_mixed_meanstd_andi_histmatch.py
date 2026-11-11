"""Build isolated mean/std + mean888 matching data; never overwrite prior runs.

export-fomo must run with read access to the registered FOMO source NIfTIs.
all can run under the MPI/OASIS/BraTS account after that export completes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_mixed_andi_histmatch as base
from andi_rewrite.data.mean_std_normalization import MEAN_STD_SPEC, mean_std_normalize_volume

import lmdb
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

ROOT = base.ROOT
NAME = "mixed_sri24_meanstd_andi_histmatch_mean888_200ep_fresh"
OUTPUT = ROOT / "outputs/datasets/mpi_oasis3_fomo45k_sri24_meanstd_andi_histmatch_mean888"
STAGE = OUTPUT.with_name(OUTPUT.name + ".staging")
REPORT = ROOT / "outputs/reports" / NAME
ANALYSIS = ROOT / "outputs/reports/brats_train888_meanstd_vs_iqr"
SPECTRUM_NAME = "mixed_train_meanstd_andi_histmatch_mean888_empirical_spectrum.npz"
EXPECTED_CASES = 669
SPEC = {"type": "andi_histmatch_mean888_mean_std", "version": 1,
        "source_normalization": MEAN_STD_SPEC,
        "reference": "equal_case_mean_BraTS_train888_mean_std_healthy_tissue_quantiles",
        "operation": "SimpleITK.HistogramMatching_per_case_per_modality",
        "matching": {"histogram_levels": 256, "match_points": 1, "threshold_at_mean_intensity": True},
        "background": -1.0, "model_normalize_input": False}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def dump(path, value):
    base.write_json(Path(path), value)


def token(case):
    return hashlib.sha256(case.encode()).hexdigest()[:20]


def reserve(path, required=5*1024**3):
    if shutil.disk_usage(path).free < required:
        raise RuntimeError(f"Insufficient disk space: need {required/1024**3:.1f} GiB free")


@contextmanager
def exclusive_lock(path):
    """Windows process-scoped lock, released automatically on process failure."""
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen input changed: {path}; inspect this experiment's staging")
    else:
        dump(path, value)


def status(phase, **kwargs):
    dump(REPORT / "build_status.json", {"status": "RUNNING", "phase": phase,
         "pid": os.getpid(), "updated_at": now(), **kwargs})


def build_reference(stage):
    """Validate the completed paired analysis before adopting its Z-score curve."""
    ids = base.reference_ids()
    report = json.loads((ANALYSIS / "comparison.json").read_text())
    rows = [json.loads(s) for s in (ANALYSIS / "per_case.jsonl").read_text().splitlines()]
    if (report["status"] != "PASS" or report["case_count"] != 888 or report["case_ids"] != ids
            or [r["case_id"] for r in rows] != ids or report["modalities"] != list(base.MODALITIES)
            or report["roster_sha256"] != base.digest(base.BRATS_IDS)
            or report["iqr_reproduction_max_abs_error"] != 0):
        raise ValueError("The completed train888 mean/std analysis contract differs")
    for row in rows:
        case = row["case_id"]
        paths = [base.BRATS / case / f"{case}_{m}.nii.gz" for m in (*base.MODALITIES, "seg")]
        if [[p.stat().st_size, p.stat().st_mtime_ns] for p in paths] != row["input_signature"]:
            raise ValueError(f"BraTS input changed since mean/std analysis: {case}")
    with np.load(ANALYSIS / "reference_comparison.npz", allow_pickle=False) as item:
        curves = item["mean_std_quantiles"].copy()
        prior_iqr = item["iqr_quantiles"].copy()
        occupancy = item["occupancy_median"].copy()
        if list(item["case_ids"]) != ids or not np.array_equal(item["probability"], base.QUANTILES):
            raise ValueError("Analysis NPZ roster/probability differs")
    old = base.OUTPUT / "reference.npz"
    if base.digest(old) != report["published_iqr_reference_sha256"]:
        raise ValueError("Published IQR reference changed since paired analysis")
    with np.load(old, allow_pickle=False) as ref:
        if not np.array_equal(prior_iqr, ref["quantiles"]) or not np.array_equal(occupancy, ref["occupancy_median"]):
            raise ValueError("Analysis no longer reproduces the published reference")
        ratios = ref["background_ratio"].copy()
    if (curves.shape != (3, 4097) or not np.isfinite(curves).all()
            or np.any(np.diff(curves, axis=1) < 0) or np.any(occupancy <= 0)):
        raise ValueError("Invalid mean/std reference curves")
    target = stage / "reference.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as item:
            for key, expected in {"quantiles": curves, "background_ratio": ratios,
                                  "probability": base.QUANTILES, "occupancy_median": occupancy,
                                  "case_ids": np.asarray(ids)}.items():
                if not np.array_equal(item[key], expected):
                    raise ValueError(f"Existing mean/std reference differs: {key}")
    else:
        stage.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, quantiles=curves, background_ratio=ratios,
                                probability=base.QUANTILES, occupancy_median=occupancy,
                                case_ids=np.asarray(ids))
        target.with_suffix(".tmp").replace(target)
    provenance = {"status": "PASS", "case_count": 888, "case_ids": ids,
                  "normalization": MEAN_STD_SPEC, "reference_sha256": base.digest(target),
                  "roster_sha256": base.digest(base.BRATS_IDS),
                  "analysis_sha256": {p.name: base.digest(p) for p in
                      (ANALYSIS / "comparison.json", ANALYSIS / "per_case.jsonl", ANALYSIS / "reference_comparison.npz")},
                  "method": "equal-case mean4097; positive full3D mean/std; resize128; exclude resized lesions",
                  "background_ratio": ratios.tolist()}
    freeze(stage / "reference_report.json", provenance)
    return curves, ratios, provenance


def sessions(ds):
    path = ROOT / f"outputs/datasets/{ds}_sri24_robust_iqr/build_report.json"
    return {r["case_id"]: r for r in json.loads(path.read_text())["sessions"]}


def source_slices(paths, stats, rows):
    images = [nib.load(str(p)) for p in paths]
    if any(im.shape != images[0].shape or not np.allclose(im.affine, images[0].affine, atol=1e-6, rtol=0) for im in images):
        raise ValueError("Source modalities have different registered geometry")
    signature_before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    raw = torch.from_numpy(np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in images]))
    normal, statistics = mean_std_normalize_volume(raw, return_statistics=True)
    support = raw > 0
    robust = torch.full_like(raw, -1)
    for c, st in enumerate(stats):
        if int(support[c].sum()) != st["foreground_voxels"]:
            raise ValueError("Source foreground changed since original dataset")
        if st["foreground_voxels"]:
            robust[c][support[c]] = (raw[c][support[c]] - st["median"]) / st["iqr"]
    z = [r["z"] for r in rows]
    values, previous, mask = [base.RESIZE(t.permute(3, 0, 1, 2))[z].numpy()
                              for t in (normal, robust, support.float())]
    provenance = [{"path": str(p.resolve()), "sha256": base.digest(p), "size": p.stat().st_size,
                   "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    if signature_before != [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
        raise ValueError("Source changed while reading")
    return values, mask > 0, previous, statistics, provenance


def verify_original(split, rows, reconstructed):
    with lmdb.open(str(base.mixed_path(base.SOURCE, split)), readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            stored = np.stack([pickle.loads(txn.get(row["key"].encode())) for row in rows])
    error = float(np.max(np.abs(stored - reconstructed)))
    if error > 2e-5:
        raise ValueError(f"Source does not reproduce original slices: max error {error}")
    return error


def export_fomo():
    if OUTPUT.exists():
        raise FileExistsError("Dataset already published; no export is needed")
    groups, manifests = base.case_groups()
    source = sessions("fomo45k")
    target = STAGE / "fomo_meanstd_inputs"
    target.mkdir(parents=True, exist_ok=True)
    freeze(target / "contract.json", {"normalization": MEAN_STD_SPEC, "source_manifests": manifests,
           "source_report_sha256": base.digest(ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json"),
           "normalizer_sha256": base.digest(ROOT / "data/mean_std_normalization.py")})
    records = []
    selected = [g for g in groups if g[1] == "fomo45k"]
    for i, (split, _, case, rows) in enumerate(selected, 1):
        reserve(target)
        paths = [Path(source[case]["input_paths"][m.upper()]) for m in base.MODALITIES]
        cache = target / f"{token(case)}.npz"
        record_path = cache.with_suffix(".json")
        if cache.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record["case_id"] != case or record["split"] != split
                    or record["keys"] != [r["key"] for r in rows] or record["z"] != [r["z"] for r in rows]
                    or base.digest(cache) != record["cache_sha256"]
                    or [base.digest(p) for p in paths] != [p["sha256"] for p in record["raw_files"]]):
                raise ValueError(f"FOMO export input/output changed: {case}")
        else:
            values, mask, robust, statistics, provenance = source_slices(paths, source[case]["statistics"], rows)
            error = verify_original(split, rows, robust)
            with cache.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(handle, values=values, support=mask,
                                    z=np.asarray([r["z"] for r in rows]), case_id=case)
            cache.with_suffix(".tmp").replace(cache)
            record = {"case_id": case, "split": split, "keys": [r["key"] for r in rows],
                      "z": [r["z"] for r in rows], "statistics": statistics, "raw_files": provenance,
                      "cache_file": cache.name, "cache_sha256": base.digest(cache),
                      "original_reconstruction_max_abs_error": error}
            dump(record_path, record)
        records.append(record)
        dump(REPORT / "export_status.json", {"status": "RUNNING", "pid": os.getpid(),
             "completed_cases": i, "total_cases": len(selected), "updated_at": now()})
        if i % 5 == 0 or i == len(selected):
            print(f"EXPORT_FOMO {i}/{len(selected)}", flush=True)
    dump(target / "export_report.json", {"status": "PASS", "cases": records,
         "contract_sha256": base.digest(target / "contract.json")})
    dump(REPORT / "export_status.json", {"status": "PASS", "cases": len(records), "updated_at": now()})


def match_volume(source, support, reference):
    if source.shape != support.shape or not np.isfinite(source).all() or not support.any():
        raise ValueError("Invalid source/support for matching")
    if np.max(np.abs(source[~support] + 1), initial=0) > 2e-5:
        raise ValueError("Source background does not match its support")
    filt = sitk.HistogramMatchingImageFilter()
    filt.SetNumberOfHistogramLevels(256)
    filt.SetNumberOfMatchPoints(1)
    filt.SetThresholdAtMeanIntensity(True)
    result = sitk.GetArrayFromImage(filt.Execute(sitk.GetImageFromArray(source.astype(np.float64)), reference)).astype(np.float32)
    result[~support] = -1
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite matched output")
    return result


def build():
    status("VERIFY_INPUTS")
    groups, manifests = base.case_groups()
    if len(groups) != EXPECTED_CASES:
        raise ValueError("Expected the original 669 source cases")
    work = OUTPUT if OUTPUT.exists() else STAGE
    work.mkdir(parents=True, exist_ok=True)
    curves, ratios, reference = build_reference(work)
    export_path = work / "fomo_meanstd_inputs/export_report.json"
    if not export_path.exists():
        raise FileNotFoundError("Run this script with export-fomo under the FOMO source account first")
    exported = json.loads(export_path.read_text())
    contract_path = export_path.parent / "contract.json"
    contract = json.loads(contract_path.read_text())
    if (exported["status"] != "PASS" or base.digest(contract_path) != exported["contract_sha256"]
            or contract["normalization"] != MEAN_STD_SPEC or contract["source_manifests"] != manifests
            or contract["normalizer_sha256"] != base.digest(ROOT / "data/mean_std_normalization.py")
            or contract["source_report_sha256"] != base.digest(ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json")):
        raise ValueError("FOMO export contract differs")
    fomo = {(r["split"], r["case_id"]): r for r in exported["cases"]}
    expected_fomo = {(s, c) for s, d, c, _ in groups if d == "fomo45k"}
    if set(fomo) != expected_fomo or len(fomo) != len(exported["cases"]):
        raise ValueError("Incomplete or duplicate FOMO export")
    signature = {"normalization": SPEC, "reference_sha256": reference["reference_sha256"],
                 "source_manifests_sha256": manifests, "SimpleITK": sitk.Version_VersionString(),
                 "export_report_sha256": base.digest(export_path),
                 "source_build_reports_sha256": {ds: base.digest(ROOT / f"outputs/datasets/{ds}_sri24_robust_iqr/build_report.json") for ds in base.SOURCES},
                 "implementation_sha256": {name: base.digest(ROOT / name) for name in
                     ("scripts/prepare_mixed_meanstd_andi_histmatch.py", "data/mean_std_normalization.py", "scripts/compute_lmdb_spectrum.py")}}
    freeze(work / "input_signature.json", signature)
    if OUTPUT.exists():
        report = json.loads((OUTPUT / "build_report.json").read_text())
        if report["status"] != "PASS" or report["input_signature"] != signature:
            raise ValueError("Existing published dataset has a different contract")
        status("PUBLISHED", status="PASS", output=str(OUTPUT))
        return
    remaining = max(0, int(sum(base.EXPECTED.values())*3*128*128*4*1.2) - sum(p.stat().st_size for p in work.rglob("data.mdb")))
    reserve(work, remaining + 15*1024**3)
    references = base.synthetic_references(curves, ratios)
    stats = {ds: sessions(ds) for ds in ("mpi", "oasis3")}
    writers, metrics = {}, []
    try:
        for split in base.SPLITS:
            folder = base.mixed_path(work, split)
            folder.mkdir(parents=True, exist_ok=True)
            writers[split] = lmdb.open(str(folder), map_size=int(base.EXPECTED[split]*3*128*128*4*1.3)+1024**3)
            freeze(folder / "normalization.json", {**SPEC, "reference_sha256": signature["reference_sha256"]})
        for i, (split, ds, case, rows) in enumerate(groups, 1):
            reserve(work)
            if ds == "fomo45k":
                record = fomo[(split, case)]
                cache = export_path.parent / record["cache_file"]
                if (cache.parent.resolve() != export_path.parent.resolve()
                        or base.digest(cache) != record["cache_sha256"]
                        or record["keys"] != [r["key"] for r in rows] or record["z"] != [r["z"] for r in rows]):
                    raise ValueError(f"FOMO cache changed or misindexed: {case}")
                with np.load(cache, allow_pickle=False) as item:
                    if str(item["case_id"]) != case or item["z"].tolist() != record["z"]:
                        raise ValueError("FOMO NPZ identity differs")
                    values, support = item["values"], item["support"]
                statistics, provenance = record["statistics"], record["raw_files"]
                error = record["original_reconstruction_max_abs_error"]
            else:
                paths = base.case_paths(ds, case)
                values, support, robust, statistics, provenance = source_slices(
                    [paths[m] for m in base.MODALITIES], stats[ds][case]["statistics"], rows)
                error = verify_original(split, rows, robust)
            if values.shape != (len(rows), 3, 128, 128) or values.dtype != np.float32 or support.dtype != bool or support.shape != values.shape:
                raise ValueError("Invalid normalized source array")
            matched = np.stack([match_volume(values[:, c], support[:, c], references[c]) for c in range(3)], axis=1)
            writer = writers[split]
            keys = [r["key"].encode("ascii") for r in rows]
            with writer.begin() as txn:
                found = [txn.get(k) is not None for k in keys]
                if all(found):
                    if any(not np.array_equal(pickle.loads(txn.get(k)), v) for k, v in zip(keys, matched)):
                        raise ValueError("Existing staging case differs from recomputed output")
            if any(found) and not all(found):
                raise ValueError("Partial case transaction")
            if not all(found):
                with writer.begin(write=True) as txn:
                    for key, value in zip(keys, matched):
                        if not txn.put(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                            raise ValueError("Duplicate key")
            with writer.begin() as txn:
                readback = np.stack([pickle.loads(txn.get(k)) for k in keys])
            if not np.array_equal(readback, matched) or not np.all(readback[~support] == -1):
                raise ValueError("Exact readback/background verification failed")
            metrics.append({"split": split, "source_dataset": ds, "case_id": case, "entries": len(rows),
                            "statistics": statistics, "raw_files": provenance,
                            "original_reconstruction_max_abs_error": error,
                            "reference_quantile_mae": [float(np.mean(np.abs(np.quantile(matched[:,c][support[:,c]], base.QUANTILES)-curves[c]))) for c in range(3)]})
            if i % 5 == 0 or i == len(groups):
                status("BUILD_LMDB", completed_cases=i, total_cases=len(groups))
                print(f"BUILD {i}/{len(groups)} {split} {ds}", flush=True)
        for writer in writers.values():
            writer.sync()
    finally:
        for writer in writers.values():
            writer.close()
    (work / "case_metrics.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in metrics))
    for split in base.SPLITS:
        shutil.copyfile(base.manifest_path(base.SOURCE, split), base.manifest_path(work, split))
    status("FULL_READBACK")
    hashes = {}
    for split, count in base.EXPECTED.items():
        digest = hashlib.sha256()
        with lmdb.open(str(base.mixed_path(work, split)), readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                if txn.stat()["entries"] != count:
                    raise ValueError("LMDB entry count differs")
                for i in range(count):
                    value = txn.get(f"{i:08d}".encode())
                    if value is None:
                        raise ValueError("Missing contiguous LMDB key")
                    array = pickle.loads(value)
                    if array.shape != (3,128,128) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError("Invalid LMDB value")
                    digest.update(value)
        if base.digest(base.manifest_path(work, split)) != manifests[split]:
            raise ValueError("Source manifest changed")
        hashes[split] = digest.hexdigest()
    status("COMPUTE_SPECTRUM")
    spectrum = work / "spectrum" / SPECTRUM_NAME
    sidecar = spectrum.with_suffix(".npz.metadata.json")
    if not spectrum.exists() or not sidecar.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/compute_lmdb_spectrum.py"),
                        "--lmdb-path", str(work / "train"), "--out", str(spectrum),
                        "--mask-mode", "robust_iqr_background", "--window", "hann", "--crop-margin", "4",
                        "--radial-bins", "64", "--channel-order", "FLAIR", "T1", "T2",
                        "--source-manifest", str(work / "source_entries.jsonl"), "--no-progress", "--overwrite"], check=True)
    metadata = json.loads(sidecar.read_text())
    if (metadata["source_serialized_values_sha256"] != hashes["train"]
            or metadata["num_slices_used"]+metadata["num_slices_skipped"] != base.EXPECTED["train"]):
        raise ValueError("Spectrum does not cover this complete train LMDB")
    with np.load(spectrum, allow_pickle=False) as item:
        for key in ("mean_amplitude", "radial_amplitude", "mean_power", "radial_power"):
            if not np.isfinite(item[key]).all() or not np.any(item[key] > 0):
                raise ValueError("Invalid empirical spectrum")
    metadata.update(source_lmdb=str(OUTPUT / "train"), source_manifest=str(OUTPUT / "source_entries.jsonl"),
                    output_npz=str(OUTPUT / "spectrum" / SPECTRUM_NAME), staging_source_lmdb=str(work / "train"))
    dump(sidecar, metadata)
    report = {"status": "PASS", "completed_at": now(), "case_count": len(groups), "entries": base.EXPECTED,
              "shape": [3,128,128], "dtype": "float32", "channel_order": ["FLAIR", "T1", "T2"],
              "input_signature": signature, "source_serialized_values_sha256": hashes,
              "reference_sha256": signature["reference_sha256"], "spectrum_sha256": base.digest(spectrum),
              "spectrum_num_slices_used": metadata["num_slices_used"],
              "spectrum_num_slices_skipped": metadata["num_slices_skipped"], "participant_overlap": 0,
              "max_original_reconstruction_abs_error": max(r["original_reconstruction_max_abs_error"] for r in metrics),
              "verification": "exact per-case readback/support, contiguous keys, finite float32, source manifests, own train spectrum"}
    dump(work / "build_report.json", report)
    reserve(work)
    work.rename(OUTPUT)
    status("PUBLISHED", status="PASS", output=str(OUTPUT), entries=base.EXPECTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export-fomo", "reference", "all"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    REPORT.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(REPORT / "build.lock"):
        try:
            if args.command == "export-fomo":
                export_fomo()
            elif args.command == "reference":
                build_reference(OUTPUT if OUTPUT.exists() else STAGE)
                print("REFERENCE PASS 888", flush=True)
            else:
                build()
        except BaseException as exc:
            destination = "export_status.json" if args.command == "export-fomo" else "build_status.json"
            dump(REPORT / destination, {"status": "FAILED", "pid": os.getpid(), "updated_at": now(),
                 "error": str(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
