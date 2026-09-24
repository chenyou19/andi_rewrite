"""Match the existing mixed robust-IQR LMDB with ANDi's SimpleITK operation.

The reference is an equal-case mean of healthy-tissue quantiles from the 888
BraTS training subjects. The source values always come from the mixed LMDB.
Registered NIfTI files (MPI/OASIS3) or the identically indexed original FOMO
p99 LMDB are read only to recover the source support mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import lmdb
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from torchvision.transforms import Resize

from andi_rewrite.data.robust_normalization import robust_normalize_volume


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr"
OUTPUT = ROOT / "outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888"
BRATS = Path("C:/ML/data/BraTS_2021")
BRATS_IDS = ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/brats_remaining_train.csv"
BRATS_VAL = ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/brats_validation50.csv"
BRATS_TEST = ROOT / "splits/BraTS21/scans_test.csv"
SOURCES = ("mpi", "oasis3", "fomo45k")
FOMO_P99 = Path("C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH")
MODALITIES = ("flair", "t1", "t2")
SPLITS = ("train", "val")
EXPECTED = {"train": 84341, "val": 9569}
QUANTILES = np.linspace(0.0, 1.0, 4097)
METRIC_GRID = np.linspace(0.0, 1.0, 101)
RESIZE = Resize(128, antialias=True)
SPEC = {
    "type": "andi_histmatch_mean888_robust_iqr",
    "version": 1,
    "source_normalization": "robust_iqr_v1",
    "reference": "equal_case_mean_BraTS_train888_healthy_tissue_quantiles",
    "operation": "SimpleITK.HistogramMatching_default_parameters_per_case_per_modality",
    "background": -1.0,
    "model_normalize_input": False,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def ids_from_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty BraTS roster: {path}")
    key = "BraTS21ID" if "BraTS21ID" in rows[0] else next(iter(rows[0]))
    return [row[key] for row in rows]


def reference_ids() -> list[str]:
    train = ids_from_csv(BRATS_IDS)
    val = set(ids_from_csv(BRATS_VAL))
    test = set(ids_from_csv(BRATS_TEST))
    if len(train) != 888 or len(set(train)) != 888 or set(train) & (val | test):
        raise ValueError("BraTS reference roster is not the isolated 888-subject training set")
    return train


def reference_case(case_id: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = BRATS / case_id
    paths = [folder / f"{case_id}_{modality}.nii.gz" for modality in (*MODALITIES, "seg")]
    signature = json.dumps([(p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    cache = cache_dir / f"{case_id}.npz"
    if cache.exists():
        with np.load(cache) as item:
            if str(item["signature"]) != signature:
                raise ValueError(f"BraTS reference file changed: {case_id}")
            return item["quantiles"], item["occupancy"]

    raw = np.stack([np.asarray(nib.load(str(p)).dataobj, dtype=np.float32) for p in paths[:3]])
    seg = np.asarray(nib.load(str(paths[3])).dataobj, dtype=np.float32)
    if raw.shape[1:] != seg.shape or not np.isfinite(raw).all() or not np.isfinite(seg).all():
        raise ValueError(f"Invalid BraTS volume: {case_id}")
    normalized = robust_normalize_volume(torch.from_numpy(raw))
    values = RESIZE(normalized.permute(3, 0, 1, 2)).numpy()
    support = RESIZE(torch.from_numpy((raw > 0).astype(np.float32)).permute(3, 0, 1, 2)).numpy() > 0
    lesion = RESIZE(torch.from_numpy((seg > 0).astype(np.float32)).permute(2, 0, 1).unsqueeze(1)).numpy()[:, 0] > 0
    curves = np.empty((3, QUANTILES.size), dtype=np.float64)
    occupancy = np.empty(3, dtype=np.float64)
    for channel in range(3):
        valid = support[:, channel] & ~lesion
        tissue = values[:, channel][valid]
        if tissue.size < 2 or not np.isfinite(tissue).all() or np.ptp(tissue) <= 0:
            raise ValueError(f"Empty or degenerate reference tissue: {case_id} {MODALITIES[channel]}")
        curves[channel] = np.quantile(tissue, QUANTILES)
        occupancy[channel] = float(valid.mean())
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, quantiles=curves, occupancy=occupancy, signature=signature)
    cache.with_suffix(".tmp").replace(cache)
    return curves, occupancy


def build_reference(stage: Path) -> tuple[np.ndarray, np.ndarray, str]:
    ids = reference_ids()
    target = stage / "reference.npz"
    if target.exists():
        with np.load(target) as item:
            if list(item["case_ids"]) != ids:
                raise ValueError("Existing reference uses different BraTS subjects")
            return item["quantiles"], item["background_ratio"], digest(target)
    cache_dir = stage / "reference_cases"
    means = np.zeros((3, QUANTILES.size), dtype=np.float64)
    occupancy = np.empty((len(ids), 3), dtype=np.float64)
    for index, case_id in enumerate(ids):
        curve, occupancy[index] = reference_case(case_id, cache_dir)
        means += curve / len(ids)
        if (index + 1) % 25 == 0 or index + 1 == len(ids):
            print(f"REFERENCE {index + 1}/{len(ids)}", flush=True)
    median_fraction = np.median(occupancy, axis=0)
    background_ratio = (1.0 - median_fraction) / median_fraction
    if not np.isfinite(means).all() or np.any(np.diff(means, axis=1) < -1e-9):
        raise ValueError("Invalid averaged BraTS reference quantiles")
    with target.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, quantiles=means, background_ratio=background_ratio,
                            case_ids=np.asarray(ids), probability=QUANTILES,
                            occupancy_median=median_fraction)
    target.with_suffix(".tmp").replace(target)
    write_json(stage / "reference_report.json", {
        "status": "PASS", "case_count": len(ids), "case_ids": ids,
        "braTS_train_roster_sha256": digest(BRATS_IDS),
        "background_ratio": background_ratio.tolist(), "reference_sha256": digest(target),
        "method": "robust-IQR full volume, resize 128, exclude resized lesion coverage, equal-case mean 4097 tissue quantiles",
    })
    return means, background_ratio, digest(target)


def synthetic_references(curves: np.ndarray, background_ratio: np.ndarray) -> list[sitk.Image]:
    images = []
    for channel in range(3):
        count = int(round(curves.shape[1] * float(background_ratio[channel])))
        voxels = np.concatenate((np.full(count, -1.0), curves[channel])).reshape(1, 1, -1)
        images.append(sitk.GetImageFromArray(voxels.astype(np.float64)))
    return images


def source_rows(dataset: str, split: str) -> dict[str, dict]:
    root = ROOT / f"outputs/datasets/{dataset}_sri24_robust_iqr"
    if dataset == "fomo45k":
        path = root / f"manifests/{split}_entries.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with (root / "entries.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = [row for row in rows if row["split"] == split]
    index = {row["key"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate source key: {dataset} {split}")
    return index


def mixed_path(root: Path, split: str) -> Path:
    return root / "train" if split == "train" else root / "validation/val"


def manifest_path(root: Path, split: str) -> Path:
    return root / "source_entries.jsonl" if split == "train" else root / "validation/source_entries.jsonl"


def case_groups() -> tuple[list[tuple[str, str, str, list[dict]]], dict[str, str]]:
    indices = {(dataset, split): source_rows(dataset, split) for dataset in SOURCES for split in SPLITS}
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    manifest_hashes = {}
    participants = {split: set() for split in SPLITS}
    for split in SPLITS:
        path = manifest_path(SOURCE, split)
        manifest_hashes[split] = digest(path)
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                mixed = json.loads(line)
                key = f"{index:08d}"
                if mixed["key"] != key or mixed["source_split"] != split:
                    raise ValueError(f"Mixed manifest ordering/split mismatch: {split} {key}")
                dataset = mixed["source_dataset"]
                row = indices[(dataset, split)][mixed["source_key"]]
                case_id = row["case_id"]
                participant = f"{dataset}:{row['participant_id']}"
                participants[split].add(participant)
                groups[(split, dataset, case_id)].append({
                    "key": key, "source_key": mixed["source_key"], "z": int(row["z"]),
                    "participant_id": participant,
                })
            if index + 1 != EXPECTED[split]:
                raise ValueError(f"Mixed manifest count mismatch: {split}")
    if participants["train"] & participants["val"]:
        raise ValueError("Mixed train/validation participant leakage")
    ordered = []
    for (split, dataset, case_id), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["z"])
        if len({row["z"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate z in case {dataset}:{case_id}")
        ordered.append((split, dataset, case_id, rows))
    return ordered, manifest_hashes


def case_paths(dataset: str, case_id: str) -> dict[str, Path]:
    folder = ROOT / f"outputs/datasets/{dataset}_sri24_robust_iqr/volumes" / case_id
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS":
        raise ValueError(f"Registered volume is not PASS: {dataset}:{case_id}")
    return {modality: folder / status["outputs"][modality] for modality in MODALITIES}


def support_mask(path: Path, z_values: list[int]) -> np.ndarray:
    raw = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if raw.ndim != 3 or max(z_values) >= raw.shape[2]:
        raise ValueError(f"Source NIfTI geometry mismatch: {path}")
    selected = np.ascontiguousarray(np.moveaxis(raw, 2, 0)[z_values] > 0, dtype=np.float32)
    return RESIZE(torch.from_numpy(selected).unsqueeze(1)).numpy()[:, 0] > 0


def match_volume(source: np.ndarray, mask: np.ndarray, reference: sitk.Image) -> np.ndarray:
    """Apply the ANDi call, then restore the known pre-resize background."""
    if source.shape != mask.shape or source.ndim != 3 or not np.isfinite(source).all():
        raise ValueError("Invalid source volume or support mask")
    if not mask.any() or np.max(np.abs(source[~mask] + 1.0), initial=0.0) > 2e-5:
        raise ValueError("Source LMDB does not match its registered-volume support mask")
    image = sitk.GetImageFromArray(source.astype(np.float64))
    result = sitk.GetArrayFromImage(sitk.HistogramMatching(image, reference)).astype(np.float32)
    result[~mask] = -1.0
    if result.shape != source.shape or not np.isfinite(result).all():
        raise ValueError("SimpleITK produced an invalid matched volume")
    return result


def case_data(split: str, dataset: str, case_id: str, rows: list[dict],
              reader: lmdb.Environment, fomo_reader: lmdb.Environment | None,
              references: list[sitk.Image], target_curves: np.ndarray):
    with reader.begin() as txn:
        source = np.stack([pickle.loads(txn.get(row["key"].encode("ascii"))) for row in rows])
    if source.shape != (len(rows), 3, 128, 128) or source.dtype != np.float32:
        raise ValueError(f"Invalid mixed LMDB shape/type: {split} {dataset} {case_id}")
    output = np.empty_like(source)
    paths = case_paths(dataset, case_id) if dataset != "fomo45k" else None
    z_values = [row["z"] for row in rows]
    fomo_support = None
    if dataset == "fomo45k":
        if fomo_reader is None:
            raise ValueError("FOMO p99 support LMDB is required")
        with fomo_reader.begin() as txn:
            fomo_support = np.stack([pickle.loads(txn.get(row["source_key"].encode("ascii"))) > 0 for row in rows])
        if fomo_support.shape != source.shape:
            raise ValueError(f"FOMO p99 support shape mismatch: {case_id}")
    metrics = []
    for channel, modality in enumerate(MODALITIES):
        mask = fomo_support[:, channel] if fomo_support is not None else support_mask(paths[modality], z_values)
        output[:, channel] = match_volume(source[:, channel], mask, references[channel])
        target = np.interp(METRIC_GRID, QUANTILES, target_curves[channel])
        before = np.quantile(source[:, channel][mask], METRIC_GRID)
        after = np.quantile(output[:, channel][mask], METRIC_GRID)
        metrics.append({"modality": modality, "before_mae": float(np.mean(np.abs(before - target))),
                        "after_mae": float(np.mean(np.abs(after - target))),
                        "foreground_fraction": float(mask.mean())})
    return source, output, metrics


def fomo_support_readers() -> dict[str, lmdb.Environment]:
    robust = ROOT / "outputs/datasets/fomo45k_sri24_robust_iqr/manifests"
    for split in SPLITS:
        name = f"{split}_entries.csv"
        if digest(robust / name) != digest(FOMO_P99 / "manifests" / name):
            raise ValueError(f"FOMO p99 support manifest differs from robust-IQR: {split}")
    return {split: lmdb.open(str(FOMO_P99 / split), readonly=True, lock=False,
                             readahead=False, max_readers=2) for split in SPLITS}


def pilot(stage: Path, groups: list, references: list[sitk.Image], curves: np.ndarray) -> None:
    report_path = stage / "pilot_report.json"
    if report_path.exists() and json.loads(report_path.read_text())["status"] == "PASS":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = {}
    for group in groups:
        chosen.setdefault((group[0], group[1]), group)
    if len(chosen) != 6:
        raise ValueError("Pilot needs one case for each source and split")
    stage.joinpath("pilot").mkdir(parents=True, exist_ok=True)
    readers = {split: lmdb.open(str(mixed_path(SOURCE, split)), readonly=True, lock=False,
                                readahead=False, max_readers=2) for split in SPLITS}
    fomo_readers = fomo_support_readers()
    results = []
    try:
        for (split, dataset), (_, _, case_id, rows) in chosen.items():
            before, after, metrics = case_data(split, dataset, case_id, rows, readers[split],
                                               fomo_readers[split] if dataset == "fomo45k" else None,
                                               references, curves)
            z = int(np.argmax(np.sum(before[:, 0] > -0.99, axis=(1, 2))))
            fig, axes = plt.subplots(3, 2, figsize=(7, 9))
            for channel, modality in enumerate(MODALITIES):
                for column, values in enumerate((before, after)):
                    axes[channel, column].imshow(values[z, channel], cmap="gray", vmin=-2, vmax=3)
                    axes[channel, column].set_title(f"{modality} {'before' if column == 0 else 'matched'}")
                    axes[channel, column].axis("off")
            fig.tight_layout()
            figure = stage / "pilot" / f"{split}_{dataset}.png"
            fig.savefig(figure, dpi=120)
            plt.close(fig)
            results.append({"split": split, "source_dataset": dataset, "case_id": case_id,
                            "entries": len(rows), "metrics": metrics, "montage": str(figure)})
            print(f"PILOT {split} {dataset} {case_id}", flush=True)
    finally:
        for reader in readers.values():
            reader.close()
        for reader in fomo_readers.values():
            reader.close()
    write_json(report_path, {"status": "PASS", "cases": results})


def input_signature(manifest_hashes: dict[str, str], reference_hash: str) -> dict:
    return {"source_root": str(SOURCE), "source_manifests_sha256": manifest_hashes,
            "source_build_reports_sha256": {split: digest(SOURCE / "build_report.json" if split == "train"
                                                      else SOURCE / "validation/build_report.json") for split in SPLITS},
            "fomo_p99_support_manifests_sha256": {split: digest(FOMO_P99 / "manifests" / f"{split}_entries.csv")
                                                   for split in SPLITS},
            "reference_sha256": reference_hash, "method_version": 1}


def build(stage: Path, groups: list, references: list[sitk.Image], curves: np.ndarray,
          manifest_hashes: dict[str, str], reference_hash: str) -> None:
    signature = input_signature(manifest_hashes, reference_hash)
    signature_path = stage / "input_signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Staging input signature changed")
    else:
        write_json(signature_path, signature)
    if shutil.disk_usage(stage).free < 45 * 1024**3:
        raise RuntimeError("Need at least 45 GiB free before writing the new LMDB")
    readers = {split: lmdb.open(str(mixed_path(SOURCE, split)), readonly=True, lock=False,
                                readahead=False, max_readers=2) for split in SPLITS}
    fomo_readers = fomo_support_readers()
    writers = {}
    for split in SPLITS:
        path = mixed_path(stage, split)
        path.mkdir(parents=True, exist_ok=True)
        writers[split] = lmdb.open(str(path), map_size=(96 if split == "train" else 16) * 1024**3)
    metric_rows = []
    try:
        for index, (split, dataset, case_id, rows) in enumerate(groups, 1):
            keys = [row["key"].encode("ascii") for row in rows]
            with writers[split].begin() as txn:
                found = [txn.get(key) is not None for key in keys]
            if any(found) and not all(found):
                raise ValueError(f"Partial case transaction in staging: {split} {dataset} {case_id}")
            source, result, metrics = case_data(split, dataset, case_id, rows, readers[split],
                                                fomo_readers[split] if dataset == "fomo45k" else None,
                                                references, curves)
            if not all(found):
                with writers[split].begin(write=True) as txn:
                    for key, value in zip(keys, result):
                        if not txn.put(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                            raise ValueError(f"Duplicate output key: {key!r}")
            metric_rows.append({"split": split, "source_dataset": dataset, "case_id": case_id,
                                "entries": len(rows), "metrics": metrics})
            if index % 10 == 0 or index == len(groups):
                print(f"BUILD {index}/{len(groups)}", flush=True)
        for writer in writers.values():
            writer.sync()
            used_bytes = (writer.info()["last_pgno"] + 1) * writer.stat()["psize"]
            compact_map_size = ((used_bytes + (1 << 30) - 1) // (1 << 30) + 1) * (1 << 30)
            writer.set_mapsize(compact_map_size)
            writer.sync()
    finally:
        for writer in writers.values():
            writer.close()
        for reader in readers.values():
            reader.close()
        for reader in fomo_readers.values():
            reader.close()
    with (stage / "case_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metric_rows:
            handle.write(json.dumps(row) + "\n")
    for split in SPLITS:
        shutil.copyfile(manifest_path(SOURCE, split), manifest_path(stage, split))
        write_json(mixed_path(stage, split) / "normalization.json", {**SPEC, "reference_sha256": reference_hash})


def audit(stage: Path, groups: list, manifest_hashes: dict[str, str], reference_hash: str) -> dict:
    counts = defaultdict(int)
    metric_sums = defaultdict(lambda: [0.0, 0.0, 0])
    with (stage / "case_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for metric in row["metrics"]:
                bucket = metric_sums[(row["split"], metric["modality"])]
                bucket[0] += metric["before_mae"]
                bucket[1] += metric["after_mae"]
                bucket[2] += 1
    for split in SPLITS:
        if digest(manifest_path(stage, split)) != manifest_hashes[split]:
            raise ValueError(f"Output provenance manifest differs from input: {split}")
        path = mixed_path(stage, split)
        spec = json.loads((path / "normalization.json").read_text())
        if spec != {**SPEC, "reference_sha256": reference_hash}:
            raise ValueError(f"Output normalization contract mismatch: {split}")
        with lmdb.open(str(path), readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                if txn.stat()["entries"] != EXPECTED[split]:
                    raise ValueError(f"Output entry count mismatch: {split}")
                for index in range(EXPECTED[split]):
                    value = txn.get(f"{index:08d}".encode("ascii"))
                    if value is None:
                        raise ValueError(f"Missing output key: {split} {index}")
                    array = pickle.loads(value)
                    if not isinstance(array, np.ndarray) or array.shape != (3, 128, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError(f"Invalid output value: {split} {index}")
                    counts[split] += 1
    report = {"status": "PASS", "completed_at": datetime.now().astimezone().isoformat(),
              "entries": dict(counts), "case_count": len(groups), "reference_sha256": reference_hash,
              "input_signature": input_signature(manifest_hashes, reference_hash),
              "mean_case_quantile_mae": {split: {modality: {
                  "before": metric_sums[(split, modality)][0] / metric_sums[(split, modality)][2],
                  "after": metric_sums[(split, modality)][1] / metric_sums[(split, modality)][2],
              } for modality in MODALITIES} for split in SPLITS},
              "verification": "Full output readback: contiguous keys, shape, dtype, finite values; exact source manifest SHA256; per-case support checked during build"}
    write_json(stage / "build_report.json", report)
    return report


def compute_spectrum(stage: Path) -> None:
    spectrum = stage / "spectrum/mixed_train_andi_histmatch_mean888_empirical_spectrum.npz"
    if spectrum.exists():
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/compute_lmdb_spectrum.py"),
                    "--lmdb-path", str(stage / "train"), "--out", str(spectrum),
                    "--mask-mode", "robust_iqr_background", "--channel-order", "FLAIR", "T1", "T2",
                    "--source-manifest", str(stage / "source_entries.jsonl"), "--no-progress"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reference", "pilot", "all"), default="all", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite published dataset: {OUTPUT}")
    stage = OUTPUT.with_name(OUTPUT.name + ".staging")
    stage.mkdir(parents=True, exist_ok=True)
    groups, manifest_hashes = case_groups()
    curves, background_ratio, reference_hash = build_reference(stage)
    if args.command == "reference":
        return
    references = synthetic_references(curves, background_ratio)
    pilot(stage, groups, references, curves)
    if args.command == "pilot":
        return
    build(stage, groups, references, curves, manifest_hashes, reference_hash)
    report = audit(stage, groups, manifest_hashes, reference_hash)
    compute_spectrum(stage)
    report["spectrum_sha256"] = digest(stage / "spectrum/mixed_train_andi_histmatch_mean888_empirical_spectrum.npz")
    write_json(stage / "build_report.json", report)
    stage.rename(OUTPUT)
    spectrum_sidecar = OUTPUT / "spectrum/mixed_train_andi_histmatch_mean888_empirical_spectrum.npz.metadata.json"
    spectrum_metadata = json.loads(spectrum_sidecar.read_text(encoding="utf-8"))
    spectrum_metadata["published_source_lmdb"] = str(OUTPUT / "train")
    spectrum_metadata["published_output_npz"] = str(OUTPUT / "spectrum/mixed_train_andi_histmatch_mean888_empirical_spectrum.npz")
    write_json(spectrum_sidecar, spectrum_metadata)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "entries": report["entries"]}), flush=True)


if __name__ == "__main__":
    main()
