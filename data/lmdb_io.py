"""LMDB sizing, writing, metadata products, and preprocessing orchestration."""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from andi_rewrite.utils.progress import ProgressReporter

from .healthy_slices import (
    BalancedSliceRecord,
    HealthySliceCandidate,
    balance_candidates_by_z,
    healthy_slice_candidates_for_subject,
    healthy_slices_for_subject,
)
from .imaging import DEFAULT_MODALITIES, load_subject_volume, resize_slices
from .subject_splits import (
    build_combined_train_test_splits,
    build_kfold_subject_splits,
    build_repeated_train_test_splits,
)


@dataclass
class SplitHealthySummary:
    """Summary returned to the healthy-slice command-line entry points."""

    subjects_seen: int
    healthy_slices: int
    lmdb_dir: str
    csv_path: str
    image_size: int
    extra: dict | None = None

    def as_dict(self) -> dict:
        summary = {
            "subjects_seen": self.subjects_seen,
            "healthy_slices": self.healthy_slices,
            "lmdb_dir": self.lmdb_dir,
            "csv_path": self.csv_path,
            "image_size": self.image_size,
        }
        if self.extra:
            summary.update(self.extra)
        return summary


@dataclass
class KFoldLMDBSummary:
    folds: int
    subjects_seen: int
    output_root: str
    lmdb_root: str
    fold_summaries: list[dict]

    def as_dict(self) -> dict:
        return {
            "folds": self.folds,
            "subjects_seen": self.subjects_seen,
            "output_root": self.output_root,
            "lmdb_root": self.lmdb_root,
            "fold_summaries": self.fold_summaries,
        }


def estimate_lmdb_map_size(dataset_path: Path, subject_ids: Iterable[str], progress: bool = False) -> int:
    """Use the original ANDi heuristic to estimate an LMDB map size."""

    subject_ids = list(subject_ids)
    progress_bar = ProgressReporter(len(subject_ids), "Estimating LMDB size", enabled=progress, unit="subject")
    total_bytes = 0
    try:
        for subject_id in subject_ids:
            subject_dir = dataset_path / subject_id
            if not subject_dir.exists():
                raise FileNotFoundError(subject_dir)
            total_bytes += sum(path.stat().st_size for path in subject_dir.iterdir() if path.is_file())
            progress_bar.update()
    finally:
        progress_bar.close()
    return max(total_bytes * 2, 1 << 28)


def estimate_balanced_lmdb_map_size(
    output_slices: int,
    channels: int,
    image_size: int,
    dtype: np.dtype = np.dtype("float32"),
) -> int:
    sample = np.zeros((channels, image_size, image_size), dtype=dtype)
    bytes_per_slice = len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
    return max(int(output_slices * bytes_per_slice * 2.5) + (64 << 20), 1 << 28)


def split_healthy_kfold_to_lmdb(
    dataset_path: str | Path,
    input_csv: str | Path,
    output_root: str | Path,
    lmdb_root: str | Path,
    test_csv: str | Path | None = None,
    combine_train_test: bool = False,
    combined_test_size: int | None = None,
    folds: int = 5,
    image_size: int = 128,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    map_size: int | None = None,
    overwrite: bool = False,
    progress: bool = False,
    sampling_mode: str = "healthy",
    per_z_count: int = 447,
    split_seed: int = 42,
    balance_seed: int = 42,
    shuffle: bool = True,
    fold_start_index: int = 0,
) -> KFoldLMDBSummary:
    """Create one training LMDB per fold while keeping the original LMDB value format."""

    dataset_path = Path(dataset_path)
    input_csv = Path(input_csv)
    output_root = Path(output_root)
    lmdb_root = Path(lmdb_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lmdb_root.mkdir(parents=True, exist_ok=True)

    combined_mode = test_csv is not None and combine_train_test
    fixed_test_mode = test_csv is not None and not combine_train_test
    if combined_mode:
        splits = build_combined_train_test_splits(
            input_csv,
            test_csv,
            folds=folds,
            seed=split_seed,
            test_size=combined_test_size,
        )
        heldout_name = "test"
    elif fixed_test_mode:
        splits = build_repeated_train_test_splits(input_csv, test_csv, folds=folds)
        heldout_name = "test"
    else:
        splits = build_kfold_subject_splits(input_csv, folds=folds, seed=split_seed, shuffle=shuffle)
        heldout_name = "val"

    fold_summaries = []
    for fold_index, train_df, heldout_df in splits:
        fold_number = fold_start_index + fold_index
        fold_name = f"fold_{fold_number}"
        fold_output_dir = output_root / fold_name
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        train_subject_csv = fold_output_dir / "scans_train.csv"
        heldout_subject_csv = fold_output_dir / f"scans_{heldout_name}.csv"
        healthy_slice_csv = fold_output_dir / "healthy_slices_train.csv"
        fold_lmdb_dir = lmdb_root / fold_name
        if fold_output_dir.resolve() == fold_lmdb_dir.resolve():
            raise ValueError(
                "--output-root and --lmdb-root must write to different fold directories; "
                f"both resolved to {fold_output_dir} for {fold_name}."
            )

        train_df.to_csv(train_subject_csv, index=False)
        heldout_df.to_csv(heldout_subject_csv, index=False)

        summary = split_healthy_to_lmdb(
            dataset_path=dataset_path,
            input_csv=train_subject_csv,
            output_csv=healthy_slice_csv,
            image_size=image_size,
            lmdb_dir=fold_lmdb_dir,
            modalities=modalities,
            map_size=map_size,
            overwrite=overwrite,
            progress=progress,
            sampling_mode=sampling_mode,
            per_z_count=per_z_count,
            balance_seed=balance_seed + fold_index,
        )
        fold_summary = summary.as_dict()
        fold_summary.update(
            {
                "fold": fold_number,
                "fold_name": fold_name,
                "split_mode": (
                    "combined_train_test"
                    if combined_mode
                    else "fixed_train_test"
                    if fixed_test_mode
                    else "kfold_validation"
                ),
                "train_subjects": int(train_df.shape[0]),
                f"{heldout_name}_subjects": int(heldout_df.shape[0]),
                "train_subject_csv": str(train_subject_csv),
                f"{heldout_name}_subject_csv": str(heldout_subject_csv),
                "healthy_slice_csv": str(healthy_slice_csv),
            }
        )
        fold_summaries.append(fold_summary)

    return KFoldLMDBSummary(
        folds=folds,
        subjects_seen=int(pd.read_csv(input_csv).shape[0]),
        output_root=str(output_root),
        lmdb_root=str(lmdb_root),
        fold_summaries=fold_summaries,
    )


def split_healthy_z_balanced_to_lmdb(
    dataset_path: Path,
    input_csv: Path,
    output_csv: Path,
    image_size: int,
    lmdb_dir: Path,
    modalities: Sequence[str],
    map_size: int | None,
    overwrite: bool,
    progress: bool,
    per_z_count: int,
    balance_seed: int,
) -> SplitHealthySummary:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("split_healthy requires the optional 'lmdb' package.") from exc

    if lmdb_dir.exists() and any(lmdb_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"LMDB directory is not empty: {lmdb_dir}. "
                "Choose a new --lmdb-dir or rerun with --overwrite."
            )
        shutil.rmtree(lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    subject_column = df.columns[0]
    subject_ids = [str(value) for value in df[subject_column].tolist()]

    candidates_by_z: dict[int, list[HealthySliceCandidate]] = {}
    z_indices_seen: set[int] = set()
    progress_bar = ProgressReporter(len(subject_ids), "Collecting healthy slices", enabled=progress, unit="subject")
    try:
        for subject_id in subject_ids:
            candidates, subject_z_indices = healthy_slice_candidates_for_subject(dataset_path, subject_id, modalities)
            z_indices_seen.update(subject_z_indices)
            for candidate in candidates:
                candidates_by_z.setdefault(candidate.z, []).append(candidate)
            progress_bar.update()
    finally:
        progress_bar.close()

    selected_records, z_summary = balance_candidates_by_z(candidates_by_z, per_z_count, balance_seed)
    skipped_empty_z = sorted(z for z in z_indices_seen if z not in candidates_by_z)
    if skipped_empty_z:
        print(
            "Warning: skipped z indices with no healthy candidates: "
            + ", ".join(str(z) for z in skipped_empty_z)
        )

    resolved_map_size = int(
        map_size
        or estimate_balanced_lmdb_map_size(
            output_slices=len(selected_records),
            channels=len(modalities),
            image_size=image_size,
        )
    )

    selected_by_subject: dict[str, list[BalancedSliceRecord]] = {}
    for record in selected_records:
        selected_by_subject.setdefault(record.candidate.subject_id, []).append(record)

    rows = []
    env = lmdb.open(str(lmdb_dir), map_size=resolved_map_size)
    progress_bar = ProgressReporter(len(subject_ids), "Writing z-balanced slices", enabled=progress, unit="subject")
    try:
        with env.begin(write=True) as txn:
            index = 0
            for subject_id in subject_ids:
                subject_records = selected_by_subject.get(subject_id, [])
                if not subject_records:
                    progress_bar.update()
                    continue

                images, _ = load_subject_volume(dataset_path, subject_id, modalities)
                subject_slices = torch.stack(
                    [images[:, :, :, record.candidate.z] for record in subject_records],
                    dim=0,
                )
                resized = resize_slices(subject_slices, image_size)
                for record, slice_tensor in zip(subject_records, resized):
                    key = f"{index:08}"
                    txn.put(key.encode("ascii"), pickle.dumps(slice_tensor.numpy(), protocol=pickle.HIGHEST_PROTOCOL))
                    rows.append(
                        {
                            subject_column: record.candidate.subject_id,
                            "Slice": record.candidate.z,
                            "original_z": record.candidate.z,
                            "balance_z_count": record.balance_z_count,
                            "sampled_index_in_z": record.sampled_index_in_z,
                            "duplicate_copy_id": record.duplicate_copy_id,
                            "is_extra_sample": record.is_extra_sample,
                        }
                    )
                    index += 1
                progress_bar.update()
    finally:
        progress_bar.close()
        env.close()

    metadata_columns = [
        subject_column,
        "Slice",
        "original_z",
        "balance_z_count",
        "sampled_index_in_z",
        "duplicate_copy_id",
        "is_extra_sample",
    ]
    pd.DataFrame(rows, columns=metadata_columns).to_csv(output_csv, index=False)

    total_candidates = sum(len(value) for value in candidates_by_z.values())
    print("Z-balanced healthy-slice summary:")
    print(f"  total subjects: {len(subject_ids)}")
    print(f"  total healthy candidates before balancing: {total_candidates}")
    print(f"  number of z groups: {len(candidates_by_z)}")
    print(f"  per-z target count: {per_z_count}")
    print(f"  total output slices: {len(rows)}")
    print(f"  skipped empty z count: {len(skipped_empty_z)}")
    for z in sorted(z_summary):
        counts = z_summary[z]
        print(f"  z={z}: original={counts['original']}, output={counts['output']}")

    return SplitHealthySummary(
        subjects_seen=len(subject_ids),
        healthy_slices=len(rows),
        lmdb_dir=str(lmdb_dir),
        csv_path=str(output_csv),
        image_size=int(image_size),
        extra={
            "sampling_mode": "z_balanced",
            "healthy_candidates_before_balancing": total_candidates,
            "z_groups": len(candidates_by_z),
            "per_z_count": int(per_z_count),
            "balance_seed": int(balance_seed),
            "skipped_empty_z_count": len(skipped_empty_z),
            "skipped_empty_z": skipped_empty_z,
            "z_counts": z_summary,
            "map_size": resolved_map_size,
        },
    )


def split_healthy_to_lmdb(
    dataset_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    image_size: int = 128,
    lmdb_dir: str | Path | None = None,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    map_size: int | None = None,
    overwrite: bool = False,
    progress: bool = False,
    sampling_mode: str = "healthy",
    per_z_count: int = 447,
    balance_seed: int = 42,
) -> SplitHealthySummary:
    """Create a healthy-slice LMDB and its slice-index CSV."""

    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("split_healthy requires the optional 'lmdb' package.") from exc

    dataset_path = Path(dataset_path)
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    lmdb_dir = Path(lmdb_dir) if lmdb_dir is not None else dataset_path / "healthy_slices"
    sampling_mode = sampling_mode.lower()
    if sampling_mode not in {"healthy", "z_balanced"}:
        raise ValueError("sampling_mode must be either 'healthy' or 'z_balanced'.")
    if sampling_mode == "z_balanced":
        return split_healthy_z_balanced_to_lmdb(
            dataset_path=dataset_path,
            input_csv=input_csv,
            output_csv=output_csv,
            image_size=image_size,
            lmdb_dir=lmdb_dir,
            modalities=modalities,
            map_size=map_size,
            overwrite=overwrite,
            progress=progress,
            per_z_count=per_z_count,
            balance_seed=balance_seed,
        )

    if lmdb_dir.exists() and any(lmdb_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"LMDB directory is not empty: {lmdb_dir}. "
                "Choose a new --lmdb-dir or rerun with --overwrite."
            )
        shutil.rmtree(lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    subject_column = df.columns[0]
    subject_ids = [str(value) for value in df[subject_column].tolist()]
    resolved_map_size = int(map_size or estimate_lmdb_map_size(dataset_path, subject_ids, progress=progress))

    rows = []
    env = lmdb.open(str(lmdb_dir), map_size=resolved_map_size)
    progress_bar = ProgressReporter(len(subject_ids), "Writing healthy slices", enabled=progress, unit="subject")
    try:
        with env.begin(write=True) as txn:
            index = 0
            for subject_id in subject_ids:
                for current_subject_id, slice_index, slice_tensor in healthy_slices_for_subject(
                    dataset_path,
                    subject_id,
                    image_size,
                    modalities,
                ):
                    key = f"{index:08}"
                    txn.put(key.encode("ascii"), pickle.dumps(slice_tensor.numpy()))
                    rows.append({subject_column: current_subject_id, "Slice": slice_index})
                    index += 1
                progress_bar.update()
    finally:
        progress_bar.close()
        env.close()

    pd.DataFrame(rows, columns=[subject_column, "Slice"]).to_csv(output_csv, index=False)
    return SplitHealthySummary(
        subjects_seen=len(subject_ids),
        healthy_slices=len(rows),
        lmdb_dir=str(lmdb_dir),
        csv_path=str(output_csv),
        image_size=int(image_size),
    )


__all__ = [
    "KFoldLMDBSummary",
    "SplitHealthySummary",
    "estimate_balanced_lmdb_map_size",
    "estimate_lmdb_map_size",
    "split_healthy_kfold_to_lmdb",
    "split_healthy_to_lmdb",
    "split_healthy_z_balanced_to_lmdb",
]
