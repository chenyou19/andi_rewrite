"""建立 ANDi healthy-slice LMDB 的前處理工具。

這個模組對應原版 split_healthy.py 的行為，同時讓邏輯可被 scripts、
notebook 或未來 experiment runner 重用。
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from andi_rewrite.utils.progress import ProgressReporter


DEFAULT_MODALITIES = ("flair", "t1", "t1ce", "t2")


@dataclass
class SplitHealthySummary:
    """前處理完成後由 CLI 印出的簡短摘要。"""

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


@dataclass(frozen=True)
class HealthySliceCandidate:
    subject_id: str
    z: int
    modality_paths: tuple[str, ...]


@dataclass(frozen=True)
class BalancedSliceRecord:
    candidate: HealthySliceCandidate
    balance_z_count: int
    sampled_index_in_z: int
    duplicate_copy_id: int
    is_extra_sample: bool


def normalize_volume(images: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """以 foreground 第 99 百分位數正規化每個 MRI modality。"""

    images = images.float()
    for modality in range(images.shape[0]):
        values = images[modality].reshape(-1)
        foreground = values[values > 0]
        if foreground.numel() == 0:
            continue
        percentile_99 = torch.quantile(foreground, 0.99).clamp_min(eps)
        images[modality] = images[modality] / percentile_99
    return images


def _load_nifti(path: Path, dtype: type = float) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("split_healthy requires the optional 'nibabel' package.") from exc

    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def load_subject_volume(
    dataset_path: Path,
    subject_id: str,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> tuple[torch.Tensor, np.ndarray]:
    """載入單一 subject 的 modalities 與 segmentation mask。

    預期檔名相容於原版 ANDi layout：
    {subject_id}_{modality}.nii.gz and {subject_id}_seg.nii.gz.
    """

    subject_dir = dataset_path / subject_id
    images = []
    for modality in modalities:
        image_path = subject_dir / f"{subject_id}_{modality}.nii.gz"
        images.append(torch.from_numpy(_load_nifti(image_path, dtype=float)))

    mask_path = subject_dir / f"{subject_id}_seg.nii.gz"
    mask = _load_nifti(mask_path, dtype=int)
    mask[mask >= 1] = 1
    return normalize_volume(torch.stack(images, dim=0)), mask


def resize_slices(slices: torch.Tensor, image_size: int) -> torch.Tensor:
    """將 [N, C, H, W] slices resize 到 config 指定的 image size。"""

    if slices.shape[-2:] == (image_size, image_size):
        return slices
    return transforms.Resize(image_size, antialias=True)(slices)


def iter_healthy_slices(
    dataset_path: Path,
    subject_ids: Iterable[str],
    image_size: int,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> Iterable[tuple[str, int, torch.Tensor]]:
    """逐一產生正規化後的 healthy slice: (subject_id, slice_index, tensor)。"""

    for subject_id in subject_ids:
        yield from healthy_slices_for_subject(dataset_path, subject_id, image_size, modalities)


def healthy_slices_for_subject(
    dataset_path: Path,
    subject_id: str,
    image_size: int,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> list[tuple[str, int, torch.Tensor]]:
    images, mask = load_subject_volume(dataset_path, subject_id, modalities)
    subject_slices = []
    metadata = []
    for slice_index in range(images.shape[3]):
        first_modality = images[0, :, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        has_foreground = bool(torch.count_nonzero(first_modality).item())
        has_anomaly = bool(np.any(mask_slice >= 1))
        if has_foreground and not has_anomaly:
            subject_slices.append(images[:, :, :, slice_index])
            metadata.append((subject_id, slice_index))

    if not subject_slices:
        return []

    resized = resize_slices(torch.stack(subject_slices, dim=0), image_size)
    return [
        (current_subject_id, slice_index, slice_tensor)
        for (current_subject_id, slice_index), slice_tensor in zip(metadata, resized)
    ]


def healthy_slice_candidates_for_subject(
    dataset_path: Path,
    subject_id: str,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> tuple[list[HealthySliceCandidate], set[int]]:
    images, mask = load_subject_volume(dataset_path, subject_id, modalities)
    subject_dir = dataset_path / subject_id
    modality_paths = tuple(str(subject_dir / f"{subject_id}_{modality}.nii.gz") for modality in modalities)
    z_indices_seen = set(range(images.shape[3]))
    candidates = []
    for slice_index in range(images.shape[3]):
        first_modality = images[0, :, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        has_foreground = bool(torch.count_nonzero(first_modality).item())
        has_anomaly = bool(np.any(mask_slice > 0))
        if has_foreground and not has_anomaly:
            candidates.append(
                HealthySliceCandidate(
                    subject_id=subject_id,
                    z=slice_index,
                    modality_paths=modality_paths,
                )
            )
    return candidates, z_indices_seen


def balance_candidates_by_z(
    candidates_by_z: dict[int, list[HealthySliceCandidate]],
    per_z_count: int,
    balance_seed: int,
) -> tuple[list[BalancedSliceRecord], dict[int, dict[str, int]]]:
    if per_z_count <= 0:
        raise ValueError("--per-z-count must be a positive integer.")

    rng = np.random.default_rng(balance_seed)
    records = []
    z_summary = {}
    for z in sorted(candidates_by_z):
        candidates = candidates_by_z[z]
        original_count = len(candidates)
        if original_count == 0:
            continue

        z_records = []
        if original_count > per_z_count:
            sampled_indices = rng.choice(original_count, size=per_z_count, replace=False).tolist()
            z_records.extend(
                BalancedSliceRecord(
                    candidate=candidates[index],
                    balance_z_count=original_count,
                    sampled_index_in_z=int(index),
                    duplicate_copy_id=0,
                    is_extra_sample=False,
                )
                for index in sampled_indices
            )
        else:
            full_repeats = per_z_count // original_count
            remainder = per_z_count % original_count
            for repeat_id in range(full_repeats):
                z_records.extend(
                    BalancedSliceRecord(
                        candidate=candidate,
                        balance_z_count=original_count,
                        sampled_index_in_z=index,
                        duplicate_copy_id=repeat_id,
                        is_extra_sample=False,
                    )
                    for index, candidate in enumerate(candidates)
                )
            if remainder:
                sampled_indices = rng.choice(original_count, size=remainder, replace=False).tolist()
                z_records.extend(
                    BalancedSliceRecord(
                        candidate=candidates[index],
                        balance_z_count=original_count,
                        sampled_index_in_z=int(index),
                        duplicate_copy_id=full_repeats,
                        is_extra_sample=True,
                    )
                    for index in sampled_indices
                )

        records.extend(z_records)
        z_summary[z] = {"original": original_count, "output": len(z_records)}

    return records, z_summary


def estimate_lmdb_map_size(dataset_path: Path, subject_ids: Iterable[str], progress: bool = False) -> int:
    """使用原版 ANDi heuristic 估計 LMDB map size。"""

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


def build_kfold_subject_splits(
    input_csv: str | Path,
    folds: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    input_csv = Path(input_csv)
    df = read_subject_csv(input_csv, label="Input CSV")
    if df.empty:
        raise ValueError(f"Input CSV contains no subjects: {input_csv}")
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    if folds > len(df):
        raise ValueError(f"--folds ({folds}) cannot exceed subject count ({len(df)}).")

    positions = np.arange(len(df))
    fold_positions = positions.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(fold_positions)

    splits = []
    for fold_index, val_positions in enumerate(np.array_split(fold_positions, folds)):
        val_positions = np.sort(val_positions)
        train_positions = np.setdiff1d(positions, val_positions, assume_unique=True)
        train_df = df.iloc[train_positions].reset_index(drop=True)
        val_df = df.iloc[val_positions].reset_index(drop=True)
        splits.append((fold_index, train_df, val_df))
    return splits


def read_subject_csv(input_csv: str | Path, label: str) -> pd.DataFrame:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"{label} contains no subjects: {input_csv}")

    subject_column = df.columns[0]
    duplicate_mask = df[subject_column].astype(str).duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(df.loc[duplicate_mask, subject_column].astype(str).unique().tolist())
        preview = ", ".join(duplicates[:10])
        if len(duplicates) > 10:
            preview += ", ..."
        raise ValueError(
            f"{label} has duplicate subject ids in first column '{subject_column}': {preview}"
        )
    return df


def build_repeated_train_test_splits(
    train_csv: str | Path,
    test_csv: str | Path,
    folds: int = 5,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    if folds < 1:
        raise ValueError("--folds must be at least 1.")

    train_df = read_subject_csv(train_csv, label="Train CSV")
    test_df = read_subject_csv(test_csv, label="Test CSV")
    return [
        (fold_index, train_df.copy().reset_index(drop=True), test_df.copy().reset_index(drop=True))
        for fold_index in range(folds)
    ]


def build_combined_train_test_splits(
    train_csv: str | Path,
    test_csv: str | Path,
    folds: int = 5,
    seed: int = 42,
    test_size: int | None = None,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2.")

    train_df = read_subject_csv(train_csv, label="Train CSV")
    test_df = read_subject_csv(test_csv, label="Test CSV")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    subject_column = combined_df.columns[0]
    duplicate_mask = combined_df[subject_column].astype(str).duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(combined_df.loc[duplicate_mask, subject_column].astype(str).unique().tolist())
        preview = ", ".join(duplicates[:10])
        if len(duplicates) > 10:
            preview += ", ..."
        raise ValueError(
            f"Combined train/test CSVs have duplicate subject ids in first column "
            f"'{subject_column}': {preview}"
        )

    total_subjects = len(combined_df)
    resolved_test_size = int(test_size or len(test_df))
    if resolved_test_size <= 0:
        raise ValueError("--combined-test-size must be positive.")
    if resolved_test_size >= total_subjects:
        raise ValueError(
            f"--combined-test-size ({resolved_test_size}) must be smaller than total subjects "
            f"({total_subjects})."
        )

    rng = np.random.default_rng(seed)
    positions = np.arange(total_subjects)
    rng.shuffle(positions)

    splits = []
    all_positions = set(range(total_subjects))
    for fold_index in range(folds):
        start = (fold_index * resolved_test_size) % total_subjects
        end = start + resolved_test_size
        if end <= total_subjects:
            test_positions = positions[start:end]
        else:
            test_positions = np.concatenate([positions[start:], positions[: end - total_subjects]])
        test_positions = np.sort(test_positions)
        train_positions = np.array(sorted(all_positions.difference(test_positions.tolist())))
        train_fold_df = combined_df.iloc[train_positions].reset_index(drop=True)
        test_fold_df = combined_df.iloc[test_positions].reset_index(drop=True)
        splits.append((fold_index, train_fold_df, test_fold_df))
    return splits


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
    """建立包含 healthy slices 的 LMDB，並輸出 slice index CSV。"""

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
