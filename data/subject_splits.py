"""Subject CSV validation and deterministic fold/window split policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


__all__ = [
    "build_combined_train_test_splits",
    "build_kfold_subject_splits",
    "build_repeated_train_test_splits",
    "read_subject_csv",
]
