"""Create healthy-slice LMDBs for k-fold BraTS experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import yaml

from andi_rewrite.data import split_healthy_kfold_to_lmdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one healthy-slice LMDB per fold for k-fold experiments."
    )
    parser.add_argument(
        "-d",
        "--data_set",
        required=True,
        help="Dataset root containing one folder per subject.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Training CSV. Without --test-file this is split into train/validation folds.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help=(
            "Optional test CSV. By default it is copied as the fixed test set for every fold; "
            "with --combine-train-test it is combined with --input_file before splitting."
        ),
    )
    parser.add_argument(
        "--combine-train-test",
        action="store_true",
        help=(
            "Combine --input_file and --test-file first, then create new per-fold "
            "train/test splits from the combined subject pool."
        ),
    )
    parser.add_argument(
        "--combined-test-size",
        type=int,
        default=None,
        help=(
            "Test subject count per fold in --combine-train-test mode. "
            "Defaults to the row count of --test-file."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where fold subject CSVs and healthy-slice CSVs will be written.",
    )
    parser.add_argument(
        "--lmdb-root",
        required=True,
        help="Directory where per-fold LMDB directories will be written.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to create. Default: 5.",
    )
    parser.add_argument(
        "--fold-start-index",
        type=int,
        default=0,
        help="First fold index used in output directory names. Default creates fold_0...fold_4.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep input CSV order when assigning folds instead of shuffling first.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic subject fold assignment.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Output slice resolution. Default keeps ANDi compatibility at 128.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["flair", "t1", "t1ce", "t2"],
        help="Modalities to load, without leading underscores or .nii.gz suffix.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=None,
        help="Optional LMDB map size in bytes for each fold. By default each fold is estimated.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["healthy", "z_balanced"],
        default="healthy",
        help="Slice sampling mode. Default keeps the original healthy-slice behavior.",
    )
    parser.add_argument(
        "--z-balanced",
        action="store_true",
        help="Alias for --sampling-mode z_balanced.",
    )
    parser.add_argument(
        "--per-z-count",
        type=int,
        default=447,
        help="Target output slice count for each z index in z-balanced mode.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Base seed for deterministic z-balanced sampling. Fold index is added to this seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing non-empty per-fold LMDB directories before writing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and ETA output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling_mode = "z_balanced" if args.z_balanced else args.sampling_mode
    summary = split_healthy_kfold_to_lmdb(
        dataset_path=Path(args.data_set),
        input_csv=Path(args.input_file),
        output_root=Path(args.output_root),
        lmdb_root=Path(args.lmdb_root),
        test_csv=Path(args.test_file) if args.test_file else None,
        combine_train_test=args.combine_train_test,
        combined_test_size=args.combined_test_size,
        folds=args.folds,
        image_size=args.resolution,
        modalities=args.modalities,
        map_size=args.map_size,
        overwrite=args.overwrite,
        progress=not args.no_progress,
        sampling_mode=sampling_mode,
        per_z_count=args.per_z_count,
        split_seed=args.split_seed,
        balance_seed=args.balance_seed,
        shuffle=not args.no_shuffle,
        fold_start_index=args.fold_start_index,
    )
    print("K-fold healthy-slice preprocessing complete:")
    print(yaml.safe_dump(summary.as_dict(), sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
