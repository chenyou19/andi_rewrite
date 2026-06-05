"""Create a healthy-slice LMDB from raw BraTS volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import yaml

from andi_rewrite.data import split_healthy_to_lmdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an LMDB of healthy MRI slices for the rewritten ANDi framework."
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
        help="CSV file listing subject ids to scan.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="CSV path where healthy subject/slice metadata will be written.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Output slice resolution. Default keeps ANDi compatibility at 128.",
    )
    parser.add_argument(
        "--lmdb-dir",
        default=None,
        help="LMDB output directory. Defaults to <data_set>/healthy_slices.",
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
        help="Optional LMDB map size in bytes. By default it is estimated from subject files.",
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
        help="Seed for deterministic z-balanced downsampling and remainder sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing non-empty LMDB output directory before writing.",
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
    summary = split_healthy_to_lmdb(
        dataset_path=Path(args.data_set),
        input_csv=Path(args.input_file),
        output_csv=Path(args.output_file),
        image_size=args.resolution,
        lmdb_dir=args.lmdb_dir,
        modalities=args.modalities,
        map_size=args.map_size,
        overwrite=args.overwrite,
        progress=not args.no_progress,
        sampling_mode=sampling_mode,
        per_z_count=args.per_z_count,
        balance_seed=args.balance_seed,
    )
    print("Healthy-slice preprocessing complete:")
    print(yaml.safe_dump(summary.as_dict(), sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
