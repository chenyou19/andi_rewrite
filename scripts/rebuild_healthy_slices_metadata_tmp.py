"""Temporary helper to rebuild healthy-slice metadata without rewriting LMDB."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import lmdb
import numpy as np
import pandas as pd

from andi_rewrite.data.preprocess import _load_nifti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lmdb-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    lmdb_dir = Path(args.lmdb_dir)

    frame = pd.read_csv(input_csv)
    subject_column = frame.columns[0]
    subject_ids = frame[subject_column].astype(str).tolist()

    rows = []
    for index, subject_id in enumerate(subject_ids, start=1):
        subject_dir = dataset / subject_id
        flair = _load_nifti(subject_dir / f"{subject_id}_flair.nii.gz", dtype=float)
        mask = _load_nifti(subject_dir / f"{subject_id}_seg.nii.gz", dtype=int)
        mask[mask >= 1] = 1

        for slice_index in range(flair.shape[2]):
            has_foreground = bool(np.count_nonzero(flair[:, :, slice_index]))
            has_anomaly = bool(np.any(mask[:, :, slice_index] >= 1))
            if has_foreground and not has_anomaly:
                rows.append({subject_column: subject_id, "Slice": int(slice_index)})

        if index % args.progress_every == 0 or index == len(subject_ids):
            print(
                f"{index}/{len(subject_ids)} subjects, {len(rows)} healthy slices",
                flush=True,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[subject_column, "Slice"]).to_csv(output_csv, index=False)

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        lmdb_entries = int(env.stat()["entries"])
    finally:
        env.close()

    print(f"wrote={output_csv}", flush=True)
    print(f"csv_rows={len(rows)}", flush=True)
    print(f"lmdb_entries={lmdb_entries}", flush=True)
    print(f"match={len(rows) == lmdb_entries}", flush=True)


if __name__ == "__main__":
    main()
