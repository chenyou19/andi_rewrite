"""Compute empirical MRI amplitude spectra from a healthy-slice LMDB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute full 2D and radial empirical spectra from an LMDB of healthy MRI slices."
    )
    parser.add_argument("--lmdb-path", required=True, help="Path to healthy-slice LMDB.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--mask-mode", default="union_nonzero", choices=["union_nonzero"])
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--crop-margin", type=int, default=4)
    parser.add_argument("--window", default="hann", choices=["none", "hann"])
    parser.add_argument("--radial-bins", type=int, help="Number of radial bins. Defaults to image_size // 2.")
    parser.add_argument("--max-slices", type=int, help="Optional maximum number of LMDB entries to inspect.")
    parser.add_argument(
        "--channel-order",
        nargs="+",
        help="Semantic source channel order, for example FLAIR T1 T2.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional LMDB entry manifest used to prove source split/session membership.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def iter_lmdb_values(lmdb_path: Path) -> tuple[int, Iterable[bytes]]:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("compute_lmdb_spectrum requires the optional 'lmdb' package.") from exc

    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        entries = int(txn.stat()["entries"])
    if entries <= 0:
        env.close()
        raise ValueError(f"LMDB contains no entries: {lmdb_path}")

    def _values() -> Iterable[bytes]:
        try:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    for _, value in cursor:
                        yield value
        finally:
            env.close()

    return entries, _values()


def maybe_progress(iterator: Iterable[bytes], total: int, enabled: bool) -> Iterable[bytes]:
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm
    except ImportError:
        return iterator
    return tqdm(iterator, total=total, unit="slice")


def foreground_bbox(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    height, width = mask.shape
    y_min = max(int(ys.min()) - margin, 0)
    y_max = min(int(ys.max()) + margin + 1, height)
    x_min = max(int(xs.min()) - margin, 0)
    x_max = min(int(xs.max()) + margin + 1, width)
    return y_min, y_max, x_min, x_max


def resize_crop(
    x: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise ImportError("compute_lmdb_spectrum requires scikit-image for crop resizing.") from exc

    y_min, y_max, x_min, x_max = bbox
    channels = x.shape[0]
    height, width = output_shape
    resized = np.empty((channels, height, width), dtype=np.float32)

    for channel in range(channels):
        resized[channel] = resize(
            x[channel, y_min:y_max, x_min:x_max],
            output_shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)

    resized_mask = resize(
        mask[y_min:y_max, x_min:x_max].astype(np.float32),
        output_shape,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized, resized_mask > 0.25


def build_centered_radius_bins(height: int, width: int, radial_bins: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    radius = np.sqrt((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2)
    max_radius = float(radius.max())
    if max_radius <= 0:
        bin_index = np.zeros((height, width), dtype=np.int64)
    else:
        bin_index = np.floor(radius / max_radius * radial_bins).astype(np.int64)
        bin_index = np.clip(bin_index, 0, radial_bins - 1)
    counts = np.bincount(bin_index.ravel(), minlength=radial_bins).astype(np.float64)
    return bin_index, counts


def radial_mean(image: np.ndarray, bin_index: np.ndarray, counts: np.ndarray, radial_bins: int) -> np.ndarray:
    sums = np.bincount(bin_index.ravel(), weights=image.ravel(), minlength=radial_bins)
    return sums / np.maximum(counts, 1.0)


def validate_slice(value: bytes, index: int) -> np.ndarray:
    x = np.asarray(pickle.loads(value))
    if x.ndim != 3:
        raise ValueError(f"LMDB entry {index} has shape {x.shape}; expected [C, H, W].")
    x = x.astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"LMDB entry {index} contains NaN or Inf.")
    return x


def source_manifest_provenance(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    split_counts: dict[str, int] = {}
    subject_ids: set[str] = set()
    session_ids: set[str] = set()
    subject_column = next(
        (name for name in ("participant_id", "case_id", "subject_id") if name in fieldnames),
        fieldnames[0] if fieldnames else None,
    )
    for row in rows:
        split = str(row.get("split", "")).strip()
        if split:
            split_counts[split] = split_counts.get(split, 0) + 1
        if subject_column and row.get(subject_column):
            subject_ids.add(str(row[subject_column]))
        identifier = str(row.get("session_identifier", "")).strip()
        if not identifier and subject_column:
            session = str(row.get("session_id", row.get("session", ""))).strip()
            identifier = f"{row.get(subject_column, '')}/{session}" if session else ""
        if identifier:
            session_ids.add(identifier)
    return {
        "sha256": digest,
        "rows": len(rows),
        "columns": fieldnames,
        "split_counts": dict(sorted(split_counts.items())),
        "subject_count": len(subject_ids),
        "session_count": len(session_ids),
    }


def compute(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lmdb_path = Path(args.lmdb_path)
    total_entries, values = iter_lmdb_values(lmdb_path)
    if args.max_slices is not None:
        total_entries = min(total_entries, int(args.max_slices))

    sum_amplitude = None
    sum_power = None
    radial_amplitude_sum = None
    radial_power_sum = None
    radial_counts = None
    radius_bin_index = None
    window = None
    used = 0
    skipped = 0
    seen = 0
    channels = height = width = radial_bin_count = None

    for raw_value in maybe_progress(values, total_entries, enabled=not args.no_progress):
        if args.max_slices is not None and seen >= args.max_slices:
            break
        x = validate_slice(raw_value, seen)
        seen += 1

        if channels is None:
            channels, height, width = map(int, x.shape)
            radial_bin_count = int(args.radial_bins or (min(height, width) // 2))
            if radial_bin_count <= 0:
                raise ValueError("--radial-bins must be positive.")
            sum_amplitude = np.zeros((channels, height, width), dtype=np.float64)
            sum_power = np.zeros((channels, height, width), dtype=np.float64)
            radial_amplitude_sum = np.zeros((channels, radial_bin_count), dtype=np.float64)
            radial_power_sum = np.zeros((channels, radial_bin_count), dtype=np.float64)
            radius_bin_index, radial_counts = build_centered_radius_bins(height, width, radial_bin_count)
            if args.window == "hann":
                wy = np.hanning(height).astype(np.float32)
                wx = np.hanning(width).astype(np.float32)
                window = wy[:, None] * wx[None, :]
        elif x.shape != (channels, height, width):
            raise ValueError(
                f"LMDB entry {seen - 1} has shape {x.shape}; expected {(channels, height, width)}."
            )

        mask = np.sum(np.abs(x), axis=0) > float(args.eps)
        bbox = foreground_bbox(mask, int(args.crop_margin))
        if bbox is None:
            skipped += 1
            continue

        crop, valid_mask = resize_crop(x, mask, bbox, (height, width))
        for channel in range(channels):
            image = crop[channel]
            if np.any(valid_mask):
                image = image - float(image[valid_mask].mean())
            else:
                image = image - float(image.mean())
            if window is not None:
                image = image * window

            fft = np.fft.fftshift(np.fft.fft2(image))
            amplitude = np.abs(fft)
            power = amplitude**2
            sum_amplitude[channel] += amplitude
            sum_power[channel] += power
            radial_amplitude_sum[channel] += radial_mean(amplitude, radius_bin_index, radial_counts, radial_bin_count)
            radial_power_sum[channel] += radial_mean(power, radius_bin_index, radial_counts, radial_bin_count)
        used += 1

    if seen == 0:
        raise ValueError(f"LMDB yielded no readable entries: {lmdb_path}")
    if used == 0:
        raise ValueError(f"No non-empty foreground slices found in LMDB: {lmdb_path}")

    channel_order = list(args.channel_order or [f"channel_{index}" for index in range(int(channels))])
    if len(channel_order) != int(channels):
        raise ValueError(
            f"--channel-order has {len(channel_order)} values but LMDB entries have C={channels}."
        )
    source_manifest = Path(args.source_manifest).resolve() if args.source_manifest else None
    if source_manifest is not None and not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest}")
    manifest_provenance = (
        source_manifest_provenance(source_manifest) if source_manifest is not None else None
    )
    if (
        manifest_provenance is not None
        and args.max_slices is None
        and int(manifest_provenance["rows"]) != seen
    ):
        raise ValueError(
            "Source manifest row count does not match the fully inspected LMDB: "
            f"{manifest_provenance['rows']} != {seen}."
        )
    if (
        manifest_provenance is not None
        and lmdb_path.name.lower() == "train"
        and manifest_provenance["split_counts"]
        and set(manifest_provenance["split_counts"]) != {"train"}
    ):
        raise ValueError(
            "Train LMDB source manifest contains non-train rows: "
            f"{manifest_provenance['split_counts']}."
        )

    mean_amplitude = (sum_amplitude / used).astype(np.float32)
    mean_power = (sum_power / used).astype(np.float32)
    radial_amplitude = (radial_amplitude_sum / used).astype(np.float32)
    radial_power = (radial_power_sum / used).astype(np.float32)

    np.savez_compressed(
        out_path,
        mean_amplitude=mean_amplitude,
        mean_power=mean_power,
        radial_amplitude=radial_amplitude,
        radial_power=radial_power,
        radial_counts=radial_counts.astype(np.int64),
        num_slices_used=np.array(used, dtype=np.int64),
        num_slices_skipped=np.array(skipped, dtype=np.int64),
        channels=np.array(channels, dtype=np.int64),
        height=np.array(height, dtype=np.int64),
        width=np.array(width, dtype=np.int64),
        radial_bins=np.array(radial_bin_count, dtype=np.int64),
        mask_mode=np.array(args.mask_mode),
        eps=np.array(float(args.eps), dtype=np.float64),
        crop_margin=np.array(int(args.crop_margin), dtype=np.int64),
        window=np.array(args.window),
        channel_order=np.asarray(channel_order),
    )

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    try:
        repository = Path(__file__).resolve().parents[1]
        git_commit = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    metadata = {
        "source_lmdb": str(lmdb_path.resolve()),
        "source_lmdb_entry_count": int(seen),
        "source_manifest": str(source_manifest) if source_manifest is not None else None,
        "source_manifest_provenance": manifest_provenance,
        "channel_order": channel_order,
        "output_npz": str(out_path.resolve()),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "python_executable": sys.executable,
        "git_commit": git_commit,
        "window": args.window,
        "crop_margin": int(args.crop_margin),
        "radial_bins": int(radial_bin_count),
        "max_slices": args.max_slices,
        "num_slices_used": int(used),
        "num_slices_skipped": int(skipped),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "npz_sha256": digest,
    }
    sidecar_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Empirical spectrum summary:")
    print(f"  used slices: {used}")
    print(f"  skipped empty slices: {skipped}")
    print(f"  shape: C={channels}, H={height}, W={width}")
    print(f"  radial bins: {radial_bin_count}")
    print(f"  output: {out_path}")
    print(f"  metadata: {sidecar_path}")
    print(f"  SHA256: {digest}")


def main() -> None:
    compute(parse_args())


if __name__ == "__main__":
    main()
