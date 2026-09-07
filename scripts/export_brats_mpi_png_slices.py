"""Export deterministic axial PNG montages from a BraTS MPI cache.

For every subject in the input manifest, the script selects the axial slices
with the largest whole-tumour (segmentation > 0) areas.  Each output PNG is a
four-panel montage: FLAIR, T1, T2, and FLAIR with categorical GT overlay.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SUBJECT_PATTERN = re.compile(r"^BraTS2021_\d+$", re.IGNORECASE)
GT_COLORS = {
    1: np.asarray((255, 64, 64), dtype=np.float32),
    2: np.asarray((255, 215, 0), dtype=np.float32),
    4: np.asarray((0, 230, 255), dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export lesion-centred PNG slices from an MPI cache."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--mode", default="mni_affine")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices-per-subject", type=int, default=2)
    parser.add_argument("--expected-subjects", type=int, default=50)
    parser.add_argument("--expected-pngs", type=int, default=100)
    return parser.parse_args()


def subject_id_from_row(row: dict[str, str]) -> str:
    for key in ("subject_id", "subject", "case_id", "patient_id", "BraTS21ID"):
        value = (row.get(key) or "").strip()
        if SUBJECT_PATTERN.fullmatch(value):
            return value
    for value in row.values():
        candidate = (value or "").strip()
        if SUBJECT_PATTERN.fullmatch(candidate):
            return candidate
    raise ValueError(f"Cannot find a BraTS2021 subject ID in manifest row: {row}")


def read_subjects(manifest: Path) -> list[str]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = [subject_id_from_row(row) for row in rows]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Manifest contains duplicate subject IDs")
    return subjects


def index_cache(cache_root: Path, mode: str, subjects: list[str]) -> dict[str, Path]:
    wanted = {subject.casefold(): subject for subject in subjects}
    matches: dict[str, list[Path]] = {subject: [] for subject in subjects}
    for path in cache_root.rglob("volume.npz"):
        path_text = path.as_posix().casefold()
        if mode.casefold() not in path_text:
            continue
        for folded, subject in wanted.items():
            if folded in path_text:
                matches[subject].append(path)
                break

    result: dict[str, Path] = {}
    for subject, paths in matches.items():
        if len(paths) != 1:
            rendered = ", ".join(str(path) for path in paths) or "none"
            raise FileNotFoundError(
                f"Expected one {mode} volume.npz for {subject}, found: {rendered}"
            )
        result[subject] = paths[0]
    return result


def to_uint8(slice_2d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(slice_2d, dtype=np.float32)
    valid = valid_mask & np.isfinite(values)
    samples = values[valid]
    if samples.size == 0:
        samples = values[np.isfinite(values)]
    if samples.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(samples, (1.0, 99.0))
    if high <= low:
        low, high = float(samples.min()), float(samples.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8)


def rgb_grayscale(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def gt_overlay(flair_gray: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    result = rgb_grayscale(flair_gray).astype(np.float32)
    alpha = 0.58
    for label in np.unique(segmentation):
        label_int = int(label)
        if label_int == 0:
            continue
        color = GT_COLORS.get(
            label_int, np.asarray((255, 0, 255), dtype=np.float32)
        )
        mask = segmentation == label_int
        result[mask] = (1.0 - alpha) * result[mask] + alpha * color
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def make_montage(
    image: np.ndarray,
    segmentation: np.ndarray,
    brain_mask: np.ndarray,
    z_index: int,
    subject: str,
    lesion_voxels: int,
) -> Image.Image:
    if image.ndim != 4 or image.shape[0] < 3:
        raise ValueError(f"Unexpected image shape for {subject}: {image.shape}")
    valid = np.asarray(brain_mask[:, :, z_index], dtype=bool)
    gray = [to_uint8(image[channel, :, :, z_index], valid) for channel in range(3)]
    seg_slice = np.asarray(segmentation[:, :, z_index])
    panels = [rgb_grayscale(channel) for channel in gray]
    panels.append(gt_overlay(gray[0], seg_slice))

    panel_height, panel_width = gray[0].shape
    header_height = 22
    footer_height = 22
    canvas = Image.new(
        "RGB", (panel_width * 4, header_height + panel_height + footer_height), "black"
    )
    draw = ImageDraw.Draw(canvas)
    titles = ("FLAIR", "T1", "T2", "FLAIR + GT")
    for index, (title, panel) in enumerate(zip(titles, panels, strict=True)):
        x = index * panel_width
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header_height))
        draw.text((x + 5, 5), title, fill="white")
        if index:
            draw.line((x, 0, x, canvas.height), fill=(80, 80, 80))

    labels = ",".join(str(int(v)) for v in np.unique(seg_slice) if int(v) != 0)
    footer = (
        f"{subject}   axial z={z_index}   lesion={lesion_voxels} px   "
        f"GT labels={labels or 'none'}"
    )
    draw.text((5, header_height + panel_height + 5), footer, fill="white")
    return canvas


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"image", "segmentation"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        image = np.asarray(archive["image"])
        segmentation = np.asarray(archive["segmentation"])
        if "brain_mask" in archive.files:
            brain_mask = np.asarray(archive["brain_mask"], dtype=bool)
        else:
            brain_mask = np.any(np.isfinite(image) & (image != 0), axis=0)
    if image.shape[1:] != segmentation.shape:
        raise ValueError(
            f"Image/segmentation mismatch in {path}: {image.shape} vs {segmentation.shape}"
        )
    if brain_mask.shape != segmentation.shape:
        raise ValueError(
            f"Brain-mask/segmentation mismatch in {path}: "
            f"{brain_mask.shape} vs {segmentation.shape}"
        )
    return image, segmentation, brain_mask


def export(args: argparse.Namespace) -> None:
    if args.slices_per_subject < 1:
        raise ValueError("--slices-per-subject must be positive")
    manifest = args.manifest.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    subjects = read_subjects(manifest)
    if len(subjects) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} subjects, found {len(subjects)}"
        )
    expected_from_selection = len(subjects) * args.slices_per_subject
    if expected_from_selection != args.expected_pngs:
        raise ValueError(
            f"Selection produces {expected_from_selection} PNGs, "
            f"not --expected-pngs={args.expected_pngs}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cache_paths = index_cache(cache_root, args.mode, subjects)
    stage = Path(tempfile.mkdtemp(prefix=".brats_slice_export_", dir=output_dir.parent))
    records: list[dict[str, object]] = []
    ordinal = 0
    try:
        for subject_index, subject in enumerate(subjects, start=1):
            image, segmentation, brain_mask = load_volume(cache_paths[subject])
            lesion_areas = np.count_nonzero(segmentation > 0, axis=(0, 1))
            nonempty = np.flatnonzero(lesion_areas > 0).tolist()
            if len(nonempty) < args.slices_per_subject:
                raise ValueError(
                    f"{subject} has only {len(nonempty)} lesion-containing slices"
                )
            selected = sorted(
                nonempty, key=lambda z: (-int(lesion_areas[z]), int(z))
            )[: args.slices_per_subject]
            for rank, z_index in enumerate(selected, start=1):
                ordinal += 1
                lesion_voxels = int(lesion_areas[z_index])
                labels = sorted(
                    int(value)
                    for value in np.unique(segmentation[:, :, z_index])
                    if int(value) != 0
                )
                filename = (
                    f"{ordinal:03d}_{subject}_rank{rank}_z{z_index:03d}_"
                    f"lesion{lesion_voxels:05d}.png"
                )
                montage = make_montage(
                    image,
                    segmentation,
                    brain_mask,
                    int(z_index),
                    subject,
                    lesion_voxels,
                )
                montage.save(stage / filename, format="PNG", optimize=True)
                records.append(
                    {
                        "ordinal": ordinal,
                        "subject_order": subject_index,
                        "subject_id": subject,
                        "selection_rank": rank,
                        "mode": args.mode,
                        "axial_z_index": int(z_index),
                        "lesion_voxels": lesion_voxels,
                        "gt_labels_present": "|".join(map(str, labels)),
                        "png_file": filename,
                        "cache_volume": str(cache_paths[subject]),
                    }
                )

        fieldnames = list(records[0])
        with (stage / "selection.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        pngs = sorted(stage.glob("*.png"))
        if len(pngs) != args.expected_pngs or len(records) != args.expected_pngs:
            raise RuntimeError(
                f"Export validation failed: {len(pngs)} PNGs, {len(records)} records"
            )
        for png in pngs:
            with Image.open(png) as check:
                check.verify()

        if output_dir.exists():
            output_dir.rmdir()  # It was verified empty above.
        stage.replace(output_dir)
        print(
            f"Exported {len(pngs)} PNGs from {len(subjects)} subjects to {output_dir}"
        )
        print(f"Selection manifest: {output_dir / 'selection.csv'}")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
