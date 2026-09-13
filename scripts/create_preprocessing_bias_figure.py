"""Create a 2x3 T1 figure comparing three datasets before/after preprocessing.

The default subjects are real, QC-passing examples available in the local data
roots.  Native images are only reoriented for display and divided by their own
non-zero p99 so that anatomy remains visible; the original p99 is printed in
each panel.  Preprocessed model inputs are displayed with one fixed [0, 1]
window, which makes residual cross-dataset contrast differences visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


CHANNEL_ORDER = ("FLAIR", "T1", "T2")
T1_INDEX = CHANNEL_ORDER.index("T1")


@dataclass(frozen=True)
class DatasetPair:
    dataset: str
    subject: str
    native: np.ndarray
    processed: np.ndarray
    native_note: str
    processed_note: str
    details: dict[str, Any]


def _positive_p99(values: np.ndarray) -> float:
    foreground = np.asarray(values, dtype=np.float32)
    foreground = foreground[np.isfinite(foreground) & (foreground > 0)]
    if foreground.size == 0:
        raise ValueError("Cannot window an image without positive finite voxels.")
    return float(np.percentile(foreground, 99.0))


def _display_native(values: np.ndarray, p99: float) -> np.ndarray:
    return np.rot90(np.clip(np.asarray(values, dtype=np.float32) / p99, 0.0, 1.0))


def _display_processed(values: np.ndarray) -> np.ndarray:
    return np.rot90(np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0))


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    values = image.get_fdata(dtype=np.float32, caching="unchanged")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    return values, np.asarray(image.affine, dtype=np.float64), spacing


def _quantile_index(indices: np.ndarray, quantile: float) -> int:
    if indices.size == 0:
        raise ValueError("Cannot select a slice from an empty index set.")
    position = int(round(float(quantile) * (indices.size - 1)))
    return int(indices[np.clip(position, 0, indices.size - 1)])


def _mapped_native_z(
    target_voxel: np.ndarray,
    target_affine: np.ndarray,
    native_to_target_world: np.ndarray,
    native_affine: np.ndarray,
    native_depth: int,
) -> int:
    target_world = np.asarray(target_affine, dtype=np.float64) @ target_voxel
    native_world = np.linalg.inv(np.asarray(native_to_target_world, dtype=np.float64)) @ target_world
    native_voxel = np.linalg.inv(np.asarray(native_affine, dtype=np.float64)) @ native_world
    return int(np.clip(round(float(native_voxel[2])), 0, native_depth - 1))


def load_brats_pair(
    raw_root: Path,
    processed_root: Path,
    subject: str,
    quantile: float,
) -> DatasetPair:
    subject_root = processed_root / "mni_affine" / "subjects" / subject
    bundle_path = subject_root / "volume.npz"
    metadata_path = subject_root / "spatial_metadata.json"
    raw_path = raw_root / subject / f"{subject}_t1.nii.gz"

    with np.load(bundle_path) as bundle:
        processed_volume = np.asarray(bundle["image"][T1_INDEX], dtype=np.float32)
        brain_mask = np.asarray(bundle["brain_mask"], dtype=bool)
        segmentation = np.asarray(bundle["segmentation"])
    occupied = np.flatnonzero(np.any(brain_mask, axis=(0, 1)))
    processed_z = _quantile_index(occupied, quantile)
    if np.any(segmentation[:, :, processed_z]):
        lesion_free = occupied[
            np.asarray([not np.any(segmentation[:, :, z]) for z in occupied], dtype=bool)
        ]
        if lesion_free.size:
            nearest = lesion_free[np.argmin(np.abs(lesion_free - processed_z))]
            if abs(int(nearest) - processed_z) <= 12:
                processed_z = int(nearest)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    native_volume, native_affine, native_spacing = _load_canonical(raw_path)
    model_affine = np.asarray(metadata["model_affine"], dtype=np.float64)
    native_to_mni = np.asarray(metadata["native_ras_to_mni"], dtype=np.float64)
    model_center = np.asarray(
        [(processed_volume.shape[0] - 1) / 2, (processed_volume.shape[1] - 1) / 2, processed_z, 1.0]
    )
    native_z = _mapped_native_z(
        model_center,
        model_affine,
        native_to_mni,
        native_affine,
        native_volume.shape[2],
    )

    p99 = _positive_p99(native_volume)
    processed_slice = processed_volume[:, :, processed_z]
    processed_p99 = _positive_p99(processed_volume)
    lesion_pixels = int(np.count_nonzero(segmentation[:, :, processed_z]))
    mni_z = int(metadata["z_start"]) + processed_z
    return DatasetPair(
        dataset="BraTS21",
        subject=subject,
        native=_display_native(native_volume[:, :, native_z], p99),
        processed=_display_processed(processed_slice),
        native_note=(
            f"native {native_volume.shape[0]}x{native_volume.shape[1]} | "
            f"{native_spacing[0]:.1f} mm | p99={p99:.0f}"
        ),
        processed_note=f"model 128x128 | p99={processed_p99:.2f}",
        details={
            "native_path": str(raw_path),
            "native_z": native_z,
            "processed_path": str(bundle_path),
            "processed_z": processed_z,
            "mni_z": mni_z,
            "lesion_pixels_in_panel": lesion_pixels,
            "native_nonzero_p99": p99,
            "processed_nonzero_p99": processed_p99,
        },
    )


def _read_jsonl_match(path: Path, key: str, expected: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get(key) == expected:
                return row
    raise KeyError(f"No {key}={expected!r} in {path}")


def load_fomo_pair(
    raw_root: Path,
    processed_root: Path,
    participant: str,
    session: str,
    quantile: float,
) -> DatasetPair:
    identifier = f"{participant}/{session}"
    manifests = processed_root / "manifests"
    session_row = _read_jsonl_match(manifests / "sessions.jsonl", "session_identifier", identifier)
    first_z = int(session_row["first_z"])
    last_z = int(session_row["last_z"])
    target_z = int(round(first_z + quantile * (last_z - first_z)))
    split = str(session_row["split"])

    entry_path = manifests / f"{split}_entries.csv"
    selected: dict[str, str] | None = None
    best_distance: int | None = None
    with entry_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["session_identifier"] != identifier:
                continue
            distance = abs(int(row["z"]) - target_z)
            if best_distance is None or distance < best_distance:
                selected = row
                best_distance = distance
    if selected is None:
        raise KeyError(f"No LMDB entry for {identifier}")

    environment = lmdb.open(
        str(processed_root / split),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    try:
        with environment.begin(write=False) as transaction:
            encoded = transaction.get(selected["key"].encode("ascii"))
        if encoded is None:
            raise KeyError(f"Missing LMDB key {selected['key']}")
        model_slice = np.asarray(pickle.loads(encoded), dtype=np.float32)[T1_INDEX]
    finally:
        environment.close()

    raw_path = Path(session_row["source_paths"]["T1"])
    native_volume, _native_affine, native_spacing = _load_canonical(raw_path)
    native_z = int(np.clip(int(selected["z"]), 0, native_volume.shape[2] - 1))
    p99 = _positive_p99(native_volume)
    processed_p99 = float(session_row["normalization"]["T1"]["normalized_max"])

    # FOMO model arrays are stored in LPS. Undo the two discrete flips only for
    # the common RAS display convention used by the other five panels.
    model_slice_ras = model_slice[::-1, ::-1]
    return DatasetPair(
        dataset="FOMO45K",
        subject=identifier,
        native=_display_native(native_volume[:, :, native_z], p99),
        processed=_display_processed(model_slice_ras),
        native_note=(
            f"native {native_volume.shape[0]}x{native_volume.shape[1]} | "
            f"{native_spacing[0]:.1f} mm | p99={p99:.0f}"
        ),
        processed_note=f"model 128x128 | clipped <= {processed_p99:.0f}",
        details={
            "native_path": str(raw_path),
            "native_z": native_z,
            "processed_root": str(processed_root / split),
            "processed_key": selected["key"],
            "processed_z": int(selected["z"]),
            "native_nonzero_p99": p99,
            "processed_max": processed_p99,
        },
    )


def _find_lemon_record(input_csv: Path, case_id: str, session: str) -> dict[str, str]:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id and row["session"] == session:
                return row
    raise KeyError(f"No LEMON record for {case_id}/{session}")


def load_lemon_pair(
    mpi_root: Path,
    processed_root: Path,
    input_csv: Path,
    case_id: str,
    session: str,
    quantile: float,
) -> DatasetPair:
    session_root = processed_root / "processed_sessions" / case_id / session
    with np.load(session_root / "slice_bundle.npz") as bundle:
        slices = np.asarray(bundle["slices"], dtype=np.float32)
        z_indices = np.asarray(bundle["z_indices"], dtype=np.int32)
    bundle_index = int(round(quantile * (len(z_indices) - 1)))
    mni_z = int(z_indices[bundle_index])
    model_slice = slices[bundle_index, T1_INDEX]

    row = _find_lemon_record(input_csv, case_id, session)
    raw_path = mpi_root / row["t1w_path"]
    native_volume, native_affine, native_spacing = _load_canonical(raw_path)
    transforms = np.load(session_root / "world_transforms.npz")
    t1_to_mni = np.asarray(transforms["T1_to_MNI_world"], dtype=np.float64)
    transforms.close()

    template_path = processed_root / "templates" / "mni2009c" / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
    template = nib.as_closest_canonical(nib.load(str(template_path)))
    roi_payload = json.loads(
        (processed_root / "templates" / "mni2009c" / "fixed_roi.json").read_text(encoding="utf-8")
    )
    roi = roi_payload["roi"]
    x_center = (float(roi["x_start"]) + float(roi["x_stop"]) - 1.0) / 2.0
    y_center = (float(roi["y_start"]) + float(roi["y_stop"]) - 1.0) / 2.0
    native_z = _mapped_native_z(
        np.asarray([x_center, y_center, mni_z, 1.0]),
        np.asarray(template.affine, dtype=np.float64),
        t1_to_mni,
        native_affine,
        native_volume.shape[2],
    )

    p99 = _positive_p99(native_volume)
    statistics = json.loads((session_root / "intensity_statistics.json").read_text(encoding="utf-8"))
    processed_p99 = float(statistics["T1"]["masked_percentiles"]["99"])
    return DatasetPair(
        dataset="LEMON",
        subject=f"{case_id}/{session}",
        native=_display_native(native_volume[:, :, native_z], p99),
        processed=_display_processed(model_slice),
        native_note=(
            f"native {native_volume.shape[0]}x{native_volume.shape[1]} | "
            f"{native_spacing[0]:.1f} mm | p99={p99:.0f}"
        ),
        processed_note=f"model 128x128 | p99={processed_p99:.2f}",
        details={
            "native_path": str(raw_path),
            "native_z": native_z,
            "processed_path": str(session_root / "slice_bundle.npz"),
            "processed_bundle_index": bundle_index,
            "mni_z": mni_z,
            "native_nonzero_p99": p99,
            "processed_nonzero_p99": processed_p99,
        },
    )


def _annotate_panel(ax: plt.Axes, note: str) -> None:
    ax.text(
        0.5,
        0.025,
        note,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
    )


def render_figure(pairs: list[DatasetPair], output_path: Path, quantile: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 9.2), facecolor="white")
    figure.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.11, wspace=0.055, hspace=0.18)

    before_color = "#C85A17"
    after_color = "#1769AA"
    for column, pair in enumerate(pairs):
        top = axes[0, column]
        bottom = axes[1, column]
        top.imshow(pair.native, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        bottom.imshow(pair.processed, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        top.set_title(f"{pair.dataset}\n{pair.subject}", pad=10)
        _annotate_panel(top, pair.native_note)
        _annotate_panel(bottom, pair.processed_note)
        for axis, color in ((top, before_color), (bottom, after_color)):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.2)
                spine.set_edgecolor(color)

    figure.suptitle("Cross-dataset T1 image bias before and after preprocessing", fontsize=19, fontweight="bold", y=0.985)
    figure.text(
        0.54,
        0.925,
        "Same superior-inferior percentile; native anatomy is display-windowed, model inputs share a fixed [0, 1] window",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.text(0.025, 0.63, "BEFORE\n(native grid)", ha="center", va="center", rotation=90, fontsize=12, fontweight="bold", color=before_color)
    figure.text(0.025, 0.285, "AFTER\n(model input)", ha="center", va="center", rotation=90, fontsize=12, fontweight="bold", color=after_color)
    figure.text(
        0.54,
        0.045,
        (
            f"Axial T1 at approximately {quantile:.0%} of each brain's inferior-to-superior extent. "
            "All panels use a common RAS display orientation. Native p99 scaling is display-only."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessing_bias_t1_2x3.png"))
    parser.add_argument("--quantile", type=float, default=0.55)
    parser.add_argument("--brats-subject", default="BraTS2021_00645")
    parser.add_argument("--fomo-participant", default="sub_10063")
    parser.add_argument("--fomo-session", default="ses_1")
    parser.add_argument("--lemon-case", default="sub-010093")
    parser.add_argument("--lemon-session", default="ses-01")
    parser.add_argument("--brats-raw-root", type=Path, default=Path(r"C:\ML\data\BraTS_2021"))
    parser.add_argument(
        "--brats-processed-root",
        type=Path,
        default=Path(r"C:\ML\data\BraTS_2021_MPI_aligned"),
    )
    parser.add_argument(
        "--fomo-raw-root",
        type=Path,
        default=Path(r"C:\ML\data\FOMO45K_healthy_243\PT007_NIMH"),
    )
    parser.add_argument(
        "--fomo-processed-root",
        type=Path,
        default=Path(r"C:\ML\data\FOMO45K_ANDi\PT007_NIMH"),
    )
    parser.add_argument("--mpi-root", type=Path, default=Path(r"C:\ML\data\MPI"))
    parser.add_argument(
        "--lemon-processed-root",
        type=Path,
        default=Path(r"C:\ML\data\MPI\LEMON_ANDi_MIP"),
    )
    parser.add_argument(
        "--lemon-input-csv",
        type=Path,
        default=Path(r"C:\ML\data\MPI\sessions_with_T1_T2_highres_FLAIR.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0, 1].")
    pairs = [
        load_brats_pair(
            args.brats_raw_root,
            args.brats_processed_root,
            args.brats_subject,
            args.quantile,
        ),
        load_fomo_pair(
            args.fomo_raw_root,
            args.fomo_processed_root,
            args.fomo_participant,
            args.fomo_session,
            args.quantile,
        ),
        load_lemon_pair(
            args.mpi_root,
            args.lemon_processed_root,
            args.lemon_input_csv,
            args.lemon_case,
            args.lemon_session,
            args.quantile,
        ),
    ]
    output_path = args.output.resolve()
    render_figure(pairs, output_path, args.quantile)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "figure": str(output_path),
                "modality": "T1",
                "slice_quantile": args.quantile,
                "native_display": "per-volume nonzero p99, clipped to [0,1]",
                "processed_display": "fixed [0,1]",
                "datasets": {pair.dataset: pair.details for pair in pairs},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
