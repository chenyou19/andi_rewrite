"""Create a reproducible 20-session FOMO45K plus one BraTS21 FLAIR montage."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


FOMO_ROOT = Path(r"C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH")
BRATS_IMAGE = Path(
    r"C:\ML\data\BraTS_2021\BraTS2021_00658\BraTS2021_00658_flair.nii.gz"
)
OUTPUT_DIR = Path(
    r"C:\ML\andi_test\Test\andi_rewrite\artifacts\fomo45k_brats21"
)
OUTPUT_IMAGE = OUTPUT_DIR / "random-20-fomo-plus-brats21-flair.png"
OUTPUT_SELECTION = OUTPUT_DIR / "random-20-fomo-plus-brats21-selection.json"
RANDOM_SEED = 73
SLICE_INDEX = 74
EXPECTED_SHAPE = (240, 240, 155)


def _fomo_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for status_path in sorted(FOMO_ROOT.glob("sub_*/ses_*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS":
            continue
        flair_name = status.get("outputs", {}).get("flair")
        if not flair_name:
            continue
        flair_path = status_path.parent / flair_name
        if not flair_path.is_file():
            continue
        candidates.append(
            {
                "case_id": status["case_id"],
                "path": flair_path,
                "dice": float(status["qc"]["atlas_mask_dice"]),
            }
        )
    return candidates


def _normalized_slice(path: Path) -> tuple[np.ma.MaskedArray, np.ndarray]:
    image = nib.load(str(path))
    if image.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape for {path}: {image.shape}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    mask = np.isfinite(data) & (data != 0)
    foreground = data[mask]
    if foreground.size == 0:
        raise ValueError(f"Empty foreground: {path}")
    std = foreground.std(dtype=np.float64)
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid foreground standard deviation: {path}")
    data[mask] = (foreground - foreground.mean(dtype=np.float64)) / std
    axial = data[:, :, SLICE_INDEX].T
    axial_mask = mask[:, :, SLICE_INDEX].T
    return np.ma.masked_where(~axial_mask, axial), image.affine


def main() -> None:
    candidates = _fomo_candidates()
    if len(candidates) < 20:
        raise RuntimeError(f"Need at least 20 PASS FOMO sessions; found {len(candidates)}")
    selected = random.Random(RANDOM_SEED).sample(candidates, 20)

    brats_slice, reference_affine = _normalized_slice(BRATS_IMAGE)
    panels: list[tuple[np.ma.MaskedArray, str, str, bool]] = []
    for index, item in enumerate(selected, start=1):
        value, affine = _normalized_slice(item["path"])
        if not np.allclose(affine, reference_affine, atol=1e-6, rtol=0):
            raise ValueError(f"Affine differs from BraTS21: {item['path']}")
        panels.append(
            (
                value,
                f"{index:02d}  {item['case_id']}",
                f"Dice {item['dice']:.3f}",
                False,
            )
        )
    panels.append((brats_slice, "21  BraTS2021_00658", "BraTS21 reference", True))

    font_path = Path(r"C:\Windows\Fonts\msjh.ttc")
    title_font = fm.FontProperties(fname=str(font_path), size=22) if font_path.is_file() else None
    label_font = fm.FontProperties(fname=str(font_path), size=10) if font_path.is_file() else None
    note_font = fm.FontProperties(fname=str(font_path), size=11) if font_path.is_file() else None
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")

    figure, axes = plt.subplots(3, 7, figsize=(21, 10.5), facecolor="#111111")
    figure.suptitle(
        "FOMO45K 隨機 20 sessions vs BraTS21：Axial FLAIR",
        color="white",
        y=0.975,
        fontproperties=title_font,
    )
    for axis, (value, label, detail, is_brats) in zip(axes.flat, panels):
        axis.imshow(value, cmap=cmap, origin="lower", vmin=-1.5, vmax=3.0)
        title_color = "#ffd166" if is_brats else "white"
        axis.set_title(
            f"{label}\n{detail}",
            color=title_color,
            pad=5,
            fontproperties=label_font,
        )
        axis.set_facecolor("black")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(is_brats)
            spine.set_color("#ffd166")
            spine.set_linewidth(3)

    figure.text(
        0.5,
        0.018,
        "固定 seed=73；相同 SRI24 grid 240×240×155、1 mm³、LPS；FLAIR z=74；腦區 z-score；顯示範圍 -1.5～3.0",
        ha="center",
        color="#dddddd",
        fontproperties=note_font,
    )
    figure.subplots_adjust(left=0.018, right=0.982, bottom=0.055, top=0.91, wspace=0.035, hspace=0.19)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_IMAGE, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)

    selection = {
        "random_seed": RANDOM_SEED,
        "slice_index": SLICE_INDEX,
        "modality": "flair",
        "fomo_candidates": len(candidates),
        "selected": [
            {
                "case_id": item["case_id"],
                "path": str(item["path"]),
                "atlas_mask_dice": item["dice"],
            }
            for item in selected
        ],
        "brats21": {"case_id": "BraTS2021_00658", "path": str(BRATS_IMAGE)},
        "output_image": str(OUTPUT_IMAGE),
    }
    OUTPUT_SELECTION.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(OUTPUT_IMAGE)
    print(OUTPUT_SELECTION)


if __name__ == "__main__":
    main()
