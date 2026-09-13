"""Render the final two-panel FOMO45K versus BraTS21 comparison image."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


FOMO = Path(
    r"C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH\sub_10063\ses_1\sub_10063_ses_1_flair.nii.gz"
)
BRATS = Path(r"C:\ML\data\BraTS_2021\BraTS2021_00658\BraTS2021_00658_flair.nii.gz")
OUTPUT = Path(
    r"C:\ML\andi_test\Test\andi_rewrite\artifacts\fomo45k_brats21\fomo-vs-brats21-flair-final.png"
)
SLICE_INDEX = 74


def _slice(path: Path) -> tuple[np.ma.MaskedArray, nib.Nifti1Image]:
    image = nib.load(str(path))
    if image.shape != (240, 240, 155):
        raise ValueError(f"Unexpected shape for {path}: {image.shape}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    mask = data != 0
    foreground = data[mask]
    data[mask] = (foreground - foreground.mean(dtype=np.float64)) / foreground.std(
        dtype=np.float64
    )
    axial = data[:, :, SLICE_INDEX].T
    return np.ma.masked_where(~mask[:, :, SLICE_INDEX].T, axial), image


def main() -> None:
    left, left_image = _slice(FOMO)
    right, right_image = _slice(BRATS)
    if not np.array_equal(left_image.affine, right_image.affine):
        raise ValueError("The two affine matrices differ")

    font_path = Path(r"C:\Windows\Fonts\msjh.ttc")
    title_font = fm.FontProperties(fname=str(font_path), size=22) if font_path.is_file() else None
    label_font = fm.FontProperties(fname=str(font_path), size=17) if font_path.is_file() else None
    note_font = fm.FontProperties(fname=str(font_path), size=13) if font_path.is_file() else None
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")

    figure, axes = plt.subplots(1, 2, figsize=(16, 8.6), facecolor="#111111")
    figure.suptitle(
        "FOMO45K 配準結果 vs BraTS21：Axial FLAIR",
        color="white",
        y=0.965,
        fontproperties=title_font,
    )
    for axis, value, label in (
        (axes[0], left, "1. FOMO45K  sub_10063 / ses_1"),
        (axes[1], right, "2. BraTS21  BraTS2021_00658"),
    ):
        axis.imshow(value, cmap=cmap, origin="lower", vmin=-1.5, vmax=3.0)
        axis.set_title(label, color="white", pad=13, fontproperties=label_font)
        axis.set_facecolor("black")
        axis.axis("off")

    figure.text(
        0.5,
        0.035,
        "相同 SRI24 grid：240×240×155、1 mm³、LPS；相同切面 z=74；非零腦區 z-score；固定顯示範圍 -1.5～3.0",
        ha="center",
        color="#dddddd",
        fontproperties=note_font,
    )
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.08, top=0.89, wspace=0.06)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
