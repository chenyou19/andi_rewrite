"""Create a two-panel, same-grid FLAIR comparison for manual visual review."""

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
    r"C:\ML\andi_test\Test\andi_rewrite\artifacts\fomo45k_brats21\fomo-vs-brats21-flair.png"
)
SLICE_INDEX = 74
DISPLAY_MIN = -1.5
DISPLAY_MAX = 3.0


def _normalized_slice(path: Path) -> tuple[np.ma.MaskedArray, nib.Nifti1Image]:
    image = nib.load(str(path))
    if image.shape != (240, 240, 155):
        raise ValueError(f"Unexpected BraTS grid: {path} shape={image.shape}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    mask = data != 0
    foreground = data[mask]
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (foreground - foreground.mean(dtype=np.float64)) / foreground.std(
        dtype=np.float64
    )
    axial = normalized[:, :, SLICE_INDEX].T
    axial_mask = mask[:, :, SLICE_INDEX].T
    return np.ma.masked_where(~axial_mask, axial), image


def main() -> None:
    left, left_image = _normalized_slice(FOMO)
    right, right_image = _normalized_slice(BRATS)
    if not np.array_equal(left_image.affine, right_image.affine):
        raise ValueError("FOMO and BraTS21 affine matrices differ")
    if left_image.header.get_zooms()[:3] != right_image.header.get_zooms()[:3]:
        raise ValueError("FOMO and BraTS21 spacing differs")

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
    panels = (
        (left, "1. FOMO45K  sub_10063 / ses_1"),
        (right, "2. BraTS21  BraTS2021_00658"),
    )
    for axis, (value, label) in zip(axes, panels):
        axis.imshow(value, cmap=cmap, origin="lower", vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
        axis.set_title(label, color="white", pad=13, fontproperties=label_font)
        axis.set_facecolor("black")
        axis.axis("off")

    figure.text(
        0.5,
        0.035,
        "相同 SRI24 grid：240×240×155、1 mm³、LPS；相同切面 z=74；非零腦區 z-score；固定顯示範圍 −1.5～3.0",
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
