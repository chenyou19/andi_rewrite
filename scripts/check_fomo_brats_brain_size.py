from pathlib import Path

import nibabel as nib
import numpy as np


IMAGES = {
    "FOMO": Path(
        r"C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH\sub_10063\ses_1"
        r"\sub_10063_ses_1_flair.nii.gz"
    ),
    "BraTS": Path(
        r"C:\ML\data\BraTS_2021\BraTS2021_00658\BraTS2021_00658_flair.nii.gz"
    ),
}


def describe(name: str, path: Path, z: int = 74) -> None:
    image = np.asarray(nib.load(path).dataobj)
    mask = np.isfinite(image) & (image != 0)
    coordinates = np.argwhere(mask)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    extent = upper - lower + 1

    slice_mask = mask[:, :, z]
    slice_coordinates = np.argwhere(slice_mask)
    slice_lower = slice_coordinates.min(axis=0)
    slice_upper = slice_coordinates.max(axis=0)
    slice_extent = slice_upper - slice_lower + 1

    print(
        name,
        "3d_bbox=", lower.tolist(), upper.tolist(),
        "3d_extent=", extent.tolist(),
        "voxels=", int(mask.sum()),
        "slice_bbox=", slice_lower.tolist(), slice_upper.tolist(),
        "slice_extent=", slice_extent.tolist(),
        "slice_voxels=", int(slice_mask.sum()),
    )


for image_name, image_path in IMAGES.items():
    describe(image_name, image_path)
