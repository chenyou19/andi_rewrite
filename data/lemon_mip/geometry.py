"""Template-derived physical square ROI and interpolation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FixedSquareROI:
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int
    margin_mm: float
    side_mm: float
    spacing_x_mm: float
    spacing_y_mm: float
    template_shape: tuple[int, int, int]
    output_size: int

    @property
    def crop_shape(self) -> tuple[int, int]:
        return self.x_stop - self.x_start, self.y_stop - self.y_start

    @property
    def resize_scale(self) -> tuple[float, float]:
        height, width = self.crop_shape
        return self.output_size / float(height), self.output_size / float(width)

    def as_dict(self) -> dict:
        return {**asdict(self), "crop_shape": list(self.crop_shape), "resize_scale": list(self.resize_scale)}


def _centered_interval(center: float, length: int) -> tuple[int, int]:
    if length <= 0:
        raise ValueError(f"ROI interval length must be positive, got {length}.")
    start = int(round(center - length / 2.0))
    return start, start + length


def derive_fixed_square_roi(
    template_mask: np.ndarray,
    template_affine: np.ndarray,
    *,
    margin_mm: float = 8.0,
    output_size: int = 128,
) -> FixedSquareROI:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ROI derivation requires nibabel.") from exc

    mask = np.asarray(template_mask) > 0.5
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError(f"Template brain mask must be a non-empty 3D array, got {mask.shape}.")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    spacing = nib.affines.voxel_sizes(np.asarray(template_affine, dtype=np.float64))
    if len(spacing) != 3 or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid template voxel spacing: {spacing}.")

    xs, ys, _ = np.nonzero(mask)
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    # Do not clip the requested physical margin at the template boundary.
    # A fixed square may extend a few voxels outside the template FOV; the
    # resize helper pads those positions with zeros explicitly.
    x_min = x_min - int(np.ceil(margin_mm / spacing[0]))
    x_max = x_max + int(np.ceil(margin_mm / spacing[0]))
    y_min = y_min - int(np.ceil(margin_mm / spacing[1]))
    y_max = y_max + int(np.ceil(margin_mm / spacing[1]))

    width_x_mm = (x_max - x_min) * float(spacing[0])
    width_y_mm = (y_max - y_min) * float(spacing[1])
    side_mm = max(width_x_mm, width_y_mm)
    length_x = int(np.ceil(side_mm / float(spacing[0])))
    length_y = int(np.ceil(side_mm / float(spacing[1])))
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    x_start, x_stop = _centered_interval(center_x, length_x)
    y_start, y_stop = _centered_interval(center_y, length_y)
    actual_x_mm = (x_stop - x_start) * float(spacing[0])
    actual_y_mm = (y_stop - y_start) * float(spacing[1])
    tolerance = max(float(spacing[0]), float(spacing[1])) + 1.0e-6
    if abs(actual_x_mm - actual_y_mm) > tolerance:
        raise ValueError(
            "Could not construct a physically square ROI: "
            f"x={actual_x_mm:.6f} mm y={actual_y_mm:.6f} mm."
        )
    return FixedSquareROI(
        x_start=x_start,
        x_stop=x_stop,
        y_start=y_start,
        y_stop=y_stop,
        margin_mm=float(margin_mm),
        side_mm=float(max(actual_x_mm, actual_y_mm)),
        spacing_x_mm=float(spacing[0]),
        spacing_y_mm=float(spacing[1]),
        template_shape=tuple(int(value) for value in mask.shape),
        output_size=int(output_size),
    )


def resize_roi_slice(
    image: np.ndarray,
    roi: FixedSquareROI,
    *,
    is_mask: bool = False,
) -> np.ndarray:
    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ROI resizing requires scikit-image.") from exc

    value = np.asarray(image)
    if value.ndim not in (2, 3):
        raise ValueError(f"Expected [H,W] or [C,H,W], got {value.shape}.")
    spatial_shape = value.shape[-2:]
    if spatial_shape != roi.template_shape[:2]:
        raise ValueError(
            f"ROI source shape mismatch: got {spatial_shape}, expected {roi.template_shape[:2]}."
        )
    crop_height, crop_width = roi.crop_shape
    padded_shape = (*value.shape[:-2], crop_height, crop_width)
    cropped = np.zeros(padded_shape, dtype=value.dtype)
    source_x_start = max(roi.x_start, 0)
    source_x_stop = min(roi.x_stop, spatial_shape[0])
    source_y_start = max(roi.y_start, 0)
    source_y_stop = min(roi.y_stop, spatial_shape[1])
    if source_x_start >= source_x_stop or source_y_start >= source_y_stop:
        raise ValueError("Fixed ROI does not overlap the template grid.")
    destination_x_start = source_x_start - roi.x_start
    destination_x_stop = destination_x_start + (source_x_stop - source_x_start)
    destination_y_start = source_y_start - roi.y_start
    destination_y_stop = destination_y_start + (source_y_stop - source_y_start)
    cropped[..., destination_x_start:destination_x_stop, destination_y_start:destination_y_stop] = value[
        ..., source_x_start:source_x_stop, source_y_start:source_y_stop
    ]
    if value.ndim == 2:
        output_shape: Sequence[int] = (roi.output_size, roi.output_size)
    else:
        output_shape = (value.shape[0], roi.output_size, roi.output_size)
    result = resize(
        cropped,
        output_shape,
        order=0 if is_mask else 1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
        anti_aliasing=not is_mask,
    )
    if is_mask:
        return (result > 0.5).astype(np.uint8)
    return result.astype(np.float32, copy=False)
