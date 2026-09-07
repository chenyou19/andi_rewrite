"""Run official SynthStrip with surfa's Win64 target-shape dtype corrected.

This reproduces ``surfa.image.interp.interpolate`` from surfa 0.6.3, changing
only its internal ``shape`` cast from C ``int`` (32-bit on Windows) to
``np.intp`` (64-bit on Win64), as required by the compiled Cython signature.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import numpy as np
import surfa.image.framed as framed


_interp_module = importlib.import_module("surfa.image.interp")


def _interpolate_win64(source, target_shape, method, affine=None, disp=None, fill=0):
    if affine is None and disp is None:
        raise ValueError("interpolation requires an affine transform and/or displacement field")
    if method not in ("linear", "nearest"):
        raise ValueError(f"interp method must be linear or nearest, got {method}")
    if not isinstance(source, np.ndarray):
        raise ValueError(f"source data must be a numpy array, got {source.__class__.__name__}")
    if source.ndim != 4:
        raise ValueError(f"source data must be 4D, but got input of shape {source.shape}")

    target_shape = tuple(target_shape)
    if len(target_shape) != 3:
        raise ValueError(f"interpolated target shape must be 3D, but got {target_shape}")

    use_affine = affine is not None
    if use_affine:
        if not isinstance(affine, np.ndarray):
            raise ValueError(f"affine must be a numpy array, got {affine.__class__.__name__}")
        if not np.array_equal(affine.shape, (4, 4)):
            raise ValueError(f"affine must be 4x4, but got input of shape {affine.shape}")
        affine = affine.astype(np.float32, copy=False)

    use_disp = disp is not None
    if use_disp:
        if not isinstance(disp, np.ndarray):
            raise ValueError(f"source data must be a numpy array, got {disp.__class__.__name__}")
        if not np.array_equal(disp.shape[:-1], target_shape):
            raise ValueError(f"warp shape {disp.shape[:-1]} must match target shape {target_shape}")
        if not disp.flags.c_contiguous and not disp.flags.f_contiguous:
            disp = np.asarray(disp, order="F")
        order = "F" if disp.flags.f_contiguous else "C"
        source = np.asarray(source, order=order)
        disp = np.asarray(disp, dtype=np.float32)
    elif not source.flags.c_contiguous and not source.flags.f_contiguous:
        source = np.asarray(source, order="F")

    order = "contiguous" if source.flags.c_contiguous else "fortran"
    interp_func = getattr(_interp_module, f"interp_3d_{order}_{method}")
    shape = np.asarray(target_shape, dtype=np.intp)

    swap_byteorder = ">" if sys.byteorder == "little" else "<"
    if source.dtype.byteorder == swap_byteorder:
        source = source.byteswap().newbyteorder()

    unsupported_dtype = None
    if source.dtype == np.dtype(bool):
        unsupported_dtype = source.dtype
        source = source.astype(np.float32)

    resampled = interp_func(source, shape, affine, disp, fill, use_affine, use_disp)
    if method == "nearest" and unsupported_dtype is not None:
        resampled = resampled.astype(unsupported_dtype)
    return resampled


framed.interpolate = _interpolate_win64
runpy.run_path(str(Path(__file__).with_name("mri_synthstrip.py")), run_name="__main__")
