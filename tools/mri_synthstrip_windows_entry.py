"""Run the official SynthStrip script with a narrow Win64 surfa dtype fix.

The conda-forge Win64 surfa extension declares ``target_shape`` as ``np.intp``
while ``ImageGeometry.shape`` is stored as ``int32``.  Values are unchanged;
only the integer dtype crossing the compiled interpolation boundary is cast.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import surfa.image.framed as framed


_ORIGINAL_INTERPOLATE = framed.interpolate


def _interpolate_intp_shape(*args, **kwargs):
    if "target_shape" in kwargs:
        kwargs["target_shape"] = np.asarray(kwargs["target_shape"], dtype=np.intp)
    elif len(args) >= 2:
        args = list(args)
        args[1] = np.asarray(args[1], dtype=np.intp)
    return _ORIGINAL_INTERPOLATE(*args, **kwargs)


framed.interpolate = _interpolate_intp_shape
runpy.run_path(str(Path(__file__).with_name("mri_synthstrip.py")), run_name="__main__")
