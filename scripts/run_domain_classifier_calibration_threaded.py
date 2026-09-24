"""Run the calibration entry point with deterministic single-threaded torch."""

from __future__ import annotations

import runpy
import sys

import torch


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
runpy.run_path(
    str(__file__).replace("_threaded.py", ".py"),
    run_name="__main__",
)
