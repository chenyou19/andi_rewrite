"""Launch the controls CLI with the frozen Stage-A CPU thread policy.

The wrapper is still executed by the ANDi interpreter.  It sets PyTorch's
intra-op and inter-op thread counts before the controls module is imported,
then delegates to the ordinary CLI without changing its arguments or data.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
SCRIPT = Path(__file__).resolve().with_name("run_domain_classifier_controls.py")
runpy.run_path(str(SCRIPT), run_name="__main__")
