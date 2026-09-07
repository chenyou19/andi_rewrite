"""Label-free, per-volume median/IQR normalization before slicing/resizing."""
from __future__ import annotations

import torch


ROBUST_METHOD = "robust_iqr"
ROBUST_SPEC = {
    "type": ROBUST_METHOD, "version": 1,
    "scope": "per_volume_per_modality", "foreground": "input > 0",
    "center": "median", "scale": "q75-q25", "background": -1.0,
    "clip": False, "eps": 1e-8, "model_normalize_input": False,
}


def robust_normalize_volume(images: torch.Tensor, *, return_statistics: bool = False):
    """Return model-space intensities; positive foreground assumes skull stripping.

    Background stays -1 as in the p99 baseline. Do not apply 2*x-1 again.
    Degenerate nonempty channels fail rather than silently amplifying noise.
    """
    if images.ndim != 4:
        raise ValueError("robust_iqr requires [C,H,W,Z], not individual slices.")
    values = images.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("robust_iqr input contains NaN/Inf.")
    output = torch.full_like(values, -1.0)
    statistics = []
    for channel in range(values.shape[0]):
        mask = values[channel] > 0
        foreground = values[channel][mask]
        if not foreground.numel():
            statistics.append({"foreground_voxels": 0, "empty": True})
            continue
        q25, median, q75 = torch.quantile(
            foreground, torch.tensor([.25, .5, .75], device=values.device)
        )
        iqr = q75 - q25
        if float(iqr) <= ROBUST_SPEC["eps"]:
            raise ValueError(f"robust_iqr channel {channel} has degenerate IQR={float(iqr)}.")
        output[channel][mask] = (foreground - median) / iqr
        statistics.append({
            "foreground_voxels": foreground.numel(), "empty": False,
            "q25": float(q25), "median": float(median), "q75": float(q75),
            "iqr": float(iqr), "output_min": float(output[channel][mask].min()),
            "output_max": float(output[channel][mask].max()),
        })
    if not bool(torch.isfinite(output).all()):
        raise ValueError("robust_iqr produced NaN/Inf.")
    return (output, statistics) if return_statistics else output
