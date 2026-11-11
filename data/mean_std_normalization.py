"""Per-volume positive-foreground population Z-score, before slicing/resizing."""
from __future__ import annotations

import numpy as np
import torch

MEAN_STD_METHOD = "mean_std"
MEAN_STD_SPEC = {
    "type": MEAN_STD_METHOD, "version": 1, "scope": "per_volume_per_modality",
    "foreground": "input > 0", "center": "mean", "scale": "population_std",
    "ddof": 0, "statistics_dtype": "float64", "output_dtype": "float32",
    "background": -1.0, "clip": False, "eps": 1e-8, "model_normalize_input": False,
}


def mean_std_normalize_volume(images: torch.Tensor, *, return_statistics: bool = False):
    if images.ndim != 4:
        raise ValueError("mean_std requires [C,H,W,Z], not individual slices.")
    values = images.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("mean_std input contains NaN/Inf.")
    output = np.full_like(values, -1.0)
    statistics = []
    for channel, volume in enumerate(values):
        mask = volume > 0
        foreground = volume[mask].astype(np.float64)
        if not foreground.size:
            statistics.append({"foreground_voxels": 0, "empty": True})
            continue
        mean = float(foreground.mean())
        std = float(foreground.std(ddof=0))
        if std <= MEAN_STD_SPEC["eps"]:
            raise ValueError(f"mean_std channel {channel} has degenerate std={std}.")
        normalized = ((foreground - mean) / std).astype(np.float32)
        output[channel][mask] = normalized
        statistics.append({"foreground_voxels": int(foreground.size), "empty": False,
                           "mean": mean, "std": std, "output_min": float(normalized.min()),
                           "output_max": float(normalized.max()),
                           "output_mean": float(normalized.mean(dtype=np.float64)),
                           "output_std": float(normalized.std(dtype=np.float64))})
    if not np.isfinite(output).all():
        raise ValueError("mean_std produced NaN/Inf.")
    result = torch.from_numpy(output).to(images.device)
    return (result, statistics) if return_statistics else result
