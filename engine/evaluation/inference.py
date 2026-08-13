"""Slice-chunk inference and volume-score reconstruction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from andi_rewrite.utils.progress import ProgressReporter


def slice_scores(
    detector: Any,
    images: torch.Tensor,
    *,
    size_splits: int,
    progress_enabled: bool,
    volume_index: int | None = None,
) -> torch.Tensor:
    """Compute raw anomaly scores for flattened ``[N, C, H, W]`` slices."""

    scores = []
    chunks = list(torch.split(images, size_splits))
    volume_label = f"Volume {volume_index}" if volume_index is not None else "Volume"
    chunk_bar = ProgressReporter(
        len(chunks),
        f"{volume_label} chunks",
        enabled=progress_enabled,
        unit="chunk",
        leave=False,
    )
    try:
        for chunk_index, chunk in enumerate(chunks, start=1):
            deviations = detector.compute_deviation_stack(
                chunk,
                progress=progress_enabled,
                progress_description=f"{volume_label} chunk {chunk_index} timesteps",
                progress_leave=False,
            )
            per_modality = detector.aggregate_time(deviations)
            scores.append(detector.pool_modalities(per_modality).detach())
            chunk_bar.update()
    finally:
        chunk_bar.close()
    return torch.cat(scores, dim=0)


def volume_scores(
    detector: Any,
    image: torch.Tensor,
    *,
    normalize_input: bool,
    size_splits: int,
    progress_enabled: bool,
    volume_index: int | None = None,
    slice_score_callback: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Reconstruct ``[B, H, W, Z]`` maps from detector slice scores.

    ``slice_score_callback`` keeps the legacy facade override point available
    while the reshaping algorithm remains a stateless leaf implementation.
    """

    image = image.to(detector.device)
    if normalize_input:
        image = image * 2.0 - 1.0
    batch_size, _, height, width, slices = image.shape
    flat = image.permute(0, 4, 1, 2, 3).reshape(-1, image.shape[1], height, width)
    if slice_score_callback is None:
        flat_scores = slice_scores(
            detector,
            flat,
            size_splits=size_splits,
            progress_enabled=progress_enabled,
            volume_index=volume_index,
        )
    else:
        flat_scores = slice_score_callback(flat, volume_index=volume_index)
    return flat_scores.view(batch_size, slices, height, width).permute(0, 2, 3, 1).contiguous()
