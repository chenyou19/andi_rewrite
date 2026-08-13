"""In-memory volume collection shared by the evaluator facade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterable

import torch

from andi_rewrite.utils.progress import ProgressReporter


def collect_volumes(
    dataloader: Iterable,
    *,
    volume_scores: Callable[..., torch.Tensor],
    split_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor | None, Any]],
    metadata_items: Callable[[Any, int], list[dict[str, Any]]],
    has_label: Callable[[torch.Tensor | None, dict[str, Any]], bool],
    accelerator: Any,
    progress_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
    """Collect raw maps, all-or-nothing labels, and subject metadata in order."""

    maps = []
    labels = []
    label_flags: list[bool] = []
    collected_metadata: list[dict[str, Any]] = []
    try:
        total = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        total = 0
    volume_bar = ProgressReporter(
        total,
        "Evaluating volumes",
        enabled=progress_enabled,
        unit="volume",
    )
    with torch.no_grad():
        try:
            for volume_index, batch in enumerate(dataloader, start=1):
                image, label, metadata = split_batch(batch)
                items = metadata_items(metadata, int(image.shape[0]))
                anomaly_map = volume_scores(image, volume_index=volume_index)
                label_available = [has_label(label, item) for item in items]
                if label is not None:
                    label = label.to(anomaly_map.device).bool()
                if accelerator is not None:
                    if label is not None:
                        anomaly_map, label = accelerator.gather_for_metrics((anomaly_map, label))
                    else:
                        anomaly_map = accelerator.gather_for_metrics(anomaly_map)
                    # Keep Python metadata in the same rank order as gathered tensors.
                    from accelerate.utils import gather_object

                    gathered_count = int(anomaly_map.shape[0])
                    items = list(gather_object(items))[:gathered_count]
                    label_available = list(gather_object(label_available))[:gathered_count]
                maps.append(anomaly_map.detach().cpu())
                if label is not None:
                    labels.append(label.detach().cpu())
                label_flags.extend(label_available)
                collected_metadata.extend(items)
                volume_bar.update()
        finally:
            volume_bar.close()
    label_tensor = torch.cat(labels, dim=0) if labels and all(label_flags) else None
    return torch.cat(maps, dim=0), label_tensor, collected_metadata
