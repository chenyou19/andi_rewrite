"""Volume -> slice flatten 與 raw anomaly map collection。

把 VolumeEvaluator 中跟「逐 volume 取得 raw anomaly map」相關的邏輯集中起來。
detector 的 scoring math 不在這裡，這個 collector 只負責 reshape、chunking、
accelerator gather 與 collect-all。行為與原版逐字一致（非 streaming）。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.utils.progress import ProgressReporter

from .config import VolumeEvaluationConfig


class VolumeScoreCollector:
    """逐 volume 計算 raw anomaly map，並 collect 成單一張量。"""

    def __init__(
        self,
        detector: ANDiDetector,
        config: VolumeEvaluationConfig,
        accelerator: Any = None,
        progress_enabled_fn: Callable[[], bool] | None = None,
    ):
        self.detector = detector
        self.config = config
        self.accelerator = accelerator
        self._progress_enabled_fn = progress_enabled_fn if progress_enabled_fn is not None else (lambda: False)

    def _progress_enabled(self) -> bool:
        return bool(self._progress_enabled_fn())

    def slice_scores(self, images: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """計算 [N, C, H, W] slices 的 raw anomaly scores。"""

        scores = []
        chunks = list(torch.split(images, self.config.size_splits))
        progress_enabled = self._progress_enabled()
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
                deviations = self.detector.compute_deviation_stack(
                    chunk,
                    progress=progress_enabled,
                    progress_description=f"{volume_label} chunk {chunk_index} timesteps",
                    progress_leave=False,
                )
                per_modality = self.detector.aggregate_time(deviations)
                scores.append(self.detector.pool_modalities(per_modality).detach())
                chunk_bar.update()
        finally:
            chunk_bar.close()
        return torch.cat(scores, dim=0)

    def volume_scores(self, image: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """將 [B, C, H, W, Z] 轉成 raw anomaly maps [B, H, W, Z]。"""

        image = image.to(self.detector.device)
        if self.config.normalize_input:
            image = image * 2.0 - 1.0
        batch_size, _, height, width, slices = image.shape
        # detector 以 2D slice 工作；先沿 slice 維度 flatten，逐 slice scoring 後再還原 volume。
        flat = image.permute(0, 4, 1, 2, 3).reshape(-1, image.shape[1], height, width)
        flat_scores = self.slice_scores(flat, volume_index=volume_index)
        return flat_scores.view(batch_size, slices, height, width).permute(0, 2, 3, 1).contiguous()

    def collect(self, dataloader: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        maps = []
        labels = []
        try:
            total = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            total = 0
        volume_bar = ProgressReporter(
            total,
            "Evaluating volumes",
            enabled=self._progress_enabled(),
            unit="volume",
        )
        with torch.no_grad():
            try:
                for volume_index, (image, label) in enumerate(dataloader, start=1):
                    anomaly_map = self.volume_scores(image, volume_index=volume_index)
                    label = label.to(anomaly_map.device).bool()
                    if self.accelerator is not None:
                        anomaly_map, label = self.accelerator.gather_for_metrics((anomaly_map, label))
                    maps.append(anomaly_map.detach().cpu())
                    labels.append(label.detach().cpu())
                    volume_bar.update()
            finally:
                volume_bar.close()
        return torch.cat(maps, dim=0), torch.cat(labels, dim=0)
