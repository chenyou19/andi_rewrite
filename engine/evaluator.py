from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from andi_rewrite.anomaly import ANDiDetector
from andi_rewrite.anomaly.postprocess import (
    apply_mask_postprocess,
    apply_score_postprocess,
    normalize_minmax,
    yen_threshold,
)
from andi_rewrite.metrics.classification import auprc, dice
from andi_rewrite.utils.progress import ProgressReporter


class VolumeEvaluator:
    """完整 volume ANDi evaluator，輸出格式相容原版 CSV。

    evaluator 負責 volume/slice reshape、threshold sweep、後處理變體與 CSV 寫出。
    anomaly scoring 交給 ANDiDetector，讓 evaluation policy 與 detection math
    可以分開擴充。
    """

    def __init__(
        self,
        detector: ANDiDetector,
        config: dict[str, Any],
        accelerator: Any = None,
    ):
        self.detector = detector
        self.config = config
        self.accelerator = accelerator
        self.size_splits = int(config.get("size_splits", config.get("slice_batch_size", 155)))
        self.normalize_input = bool(config.get("normalize_input", True))
        self.output_csv = Path(config.get("output_csv", config.get("output", "outputs/metrics/ANDi.csv")))
        self.output_mf_csv = Path(config.get("output_mf_csv", config.get("output_mf", "outputs/metrics/ANDi_mf.csv")))
        self.threshold_start = float(config.get("thr_start", config.get("threshold_start", 0.01)))
        self.threshold_end = float(config.get("thr_end", config.get("threshold_end", 0.3)))
        self.threshold_step = float(config.get("thr_step", config.get("threshold_step", 0.001)))
        self.metric_threshold = float(config.get("threshold", 0.5))
        self.compute_auprc = bool(config.get("compute_auprc", True))
        auprc_max_samples = config.get("auprc_max_samples")
        self.auprc_max_samples = int(auprc_max_samples) if auprc_max_samples not in (None, "", 0, False) else None
        self.auprc_seed = int(config.get("auprc_seed", 73))
        self.rank = int(config.get("rank", 3))
        self.connectivity = int(config.get("connectivity", 1))
        median_config = config.get("median_filter", {})
        self.median_enabled = bool(median_config.get("enabled", config.get("median_enabled", True)))
        self.kernel_size = int(median_config.get("kernel_size", config.get("kernel_size", 5)))
        postprocess_config = config.get("postprocess", {})
        self.score_postprocess = postprocess_config.get("score", {})
        self.score_mf_postprocess = postprocess_config.get("score_mf")
        if self.score_mf_postprocess is None:
            # 與原版 ANDi_mf.csv 相容的預設路徑；新實驗可用
            # postprocess.score_mf.pipeline 換成任意 MF 後處理流程。
            self.score_mf_postprocess = (
                {
                    "pipeline": [
                        {
                            "type": "median_filter",
                            "kernel_size": self.kernel_size,
                            "mode": str(median_config.get("mode", "3d")),
                        }
                    ]
                }
                if self.median_enabled
                else {}
            )
        self.threshold_mask_postprocess = postprocess_config.get("threshold_mask", {})
        self.yen_mask_postprocess = postprocess_config.get(
            "yen_mask",
            {
                "binary_dilation": {
                    "enabled": bool(config.get("yen_binary_dilation", True)),
                    "rank": self.rank,
                    "connectivity": self.connectivity,
                    "iterations": 1,
                }
            },
        )
        self.prediction_output = config.get("prediction_output", {})
        self.prediction_enabled = bool(self.prediction_output.get("enabled", False))
        self.model_config = config.get("model", {})
        self.anomaly_config = config.get("anomaly", {})
        self._prediction_index = 0

    def _progress_enabled(self) -> bool:
        progress_config = self.config.get("progress", True)
        if isinstance(progress_config, dict):
            return bool(progress_config.get("enabled", True)) and self.is_main_process
        return bool(progress_config) and self.is_main_process

    @property
    def is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def prepare(self, dataloader: Iterable) -> Iterable:
        if self.accelerator is None:
            return dataloader
        model, dataloader = self.accelerator.prepare(self.detector.model, dataloader)
        self.detector.model = model
        return dataloader

    def _slice_scores(self, images: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """計算 [N, C, H, W] slices 的 raw anomaly scores。"""

        scores = []
        chunks = list(torch.split(images, self.size_splits))
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

    def _volume_scores(self, image: torch.Tensor, volume_index: int | None = None) -> torch.Tensor:
        """將 [B, C, H, W, Z] 轉成 raw anomaly maps [B, H, W, Z]。"""

        image = image.to(self.detector.device)
        if self.normalize_input:
            image = image * 2.0 - 1.0
        batch_size, _, height, width, slices = image.shape
        # detector 以 2D slice 工作；先沿 slice 維度 flatten，逐 slice scoring 後再還原 volume。
        flat = image.permute(0, 4, 1, 2, 3).reshape(-1, image.shape[1], height, width)
        flat_scores = self._slice_scores(flat, volume_index=volume_index)
        return flat_scores.view(batch_size, slices, height, width).permute(0, 2, 3, 1).contiguous()

    def _split_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        if isinstance(batch, dict):
            image = batch.get("image")
            label = batch.get("label", batch.get("mask"))
            metadata = batch.get("metadata", batch.get("meta"))
            if image is None:
                raise ValueError("Dictionary batches must contain an 'image' key.")
            return image, label, metadata
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Empty evaluation batch.")
            image = batch[0]
            label = batch[1] if len(batch) > 1 else None
            metadata = batch[2] if len(batch) > 2 else None
            return image, label, metadata
        raise TypeError(f"Unsupported evaluation batch type: {type(batch)!r}")

    def _metadata_items(self, metadata: Any, batch_size: int) -> list[dict[str, Any]]:
        if metadata is None:
            return [{} for _ in range(batch_size)]
        return [self._metadata_item(metadata, index, batch_size) for index in range(batch_size)]

    def _metadata_item(self, value: Any, index: int, batch_size: int) -> Any:
        if isinstance(value, dict):
            return {key: self._metadata_item(item, index, batch_size) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            item = value
            if item.ndim > 0 and item.shape[0] == batch_size:
                item = item[index]
            if item.ndim == 0:
                return item.item()
            return item.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            item = value
            if item.ndim > 0 and item.shape[0] == batch_size:
                item = item[index]
            return item.item() if item.ndim == 0 else item.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                return self._metadata_item(value[index], 0, 1)
            return [self._metadata_item(item, index, batch_size) for item in value]
        return value

    @staticmethod
    def _truthy(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _safe_subject_id(value: Any) -> str:
        text = str(value or "subject").strip()
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
        return safe or "subject"

    @staticmethod
    def _shape_from_text(value: Any) -> tuple[int, ...] | None:
        if isinstance(value, str) and value:
            try:
                return tuple(int(part) for part in value.split(",") if part)
            except ValueError:
                return None
        if isinstance(value, (list, tuple)):
            try:
                return tuple(int(part) for part in value)
            except (TypeError, ValueError):
                return None
        return None

    def _has_label(self, label: torch.Tensor | None, metadata: dict[str, Any]) -> bool:
        if label is None:
            return False
        return self._truthy(metadata.get("has_label"), default=True)

    def _processed_prediction_maps(self, raw_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_maps = torch.nan_to_num(raw_maps.float(), nan=0.0, posinf=0.0, neginf=0.0)
        score_raw = normalize_minmax(raw_maps)
        score_raw = apply_score_postprocess(score_raw, self.score_postprocess)
        score_raw = normalize_minmax(torch.nan_to_num(score_raw, nan=0.0, posinf=0.0, neginf=0.0))

        score_mf = apply_score_postprocess(score_raw, self.score_mf_postprocess)
        score_mf = normalize_minmax(torch.nan_to_num(score_mf, nan=0.0, posinf=0.0, neginf=0.0))

        yen_mask, yen_thresholds = yen_threshold(score_mf)
        yen_mask = apply_mask_postprocess(yen_mask, self.yen_mask_postprocess)

        threshold = float(self.prediction_output.get("threshold", self.metric_threshold))
        threshold_source = str(self.prediction_output.get("threshold_source", "score_mf")).lower()
        threshold_score = score_raw if threshold_source in {"raw", "score_raw", "anomaly_score_raw"} else score_mf
        threshold_mask = apply_mask_postprocess(threshold_score > threshold, self.threshold_mask_postprocess)
        return {
            "score_raw": score_raw,
            "score_mf": score_mf,
            "yen_mask": yen_mask,
            "yen_thresholds": yen_thresholds,
            "threshold_mask": threshold_mask,
        }

    @staticmethod
    def _resize_volume_to_shape(volume: torch.Tensor, shape: tuple[int, int, int], continuous: bool) -> torch.Tensor:
        if tuple(volume.shape) == tuple(shape):
            return volume
        tensor = volume[None, None].float()
        if continuous:
            resized = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
        else:
            resized = F.interpolate(tensor, size=shape, mode="nearest")
        return resized[0, 0]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): VolumeEvaluator._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [VolumeEvaluator._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _load_reference_image(self, metadata: dict[str, Any]):
        reference_path = metadata.get("reference_path")
        if not reference_path:
            return None
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("prediction_output NIfTI export requires nibabel.") from exc
        path = Path(str(reference_path))
        if not path.exists():
            raise FileNotFoundError(f"Reference NIfTI for prediction export does not exist: {path}")
        return nib.load(str(path))

    def _save_nifti(self, array: np.ndarray, reference_image: Any, path: Path, dtype: Any) -> None:
        import nibabel as nib

        path.parent.mkdir(parents=True, exist_ok=True)
        header = reference_image.header.copy()
        header.set_data_dtype(dtype)
        nib.save(nib.Nifti1Image(array.astype(dtype), reference_image.affine, header), str(path))

    def _export_predictions(self, raw_maps: torch.Tensor, metadata_items: list[dict[str, Any]]) -> None:
        if not self.prediction_enabled or not self.is_main_process:
            return
        output_dir = Path(self.prediction_output.get("directory", "outputs/predictions"))
        restore_native = bool(self.prediction_output.get("restore_native_grid", True))
        processed = self._processed_prediction_maps(raw_maps.detach().cpu())
        batch_size = raw_maps.shape[0]
        for batch_index in range(batch_size):
            metadata = metadata_items[batch_index] if batch_index < len(metadata_items) else {}
            self._prediction_index += 1
            subject_id = self._safe_subject_id(metadata.get("subject_id", f"subject_{self._prediction_index:04d}"))
            subject_dir = output_dir / subject_id
            reference_image = self._load_reference_image(metadata)
            if reference_image is None:
                if restore_native:
                    raise ValueError("prediction_output.restore_native_grid requires metadata.reference_path.")
                try:
                    import nibabel as nib
                except ImportError as exc:
                    raise ImportError("prediction_output NIfTI export requires nibabel.") from exc
                reference_image = nib.Nifti1Image(np.zeros(tuple(raw_maps[batch_index].shape), dtype=np.float32), np.eye(4))

            native_shape = tuple(int(item) for item in reference_image.shape[:3])
            model_shape = tuple(int(item) for item in raw_maps[batch_index].shape)

            def restore(item: torch.Tensor, continuous: bool) -> torch.Tensor:
                if not restore_native:
                    return item
                return self._resize_volume_to_shape(item, native_shape, continuous=continuous)

            raw_score = restore(processed["score_raw"][batch_index], continuous=True)
            mf_score = restore(processed["score_mf"][batch_index], continuous=True)
            yen_mask = restore(processed["yen_mask"][batch_index].float(), continuous=False).bool()
            threshold_mask = restore(processed["threshold_mask"][batch_index].float(), continuous=False).bool()

            finite_raw = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
            finite_mf = torch.nan_to_num(mf_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if bool(self.prediction_output.get("save_raw_score", True)):
                self._save_nifti(finite_raw.numpy(), reference_image, subject_dir / "anomaly_score_raw.nii.gz", np.float32)
            if bool(self.prediction_output.get("save_median_filtered_score", True)):
                self._save_nifti(finite_mf.numpy(), reference_image, subject_dir / "anomaly_score_mf.nii.gz", np.float32)
            if bool(self.prediction_output.get("save_yen_mask", True)):
                self._save_nifti(yen_mask.cpu().numpy().astype(np.uint8), reference_image, subject_dir / "lesion_mask_yen.nii.gz", np.uint8)
            if bool(self.prediction_output.get("save_threshold_mask", False)):
                self._save_nifti(
                    threshold_mask.cpu().numpy().astype(np.uint8),
                    reference_image,
                    subject_dir / "lesion_mask_threshold.nii.gz",
                    np.uint8,
                )

            payload = {
                "subject_id": subject_id,
                "location": metadata.get("location"),
                "split": metadata.get("split"),
                "input_paths": metadata.get("input_paths", {}),
                "native_shape": list(native_shape),
                "model_shape": list(model_shape),
                "export_shape": list(native_shape if restore_native else model_shape),
                "restored_to_native_grid": restore_native,
                "reference_modality": metadata.get("reference_modality"),
                "reference_path": metadata.get("reference_path"),
                "checkpoint": self.model_config.get("checkpoint"),
                "use_ema": self.model_config.get("use_ema"),
                "anomaly_timestep": {
                    "t_lower": self.detector.t_lower,
                    "t_upper": self.detector.t_upper,
                },
                "modality_mapping": metadata.get("modality_mapping", {}),
                "resampled_modalities": metadata.get("resampled_modalities", ""),
                "yen_threshold": processed["yen_thresholds"][batch_index].item()
                if processed["yen_thresholds"].numel() > batch_index
                else None,
                "prediction_output": self.prediction_output,
            }
            metadata_path = subject_dir / "prediction_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(self._json_safe(payload), indent=2), encoding="utf-8")

    def collect(self, dataloader: Iterable) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
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
            enabled=self._progress_enabled(),
            unit="volume",
        )
        with torch.no_grad():
            try:
                for volume_index, batch in enumerate(dataloader, start=1):
                    image, label, metadata = self._split_batch(batch)
                    metadata_items = self._metadata_items(metadata, int(image.shape[0]))
                    anomaly_map = self._volume_scores(image, volume_index=volume_index)
                    label_available = [self._has_label(label, item) for item in metadata_items]
                    if label is not None:
                        label = label.to(anomaly_map.device).bool()
                    if self.accelerator is not None:
                        if label is not None:
                            anomaly_map, label = self.accelerator.gather_for_metrics((anomaly_map, label))
                        else:
                            anomaly_map = self.accelerator.gather_for_metrics(anomaly_map)
                    self._export_predictions(anomaly_map.detach().cpu(), metadata_items)
                    maps.append(anomaly_map.detach().cpu())
                    if label is not None:
                        labels.append(label.detach().cpu())
                    label_flags.extend(label_available)
                    collected_metadata.extend(metadata_items)
                    volume_bar.update()
            finally:
                volume_bar.close()
        label_tensor = torch.cat(labels, dim=0) if labels and all(label_flags) else None
        return torch.cat(maps, dim=0), label_tensor, collected_metadata

    def threshold_values(self) -> list[float]:
        values = np.arange(self.threshold_start, self.threshold_end, self.threshold_step)
        return [round(float(value), 3) for value in values]

    def _dice_mean(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
        values = []
        for pred_item, label_item in zip(prediction, label):
            values.append(dice(pred_item, label_item))
        return float(np.mean(values)) if values else 0.0

    def _binary_rates(self, prediction: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        prediction = prediction.bool()
        label = label.bool()
        tp = torch.logical_and(prediction, label).sum().item()
        tn = torch.logical_and(~prediction, ~label).sum().item()
        fp = torch.logical_and(prediction, ~label).sum().item()
        fn = torch.logical_and(~prediction, label).sum().item()
        return {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
        }

    def _segmentation_metrics(self, segmentation: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        rates = self._binary_rates(segmentation, label)
        return {
            "dice": self._dice_mean(segmentation, label),
            "sensitivity": rates["sensitivity"],
            "precision": rates["precision"],
        }

    def _yen_metrics(self, anomaly_map: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
        segmentation, thresholds = yen_threshold(anomaly_map)
        segmentation = apply_mask_postprocess(segmentation, self.yen_mask_postprocess)
        metrics = self._segmentation_metrics(segmentation, label)
        metrics["threshold"] = float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        return metrics

    def _summarize_without_labels(
        self,
        anomaly_map: torch.Tensor,
        anomaly_map_mf: torch.Tensor,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        scores: dict[Any, Any] = {}
        scores_mf: dict[Any, Any] = {}
        for threshold in self.threshold_values():
            scores[threshold] = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}
            scores_mf[threshold] = {"dice": "N/A", "sensitivity": "N/A", "precision": "N/A"}

        _, thresholds = yen_threshold(anomaly_map)
        _, thresholds_mf = yen_threshold(anomaly_map_mf)
        scores["yen"] = "N/A"
        scores["yenthr"] = float(thresholds.float().mean().item()) if thresholds.numel() else 0.0
        scores["yensen"] = "N/A"
        scores["yenpre"] = "N/A"
        scores_mf["yen"] = "N/A"
        scores_mf["yenthr"] = float(thresholds_mf.float().mean().item()) if thresholds_mf.numel() else 0.0
        scores_mf["yensen"] = "N/A"
        scores_mf["yenpre"] = "N/A"
        if self.compute_auprc:
            scores["AUPRC"] = "N/A"
            scores_mf["AUPRC"] = "N/A"
        scores.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
        scores_mf.update({"sensitivity": "N/A", "specificity": "N/A", "precision": "N/A"})
        return scores, scores_mf

    def summarize(self, raw_maps: torch.Tensor, labels: torch.Tensor | None) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """針對 raw 與 MF map 版本計算 threshold-sweep 與 Yen metrics。"""

        anomaly_map = normalize_minmax(torch.nan_to_num(raw_maps.float(), nan=0.0, posinf=0.0, neginf=0.0))
        anomaly_map = apply_score_postprocess(anomaly_map, self.score_postprocess)
        anomaly_map = normalize_minmax(torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0))

        anomaly_map_mf = apply_score_postprocess(anomaly_map, self.score_mf_postprocess)
        anomaly_map_mf = normalize_minmax(torch.nan_to_num(anomaly_map_mf, nan=0.0, posinf=0.0, neginf=0.0))

        if labels is None:
            return self._summarize_without_labels(anomaly_map, anomaly_map_mf)

        scores = {}
        scores_mf = {}
        thresholds = self.threshold_values()
        summary_bar = ProgressReporter(
            len(thresholds) + 3,
            "Summarizing metrics",
            enabled=self._progress_enabled(),
            unit="metric",
        )
        try:
            for threshold in thresholds:
                segmentation = apply_mask_postprocess(anomaly_map > threshold, self.threshold_mask_postprocess)
                segmentation_mf = apply_mask_postprocess(anomaly_map_mf > threshold, self.threshold_mask_postprocess)
                scores[threshold] = self._segmentation_metrics(segmentation, labels)
                scores_mf[threshold] = self._segmentation_metrics(segmentation_mf, labels)
                summary_bar.update(postfix={"thr": threshold})

            yen_metrics = self._yen_metrics(anomaly_map, labels)
            yen_metrics_mf = self._yen_metrics(anomaly_map_mf, labels)
            scores["yen"] = yen_metrics["dice"]
            scores["yenthr"] = yen_metrics["threshold"]
            scores["yensen"] = yen_metrics["sensitivity"]
            scores["yenpre"] = yen_metrics["precision"]
            scores_mf["yen"] = yen_metrics_mf["dice"]
            scores_mf["yenthr"] = yen_metrics_mf["threshold"]
            scores_mf["yensen"] = yen_metrics_mf["sensitivity"]
            scores_mf["yenpre"] = yen_metrics_mf["precision"]
            summary_bar.update(postfix="yen")
            if self.compute_auprc:
                scores["AUPRC"] = auprc(
                    anomaly_map,
                    labels,
                    max_samples=self.auprc_max_samples,
                    seed=self.auprc_seed,
                )
                scores_mf["AUPRC"] = auprc(
                    anomaly_map_mf,
                    labels,
                    max_samples=self.auprc_max_samples,
                    seed=self.auprc_seed,
                )
                if self.auprc_max_samples is not None:
                    sampled = min(int(anomaly_map.numel()), self.auprc_max_samples)
                    scores["AUPRC_samples"] = sampled
                    scores_mf["AUPRC_samples"] = sampled
                summary_bar.update(postfix="AUPRC")

            rates = self._binary_rates(anomaly_map > self.metric_threshold, labels)
            rates_mf = self._binary_rates(anomaly_map_mf > self.metric_threshold, labels)
            scores.update(rates)
            scores_mf.update(rates_mf)
            summary_bar.update(postfix="rates")
        finally:
            summary_bar.close()
        return scores, scores_mf

    @staticmethod
    def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, value in scores.items():
            if isinstance(value, dict):
                rows.append(
                    {
                        "thr": key,
                        "value": value.get("value"),
                        "dice": value.get("dice"),
                        "sensitivity": value.get("sensitivity"),
                        "precision": value.get("precision"),
                    }
                )
            else:
                rows.append({"thr": key, "value": value, "dice": None, "sensitivity": None, "precision": None})
        frame = pd.DataFrame(rows, columns=["thr", "value", "dice", "sensitivity", "precision"])
        frame.to_csv(path, index=False)

    def evaluate(self, dataloader: Iterable) -> dict[str, Any]:
        raw_maps, labels, metadata = self.collect(dataloader)
        if not self.is_main_process:
            return {}
        scores, scores_mf = self.summarize(raw_maps, labels)
        self.write_original_style_csv(scores, self.output_csv)
        self.write_original_style_csv(scores_mf, self.output_mf_csv)
        return {
            "output": str(self.output_csv),
            "output_mf": str(self.output_mf_csv),
            "subjects": len(metadata),
            "labels_available": labels is not None,
            "AUPRC": scores.get("AUPRC"),
            "AUPRC_mf": scores_mf.get("AUPRC"),
            "DiceYen": scores.get("yen"),
            "DiceYen_mf": scores_mf.get("yen"),
            "YenThr": scores.get("yenthr"),
            "YenThr_mf": scores_mf.get("yenthr"),
        }
