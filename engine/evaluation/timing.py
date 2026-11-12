"""Opt-in wall-clock timing for sequential evaluation stages."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from itertools import repeat
from math import ceil
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import torch


STAGES = ("data_read", "inference", "postprocessing", "file_write", "metrics")
TIMING_JSON = "evaluation_timing.json"
TIMING_CSV = "evaluation_timing_subjects.csv"
_NO_SPAN = nullcontext()


def timing_enabled(config: dict[str, Any]) -> bool:
    options = config.get("timing", {})
    if not isinstance(options, dict):
        raise TypeError("evaluation.timing must be a mapping.")
    return bool(options.get("enabled", False))


class EvaluationTimer:
    """Record exclusive stage times and attribute measured work to subjects."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        inference_device: torch.device | None = None,
        postprocess_device: torch.device | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.enabled = enabled
        self.clock = clock
        self.devices = {"inference": inference_device, "postprocessing": postprocess_device}
        self.start = clock() if enabled else 0.0
        self.stage_seconds = dict.fromkeys(STAGES, 0.0)
        self.shared_stage_seconds = dict.fromkeys(STAGES, 0.0)
        self.subjects: list[dict[str, Any]] = []
        # Each stack entry accumulates the inclusive time of its immediate children.
        self._nested_seconds: list[float] = []

    def register_subjects(self, items: list[dict[str, Any]]) -> tuple[int, ...]:
        if not self.enabled:
            return ()
        start_index = len(self.subjects)
        for index, item in enumerate(items, start=start_index):
            self.subjects.append({
                "subject_index": index,
                "subject_id": str(item.get("subject_id", f"subject_{index + 1:04d}")),
                "cache_hit": False,
                **{f"{stage}_seconds": 0.0 for stage in STAGES},
                "amortized_stages": [],
            })
        return tuple(range(start_index, len(self.subjects)))

    def mark_cache_hit(self, index: int) -> None:
        if self.enabled:
            self.subjects[index]["cache_hit"] = True

    def batches(self, data: Iterable):
        """Yield a batch with its foreground iterator wait in seconds."""
        if not self.enabled:
            return zip(data, repeat(0.0))
        return self._timed_batches(data)

    def _timed_batches(self, data: Iterable):
        with self.span("data_read"):
            iterator = iter(data)
        while True:
            start = self.clock()
            try:
                batch = next(iterator)
            except StopIteration:
                self.record("data_read", self.clock() - start)
                return
            yield batch, self.clock() - start

    def record(
        self, stage: str, elapsed: float, subjects: int | tuple[int, ...] = (),
    ) -> None:
        if not self.enabled:
            return
        self.stage_seconds[stage] += elapsed
        indices = (subjects,) if isinstance(subjects, int) else subjects
        if not indices:
            self.shared_stage_seconds[stage] += elapsed
            return
        portion = elapsed / len(indices)
        for index in indices:
            subject = self.subjects[index]
            subject[f"{stage}_seconds"] += portion
            if len(indices) > 1 and stage not in subject["amortized_stages"]:
                subject["amortized_stages"].append(stage)

    def span(
        self, stage: str, subjects: int | tuple[int, ...] = (), *, synchronize: bool = True,
    ):
        if not self.enabled:
            return _NO_SPAN
        return self._measure(stage, subjects, synchronize)

    @contextmanager
    def _measure(self, stage: str, subjects: int | tuple[int, ...], synchronize: bool):
        device = self.devices.get(stage) if synchronize else None
        cuda = device is not None and device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(device)
        started = self.clock()
        self._nested_seconds.append(0.0)
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(device)
            elapsed = self.clock() - started
            nested = self._nested_seconds.pop()
            if self._nested_seconds:
                self._nested_seconds[-1] += elapsed
            self.record(stage, elapsed - nested, subjects)

    def finish(self) -> dict[str, Any]:
        elapsed = self.clock() - self.start
        for subject in self.subjects:
            subject["attributed_total_seconds"] = sum(
                subject[f"{stage}_seconds"] for stage in STAGES
            )
        columns = [*(f"{stage}_seconds" for stage in STAGES), "attributed_total_seconds"]
        return {
            "inference_device": str(self.devices["inference"]),
            "subjects_inferred": sum(not subject["cache_hit"] for subject in self.subjects),
            "cache_hits": sum(subject["cache_hit"] for subject in self.subjects),
            "evaluator_wall_seconds": elapsed,
            "stage_seconds": self.stage_seconds,
            "shared_stage_seconds": self.shared_stage_seconds,
            "overhead_seconds": elapsed - sum(self.stage_seconds.values()),
            "subject_statistics": {
                column: _statistics([subject[column] for subject in self.subjects])
                for column in columns
            },
            "subject_csv": TIMING_CSV,
            "timing_json": TIMING_JSON,
            "scope": {
                "data_read": (
                    "Foreground DataLoader iterator startup, batch wait and shutdown, plus "
                    "evaluation-cache reads and prediction reference loads. Worker I/O may overlap "
                    "inference. Cache hits still fetch input batches."
                ),
                "inference": (
                    "Wall-clock detector latency including input preparation, transfers, score "
                    "aggregation and host score copy. CUDA scoring is synchronized at its boundaries."
                ),
                "postprocessing": (
                    "Score normalization, filtering, adaptive masks and export preparation. "
                    "GPU median filtering is timed with CUDA synchronization."
                ),
                "file_write": (
                    "Metrics CSV, prediction NIfTI and JSON, and evaluation-cache writes and "
                    "cleanup. Includes serialization and NIfTI compression."
                ),
                "metrics": (
                    "Threshold sweep, metric reduction and AUPRC, including external sort workspace "
                    "I/O in exact disk-streaming AUPRC."
                ),
                "overhead_seconds": (
                    "Evaluator wall time remaining after exclusive stages. Includes bookkeeping, "
                    "tensor collection and timing overhead."
                ),
                "subjects": (
                    "Directly measured subject spans and equal allocations of shared batch spans "
                    "listed in amortized_stages. Dataset-wide work stays in shared_stage_seconds. "
                    "Statistics include cache hits with zero inference and use nearest-rank p95."
                ),
            },
        }


TIMING_DISABLED = EvaluationTimer(enabled=False)


def _statistics(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": sorted(values)[ceil(len(values) * 0.95) - 1],
    }


def write_timing(
    directory: Path,
    summary: dict[str, Any],
    subjects: list[dict[str, Any]],
    *,
    cli_wall_seconds: float,
    cli_setup_seconds: float,
    cli_report_seconds: float,
) -> None:
    """Write the timing artifacts after the ordinary inference reports."""
    directory.mkdir(parents=True, exist_ok=True)
    columns = [
        "subject_index", "subject_id", "cache_hit",
        *(f"{stage}_seconds" for stage in STAGES),
        "amortized_stages", "attributed_total_seconds",
    ]
    with (directory / TIMING_CSV).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({
            **subject, "amortized_stages": ",".join(subject["amortized_stages"]),
        } for subject in subjects)
    payload = {
        **summary,
        "cli_wall_seconds": cli_wall_seconds,
        "cli_setup_seconds": cli_setup_seconds,
        "cli_report_seconds": cli_report_seconds,
        "cli_scope": (
            "Wall time starts after argument parsing and ends after writing the inference reports. "
            "Setup includes config, detector and DataLoader construction. Report time includes report "
            "preparation and writes. Timing artifact writes are outside the measured interval."
        ),
        "subjects": subjects,
    }
    (directory / TIMING_JSON).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
