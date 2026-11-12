"""Bounded, single-producer NIfTI writing on CPU worker threads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from os.path import normcase
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AsyncWriteConfig:
    enabled: bool
    workers: int
    max_pending_jobs: int
    max_pending_bytes: int

    @classmethod
    def from_mapping(cls, config: Any) -> AsyncWriteConfig:
        if not isinstance(config, Mapping):
            raise TypeError("prediction_output.async_write must be a mapping.")
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("prediction_output.async_write.enabled must be a boolean.")
        workers = _positive_int(config, "workers", 4)
        return cls(
            enabled,
            workers,
            _positive_int(config, "max_pending_jobs", 2 * workers),
            _positive_int(config, "max_pending_bytes", 256 * 1024**2),
        )


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"prediction_output.async_write.{key} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"prediction_output.async_write.{key} must be a positive integer.")
    return value


@dataclass(frozen=True)
class _PendingWrite:
    path: Path
    path_key: str
    array_bytes: int
    enqueued_at: float


class AsyncPredictionWriter:
    """Write owned array snapshots, bounding both queued and running jobs."""

    def __init__(
        self,
        save_image: Callable[[np.ndarray, Any, Path, Any], None],
        config: AsyncWriteConfig,
    ) -> None:
        self._config = config
        self._save_image = save_image
        self._pool = ThreadPoolExecutor(
            max_workers=config.workers, thread_name_prefix="prediction",
        )
        self._pending: dict[Future[tuple[float, float]], _PendingWrite] = {}
        self._pending_bytes = 0
        self._closed = False
        self._stats: dict[str, int | float] = {
            "files_submitted": 0,
            "files_completed": 0,
            "array_bytes_submitted": 0,
            "peak_pending_jobs": 0,
            "peak_pending_array_bytes": 0,
            "foreground_submit_seconds": 0.0,
            "backpressure_seconds": 0.0,
            "worker_queue_wait_seconds": 0.0,
            "worker_service_seconds": 0.0,
            "drain_seconds": 0.0,
        }

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(self, array: np.ndarray, reference_image: Any, path: Path, dtype: Any) -> None:
        """Wait for capacity, snapshot the callback inputs, and queue one file."""
        if self._closed:
            raise RuntimeError("The asynchronous prediction writer is closed.")
        started = perf_counter()
        self._collect_completed()
        array_bytes = array.size * np.dtype(dtype).itemsize
        if array_bytes > self._config.max_pending_bytes:
            raise ValueError(
                f"Prediction {path} needs {array_bytes} array bytes, exceeding "
                f"prediction_output.async_write.max_pending_bytes ({self._config.max_pending_bytes}). "
                "Increase the byte budget or disable asynchronous writing."
            )
        path_key = normcase(str(path.resolve()))
        while (
            len(self._pending) >= self._config.max_pending_jobs
            or self._pending_bytes + array_bytes > self._config.max_pending_bytes
            or any(job.path_key == path_key for job in self._pending.values())
        ):
            wait_started = perf_counter()
            self._collect_completed(block=True)
            self._stats["backpressure_seconds"] += perf_counter() - wait_started

        snapshot = np.array(array, dtype=dtype, copy=True, order="K")
        snapshot.flags.writeable = False
        reference = SimpleNamespace(
            affine=None if reference_image.affine is None else reference_image.affine.copy(),
            header=deepcopy(reference_image.header),
        )
        enqueued_at = perf_counter()
        # Drop the executor's reference to the payload before it signals completion.
        payload = [(snapshot, reference)]
        future = self._pool.submit(self._write, payload, path, dtype)
        self._pending[future] = _PendingWrite(path, path_key, array_bytes, enqueued_at)
        self._pending_bytes += array_bytes
        self._stats["files_submitted"] += 1
        self._stats["array_bytes_submitted"] += array_bytes
        self._stats["peak_pending_jobs"] = max(self._stats["peak_pending_jobs"], len(self._pending))
        self._stats["peak_pending_array_bytes"] = max(
            self._stats["peak_pending_array_bytes"], self._pending_bytes,
        )
        self._stats["foreground_submit_seconds"] += perf_counter() - started

    def _write(
        self, payload: list[tuple[np.ndarray, Any]], path: Path, dtype: Any,
    ) -> tuple[float, float]:
        array, reference = payload.pop()
        started = perf_counter()
        self._save_image(array, reference, path, dtype)
        return started, perf_counter()

    def _collect_completed(self, *, block: bool = False) -> None:
        completed = [future for future in self._pending if future.done()]
        if block and not completed:
            done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
            completed = list(done)
        for future in completed:
            job = self._pending.pop(future)
            self._pending_bytes -= job.array_bytes
            try:
                started, ended = future.result()
            except Exception as exc:
                raise RuntimeError(f"Asynchronous NIfTI write failed for {job.path}") from exc
            self._stats["files_completed"] += 1
            self._stats["worker_queue_wait_seconds"] += started - job.enqueued_at
            self._stats["worker_service_seconds"] += ended - started

    def close(self, *, abort: bool = False) -> None:
        """Drain on success, or cancel pending jobs and join running writes."""
        if self._closed:
            return
        self._closed = True
        started = perf_counter()
        try:
            if not abort:
                while self._pending:
                    self._collect_completed(block=True)
        except BaseException:
            abort = True
            raise
        finally:
            self._pool.shutdown(wait=True, cancel_futures=abort)
            self._pending.clear()
            self._pending_bytes = 0
            self._stats["drain_seconds"] = perf_counter() - started

    def summary(self) -> dict[str, int | float]:
        return {
            "workers": self._config.workers,
            "max_pending_jobs": self._config.max_pending_jobs,
            "max_pending_array_bytes": self._config.max_pending_bytes,
            **self._stats,
        }
