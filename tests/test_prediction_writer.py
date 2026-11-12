"""Ownership, lifecycle and parity tests for asynchronous prediction export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.engine.evaluation.prediction_export import save_nifti
from andi_rewrite.engine.evaluation.prediction_writer import AsyncPredictionWriter, AsyncWriteConfig
from andi_rewrite.engine.evaluator import VolumeEvaluator
from andi_rewrite.reporting.io import save_inference_report


def _reference() -> nib.Nifti1Image:
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    affine[:3, 3] = [10.0, -2.0, 3.0]
    reference = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.int16), affine)
    reference.set_qform(affine, code=1)
    reference.set_sform(affine, code=2)
    reference.header["descrip"] = b"writer parity"
    reference.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"reference extension"))
    return reference


def _writer_config(**options) -> AsyncWriteConfig:
    return AsyncWriteConfig.from_mapping({"enabled": True, **options})


class _Detector:
    t_lower = 1
    t_upper = 2
    device = torch.device("cpu")
    postprocess_policy = None


class _Evaluator(VolumeEvaluator):
    def __init__(self, config, *, failure=None, accelerator=None):
        self.failure = failure
        self.saved = []
        self.result_calls = 0
        self.csv_write_counts = []
        super().__init__(_Detector(), config, accelerator=accelerator)

    def _volume_scores(self, image, volume_index=None):
        return image[:, 0].clone()

    def _export_predictions(self, *args, **kwargs):
        super()._export_predictions(*args, **kwargs)
        if self.failure == "export":
            raise OSError("prediction metadata write failed")

    def _save_nifti(self, array, reference_image, path, dtype):
        if self.failure == "worker" and path.name == "lesion_mask_yen.nii.gz":
            raise OSError("prediction write failed")
        super()._save_nifti(array, reference_image, path, dtype)
        self.saved.append((path, current_thread()))

    def summarize_processed(self, *args, **kwargs):
        if self.failure == "metrics":
            raise OSError("metrics failed")
        return super().summarize_processed(*args, **kwargs)

    def _summarize_streaming(self, *args, **kwargs):
        scores = super()._summarize_streaming(*args, **kwargs)
        if self.failure == "metrics":
            raise OSError("metrics failed")
        return scores

    def write_original_style_csv(self, scores, path):
        self.csv_write_counts.append(len(self.saved))
        super().write_original_style_csv(scores, path)

    def _result(self, **kwargs):
        self.result_calls += 1
        return super()._result(**kwargs)


def _config(root: Path, mode="in_memory", *, async_enabled=False, timed=False) -> dict:
    prediction_output = {
        "enabled": True,
        "directory": str(root / "predictions"),
        "restore_native_grid": True,
    }
    if async_enabled:
        prediction_output["async_write"] = {"enabled": True}
    return {
        "memory_mode": mode,
        "output_csv": str(root / "ANDi.csv"),
        "output_mf_csv": str(root / "ANDi_mf.csv"),
        "thr_start": 0.2,
        "thr_end": 0.4,
        "thr_step": 0.1,
        "compute_auprc": False,
        "progress": False,
        "postprocess_mode": "rewrite",
        "postprocess": {
            "score": {"pipeline": [{"type": "normalize"}]},
            "score_mf": {"pipeline": [{"type": "normalize"}]},
            "binary_mask": {"pipeline": []},
            "threshold_mask": {"pipeline": []},
        },
        "cache": {"directory": str(root / "cache"), "keep_on_success": False},
        "timing": {"enabled": timed},
        "prediction_output": prediction_output,
    }


def _batch(reference_path: Path) -> list[dict]:
    image = torch.rand((2, 1, 4, 4, 4), generator=torch.Generator().manual_seed(73))
    return [{
        "image": image,
        "label": image[:, 0] > 0.6,
        "metadata": {
            "subject_id": ["A", "B"],
            "reference_path": [str(reference_path)] * 2,
        },
    }]


class AsyncPredictionWriterTest(unittest.TestCase):
    def test_validates_async_config_at_evaluator_construction(self):
        self.assertEqual(
            AsyncWriteConfig.from_mapping({}),
            AsyncWriteConfig(False, 4, 8, 268435456),
        )
        self.assertEqual(_writer_config(workers=3).max_pending_jobs, 6)
        invalid = [
            (None, TypeError),
            ([], TypeError),
            ({"enabled": "true"}, TypeError),
        ]
        for key in ("workers", "max_pending_jobs", "max_pending_bytes"):
            invalid.extend(({key: value}, TypeError) for value in (True, 1.5, "1"))
            invalid.extend(({key: value}, ValueError) for value in (0, -1))
        for options, error in invalid:
            with self.subTest(options=options):
                config = _config(Path("unused"))
                config["prediction_output"]["async_write"] = options
                with self.assertRaisesRegex(error, r"prediction_output\.async_write"):
                    _Evaluator(config)

    def test_job_and_byte_budgets_apply_backpressure(self):
        array = np.arange(12, dtype=np.uint8)
        reference = _reference()
        for max_jobs, max_bytes in ((2, 10 * array.nbytes), (10, 2 * array.nbytes)):
            with self.subTest(max_jobs=max_jobs, max_bytes=max_bytes):
                release = Event()
                queue_full = Event()
                submitted = Event()
                errors = []

                def save_image(*args):
                    if not release.wait(5.0):
                        raise TimeoutError("Worker was not released.")

                writer = AsyncPredictionWriter(
                    save_image,
                    _writer_config(workers=1, max_pending_jobs=max_jobs, max_pending_bytes=max_bytes),
                )

                def produce():
                    try:
                        for index in range(3):
                            if index == 2:
                                queue_full.set()
                            writer.submit(array, reference, Path(f"A/{index}.nii.gz"), np.uint8)
                        submitted.set()
                        writer.close()
                    except BaseException as exc:
                        errors.append(exc)
                        writer.close(abort=True)

                producer = Thread(target=produce)
                producer.start()
                try:
                    self.assertTrue(queue_full.wait(5.0))
                    self.assertFalse(submitted.wait(0.05))
                finally:
                    release.set()
                    producer.join(6.0)
                self.assertFalse(producer.is_alive())
                self.assertEqual(errors, [])
                result = writer.summary()
                self.assertEqual(result["files_completed"], 3)
                self.assertEqual(result["peak_pending_jobs"], 2)
                self.assertEqual(result["peak_pending_array_bytes"], 2 * array.nbytes)
                self.assertGreater(result["backpressure_seconds"], 0.0)

    def test_oversized_job_raises_before_snapshot(self):
        writer = AsyncPredictionWriter(save_nifti, _writer_config(max_pending_bytes=47))
        try:
            array = np.arange(12, dtype=np.uint8)
            with patch("andi_rewrite.engine.evaluation.prediction_writer.np.array") as snapshot:
                with self.assertRaisesRegex(ValueError, r"48 array bytes.*max_pending_bytes \(47\)"):
                    writer.submit(array, None, Path("oversized.nii.gz"), np.float32)
                snapshot.assert_not_called()
            self.assertEqual(writer.summary()["files_submitted"], 0)
        finally:
            writer.close()

    def test_snapshot_preserves_array_affine_header_and_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _reference()
            array = np.arange(np.prod(reference.shape), dtype=np.float32).reshape(reference.shape)
            expected = root / "expected.nii.gz"
            actual = root / "actual.nii.gz"
            save_nifti(array, reference, expected, np.float32)
            release = Event()

            def save_after_mutation(*args):
                if not release.wait(5.0):
                    raise TimeoutError("Producer did not finish mutating its inputs.")
                save_nifti(*args)

            writer = AsyncPredictionWriter(save_after_mutation, _writer_config())
            try:
                writer.submit(array, reference, actual, np.float32)
                array.fill(-7)
                reference.affine[0, 3] = 91.0
                reference.header["descrip"] = b"mutated"
                reference.header.extensions[0]._raw = b"mutated extension"
                reference.header.extensions.clear()
            finally:
                release.set()
                writer.close()
            self.assertEqual(expected.read_bytes(), actual.read_bytes())

    def test_writes_to_the_same_path_are_serialized(self):
        first_started = Event()
        release = Event()
        second_started = Event()
        written = []

        def save_image(array, reference, path, dtype):
            if array[0] == 1:
                first_started.set()
                if not release.wait(5.0):
                    raise TimeoutError("First write was not released.")
            else:
                second_started.set()
            written.append(array[0])

        writer = AsyncPredictionWriter(save_image, _writer_config(workers=2))
        reference = _reference()
        errors = []
        second_submitting = Event()

        def produce():
            try:
                writer.submit(np.array([1]), reference, Path("A/image.nii.gz"), np.uint8)
                second_submitting.set()
                writer.submit(np.array([2]), reference, Path("A/../A/image.nii.gz"), np.uint8)
                writer.close()
            except BaseException as exc:
                errors.append(exc)
                writer.close(abort=True)

        producer = Thread(target=produce)
        producer.start()
        try:
            self.assertTrue(first_started.wait(5.0))
            self.assertTrue(second_submitting.wait(5.0))
            self.assertFalse(second_started.wait(0.05))
        finally:
            release.set()
            producer.join(6.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(written, [1, 2])

    def test_abort_cancels_queued_job_and_joins_running_job(self):
        started = Event()
        release = Event()
        written = []

        def save_image(array, reference, path, dtype):
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("Worker was not released.")
            written.append(path)

        writer = AsyncPredictionWriter(save_image, _writer_config(workers=1))
        array = np.zeros(1, dtype=np.uint8)
        reference = _reference()
        running_path = Path("running.nii.gz")
        queued_path = Path("queued.nii.gz")
        try:
            writer.submit(array, reference, running_path, np.uint8)
            self.assertTrue(started.wait(5.0))
            writer.submit(array, reference, queued_path, np.uint8)
            queued = next(future for future, job in writer._pending.items() if job.path == queued_path)
            cancelled = Event()
            queued.add_done_callback(lambda future: cancelled.set())
            closer = Thread(target=writer.close, kwargs={"abort": True})
            closer.start()
            try:
                self.assertTrue(cancelled.wait(5.0))
                self.assertTrue(queued.cancelled())
                self.assertTrue(closer.is_alive())
            finally:
                release.set()
                closer.join(6.0)
            self.assertFalse(closer.is_alive())
            self.assertEqual(written, [running_path])
        finally:
            release.set()
            writer.close(abort=True)


class EvaluationPredictionWriterTest(unittest.TestCase):
    def test_parity_drain_and_reporting_in_both_evaluation_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.nii.gz"
            nib.save(_reference(), reference)
            for mode in ("in_memory", "disk_streaming"):
                with self.subTest(mode=mode):
                    baseline_root = root / mode / "sync"
                    candidate_root = root / mode / "async"
                    baseline = _Evaluator(_config(baseline_root, mode))
                    config = _config(
                        candidate_root, mode, async_enabled=True, timed=mode == "disk_streaming",
                    )
                    candidate = _Evaluator(config)
                    with patch(
                        "andi_rewrite.engine.evaluation.prediction_writer.ThreadPoolExecutor",
                        side_effect=AssertionError("Synchronous evaluation created a writer"),
                    ):
                        baseline_result = baseline.evaluate(_batch(reference))
                    self.assertNotIn("async_prediction_writer", baseline_result)
                    result = candidate.evaluate(_batch(reference))
                    summary = result["async_prediction_writer"]
                    self.assertEqual(summary["files_submitted"], 10)
                    self.assertEqual(summary["files_completed"], 10)
                    self.assertGreater(summary["worker_service_seconds"], 0.0)
                    self.assertEqual(candidate.csv_write_counts, [10, 10])
                    self.assertEqual(candidate.result_calls, 1)
                    self.assertTrue(all(thread is not current_thread() for _, thread in candidate.saved))
                    self.assertTrue(all(not thread.is_alive() for _, thread in candidate.saved))
                    self.assertEqual(baseline.output_csv.read_bytes(), candidate.output_csv.read_bytes())
                    self.assertEqual(baseline.output_mf_csv.read_bytes(), candidate.output_mf_csv.read_bytes())
                    if mode == "disk_streaming":
                        timing = result["timing"]
                        self.assertIn("backpressure", timing["scope"]["file_write"])
                        self.assertLessEqual(
                            summary["drain_seconds"], timing["shared_stage_seconds"]["file_write"],
                        )
                        self.assertFalse((candidate_root / "cache").exists())
                    else:
                        self.assertNotIn("timing", result)

                    sync_predictions = baseline_root / "predictions"
                    async_predictions = candidate_root / "predictions"
                    expected = sorted(
                        path.relative_to(sync_predictions) for path in sync_predictions.glob("*/*.nii.gz")
                    )
                    actual = sorted(
                        path.relative_to(async_predictions) for path in async_predictions.glob("*/*.nii.gz")
                    )
                    self.assertEqual(len(expected), 10)
                    self.assertEqual(expected, actual)
                    for path in expected:
                        self.assertEqual(
                            (sync_predictions / path).read_bytes(),
                            (async_predictions / path).read_bytes(),
                            str(path),
                        )
                    for subject in ("A", "B"):
                        baseline_metadata = json.loads(
                            (sync_predictions / subject / "prediction_metadata.json").read_text()
                        )
                        self.assertNotIn("async_write", baseline_metadata["prediction_output"])
                        metadata = json.loads(
                            (async_predictions / subject / "prediction_metadata.json").read_text()
                        )
                        self.assertEqual(metadata["prediction_output"]["async_write"], {"enabled": True})
                    report_dir = save_inference_report(config=config, evaluator=candidate, result=result)
                    self.assertEqual(report_dir, candidate_root)
                    report = json.loads((report_dir / "inference_report.json").read_text())
                    self.assertEqual(report["evaluation_result"]["async_prediction_writer"], summary)
                    self.assertIn(
                        "Asynchronous Prediction Writes", (report_dir / "inference_report.md").read_text(),
                    )

    def test_failed_worker_prevents_csv_result_and_cache_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.nii.gz"
            nib.save(_reference(), reference)
            config = _config(root, "disk_streaming", async_enabled=True, timed=True)
            evaluator = _Evaluator(config, failure="worker")
            with self.assertRaisesRegex(RuntimeError, r"Asynchronous NIfTI write failed.*lesion_mask_yen") as caught:
                evaluator.evaluate(_batch(reference))
            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertTrue((root / "cache" / "manifest.json").is_file())
            self.assertFalse(evaluator.output_csv.exists())
            self.assertFalse(evaluator.output_mf_csv.exists())
            self.assertEqual(evaluator.result_calls, 0)
            self.assertEqual(evaluator.last_timing_subjects, [])
            self.assertIsNone(evaluator._prediction_writer)

            evaluator.failure = None
            resumed = evaluator.evaluate(_batch(reference))
            self.assertEqual(resumed["cache_hits"], 2)
            self.assertEqual(resumed["async_prediction_writer"]["files_completed"], 10)
            self.assertFalse((root / "cache").exists())

    def test_foreground_failure_closes_writer_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.nii.gz"
            nib.save(_reference(), reference)
            for mode in ("in_memory", "disk_streaming"):
                for failure in ("export", "metrics"):
                    with self.subTest(mode=mode, failure=failure):
                        run_root = root / mode / failure
                        evaluator = _Evaluator(
                            _config(run_root, mode, async_enabled=True), failure=failure,
                        )
                        with self.assertRaisesRegex(OSError, "failed"):
                            evaluator.evaluate(_batch(reference))
                        self.assertFalse(evaluator.output_csv.exists())
                        self.assertEqual(evaluator.result_calls, 0)
                        self.assertIsNone(evaluator._prediction_writer)
                        self.assertTrue(all(not thread.is_alive() for _, thread in evaluator.saved))
                        if mode == "disk_streaming":
                            self.assertTrue((run_root / "cache" / "manifest.json").is_file())
                        evaluator.failure = None
                        for _ in range(2):
                            result = evaluator.evaluate(_batch(reference))
                            self.assertEqual(result["async_prediction_writer"]["files_submitted"], 10)
                            self.assertEqual(result["async_prediction_writer"]["files_completed"], 10)

    def test_pool_requires_prediction_export_and_main_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disabled = _config(root, async_enabled=True)
            disabled["prediction_output"]["enabled"] = False
            non_main = _Evaluator(
                _config(root, async_enabled=True),
                accelerator=SimpleNamespace(is_main_process=False),
            )
            with patch(
                "andi_rewrite.engine.evaluation.prediction_writer.ThreadPoolExecutor",
                side_effect=AssertionError("Evaluation unexpectedly created a writer"),
            ):
                self.assertNotIn("async_prediction_writer", _Evaluator(disabled).evaluate(_batch(root)))
                with patch.object(non_main, "collect", return_value=(None, None, [])):
                    self.assertEqual(non_main.evaluate([]), {})

    def test_direct_exports_are_synchronous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.nii.gz"
            nib.save(_reference(), reference)
            evaluator = _Evaluator(_config(root, async_enabled=True))
            raw_maps, _, metadata = evaluator.collect(_batch(reference))
            evaluator._export_predictions(raw_maps, metadata)
            self.assertEqual(len(evaluator.saved), 10)
            self.assertTrue(all(thread is current_thread() for _, thread in evaluator.saved))
