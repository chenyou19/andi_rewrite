from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.engine.evaluation.timing import EvaluationTimer, STAGES, TIMING_CSV, TIMING_JSON
from andi_rewrite.engine.evaluator import VolumeEvaluator
from andi_rewrite.scripts import eval as eval_script


class _Detector:
    t_lower = 1
    t_upper = 2
    device = torch.device("cpu")
    diffusion = None
    noise_plan = None
    postprocess_policy = None


class _Evaluator(VolumeEvaluator):
    def _volume_scores(self, image, volume_index=None):
        return image[:, 0].clone()


def _config(root: Path) -> dict:
    return {
        "output_csv": str(root / "ANDi.csv"),
        "output_mf_csv": str(root / "ANDi_mf.csv"),
        "thr_start": 0.2,
        "thr_end": 0.4,
        "thr_step": 0.1,
        "auprc_mode": "sampled",
        "auprc_max_samples": 32,
        "progress": False,
        "postprocess_mode": "rewrite",
        "postprocess": {
            "score": {"pipeline": [{"type": "normalize"}]},
            "score_mf": {"pipeline": [{"type": "normalize"}]},
            "binary_mask": {"pipeline": []},
            "threshold_mask": {"pipeline": []},
        },
        "prediction_output": {
            "enabled": True,
            "directory": str(root / "predictions"),
            "restore_native_grid": False,
        },
    }


def _batches():
    images = torch.rand((3, 1, 5, 5, 5), generator=torch.Generator().manual_seed(73))
    return [
        {
            "image": images[start:end],
            "label": images[start:end, 0] > 0.6,
            "metadata": {"subject_id": ["A", "B", "C"][start:end]},
        }
        for start, end in ((0, 1), (1, 3))
    ]


class EvaluationTimerTest(unittest.TestCase):
    def test_exclusive_nested_spans_and_batch_attribution(self):
        now = 0.0
        timer = EvaluationTimer(clock=lambda: now)
        indices = timer.register_subjects([{"subject_id": "a"}, {"subject_id": "b"}])
        with timer.span("metrics"):
            now += 2
            with timer.span("data_read", indices):
                now += 6
            with timer.span("postprocessing", 0):
                now += 4
                with timer.span("data_read", 0):
                    now += 1
            now += 3
        now += 5
        summary = timer.finish()
        self.assertEqual(summary["evaluator_wall_seconds"], 21)
        self.assertEqual(summary["stage_seconds"], {
            "data_read": 7, "inference": 0, "postprocessing": 4, "file_write": 0, "metrics": 5,
        })
        self.assertEqual(summary["shared_stage_seconds"]["metrics"], 5)
        self.assertEqual(summary["overhead_seconds"], 5)
        self.assertEqual(summary["subject_statistics"]["data_read_seconds"], {
            "count": 2, "mean": 3.5, "median": 3.5, "p95": 4.0,
        })
        self.assertEqual(timer.subjects[0]["attributed_total_seconds"], 8)
        self.assertEqual(timer.subjects[1]["attributed_total_seconds"], 3)
        self.assertEqual(timer.subjects[0]["amortized_stages"], ["data_read"])

    def test_cuda_sync_uses_the_measured_device_and_never_runs_on_cpu(self):
        cuda = torch.device("cuda:2")
        timer = EvaluationTimer(inference_device=cuda, postprocess_device=torch.device("cpu"))
        with patch("torch.cuda.synchronize") as sync:
            with timer.span("inference"):
                self.assertEqual(sync.call_count, 1)
            with timer.span("postprocessing"):
                self.assertEqual(sync.call_count, 2)
            self.assertEqual([call.args for call in sync.call_args_list], [(cuda,), (cuda,)])
            with EvaluationTimer(enabled=False, inference_device=cuda).span("inference"):
                self.assertEqual(sync.call_count, 2)

    def test_data_read_times_iterator_wait_and_leaves_consumer_work_out(self):
        now = 0.0
        timer = EvaluationTimer(clock=lambda: now)
        timer.register_subjects([{}, {}])

        class Loader:
            batches = iter(["first", "second"])

            def __iter__(self):
                nonlocal now
                now += 2
                return self

            def __next__(self):
                nonlocal now
                now += 3
                return next(self.batches)

        for index, (_, wait_seconds) in enumerate(timer.batches(Loader())):
            timer.record("data_read", wait_seconds, index)
            now += 10
        summary = timer.finish()
        self.assertEqual(summary["stage_seconds"]["data_read"], 11)
        self.assertEqual(summary["shared_stage_seconds"]["data_read"], 5)
        self.assertEqual(summary["overhead_seconds"], 20)


class EvaluationTimingIntegrationTest(unittest.TestCase):
    def assert_stage_accounting(self, result, subjects):
        timing = result["timing"]
        for stage in STAGES:
            measured = sum(row[f"{stage}_seconds"] for row in subjects)
            self.assertAlmostEqual(
                measured + timing["shared_stage_seconds"][stage],
                timing["stage_seconds"][stage],
            )
        self.assertAlmostEqual(
            timing["overhead_seconds"] + sum(timing["stage_seconds"].values()),
            timing["evaluator_wall_seconds"],
        )
        self.assertGreaterEqual(timing["overhead_seconds"], 0.0)

    def test_in_memory_parity_and_repeated_runs(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.synchronize") as sync:
            root = Path(directory)
            config = _config(root)
            evaluator = _Evaluator(_Detector(), config)
            expected = evaluator.evaluate(_batches())
            expected_csv = evaluator.output_csv.read_bytes()
            image_path = root / "predictions/A/anomaly_score_mf.nii.gz"
            expected_image = nib.load(image_path).get_fdata()
            timed = _Evaluator(_Detector(), {**config, "timing": {"enabled": True}})
            for _ in range(2):
                result = timed.evaluate(_batches())
                subjects = timed.last_timing_subjects
                self.assert_stage_accounting(result, subjects)
                self.assertEqual({key: value for key, value in result.items() if key != "timing"}, expected)
                self.assertEqual(timed.output_csv.read_bytes(), expected_csv)
                np.testing.assert_array_equal(nib.load(image_path).get_fdata(), expected_image)
                self.assertEqual([row["subject_id"] for row in subjects], ["A", "B", "C"])
                self.assertGreater(result["timing"]["shared_stage_seconds"]["postprocessing"], 0)
                self.assertGreater(result["timing"]["shared_stage_seconds"]["metrics"], 0)
                self.assertGreater(subjects[0]["file_write_seconds"], 0)
                self.assertEqual(subjects[0]["amortized_stages"], [])
                self.assertEqual(subjects[1]["amortized_stages"], ["data_read", "inference"])
                self.assertEqual(subjects[1]["inference_seconds"], subjects[2]["inference_seconds"])
            sync.assert_not_called()

    def test_disk_cache_resume_from_run_without_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                **_config(root),
                "memory_mode": "disk_streaming",
                "auprc_mode": "exact",
                "auprc_max_samples": None,
                "cache": {"directory": str(root / "cache")},
            }
            cold = _Evaluator(_Detector(), config)
            expected = cold.evaluate(_batches())
            expected_csv = cold.output_csv.read_bytes()
            timed = _Evaluator(_Detector(), {**config, "timing": {"enabled": True}})
            with patch.object(timed, "_volume_scores", side_effect=AssertionError("cache miss")):
                result = timed.evaluate(_batches())
            subjects = timed.last_timing_subjects
            self.assert_stage_accounting(result, subjects)
            self.assertEqual(expected["subjects_inferred"], 3)
            self.assertEqual(result["timing"]["cache_hits"], 3)
            self.assertEqual(result["timing"]["stage_seconds"]["inference"], 0.0)
            for row in subjects:
                self.assertTrue(row["cache_hit"])
                self.assertEqual(row["inference_seconds"], 0.0)
                self.assertGreater(row["data_read_seconds"], 0)
                self.assertGreater(row["postprocessing_seconds"], 0)
                self.assertGreater(row["metrics_seconds"], 0)
            self.assertEqual(timed.output_csv.read_bytes(), expected_csv)

            partial = {
                **config, "cache": {"directory": str(root / "partial_cache")},
            }

            def interrupted():
                yield _batches()[0]
                raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _Evaluator(_Detector(), partial).evaluate(interrupted())
            batches = _batches()
            mixed = [{
                "image": torch.cat([batch["image"] for batch in batches]),
                "label": torch.cat([batch["label"] for batch in batches]),
                "metadata": {"subject_id": ["A", "B", "C"]},
            }]
            timed = _Evaluator(_Detector(), {**partial, "timing": {"enabled": True}})
            result = timed.evaluate(mixed)
            subjects = timed.last_timing_subjects
            self.assert_stage_accounting(result, subjects)
            self.assertEqual([row["cache_hit"] for row in subjects], [True, False, False])
            self.assertEqual(subjects[0]["inference_seconds"], 0)
            self.assertGreater(subjects[1]["inference_seconds"], 0)
            self.assertEqual(subjects[1]["inference_seconds"], subjects[2]["inference_seconds"])
            self.assertEqual(subjects[1]["amortized_stages"], ["data_read", "inference"])
            self.assertEqual(timed.output_csv.read_bytes(), expected_csv)

    def test_cli_flag_writes_artifacts_and_inference_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            prediction_output = config.pop("prediction_output")
            config_path = root / "eval.yaml"
            config_path.write_text(yaml.safe_dump({
                "metrics": config,
                "evaluation": {"timing": {"enabled": False}},
                "prediction_output": prediction_output,
            }), encoding="utf-8")
            with (
                patch.object(sys, "argv", ["eval.py", "--run-eval", "--timing", "--config", str(config_path)]),
                patch.object(eval_script, "build_detector_from_config", return_value=(_Detector(), None)),
                patch.object(eval_script, "VolumeEvaluator", _Evaluator),
                patch.object(eval_script, "build_dataloader", return_value=_batches()),
                patch.object(eval_script, "print_config"),
            ):
                eval_script.main()

            artifact = json.loads((root / TIMING_JSON).read_text(encoding="utf-8"))
            self.assertGreater(artifact["cli_wall_seconds"], artifact["evaluator_wall_seconds"])
            self.assertGreater(artifact["cli_setup_seconds"], 0)
            self.assertGreater(artifact["cli_report_seconds"], 0)
            with (root / TIMING_CSV).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["subject_id"] for row in rows], ["A", "B", "C"])
            report = json.loads((root / "inference_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_result"]["timing"]["stage_seconds"], artifact["stage_seconds"])
            self.assertIn("## Evaluation Timing", (root / "inference_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
