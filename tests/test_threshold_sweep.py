"""Exact-count and end-to-end parity checks for disk-streaming threshold sweeps."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.anomaly.postprocess import RewritePostprocessPolicy
from andi_rewrite.engine.evaluation import streaming, threshold_sweep
from andi_rewrite.engine.evaluation.metrics import empty_stream_stats, update_stream_stats
from andi_rewrite.engine.evaluator import VolumeEvaluator
from andi_rewrite.tests import test_streaming_evaluator as fixtures


class BucketCountTest(unittest.TestCase):
    def test_counts_at_float32_boundaries_and_duplicate_thresholds(self):
        rounded = 0.1
        edge = np.float32(rounded)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        max_score = np.finfo(np.float32).max
        scores = np.array(
            [
                np.nextafter(edge, -np.inf, dtype=np.float32),
                edge,
                np.nextafter(edge, np.inf, dtype=np.float32),
                -0.0,
                0.0,
                -tiny,
                tiny,
                np.nan,
                np.inf,
                -np.inf,
                -max_score,
                max_score,
            ],
            dtype=np.float32,
        )
        thresholds = [
            0.299, rounded, -0.0, float(-max_score), -2.0, float(max_score), rounded,
            float(np.nextafter(edge, 0, dtype=np.float32)), float(edge), float(tiny),
            0.0, float(-tiny), 2.0, -0.01,
        ]
        policy = RewritePostprocessPolicy({"postprocess": {"threshold_mask": {"pipeline": []}}})
        score = torch.from_numpy(scores).reshape(1, 3, 4).transpose(1, 2)
        truth = torch.tensor(
            [1, 0, -1, np.nan, 0, 0, np.inf, -np.inf, 1, 0, 1, 0], dtype=torch.float32,
        ).reshape(1, 3, 4).transpose(1, 2)
        self.assertFalse(score.is_contiguous())
        self.assertFalse(truth.is_contiguous())
        cases = [
            (score, truth),
            (score[..., :2], truth[..., :2].bool().contiguous()),
            (score[:, :0], truth[:, :0]),
            (torch.zeros_like(score), torch.zeros_like(truth)),
            (score[..., :1], torch.ones_like(truth[..., :1])),
        ]
        expected = {threshold: empty_stream_stats() for threshold in thresholds}
        actual = {threshold: empty_stream_stats() for threshold in thresholds}
        cutoffs = threshold_sweep.prepare_cutoffs(thresholds)
        assert cutoffs is not None
        with patch.object(threshold_sweep, "_CHUNK_VOXELS", 3):
            for case_score, case_label in cases:
                for threshold in thresholds:
                    update_stream_stats(
                        expected[threshold],
                        policy.fixed_threshold_mask(case_score, threshold),
                        case_label,
                    )
                threshold_sweep.update_bucket_stats(
                    actual, thresholds, cutoffs, case_score, case_label.bool().numpy()
                )
                self.assertEqual(expected, actual)
        self.assertEqual(actual[rounded]["subjects"], 2 * len(cases))
        self.assertEqual(actual[0.0]["subjects"], 2 * len(cases))

    def test_unsupported_cutoffs(self):
        for thresholds in ([], [np.nan], [np.inf], [-np.inf], [float(np.finfo(np.float32).max) * 2]):
            with self.subTest(thresholds=thresholds):
                self.assertIsNone(threshold_sweep.prepare_cutoffs(thresholds))

    def test_unsupported_tensor_layout_device_or_precision(self):
        score = torch.ones(2, 3)
        label = score.bool()
        self.assertTrue(threshold_sweep.supports_inputs(score, score, label))
        for unsupported in (score.to("meta"), score.double(), score.to_sparse(), score[:, :1]):
            with self.subTest(device=unsupported.device, dtype=unsupported.dtype, layout=unsupported.layout):
                self.assertFalse(threshold_sweep.supports_inputs(score, unsupported, label))


class ThresholdSweepEvaluatorTest(unittest.TestCase):
    @staticmethod
    def _batches():
        return fixtures.StreamingEvaluatorTest._variable_depth_batches()

    def _evaluate(self, root, backend, batches, variant):
        memory_mode = "in_memory" if variant == "in_memory" else "disk_streaming"
        config = fixtures.StreamingEvaluatorTest._config(
            root, memory_mode=memory_mode, auprc_mode="sampled"
        )
        if backend is not None:
            config["threshold_sweep_backend"] = backend
        config.update(thr_start=0.01, thr_end=0.3, thr_step=0.001, threshold_method="yen")
        config["postprocess"]["score_mf"]["pipeline"][0]["kernel_size"] = 3
        evaluator_type = fixtures._ScoreFromImageEvaluator
        if variant == "morphology":
            config["postprocess"]["threshold_mask"]["pipeline"] = [
                {"type": "binary_dilation", "rank": 3, "connectivity": 1, "iterations": 1}
            ]
        elif variant == "callback":
            evaluator_type = fixtures._StreamingHookEvaluator
        elif variant == "original_andi":
            config["postprocess_mode"] = "original_andi"

        evaluator = evaluator_type(fixtures._DummyDetector(), config)
        if variant == "custom_policy":
            fixtures.StreamingEvaluatorTest._install_policy(evaluator, fixtures._CustomStreamingPolicy())
        elif variant == "mask_override":
            original_mask = evaluator.postprocess_policy.fixed_threshold_mask
            evaluator.postprocess_policy.fixed_threshold_mask = lambda score, threshold: (
                ~original_mask(score, threshold)
            )
        elif variant == "mask_override_during_processing":
            process_entry = evaluator._processed_stream_entry
            original_mask = evaluator.postprocess_policy.fixed_threshold_mask

            def process_and_change_mask(*args):
                evaluator.postprocess_policy.fixed_threshold_mask = lambda score, threshold: (
                    ~original_mask(score, threshold)
                )
                return process_entry(*args)

            evaluator._processed_stream_entry = process_and_change_mask
        elif variant == "float64":
            process_entry = evaluator._processed_stream_entry

            def process_with_float64(*args):
                raw, processed = process_entry(*args)
                return raw, replace(processed, score_mf=processed.score_mf.double())

            evaluator._processed_stream_entry = process_with_float64
        elif variant in {"default", "unlabeled", "empty"}:
            self.assertEqual(evaluator.postprocess_policy.score_pipeline_spec().mf_source, "score_raw")

        ap_inputs = []
        real_auprc = streaming.auprc

        def capture_auprc(scores, labels, **options):
            ap_inputs.append((scores.copy(), labels.copy(), options.copy()))
            return real_auprc(scores, labels, **options)

        with ExitStack() as stack:
            class_calls = []
            if variant in {"facade_static", "streaming_static"}:
                cls, name = (
                    (VolumeEvaluator, "_update_stream_stats")
                    if variant == "facade_static"
                    else (streaming.StreamingEvaluator, "update_stream_stats")
                )
                original_stats = getattr(cls, name)

                def record_stats(*args):
                    class_calls.append(None)
                    original_stats(*args)

                stack.enter_context(patch.object(cls, name, staticmethod(record_stats)))
            stack.enter_context(patch.object(streaming, "auprc", capture_auprc))
            bucket = stack.enter_context(patch.object(
                threshold_sweep, "update_bucket_stats", wraps=threshold_sweep.update_bucket_stats,
            ))
            stack.enter_context(redirect_stdout(StringIO()))
            result = evaluator.evaluate(batches)

        raw_csv = Path(result.pop("output")).read_bytes()
        mf_csv = Path(result.pop("output_mf")).read_bytes()
        result.pop("cache_directory", None)
        calls = getattr(evaluator, "streaming_hook_calls", {}).get("update_stream_stats", 0)
        return result, raw_csv, mf_csv, ap_inputs, bucket.call_count, calls + len(class_calls)

    def _compare(self, batches, variant, *, expected_bucket_calls, backend="auto"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop = self._evaluate(root / "loop", "loop", batches, variant)
            auto = self._evaluate(root / "auto", backend, batches, variant)
        self.assertEqual(loop[:3], auto[:3])
        self.assertEqual(loop[4], 0)
        self.assertEqual(auto[4], expected_bucket_calls)
        self.assertEqual(loop[5], auto[5])
        self.assertEqual(len(loop[3]), len(auto[3]))
        for (old_scores, old_labels, old_options), (scores, labels, options) in zip(loop[3], auto[3]):
            np.testing.assert_array_equal(old_scores, scores)
            np.testing.assert_array_equal(old_labels, labels)
            self.assertEqual(old_options, options)
        return auto

    def test_default_dataset_normalization_yen_and_sampled_auprc(self):
        result, _, _, inputs, _, _ = self._compare(
            self._batches(), "default", expected_bucket_calls=4, backend=None
        )
        self.assertEqual(result["normalization_scope"], "dataset")
        self.assertEqual(result["threshold_method"], "yen")
        self.assertEqual(result["AUPRC_samples"], 37)
        self.assertEqual(result["AUPRC_seed"], 73)
        self.assertEqual([len(scores) for scores, _, _ in inputs], [37, 37])

    def test_original_andi_policy_uses_buckets(self):
        self._compare(self._batches(), "original_andi", expected_bucket_calls=4)

    def test_unlabeled_subject_disables_dataset_label_metrics(self):
        batches = self._batches()
        batches[1].pop("label")
        batches[1]["metadata"]["has_label"] = False
        result, _, _, inputs, _, _ = self._compare(batches, "unlabeled", expected_bucket_calls=0)
        self.assertFalse(result["labels_available"])
        self.assertEqual(result["AUPRC"], "N/A")
        self.assertEqual(inputs, [])

    def test_no_subjects(self):
        result, *_ = self._compare([], "empty", expected_bucket_calls=0)
        self.assertEqual(result["subjects"], 0)

    def test_custom_policy_mask_pipeline_or_mask_method_uses_loop(self):
        for variant in ("custom_policy", "morphology", "mask_override", "mask_override_during_processing"):
            with self.subTest(variant=variant):
                self._compare(self._batches(), variant, expected_bucket_calls=0)

    def test_custom_stats_callbacks_and_static_helpers_use_loop(self):
        for variant in ("callback", "facade_static", "streaming_static"):
            with self.subTest(variant=variant):
                *_, calls = self._compare(self._batches(), variant, expected_bucket_calls=0)
                self.assertEqual(calls, 2 * (290 * 2 + 4))

    def test_float64_score_uses_loop(self):
        self._compare(self._batches(), "float64", expected_bucket_calls=0)

    def test_in_memory_sweep_uses_existing_reduction(self):
        self._compare(fixtures.StreamingEvaluatorTest._batches(), "in_memory", expected_bucket_calls=0)


if __name__ == "__main__":
    unittest.main()
