from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.anomaly.postprocess import (  # noqa: E402
    NormalizePostprocessor,
    PostprocessPolicy,
    PostprocessResult,
    ScorePipelineSpec,
    apply_postprocess_pipeline,
    sanitize_scores,
)
from andi_rewrite.engine.evaluator import VolumeEvaluator  # noqa: E402
from andi_rewrite.metrics.classification import (  # noqa: E402
    external_auprc,
    resolve_auprc_mode,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyDetector:
    t_lower = 1
    t_upper = 2
    device = torch.device("cpu")
    config: dict[str, Any] = {}
    postprocess_policy = None

    def set_postprocess_policy(self, policy: Any) -> None:
        self.postprocess_policy = policy


class _ScoreFromImageEvaluator(VolumeEvaluator):
    def __init__(self, *args: Any, fail_on_call: int | None = None, **kwargs: Any) -> None:
        self.score_calls = 0
        self.fail_on_call = fail_on_call
        super().__init__(*args, **kwargs)

    def _volume_scores(
        self,
        image: torch.Tensor,
        volume_index: int | None = None,
    ) -> torch.Tensor:
        self.score_calls += int(image.shape[0])
        if self.fail_on_call is not None and self.score_calls >= self.fail_on_call:
            raise RuntimeError("intentional interruption")
        return image[:, 0].float().clone()


class _StreamingHookEvaluator(_ScoreFromImageEvaluator):
    """Freeze the private streaming override seams from the former monolith."""

    HOOK_NAMES = (
        "pipeline_plans",
        "prepared_bounds",
        "raw_score_from_entry",
        "empty_stream_stats",
        "update_stream_stats",
        "finalize_stream_stats",
        "processed_stream_entry",
        "exact_auprc_chunks",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.streaming_hook_calls = {name: 0 for name in self.HOOK_NAMES}
        super().__init__(*args, **kwargs)

    def _streaming_pipeline_plans(self):
        self.streaming_hook_calls["pipeline_plans"] += 1
        return super()._streaming_pipeline_plans()

    def _prepared_bounds(self, tensor: torch.Tensor) -> tuple[float, float]:
        self.streaming_hook_calls["prepared_bounds"] += 1
        return super()._prepared_bounds(tensor)

    def _raw_score_from_entry(self, cache, entry, raw_plan, raw_bounds):
        self.streaming_hook_calls["raw_score_from_entry"] += 1
        return super()._raw_score_from_entry(cache, entry, raw_plan, raw_bounds)

    def _empty_stream_stats(self) -> dict[str, float | int]:
        self.streaming_hook_calls["empty_stream_stats"] += 1
        return super()._empty_stream_stats()

    def _update_stream_stats(self, stats, prediction, label) -> None:
        self.streaming_hook_calls["update_stream_stats"] += 1
        super()._update_stream_stats(stats, prediction, label)

    def _finalize_stream_stats(self, stats) -> dict[str, float]:
        self.streaming_hook_calls["finalize_stream_stats"] += 1
        return super()._finalize_stream_stats(stats)

    def _processed_stream_entry(self, cache, entry, raw_plan, raw_bounds):
        self.streaming_hook_calls["processed_stream_entry"] += 1
        return super()._processed_stream_entry(cache, entry, raw_plan, raw_bounds)

    def _exact_auprc_chunks(self, cache, branch, raw_plan, raw_bounds):
        self.streaming_hook_calls["exact_auprc_chunks"] += 1
        return super()._exact_auprc_chunks(cache, branch, raw_plan, raw_bounds)


class _CustomStreamingPolicy(PostprocessPolicy):
    """Third-party-style policy opting into dataset-normalized streaming."""

    mode = "custom_streaming_contract"

    def __init__(self) -> None:
        super().__init__(
            normalization_scope="dataset",
            threshold_method="otsu",
            threshold_mask_config={"pipeline": []},
            binary_mask_config={"pipeline": []},
        )
        self.raw_steps = [NormalizePostprocessor()]
        self.mf_steps = [NormalizePostprocessor()]
        self.complete_calls = 0

    def score_pipeline_spec(self) -> ScorePipelineSpec:
        return ScorePipelineSpec(
            raw_steps=tuple(self.raw_steps),
            mf_steps=tuple(self.mf_steps),
            mf_source="score_raw",
        )

    def process(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        scope = normalization_scope or self.normalization_scope
        score_raw = apply_postprocess_pipeline(
            sanitize_scores(raw_maps),
            self.raw_steps,
            normalization_scope=scope,
        )
        score_mf = apply_postprocess_pipeline(
            score_raw,
            self.mf_steps,
            normalization_scope=scope,
        )
        return self.complete_scores(score_raw, score_mf, scope)

    def _complete(
        self,
        score_raw: torch.Tensor,
        score_mf: torch.Tensor,
        normalization_scope: str,
    ) -> PostprocessResult:
        self.complete_calls += 1
        return super()._complete(score_raw, score_mf, normalization_scope)

    def describe(self) -> dict[str, Any]:
        return {
            "postprocess_mode": self.mode,
            "normalization_scope": self.normalization_scope,
            "raw_score_pipeline": ["nan_to_num", "dataset_minmax"],
            "mf_score_pipeline": ["dataset_minmax"],
            "threshold_method": self.threshold_method,
        }


class StreamingEvaluatorTest(unittest.TestCase):
    @staticmethod
    def _batches() -> list[dict[str, Any]]:
        base = torch.linspace(0.0, 1.0, 64, dtype=torch.float32).reshape(4, 4, 4)
        raw_values = [base, 0.2 + 3.0 * base.square()]
        batches = []
        for index, raw in enumerate(raw_values):
            batches.append(
                {
                    "image": raw[None, None],
                    "label": (raw > (0.55 + index * 0.1))[None],
                    "metadata": {
                        "subject_id": f"case_{index}",
                        "has_label": True,
                    },
                }
            )
        return batches

    @staticmethod
    def _config(
        root: Path,
        *,
        memory_mode: str,
        auprc_mode: str,
        cache_name: str = "cache",
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "normalize_input": False,
            "memory_mode": memory_mode,
            "cache": {
                "directory": str(root / cache_name),
                "resume": True,
                "keep_on_success": True,
            },
            "external_sort": {"chunk_bytes": 96},
            "output_csv": str(root / f"{memory_mode}_{auprc_mode}_raw.csv"),
            "output_mf_csv": str(root / f"{memory_mode}_{auprc_mode}_mf.csv"),
            "postprocess_mode": "rewrite",
            "threshold_method": "otsu",
            "normalization_scope": "dataset",
            "thr_start": 0.2,
            "thr_end": 0.5,
            "thr_step": 0.1,
            "threshold": 0.5,
            "compute_auprc": True,
            "auprc_mode": auprc_mode,
            "auprc_seed": 73,
            "progress": False,
            "postprocess": {
                "score": {"pipeline": [{"type": "normalize"}]},
                "score_mf": {
                    "pipeline": [
                        {"type": "median_filter", "kernel_size": 1, "mode": "3d"},
                        {"type": "normalize"},
                    ]
                },
                "threshold_mask": {"pipeline": []},
                "binary_mask": {"pipeline": []},
            },
            "_run_config": {
                "runtime": {"seed": 73},
                "data": {"type": "synthetic", "split": "two_subjects"},
                "model": {"type": "dummy", "checkpoint": None},
                "diffusion": {"steps": 2},
                "noise": {"schedule": {"type": "static"}},
                "anomaly": {"t_lower": 1, "t_upper": 2},
            },
        }
        if auprc_mode == "sampled":
            config["auprc_max_samples"] = 37
        return config

    def _assert_csv_equivalent(self, left: Path, right: Path) -> None:
        left_frame = pd.read_csv(left)
        right_frame = pd.read_csv(right)
        pd.testing.assert_frame_equal(
            left_frame,
            right_frame,
            check_exact=False,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    @staticmethod
    def _install_policy(
        evaluator: VolumeEvaluator,
        policy: PostprocessPolicy,
    ) -> None:
        evaluator.postprocess_policy = policy
        evaluator.detector.set_postprocess_policy(policy)
        evaluator.metric_normalization_scope = policy.normalization_scope
        evaluator.threshold_method = policy.threshold_method

    def test_custom_policy_can_opt_into_dataset_streaming_via_public_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_memory = _ScoreFromImageEvaluator(
                _DummyDetector(),
                self._config(root, memory_mode="in_memory", auprc_mode="sampled"),
            )
            in_memory_policy = _CustomStreamingPolicy()
            self._install_policy(in_memory, in_memory_policy)
            expected = in_memory.evaluate(self._batches())

            streaming = _ScoreFromImageEvaluator(
                _DummyDetector(),
                self._config(
                    root,
                    memory_mode="disk_streaming",
                    auprc_mode="sampled",
                    cache_name="custom_policy_cache",
                ),
            )
            streaming_policy = _CustomStreamingPolicy()
            description_before = streaming_policy.describe()
            self._install_policy(streaming, streaming_policy)
            actual = streaming.evaluate(self._batches())

            self.assertEqual(streaming_policy.describe(), description_before)
            self.assertEqual(in_memory_policy.complete_calls, 1)
            self.assertEqual(streaming_policy.complete_calls, len(self._batches()))
            self.assertEqual(actual["threshold_method"], expected["threshold_method"])
            self.assertAlmostEqual(actual["AUPRC"], expected["AUPRC"], places=12)
            self.assertAlmostEqual(actual["AUPRC_mf"], expected["AUPRC_mf"], places=12)
            self._assert_csv_equivalent(
                Path(streaming.output_csv),
                Path(in_memory.output_csv),
            )
            self._assert_csv_equivalent(
                Path(streaming.output_mf_csv),
                Path(in_memory.output_mf_csv),
            )

    def test_disk_streaming_preserves_facade_subclass_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _StreamingHookEvaluator(
                _DummyDetector(),
                self._config(
                    root,
                    memory_mode="disk_streaming",
                    auprc_mode="exact",
                    cache_name="hook_cache",
                ),
            )

            result = evaluator.evaluate(self._batches())

        self.assertEqual(result["subjects"], 2)
        self.assertEqual(
            set(evaluator.streaming_hook_calls),
            set(_StreamingHookEvaluator.HOOK_NAMES),
        )
        for name, calls in evaluator.streaming_hook_calls.items():
            with self.subTest(hook=name):
                self.assertGreater(calls, 0)

    def test_sampled_disk_streaming_matches_in_memory_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_memory = _ScoreFromImageEvaluator(
                _DummyDetector(),
                self._config(root, memory_mode="in_memory", auprc_mode="sampled"),
            )
            expected = in_memory.evaluate(self._batches())

            streaming_config = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            streaming = _ScoreFromImageEvaluator(_DummyDetector(), streaming_config)
            actual = streaming.evaluate(self._batches())

            self.assertAlmostEqual(actual["AUPRC"], expected["AUPRC"], places=12)
            self.assertAlmostEqual(actual["AUPRC_mf"], expected["AUPRC_mf"], places=12)
            self.assertEqual(actual["AUPRC_mode"], "sampled")
            self.assertEqual(actual["AUPRC_samples"], 37)
            self.assertEqual(streaming.score_calls, 2)
            self._assert_csv_equivalent(
                Path(streaming.output_csv),
                Path(in_memory.output_csv),
            )
            self._assert_csv_equivalent(
                Path(streaming.output_mf_csv),
                Path(in_memory.output_mf_csv),
            )

            resumed = _ScoreFromImageEvaluator(_DummyDetector(), streaming_config)
            resumed_result = resumed.evaluate(self._batches())
            self.assertEqual(resumed.score_calls, 0)
            self.assertEqual(resumed_result["cache_hits"], 2)
            self.assertEqual(resumed_result["subjects_inferred"], 0)

            manifest = json.loads(
                (root / "cache" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["collection_complete"])
            self.assertEqual(len(manifest["entries"]), 2)
            self.assertTrue(all("mf" in entry for entry in manifest["entries"]))
            self.assertTrue(all("mf_pre" not in entry for entry in manifest["entries"]))

    def test_exact_disk_streaming_matches_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_memory = _ScoreFromImageEvaluator(
                _DummyDetector(),
                self._config(root, memory_mode="in_memory", auprc_mode="exact"),
            )
            expected = in_memory.evaluate(self._batches())
            streaming = _ScoreFromImageEvaluator(
                _DummyDetector(),
                self._config(
                    root,
                    memory_mode="disk_streaming",
                    auprc_mode="exact",
                    cache_name="exact_cache",
                ),
            )

            actual = streaming.evaluate(self._batches())

            self.assertAlmostEqual(actual["AUPRC"], expected["AUPRC"], places=12)
            self.assertAlmostEqual(actual["AUPRC_mf"], expected["AUPRC_mf"], places=12)
            self.assertEqual(actual["AUPRC_mode"], "exact")
            self.assertIsNone(actual["AUPRC_samples"])
            self.assertEqual(list((root / "exact_cache" / "external_sort").iterdir()), [])

    def test_interrupted_collection_resumes_only_missing_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            interrupted = _ScoreFromImageEvaluator(
                _DummyDetector(),
                config,
                fail_on_call=2,
            )
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                interrupted.evaluate(self._batches())

            manifest = json.loads(
                (root / "cache" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["entries"]), 1)

            resumed = _ScoreFromImageEvaluator(_DummyDetector(), config)
            result = resumed.evaluate(self._batches())
            self.assertEqual(resumed.score_calls, 1)
            self.assertEqual(result["cache_hits"], 1)
            self.assertEqual(result["subjects_inferred"], 1)

    def test_cache_fingerprint_mismatch_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            _ScoreFromImageEvaluator(_DummyDetector(), config).evaluate(self._batches())
            changed = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            changed["_run_config"]["anomaly"]["t_lower"] = 99

            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                _ScoreFromImageEvaluator(_DummyDetector(), changed).evaluate(self._batches())

    def test_threshold_method_change_reuses_score_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            first = _ScoreFromImageEvaluator(_DummyDetector(), config)
            first.evaluate(self._batches())
            changed = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            changed["threshold_method"] = "yen"
            resumed = _ScoreFromImageEvaluator(_DummyDetector(), changed)

            result = resumed.evaluate(self._batches())

            self.assertEqual(resumed.score_calls, 0)
            self.assertEqual(result["cache_hits"], 2)
            self.assertEqual(result["threshold_method"], "yen")

    def test_corrupt_cache_file_stops_without_reinference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(
                root,
                memory_mode="disk_streaming",
                auprc_mode="sampled",
            )
            _ScoreFromImageEvaluator(_DummyDetector(), config).evaluate(self._batches())
            manifest = json.loads(
                (root / "cache" / "manifest.json").read_text(encoding="utf-8")
            )
            raw_path = root / "cache" / manifest["entries"][0]["raw_file"]
            raw_path.write_bytes(b"not a numpy file")
            resumed = _ScoreFromImageEvaluator(_DummyDetector(), config)

            with self.assertRaisesRegex(RuntimeError, "cannot be read"):
                resumed.evaluate(self._batches())
            self.assertEqual(resumed.score_calls, 0)

    def test_target_otsu_config_uses_streaming_sampled_auprc(self) -> None:
        path = (
            REPO_ROOT
            / "configs"
            / "eval_brats21_full_empirical_spectrum_epoch0232_20260609_otsu.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(config["evaluation"]["memory_mode"], "disk_streaming")
        self.assertEqual(config["evaluation"]["auprc_mode"], "sampled")
        self.assertEqual(config["evaluation"]["auprc_max_samples"], 5_000_000)
        self.assertEqual(config["evaluation"]["auprc_seed"], 73)
        self.assertTrue(config["evaluation"]["cache"]["resume"])
        self.assertTrue(config["evaluation"]["cache"]["keep_on_success"])


class ExternalAUPRCTest(unittest.TestCase):
    def test_external_auprc_matches_sklearn_with_cross_run_ties(self) -> None:
        scores = np.array([0.2, 0.8, 0.2, 0.9, 0.8, 0.1], dtype=np.float32)
        labels = np.array([0, 1, 1, 0, 1, 0], dtype=np.bool_)
        with tempfile.TemporaryDirectory() as directory:
            actual = external_auprc(
                [(scores[:3], labels[:3]), (scores[3:], labels[3:])],
                Path(directory) / "sort",
                chunk_bytes=48,
            )

        self.assertAlmostEqual(actual, average_precision_score(labels, scores), places=15)

    def test_external_auprc_constant_and_all_negative_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constant_scores = np.ones(8, dtype=np.float32)
            mixed_labels = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool_)
            constant = external_auprc(
                [(constant_scores, mixed_labels)],
                root / "constant",
                chunk_bytes=48,
            )
            all_negative = external_auprc(
                [(np.arange(8, dtype=np.float32), np.zeros(8, dtype=np.bool_))],
                root / "negative",
                chunk_bytes=48,
            )

        self.assertAlmostEqual(
            constant,
            average_precision_score(mixed_labels, constant_scores),
            places=15,
        )
        self.assertEqual(all_negative, 0.0)

    def test_external_auprc_sanitizes_nonfinite_scores(self) -> None:
        scores = np.array([np.nan, np.inf, -np.inf, 0.5, 0.1], dtype=np.float32)
        labels = np.array([1, 0, 1, 1, 0], dtype=np.bool_)
        sanitized = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        with tempfile.TemporaryDirectory() as directory:
            actual = external_auprc(
                [(scores, labels)],
                Path(directory) / "nonfinite",
                chunk_bytes=48,
            )

        self.assertAlmostEqual(
            actual,
            average_precision_score(labels, sanitized),
            places=15,
        )

    def test_auprc_mode_validation_and_legacy_inference(self) -> None:
        self.assertEqual(resolve_auprc_mode(None, 100), ("sampled", 100))
        self.assertEqual(resolve_auprc_mode(None, None), ("exact", None))
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_auprc_mode("sampled", None)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_auprc_mode("exact", 100)
        with self.assertRaisesRegex(ValueError, "Unknown AUPRC mode"):
            resolve_auprc_mode("approximate", None)


if __name__ == "__main__":
    unittest.main()
