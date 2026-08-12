"""Focused public-contract tests for the evaluator refactor.

These tests deliberately exercise stable caller-visible behavior rather than
the layout of the extracted implementation modules.
"""

from __future__ import annotations

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.anomaly import (  # noqa: E402
    OriginalANDiPostprocessPolicy as AnomalyOriginalANDiPostprocessPolicy,
)
from andi_rewrite.anomaly import PostprocessResult as AnomalyPostprocessResult  # noqa: E402
from andi_rewrite.anomaly import (  # noqa: E402
    RewritePostprocessPolicy as AnomalyRewritePostprocessPolicy,
)
from andi_rewrite.anomaly.postprocess import (  # noqa: E402
    OriginalANDiPostprocessPolicy,
    PostprocessResult,
    RewritePostprocessPolicy,
)
from andi_rewrite.engine import VolumeEvaluator as EngineVolumeEvaluator  # noqa: E402
from andi_rewrite.engine.evaluation.metrics import (  # noqa: E402
    empty_stream_stats,
    finalize_stream_stats,
    update_stream_stats,
)
from andi_rewrite.engine.evaluator import VolumeEvaluator  # noqa: E402


class _DummyDetector:
    t_lower = 1
    t_upper = 2
    device = torch.device("cpu")
    config: dict[str, Any] = {}
    postprocess_policy: Any = None

    def set_postprocess_policy(self, policy: Any) -> None:
        self.postprocess_policy = policy


class _RecordingScoreEvaluator(VolumeEvaluator):
    """A caller extension point used by lightweight evaluation integrations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.volume_score_calls: list[tuple[torch.Tensor, int | None]] = []
        super().__init__(*args, **kwargs)

    def _volume_scores(
        self,
        image: torch.Tensor,
        volume_index: int | None = None,
    ) -> torch.Tensor:
        self.volume_score_calls.append((image.detach().clone(), volume_index))
        return image[:, 0].float().clone()


class _MetricHookEvaluator(_RecordingScoreEvaluator):
    """Freeze the subclass hooks historically invoked by metric orchestration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metric_hook_calls = {
            "segmentation": 0,
            "threshold_method": 0,
            "binary_rates": 0,
            "without_labels": 0,
        }

    def _segmentation_metrics(
        self,
        segmentation: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, float]:
        self.metric_hook_calls["segmentation"] += 1
        return {"dice": 0.11, "sensitivity": 0.22, "precision": 0.33}

    def _threshold_method_metrics(
        self,
        segmentation: torch.Tensor,
        thresholds: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, float]:
        self.metric_hook_calls["threshold_method"] += 1
        return {
            "dice": 0.44,
            "sensitivity": 0.55,
            "precision": 0.66,
            "threshold": 0.77,
        }

    def _binary_rates(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, float]:
        self.metric_hook_calls["binary_rates"] += 1
        return {"sensitivity": 0.88, "specificity": 0.89, "precision": 0.9}

    def _summarize_without_labels(
        self,
        processed: PostprocessResult,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        self.metric_hook_calls["without_labels"] += 1
        return {"hook": "raw"}, {"hook": "mf"}


class _MethodSwitchingEvaluator(_RecordingScoreEvaluator):
    """Model a caller override that changes the processed adaptive method."""

    def process_raw_maps(
        self,
        raw_maps: torch.Tensor,
        normalization_scope: str | None = None,
    ) -> PostprocessResult:
        processed = super().process_raw_maps(raw_maps, normalization_scope)
        return PostprocessResult(
            **{
                **processed.__dict__,
                "threshold_method": "otsu",
            }
        )


def _evaluation_config(
    *,
    threshold_method: str = "yen",
    score_pipeline: list[dict[str, Any]] | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Small deterministic config with no file identities in its fingerprint."""

    root = output_root or Path("outputs")
    return {
        "normalize_input": False,
        "postprocess_mode": "rewrite",
        "threshold_method": threshold_method,
        "thr_start": 0.2,
        "thr_end": 0.4,
        "thr_step": 0.1,
        "threshold": 0.5,
        "compute_auprc": False,
        "progress": False,
        "output_csv": str(root / "ANDi.csv"),
        "output_mf_csv": str(root / "ANDi_mf.csv"),
        "postprocess": {
            "score": {"pipeline": score_pipeline if score_pipeline is not None else [{"type": "normalize"}]},
            "score_mf": {"pipeline": [{"type": "normalize"}]},
            "threshold_mask": {"pipeline": []},
            "binary_mask": {"pipeline": []},
        },
        "_run_config": {
            "runtime": {"seed": 73},
            "data": {"type": "synthetic", "split": "refactor-contract"},
            "model": {"type": "dummy", "checkpoint": None},
            "diffusion": {"steps": 2},
            "noise": {"schedule": {"type": "static"}},
            "anomaly": {"t_lower": 1, "t_upper": 2},
        },
    }


class EvaluatorRefactorContractTest(unittest.TestCase):
    def test_legacy_imports_reexport_the_same_public_types(self) -> None:
        self.assertIs(EngineVolumeEvaluator, VolumeEvaluator)
        self.assertIs(AnomalyOriginalANDiPostprocessPolicy, OriginalANDiPostprocessPolicy)
        self.assertIs(AnomalyRewritePostprocessPolicy, RewritePostprocessPolicy)
        self.assertIs(AnomalyPostprocessResult, PostprocessResult)

    def test_practical_public_methods_and_score_override_remain_available(self) -> None:
        expected_public_methods = {
            "prepare",
            "collect",
            "process_raw_maps",
            "summarize",
            "summarize_processed",
            "threshold_values",
            "write_original_style_csv",
            "evaluate",
        }
        for method in expected_public_methods:
            self.assertTrue(callable(getattr(VolumeEvaluator, method, None)), method)

        images = torch.linspace(0.0, 1.0, 48, dtype=torch.float32).reshape(2, 1, 2, 3, 4)
        labels = images[:, 0] > 0.5
        evaluator = _RecordingScoreEvaluator(_DummyDetector(), _evaluation_config())
        maps, collected_labels, metadata = evaluator.collect(
            [
                {
                    "image": images,
                    "label": labels,
                    "metadata": {
                        "subject_id": ["first", "second"],
                        "has_label": [True, True],
                    },
                }
            ]
        )

        self.assertEqual(len(evaluator.volume_score_calls), 1)
        called_images, volume_index = evaluator.volume_score_calls[0]
        self.assertEqual(volume_index, 1)
        torch.testing.assert_close(called_images, images)
        torch.testing.assert_close(maps, images[:, 0])
        self.assertIsNotNone(collected_labels)
        assert collected_labels is not None
        self.assertTrue(torch.equal(collected_labels, labels))
        self.assertEqual([item["subject_id"] for item in metadata], ["first", "second"])

    def test_legacy_csv_schema_and_mapping_order_are_preserved(self) -> None:
        scores: dict[Any, Any] = {
            0.1: {"dice": 0.25, "sensitivity": 0.5, "precision": 0.75},
            "yen": 0.4,
            "yenthr": 0.3,
            "AUPRC": 0.6,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            VolumeEvaluator.write_original_style_csv(scores, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], ["thr", "value", "dice", "sensitivity", "precision"])
        self.assertEqual([row[0] for row in rows[1:]], ["0.1", "yen", "yenthr", "AUPRC"])
        self.assertEqual(rows[1][1:], ["", "0.25", "0.5", "0.75"])
        self.assertEqual(rows[2], ["yen", "0.4", "", "", ""])

    def test_final_result_uses_threshold_method_specific_aliases(self) -> None:
        raw = torch.linspace(0.0, 1.0, 80, dtype=torch.float32).reshape(1, 1, 4, 4, 5)
        batch = {
            "image": raw,
            "label": raw[:, 0] > 0.6,
            "metadata": {"subject_id": "contract", "has_label": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method, other_method in (("yen", "otsu"), ("otsu", "yen")):
                evaluator = _RecordingScoreEvaluator(
                    _DummyDetector(),
                    _evaluation_config(
                        threshold_method=method,
                        output_root=root / method,
                    ),
                )
                result = evaluator.evaluate([batch])
                method_label = method.title()
                other_label = other_method.title()

                self.assertEqual(result["threshold_method"], method)
                self.assertEqual(result[f"Dice{method_label}"], result["ThresholdDice"])
                self.assertEqual(result[f"Dice{method_label}_mf"], result["ThresholdDice_mf"])
                self.assertEqual(result[f"{method_label}Thr"], result["Threshold"])
                self.assertEqual(result[f"{method_label}Thr_mf"], result["Threshold_mf"])
                self.assertNotIn(f"Dice{other_label}", result)

    def test_in_memory_result_uses_the_processed_threshold_method(self) -> None:
        raw = torch.linspace(0.0, 1.0, 80, dtype=torch.float32).reshape(1, 1, 4, 4, 5)
        batch = {
            "image": raw,
            "label": raw[:, 0] > 0.6,
            "metadata": {"subject_id": "contract", "has_label": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            evaluator = _MethodSwitchingEvaluator(
                _DummyDetector(),
                _evaluation_config(
                    threshold_method="yen",
                    output_root=Path(directory),
                ),
            )
            result = evaluator.evaluate([batch])

        self.assertEqual(result["threshold_method"], "otsu")
        self.assertEqual(result["DiceOtsu"], result["ThresholdDice"])
        self.assertNotIn("DiceYen", result)

    def test_cache_fingerprints_ignore_threshold_choice_but_track_score_pipeline(self) -> None:
        baseline_config = _evaluation_config(threshold_method="yen")
        threshold_only_config = copy.deepcopy(baseline_config)
        threshold_only_config["threshold_method"] = "otsu"
        pipeline_config = copy.deepcopy(baseline_config)
        pipeline_config["postprocess"]["score"]["pipeline"] = []

        baseline = _RecordingScoreEvaluator(_DummyDetector(), baseline_config)
        threshold_only = _RecordingScoreEvaluator(_DummyDetector(), threshold_only_config)
        changed_pipeline = _RecordingScoreEvaluator(_DummyDetector(), pipeline_config)

        baseline_fingerprints = baseline._cache_fingerprints()
        self.assertEqual(baseline._cache_fingerprints(), baseline_fingerprints)
        self.assertEqual(threshold_only._cache_fingerprints(), baseline_fingerprints)
        self.assertEqual(changed_pipeline._cache_fingerprints()[0], baseline_fingerprints[0])
        self.assertNotEqual(changed_pipeline._cache_fingerprints()[1], baseline_fingerprints[1])

    def test_metric_orchestration_preserves_evaluator_subclass_hooks(self) -> None:
        raw = torch.linspace(0.0, 1.0, 80, dtype=torch.float32).reshape(1, 1, 4, 4, 5)
        labels = raw[:, 0] > 0.6
        evaluator = _MetricHookEvaluator(_DummyDetector(), _evaluation_config())
        processed = evaluator.process_raw_maps(raw[:, 0])

        scores, scores_mf = evaluator.summarize_processed(processed, labels)

        for threshold in evaluator.threshold_values():
            self.assertEqual(
                scores[threshold],
                {"dice": 0.11, "sensitivity": 0.22, "precision": 0.33},
            )
            self.assertEqual(scores_mf[threshold], scores[threshold])
        self.assertEqual(scores[processed.threshold_method], 0.44)
        self.assertEqual(scores[f"{processed.threshold_method}thr"], 0.77)
        self.assertEqual(scores[f"{processed.threshold_method}sen"], 0.55)
        self.assertEqual(scores[f"{processed.threshold_method}pre"], 0.66)
        self.assertEqual(
            {key: scores[key] for key in ("sensitivity", "specificity", "precision")},
            {"sensitivity": 0.88, "specificity": 0.89, "precision": 0.9},
        )
        self.assertEqual(evaluator.metric_hook_calls["segmentation"], 4)
        self.assertEqual(evaluator.metric_hook_calls["threshold_method"], 2)
        self.assertEqual(evaluator.metric_hook_calls["binary_rates"], 2)

        unlabeled_raw, unlabeled_mf = evaluator.summarize_processed(processed, None)
        self.assertEqual(unlabeled_raw, {"hook": "raw"})
        self.assertEqual(unlabeled_mf, {"hook": "mf"})
        self.assertEqual(evaluator.metric_hook_calls["without_labels"], 1)

    def test_shared_streaming_metric_accumulator_preserves_legacy_reduction(self) -> None:
        stats = empty_stream_stats()
        update_stream_stats(
            stats,
            torch.tensor([[True, False], [True, False]]),
            torch.tensor([[True, False], [False, True]]),
        )
        update_stream_stats(
            stats,
            torch.tensor([[False, True], [False, True]]),
            torch.tensor([[False, True], [True, True]]),
        )

        self.assertEqual(
            stats,
            {
                "dice_sum": 0.5 + 0.8,
                "subjects": 2,
                "tp": 3,
                "tn": 2,
                "fp": 1,
                "fn": 2,
            },
        )
        self.assertEqual(
            finalize_stream_stats(stats),
            {
                "dice": 0.65,
                "sensitivity": 0.6,
                "specificity": 2.0 / 3.0,
                "precision": 0.75,
            },
        )


if __name__ == "__main__":
    unittest.main()
