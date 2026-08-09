from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_dilation, generate_binary_structure, median_filter
from skimage.filters import threshold_otsu, threshold_yen
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.anomaly.detector import ANDiDetector  # noqa: E402
from andi_rewrite.engine.evaluator import VolumeEvaluator  # noqa: E402
from andi_rewrite.anomaly.postprocess import (  # noqa: E402
    MASK_POSTPROCESSORS,
    OriginalANDiPostprocessPolicy,
    PostprocessResult,
    _legacy_mask_pipeline,
    apply_mask_postprocess,
    apply_score_postprocess,
    binary_dilation_tensor,
    build_postprocess_policy,
    build_postprocess_pipeline,
    normalize_minmax,
    otsu_threshold,
    sanitize_scores,
    threshold_anomaly_map,
    yen_threshold,
)
from andi_rewrite.metrics.classification import auprc, dice  # noqa: E402
from andi_rewrite.scripts.eval import (  # noqa: E402
    apply_threshold_method_override,
    build_parser as build_eval_parser,
)
from andi_rewrite.utils.reporting import save_inference_report  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_metric_postprocess_steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    postprocess = config.get("metrics", {}).get("postprocess", {})
    if not isinstance(postprocess, dict):
        return steps
    for postprocess_config in postprocess.values():
        if not isinstance(postprocess_config, dict):
            continue
        pipeline = postprocess_config.get("pipeline")
        if isinstance(pipeline, dict):
            pipeline = [pipeline]
        if isinstance(pipeline, list):
            steps.extend(step for step in pipeline if isinstance(step, dict))
    return steps


def _reference_original_andi(
    raw: torch.Tensor,
    kernel_size: int = 3,
) -> dict[str, torch.Tensor]:
    """Minimal line-by-line reference for the original eval.py postprocessing."""

    finite = sanitize_scores(raw).cpu()
    raw_array = finite.numpy()
    mf_array = np.empty_like(raw_array)
    for index in range(raw_array.shape[0]):
        mf_array[index] = median_filter(
            raw_array[index],
            size=(kernel_size, kernel_size, kernel_size),
        )
    raw_mf = torch.from_numpy(mf_array)

    def original_norm(tensor: torch.Tensor) -> torch.Tensor:
        minimum = tensor.min()
        maximum = tensor.max()
        denominator = maximum - minimum
        if float(denominator) == 0.0:
            return torch.zeros_like(tensor)
        return (tensor - minimum) / denominator

    score_raw = original_norm(finite)
    score_mf = original_norm(raw_mf)
    structure = generate_binary_structure(3, 1)

    def yen_products(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = []
        dilated = []
        thresholds = []
        for volume in scores.numpy():
            threshold = float(threshold_yen(volume))
            mask = volume > threshold
            thresholds.append(threshold)
            masks.append(mask)
            dilated.append(binary_dilation(mask, structure=structure, iterations=1))
        return (
            torch.from_numpy(np.stack(masks)).bool(),
            torch.from_numpy(np.stack(dilated)).bool(),
            torch.tensor(thresholds, dtype=scores.dtype),
        )

    yen_raw, yen_raw_dilated, thresholds_raw = yen_products(score_raw)
    yen_mf, yen_mf_dilated, thresholds_mf = yen_products(score_mf)
    return {
        "score_raw": score_raw,
        "score_mf": score_mf,
        "yen_mask_raw": yen_raw,
        "yen_mask_mf": yen_mf,
        "yen_mask_raw_postprocessed": yen_raw_dilated,
        "yen_mask_mf_postprocessed": yen_mf_dilated,
        "yen_thresholds_raw": thresholds_raw,
        "yen_thresholds_mf": thresholds_mf,
    }


def _original_policy_config(kernel_size: int = 3) -> dict[str, Any]:
    return {
        "normalization_scope": "dataset",
        "median_filter": {"enabled": True, "kernel_size": kernel_size, "mode": "3d"},
        "yen": {
            "binary_dilation": {
                "enabled": True,
                "rank": 3,
                "connectivity": 1,
                "iterations": 1,
            }
        },
    }


class PostprocessPipelineTest(unittest.TestCase):
    def test_empty_yen_mask_pipeline_runs(self) -> None:
        mask = torch.tensor([[[[True, False], [False, True]]]])

        result = apply_mask_postprocess(mask, {"pipeline": []})

        self.assertTrue(torch.equal(result, mask))
        self.assertEqual(result.dtype, torch.bool)

    def test_yen_mask_binary_dilation_pipeline_builds(self) -> None:
        steps = build_postprocess_pipeline(
            {
                "pipeline": [
                    {
                        "type": "binary_dilation",
                        "rank": 3,
                        "connectivity": 1,
                        "iterations": 1,
                    }
                ]
            },
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )

        self.assertEqual([step.name for step in steps], ["binary_dilation"])

    def test_yen_mask_connected_components_pipeline_builds(self) -> None:
        steps = build_postprocess_pipeline(
            {
                "pipeline": [
                    {
                        "type": "connected_components",
                        "min_size": 20,
                        "connectivity": 3,
                    }
                ]
            },
            MASK_POSTPROCESSORS,
            _legacy_mask_pipeline,
            "mask",
        )

        self.assertEqual([step.name for step in steps], ["connected_components"])

    def test_yen_threshold_as_mask_postprocess_has_clear_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            build_postprocess_pipeline(
                {"pipeline": [{"type": "yen_threshold"}]},
                MASK_POSTPROCESSORS,
                _legacy_mask_pipeline,
                "mask",
            )

        message = str(context.exception)
        self.assertIn("Unknown mask postprocess step: yen_threshold.", message)
        self.assertIn(
            "Supported mask postprocess steps: binary_dilation, dilation, "
            "connected_components, remove_small_components, cc.",
            message,
        )
        self.assertIn("inside PostprocessPolicy", message)
        self.assertIn("score-to-mask thresholding stage", message)
        self.assertIn("should not be configured as a mask postprocess step", message)

    def test_repo_configs_do_not_use_yen_threshold_as_metric_postprocess(self) -> None:
        offenders: list[str] = []
        for path in sorted(REPO_ROOT.glob("configs/**/*.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            for step in _iter_metric_postprocess_steps(config):
                if str(step.get("type", "")).lower() == "yen_threshold":
                    offenders.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(offenders, [])

    def test_volume_evaluator_without_labels_reports_na_metrics(self) -> None:
        class DummyDetector:
            t_lower = 1
            t_upper = 2
            device = torch.device("cpu")

        evaluator = VolumeEvaluator(
            DummyDetector(),  # type: ignore[arg-type]
            {
                "thr_start": 0.1,
                "thr_end": 0.3,
                "thr_step": 0.1,
                "compute_auprc": True,
                "postprocess": {"score": {"pipeline": []}, "score_mf": {"pipeline": []}},
                "progress": False,
            },
        )

        scores, scores_mf = evaluator.summarize(torch.rand(1, 8, 8, 3), labels=None)

        self.assertEqual(scores["AUPRC"], "N/A")
        self.assertEqual(scores["yen"], "N/A")
        self.assertEqual(scores[0.1]["dice"], "N/A")
        self.assertEqual(scores_mf["yenpre"], "N/A")


class ThresholdMethodTest(unittest.TestCase):
    class DummyDetector:
        t_lower = 1
        t_upper = 2
        device = torch.device("cpu")
        config: dict[str, Any] = {}

    @staticmethod
    def _evaluator(method: str) -> VolumeEvaluator:
        return VolumeEvaluator(
            ThresholdMethodTest.DummyDetector(),  # type: ignore[arg-type]
            {
                "postprocess_mode": "rewrite",
                "threshold_method": method,
                "thr_start": 0.2,
                "thr_end": 0.4,
                "thr_step": 0.1,
                "compute_auprc": True,
                "postprocess": {
                    "score": {"pipeline": [{"type": "normalize"}]},
                    "score_mf": {
                        "pipeline": [
                            {"type": "median_filter", "kernel_size": 3, "mode": "3d"},
                            {"type": "normalize"},
                        ]
                    },
                    "binary_mask": {"pipeline": []},
                },
                "progress": False,
            },
        )

    def test_yen_generic_api_is_backward_compatible(self) -> None:
        scores = torch.linspace(0.0, 1.0, 128).reshape(2, 4, 4, 4)

        legacy_masks, legacy_thresholds = yen_threshold(scores)
        generic_masks, generic_thresholds = threshold_anomaly_map(scores, method="yen")

        torch.testing.assert_close(generic_thresholds, legacy_thresholds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(generic_masks, legacy_masks, rtol=0.0, atol=0.0)
        for index in range(scores.shape[0]):
            expected_threshold = float(threshold_yen(scores[index].numpy()))
            self.assertAlmostEqual(float(generic_thresholds[index]), expected_threshold, places=7)
            torch.testing.assert_close(generic_masks[index], scores[index] > expected_threshold)

    def test_otsu_matches_skimage_and_separates_bimodal_values(self) -> None:
        low = torch.full((4, 5, 5), 0.1)
        high = torch.full((4, 5, 5), 0.9)
        scores = torch.cat([low, high], dim=0).reshape(1, 5, 5, 8)

        masks, thresholds = otsu_threshold(scores)
        expected_threshold = float(threshold_otsu(scores[0].numpy()))

        self.assertAlmostEqual(float(thresholds[0]), expected_threshold, places=7)
        torch.testing.assert_close(masks[0], scores[0] > expected_threshold)
        self.assertFalse(bool(masks[0][scores[0] == 0.1].any()))
        self.assertTrue(bool(masks[0][scores[0] == 0.9].all()))

    def test_invalid_threshold_method_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown threshold method"):
            threshold_anomaly_map(torch.rand(1, 3, 3, 3), method="abc")
        with self.assertRaisesRegex(ValueError, "Unknown threshold method"):
            build_postprocess_policy(
                {"postprocess_mode": "rewrite", "threshold_method": "abc"}
            )

    def test_cli_override_reaches_evaluator_policy(self) -> None:
        args = build_eval_parser().parse_args(["--threshold-method", "otsu"])
        config = {
            "metrics": {"postprocess_mode": "rewrite"},
            "anomaly": {"threshold": "yen"},
        }

        apply_threshold_method_override(config, args.threshold_method)
        evaluator = VolumeEvaluator(
            self.DummyDetector(),  # type: ignore[arg-type]
            {
                **config["metrics"],
                "anomaly": config["anomaly"],
                "progress": False,
            },
        )

        self.assertEqual(config["metrics"]["threshold_method"], "otsu")
        self.assertEqual(config["anomaly"]["threshold"], "otsu")
        self.assertEqual(evaluator.threshold_method, "otsu")
        self.assertEqual(evaluator.postprocess_policy.threshold_method, "otsu")

    def test_constant_otsu_volume_warns_and_evaluation_continues(self) -> None:
        evaluator = self._evaluator("otsu")
        raw = torch.full((2, 4, 4, 4), 7.0)
        labels = torch.zeros_like(raw, dtype=torch.bool)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores, scores_mf = evaluator.summarize(raw, labels)

        self.assertTrue(any("constant volume" in str(item.message) for item in caught))
        self.assertTrue(np.isfinite(scores["otsuthr"]))
        self.assertTrue(np.isfinite(scores_mf["otsuthr"]))
        self.assertEqual(scores["otsu"], 0.0)
        self.assertEqual(scores_mf["otsu"], 0.0)

    def test_otsu_export_uses_method_specific_names_and_metadata(self) -> None:
        evaluator = self._evaluator("otsu")
        raw = torch.linspace(0.0, 1.0, 80).reshape(1, 4, 4, 5)
        processed = evaluator.process_raw_maps(raw)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "predictions"
            evaluator.prediction_output = {
                "enabled": True,
                "directory": str(output_dir),
                "restore_native_grid": False,
                "save_raw_score": False,
                "save_median_filtered_score": False,
                "save_binary_mask": True,
                "binary_mask_source": "score_mf",
            }
            evaluator.prediction_enabled = True
            evaluator._export_predictions(
                raw,
                [{"subject_id": "case_otsu"}],
                processed=processed,
            )

            subject_dir = output_dir / "case_otsu"
            for filename in (
                "lesion_mask_otsu_raw.nii.gz",
                "lesion_mask_otsu_mf.nii.gz",
                "lesion_mask_otsu.nii.gz",
            ):
                self.assertTrue((subject_dir / filename).is_file())
            self.assertFalse((subject_dir / "lesion_mask_yen.nii.gz").exists())
            payload = json.loads(
                (subject_dir / "prediction_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["threshold_method"], "otsu")
            self.assertEqual(payload["binary_mask_source"], "score_mf")
            self.assertAlmostEqual(payload["threshold"], processed.thresholds_mf[0].item())
            self.assertNotIn("yen_threshold", payload)

    def test_otsu_report_records_method_and_adaptive_metrics(self) -> None:
        evaluator = self._evaluator("otsu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator.output_csv = root / "ANDi.csv"
            evaluator.output_mf_csv = root / "ANDi_mf.csv"
            evaluator.write_original_style_csv(
                {"AUPRC": 0.8, "otsu": 0.6, "otsuthr": 0.35, "otsusen": 0.7, "otsupre": 0.5},
                evaluator.output_csv,
            )
            evaluator.write_original_style_csv(
                {"AUPRC": 0.85, "otsu": 0.65, "otsuthr": 0.4, "otsusen": 0.75, "otsupre": 0.55},
                evaluator.output_mf_csv,
            )

            report_dir = save_inference_report(
                config={
                    "metrics": {
                        "postprocess_mode": "rewrite",
                        "threshold_method": "otsu",
                        "output_csv": str(evaluator.output_csv),
                        "output_mf_csv": str(evaluator.output_mf_csv),
                    }
                },
                evaluator=evaluator,
                result={"threshold_method": "otsu", "Threshold": 0.35, "Threshold_mf": 0.4},
                output_dir=root / "report",
            )

            self.assertIsNotNone(report_dir)
            payload = json.loads((root / "report" / "inference_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["inference_settings"]["threshold_method"], "otsu")
            self.assertEqual(payload["post_processing_settings"]["threshold_method"], "otsu")
            self.assertEqual(payload["metrics_summary"]["raw"]["adaptive_method"], "otsu")
            self.assertAlmostEqual(payload["metrics_summary"]["raw"]["adaptive_threshold"], 0.35)

    def test_yen_and_otsu_smoke_share_scores_and_auprc(self) -> None:
        base = torch.linspace(0.0, 1.0, 125).reshape(5, 5, 5)
        raw = torch.stack([base, base.square() * 1.7 + 0.2])
        labels = raw > 0.72

        results: dict[str, tuple[PostprocessResult, dict[Any, Any], dict[Any, Any]]] = {}
        for method in ("yen", "otsu"):
            evaluator = self._evaluator(method)
            processed = evaluator.process_raw_maps(raw)
            scores, scores_mf = evaluator.summarize_processed(processed, labels)
            results[method] = (processed, scores, scores_mf)

        yen_processed, yen_scores, yen_scores_mf = results["yen"]
        otsu_processed, otsu_scores, otsu_scores_mf = results["otsu"]
        torch.testing.assert_close(yen_processed.score_raw, otsu_processed.score_raw)
        torch.testing.assert_close(yen_processed.score_mf, otsu_processed.score_mf)
        self.assertAlmostEqual(yen_scores["AUPRC"], otsu_scores["AUPRC"], places=12)
        self.assertAlmostEqual(yen_scores_mf["AUPRC"], otsu_scores_mf["AUPRC"], places=12)
        self.assertIn("yen", yen_scores_mf)
        self.assertIn("otsu", otsu_scores_mf)


class OriginalANDiPostprocessPolicyTest(unittest.TestCase):
    @staticmethod
    def _detector(policy: OriginalANDiPostprocessPolicy) -> ANDiDetector:
        detector = object.__new__(ANDiDetector)
        detector.config = {}
        detector.device = torch.device("cpu")
        detector.t_lower = 1
        detector.t_upper = 2
        detector.threshold = "yen"
        detector.postprocess_policy = policy
        return detector

    def test_original_order_filters_raw_before_any_normalization(self) -> None:
        raw = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3)
        policy = OriginalANDiPostprocessPolicy(
            {
                **_original_policy_config(kernel_size=3),
                "yen": {"binary_dilation": {"enabled": False}},
            }
        )
        events: list[str] = []
        median_inputs: list[torch.Tensor] = []
        normalize_inputs: list[torch.Tensor] = []

        def fake_median(
            tensor: torch.Tensor,
            kernel_size: int,
            mode: str,
        ) -> torch.Tensor:
            events.append("median_filter")
            median_inputs.append(tensor.clone())
            self.assertEqual(kernel_size, 3)
            self.assertEqual(mode, "3d")
            return tensor + 10.0

        def fake_normalize(
            tensor: torch.Tensor,
            eps: float = 1.0e-8,
            scope: str = "dataset",
        ) -> torch.Tensor:
            events.append(f"normalize:{scope}")
            normalize_inputs.append(tensor.clone())
            return tensor

        with (
            mock.patch(
                "andi_rewrite.anomaly.postprocess.median_filter_tensor",
                side_effect=fake_median,
            ),
            mock.patch(
                "andi_rewrite.anomaly.postprocess.normalize_minmax",
                side_effect=fake_normalize,
            ),
        ):
            policy.process(raw)

        self.assertEqual(events, ["median_filter", "normalize:dataset", "normalize:dataset"])
        torch.testing.assert_close(median_inputs[0], raw)
        torch.testing.assert_close(normalize_inputs[0], raw)
        torch.testing.assert_close(normalize_inputs[1], raw + 10.0)

    def test_original_mf_matches_filter_raw_then_normalize(self) -> None:
        raw = torch.arange(125, dtype=torch.float32).reshape(1, 5, 5, 5)
        raw[0, 2, 2, 2] = 1000.0
        policy = OriginalANDiPostprocessPolicy(_original_policy_config(kernel_size=3))

        processed = policy.process(raw)
        filtered = torch.from_numpy(
            median_filter(raw[0].numpy(), size=(3, 3, 3))[None]
        )
        expected_mf = normalize_minmax(filtered, scope="dataset")

        torch.testing.assert_close(processed.score_mf, expected_mf, rtol=1e-6, atol=1e-7)

    def test_dataset_minmax_uses_one_range_for_all_subjects(self) -> None:
        subject_a = torch.linspace(0.0, 1.0, 8).reshape(2, 2, 2)
        subject_b = torch.linspace(0.0, 100.0, 8).reshape(2, 2, 2)
        raw = torch.stack([subject_a, subject_b])
        policy = OriginalANDiPostprocessPolicy(
            {
                "normalization_scope": "dataset",
                "median_filter": {"enabled": False, "kernel_size": 5, "mode": "3d"},
                "yen": {"binary_dilation": {"enabled": False}},
            }
        )

        dataset_processed = policy.process(raw)
        subject_processed = policy.process(raw, normalization_scope="subject")

        self.assertAlmostEqual(float(dataset_processed.score_raw[0].max()), 0.01, places=7)
        self.assertAlmostEqual(float(dataset_processed.score_raw[1].max()), 1.0, places=7)
        self.assertAlmostEqual(float(subject_processed.score_raw[0].max()), 1.0, places=7)
        self.assertAlmostEqual(float(subject_processed.score_raw[1].max()), 1.0, places=7)

    def test_original_metric_scope_rejects_subject_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'dataset'"):
            OriginalANDiPostprocessPolicy(
                {
                    "normalization_scope": "subject",
                    "median_filter": {"enabled": True, "kernel_size": 5, "mode": "3d"},
                }
            )

    def test_raw_and_mf_branches_use_independent_extrema(self) -> None:
        grid = torch.tensor(np.indices((5, 5, 5)).sum(axis=0), dtype=torch.float32)
        raw = grid[None]
        raw[0, 2, 2, 2] = 1000.0
        policy = OriginalANDiPostprocessPolicy(_original_policy_config(kernel_size=3))

        processed = policy.process(raw)
        filtered = torch.from_numpy(
            median_filter(raw[0].numpy(), size=(3, 3, 3))[None]
        )

        self.assertLess(float(filtered.max()), float(raw.max()))
        self.assertAlmostEqual(float(processed.score_raw.max()), 1.0, places=7)
        self.assertAlmostEqual(float(processed.score_mf.max()), 1.0, places=7)
        torch.testing.assert_close(
            processed.score_mf,
            normalize_minmax(filtered, scope="dataset"),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_yen_threshold_is_computed_per_subject(self) -> None:
        values = torch.linspace(0.0, 1.0, 64)
        scores = torch.stack(
            [
                values.reshape(4, 4, 4),
                values.square().reshape(4, 4, 4),
            ]
        )

        masks, thresholds = yen_threshold(scores)

        self.assertEqual(tuple(thresholds.shape), (2,))
        self.assertEqual(tuple(masks.shape), (2, 4, 4, 4))
        self.assertNotAlmostEqual(float(thresholds[0]), float(thresholds[1]), places=4)
        self.assertAlmostEqual(float(thresholds[0]), float(threshold_yen(scores[0].numpy())), places=6)
        self.assertAlmostEqual(float(thresholds[1]), float(threshold_yen(scores[1].numpy())), places=6)

    def test_connectivity_one_dilation_has_six_face_neighbours(self) -> None:
        mask = torch.zeros((1, 5, 5, 5), dtype=torch.bool)
        mask[0, 2, 2, 2] = True

        dilated = binary_dilation_tensor(mask, rank=3, connectivity=1, iterations=1)

        expected = torch.zeros_like(mask)
        expected[0, 2, 2, 2] = True
        expected[0, 1, 2, 2] = True
        expected[0, 3, 2, 2] = True
        expected[0, 2, 1, 2] = True
        expected[0, 2, 3, 2] = True
        expected[0, 2, 2, 1] = True
        expected[0, 2, 2, 3] = True
        self.assertTrue(torch.equal(dilated, expected))
        self.assertEqual(int(dilated.sum()), 7)
        self.assertFalse(bool(dilated[0, 1, 1, 1]))

    def test_detector_and_evaluator_share_identical_products(self) -> None:
        torch.manual_seed(73)
        raw = torch.rand((2, 5, 5, 5), dtype=torch.float32)
        policy_config = _original_policy_config(kernel_size=3)
        policy = OriginalANDiPostprocessPolicy(policy_config)
        detector = self._detector(policy)
        evaluator = VolumeEvaluator(
            detector,
            {
                "postprocess_mode": "original_andi",
                "original_andi": policy_config,
                "thr_start": 0.1,
                "thr_end": 0.3,
                "thr_step": 0.1,
                "progress": False,
                "prediction_output": {"enabled": False},
            },
        )

        detector_output = detector.postprocess(raw)
        evaluator_output = evaluator.process_raw_maps(raw)

        self.assertIs(detector.postprocess_policy, evaluator.postprocess_policy)
        comparisons = {
            "score_raw": evaluator_output.score_raw,
            "score_mf": evaluator_output.score_mf,
            "yen_thresholds_raw": evaluator_output.yen_thresholds_raw,
            "yen_thresholds_mf": evaluator_output.yen_thresholds_mf,
            "yen_mask_raw": evaluator_output.yen_mask_raw,
            "yen_mask_mf": evaluator_output.yen_mask_mf,
            "yen_mask_raw_postprocessed": evaluator_output.yen_mask_raw_postprocessed,
            "yen_mask_mf_postprocessed": evaluator_output.yen_mask_mf_postprocessed,
        }
        for key, expected in comparisons.items():
            if expected.dtype == torch.bool:
                self.assertTrue(torch.equal(detector_output[key], expected), key)
            else:
                torch.testing.assert_close(detector_output[key], expected, rtol=1e-6, atol=1e-7)

    def test_deterministic_reference_regression(self) -> None:
        torch.manual_seed(73)
        raw = torch.rand((2, 5, 5, 5), dtype=torch.float32)
        raw[0, 2, 2, 2] += 2.0
        raw[1, 1:4, 1:4, 1:4] += 0.5
        labels = raw > 0.8
        policy_config = _original_policy_config(kernel_size=3)
        policy = OriginalANDiPostprocessPolicy(policy_config)

        processed = policy.process(raw)
        reference = _reference_original_andi(raw, kernel_size=3)

        for key in ["score_raw", "score_mf", "yen_thresholds_raw", "yen_thresholds_mf"]:
            torch.testing.assert_close(
                getattr(processed, key),
                reference[key],
                rtol=1e-6,
                atol=1e-7,
            )
        for key in [
            "yen_mask_raw",
            "yen_mask_mf",
            "yen_mask_raw_postprocessed",
            "yen_mask_mf_postprocessed",
        ]:
            self.assertTrue(torch.equal(getattr(processed, key), reference[key]), key)
        for threshold in [0.01, 0.1, 0.299]:
            self.assertTrue(
                torch.equal(
                    policy.fixed_threshold_mask(processed.score_raw, threshold),
                    reference["score_raw"] > threshold,
                )
            )
            self.assertTrue(
                torch.equal(
                    policy.fixed_threshold_mask(processed.score_mf, threshold),
                    reference["score_mf"] > threshold,
                )
            )

        detector = self._detector(policy)
        evaluator = VolumeEvaluator(
            detector,
            {
                "postprocess_mode": "original_andi",
                "original_andi": policy_config,
                "thr_start": 0.1,
                "thr_end": 0.3,
                "thr_step": 0.1,
                "progress": False,
            },
        )
        scores, scores_mf = evaluator.summarize_processed(processed, labels)
        self.assertAlmostEqual(
            scores["AUPRC"],
            float(average_precision_score(labels.reshape(-1), reference["score_raw"].reshape(-1))),
            places=7,
        )
        self.assertAlmostEqual(
            scores_mf["AUPRC"],
            float(average_precision_score(labels.reshape(-1), reference["score_mf"].reshape(-1))),
            places=7,
        )
        expected_dice_yen = float(
            np.mean(
                [
                    dice(prediction, label)
                    for prediction, label in zip(reference["yen_mask_raw_postprocessed"], labels)
                ]
            )
        )
        expected_dice_yen_mf = float(
            np.mean(
                [
                    dice(prediction, label)
                    for prediction, label in zip(reference["yen_mask_mf_postprocessed"], labels)
                ]
            )
        )
        self.assertAlmostEqual(scores["yen"], expected_dice_yen, places=7)
        self.assertAlmostEqual(scores_mf["yen"], expected_dice_yen_mf, places=7)

    def test_dataset_export_reuses_metric_tensor_and_records_scope(self) -> None:
        import nibabel as nib

        torch.manual_seed(73)
        raw = torch.rand((2, 5, 5, 5), dtype=torch.float32)
        labels = raw > 0.7
        policy_config = _original_policy_config(kernel_size=3)
        policy = OriginalANDiPostprocessPolicy(policy_config)
        detector = self._detector(policy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)),
                str(reference_path),
            )
            evaluator = VolumeEvaluator(
                detector,
                {
                    "postprocess_mode": "original_andi",
                    "original_andi": policy_config,
                    "thr_start": 0.1,
                    "thr_end": 0.3,
                    "thr_step": 0.1,
                    "progress": False,
                    "prediction_output": {
                        "enabled": True,
                        "directory": str(root / "predictions"),
                        "normalization_scope": "dataset",
                        "restore_native_grid": True,
                        "save_threshold_mask": True,
                    },
                },
            )
            processed = evaluator.process_raw_maps(raw)
            prediction_processed = evaluator._prediction_postprocess(raw, processed)
            self.assertIs(prediction_processed, processed)
            metadata = [
                {"subject_id": f"case_{index}", "reference_path": str(reference_path)}
                for index in range(raw.shape[0])
            ]

            evaluator._export_predictions(raw, metadata, processed=prediction_processed)
            scores, scores_mf = evaluator.summarize_processed(processed, labels)

            self.assertIs(evaluator.last_prediction_processed, processed)
            for index in range(raw.shape[0]):
                subject_dir = root / "predictions" / f"case_{index}"
                exported_raw = np.asarray(
                    nib.load(str(subject_dir / "anomaly_score_raw.nii.gz")).dataobj,
                    dtype=np.float32,
                )
                exported_mf = np.asarray(
                    nib.load(str(subject_dir / "anomaly_score_mf.nii.gz")).dataobj,
                    dtype=np.float32,
                )
                exported_yen = np.asarray(
                    nib.load(str(subject_dir / "lesion_mask_yen.nii.gz")).dataobj,
                    dtype=np.uint8,
                )
                exported_yen_raw = np.asarray(
                    nib.load(str(subject_dir / "lesion_mask_yen_raw.nii.gz")).dataobj,
                    dtype=np.uint8,
                )
                exported_yen_mf = np.asarray(
                    nib.load(str(subject_dir / "lesion_mask_yen_mf.nii.gz")).dataobj,
                    dtype=np.uint8,
                )
                np.testing.assert_allclose(exported_raw, processed.score_raw[index].numpy(), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(exported_mf, processed.score_mf[index].numpy(), rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(
                    exported_yen_raw.astype(bool),
                    processed.yen_mask_raw_postprocessed[index].numpy(),
                )
                np.testing.assert_array_equal(
                    exported_yen_mf.astype(bool),
                    processed.yen_mask_mf_postprocessed[index].numpy(),
                )
                np.testing.assert_array_equal(
                    exported_yen,
                    exported_yen_mf,
                )
                metadata_payload = json.loads(
                    (subject_dir / "prediction_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata_payload["postprocess_mode"], "original_andi")
                self.assertEqual(metadata_payload["normalization_scope"], "dataset")
                self.assertEqual(
                    metadata_payload["prediction_output"]["normalization_scope"],
                    "dataset",
                )
                self.assertEqual(metadata_payload["yen_source"], "score_mf")
                self.assertAlmostEqual(
                    metadata_payload["yen_threshold_raw"],
                    processed.yen_thresholds_raw[index].item(),
                    places=7,
                )
                self.assertAlmostEqual(
                    metadata_payload["yen_threshold_mf"],
                    processed.yen_thresholds_mf[index].item(),
                    places=7,
                )
                self.assertAlmostEqual(
                    metadata_payload["yen_threshold"],
                    processed.yen_thresholds_mf[index].item(),
                    places=7,
                )

            evaluator.prediction_output["directory"] = str(root / "predictions_without_yen")
            evaluator.prediction_output["save_yen_mask"] = False
            evaluator._export_predictions(raw, metadata, processed=prediction_processed)
            for index in range(raw.shape[0]):
                subject_dir = root / "predictions_without_yen" / f"case_{index}"
                for filename in (
                    "lesion_mask_yen_raw.nii.gz",
                    "lesion_mask_yen_mf.nii.gz",
                    "lesion_mask_yen.nii.gz",
                ):
                    self.assertFalse((subject_dir / filename).exists())
            self.assertAlmostEqual(scores["AUPRC"], auprc(processed.score_raw, labels), places=7)
            self.assertAlmostEqual(scores_mf["AUPRC"], auprc(processed.score_mf, labels), places=7)

    def test_legacy_config_warns_and_preserves_rewrite_result(self) -> None:
        config = {
            "median_filter": {"enabled": True, "kernel_size": 3, "mode": "3d"},
            "postprocess": {
                "score": {"pipeline": [{"type": "normalize"}]},
                "score_mf": {
                    "pipeline": [
                        {"type": "median_filter", "kernel_size": 3, "mode": "3d"},
                        {"type": "normalize"},
                    ]
                },
                "yen_mask": {"pipeline": []},
            },
        }
        raw = torch.linspace(0.0, 2.0, 250).reshape(2, 5, 5, 5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = build_postprocess_policy(config)

        finite = sanitize_scores(raw.float())
        legacy_raw = normalize_minmax(finite)
        legacy_raw = apply_score_postprocess(legacy_raw, config["postprocess"]["score"])
        legacy_raw = normalize_minmax(sanitize_scores(legacy_raw))
        legacy_mf = apply_score_postprocess(legacy_raw, config["postprocess"]["score_mf"])
        legacy_mf = normalize_minmax(sanitize_scores(legacy_mf))
        processed = policy.process(raw)

        self.assertTrue(any("postprocess_mode is not set" in str(item.message) for item in caught))
        self.assertTrue(policy.describe()["legacy_compatibility"])
        torch.testing.assert_close(processed.score_raw, legacy_raw, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(processed.score_mf, legacy_mf, rtol=1e-6, atol=1e-7)

    def test_rewrite_prediction_can_use_explicit_subject_scope(self) -> None:
        rewrite_config = {
            "postprocess_mode": "rewrite",
            "postprocess": {
                "score": {"pipeline": [{"type": "normalize"}]},
                "score_mf": {"pipeline": [{"type": "normalize"}]},
                "yen_mask": {"pipeline": []},
            },
        }
        policy = build_postprocess_policy(rewrite_config)
        detector = self._detector(policy)  # type: ignore[arg-type]
        evaluator = VolumeEvaluator(
            detector,
            {
                **rewrite_config,
                "prediction_output": {
                    "enabled": True,
                    "normalization_scope": "subject",
                    "restore_native_grid": False,
                },
                "progress": False,
            },
        )
        raw = torch.stack(
            [
                torch.linspace(0.0, 1.0, 27).reshape(3, 3, 3),
                torch.linspace(0.0, 100.0, 27).reshape(3, 3, 3),
            ]
        )

        metric_processed = evaluator.process_raw_maps(raw, normalization_scope="dataset")
        prediction_processed = evaluator._prediction_postprocess(raw, metric_processed)

        self.assertAlmostEqual(float(metric_processed.score_raw[0].max()), 0.01, places=7)
        self.assertAlmostEqual(float(prediction_processed.score_raw[0].max()), 1.0, places=7)
        self.assertEqual(prediction_processed.normalization_scope, "subject")

    def test_numerical_safety_and_empty_foreground(self) -> None:
        policy_config = {
            "normalization_scope": "dataset",
            "median_filter": {"enabled": False, "kernel_size": 5, "mode": "3d"},
            "yen": {"binary_dilation": {"enabled": False}},
        }
        policy = OriginalANDiPostprocessPolicy(policy_config)
        cases = {
            "constant": torch.full((2, 3, 3, 3), 4.0),
            "nan": torch.tensor([float("nan"), 0.0, 1.0]).repeat(18).reshape(2, 3, 3, 3),
            "posinf": torch.tensor([float("inf"), 0.0, 1.0]).repeat(18).reshape(2, 3, 3, 3),
            "neginf": torch.tensor([float("-inf"), 0.0, 1.0]).repeat(18).reshape(2, 3, 3, 3),
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                processed = policy.process(raw)
                self.assertTrue(torch.isfinite(processed.score_raw).all())
                self.assertTrue(torch.isfinite(processed.score_mf).all())
                self.assertTrue(torch.isfinite(processed.yen_thresholds_raw).all())
                self.assertEqual(tuple(processed.score_raw.shape), (2, 3, 3, 3))
        constant = policy.process(cases["constant"])
        self.assertEqual(int(torch.count_nonzero(constant.score_raw)), 0)
        self.assertEqual(int(torch.count_nonzero(constant.yen_mask_raw_postprocessed)), 0)

        detector = self._detector(policy)
        evaluator = VolumeEvaluator(
            detector,
            {
                "postprocess_mode": "original_andi",
                "original_andi": policy_config,
                "thr_start": 0.1,
                "thr_end": 0.3,
                "thr_step": 0.1,
                "compute_auprc": False,
                "progress": False,
            },
        )
        processed = policy.process(torch.rand((2, 3, 3, 3)))
        scores, scores_mf = evaluator.summarize_processed(
            processed,
            torch.zeros((2, 3, 3, 3), dtype=torch.bool),
        )
        self.assertTrue(np.isfinite(scores["yen"]))
        self.assertTrue(np.isfinite(scores_mf["yen"]))

    def test_policy_description_and_eval_config_are_explicit(self) -> None:
        policy = OriginalANDiPostprocessPolicy(_original_policy_config(kernel_size=5))
        description = policy.describe()

        self.assertEqual(description["postprocess_mode"], "original_andi")
        self.assertEqual(description["normalization_scope"], "dataset")
        self.assertEqual(description["threshold_method"], "yen")
        self.assertEqual(description["raw_score_pipeline"], ["nan_to_num", "dataset_minmax"])
        self.assertEqual(
            description["mf_score_pipeline"],
            ["nan_to_num", "median_filter_3d(kernel=5)", "dataset_minmax"],
        )
        self.assertEqual(description["yen_threshold_strategy"], "per_subject_3d_volume")
        self.assertEqual(description["dilation_settings"]["connectivity"], 1)

        with (REPO_ROOT / "configs" / "eval.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["metrics"]["postprocess_mode"], "rewrite")
        self.assertEqual(config["metrics"]["threshold_method"], "yen")
        self.assertEqual(
            config["metrics"]["original_andi"]["normalization_scope"],
            "dataset",
        )
        self.assertEqual(
            config["prediction_output"]["normalization_scope"],
            {"rewrite": "subject", "original_andi": "dataset"},
        )
        switched_metrics = {**config["metrics"], "postprocess_mode": "original_andi"}
        switched_evaluator = VolumeEvaluator(
            self._detector(OriginalANDiPostprocessPolicy(config["metrics"]["original_andi"])),
            {
                **switched_metrics,
                "prediction_output": config["prediction_output"],
                "progress": False,
            },
        )
        self.assertEqual(switched_evaluator.prediction_normalization_scope, "dataset")
        with (REPO_ROOT / "configs" / "eval_original_andi.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            original_config = yaml.safe_load(handle)
        self.assertEqual(original_config["metrics"]["postprocess_mode"], "original_andi")
        self.assertEqual(original_config["prediction_output"]["normalization_scope"], "dataset")

    def test_inference_report_records_resolved_policy(self) -> None:
        policy_config = _original_policy_config(kernel_size=5)
        policy = OriginalANDiPostprocessPolicy(policy_config)
        detector = self._detector(policy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_csv = root / "ANDi.csv"
            mf_csv = root / "ANDi_mf.csv"
            evaluator = VolumeEvaluator(
                detector,
                {
                    "postprocess_mode": "original_andi",
                    "original_andi": policy_config,
                    "output_csv": str(raw_csv),
                    "output_mf_csv": str(mf_csv),
                    "prediction_output": {
                        "enabled": False,
                        "normalization_scope": "dataset",
                    },
                    "progress": False,
                },
            )
            evaluator.write_original_style_csv({"AUPRC": 0.5, "yen": 0.4}, raw_csv)
            evaluator.write_original_style_csv({"AUPRC": 0.6, "yen": 0.45}, mf_csv)
            config = {
                "experiment": {"name": "report-test"},
                "metrics": {
                    "postprocess_mode": "original_andi",
                    "original_andi": policy_config,
                    "output_csv": str(raw_csv),
                    "output_mf_csv": str(mf_csv),
                },
                "prediction_output": {"normalization_scope": "dataset"},
            }

            report_dir = save_inference_report(
                config=config,
                evaluator=evaluator,
                result={
                    "output": str(raw_csv),
                    "output_mf": str(mf_csv),
                    "postprocessing": policy.describe(),
                },
                output_dir=root / "report",
            )

            self.assertIsNotNone(report_dir)
            payload = json.loads((root / "report" / "inference_report.json").read_text(encoding="utf-8"))
            settings = payload["post_processing_settings"]
            self.assertEqual(settings["postprocess_mode"], "original_andi")
            self.assertEqual(settings["normalization_scope"], "dataset")
            self.assertEqual(settings["raw_score_pipeline"], ["nan_to_num", "dataset_minmax"])
            self.assertIn("median_filter_3d(kernel=5)", settings["mf_score_pipeline"])
            self.assertEqual(settings["yen_threshold_strategy"], "per_subject_3d_volume")
            self.assertEqual(settings["dilation_settings"]["connectivity"], 1)
            self.assertEqual(settings["export_normalization_scope"], "dataset")


if __name__ == "__main__":
    unittest.main()
