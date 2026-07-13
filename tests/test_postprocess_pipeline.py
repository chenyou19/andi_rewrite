from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.engine.evaluator import VolumeEvaluator  # noqa: E402
from andi_rewrite.anomaly.postprocess import (  # noqa: E402
    MASK_POSTPROCESSORS,
    _legacy_mask_pipeline,
    apply_mask_postprocess,
    build_postprocess_pipeline,
)


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
        self.assertIn("VolumeEvaluator._yen_metrics()", message)
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


if __name__ == "__main__":
    unittest.main()
