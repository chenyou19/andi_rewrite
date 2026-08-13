"""Public contracts for the modular ``anomaly.postprocess`` package."""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_MODULE = "andi_rewrite.anomaly.postprocess"
ANOMALY_MODULE = "andi_rewrite.anomaly"


class PostprocessPackageContractTest(unittest.TestCase):
    """Exercise caller-visible compatibility, not implementation details."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.postprocess = importlib.import_module(POSTPROCESS_MODULE)
        cls.anomaly = importlib.import_module(ANOMALY_MODULE)

    def test_anomaly_package_reexports_keep_direct_module_identity(self) -> None:
        names = (
            "BasePostprocessor",
            "OriginalANDiPostprocessPolicy",
            "PostprocessPolicy",
            "PostprocessResult",
            "RewritePostprocessPolicy",
            "ScorePipelineSpec",
            "SUPPORTED_THRESHOLD_METHODS",
            "build_postprocess_policy",
            "build_postprocess_pipeline",
            "otsu_threshold",
            "register_mask_postprocessor",
            "register_score_postprocessor",
            "register_threshold_method",
            "supported_threshold_methods",
            "threshold_anomaly_map",
            "yen_threshold",
        )

        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(self.anomaly, name), getattr(self.postprocess, name))

    def test_direct_postprocess_facade_keeps_established_exports(self) -> None:
        names = (
            "BasePostprocessor",
            "NORMALIZATION_SCOPES",
            "SUPPORTED_THRESHOLD_METHODS",
            "SCORE_POSTPROCESSORS",
            "MASK_POSTPROCESSORS",
            "NormalizePostprocessor",
            "MedianFilterPostprocessor",
            "GrayDilationPostprocessor",
            "BinaryDilationPostprocessor",
            "ConnectedComponentsPostprocessor",
            "sanitize_scores",
            "normalize_minmax",
            "median_filter_tensor",
            "gray_dilation_tensor",
            "binary_dilation_tensor",
            "connected_components_tensor",
            "remove_small_components_tensor",
            "apply_postprocess_pipeline",
            "apply_score_postprocess",
            "apply_mask_postprocess",
            "build_postprocess_policy",
            "OriginalANDiPostprocessPolicy",
            "PostprocessPolicy",
            "PostprocessResult",
            "RewritePostprocessPolicy",
            "ScorePipelineSpec",
            "otsu_threshold",
            "threshold_anomaly_map",
            "yen_threshold",
            "register_mask_postprocessor",
            "register_score_postprocessor",
            "register_threshold_method",
            "supported_threshold_methods",
        )

        missing = [name for name in names if not hasattr(self.postprocess, name)]
        self.assertFalse(missing, f"postprocess facade lost direct imports: {missing}")

    def test_registries_remain_shared_and_accept_alias_extensions(self) -> None:
        base = importlib.import_module(f"{POSTPROCESS_MODULE}.base")
        self.assertIs(self.postprocess.SCORE_POSTPROCESSORS, base.SCORE_POSTPROCESSORS)
        self.assertIs(self.postprocess.MASK_POSTPROCESSORS, base.MASK_POSTPROCESSORS)

        for registry, aliases in (
            (self.postprocess.SCORE_POSTPROCESSORS, ("normalize", "minmax", "normalize_minmax")),
            (self.postprocess.SCORE_POSTPROCESSORS, ("median_filter", "median", "mf")),
            (self.postprocess.SCORE_POSTPROCESSORS, ("gray_dilation", "grey_dilation")),
            (self.postprocess.MASK_POSTPROCESSORS, ("binary_dilation", "dilation")),
            (
                self.postprocess.MASK_POSTPROCESSORS,
                ("connected_components", "remove_small_components", "cc"),
            ),
        ):
            canonical = registry[aliases[0]]
            for alias in aliases[1:]:
                with self.subTest(aliases=aliases, alias=alias):
                    self.assertIs(registry[alias], canonical)

        class IdentityPostprocessor(self.postprocess.BasePostprocessor):
            name = "phase2_contract_identity"

            def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor

        registry = self.postprocess.SCORE_POSTPROCESSORS
        aliases = ("phase2_contract_identity", "phase2_contract_identity_alias")
        sentinel = object()
        previous = {name: registry.get(name, sentinel) for name in aliases}
        try:
            registered = self.postprocess.register_score_postprocessor(*aliases)(
                IdentityPostprocessor
            )
            self.assertIs(registered, IdentityPostprocessor)
            self.assertIs(registry[aliases[0]], IdentityPostprocessor)
            self.assertIs(registry[aliases[1]], IdentityPostprocessor)

            steps = self.postprocess.build_postprocess_pipeline(
                {"pipeline": [{"type": aliases[1]}]},
                registry,
                lambda _config: [],
                "score-map",
            )
            self.assertIsInstance(steps[0], IdentityPostprocessor)
        finally:
            for name, value in previous.items():
                if value is sentinel:
                    registry.pop(name, None)
                else:
                    registry[name] = value

    def test_threshold_registry_is_authoritative_extensible_and_restored(self) -> None:
        threshold_module = importlib.import_module(f"{POSTPROCESS_MODULE}.threshold")
        registry = self.postprocess.THRESHOLD_FUNCTION_LOADERS
        self.assertIs(registry, threshold_module.THRESHOLD_FUNCTION_LOADERS)
        self.assertEqual(self.postprocess.SUPPORTED_THRESHOLD_METHODS, ("yen", "otsu"))
        self.assertEqual(self.postprocess.supported_threshold_methods(), tuple(registry))

        direct_name = "phase2_contract_direct_threshold"
        registered_name = "phase2_contract_registered_threshold"
        sentinel = object()
        previous = {
            name: registry.get(name, sentinel)
            for name in (direct_name, registered_name)
        }
        original_order = tuple(registry)

        def mean_loader():
            return lambda array: float(np.mean(array))

        try:
            registry[direct_name] = mean_loader
            registered = self.postprocess.register_threshold_method(registered_name)(
                mean_loader
            )
            self.assertIs(registered, mean_loader)
            self.assertEqual(
                self.postprocess.supported_threshold_methods(),
                (*original_order, direct_name, registered_name),
            )

            scores = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
            mask, thresholds = self.postprocess.threshold_anomaly_map(
                scores,
                method=direct_name.upper(),
            )
            torch.testing.assert_close(thresholds, torch.tensor([1.5]))
            self.assertTrue(
                torch.equal(
                    mask,
                    torch.tensor([[[[False, False, True, True]]]]),
                )
            )

            policy = self.postprocess.build_postprocess_policy(
                {
                    "postprocess_mode": "rewrite",
                    "threshold_method": registered_name,
                    "normalization_scope": "subject",
                    "postprocess": {
                        "score": {"pipeline": []},
                        "score_mf": {"pipeline": []},
                        "threshold_mask": {"pipeline": []},
                        "binary_mask": {"pipeline": []},
                    },
                },
                warn_legacy=False,
            )
            result = policy.process(scores)
            self.assertEqual(result.threshold_method, registered_name)
            torch.testing.assert_close(result.thresholds_raw, torch.tensor([1.5]))
            self.assertTrue(torch.equal(result.binary_mask_raw, mask))

            detector = object.__new__(self.anomaly.ANDiDetector)
            detector.threshold = registered_name
            detector.set_postprocess_policy(policy)
            detector_output = detector.postprocess(scores)
            self.assertEqual(detector.threshold, registered_name)
            self.assertEqual(detector_output["threshold_method"], registered_name)
            self.assertTrue(torch.equal(detector_output["segmentation"], mask))

            eval_script = importlib.import_module("andi_rewrite.scripts.eval")
            args = eval_script.build_parser().parse_args(
                ["--threshold-method", registered_name]
            )
            cli_config = {
                "metrics": {"postprocess_mode": "rewrite"},
                "anomaly": {"threshold": direct_name},
            }
            eval_script.apply_threshold_method_override(
                cli_config,
                args.threshold_method,
            )
            self.assertEqual(
                cli_config["metrics"]["threshold_method"],
                registered_name,
            )
            self.assertEqual(cli_config["anomaly"]["threshold"], registered_name)
        finally:
            for name, value in previous.items():
                if value is sentinel:
                    registry.pop(name, None)
                else:
                    registry[name] = value

        self.assertEqual(tuple(registry), original_order)
        self.assertEqual(self.postprocess.supported_threshold_methods(), original_order)

    def test_builtin_threshold_import_failures_keep_legacy_behavior(self) -> None:
        registry = self.postprocess.THRESHOLD_FUNCTION_LOADERS
        original_yen = registry["yen"]
        original_otsu = registry["otsu"]

        def missing_dependency_loader():
            raise ImportError("synthetic missing dependency")

        scores = torch.tensor([[[[0.0, 1.0, 3.0]]]])
        try:
            registry["yen"] = missing_dependency_loader
            with self.assertWarnsRegex(
                RuntimeWarning,
                "legacy mean fallback for Yen thresholding",
            ) as caught:
                mask, thresholds = self.postprocess.yen_threshold(scores)
            threshold_module = importlib.import_module(f"{POSTPROCESS_MODULE}.threshold")
            self.assertEqual(
                Path(caught.filename).resolve(),
                Path(threshold_module.__file__).resolve(),
            )
            torch.testing.assert_close(thresholds, torch.tensor([4.0 / 3.0]))
            self.assertTrue(
                torch.equal(mask, torch.tensor([[[[False, False, True]]]]))
            )

            registry["otsu"] = missing_dependency_loader
            with self.assertRaisesRegex(
                ImportError,
                "Otsu thresholding requires scikit-image",
            ):
                self.postprocess.otsu_threshold(scores)
        finally:
            registry["yen"] = original_yen
            registry["otsu"] = original_otsu

        self.assertEqual(tuple(registry)[:2], ("yen", "otsu"))

    def test_builtin_streaming_specs_do_not_change_policy_descriptions(self) -> None:
        original = self.postprocess.OriginalANDiPostprocessPolicy(
            {
                "median_filter": {"enabled": True, "kernel_size": 3, "mode": "3d"},
                "binary_mask": {"binary_dilation": {"enabled": False}},
            }
        )
        rewrite = self.postprocess.build_postprocess_policy(
            {
                "postprocess_mode": "rewrite",
                "normalization_scope": "dataset",
                "postprocess": {
                    "score": {"pipeline": [{"type": "normalize"}]},
                    "score_mf": {
                        "pipeline": [
                            {"type": "median_filter", "kernel_size": 3, "mode": "3d"},
                            {"type": "normalize"},
                        ]
                    },
                    "threshold_mask": {"pipeline": []},
                    "binary_mask": {"pipeline": []},
                },
            },
            warn_legacy=False,
        )

        for policy, expected_source in ((original, "raw"), (rewrite, "score_raw")):
            with self.subTest(policy=policy.mode):
                description = policy.describe()
                spec = policy.score_pipeline_spec()
                self.assertIsInstance(spec, self.postprocess.ScorePipelineSpec)
                self.assertEqual(spec.mf_source, expected_source)
                self.assertEqual(policy.describe(), description)

    @staticmethod
    def _result(threshold_method: str):
        postprocess = importlib.import_module(POSTPROCESS_MODULE)
        return postprocess.PostprocessResult(
            score_raw=torch.tensor([0.1]),
            score_mf=torch.tensor([0.2]),
            thresholds_raw=torch.tensor([0.3]),
            thresholds_mf=torch.tensor([0.4]),
            binary_mask_raw=torch.tensor([True]),
            binary_mask_mf=torch.tensor([False]),
            binary_mask_raw_postprocessed=torch.tensor([True]),
            binary_mask_mf_postprocessed=torch.tensor([False]),
            threshold_method=threshold_method,
            normalization_scope="dataset",
        )

    def test_result_uses_method_neutral_fields_and_conditional_yen_aliases(self) -> None:
        tensor_fields = (
            "score_raw",
            "score_mf",
            "thresholds_raw",
            "thresholds_mf",
            "binary_mask_raw",
            "binary_mask_mf",
            "binary_mask_raw_postprocessed",
            "binary_mask_mf_postprocessed",
        )
        yen_aliases = {
            "yen_thresholds_raw": "thresholds_raw",
            "yen_thresholds_mf": "thresholds_mf",
            "yen_mask_raw": "binary_mask_raw",
            "yen_mask_mf": "binary_mask_mf",
            "yen_mask_raw_postprocessed": "binary_mask_raw_postprocessed",
            "yen_mask_mf_postprocessed": "binary_mask_mf_postprocessed",
        }

        otsu = self._result("otsu")
        otsu_payload = otsu.as_dict()
        for field in tensor_fields:
            with self.subTest(method="otsu", field=field):
                self.assertIs(otsu_payload[field], getattr(otsu, field))
        self.assertEqual(otsu_payload["threshold_method"], "otsu")
        self.assertEqual(otsu_payload["normalization_scope"], "dataset")
        self.assertTrue(set(yen_aliases).isdisjoint(otsu_payload))

        yen = self._result("yen")
        yen_payload = yen.as_dict()
        for legacy_name, neutral_name in yen_aliases.items():
            with self.subTest(method="yen", alias=legacy_name):
                self.assertIs(yen_payload[legacy_name], yen_payload[neutral_name])
                self.assertIs(getattr(yen, legacy_name), getattr(yen, neutral_name))

    def test_fresh_interpreter_imports_package_and_submodules(self) -> None:
        modules = (
            ANOMALY_MODULE,
            POSTPROCESS_MODULE,
            f"{POSTPROCESS_MODULE}.base",
            f"{POSTPROCESS_MODULE}.numerics",
            f"{POSTPROCESS_MODULE}.transforms",
            f"{POSTPROCESS_MODULE}.pipeline",
            f"{POSTPROCESS_MODULE}.threshold",
            f"{POSTPROCESS_MODULE}.result",
            f"{POSTPROCESS_MODULE}.policies",
            f"{POSTPROCESS_MODULE}.factory",
        )
        script = "import importlib\n" + f"for name in {modules!r}:\n    importlib.import_module(name)\n"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT.parent,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            completed.returncode,
            0,
            "fresh postprocess package import failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
