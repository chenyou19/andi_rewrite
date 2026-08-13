"""Regression coverage for compatibility of all checked-in YAML configs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from andi_rewrite.anomaly.postprocess import (  # noqa: E402
    PostprocessPolicy,
    build_postprocess_policy,
)
from andi_rewrite.data.datasets import DATASET_BUILDERS  # noqa: E402
from andi_rewrite.utils import load_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


class ConfigCompatibilityTest(unittest.TestCase):
    """Keep historical and current config shapes compatible with public facades."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config_paths = tuple(sorted(CONFIG_ROOT.rglob("*.yaml")))
        cls.configs = {
            path: load_config(path)
            for path in cls.config_paths
        }

    @staticmethod
    def _relative(path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix()

    def test_every_checked_in_yaml_loads_as_a_mapping(self) -> None:
        self.assertTrue(self.config_paths, "No checked-in YAML configs were discovered.")

        for path, config in self.configs.items():
            relative_path = self._relative(path)
            with self.subTest(config=relative_path):
                self.assertIsInstance(config, Mapping)
                self.assertEqual(config.get("_config_path"), str(path))

    def test_every_configured_dataset_type_has_a_registered_builder(self) -> None:
        observed: dict[str, list[str]] = {}

        for path, config in self.configs.items():
            data_config = config.get("data")
            if not isinstance(data_config, Mapping) or "type" not in data_config:
                continue

            relative_path = self._relative(path)
            data_type = data_config["type"]
            with self.subTest(config=relative_path, data_type=data_type):
                self.assertIsInstance(data_type, str, "data.type must be a string")
                if not isinstance(data_type, str):
                    continue

                # Match the public factory normalization without invoking a
                # builder, which keeps this compatibility check data-free.
                normalized_type = str(data_type).lower()
                observed.setdefault(normalized_type, []).append(relative_path)
                self.assertIn(normalized_type, DATASET_BUILDERS)

        self.assertTrue(observed, "No checked-in YAML data.type values were discovered.")

    def test_every_eval_config_builds_a_postprocess_policy(self) -> None:
        eval_paths = tuple(
            path for path in self.config_paths if path.name.startswith("eval")
        )
        self.assertTrue(eval_paths, "No eval*.yaml configs were discovered.")

        relative_paths = {self._relative(path) for path in eval_paths}
        self.assertIn("configs/eval.yaml", relative_paths)
        self.assertIn("configs/eval_original_andi.yaml", relative_paths)

        discovered = {
            "legacy": False,
            "current": False,
            "ucsf": False,
            "brats": False,
            "empirical_spectrum": False,
            "pyramid": False,
        }
        for path in eval_paths:
            relative_path = self._relative(path)
            config = self.configs[path]
            metrics_config = config.get("metrics", {})
            anomaly_config = config.get("anomaly", {})

            with self.subTest(config=relative_path):
                self.assertIsInstance(metrics_config, Mapping, "metrics must be a mapping")
                self.assertIsInstance(anomaly_config, Mapping, "anomaly must be a mapping")
                if not isinstance(metrics_config, Mapping) or not isinstance(anomaly_config, Mapping):
                    continue

                # Copy only the policy inputs: policy construction must remain
                # independent of configured dataset paths and checkpoints.
                policy = build_postprocess_policy(
                    dict(metrics_config),
                    dict(anomaly_config),
                    warn_legacy=False,
                )
                self.assertIsInstance(policy, PostprocessPolicy)

            normalized_path = relative_path.lower()
            discovered["legacy"] |= "postprocess_mode" not in metrics_config
            discovered["current"] |= "postprocess_mode" in metrics_config
            discovered["ucsf"] |= "ucsf" in normalized_path
            discovered["brats"] |= "brats" in normalized_path
            discovered["empirical_spectrum"] |= "empirical_spectrum" in normalized_path
            discovered["pyramid"] |= "pyramid" in normalized_path

        for family, was_discovered in discovered.items():
            with self.subTest(config_family=family):
                self.assertTrue(was_discovered, f"No {family} eval config was discovered.")

    def test_fresh_interpreter_imports_public_pipeline_entrypoints(self) -> None:
        modules = (
            "andi_rewrite.scripts.train",
            "andi_rewrite.scripts.eval",
            "andi_rewrite.engine.trainer",
            "andi_rewrite.engine.evaluator",
            "andi_rewrite.data.datasets",
            "andi_rewrite.anomaly.postprocess",
            "andi_rewrite.reporting",
        )
        script = "import importlib\n" + (
            f"for name in {modules!r}:\n    importlib.import_module(name)\n"
        )

        with tempfile.TemporaryDirectory() as pycache_directory:
            environment = os.environ.copy()
            environment["PYTHONPYCACHEPREFIX"] = pycache_directory
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT.parent,
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
                env=environment,
            )

        self.assertEqual(
            completed.returncode,
            0,
            "fresh public-entrypoint import failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
