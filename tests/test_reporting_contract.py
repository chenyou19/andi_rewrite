"""Compatibility contracts for the modular reporting implementation."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[1]
FACADE_MODULE = "andi_rewrite.utils.reporting"
REPORTING_PACKAGE = "andi_rewrite.reporting"


class _Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _Model:
    def parameters(self):
        return iter((_Parameter(2), _Parameter(5)))


class _FakeTrainer:
    def __init__(self, root: Path) -> None:
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-3, "weight_decay": 1.0e-4}])
        self.model = _Model()
        self.device = "cpu"
        self.noise_plan = {"kind": "contract"}
        self.checkpoint_dir = root / "checkpoints"
        self.last_sample_path = root / "sample.pt"
        self.last_checkpoint_path = root / "checkpoint.pt"
        self.last_loss = 0.125
        self.best_loss = 0.1
        self.last_epoch = 3


class ReportingContractTest(unittest.TestCase):
    """Exercise reports as a dependency leaf with stable artifacts and imports."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.facade = importlib.import_module(FACADE_MODULE)
        cls.reporting = importlib.import_module(REPORTING_PACKAGE)

    def test_facade_exports_direct_owner_objects(self) -> None:
        owners = {
            "safe_get": ("serialization", "safe_get"),
            "snapshot_config": ("serialization", "snapshot_config"),
            "SUMMARY_COLUMNS": ("metrics_csv", "SUMMARY_COLUMNS"),
            "summarize_eval_metrics": ("metrics_csv", "summarize_eval_metrics"),
            "save_training_report": ("io", "save_training_report"),
            "save_inference_report": ("io", "save_inference_report"),
        }

        for facade_name, (owner_module, owner_name) in owners.items():
            owner = importlib.import_module(f"{REPORTING_PACKAGE}.{owner_module}")
            with self.subTest(facade_name=facade_name, owner=owner_module):
                self.assertIs(getattr(self.facade, facade_name), getattr(owner, owner_name))

        self.assertIs(self.reporting.save_training_report, self.facade.save_training_report)
        self.assertIs(self.reporting.save_inference_report, self.facade.save_inference_report)

    def test_facade_imports_from_package_and_top_level_contexts(self) -> None:
        contexts = (
            (REPO_ROOT.parent, "andi_rewrite.utils.reporting"),
            (REPO_ROOT, "utils.reporting"),
        )
        script_template = (
            "import importlib\n"
            "module = importlib.import_module({module_name!r})\n"
            "assert callable(module.save_training_report)\n"
            "assert callable(module.save_inference_report)\n"
            "assert callable(module.summarize_eval_metrics)\n"
            "assert module.SUMMARY_COLUMNS\n"
        )

        for cwd, module_name in contexts:
            with self.subTest(module_name=module_name):
                completed = subprocess.run(
                    [sys.executable, "-c", script_template.format(module_name=module_name)],
                    cwd=cwd,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"failed to import {module_name}:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}",
                )

    def test_parser_preserves_five_column_threshold_rows_and_legacy_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            five_column = root / "legacy-five-column.csv"
            metric_value = root / "metric-value.csv"
            five_column.write_text(
                "thr,value,dice,sensitivity,precision\n"
                "0.10,,0.20,0.30,0.40\n"
                "0.20,,0.90,0.80,0.70\n"
                "yun,0.51,,,\n"
                "yunthr,0.26,,,\n"
                "yunsen,0.61,,,\n"
                "yunpre,0.71,,,\n"
                "AUPRC,0.81,,,\n",
                encoding="utf-8",
            )
            metric_value.write_text(
                "metric,value\n"
                "Average Precision,0.91\n"
                "best dice,0.77\n"
                "best threshold,0.27\n"
                "Otsu,0.67\n"
                "Otsu Threshold,0.37\n"
                "Otsu Sensitivity,0.57\n"
                "Otsu Precision,0.47\n",
                encoding="utf-8",
            )

            summary = self.facade.summarize_eval_metrics(five_column, metric_value)

        raw = summary["raw"]
        self.assertEqual(raw["source_csv"], str(five_column))
        self.assertEqual(raw["AUPRC"], 0.81)
        self.assertEqual(raw["bestdice"], 0.9)
        self.assertEqual(raw["bestthr"], 0.2)
        self.assertEqual(raw["bestsen"], 0.8)
        self.assertEqual(raw["bestpre"], 0.7)
        self.assertEqual(raw["yendice"], 0.51)
        self.assertEqual(raw["yenthr"], 0.26)
        self.assertEqual(raw["yensen"], 0.61)
        self.assertEqual(raw["yenpre"], 0.71)
        self.assertEqual(raw["adaptive_method"], "yen")
        self.assertEqual(raw["adaptive_dice"], 0.51)
        self.assertEqual(raw["adaptive_threshold"], 0.26)

        median_filter = summary["median_filter"]
        self.assertEqual(median_filter["source_csv"], str(metric_value))
        self.assertEqual(median_filter["AUPRC"], 0.91)
        self.assertEqual(median_filter["bestdice"], 0.77)
        self.assertEqual(median_filter["bestthr"], 0.27)
        self.assertEqual(median_filter["otsudice"], 0.67)
        self.assertEqual(median_filter["otsuthr"], 0.37)
        self.assertEqual(median_filter["otsusen"], 0.57)
        self.assertEqual(median_filter["otsupre"], 0.47)
        self.assertEqual(median_filter["adaptive_method"], "otsu")
        self.assertEqual(median_filter["adaptive_dice"], 0.67)
        self.assertEqual(median_filter["adaptive_threshold"], 0.37)
        self.assertEqual(median_filter["adaptive_sensitivity"], 0.57)
        self.assertEqual(median_filter["adaptive_precision"], 0.47)

    def test_training_report_has_stable_artifact_names_and_core_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "reports"
            config = {
                "experiment": {"name": "report-contract"},
                "runtime": {"seed": 73, "device": "cpu", "mixed_precision": False},
                "data": {
                    "type": "healthy_slices",
                    "path": "healthy.lmdb",
                    "path_to_csv": "healthy_slices.csv",
                    "image_size": 32,
                    "channels": 4,
                    "batch_size": 2,
                    "workers": 0,
                    "shuffle": True,
                    "sampling_mode": "z_balanced",
                    "per_z_count": 3,
                },
                "model": {"type": "contract-model", "in_channels": 4, "base_channels": 8},
                "diffusion": {"steps": 2, "beta_start": 0.001, "beta_end": 0.02},
                "noise": {"schedule": {"type": "static"}},
                "training": {
                    "run_name": "training-contract",
                    "epochs": 3,
                    "checkpoint": {"dir": str(root / "configured-checkpoints")},
                    "scheduler": {"type": "cosine", "start_lr": 1.0e-4, "target_lr": 1.0e-3},
                    "ema": {"decay": 0.999},
                },
            }
            result = self.facade.save_training_report(
                config=config,
                trainer=_FakeTrainer(root),
                dataloader=SimpleNamespace(dataset=["one", "two", "three"]),
                start_time=datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 8, 13, 9, 0, 2, tzinfo=timezone.utc),
                config_path=root / "training.yaml",
                cli_args={"config": "training.yaml"},
                output_dir=report_dir,
            )

            self.assertEqual(result, report_dir)
            self.assertEqual(
                sorted(path.name for path in report_dir.iterdir()),
                ["training_report.json", "training_report.md"],
            )
            payload = json.loads((report_dir / "training_report.json").read_text(encoding="utf-8"))
            markdown = (report_dir / "training_report.md").read_text(encoding="utf-8")

        self.assertEqual(
            set(payload),
            {
                "basic_info",
                "dataset_settings",
                "model_settings",
                "diffusion_noise_settings",
                "optimizer_scheduler_settings",
                "training_result",
                "full_config_snapshot",
            },
        )
        self.assertEqual(payload["basic_info"]["experiment_name"], "report-contract")
        self.assertEqual(payload["basic_info"]["output_directory"], str(report_dir))
        self.assertEqual(payload["basic_info"]["checkpoint_output_directory"], str(root / "checkpoints" / "training-contract"))
        self.assertEqual(payload["basic_info"]["total_training_time_seconds"], 2.0)
        self.assertEqual(payload["dataset_settings"]["number_of_training_samples"], 3)
        self.assertTrue(payload["dataset_settings"]["z_balanced_sampling"])
        self.assertEqual(payload["model_settings"]["parameter_count"], 7)
        self.assertEqual(payload["optimizer_scheduler_settings"]["learning_rate"], 1.0e-3)
        self.assertEqual(payload["training_result"]["final_loss"], 0.125)
        self.assertEqual(payload["training_result"]["best_loss"], 0.1)
        self.assertEqual(payload["training_result"]["last_epoch"], 3)
        self.assertEqual(payload["full_config_snapshot"], config)
        self.assertTrue(markdown.startswith("# Training Report\n"))

    def test_report_write_failure_warns_and_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocked_path = Path(directory) / "not-a-report-directory"
            blocked_path.write_text("file blocks report directory creation", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = self.facade.save_training_report(
                    config={},
                    trainer=object(),
                    output_dir=blocked_path,
                )

        self.assertIsNone(result)
        self.assertIn("Warning: failed to write training report:", stdout.getvalue())

    def test_fresh_reporting_imports_are_engine_independent(self) -> None:
        modules = (
            REPORTING_PACKAGE,
            f"{REPORTING_PACKAGE}.serialization",
            f"{REPORTING_PACKAGE}.metadata",
            f"{REPORTING_PACKAGE}.metrics_csv",
            f"{REPORTING_PACKAGE}.training",
            f"{REPORTING_PACKAGE}.inference",
            f"{REPORTING_PACKAGE}.io",
            FACADE_MODULE,
        )
        script = (
            "import importlib\n"
            "import sys\n"
            f"for module in {modules!r}:\n"
            "    importlib.import_module(module)\n"
            "engine_modules = sorted(\n"
            "    name for name in sys.modules\n"
            "    if name == 'andi_rewrite.engine' or name.startswith('andi_rewrite.engine.')\n"
            ")\n"
            "if engine_modules:\n"
            "    raise RuntimeError(f'reporting imported engine modules: {engine_modules}')\n"
        )
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
            "fresh reporting import failed or imported engine:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
