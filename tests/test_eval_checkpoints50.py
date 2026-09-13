from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.scripts.eval_checkpoints50 import (  # noqa: E402
    completed_output,
    selected_checkpoints,
    validate_subject_csv,
)


class EvalCheckpoints50Test(unittest.TestCase):
    def test_checkpoints_are_sorted_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["epoch_0119.pt", "epoch_0019.pt", "latest.pt"]:
                (root / name).touch()
            args = SimpleNamespace(
                checkpoint_dir=str(root),
                pattern="**/*.pt",
                only=None,
            )

            checkpoints = selected_checkpoints(args)

            self.assertEqual(
                [path.stem for path in checkpoints],
                ["epoch_0019", "epoch_0119"],
            )

    def test_duplicate_epoch_snapshots_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "epoch_0019.pt").touch()
            (root / "b" / "epoch_0019.pt").touch()
            args = SimpleNamespace(
                checkpoint_dir=str(root),
                pattern="**/epoch_*.pt",
                only=None,
            )

            with self.assertRaisesRegex(ValueError, "Duplicate epoch"):
                selected_checkpoints(args)

    def test_subject_csv_requires_expected_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "subjects.csv"
            csv_path.write_text(
                "BraTS21ID\nBraTS2021_00001\nBraTS2021_00002\n",
                encoding="utf-8",
            )
            self.assertEqual(
                validate_subject_csv(csv_path, 2),
                ["BraTS2021_00001", "BraTS2021_00002"],
            )
            csv_path.write_text(
                "BraTS21ID\nBraTS2021_00001\nBraTS2021_00001\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate subject"):
                validate_subject_csv(csv_path, 2)

    def test_completed_output_requires_all_artifacts_and_matching_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch_0019.pt"
            checkpoint.touch()
            output_dir = root / "metrics" / checkpoint.stem
            output_dir.mkdir(parents=True)
            for name in [
                "ANDi.csv",
                "ANDi_mf.csv",
                "inference_metrics_summary.csv",
                "inference_report.md",
            ]:
                (output_dir / name).write_text("non-empty\n", encoding="utf-8")
            report_path = output_dir / "inference_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "inference_settings": {
                            "checkpoint_path": str(checkpoint),
                        },
                        "evaluation_result": {"subjects": 50},
                    }
                ),
                encoding="utf-8",
            )
            config = {"metrics": {"output_csv": str(output_dir / "ANDi.csv")}}

            self.assertTrue(completed_output(config, checkpoint, 50))
            (output_dir / "inference_report.md").unlink()
            self.assertFalse(completed_output(config, checkpoint, 50))


if __name__ == "__main__":
    unittest.main()
