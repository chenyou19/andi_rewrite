from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_brats21_251_lemon_models import (
    build_comparison,
    write_comparison,
)
from scripts.run_brats21_251_lemon_comparison import BAD_LOG_PATTERN


class BraTS21LemonComparisonTest(unittest.TestCase):
    @staticmethod
    def _summary(path: Path, *, scale: float) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["version", "AUPRC", "bestdice", "bestthr"],
            )
            writer.writeheader()
            writer.writerow(
                {"version": "raw", "AUPRC": 0.4 * scale, "bestdice": 0.5 * scale, "bestthr": 0.1}
            )
            writer.writerow(
                {
                    "version": "median_filter",
                    "AUPRC": 0.6 * scale,
                    "bestdice": 0.7 * scale,
                    "bestthr": 0.2,
                }
            )

    def test_build_and_write_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.csv"
            cross = root / "cross.csv"
            self._summary(native, scale=1.0)
            self._summary(cross, scale=1.1)
            report = build_comparison(native, cross)
            auprc = next(
                row
                for row in report["rows"]
                if row["version"] == "raw" and row["metric"] == "AUPRC"
            )
            self.assertAlmostEqual(auprc["cross_minus_native"], 0.04)
            self.assertEqual(auprc["winner"], "lemon_brats21_noise")
            threshold = next(
                row
                for row in report["rows"]
                if row["version"] == "raw" and row["metric"] == "bestthr"
            )
            self.assertEqual(threshold["winner"], "not_applicable")
            write_comparison(report, root / "result")
            self.assertTrue((root / "result" / "comparison_summary.csv").is_file())
            self.assertTrue((root / "result" / "comparison_summary.json").is_file())
            self.assertTrue((root / "result" / "comparison_report.md").is_file())

    def test_mismatched_metric_columns_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.csv"
            cross = root / "cross.csv"
            self._summary(native, scale=1.0)
            self._summary(cross, scale=1.0)
            text = cross.read_text(encoding="utf-8").replace("bestthr", "yenthr")
            cross.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                build_comparison(native, cross)

    def test_log_scan_ignores_numerical_safety_config_but_catches_failures(self) -> None:
        self.assertIsNone(BAD_LOG_PATTERN.search("nan_to_num:\n  nan: 0.0\n  posinf: 0.0"))
        self.assertIsNotNone(BAD_LOG_PATTERN.search("loss: nan"))
        self.assertIsNotNone(BAD_LOG_PATTERN.search("CUDA out of memory"))
        self.assertIsNotNone(BAD_LOG_PATTERN.search("Traceback (most recent call last):"))


if __name__ == "__main__":
    unittest.main()
