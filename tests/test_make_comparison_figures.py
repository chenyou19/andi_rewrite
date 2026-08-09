from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import make_comparison_figures as figures  # noqa: E402


class MakeComparisonFiguresTest(unittest.TestCase):
    @staticmethod
    def _save(path: Path, array: np.ndarray, affine: np.ndarray | None = None) -> None:
        nib.save(
            nib.Nifti1Image(array, np.eye(4) if affine is None else affine),
            str(path),
        )

    def _write_case(
        self,
        root: Path,
        *,
        yen_threshold: float = 0.375,
        yen_source: str = "score_mf",
        mf_affine: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (4, 5, 3)
        raw = np.linspace(0.0, 1.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
        mf = np.flip(raw, axis=0).copy()
        # Deliberately unrelated to score > threshold. The loader must use this
        # artifact verbatim rather than reconstructing a Yen mask.
        mask = np.zeros(shape, dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[3, 4, 2] = 1

        self._save(root / "anomaly_score_raw.nii.gz", raw)
        self._save(root / "anomaly_score_mf.nii.gz", mf, mf_affine)
        self._save(root / "lesion_mask_yen.nii.gz", mask)
        (root / "prediction_metadata.json").write_text(
            json.dumps(
                {
                    "subject_id": "case_001",
                    "yen_threshold": yen_threshold,
                    "yen_source": yen_source,
                    "postprocess_mode": "original_andi",
                    "normalization_scope": "dataset",
                }
            ),
            encoding="utf-8",
        )
        return raw, mf, mask.astype(bool)

    def _write_branch_products(
        self,
        root: Path,
        *,
        raw_threshold: float = 0.21875,
        mf_threshold: float = 0.6875,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (4, 5, 3)
        raw_mask = np.zeros(shape, dtype=np.uint8)
        raw_mask[0, 1, 0] = 1
        raw_mask[1, 2, 1] = 1
        mf_mask = np.zeros(shape, dtype=np.uint8)
        mf_mask[2, 3, 1] = 1
        mf_mask[3, 4, 2] = 1
        self._save(root / "lesion_mask_yen_raw.nii.gz", raw_mask)
        self._save(root / "lesion_mask_yen_mf.nii.gz", mf_mask)

        metadata_path = root / "prediction_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["yen_threshold_raw"] = raw_threshold
        metadata["yen_threshold_mf"] = mf_threshold
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        selected = raw_mask if str(metadata["yen_source"]).lower() in {"raw", "score_raw"} else mf_mask
        self._save(root / "lesion_mask_yen.nii.gz", selected)
        return raw_mask.astype(bool), mf_mask.astype(bool)

    def test_load_prediction_uses_exported_mask_and_metadata_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, mf, mask = self._write_case(
                root,
                yen_threshold=0.8125,
                yen_source="raw",
            )

            prediction = figures._load_prediction(root, None)

            np.testing.assert_array_equal(prediction.raw_score, raw)
            np.testing.assert_array_equal(prediction.mf_score, mf)
            np.testing.assert_array_equal(prediction.yen_mask, mask)
            self.assertEqual(prediction.yen_source, "score_raw")
            self.assertEqual(prediction.yen_threshold, 0.8125)
            self.assertEqual(prediction.postprocess_mode, "original_andi")
            self.assertEqual(prediction.normalization_scope, "dataset")
            self.assertIsNone(prediction.yen_mask_raw)
            self.assertIsNone(prediction.yen_mask_mf)
            self.assertIsNone(prediction.yen_threshold_raw)
            self.assertIsNone(prediction.yen_threshold_mf)

    def test_load_prediction_uses_exact_exported_branch_masks_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, yen_threshold=0.6875, yen_source="score_mf")
            raw_mask, mf_mask = self._write_branch_products(root)

            prediction = figures._load_prediction(root, None)

            np.testing.assert_array_equal(prediction.yen_mask_raw, raw_mask)
            np.testing.assert_array_equal(prediction.yen_mask_mf, mf_mask)
            np.testing.assert_array_equal(prediction.yen_mask, mf_mask)
            self.assertEqual(prediction.yen_threshold_raw, 0.21875)
            self.assertEqual(prediction.yen_threshold_mf, 0.6875)
            figures._require_comparison_yen_products(prediction)

    def test_compare_rejects_legacy_prediction_without_branch_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root)
            prediction = figures._load_prediction(root, None)

            with self.assertRaisesRegex(figures.FigureInputError, "Re-run the Evaluator"):
                figures._require_comparison_yen_products(prediction)

    def test_compare_rejects_missing_branch_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, yen_threshold=0.6875)
            self._write_branch_products(root)
            metadata_path = root / "prediction_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["yen_threshold_raw"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            prediction = figures._load_prediction(root, None)

            with self.assertRaisesRegex(figures.FigureInputError, "yen_threshold_raw"):
                figures._require_comparison_yen_products(prediction)

    def test_nonbinary_exported_branch_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, yen_threshold=0.6875)
            self._write_branch_products(root)
            invalid = np.zeros((4, 5, 3), dtype=np.float32)
            invalid[0, 0, 0] = 0.5
            self._save(root / "lesion_mask_yen_raw.nii.gz", invalid)

            with self.assertRaisesRegex(figures.FigureInputError, "binary 0/1 mask"):
                figures._load_prediction(root, None)

    def test_mismatched_exported_branch_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, yen_threshold=0.6875)
            raw_mask, _ = self._write_branch_products(root)
            affine = np.eye(4)
            affine[1, 3] = 3.0
            self._save(root / "lesion_mask_yen_raw.nii.gz", raw_mask.astype(np.uint8), affine)

            with self.assertRaisesRegex(figures.FigureInputError, "refusing to resample"):
                figures._load_prediction(root, None)

    def test_compare_layout_uses_raw_and_mf_branch_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_a = root / "a"
            case_b = root / "b"
            case_a.mkdir()
            case_b.mkdir()
            for case in (case_a, case_b):
                self._write_case(case, yen_threshold=0.6875)
                self._write_branch_products(case)
            prediction_a = figures._load_prediction(case_a, None)
            prediction_b = figures._load_prediction(case_b, None)
            shape = prediction_a.raw_score.shape
            inputs = figures.InputVolumes(
                arrays={name: np.zeros(shape, dtype=np.float32) for name in figures.MODALITIES},
                display_limits={name: (0.0, 1.0) for name in figures.MODALITIES},
                gt=np.zeros(shape, dtype=bool),
            )
            metrics = figures.EvaluatorYenMetrics(
                csv_path=root / "ANDi_mf.csv",
                mean_threshold=0.5,
                dice=0.4,
                sensitivity=0.3,
                precision=0.2,
            )

            figure = figures._compare_figure(
                prediction_a,
                prediction_b,
                inputs,
                metrics,
                metrics,
                "A",
                "B",
                "manual",
                1,
            )
            try:
                titles = {axis.get_title() for axis in figure.axes}
                self.assertIn("Model A Yen Mask before MF (thr=0.218750)", titles)
                self.assertIn("Model A Yen Mask after MF (thr=0.687500)", titles)
                self.assertIn("Model A MF Error Map", titles)
                self.assertIn("Model A MF Mask on FLAIR", titles)
                self.assertIn("Model B Yen Mask before MF (thr=0.218750)", titles)
                self.assertIn("Model B Yen Mask after MF (thr=0.687500)", titles)
            finally:
                figures.plt.close(figure)

    def test_missing_metadata_threshold_is_rejected_instead_of_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root)
            metadata_path = root / "prediction_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["yen_threshold"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(figures.FigureInputError, "will not recompute"):
                figures._load_prediction(root, None)

    def test_nonbinary_exported_yen_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root)
            invalid = np.zeros((4, 5, 3), dtype=np.float32)
            invalid[0, 0, 0] = 0.5
            self._save(root / "lesion_mask_yen.nii.gz", invalid)

            with self.assertRaisesRegex(figures.FigureInputError, "binary 0/1 mask"):
                figures._load_prediction(root, None)

    def test_mismatched_evaluator_artifact_grids_are_not_resampled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.eye(4)
            affine[0, 3] = 2.0
            self._write_case(root, mf_affine=affine)

            with self.assertRaisesRegex(figures.FigureInputError, "refusing to resample"):
                figures._load_prediction(root, None)

    def test_evaluator_metrics_are_loaded_verbatim_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ANDi_mf.csv"
            path.write_text(
                "thr,value,dice,sensitivity,precision\n"
                "yen,0.5760646855179183,,,\n"
                "yenthr,0.12404581159353256,,,\n"
                "yensen,0.5751360718014795,,,\n"
                "yenpre,0.9393405030405945,,,\n",
                encoding="utf-8",
            )

            metrics = figures._read_evaluator_yen_metrics(path)

            self.assertEqual(metrics.csv_path, path)
            self.assertEqual(metrics.dice, 0.5760646855179183)
            self.assertEqual(metrics.mean_threshold, 0.12404581159353256)
            self.assertEqual(metrics.sensitivity, 0.5751360718014795)
            self.assertEqual(metrics.precision, 0.9393405030405945)

    def test_missing_evaluator_csv_is_rejected_instead_of_recalculated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root)
            prediction = figures._load_prediction(root, None)
            with (
                mock.patch.object(figures, "_metadata_metrics_csv", return_value=None),
                mock.patch.object(figures, "_discover_metrics_csv", return_value=None),
                self.assertRaisesRegex(figures.FigureInputError, "will not recalculate"),
            ):
                figures._resolve_metrics_csv(prediction, None, "--metrics-csv")


if __name__ == "__main__":
    unittest.main()
