"""Synthetic contract tests for BraTS-to-MPI preprocessing and restoration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.data.brats_mpi.geometry import (  # noqa: E402
    CropWindow,
    crop_resize_xy,
    restore_model_to_source,
)
from andi_rewrite.data.brats_mpi.manifest import BraTSMPIRecord  # noqa: E402
from andi_rewrite.data.brats_mpi.processing import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    _resample_affine,
    process_subject,
)
from andi_rewrite.data.datasets.brats_mpi import BraTSMPICacheDataset  # noqa: E402
from andi_rewrite.anomaly.postprocess import PostprocessResult  # noqa: E402
from andi_rewrite.engine.evaluation.prediction_export import (  # noqa: E402
    export_predictions,
    restore_spatial_volume,
)


class BraTSMPIGeometryTest(unittest.TestCase):
    def test_mask_resize_preserves_categorical_labels_and_restore_shape(self) -> None:
        segmentation = np.zeros((12, 12, 3), dtype=np.uint8)
        segmentation[2:5, 2:5, :] = 1
        segmentation[5:8, 5:8, :] = 2
        segmentation[8:11, 8:11, :] = 4
        window = CropWindow(1, 11, 1, 11)
        model = crop_resize_xy(segmentation, window, 20, is_mask=True)
        self.assertTrue(set(np.unique(model)).issubset({0, 1, 2, 4}))
        restored = restore_model_to_source(
            model,
            source_shape=segmentation.shape,
            window=window,
            z_start=0,
            continuous=False,
        )
        self.assertEqual(restored.shape, segmentation.shape)
        self.assertTrue(set(np.unique(restored)).issubset({0, 1, 2, 4}))

    def test_identity_affine_nearest_resampling_keeps_zero_one_two_four(self) -> None:
        segmentation = np.zeros((8, 9, 7), dtype=np.uint8)
        segmentation[1:3] = 1
        segmentation[3:5] = 2
        segmentation[5:7] = 4
        restored = _resample_affine(
            segmentation,
            np.eye(4),
            segmentation.shape,
            np.eye(4),
            np.eye(4),
            order=0,
        )
        np.testing.assert_array_equal(restored, segmentation)


class BraTSMPIFixedModeTest(unittest.TestCase):
    def _synthetic_subject(self, root: Path) -> BraTSMPIRecord:
        subject_id = "BraTS2021_SYNTH"
        subject_dir = root / subject_id
        subject_dir.mkdir(parents=True)
        shape = (240, 240, 8)
        affine = np.array(
            [[-1.0, 0.0, 0.0, 239.0], [0.0, -1.0, 0.0, 239.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        base = np.zeros(shape, dtype=np.float32)
        base[40:200, 30:210, 1:7] = 20.0
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[80:100, 80:100, 2:5] = 1
        segmentation[120:135, 100:120, 3:6] = 2
        # Label 4 deliberately lies outside T1>0 but inside the fixed FOV.
        segmentation[205:215, 70:90, 2:6] = 4

        paths: dict[str, Path] = {}
        for modality, scale in (("flair", 0.75), ("t1", 1.0), ("t2", 1.25)):
            path = subject_dir / f"{subject_id}_{modality}.nii.gz"
            nib.save(nib.Nifti1Image(base * scale, affine), str(path))
            paths[modality] = path
        seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
        nib.save(nib.Nifti1Image(segmentation, affine), str(seg_path))
        return BraTSMPIRecord(0, subject_id, paths["flair"], paths["t1"], paths["t2"], seg_path)

    def test_fixed_mode_keeps_gt_independent_from_brain_mask_and_restores_native(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = self._synthetic_subject(Path(temporary))
            processed = process_subject(
                record,
                "ras_fixed_fov",
                {
                    "model_size": 64,
                    "qc": {"foreground_retention_min": 0.999, "lesion_retention_min": 1.0},
                },
            )
            self.assertEqual(processed.image.shape[:3], (3, 64, 64))
            self.assertEqual(processed.segmentation.shape, processed.image.shape[1:])
            self.assertEqual(processed.qc["status"], "PASS")
            self.assertIn(4, np.unique(processed.segmentation))
            self.assertTrue(set(np.unique(processed.segmentation)).issubset({0, 1, 2, 4}))

            restored = restore_spatial_volume(
                torch.from_numpy((processed.segmentation > 0).astype(np.float32)),
                processed.metadata,
                continuous=False,
            )
            self.assertEqual(tuple(restored.shape), (240, 240, 8))
            self.assertGreater(int(restored.sum()), 0)


class BraTSMPICacheDatasetTest(unittest.TestCase):
    def test_adapter_requires_pass_fingerprint_and_derives_whole_tumour(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject_id = "BraTS2021_CACHE"
            subject_dir = root / "mni_affine" / "subjects" / subject_id
            subject_dir.mkdir(parents=True)
            image = np.ones((3, 16, 16, 4), dtype=np.float32)
            segmentation = np.zeros((16, 16, 4), dtype=np.uint8)
            segmentation[1:4] = 1
            segmentation[4:7] = 2
            segmentation[7:10] = 4
            np.savez_compressed(
                subject_dir / "volume.npz",
                image=image,
                segmentation=segmentation,
                brain_mask=np.ones_like(segmentation, dtype=np.uint8),
            )
            metadata = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "subject_id": subject_id,
                "mode": "mni_affine",
                "fingerprint": "synthetic-fingerprint",
                "qc": {"status": "PASS", "failures": []},
                "native_reference_path": str(root / "native.nii.gz"),
                "segmentation_path": str(root / "seg.nii.gz"),
            }
            metadata_path = subject_dir / "spatial_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest = root / "pass.csv"
            manifest.write_text(f"subject_id\n{subject_id}\n", encoding="utf-8")

            dataset = BraTSMPICacheDataset(
                cache_root=root,
                mode="mni_affine",
                path_to_csv=manifest,
                image_size=16,
                return_metadata=True,
            )
            returned_image, whole_tumour, item_metadata = dataset[0]
            self.assertEqual(tuple(returned_image.shape), image.shape)
            self.assertEqual(whole_tumour.dtype, torch.bool)
            np.testing.assert_array_equal(whole_tumour.numpy(), segmentation > 0)
            self.assertEqual(item_metadata["cache_fingerprint"], "synthetic-fingerprint")
            with np.load(subject_dir / "volume.npz", allow_pickle=False) as cached:
                self.assertEqual(set(np.unique(cached["segmentation"])), {0, 1, 2, 4})


class BraTSMPIPredictionExportTest(unittest.TestCase):
    def test_spatial_sidecar_exports_model_and_native_grids(self) -> None:
        class Policy:
            mode = "rewrite"

            @staticmethod
            def fixed_threshold_mask(scores, threshold):
                return scores > threshold

            @staticmethod
            def describe():
                return {"postprocess_mode": "rewrite"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 3), dtype=np.float32), np.eye(4)), str(native_path))
            spatial_path = root / "spatial_metadata.json"
            spatial_path.write_text(
                json.dumps(
                    {
                        "subject_id": "case_dual",
                        "fingerprint": "dual-fingerprint",
                        "qc": {"status": "PASS", "failures": []},
                        "model_affine": np.eye(4).tolist(),
                        "source_shape": [8, 8, 3],
                        "source_affine": np.eye(4).tolist(),
                        "crop_window": {"x_start": 2, "x_stop": 6, "y_start": 2, "y_stop": 6},
                        "z_start": 0,
                        "restore_kind": "canonical_ras_to_native",
                        "native_reference_path": str(native_path),
                    }
                ),
                encoding="utf-8",
            )
            score = torch.linspace(0.0, 1.0, 12).reshape(1, 2, 2, 3)
            mask = score > 0.5
            processed = PostprocessResult(
                score_raw=score,
                score_mf=score,
                thresholds_raw=torch.tensor([0.5]),
                thresholds_mf=torch.tensor([0.5]),
                binary_mask_raw=mask,
                binary_mask_mf=mask,
                binary_mask_raw_postprocessed=mask,
                binary_mask_mf_postprocessed=mask,
                threshold_method="yen",
                normalization_scope="subject",
            )
            next_index = export_predictions(
                score,
                [
                    {
                        "subject_id": "case_dual",
                        "reference_path": str(native_path),
                        "spatial_metadata_path": str(spatial_path),
                        "cache_fingerprint": "dual-fingerprint",
                    }
                ],
                processed=processed,
                prediction_output={
                    "directory": str(root / "predictions"),
                    "restore_native_grid": True,
                    "save_model_grid": True,
                },
                metric_threshold=0.5,
                model_config={"checkpoint": "synthetic.pt", "use_ema": True},
                detector=SimpleNamespace(t_lower=75, t_upper=200),
                postprocess_policy=Policy(),
                prediction_normalization_scope="subject",
                prediction_index=0,
            )
            self.assertEqual(next_index, 1)
            subject_root = root / "predictions" / "case_dual"
            model_score = nib.load(str(subject_root / "model_grid" / "anomaly_score_raw.nii.gz"))
            native_score = nib.load(str(subject_root / "native_grid" / "anomaly_score_raw.nii.gz"))
            self.assertEqual(model_score.shape, (2, 2, 3))
            self.assertEqual(native_score.shape, (8, 8, 3))
            payload = json.loads((subject_root / "prediction_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["model_grid_saved"])
            self.assertTrue(payload["restored_to_native_grid"])
            self.assertEqual(payload["spatial_restore_kind"], "canonical_ras_to_native")


if __name__ == "__main__":
    unittest.main()
