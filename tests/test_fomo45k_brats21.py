from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from andi_rewrite.data.datasets import fomo45k_brats21 as adapter
from andi_rewrite.data.fomo45k.brats21 import (
    RegistrationSettings,
    Toolchain,
    build_apply_command,
    build_registration_command,
    classify_modality,
    discover_sessions,
    geometry_matches,
    preflight_report,
)


class FOMO45KBraTS21DiscoveryTest(unittest.TestCase):
    def test_classification_does_not_turn_t1ce_into_t1(self) -> None:
        self.assertEqual(classify_modality("sub_t1.nii.gz"), "t1")
        self.assertEqual(classify_modality("sub_T2w.nii.gz"), "t2")
        self.assertEqual(classify_modality("sub_FLAIR.nii.gz"), "flair")
        self.assertIsNone(classify_modality("sub_t1ce.nii.gz"))
        self.assertIsNone(classify_modality("sub_t1_post_Gd.nii.gz"))
        self.assertIsNone(classify_modality("sub_t1_brainmask.nii.gz"))

    def test_tsv_discovery_is_fail_closed_and_keeps_literal_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "sub_1" / "ses_&#x20;"
            session.mkdir(parents=True)
            affine = np.eye(4)
            for name in ("t1.nii.gz", "flair.nii.gz"):
                nib.save(nib.Nifti1Image(np.ones((3, 4, 2), dtype=np.float32), affine), session / name)
            tsv = root / "metadata.tsv"
            with tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    delimiter="\t",
                    fieldnames=(
                        "participant_id",
                        "session_id",
                        "T1_filename",
                        "T2_filename",
                        "FLAIR_filename",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "participant_id": "sub_1",
                        "session_id": "ses_&#x20;",
                        "T1_filename": "t1.nii.gz",
                        "T2_filename": "t2.nii.gz",
                        "FLAIR_filename": "flair.nii.gz",
                    }
                )
            records = discover_sessions(root, tsv)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].relative_dir, "sub_1/ses_&#x20;")
            self.assertEqual(records[0].status, "PARTIAL")
            self.assertIn("missing_t2", records[0].reasons)

    def test_geometry_match_requires_shape_and_affine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.nii.gz"
            same = root / "same.nii.gz"
            moved = root / "moved.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 4, 2)), np.eye(4)), left)
            nib.save(nib.Nifti1Image(np.ones((3, 4, 2)), np.eye(4)), same)
            affine = np.eye(4)
            affine[0, 3] = 1.0
            nib.save(nib.Nifti1Image(np.ones((3, 4, 2)), affine), moved)
            self.assertTrue(geometry_matches(left, same))
            self.assertFalse(geometry_matches(left, moved))


class FOMO45KBraTS21CommandTest(unittest.TestCase):
    def test_registration_command_has_rigid_then_affine_and_no_syn(self) -> None:
        command = build_registration_command(
            "antsRegistration",
            "fixed.nii.gz",
            "moving.nii.gz",
            "fixed_mask.nii.gz",
            "moving_mask.nii.gz",
            "out_",
            affine=True,
            random_seed=73,
        )
        transforms = [command[index + 1] for index, value in enumerate(command) if value == "--transform"]
        self.assertEqual(transforms, ["Rigid[0.1]", "Affine[0.1]"])
        self.assertNotIn("SyN", " ".join(command))
        self.assertIn("--random-seed", command)

    def test_apply_command_preserves_composite_order_and_interpolation(self) -> None:
        command = build_apply_command(
            "antsApplyTransforms",
            "flair.nii.gz",
            "atlas.nii.gz",
            "out.nii.gz",
            ["t1_to_sri.mat", "flair_to_t1.mat"],
        )
        transforms = [command[index + 1] for index, value in enumerate(command) if value == "-t"]
        self.assertEqual(transforms, ["t1_to_sri.mat", "flair_to_t1.mat"])
        self.assertEqual(command[command.index("-n") + 1], "Linear")

    def test_preflight_reports_all_missing_dependencies(self) -> None:
        report = preflight_report(
            Toolchain(
                ants_registration="definitely_missing_antsRegistration",
                ants_apply_transforms="definitely_missing_antsApplyTransforms",
                n4_bias_field_correction="definitely_missing_N4",
                synthstrip="definitely_missing_synthstrip",
                atlas_t1="missing_t1.nii",
                atlas_brain="missing_brain.nii",
                atlas_mask="missing_mask.nii",
            )
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(len(report["errors"]), 7)


class FOMO45KBraTS21AdapterTest(unittest.TestCase):
    def test_nonzero_zscore_preserves_zero_background(self) -> None:
        value = np.zeros((2, 2, 2), dtype=np.float32)
        mask = np.zeros_like(value, dtype=bool)
        mask[0, :, :] = True
        value[mask] = np.asarray([1.0, 2.0, 3.0, 4.0])
        normalized = adapter.zscore_nonzero(value, mask)
        self.assertAlmostEqual(float(normalized[mask].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[mask].std()), 1.0, places=6)
        np.testing.assert_array_equal(normalized[~mask], 0.0)

    def test_volume_and_slice_adapters_use_flair_t1_t2_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "sub_1" / "ses_1"
            case.mkdir(parents=True)
            shape = (4, 5, 3)
            affine = np.eye(4)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[1:3, 1:4, :] = 1
            paths: dict[str, Path] = {}
            foreground = np.arange(1, mask.sum() + 1, dtype=np.float32)
            modality_values = {
                "t1": foreground,
                "t2": foreground**2,
                "flair": foreground[::-1],
            }
            for modality in ("t1", "t2", "flair"):
                value = np.zeros(shape, dtype=np.float32)
                value[mask > 0] = modality_values[modality]
                path = case / f"sub_1_ses_1_{modality}.nii.gz"
                nib.save(nib.Nifti1Image(value, affine), path)
                paths[modality] = path
            mask_path = case / "sub_1_ses_1_brainmask.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            manifest = root / "dataset_manifest.csv"
            fields = (
                "case_id",
                "processing_status",
                "t1_output",
                "t2_output",
                "flair_output",
                "brain_mask_output",
            )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "sub_1/ses_1",
                        "processing_status": "PASS",
                        "t1_output": paths["t1"],
                        "t2_output": paths["t2"],
                        "flair_output": paths["flair"],
                        "brain_mask_output": mask_path,
                    }
                )
            slice_manifest = root / "slice_manifest.csv"
            slice_manifest.write_text("case_id,slice\nsub_1/ses_1,1\n", encoding="utf-8")
            with (
                mock.patch.object(adapter, "BRATS_SHAPE", shape),
                mock.patch.object(adapter, "BRATS_SPACING_MM", (1.0, 1.0, 1.0)),
            ):
                volume_dataset = adapter.FOMO45KBraTS21VolumeDataset(root)
                item = volume_dataset[0]
                self.assertEqual(tuple(item["image"].shape), (3, *shape))
                self.assertEqual(volume_dataset.modalities, ["flair", "t1", "t2"])
                self.assertTrue(torch.equal(item["image"][0], item["flair"]))
                self.assertTrue(torch.equal(item["image"][1], item["t1"]))
                self.assertTrue(torch.equal(item["image"][2], item["t2"]))
                slice_dataset = adapter.FOMO45KBraTS21SliceDataset(root, image_size=4)
                image = slice_dataset[0]
                self.assertEqual(tuple(image.shape), (3, 4, 4))
                self.assertTrue(torch.all(image[:, 0, :] == 0))


if __name__ == "__main__":
    unittest.main()
