from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.data.datasets import (
    UCSFPDGMVolumeDataset,
    _ucsf_pdgm_file_stem,
    _validate_nifti_geometry,
    build_dataset,
)


MODALITY_SUFFIXES = {
    "flair": "FLAIR",
    "t1": "T1",
    "t1ce": "T1c",
    "t2": "T2",
}


def _lps_affine(height: int) -> np.ndarray:
    return np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, float(height - 1)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _touch_case(parent: Path, subject_id: str, missing: set[str] | None = None) -> Path:
    missing = missing or set()
    folder = parent / f"{subject_id}_nifti"
    folder.mkdir(parents=True)
    for logical_name, suffix in MODALITY_SUFFIXES.items():
        if logical_name not in missing:
            (folder / f"{subject_id}_{suffix}.nii.gz").write_bytes(b"")
    if "segmentation" not in missing:
        (folder / f"{subject_id}_tumor_segmentation.nii.gz").write_bytes(b"")
    return folder


def _write_case(
    parent: Path,
    subject_id: str,
    *,
    shape: tuple[int, int, int] = (6, 6, 2),
    modality_values: dict[str, float] | None = None,
    segmentation: np.ndarray | None = None,
) -> Path:
    folder = parent / f"{subject_id}_nifti"
    folder.mkdir(parents=True)
    values = modality_values or {"flair": 10.0, "t1": 20.0, "t1ce": 30.0, "t2": 40.0}
    affine = _lps_affine(shape[1])
    for logical_name, suffix in MODALITY_SUFFIXES.items():
        data = np.full(shape, values[logical_name], dtype=np.float32)
        nib.save(nib.Nifti1Image(data, affine), folder / f"{subject_id}_{suffix}.nii.gz")
    if segmentation is None:
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[1, 1, 0] = 1
    nib.save(
        nib.Nifti1Image(np.asarray(segmentation, dtype=np.uint8), affine),
        folder / f"{subject_id}_tumor_segmentation.nii.gz",
    )
    return folder


def _write_subject_csv(path: Path, subject_ids: list[str], header: str = "subject_id") -> Path:
    path.write_text(
        "\n".join([header, *subject_ids]) + "\n",
        encoding="utf-8",
    )
    return path


class UCSFPDGMVolumeDatasetTest(unittest.TestCase):
    def test_folder_stem_parsing(self) -> None:
        self.assertEqual(_ucsf_pdgm_file_stem("UCSF-PDGM-0004_nifti"), "UCSF-PDGM-0004")

    def test_follow_up_folder_stem_parsing(self) -> None:
        self.assertEqual(
            _ucsf_pdgm_file_stem("UCSF-PDGM-0391_FU016d_nifti"),
            "UCSF-PDGM-0391_FU016d",
        )

    def test_default_modality_mapping_preserves_model_channel_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            dataset = UCSFPDGMVolumeDataset(root_path)
            subject = dataset.subjects[0]

        self.assertEqual(dataset.modalities, ["flair", "t1", "t1ce", "t2"])
        self.assertEqual(
            [subject.modality_paths[name].name for name in dataset.modalities],
            [
                "UCSF-PDGM-0001_FLAIR.nii.gz",
                "UCSF-PDGM-0001_T1.nii.gz",
                "UCSF-PDGM-0001_T1c.nii.gz",
                "UCSF-PDGM-0001_T2.nii.gz",
            ],
        )

    def test_recursive_discovery_excludes_incomplete_cases_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            _touch_case(root_path, "UCSF-PDGM-0002", missing={"t1ce"})
            package = root_path / "PKG - UCSF-PDGM Version 5" / "UCSF-PDGM-v5"
            _touch_case(package, "UCSF-PDGM-0003")
            dataset = UCSFPDGMVolumeDataset(root_path, subject_limit=1)

        self.assertEqual([subject.subject_id for subject in dataset.subjects], ["UCSF-PDGM-0001"])
        self.assertEqual(dataset.discovery_summary["candidate_folders"], 3)
        self.assertEqual(dataset.discovery_summary["complete_cases"], 2)
        self.assertEqual(dataset.discovery_summary["incomplete_cases"], 1)
        self.assertEqual(dataset.discovery_summary["requested_cases"], 2)
        self.assertEqual(dataset.discovery_summary["selected_cases"], 1)
        self.assertEqual(dataset.discovery_summary["selection_source"], "discovery")

    def test_csv_selection_preserves_order_excludes_extra_and_applies_limit_last(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            _touch_case(root_path, "UCSF-PDGM-0002")
            _touch_case(root_path, "UCSF-PDGM-0002_FU010d")
            csv_path = _write_subject_csv(
                root_path / "subjects.csv",
                ["UCSF-PDGM-0002", "UCSF-PDGM-0001"],
            )

            dataset = UCSFPDGMVolumeDataset(root_path, csv_path=csv_path)
            limited = UCSFPDGMVolumeDataset(root_path, csv_path=csv_path, subject_limit=1)

        self.assertEqual(
            [subject.subject_id for subject in dataset.subjects],
            ["UCSF-PDGM-0002", "UCSF-PDGM-0001"],
        )
        self.assertEqual([subject.subject_id for subject in limited.subjects], ["UCSF-PDGM-0002"])
        self.assertEqual(dataset.discovery_summary["complete_cases"], 3)
        self.assertEqual(dataset.discovery_summary["requested_cases"], 2)
        self.assertEqual(dataset.discovery_summary["selected_cases"], 2)
        self.assertEqual(dataset.discovery_summary["selection_source"], "csv")

    def test_csv_rejects_missing_column_blank_and_duplicate_subject_ids(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            csv_path = root_path / "subjects.csv"

            _write_subject_csv(csv_path, ["UCSF-PDGM-0001"], header="patient_id")
            with self.assertRaisesRegex(ValueError, "must contain a 'subject_id' column"):
                UCSFPDGMVolumeDataset(root_path, csv_path=csv_path)

            csv_path.write_text('subject_id\n" "\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank subject_id"):
                UCSFPDGMVolumeDataset(root_path, csv_path=csv_path)

            _write_subject_csv(
                csv_path,
                ["UCSF-PDGM-0001", "UCSF-PDGM-0001"],
            )
            with self.assertRaisesRegex(ValueError, "duplicate subject_id"):
                UCSFPDGMVolumeDataset(root_path, csv_path=csv_path)

    def test_csv_missing_or_incomplete_subject_raises_instead_of_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            _touch_case(root_path, "UCSF-PDGM-0002", missing={"t1ce"})
            csv_path = _write_subject_csv(
                root_path / "subjects.csv",
                ["UCSF-PDGM-0001", "UCSF-PDGM-0002"],
            )

            with self.assertRaisesRegex(FileNotFoundError, "UCSF-PDGM-0002"):
                UCSFPDGMVolumeDataset(root_path, csv_path=csv_path)

    def test_partial_and_aspera_checkpoint_files_do_not_make_a_case_complete(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            partial = root_path / "UCSF-PDGM-0002_nifti"
            partial.mkdir()
            (partial / "UCSF-PDGM-0002_T1c.nii.gz.partial").write_bytes(b"")
            (partial / "UCSF-PDGM-0002_T1c.nii.gz.aspera-ckpt").write_bytes(b"")
            dataset = UCSFPDGMVolumeDataset(root_path)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.incomplete_cases[0]["subject_id"], "UCSF-PDGM-0002")

    def test_pre_normalization_channel_order_reads_flair_t1_t1c_t2(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _write_case(root_path, "UCSF-PDGM-0001")
            dataset = UCSFPDGMVolumeDataset(root_path, image_size=6)
            images, _, _, _ = dataset._load_case_arrays(dataset.subjects[0])

        np.testing.assert_allclose(images[:, 0, 0, 0], [10.0, 20.0, 30.0, 40.0])

    def test_multiclass_tumor_labels_become_binary_whole_tumor(self) -> None:
        segmentation = np.asarray([0, 1, 2, 4], dtype=np.uint8).reshape(2, 2, 1)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _write_case(
                root_path,
                "UCSF-PDGM-0001",
                shape=(2, 2, 1),
                segmentation=segmentation,
            )
            dataset = UCSFPDGMVolumeDataset(root_path, image_size=2, return_metadata=False)
            _, mask = dataset[0]

        np.testing.assert_array_equal(mask.numpy().reshape(-1), [False, True, True, True])

    def test_resize_changes_xy_to_128_and_preserves_155_slices(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            dataset = UCSFPDGMVolumeDataset(root_path, image_size=128)
            image = torch.zeros((4, 240, 240, 155), dtype=torch.float32)
            mask = torch.zeros((240, 240, 155), dtype=torch.bool)
            resized_image = dataset._resize_image_volume(image)
            resized_mask = dataset._resize_mask_volume(mask)

        self.assertEqual(tuple(resized_image.shape), (4, 128, 128, 155))
        self.assertEqual(tuple(resized_mask.shape), (128, 128, 155))
        self.assertEqual(resized_mask.dtype, torch.bool)

    def test_metadata_contains_native_and_model_space_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            folder = _write_case(root_path, "UCSF-PDGM-0001", shape=(6, 6, 2))
            dataset = UCSFPDGMVolumeDataset(root_path, image_size=4)
            image, mask, metadata = dataset[0]

        self.assertEqual(tuple(image.shape), (4, 4, 4, 2))
        self.assertEqual(tuple(mask.shape), (4, 4, 2))
        self.assertEqual(metadata["subject_id"], "UCSF-PDGM-0001")
        self.assertEqual(Path(metadata["case_folder"]), folder)
        self.assertEqual(metadata["native_shape"], "6,6,2")
        self.assertEqual(metadata["model_shape"], "4,4,2")
        self.assertEqual(metadata["native_orientation"], "LPS")
        self.assertEqual(metadata["model_orientation"], "LPS")
        self.assertEqual(metadata["orientation_transform"], "identity")
        self.assertEqual(metadata["reference_path"], metadata["input_paths"]["flair"])
        self.assertTrue(metadata["segmentation_path"].endswith("_tumor_segmentation.nii.gz"))

    def test_dataset_builder_supports_ucsf_pdgm_volume(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path, "UCSF-PDGM-0001")
            _touch_case(root_path, "UCSF-PDGM-0002")
            csv_path = _write_subject_csv(root_path / "subjects.csv", ["UCSF-PDGM-0002"])
            dataset = build_dataset(
                {
                    "type": "ucsf_pdgm_volume",
                    "dataset_path": str(root_path),
                    "path_to_csv": str(csv_path),
                    "subject_limit": 1,
                }
            )

        self.assertIsInstance(dataset, UCSFPDGMVolumeDataset)
        self.assertEqual([subject.subject_id for subject in dataset.subjects], ["UCSF-PDGM-0002"])
        self.assertEqual(dataset.csv_path, csv_path)

    def test_duplicate_complete_case_ids_raise_before_subject_limit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _touch_case(root_path / "direct", "UCSF-PDGM-0001")
            _touch_case(root_path / "package", "UCSF-PDGM-0001")

            with self.assertRaisesRegex(ValueError, "Duplicate complete UCSF-PDGM"):
                UCSFPDGMVolumeDataset(root_path, subject_limit=1)

    def test_geometry_validation_accepts_lps_identity_and_rejects_unexpected_orientation(self) -> None:
        shape = (4, 5, 2)
        reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), _lps_affine(shape[1]))
        matching = nib.Nifti1Image(np.ones(shape, dtype=np.float32), _lps_affine(shape[1]))

        self.assertEqual(_validate_nifti_geometry(reference, {"t1": matching}, "LPS"), "LPS")

        ras = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
        with self.assertRaisesRegex(ValueError, "verified BraTS model orientation"):
            _validate_nifti_geometry(ras, {"t1": ras}, "LPS")


if __name__ == "__main__":
    unittest.main()
