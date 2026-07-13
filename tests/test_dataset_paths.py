from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.data.datasets import MRIDataVolume, ShiftsMSVolumeDataset, _subject_file_path, build_dataset


class DatasetPathTest(unittest.TestCase):
    def test_subject_file_path_uses_configured_separator(self) -> None:
        subject_dir = Path("BraTS-GLI-00000-000")

        path = _subject_file_path(subject_dir, "BraTS-GLI-00000-000", "t2f", "-")

        self.assertEqual(path, subject_dir / "BraTS-GLI-00000-000-t2f.nii.gz")

    def test_volume_dataset_can_build_subject_list_from_directories(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root)
            (dataset_path / "BraTS-GLI-00002-000").mkdir()
            (dataset_path / "BraTS-GLI-00000-000").mkdir()

            dataset = MRIDataVolume(
                csv_path=None,
                dataset_path=dataset_path,
                modalities=["t2f", "t1n", "t1c", "t2w"],
                filename_separator="-",
            )

        self.assertEqual(dataset.df["subject_id"].tolist(), ["BraTS-GLI-00000-000", "BraTS-GLI-00002-000"])
        self.assertEqual(dataset.filename_separator, "-")

    def test_shifts_ms_dataset_discovers_archive_layout_and_pd_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root)
            split_dir = dataset_path / "shifts_ms_pt2" / "best" / "dev_in"
            for folder in ["flair", "t1", "pd", "t2", "gt", "fg_mask"]:
                (split_dir / folder).mkdir(parents=True)
            (split_dir / "flair" / "6_FLAIR_isovox.nii.gz").write_bytes(b"")
            (split_dir / "t1" / "6_T1_isovox.nii.gz").write_bytes(b"")
            (split_dir / "pd" / "6_PD_isovox.nii.gz").write_bytes(b"")
            (split_dir / "t2" / "6_T2_isovox.nii.gz").write_bytes(b"")
            (split_dir / "gt" / "6_gt_isovox.nii.gz").write_bytes(b"")
            (split_dir / "fg_mask" / "6_isovox_fg_mask.nii.gz").write_bytes(b"")

            dataset = ShiftsMSVolumeDataset(
                dataset_path=dataset_path,
                splits=["dev_in"],
                require_segmentation=True,
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.root.name, "shifts_ms_pt2")
        self.assertEqual(dataset.subjects[0].subject_id, "6")
        self.assertEqual(dataset.subjects[0].modality_sources["t1ce"], "pd")
        self.assertTrue(dataset.subjects[0].brain_mask_path)

    def test_build_dataset_supports_ljubljana_ms_volume_type(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root)
            split_dir = dataset_path / "ljubljana" / "dev_out"
            for folder in ["flair", "t1", "t1ce", "t2", "Gold_Standard"]:
                (split_dir / folder).mkdir(parents=True)
            (split_dir / "flair" / "1_FLAIR_isovox.nii.gz").write_bytes(b"")
            (split_dir / "t1" / "1_T1_isovox.nii.gz").write_bytes(b"")
            (split_dir / "t1ce" / "1_T1Post_isovox.nii.gz").write_bytes(b"")
            (split_dir / "t2" / "1_T2_isovox.nii.gz").write_bytes(b"")
            (split_dir / "Gold_Standard" / "1_Gold_Standard_isovox.nii.gz").write_bytes(b"")

            dataset = build_dataset(
                {
                    "type": "ljubljana_ms_volume",
                    "dataset_path": str(dataset_path),
                    "splits": ["dev_out"],
                    "locations": ["ljubljana"],
                }
            )

        self.assertIsInstance(dataset, ShiftsMSVolumeDataset)
        self.assertEqual(dataset.subjects[0].location, "ljubljana")


if __name__ == "__main__":
    unittest.main()
