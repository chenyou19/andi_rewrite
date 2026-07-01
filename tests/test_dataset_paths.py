from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.data.datasets import MRIDataVolume, _subject_file_path


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


if __name__ == "__main__":
    unittest.main()
