"""Compatibility contracts for modular healthy-slice preprocessing."""

from __future__ import annotations

import csv
import importlib
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[1]
FACADE_MODULE = "andi_rewrite.data.preprocess"


class PreprocessContractTest(unittest.TestCase):
    """Freeze public aliases and deterministic preprocessing semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preprocess = importlib.import_module(FACADE_MODULE)
        cls.imaging = importlib.import_module("andi_rewrite.data.imaging")
        cls.healthy_slices = importlib.import_module("andi_rewrite.data.healthy_slices")
        cls.subject_splits = importlib.import_module("andi_rewrite.data.subject_splits")
        cls.lmdb_io = importlib.import_module("andi_rewrite.data.lmdb_io")
        cls.datasets = importlib.import_module("andi_rewrite.data.datasets")
        cls.datasets_imaging = importlib.import_module("andi_rewrite.data.datasets.imaging")

    def test_old_preprocess_facade_preserves_direct_owner_identity(self) -> None:
        owners = {
            "DEFAULT_MODALITIES": (self.imaging, "DEFAULT_MODALITIES"),
            "normalize_volume": (self.imaging, "normalize_volume"),
            "_load_nifti": (self.imaging, "_load_nifti"),
            "load_subject_volume": (self.imaging, "load_subject_volume"),
            "resize_slices": (self.imaging, "resize_slices"),
            "HealthySliceCandidate": (self.healthy_slices, "HealthySliceCandidate"),
            "BalancedSliceRecord": (self.healthy_slices, "BalancedSliceRecord"),
            "iter_healthy_slices": (self.healthy_slices, "iter_healthy_slices"),
            "healthy_slices_for_subject": (self.healthy_slices, "healthy_slices_for_subject"),
            "healthy_slice_candidates_for_subject": (self.healthy_slices, "healthy_slice_candidates_for_subject"),
            "balance_candidates_by_z": (self.healthy_slices, "balance_candidates_by_z"),
            "read_subject_csv": (self.subject_splits, "read_subject_csv"),
            "build_kfold_subject_splits": (self.subject_splits, "build_kfold_subject_splits"),
            "build_repeated_train_test_splits": (self.subject_splits, "build_repeated_train_test_splits"),
            "build_combined_train_test_splits": (self.subject_splits, "build_combined_train_test_splits"),
            "SplitHealthySummary": (self.lmdb_io, "SplitHealthySummary"),
            "KFoldLMDBSummary": (self.lmdb_io, "KFoldLMDBSummary"),
            "estimate_lmdb_map_size": (self.lmdb_io, "estimate_lmdb_map_size"),
            "estimate_balanced_lmdb_map_size": (self.lmdb_io, "estimate_balanced_lmdb_map_size"),
            "split_healthy_to_lmdb": (self.lmdb_io, "split_healthy_to_lmdb"),
            "split_healthy_z_balanced_to_lmdb": (self.lmdb_io, "split_healthy_z_balanced_to_lmdb"),
            "split_healthy_kfold_to_lmdb": (self.lmdb_io, "split_healthy_kfold_to_lmdb"),
        }
        for name, (owner, owner_name) in owners.items():
            with self.subTest(name=name):
                self.assertIn(name, self.preprocess.__all__)
                self.assertIs(getattr(self.preprocess, name), getattr(owner, owner_name))

    def test_normalize_volume_is_shared_and_keeps_mutating_foreground_p99_behavior(self) -> None:
        self.assertIs(self.preprocess.normalize_volume, self.imaging.normalize_volume)
        self.assertIs(self.preprocess.normalize_volume, self.datasets.normalize_volume)
        self.assertIs(self.preprocess.normalize_volume, self.datasets_imaging.normalize_volume)

        positive = torch.arange(1, 101, dtype=torch.float32)
        image = torch.cat((torch.tensor([-5.0, 0.0]), positive)).reshape(1, 1, 1, -1)
        original = image.clone()
        result = self.preprocess.normalize_volume(image)

        self.assertIs(result, image)
        p99 = torch.quantile(positive, 0.99)
        torch.testing.assert_close(result[0, 0, 0, 0], torch.tensor(-5.0) / p99)
        self.assertEqual(result[0, 0, 0, 1].item(), 0.0)
        torch.testing.assert_close(result[0, 0, 0, -1], torch.tensor(100.0) / p99)
        torch.testing.assert_close(original[0, 0, 0, 2:], positive)

        all_zero = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
        normalized_zero = self.preprocess.normalize_volume(all_zero)
        self.assertIsNot(normalized_zero, all_zero)
        self.assertEqual(normalized_zero.dtype, torch.float32)
        self.assertTrue(torch.equal(normalized_zero, torch.zeros_like(normalized_zero)))
        self.assertEqual(all_zero.dtype, torch.float64)
        self.assertTrue(torch.equal(all_zero, torch.zeros_like(all_zero)))

    def test_z_balancing_is_seeded_sorted_and_preserves_repeat_metadata(self) -> None:
        candidate = self.healthy_slices.HealthySliceCandidate
        groups = {
            4: [candidate("four-a", 4, ()), candidate("four-b", 4, ()), candidate("four-c", 4, ())],
            1: [candidate("one-a", 1, ()), candidate("one-b", 1, ())],
        }
        first_records, first_summary = self.preprocess.balance_candidates_by_z(groups, per_z_count=2, balance_seed=17)
        second_records, second_summary = self.preprocess.balance_candidates_by_z(groups, per_z_count=2, balance_seed=17)

        self.assertEqual(first_records, second_records)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary, {1: {"original": 2, "output": 2}, 4: {"original": 3, "output": 2}})
        self.assertEqual(
            [(record.candidate.subject_id, record.candidate.z, record.sampled_index_in_z) for record in first_records],
            [("one-a", 1, 0), ("one-b", 1, 1), ("four-c", 4, 2), ("four-b", 4, 1)],
        )

        repeated, summary = self.preprocess.balance_candidates_by_z(
            {5: [candidate("a", 5, ()), candidate("b", 5, ())]},
            per_z_count=5,
            balance_seed=17,
        )
        self.assertEqual(summary, {5: {"original": 2, "output": 5}})
        self.assertEqual(
            [(record.candidate.subject_id, record.sampled_index_in_z, record.duplicate_copy_id, record.is_extra_sample) for record in repeated],
            [("a", 0, 0, False), ("b", 1, 0, False), ("a", 0, 1, False), ("b", 1, 1, False), ("b", 1, 2, True)],
        )
        with self.assertRaisesRegex(ValueError, "per-z-count must be a positive integer"):
            self.preprocess.balance_candidates_by_z(groups, per_z_count=0, balance_seed=17)

    def test_subject_splits_are_deterministic_ordered_and_validate_edge_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subjects.csv"
            train = root / "train.csv"
            test = root / "test.csv"
            duplicate = root / "duplicate.csv"
            source.write_text("subject_id,site\ns0,a\ns1,a\ns2,b\ns3,b\ns4,c\n", encoding="utf-8")
            train.write_text("subject_id\ntr0\ntr1\n", encoding="utf-8")
            test.write_text("subject_id\nte0\nte1\n", encoding="utf-8")
            duplicate.write_text("subject_id\na\na\n", encoding="utf-8")

            first = self.preprocess.build_kfold_subject_splits(source, folds=3, seed=13, shuffle=True)
            second = self.preprocess.build_kfold_subject_splits(source, folds=3, seed=13, shuffle=True)
            no_shuffle = self.preprocess.build_kfold_subject_splits(source, folds=3, seed=999, shuffle=False)
            repeated = self.preprocess.build_repeated_train_test_splits(train, test, folds=2)
            combined = self.preprocess.build_combined_train_test_splits(train, test, folds=3, seed=13, test_size=1)

            self.assertEqual(
                [(index, train_fold.subject_id.tolist(), val_fold.subject_id.tolist()) for index, train_fold, val_fold in first],
                [(index, train_fold.subject_id.tolist(), val_fold.subject_id.tolist()) for index, train_fold, val_fold in second],
            )
            self.assertEqual(
                [(index, val_fold.subject_id.tolist()) for index, _train_fold, val_fold in first],
                [(0, ["s0", "s2"]), (1, ["s1", "s3"]), (2, ["s4"])],
            )
            self.assertEqual(
                [(index, val_fold.subject_id.tolist()) for index, _train_fold, val_fold in no_shuffle],
                [(0, ["s0", "s1"]), (1, ["s2", "s3"]), (2, ["s4"])],
            )
            self.assertEqual(
                [(index, train_fold.subject_id.tolist(), test_fold.subject_id.tolist()) for index, train_fold, test_fold in repeated],
                [(0, ["tr0", "tr1"], ["te0", "te1"]), (1, ["tr0", "tr1"], ["te0", "te1"])],
            )
            self.assertEqual([index for index, _train_fold, _test_fold in combined], [0, 1, 2])
            for _index, train_fold, test_fold in combined:
                self.assertEqual(train_fold.index.tolist(), list(range(len(train_fold))))
                self.assertEqual(test_fold.index.tolist(), list(range(len(test_fold))))
                self.assertEqual(len(train_fold), 3)
                self.assertEqual(len(test_fold), 1)
                self.assertTrue(set(train_fold.subject_id).isdisjoint(test_fold.subject_id))

            with self.assertRaisesRegex(ValueError, "duplicate subject ids"):
                self.preprocess.read_subject_csv(duplicate, label="Input CSV")
            with self.assertRaisesRegex(ValueError, "folds must be at least 2"):
                self.preprocess.build_kfold_subject_splits(source, folds=1)
            with self.assertRaisesRegex(ValueError, "cannot exceed subject count"):
                self.preprocess.build_kfold_subject_splits(source, folds=6)
            with self.assertRaisesRegex(ValueError, "combined-test-size must be positive"):
                self.preprocess.build_combined_train_test_splits(train, test, test_size=-1)

    def test_temp_nifti_to_lmdb_product_has_legacy_keys_values_and_safe_existing_target_guard(self) -> None:
        nib = importlib.import_module("nibabel")
        lmdb = importlib.import_module("lmdb")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            subject_id = "subject-01"
            subject_dir = dataset_root / subject_id
            subject_dir.mkdir(parents=True)
            volumes = {
                "flair": np.array([[[2.0, 0.0], [4.0, 0.0]], [[6.0, 0.0], [8.0, 0.0]]], dtype=np.float32),
                "t1": np.array([[[1.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [7.0, 0.0]]], dtype=np.float32),
            }
            for modality, volume in volumes.items():
                nib.save(nib.Nifti1Image(volume, np.eye(4)), subject_dir / f"{subject_id}_{modality}.nii.gz")
            mask = np.zeros((2, 2, 2), dtype=np.int16)
            mask[0, 0, 1] = 3
            nib.save(nib.Nifti1Image(mask, np.eye(4)), subject_dir / f"{subject_id}_seg.nii.gz")
            input_csv = root / "subjects.csv"
            input_csv.write_text("subject_id\nsubject-01\n", encoding="utf-8")
            output_csv = root / "healthy_slices.csv"
            lmdb_dir = root / "healthy.lmdb"

            summary = self.preprocess.split_healthy_to_lmdb(
                dataset_path=dataset_root,
                input_csv=input_csv,
                output_csv=output_csv,
                image_size=2,
                lmdb_dir=lmdb_dir,
                modalities=("flair", "t1"),
                map_size=1 << 20,
                progress=False,
            )

            self.assertEqual(summary.as_dict(), {
                "subjects_seen": 1,
                "healthy_slices": 1,
                "lmdb_dir": str(lmdb_dir),
                "csv_path": str(output_csv),
                "image_size": 2,
            })
            with output_csv.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.reader(handle)), [["subject_id", "Slice"], [subject_id, "0"]])
            environment = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
            try:
                with environment.begin() as transaction:
                    payload = transaction.get(b"00000000")
                    self.assertIsNotNone(payload)
                    stored = pickle.loads(payload)
                    self.assertEqual(stored.shape, (2, 2, 2))
                    self.assertEqual(stored.dtype, np.float32)
                    self.assertAlmostEqual(float(stored[0, 1, 1]), 8.0 / 7.94, places=6)
            finally:
                environment.close()

            original_csv = output_csv.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "LMDB directory is not empty"):
                self.preprocess.split_healthy_to_lmdb(
                    dataset_path=dataset_root,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    image_size=2,
                    lmdb_dir=lmdb_dir,
                    modalities=("flair", "t1"),
                    map_size=1 << 20,
                    progress=False,
                    overwrite=False,
                )
            self.assertEqual(output_csv.read_bytes(), original_csv)

    def test_fresh_preprocess_imports_do_not_reverse_import_engine(self) -> None:
        modules = (
            "andi_rewrite.data.imaging",
            "andi_rewrite.data.healthy_slices",
            "andi_rewrite.data.subject_splits",
            "andi_rewrite.data.lmdb_io",
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
            "    raise RuntimeError(f'preprocess imported engine modules: {engine_modules}')\n"
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
            "fresh preprocessing import failed or imported engine:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
