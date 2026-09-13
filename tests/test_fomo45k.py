from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from andi_rewrite.data.fomo45k import (
    DatasetExpectations,
    audit_output,
    build_output,
    dataset_status,
    normalize_nonzero_p99,
    reorient_volume,
    validate_dataset,
)
from andi_rewrite.scripts.compute_lmdb_spectrum import source_manifest_provenance


REVISION = "bf2bb12bd5cfeaed65e4a003e72d55fb8322c28a"


class FOMOOrientationTest(unittest.TestCase):
    def test_ras_to_lps_is_exact_xy_flip_with_updated_affine(self) -> None:
        source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        reoriented, affine, transform = reorient_volume(source, np.eye(4), ("L", "P", "S"))

        np.testing.assert_array_equal(reoriented, source[::-1, ::-1, :])
        np.testing.assert_array_equal(
            transform,
            np.asarray([[0.0, -1.0], [1.0, -1.0], [2.0, 1.0]]),
        )
        np.testing.assert_allclose(
            affine,
            np.asarray(
                [
                    [-1.0, 0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertEqual(nib.aff2axcodes(affine), ("L", "P", "S"))
        np.testing.assert_array_equal(np.sort(reoriented, axis=None), np.sort(source, axis=None))

    def test_nonzero_p99_clips_without_creating_a_mask(self) -> None:
        source = np.asarray([[[0.0, 1.0, 2.0, 100.0]]], dtype=np.float32)
        normalized, statistics = normalize_nonzero_p99(source)
        self.assertEqual(float(normalized[0, 0, 0]), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)
        self.assertEqual(statistics["nonzero_voxels"], 3)
        self.assertTrue(np.all(np.isfinite(normalized)))


class FOMOBuildTest(unittest.TestCase):
    def _save_fixture(self, root: Path) -> tuple[Path, Path]:
        download_root = root / "FOMO45K_healthy_243"
        source_root = download_root / "PT007_NIMH"
        cache_root = download_root / ".cache" / "huggingface" / "download"
        rows = []
        for subject_index, participant_id in enumerate(("sub_1", "sub_2"), start=1):
            session_id = "ses_1"
            session_root = source_root / participant_id / session_id
            session_root.mkdir(parents=True)
            filenames = {
                "T1": "t1.nii.gz",
                "T2": "t2_2.nii.gz" if subject_index == 2 else "t2.nii.gz",
                "FLAIR": "flair.nii.gz",
            }
            base = np.zeros((4, 6, 3), dtype=np.float32)
            base[1:3, 1:5, 0] = float(subject_index)
            base[1:3, 2:4, 2] = float(subject_index + 1)
            paths = {}
            for modality, scale in (("FLAIR", 3.0), ("T1", 1.0), ("T2", 2.0)):
                path = session_root / filenames[modality]
                nib.save(nib.Nifti1Image(base * scale, np.eye(4)), path)
                repo_path = Path("PT007_NIMH") / participant_id / session_id / path.name
                metadata_path = cache_root / repo_path.parent / f"{repo_path.name}.metadata"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                metadata_path.write_text(f"{REVISION}\n{digest}\n0.0\n", encoding="utf-8")
                paths[modality] = repo_path.as_posix()
            rows.append(
                {
                    "dataset": "PT007_NIMH",
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "sex": "F" if subject_index == 1 else "M",
                    "age": np.nan if subject_index == 2 else 20.0 + subject_index,
                    "group": "Control",
                    "T1_filename": filenames["T1"],
                    "T1_repo_path": paths["T1"],
                    "T2_filename": filenames["T2"],
                    "T2_repo_path": paths["T2"],
                    "FLAIR_filename": filenames["FLAIR"],
                    "FLAIR_repo_path": paths["FLAIR"],
                }
            )
        metadata_tsv = root / "metadata.tsv"
        pd.DataFrame(rows).to_csv(metadata_tsv, sep="\t", index=False)
        return source_root, metadata_tsv

    def test_validate_build_audit_and_atomic_publication(self) -> None:
        expectations = DatasetExpectations(
            subjects=2,
            sessions=2,
            files=6,
            train_subjects=1,
            validation_subjects=1,
            total_slices=4,
            train_slices=2,
            validation_slices=2,
            source_axcodes="RAS",
            hf_revision=REVISION,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, metadata_tsv = self._save_fixture(root)
            records, validation = validate_dataset(
                source_root,
                metadata_tsv,
                validation_subjects=("sub_2",),
                expectations=expectations,
                pad_size=8,
            )
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["split_slice_counts"], {"train": 2, "val": 2})
            self.assertEqual(validation["missing_age_count"], 1)
            self.assertEqual([record.split for record in records], ["train", "val"])

            output_root = root / "prepared" / "PT007_NIMH"
            publication = build_output(
                source_root,
                metadata_tsv,
                output_root,
                validation_subjects=("sub_2",),
                expectations=expectations,
                pad_size=8,
                image_size=4,
                map_size_override=16 * 1024**2,
            )
            self.assertEqual(publication["status"], "PUBLISHED")
            self.assertTrue(output_root.is_dir())
            self.assertEqual(audit_output(output_root)["total_entries"], 4)
            self.assertEqual(dataset_status(output_root)["audit_report.json"]["status"], "PASS")

            sessions = [
                json.loads(line)
                for line in (output_root / "manifests" / "sessions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual({row["target_axcodes"] for row in sessions}, {"LPS"})
            self.assertTrue(all(row["mask_created"] is False for row in sessions))
            self.assertTrue(all(row["channel_order"] == ["FLAIR", "T1", "T2"] for row in sessions))

            import lmdb

            environment = lmdb.open(str(output_root / "train"), readonly=True, lock=False)
            try:
                with environment.begin() as transaction:
                    value = np.asarray(pickle.loads(transaction.get(b"00000000")))
            finally:
                environment.close()
            self.assertEqual(value.shape, (3, 4, 4))
            self.assertEqual(value.dtype, np.float32)
            self.assertGreaterEqual(float(value.min()), 0.0)
            self.assertLessEqual(float(value.max()), 1.0)


class FOMOConfigContractTest(unittest.TestCase):
    def test_spectrum_manifest_provenance_proves_train_only_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_entries.csv"
            pd.DataFrame(
                [
                    {
                        "participant_id": "sub_1",
                        "session_identifier": "sub_1/ses_1",
                        "split": "train",
                    },
                    {
                        "participant_id": "sub_2",
                        "session_identifier": "sub_2/ses_1",
                        "split": "train",
                    },
                ]
            ).to_csv(path, index=False)
            provenance = source_manifest_provenance(path)
            self.assertEqual(provenance["rows"], 2)
            self.assertEqual(provenance["split_counts"], {"train": 2})
            self.assertEqual(provenance["subject_count"], 2)
            self.assertEqual(provenance["session_count"], 2)
            self.assertRegex(str(provenance["sha256"]), r"^[0-9a-f]{64}$")

    def test_train_eval_and_hidden_launch_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        train = yaml.safe_load(
            (root / "configs" / "train_fomo45k_nimh_flair_t1_t2_empirical_spectrum233.yaml")
            .read_text(encoding="utf-8")
        )
        evaluation = yaml.safe_load(
            (root / "configs" / "eval_brats21_50_fomo45k_nimh_model_matched_noise.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(train["data"]["batch_size"], 52)
        self.assertEqual(train["training"]["epochs"], 233)
        self.assertEqual(train["model"]["in_channels"], 3)
        self.assertIn("/train", train["data"]["path"])
        self.assertIn("PT007_NIMH/spectrum/", train["noise"]["schedule"]["sampler"]["stats_path"])
        self.assertEqual(evaluation["data"]["type"], "volume")
        self.assertEqual(evaluation["data"]["modalities"], ["flair", "t1", "t2"])
        self.assertTrue(evaluation["data"]["return_metadata"])
        self.assertTrue(evaluation["model"]["use_ema"])
        self.assertIn("epoch_0232.pt", evaluation["model"]["checkpoint"])
        self.assertEqual(evaluation["metrics"]["auprc_max_samples"], 5_000_000)
        self.assertTrue(evaluation["prediction_output"]["restore_native_grid"])
        self.assertTrue(evaluation["prediction_output"]["save_binary_mask"])

        launcher = (root / "scripts" / "launch_training_hidden.ps1").read_text(encoding="utf-8")
        trainer = (root / "scripts" / "train.py").read_text(encoding="utf-8")
        self.assertIn("[string]$EvalConfig", launcher)
        self.assertIn("eval_config_snapshot.yaml", launcher)
        self.assertIn("'--eval-config'", launcher)
        self.assertIn('"prediction_output": eval_config.get("prediction_output", {})', trainer)


if __name__ == "__main__":
    unittest.main()
