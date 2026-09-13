from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from andi_rewrite.data.fomo45k.brats21 import (
    RegistrationSettings,
    SessionRecord,
    Toolchain,
    _signature,
    _source_state,
    build_registration_command,
    discover_sessions,
    geometry_matches,
    process_session,
)
from andi_rewrite.scripts.train_fomo45k_brats21_checked import (
    validate_fomo45k_brats21_config,
)


class FOMO45KBraTS21EdgeCaseTest(unittest.TestCase):
    def test_recursive_discovery_excludes_duplicate_t1_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "sub_1" / "ses_1"
            session.mkdir(parents=True)
            for name in ("t1.nii.gz", "scan_t1w.nii.gz", "t2.nii.gz", "flair.nii.gz"):
                nib.save(nib.Nifti1Image(np.ones((3, 4, 2)), np.eye(4)), session / name)
            record = discover_sessions(root)[0]
            self.assertEqual(record.status, "EXCLUDED")
            self.assertIn("ambiguous_t1:2", record.reasons)
            self.assertIsNone(record.t1_path)

    def test_corrupt_nifti_is_rejected_without_guessing_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.nii.gz"
            broken = root / "broken.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 4, 2)), np.eye(4)), good)
            broken.write_bytes(b"not a nifti")
            with self.assertRaises(Exception):
                geometry_matches(good, broken)

    def test_rigid_only_modality_command_has_no_affine_or_syn(self) -> None:
        command = build_registration_command(
            "antsRegistration",
            "t1.nii.gz",
            "flair.nii.gz",
            "t1_mask.nii.gz",
            "flair_mask.nii.gz",
            "out_",
            affine=False,
            random_seed=73,
        )
        transforms = [command[i + 1] for i, value in enumerate(command) if value == "--transform"]
        self.assertEqual(transforms, ["Rigid[0.1]"])
        self.assertNotIn("SyN", " ".join(command))

    def test_matching_signature_resumes_without_running_external_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "sub_1" / "ses_1"
            source.mkdir(parents=True)
            paths = {}
            for modality in ("t1", "t2", "flair"):
                path = source / f"{modality}.nii.gz"
                path.write_bytes(modality.encode("ascii"))
                paths[modality] = path
            record = SessionRecord(
                participant_id="sub_1",
                session_id="ses_1",
                relative_dir="sub_1/ses_1",
                t1_path=str(paths["t1"]),
                t2_path=str(paths["t2"]),
                flair_path=str(paths["flair"]),
                status="READY",
            )
            settings = RegistrationSettings()
            preflight = {"atlas": {"atlas_t1": {"sha256": "atlas"}}}
            signature = _signature(_source_state(record), preflight, settings)
            target = root / "output" / "sub_1" / "ses_1"
            target.mkdir(parents=True)
            (target / "status.json").write_text(
                json.dumps({"case_id": record.case_id, "status": "PASS", "signature": signature}),
                encoding="utf-8",
            )
            toolchain = Toolchain("a", "b", "c", "d", "e", "f", "g")
            with mock.patch(
                "andi_rewrite.data.fomo45k.brats21._prepare_registration_input",
                side_effect=AssertionError("resume must not start processing"),
            ):
                result = process_session(
                    record, root / "output", toolchain, settings, preflight, resume=True
                )
            self.assertTrue(result["resumed"])
            self.assertEqual(result["status"], "PASS")

    def test_checked_training_rejects_non_three_channel_config_and_checkpoint(self) -> None:
        base = {
            "data": {"type": "fomo45k_brats21"},
            "model": {"in_channels": 3, "out_channels": 3},
            "training": {},
        }
        invalid_config = {**base, "model": {"in_channels": 4, "out_channels": 4}}
        with self.assertRaisesRegex(ValueError, "exactly FLAIR,T1,T2"):
            validate_fomo45k_brats21_config(invalid_config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "four_channel.pt"
            torch.save({"model": {"inc.double_conv.0.weight": torch.zeros(32, 4, 3, 3)}}, checkpoint)
            config = {
                **base,
                "model": {"in_channels": 3, "out_channels": 3, "checkpoint": str(checkpoint)},
            }
            with self.assertRaisesRegex(ValueError, "expects 4 input channels"):
                validate_fomo45k_brats21_config(config)


if __name__ == "__main__":
    unittest.main()
