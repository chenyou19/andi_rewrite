from __future__ import annotations

import csv
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.data.lemon_mip.geometry import derive_fixed_square_roi, resize_roi_slice
from andi_rewrite.data.lemon_mip.manifest import (
    InputValidationError,
    SessionRecord,
    deterministic_session_split,
    validate_session_csv,
)
from andi_rewrite.data.lemon_mip.processing import _dilate_mask_mm
from andi_rewrite.data.lemon_mip.pipeline import (
    clear_explicit_product_dir,
    explicit_exclusion_ids,
    select_release_records,
)
from andi_rewrite.data.lemon_mip.qc import affine_qc_metrics, automatic_qc_status, dice_score
from andi_rewrite.data.lemon_mip.registration import (
    RegistrationSettings,
    compose_forward_transforms,
    dipy_pull_to_forward,
    resample_with_forward_transform,
    register_affine_multistage,
)
from andi_rewrite.data.lemon_mip.store import (
    audit_published_lmdb,
    audit_staging_lmdb,
    build_staging_lmdb,
    publish_lmdb,
    save_session_bundle,
)
from andi_rewrite.data.datasets.lmdb import LMDBSliceDataset


class LemonManifestTest(unittest.TestCase):
    def _save(self, path: Path, value: np.ndarray, affine: np.ndarray | None = None) -> None:
        nib.save(nib.Nifti1Image(value, np.eye(4) if affine is None else affine), path)

    def _fixture(self, root: Path, count: int = 2) -> Path:
        rows = []
        for index in range(count):
            case = f"sub-{index:06}"
            directory = root / case / "ses-01" / "anat"
            directory.mkdir(parents=True)
            paths = {}
            for name in ("t1", "t2", "flair"):
                path = directory / f"{name}.nii.gz"
                self._save(path, np.ones((4, 5, 6), dtype=np.float32) * (index + 1))
                paths[name] = path.relative_to(root)
            rows.append(
                {
                    "case_id": case,
                    "session": "ses-01",
                    "t1w_path": str(paths["t1"]),
                    "t2w_path": str(paths["t2"]),
                    "highres_flair_path": str(paths["flair"]),
                }
            )
        path = root / "sessions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_relative_paths_full_voxel_validation_and_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._fixture(root)
            records = validate_session_csv(source, expected_sessions=2, validate_voxels=True)
            self.assertEqual([record.csv_index for record in records], [0, 1])
            self.assertEqual(records[0].session_id, "sub-000000/ses-01")
            self.assertTrue(records[0].t1_path.is_absolute())

    def test_schema_duplicate_missing_path_wrong_dimensionality_nonfinite_and_unsafe_affine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._fixture(root)
            original = list(csv.DictReader(source.open(encoding="utf-8")))

            missing_column = root / "missing_column.csv"
            with missing_column.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "session", "t1w_path", "t2w_path"])
                writer.writeheader()
                writer.writerows({key: value for key, value in row.items() if key != "highres_flair_path"} for row in original)
            with self.assertRaisesRegex(InputValidationError, "missing required columns"):
                validate_session_csv(missing_column, expected_sessions=2)

            extra = root / "extra.csv"
            with extra.open("w", newline="", encoding="utf-8") as handle:
                fields = [*original[0], "pd_path"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({**row, "pd_path": row["t1w_path"]} for row in original)
            with self.assertRaisesRegex(InputValidationError, "unexpected modality/path columns"):
                validate_session_csv(extra, expected_sessions=2)

            broken = root / "broken.csv"
            rows = [dict(row) for row in original]
            rows[1]["case_id"] = rows[0]["case_id"]
            rows[1]["session"] = rows[0]["session"]
            rows[0]["t2w_path"] = "missing.nii.gz"
            self._save(root / rows[0]["t1w_path"], np.ones((2, 2, 2, 2), dtype=np.float32))
            bad_flair = np.ones((4, 5, 6), dtype=np.float32)
            bad_flair[0, 0, 0] = np.nan
            self._save(root / rows[0]["highres_flair_path"], bad_flair)
            singular = np.eye(4)
            singular[2, 2] = 0
            singular_image = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4))
            singular_image.set_qform(np.eye(4), code=0)
            singular_image.set_sform(singular, code=1)
            nib.save(singular_image, root / rows[1]["t1w_path"])
            with broken.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(InputValidationError) as caught:
                validate_session_csv(broken, expected_sessions=2, validate_voxels=True)
            message = str(caught.exception)
            for expected in (
                "duplicate case/session",
                "path does not exist",
                "expected exactly 3D",
                "voxel data has 1 NaN/Inf",
                "affine spatial matrix is singular",
                "case/session=",
                "modality=",
                "path=",
                "reason=",
            ):
                self.assertIn(expected, message)

    def test_split_is_pcg64_seeded_and_preserves_csv_order(self) -> None:
        records = [
            SessionRecord(index, f"case-{index:03}", "ses-01", Path("t1"), Path("t2"), Path("f"))
            for index in reversed(range(20))
        ]
        train, validation, metadata = deterministic_session_split(records, validation_count=5, seed=73)
        sorted_ids = sorted(record.session_id for record in records)
        positions = np.random.Generator(np.random.PCG64(73)).permutation(20)[:5]
        expected = {sorted_ids[int(position)] for position in positions}
        self.assertEqual({record.session_id for record in validation}, expected)
        self.assertEqual(
            [record.csv_index for record in train],
            sorted((record.csv_index for record in train), reverse=True),
        )
        self.assertEqual(metadata["seed"], 73)
        self.assertTrue({record.session_id for record in train}.isdisjoint(expected))


class LemonGeometryRegistrationTest(unittest.TestCase):
    def test_fixed_roi_is_physical_square_margin_fixed_and_binary_resize(self) -> None:
        mask = np.zeros((20, 30, 8), dtype=np.uint8)
        mask[5:9, 8:16, 2:7] = 1
        affine = np.diag([2.0, 1.0, 3.0, 1.0])
        roi = derive_fixed_square_roi(mask, affine, margin_mm=0.0, output_size=128)
        self.assertAlmostEqual(roi.crop_shape[0] * 2.0, roi.crop_shape[1] * 1.0, delta=2.0)
        self.assertEqual(roi.output_size, 128)
        self.assertEqual(roi, derive_fixed_square_roi(mask.copy(), affine.copy(), margin_mm=0.0, output_size=128))
        resized = resize_roi_slice(mask[:, :, 3], roi, is_mask=True)
        self.assertEqual(resized.shape, (128, 128))
        self.assertEqual(set(np.unique(resized)), {1})

        margin_roi = derive_fixed_square_roi(mask, affine, margin_mm=8.0, output_size=128)
        self.assertGreater(margin_roi.side_mm, roi.side_mm)

        boundary_mask = np.ones((10, 14, 3), dtype=np.uint8)
        padded_roi = derive_fixed_square_roi(
            boundary_mask, np.eye(4), margin_mm=2.0, output_size=32
        )
        self.assertTrue(
            padded_roi.x_start < 0
            or padded_roi.y_start < 0
            or padded_roi.x_stop > boundary_mask.shape[0]
            or padded_roi.y_stop > boundary_mask.shape[1]
        )
        padded = resize_roi_slice(boundary_mask[:, :, 1], padded_roi, is_mask=True)
        self.assertEqual(padded.shape, (32, 32))
        self.assertIn(0, np.unique(padded))
        self.assertIn(1, np.unique(padded))

    def test_physical_guard_dilation_and_clipping_retention(self) -> None:
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[4, 4, 4] = 1
        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        dilated = _dilate_mask_mm(mask, affine, 2.1)
        self.assertEqual(dilated[5, 4, 4], 1)
        self.assertEqual(dilated[6, 4, 4], 0)
        self.assertEqual(dilated[4, 6, 4], 1)
        self.assertEqual(dilated[4, 7, 4], 0)
        subject = dilated.astype(bool)
        final = subject & dilated.astype(bool)
        self.assertEqual(float(final.sum() / subject.sum()), 1.0)

    def test_pull_forward_convention_composition_and_interpolation(self) -> None:
        flair_to_t1 = np.eye(4)
        flair_to_t1[0, 3] = 1.0
        t1_to_mni = np.eye(4)
        t1_to_mni[1, 3] = 2.0
        composed = compose_forward_transforms(flair_to_t1, t1_to_mni)
        np.testing.assert_allclose(composed, t1_to_mni @ flair_to_t1)
        np.testing.assert_allclose(dipy_pull_to_forward(np.linalg.inv(flair_to_t1)), flair_to_t1)

        moving = np.zeros((7, 7, 7), dtype=np.float32)
        moving[2, 3, 3] = 1.0
        linear = resample_with_forward_transform(
            moving, np.eye(4), moving.shape, np.eye(4), flair_to_t1, interpolation="linear"
        )
        nearest = resample_with_forward_transform(
            moving, np.eye(4), moving.shape, np.eye(4), flair_to_t1, interpolation="nearest"
        )
        self.assertAlmostEqual(float(linear[3, 3, 3]), 1.0, places=6)
        self.assertEqual(nearest[3, 3, 3], 1)
        self.assertEqual(set(np.unique(nearest)), {0, 1})

    def test_affine_polar_rotation_singular_values_shear_and_qc_thresholds(self) -> None:
        angle = np.deg2rad(30.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        stretch = np.array([[2.0, 0.2, 0], [0.2, 1.0, 0.1], [0, 0.1, 0.5]])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ stretch
        affine[:3, 3] = [3, 4, 0]
        metrics = affine_qc_metrics(affine)
        self.assertAlmostEqual(metrics["translation_norm"], 5.0, places=6)
        self.assertGreater(metrics["rotation_norm"], 20.0)
        self.assertGreaterEqual(metrics["sv1"], metrics["sv2"])
        self.assertGreaterEqual(metrics["sv2"], metrics["sv3"])
        self.assertGreater(metrics["shear_norm"], 0)

        base = {
            "image_finite": True,
            "mask_finite": True,
            "geometry_valid": True,
            "shape_valid": True,
            "brain_volume_L": 1.2,
            "det": 1.0,
            "sv_min": 0.8,
            "sv_max": 1.2,
            "mask_dice": 0.8,
            "T1_MNI_MI_improvement": 0.1,
            "FLAIR_T1_MI_improvement": 0.1,
            "T2_T1_MI_improvement": 0.1,
            "guard_retention": 0.99,
        }
        self.assertEqual(automatic_qc_status(base), ("PASS", []))
        status, flags = automatic_qc_status({**base, "mask_dice": 0.699})
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("mask_dice" in flag for flag in flags))
        self.assertEqual(dice_score(np.ones(3), np.ones(3)), 1.0)

    def test_real_dipy_registration_accepts_uint8_masks(self) -> None:
        static = np.zeros((16, 16, 16), dtype=np.float32)
        static[4:12, 4:12, 4:12] = 1.0
        moving = np.roll(static, shift=1, axis=0)
        mask = (static > 0).astype(np.uint8)
        result = register_affine_multistage(
            static,
            np.eye(4),
            moving,
            np.eye(4),
            final_transform="rigid",
            settings=RegistrationSettings(
                nbins=16, level_iters=(10, 3, 1), sigmas=(1, 0, 0), factors=(2, 1, 1)
            ),
            static_mask=mask,
        )
        self.assertEqual(result.transformed.shape, static.shape)
        self.assertTrue(np.all(np.isfinite(result.forward_moving_to_static_world)))


class LemonLMDBTest(unittest.TestCase):
    def _result(self, record: SessionRecord, z_values: list[int]):
        slices = []
        for z in z_values:
            value = np.stack(
                [
                    np.full((128, 128), z + 0.1, dtype=np.float32),
                    np.full((128, 128), z + 0.2, dtype=np.float32),
                    np.full((128, 128), z + 0.3, dtype=np.float32),
                ]
            )
            slices.append((z, value))
        stats = {
            "channel_order": ["FLAIR", "T1", "T2"],
            **{
                modality: {
                    "masked_percentiles": {},
                    "roi_fraction_lt_0": 0.0,
                    "roi_fraction_gt_1": 0.0,
                    "roi_fraction_eq_0": 0.0,
                }
                for modality in ("FLAIR", "T1", "T2")
            },
        }
        return SimpleNamespace(
            record=record,
            slices=slices,
            metrics={"slice_count": len(slices), "case_id": record.case_id, "session": record.session},
            intensity_statistics=stats,
        )

    def test_staging_keys_values_manifest_audit_and_publication(self) -> None:
        import lmdb

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                SessionRecord(0, "case-a", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
                SessionRecord(1, "case-b", "ses-01", Path("t1"), Path("t2"), Path("f"), "val"),
                SessionRecord(2, "case-c", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
            ]
            for record, z_values in zip(records, ([2, 4], [1], [3])):
                save_session_bundle(self._result(record, list(z_values)), root, processing_fingerprint="fixture")
            report = build_staging_lmdb(
                records, root, processing_fingerprint="fixture", map_size=16 * 1024**2
            )
            self.assertEqual(report["split_counts"], {"train": 3, "val": 1})
            self.assertEqual(report["session_count"], 3)
            audit = audit_staging_lmdb(records, root, processing_fingerprint="fixture")
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["shape"], [3, 128, 128])

            with (root / "MIP_lmdb" / "manifest.building.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            train_rows = [row for row in rows if row["split"] == "train"]
            self.assertEqual([row["key"] for row in train_rows], ["00000000", "00000001", "00000002"])
            self.assertEqual([(row["case_id"], int(row["z"])) for row in train_rows], [("case-a", 2), ("case-a", 4), ("case-c", 3)])
            self.assertTrue({row["session_id"] for row in train_rows}.isdisjoint({row["session_id"] for row in rows if row["split"] == "val"}))

            environment = lmdb.open(str(root / "MIP_lmdb" / "train.building"), readonly=True, lock=False)
            try:
                with environment.begin() as transaction:
                    value = pickle.loads(transaction.get(b"00000000"))
                    self.assertEqual(value.shape, (3, 128, 128))
                    self.assertEqual(value.dtype, np.float32)
                    self.assertAlmostEqual(float(value[0, 0, 0]), 2.1, places=6)
            finally:
                environment.close()

            publication = publish_lmdb(root)
            self.assertEqual(publication["status"], "PUBLISHED")
            self.assertEqual(publication["product_state"], "PUBLISHED")
            self.assertTrue((root / "MIP_lmdb" / "train").is_dir())
            self.assertTrue((root / "MIP_lmdb" / "val").is_dir())
            self.assertTrue((root / "MIP_lmdb" / "manifest.csv").is_file())
            self.assertFalse((root / "MIP_lmdb" / "train.building").exists())
            self.assertEqual(
                audit_published_lmdb(records, root, processing_fingerprint="fixture")["product_state"],
                "PUBLISHED",
            )
            selected = LMDBSliceDataset(
                root / "MIP_lmdb" / "train", image_size=128, channel_indices=[2, 0, 1]
            )[0]
            self.assertEqual(selected.shape, (3, 128, 128))
            self.assertAlmostEqual(float(selected[0, 0, 0]), 2.3, places=6)
            self.assertAlmostEqual(float(selected[1, 0, 0]), 2.1, places=6)
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                LMDBSliceDataset(root / "MIP_lmdb" / "train", channel_indices=[])
            with self.assertRaisesRegex(IndexError, "out of range"):
                LMDBSliceDataset(root / "MIP_lmdb" / "train", channel_indices=[3])[0]

    def test_explicit_release_exclusion_preserves_split_and_requires_exact_adjudication(self) -> None:
        records = [
            SessionRecord(0, "case-a", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
            SessionRecord(1, "case-b", "ses-01", Path("t1"), Path("t2"), Path("f"), "val"),
            SessionRecord(2, "case-c", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
        ]
        config = {
            "explicit_exclusions": {
                "authorized": True,
                "preserve_original_split": True,
                "expected_excluded_sessions": 1,
                "expected_remaining_sessions": 2,
                "expected_remaining_split_counts": {"train": 1, "val": 1},
                "session_ids": ["case-c/ses-01"],
            }
        }
        reviews = [{"session_id": "case-c/ses-01", "manual_status": "EXCLUDE"}]
        selected, metadata = select_release_records(config, records, reviews)
        self.assertEqual(explicit_exclusion_ids(config), ("case-c/ses-01",))
        self.assertEqual([record.session_id for record in selected], ["case-a/ses-01", "case-b/ses-01"])
        self.assertEqual([record.split for record in selected], ["train", "val"])
        self.assertEqual(metadata["selected_split_counts"], {"train": 1, "val": 1})
        with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
            select_release_records(
                config,
                records,
                [{"session_id": "case-c/ses-01", "manual_status": "PASS"}],
            )

    def test_existing_lmdb_build_rejects_different_release_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                SessionRecord(0, "case-a", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
                SessionRecord(1, "case-b", "ses-01", Path("t1"), Path("t2"), Path("f"), "val"),
                SessionRecord(2, "case-c", "ses-01", Path("t1"), Path("t2"), Path("f"), "train"),
            ]
            for record in records:
                save_session_bundle(self._result(record, [1]), root, processing_fingerprint="fixture")
            build_staging_lmdb(
                records,
                root,
                processing_fingerprint="fixture",
                map_size=16 * 1024**2,
            )
            with self.assertRaisesRegex(ValueError, "different release selection"):
                build_staging_lmdb(
                    records[:2],
                    root,
                    processing_fingerprint="fixture",
                    map_size=16 * 1024**2,
                )

    def test_overwrite_requires_one_explicit_directory_below_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = root / "one" / "product"
            product.mkdir(parents=True)
            (product / "owned.txt").write_text("fixture", encoding="utf-8")
            config = {"output_root": str(root)}
            self.assertEqual(clear_explicit_product_dir(config, product), product.resolve())
            self.assertFalse(product.exists())
            with self.assertRaisesRegex(ValueError, "specific directory below"):
                clear_explicit_product_dir(config, root)


if __name__ == "__main__":
    unittest.main()
