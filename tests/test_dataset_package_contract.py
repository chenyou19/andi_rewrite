"""Public contracts for the modular ``data.datasets`` package."""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from collections.abc import MutableMapping
from pathlib import Path
from unittest.mock import patch

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_MODULE = "andi_rewrite.data"
DATASETS_MODULE = "andi_rewrite.data.datasets"

# The stable aliases deliberately point at adapter-local builders, rather than
# adapter classes, so each adapter retains ownership of its config translation.
ALIAS_BUILDERS = {
    "lmdb": ("lmdb", "build_lmdb_dataset"),
    "brats_healthy_slices": ("brats", "build_brats_healthy_slices_dataset"),
    "healthy_slices": ("brats", "build_brats_healthy_slices_dataset"),
    "volume": ("brats", "build_mri_volume_dataset"),
    "mri_volume": ("brats", "build_mri_volume_dataset"),
    "brats_volume": ("brats", "build_mri_volume_dataset"),
    "ljubljana_ms_volume": ("shifts_ms", "build_shifts_ms_volume_dataset"),
    "shifts_ms_volume": ("shifts_ms", "build_shifts_ms_volume_dataset"),
    "shifts_volume": ("shifts_ms", "build_shifts_ms_volume_dataset"),
    "ucsf_pdgm": ("ucsf_pdgm", "build_ucsf_pdgm_dataset"),
    "ucsf_pdgm_volume": ("ucsf_pdgm", "build_ucsf_pdgm_dataset"),
}

FACADE_OBJECTS = {
    "LMDBSliceDataset": ("lmdb", "LMDBSliceDataset"),
    "BraTSHealthySliceDataset": ("brats", "BraTSHealthySliceDataset"),
    "MRIDataVolume": ("brats", "MRIDataVolume"),
    "ShiftsMSVolumeDataset": ("shifts_ms", "ShiftsMSVolumeDataset"),
    "UCSFPDGMVolumeDataset": ("ucsf_pdgm", "UCSFPDGMVolumeDataset"),
    "build_dataset": ("factory", "build_dataset"),
    "build_dataloader": ("factory", "build_dataloader"),
}

DATA_PACKAGE_EXPORTS = (
    "BraTSHealthySliceDataset",
    "MRIDataVolume",
    "ShiftsMSVolumeDataset",
    "UCSFPDGMVolumeDataset",
    "build_dataset",
    "build_dataloader",
)


class DatasetPackageContractTest(unittest.TestCase):
    """Characterize package facades and factory routing without opening data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = importlib.import_module(DATA_MODULE)
        cls.datasets = importlib.import_module(DATASETS_MODULE)
        cls.factory = importlib.import_module(f"{DATASETS_MODULE}.factory")

    @staticmethod
    def _adapter(module_name: str):
        return importlib.import_module(f"{DATASETS_MODULE}.{module_name}")

    def test_facade_and_data_package_keep_adapter_object_identity(self) -> None:
        for facade_name, (module_name, adapter_name) in FACADE_OBJECTS.items():
            adapter = self._adapter(module_name)
            with self.subTest(facade_name=facade_name, adapter=module_name):
                self.assertIs(
                    getattr(self.datasets, facade_name),
                    getattr(adapter, adapter_name),
                )

        self.assertIs(self.datasets.DATASET_BUILDERS, self.factory.DATASET_BUILDERS)
        for name in DATA_PACKAGE_EXPORTS:
            with self.subTest(data_package_export=name):
                self.assertIn(name, self.data.__all__)
                self.assertIs(getattr(self.data, name), getattr(self.datasets, name))

    def test_factory_aliases_select_adapter_builders_and_pass_the_exact_config(self) -> None:
        registry = self.factory.DATASET_BUILDERS

        for alias, (module_name, builder_name) in ALIAS_BUILDERS.items():
            adapter_builder = getattr(self._adapter(module_name), builder_name)
            with self.subTest(alias=alias):
                self.assertIn(alias, registry)
                self.assertIs(registry[alias], adapter_builder)

                product = object()
                received: list[dict[str, object]] = []

                def record_builder(config: dict[str, object]) -> object:
                    received.append(config)
                    return product

                config = {"type": alias, "phase3_contract_marker": alias}
                with patch.dict(registry, {alias: record_builder}):
                    self.assertIs(self.datasets.build_dataset(config), product)

                self.assertEqual(received, [config])
                self.assertIs(received[0], config)

    def test_factory_mapping_accepts_a_scoped_extension_when_it_is_public(self) -> None:
        registry = getattr(self.factory, "DATASET_BUILDERS", None)
        if not isinstance(registry, MutableMapping):
            return

        alias = "phase3_contract_extension"
        absent = object()
        previous = registry.get(alias, absent)
        product = object()
        received: list[dict[str, object]] = []

        def extension_builder(config: dict[str, object]) -> object:
            received.append(config)
            return product

        try:
            registry[alias] = extension_builder
            config = {"type": alias, "phase3_contract_marker": True}
            self.assertIs(self.datasets.build_dataset(config), product)
            self.assertEqual(received, [config])
            self.assertIs(received[0], config)
        finally:
            if previous is absent:
                registry.pop(alias, None)
            else:
                registry[alias] = previous

    def test_adapter_builders_preserve_legacy_config_translation(self) -> None:
        cases = (
            (
                "lmdb",
                "LMDBSliceDataset",
                {"type": "lmdb", "path": "healthy.lmdb", "image_size": "64"},
                (("healthy.lmdb",), {"image_size": 64}),
            ),
            (
                "brats",
                "BraTSHealthySliceDataset",
                {
                    "type": "healthy_slices",
                    "path_to_csv": "healthy.csv",
                    "dataset_path": "brats",
                    "image_size": "96",
                    "modalities": ["flair", "t1"],
                },
                (
                    (),
                    {
                        "csv_path": "healthy.csv",
                        "dataset_path": "brats",
                        "image_size": 96,
                        "modalities": ["flair", "t1"],
                        "slice_column": "Slice",
                        "filename_separator": "_",
                    },
                ),
            ),
            (
                "brats",
                "MRIDataVolume",
                {
                    "type": "volume",
                    "dataset_path": "C:/datasets/shifts-like",
                    "image_size": "80",
                },
                (
                    (),
                    {
                        "csv_path": None,
                        "dataset_path": "C:/datasets/shifts-like",
                        "image_size": 80,
                        "modalities": None,
                        "segmentation_suffix": "seg",
                        "histogram_normalization": False,
                        "shift_naming": True,
                        "filename_separator": "_",
                        "return_metadata": False,
                    },
                ),
            ),
            (
                "shifts_ms",
                "ShiftsMSVolumeDataset",
                {"type": "ljubljana_ms_volume", "dataset_path": "shifts", "image_size": "72"},
                (
                    (),
                    {
                        "dataset_path": "shifts",
                        "image_size": 72,
                        "modalities": None,
                        "modality_mapping": None,
                        "dataset_subdir": None,
                        "locations": None,
                        "location": None,
                        "preferred_locations": None,
                        "splits": None,
                        "reference_modality": "flair",
                        "require_segmentation": True,
                        "require_modalities": True,
                        "resample_to_reference": True,
                        "histogram_normalization": False,
                        "return_metadata": True,
                        "subject_limit": None,
                    },
                ),
            ),
            (
                "ucsf_pdgm",
                "UCSFPDGMVolumeDataset",
                {"type": "ucsf_pdgm", "dataset_path": "ucsf", "image_size": "88"},
                (
                    (),
                    {
                        "dataset_path": "ucsf",
                        "image_size": 88,
                        "modalities": None,
                        "modality_mapping": None,
                        "segmentation_suffix": "tumor_segmentation",
                        "reference_modality": "flair",
                        "model_orientation": "LPS",
                        "histogram_normalization": False,
                        "return_metadata": True,
                        "subject_limit": None,
                        "duplicate_policy": "error",
                        "csv_path": None,
                    },
                ),
            ),
        )

        for module_name, constructor_name, config, expected in cases:
            adapter = self._adapter(module_name)
            product = object()
            with self.subTest(type=config["type"]), patch.object(
                adapter,
                constructor_name,
                return_value=product,
            ) as constructor:
                self.assertIs(self.datasets.build_dataset(config), product)
                expected_args, expected_kwargs = expected
                constructor.assert_called_once_with(*expected_args, **expected_kwargs)

    def test_image_size_is_coerced_once_and_legacy_error_precedence_is_kept(self) -> None:
        class OneShotInteger:
            def __init__(self) -> None:
                self.calls = 0

            def __int__(self) -> int:
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("image_size was coerced more than once")
                return 64

        image_size = OneShotInteger()
        lmdb = self._adapter("lmdb")
        with patch.object(lmdb, "LMDBSliceDataset", return_value=object()):
            self.datasets.build_dataset(
                {"type": "lmdb", "path": "healthy.lmdb", "image_size": image_size}
            )
        self.assertEqual(image_size.calls, 1)

        with self.assertRaisesRegex(ValueError, "invalid literal"):
            self.datasets.build_dataset({"type": "unknown", "image_size": "not-an-int"})
        with self.assertRaisesRegex(ValueError, "Unknown dataset type: unknown"):
            self.datasets.build_dataset({"type": "unknown", "image_size": "64"})

        for data_type in (
            "brats_healthy_slices",
            "volume",
            "ljubljana_ms_volume",
            "ucsf_pdgm_volume",
        ):
            with self.subTest(error_precedence=data_type), self.assertRaisesRegex(
                ValueError,
                "invalid literal",
            ):
                self.datasets.build_dataset(
                    {"type": data_type, "image_size": "not-an-int"}
                )

    def test_every_checked_in_data_type_has_a_factory_alias(self) -> None:
        observed: dict[str, list[str]] = {}
        config_paths = sorted(REPO_ROOT.joinpath("configs").rglob("*.yaml"))
        config_paths.extend(sorted(REPO_ROOT.joinpath("configs").rglob("*.yml")))

        for path in config_paths:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or not isinstance(document.get("data"), dict):
                continue
            data_config = document["data"]
            if "type" not in data_config:
                continue
            data_type = data_config["type"]
            self.assertIsInstance(data_type, str, f"{path} data.type must be a string")
            observed.setdefault(data_type.lower(), []).append(path.relative_to(REPO_ROOT).as_posix())

        self.assertTrue(observed, "No checked-in YAML data.type values were discovered.")
        missing = sorted(set(observed).difference(self.factory.DATASET_BUILDERS))
        self.assertFalse(
            missing,
            "Checked-in YAML data.type values are not factory aliases: "
            f"{[(name, observed[name]) for name in missing]}",
        )

    def test_fresh_interpreter_imports_dataset_package_and_leaf_modules(self) -> None:
        modules = (
            DATA_MODULE,
            DATASETS_MODULE,
            f"{DATASETS_MODULE}.common",
            f"{DATASETS_MODULE}.imaging",
            f"{DATASETS_MODULE}.lmdb",
            f"{DATASETS_MODULE}.brats",
            f"{DATASETS_MODULE}.shifts_ms",
            f"{DATASETS_MODULE}.ucsf_pdgm",
            f"{DATASETS_MODULE}.factory",
        )
        script = "import importlib\n" + f"for name in {modules!r}:\n    importlib.import_module(name)\n"
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
            "fresh dataset package import failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
