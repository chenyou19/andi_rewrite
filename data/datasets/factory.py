"""Dataset adapter selection and DataLoader construction.

All adapter-specific configuration interpretation stays in the owning adapter
module. This module only maps stable type aliases to those builders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.utils.data import DataLoader, Dataset

from .brats import build_brats_healthy_slices_dataset, build_mri_volume_dataset
from .lmdb import build_lmdb_dataset
from .shifts_ms import build_shifts_ms_volume_dataset
from .ucsf_pdgm import build_ucsf_pdgm_dataset


DatasetBuilder = Callable[[dict[str, Any]], Dataset]


DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "lmdb": build_lmdb_dataset,
    "brats_healthy_slices": build_brats_healthy_slices_dataset,
    "healthy_slices": build_brats_healthy_slices_dataset,
    "volume": build_mri_volume_dataset,
    "mri_volume": build_mri_volume_dataset,
    "brats_volume": build_mri_volume_dataset,
    "ljubljana_ms_volume": build_shifts_ms_volume_dataset,
    "shifts_ms_volume": build_shifts_ms_volume_dataset,
    "shifts_volume": build_shifts_ms_volume_dataset,
    "ucsf_pdgm": build_ucsf_pdgm_dataset,
    "ucsf_pdgm_volume": build_ucsf_pdgm_dataset,
}


def build_dataset(config: dict[str, Any]) -> Dataset:
    """Build a dataset from the legacy data configuration mapping."""

    data_type = str(config.get("type", "lmdb")).lower()
    builder = DATASET_BUILDERS.get(data_type)
    if builder is None:
        # The monolithic factory coerced image_size before reporting an unknown
        # type. Preserve that error precedence without coercing known adapters
        # twice; their owning builders perform the one normal conversion.
        int(config.get("image_size", 128))
        raise ValueError(f"Unknown dataset type: {data_type}")
    return builder(config)


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=bool(config.get("shuffle", False)),
        num_workers=int(config.get("workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )
