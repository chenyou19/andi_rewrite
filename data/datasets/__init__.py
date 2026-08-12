"""Compatibility facade for the dataset package.

The historical ``andi_rewrite.data.datasets`` import path remains the complete
dataset surface while implementations live in independent adapter modules.
"""

from .brats import (
    BraTSHealthySliceDataset,
    MRIDataVolume,
    _subject_frame_from_directory,
    build_brats_healthy_slices_dataset,
    build_mri_volume_dataset,
)
from .common import _as_list, _shape_text, _subject_file_path
from .factory import DATASET_BUILDERS, DatasetBuilder, build_dataloader, build_dataset
from .imaging import histogram_normalize_volume, normalize_volume
from .lmdb import LMDBSliceDataset, build_lmdb_dataset
from .shifts_ms import (
    ShiftsMSSubject,
    ShiftsMSVolumeDataset,
    _NIFTI_SUFFIXES,
    _SHIFTS_LOCATIONS,
    _SHIFTS_SPLITS,
    _is_nifti,
    _strip_nifti_suffix,
    _subject_id_from_shifts_path,
    build_shifts_ms_dataset,
    build_shifts_ms_volume_dataset,
)
from .ucsf_pdgm import (
    UCSFPDGMSubject,
    UCSFPDGMVolumeDataset,
    _nifti_orientation_code,
    _ucsf_pdgm_file_stem,
    _validate_nifti_geometry,
    build_ucsf_pdgm_dataset,
)

__all__ = [
    "BraTSHealthySliceDataset",
    "DATASET_BUILDERS",
    "DatasetBuilder",
    "LMDBSliceDataset",
    "MRIDataVolume",
    "ShiftsMSSubject",
    "ShiftsMSVolumeDataset",
    "UCSFPDGMSubject",
    "UCSFPDGMVolumeDataset",
    "build_brats_healthy_slices_dataset",
    "build_dataloader",
    "build_dataset",
    "build_lmdb_dataset",
    "build_mri_volume_dataset",
    "build_shifts_ms_dataset",
    "build_shifts_ms_volume_dataset",
    "build_ucsf_pdgm_dataset",
    "histogram_normalize_volume",
    "normalize_volume",
]
