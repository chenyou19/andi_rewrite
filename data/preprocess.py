"""Compatibility facade for healthy-slice preprocessing.

The established preprocessing import path remains stable while MRI primitives,
selection, split policy, and LMDB product ownership live in cohesive leaf modules.
"""

from .healthy_slices import (
    BalancedSliceRecord,
    HealthySliceCandidate,
    balance_candidates_by_z,
    healthy_slice_candidates_for_subject,
    healthy_slices_for_subject,
    iter_healthy_slices,
)
from .imaging import (
    DEFAULT_MODALITIES,
    _load_nifti,
    load_subject_volume,
    normalize_volume,
    resize_slices,
)
from .lmdb_io import (
    KFoldLMDBSummary,
    SplitHealthySummary,
    estimate_balanced_lmdb_map_size,
    estimate_lmdb_map_size,
    split_healthy_kfold_to_lmdb,
    split_healthy_to_lmdb,
    split_healthy_z_balanced_to_lmdb,
)
from .subject_splits import (
    build_combined_train_test_splits,
    build_kfold_subject_splits,
    build_repeated_train_test_splits,
    read_subject_csv,
)


__all__ = [
    "DEFAULT_MODALITIES",
    "BalancedSliceRecord",
    "HealthySliceCandidate",
    "KFoldLMDBSummary",
    "SplitHealthySummary",
    "_load_nifti",
    "balance_candidates_by_z",
    "build_combined_train_test_splits",
    "build_kfold_subject_splits",
    "build_repeated_train_test_splits",
    "estimate_balanced_lmdb_map_size",
    "estimate_lmdb_map_size",
    "healthy_slice_candidates_for_subject",
    "healthy_slices_for_subject",
    "iter_healthy_slices",
    "load_subject_volume",
    "normalize_volume",
    "read_subject_csv",
    "resize_slices",
    "split_healthy_kfold_to_lmdb",
    "split_healthy_to_lmdb",
    "split_healthy_z_balanced_to_lmdb",
]
