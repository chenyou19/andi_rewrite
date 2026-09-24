"""Auditable Stage1 data layer for the domain-classifier experiment."""

from .audit import audit_records, audit_source_provenance
from .matching import (
    PairingResult,
    SPLITS,
    apply_participant_splits,
    assert_participant_split_disjoint,
    build_pairs,
    participant_split_map,
)
from .readers import (
    DomainClassifierDataset,
    file_fingerprint,
    load_mask_slice,
    load_records,
    load_slice,
    validate_tumor_free,
    write_records,
)
from .records import (
    DEFAULT_Z_BINS,
    MODALITIES,
    MODEL_SHAPE,
    SliceRecord,
    namespaced_participant,
    record_from_paths,
    z_bin_for,
)

__all__ = [
    "DEFAULT_Z_BINS",
    "DomainClassifierDataset",
    "MODALITIES",
    "MODEL_SHAPE",
    "PairingResult",
    "SPLITS",
    "SliceRecord",
    "apply_participant_splits",
    "assert_participant_split_disjoint",
    "audit_records",
    "audit_source_provenance",
    "build_pairs",
    "file_fingerprint",
    "load_mask_slice",
    "load_records",
    "load_slice",
    "namespaced_participant",
    "participant_split_map",
    "record_from_paths",
    "validate_tumor_free",
    "write_records",
    "z_bin_for",
]
