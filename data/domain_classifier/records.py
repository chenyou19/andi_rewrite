"""Canonical records used by the domain-classifier data preparation stage.

The records deliberately keep source metadata beside, rather than inside, the
model input.  A classifier can therefore be given only the array returned by
``load_slice`` while split, matching, and provenance audits still have a
stable representation to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

MODALITIES = ("flair", "t1", "t2")
MODEL_SHAPE = (3, 128, 128)
DEFAULT_Z_BINS = 20


def namespaced_participant(source_dataset: str, participant_id: str) -> str:
    """Return the collision-proof identity used for global split maps."""

    dataset = str(source_dataset).strip().lower().replace("/", "_").replace("\\", "_")
    participant = str(participant_id).strip()
    if not dataset or not participant:
        raise ValueError("source_dataset and participant_id are required")
    return f"{dataset}:{participant}"


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _canonical_paths(value: Mapping[str, Any] | None) -> dict[str, str]:
    """Return paths with lower-case logical modality names.

    Existing manifests use both ``FLAIR``/``T1``/``T2`` and lower-case keys.
    Accepting both here prevents an accidental channel reordering in readers.
    """

    if not value:
        return {}
    result: dict[str, str] = {}
    aliases = {
        "flair": "flair",
        "t1": "t1",
        "t2": "t2",
        "FLAIR": "flair",
        "T1": "t1",
        "T2": "t2",
    }
    for key, path in value.items():
        logical = aliases.get(str(key), str(key).lower())
        result[logical] = _as_text(path)
    return result


def _normalise_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if not float(value).is_integer() or int(value) not in (0, 1):
            raise ValueError(f"domain label must be 0 or 1, found {value!r}")
        return int(value)
    text = _as_text(value).strip().lower()
    if text in {"0", "healthy", "control", "source", "a"}:
        return 0
    if text in {"1", "brats", "brats21", "target", "lesion", "b"}:
        return 1
    raise ValueError(f"domain label must be 0/1 or a known name, found {value!r}")


def z_bin_for(z_norm: float, bins: int = DEFAULT_Z_BINS) -> int:
    if bins < 1:
        raise ValueError("bins must be positive")
    value = float(z_norm)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"z_norm must lie in [0, 1], found {value}")
    return min(bins - 1, int(value * bins))


@dataclass(frozen=True)
class SliceRecord:
    """One model-input slice and its auditable source identity."""

    split: str
    label: int
    domain: str
    participant_id: str
    session_id: str
    case_id: str
    z: int
    z_norm: float
    z_bin: int
    source_dataset: str
    source_key: str = ""
    source_split: str = ""
    image_paths: dict[str, str] = field(default_factory=dict)
    seg_path: str = ""
    model_mask_path: str = ""
    registered_paths: dict[str, str] = field(default_factory=dict)
    geometry_shape: tuple[int, ...] = ()
    model_shape: tuple[int, ...] = MODEL_SHAPE
    pair_id: str = ""
    stage: str = "final"
    native_seg_voxels: int | None = None
    model_mask_voxels: int | None = None
    foreground_fraction: tuple[float, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        if self.source_key:
            return f"{self.source_dataset}:{self.source_split}:{self.source_key}"
        return f"{self.source_dataset}:{self.case_id}:{self.z}"

    @property
    def modalities(self) -> tuple[str, str, str]:
        return MODALITIES

    def with_updates(self, **updates: Any) -> "SliceRecord":
        values = self.to_dict()
        values.update(updates)
        return SliceRecord.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to one JSONL-compatible mapping."""

        payload: dict[str, Any] = {
            "split": self.split,
            "label": int(self.label),
            "domain": self.domain,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "case_id": self.case_id,
            "z": int(self.z),
            "z_norm": float(self.z_norm),
            "z_bin": int(self.z_bin),
            "source_dataset": self.source_dataset,
            "source_key": self.source_key,
            "source_split": self.source_split,
            "image_paths": dict(self.image_paths),
            "seg_path": self.seg_path,
            "model_mask_path": self.model_mask_path,
            "registered_paths": dict(self.registered_paths),
            "geometry_shape": list(self.geometry_shape),
            "model_shape": list(self.model_shape),
            "pair_id": self.pair_id,
            "stage": self.stage,
            "native_seg_voxels": self.native_seg_voxels,
            "model_mask_voxels": self.model_mask_voxels,
            "foreground_fraction": list(self.foreground_fraction),
            "provenance": dict(self.provenance),
        }
        # Keep additional source metadata available for diagnostics without
        # flattening it into fields used by the reader.
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SliceRecord":
        paths_value = row.get("image_paths", row.get("modality_paths", {}))
        registered_value = row.get("registered_paths", row.get("raw_registered_paths", {}))
        paths = _canonical_paths(paths_value if isinstance(paths_value, Mapping) else {})
        registered = _canonical_paths(
            registered_value if isinstance(registered_value, Mapping) else {}
        )
        geometry = row.get("geometry_shape", row.get("native_shape", ()))
        model = row.get("model_shape", MODEL_SHAPE)
        geometry_shape = tuple(int(v) for v in geometry) if geometry else ()
        model_shape = tuple(int(v) for v in model) if model else MODEL_SHAPE
        z = int(row.get("z", row.get("slice", 0)))
        if "z_norm" in row and row.get("z_norm") is not None:
            z_norm = float(row["z_norm"])
        elif len(geometry_shape) >= 3:
            z_norm = float(z) / max(1, geometry_shape[-1] - 1)
        else:
            # A manifest without native geometry is still useful for tests and
            # synthetic arrays.  Its z coordinate is interpreted as already
            # normalized only when it is in [0, 1].
            if z not in (0, 1):
                raise ValueError(
                    "Manifest rows without geometry_shape must provide explicit z_norm "
                    f"for z={z}."
                )
            z_norm = float(z)
        if not 0.0 <= z_norm <= 1.0:
            raise ValueError(f"z_norm must lie in [0, 1], found {z_norm}")
        bins = int(row.get("z_bins", DEFAULT_Z_BINS))
        expected_bin = z_bin_for(z_norm, bins)
        z_bin = int(row.get("z_bin", expected_bin))
        if z_bin != expected_bin:
            raise ValueError(
                f"z_bin={z_bin} is inconsistent with z_norm={z_norm} and bins={bins}"
            )
        label = _normalise_label(row.get("label", 0))
        domain_value = row.get("domain")
        domain = _as_text(domain_value) if domain_value is not None else ("healthy" if label == 0 else "brats")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        # Preserve fields added by a producer that are not part of the core
        # schema.  This is useful for audit counts and cache identities.
        known = {
            "split", "label", "domain", "participant_id", "session_id", "case_id",
            "z", "slice", "z_norm", "z_bin", "z_bins", "source_dataset",
            "source_key", "source_split", "image_paths", "modality_paths", "seg_path",
            "segmentation_path", "model_mask_path", "registered_paths",
            "raw_registered_paths", "geometry_shape", "native_shape", "model_shape",
            "pair_id", "stage", "native_seg_voxels", "model_mask_voxels",
            "foreground_fraction", "provenance", "metadata",
        }
        extras = dict(metadata)
        extras.update({str(k): v for k, v in row.items() if k not in known})
        foreground = row.get("foreground_fraction", ())
        return cls(
            split=_as_text(row.get("split", "")),
            label=label,
            domain=domain,
            participant_id=_as_text(row.get("participant_id", row.get("subject_id", ""))),
            session_id=_as_text(row.get("session_id", "")),
            case_id=_as_text(row.get("case_id", row.get("subject_id", ""))),
            z=z,
            z_norm=z_norm,
            z_bin=z_bin,
            source_dataset=_as_text(row.get("source_dataset", row.get("dataset", ""))),
            source_key=_as_text(row.get("source_key", row.get("key", ""))),
            source_split=_as_text(row.get("source_split", row.get("split", ""))),
            image_paths=paths,
            seg_path=_as_text(row.get("seg_path", row.get("segmentation_path", ""))),
            model_mask_path=_as_text(row.get("model_mask_path", "")),
            registered_paths=registered,
            geometry_shape=geometry_shape,
            model_shape=model_shape,
            pair_id=_as_text(row.get("pair_id", "")),
            stage=_as_text(row.get("stage", "final")) or "final",
            native_seg_voxels=(
                None if row.get("native_seg_voxels") is None else int(row["native_seg_voxels"])
            ),
            model_mask_voxels=(
                None if row.get("model_mask_voxels") is None else int(row["model_mask_voxels"])
            ),
            foreground_fraction=tuple(float(v) for v in foreground) if foreground else (),
            provenance=dict(row.get("provenance", {}) or {}),
            metadata=extras,
        )


def record_from_paths(
    *,
    split: str,
    label: int,
    domain: str,
    participant_id: str,
    session_id: str,
    case_id: str,
    z: int,
    geometry_shape: tuple[int, ...],
    source_dataset: str,
    image_paths: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SliceRecord:
    """Convenience constructor that computes normalized z and its bin."""

    z_norm = float(z) / max(1, int(geometry_shape[-1]) - 1)
    return SliceRecord(
        split=split,
        label=_normalise_label(label),
        domain=domain,
        participant_id=participant_id,
        session_id=session_id,
        case_id=case_id,
        z=int(z),
        z_norm=z_norm,
        z_bin=z_bin_for(z_norm),
        source_dataset=source_dataset,
        image_paths=_canonical_paths(image_paths),
        geometry_shape=tuple(int(v) for v in geometry_shape),
        **kwargs,
    )


__all__ = [
    "DEFAULT_Z_BINS",
    "MODALITIES",
    "MODEL_SHAPE",
    "SliceRecord",
    "namespaced_participant",
    "record_from_paths",
    "z_bin_for",
]
