"""Strict path-CSV validation and deterministic session-level splitting."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


MODALITY_ORDER = ("FLAIR", "T1", "T2")
REQUIRED_COLUMNS = (
    "case_id",
    "session",
    "t1w_path",
    "t2w_path",
    "highres_flair_path",
)
PATH_COLUMNS = {
    "T1": "t1w_path",
    "T2": "t2w_path",
    "FLAIR": "highres_flair_path",
}


class InputValidationError(ValueError):
    """Raised with every discoverable path-CSV validation failure."""


@dataclass(frozen=True)
class SessionRecord:
    csv_index: int
    case_id: str
    session: str
    t1_path: Path
    t2_path: Path
    flair_path: Path
    split: str | None = None

    @property
    def session_id(self) -> str:
        return f"{self.case_id}/{self.session}"

    def modality_path(self, modality: str) -> Path:
        return {
            "T1": self.t1_path,
            "T2": self.t2_path,
            "FLAIR": self.flair_path,
        }[modality]

    def with_split(self, split: str) -> "SessionRecord":
        return SessionRecord(**{**asdict(self), "split": split})


def _resolve_path(csv_dir: Path, value: str) -> Path:
    candidate = Path(str(value).strip())
    return candidate if candidate.is_absolute() else csv_dir / candidate


def _error(
    case_id: str,
    session: str,
    modality: str,
    path: Path | str,
    reason: str,
) -> str:
    return (
        f"case/session={case_id}/{session} modality={modality} "
        f"path={path} reason={reason}"
    )


def _validate_nifti(
    case_id: str,
    session: str,
    modality: str,
    path: Path,
    *,
    validate_voxels: bool,
) -> list[str]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment contract
        raise ImportError("LEMON preprocessing requires nibabel.") from exc

    failures: list[str] = []
    if not path.is_file():
        return [_error(case_id, session, modality, path, "path does not exist or is not a file")]
    try:
        image = nib.load(str(path))
    except Exception as exc:
        return [_error(case_id, session, modality, path, f"NIfTI unreadable: {exc}")]

    if len(image.shape) != 3:
        failures.append(_error(case_id, session, modality, path, f"expected exactly 3D, got shape={image.shape}"))
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        failures.append(_error(case_id, session, modality, path, "affine is not finite 4x4"))
    elif abs(float(np.linalg.det(affine[:3, :3]))) <= np.finfo(float).eps:
        failures.append(_error(case_id, session, modality, path, "affine spatial matrix is singular"))

    try:
        orientation = nib.orientations.io_orientation(affine)
        if orientation.shape != (3, 2) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"invalid orientation array {orientation}")
        if len(set(int(value) for value in orientation[:, 0])) != 3:
            raise ValueError(f"orientation axes are not unique: {orientation}")
        canonical = nib.as_closest_canonical(image, enforce_diag=False)
        if tuple(nib.aff2axcodes(canonical.affine)) != ("R", "A", "S"):
            raise ValueError(f"canonical axes are {nib.aff2axcodes(canonical.affine)}, expected RAS")
    except Exception as exc:
        failures.append(_error(case_id, session, modality, path, f"unsafe canonical RAS conversion: {exc}"))

    if validate_voxels and len(image.shape) == 3:
        try:
            values = image.get_fdata(dtype=np.float32, caching="unchanged")
            if not np.all(np.isfinite(values)):
                nonfinite = int(values.size - np.count_nonzero(np.isfinite(values)))
                failures.append(_error(case_id, session, modality, path, f"voxel data has {nonfinite} NaN/Inf values"))
        except Exception as exc:
            failures.append(_error(case_id, session, modality, path, f"voxel data unreadable: {exc}"))
    return failures


def validate_session_csv(
    csv_path: str | Path,
    *,
    expected_sessions: int = 115,
    validate_voxels: bool = True,
) -> list[SessionRecord]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        extra_path_columns = [column for column in columns if column.endswith("_path") and column not in REQUIRED_COLUMNS]
        if missing or extra_path_columns:
            messages = []
            if missing:
                messages.append(f"missing required columns: {missing}")
            if extra_path_columns:
                messages.append(f"unexpected modality/path columns: {extra_path_columns}")
            raise InputValidationError("CSV schema validation failed: " + "; ".join(messages))
        rows = list(reader)

    failures: list[str] = []
    if len(rows) != int(expected_sessions):
        failures.append(
            f"case/session=<dataset> modality=<all> path={csv_path} "
            f"reason=expected exactly {expected_sessions} sessions, got {len(rows)}"
        )

    csv_dir = csv_path.resolve().parent
    records: list[SessionRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        session = str(row.get("session", "")).strip()
        if not case_id or not session:
            failures.append(_error(case_id or "<missing>", session or "<missing>", "<all>", csv_path, "empty case_id/session"))
        pair = (case_id, session)
        if pair in seen_pairs:
            failures.append(_error(case_id, session, "<all>", csv_path, "duplicate case/session"))
        seen_pairs.add(pair)

        resolved = {
            modality: _resolve_path(csv_dir, str(row.get(column, "")))
            for modality, column in PATH_COLUMNS.items()
        }
        if len({str(path.resolve()) for path in resolved.values()}) != len(resolved):
            failures.append(_error(case_id, session, "<all>", csv_path, "duplicate modality paths within session"))
        for modality in MODALITY_ORDER:
            failures.extend(
                _validate_nifti(
                    case_id,
                    session,
                    modality,
                    resolved[modality],
                    validate_voxels=validate_voxels,
                )
            )
        records.append(
            SessionRecord(
                csv_index=index,
                case_id=case_id,
                session=session,
                t1_path=resolved["T1"].resolve(),
                t2_path=resolved["T2"].resolve(),
                flair_path=resolved["FLAIR"].resolve(),
            )
        )

    if failures:
        raise InputValidationError("Input dataset validation failed:\n" + "\n".join(failures))
    return records


def deterministic_session_split(
    records: Iterable[SessionRecord],
    *,
    validation_count: int = 5,
    seed: int = 73,
) -> tuple[list[SessionRecord], list[SessionRecord], dict]:
    records = list(records)
    if validation_count <= 0 or validation_count >= len(records):
        raise ValueError("validation_count must be positive and smaller than the session count.")
    sorted_ids = sorted(record.session_id for record in records)
    permutation = np.random.Generator(np.random.PCG64(int(seed))).permutation(len(sorted_ids))
    validation_ids_in_selection_order = [
        sorted_ids[int(index)] for index in permutation[:validation_count]
    ]
    validation_ids = set(validation_ids_in_selection_order)
    train = [record.with_split("train") for record in records if record.session_id not in validation_ids]
    validation = [record.with_split("val") for record in records if record.session_id in validation_ids]
    metadata = {
        "seed": int(seed),
        "algorithm": "sort session_id; numpy.random.Generator(PCG64(seed)).permutation; first validation_count",
        "numpy_version": np.__version__,
        "validation_count": int(validation_count),
        "train_session_ids": [record.session_id for record in train],
        "validation_session_ids": [record.session_id for record in validation],
        "validation_session_ids_permutation_order": validation_ids_in_selection_order,
        "permutation_first_validation_positions": [int(value) for value in permutation[:validation_count]],
    }
    return train, validation, metadata


def _record_row(record: SessionRecord) -> dict[str, object]:
    return {
        "csv_index": record.csv_index,
        "case_id": record.case_id,
        "session": record.session,
        "session_id": record.session_id,
        "split": record.split,
        "channel_0": "FLAIR",
        "channel_1": "T1",
        "channel_2": "T2",
        "flair_path": str(record.flair_path),
        "t1_path": str(record.t1_path),
        "t2_path": str(record.t2_path),
    }


def write_manifest_products(
    output_root: str | Path,
    csv_path: str | Path,
    records: Iterable[SessionRecord],
    split_metadata: dict,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = list(records)
    fieldnames = list(_record_row(records[0]).keys()) if records else []
    for name, subset in (
        ("manifest.csv", records),
        ("train_split.csv", [record for record in records if record.split == "train"]),
        ("validation_split.csv", [record for record in records if record.split == "val"]),
    ):
        with (output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_record_row(record) for record in subset)

    source_csv = Path(csv_path).resolve()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_csv": str(source_csv),
        "input_csv_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "expected_channel_order": list(MODALITY_ORDER),
        "session_count": len(records),
        "split": split_metadata,
        "sessions": [_record_row(record) for record in records],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
