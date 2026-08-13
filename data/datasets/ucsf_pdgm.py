"""UCSF-PDGM dataset adapter.

This module owns UCSF discovery, geometry validation, interpolation, label, and
metadata behaviour; it is deliberately not shared with other MRI adapters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import _shape_text
from .imaging import histogram_normalize_volume, normalize_volume


@dataclass(frozen=True)
class UCSFPDGMSubject:
    subject_id: str
    folder: Path
    reference_path: Path
    modality_paths: dict[str, Path]
    segmentation_path: Path


def _ucsf_pdgm_file_stem(folder_name: str) -> str:
    """Return the case/timepoint identifier without damaging follow-up IDs."""

    suffix = "_nifti"
    if not folder_name.endswith(suffix):
        raise ValueError(f"UCSF-PDGM case folder must end with {suffix!r}: {folder_name!r}")
    file_stem = folder_name[: -len(suffix)]
    if not file_stem:
        raise ValueError(f"UCSF-PDGM case folder has an empty identifier: {folder_name!r}")
    return file_stem


def _nifti_orientation_code(image: Any) -> str:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("NIfTI orientation validation requires the optional 'nibabel' package.") from exc
    return "".join(str(code) for code in nib.aff2axcodes(image.affine))


def _validate_nifti_geometry(
    reference_image: Any,
    images: dict[str, Any],
    expected_orientation: str | None,
    *,
    affine_tolerance: float = 1.0e-4,
) -> str:
    """Validate identical voxel grids and the verified BraTS model orientation."""

    if len(reference_image.shape) != 3:
        raise ValueError(f"Reference NIfTI must be 3-D, got shape {reference_image.shape}.")
    reference_orientation = _nifti_orientation_code(reference_image)
    expected = str(expected_orientation).strip().upper() if expected_orientation else None
    if expected and reference_orientation != expected:
        raise ValueError(
            "NIfTI orientation does not match the verified BraTS model orientation: "
            f"expected={expected}, actual={reference_orientation}."
        )
    reference_spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
    for name, image in images.items():
        orientation = _nifti_orientation_code(image)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if len(image.shape) != 3:
            raise ValueError(f"{name} NIfTI must be 3-D, got shape {image.shape}.")
        if tuple(image.shape) != tuple(reference_image.shape):
            raise ValueError(
                f"{name} shape {image.shape} does not match reference shape {reference_image.shape}."
            )
        if not np.allclose(spacing, reference_spacing, atol=affine_tolerance, rtol=0.0):
            raise ValueError(
                f"{name} spacing {spacing} does not match reference spacing {reference_spacing}."
            )
        if orientation != reference_orientation:
            raise ValueError(
                f"{name} orientation {orientation} does not match reference orientation "
                f"{reference_orientation}."
            )
        if not np.allclose(image.affine, reference_image.affine, atol=affine_tolerance, rtol=0.0):
            raise ValueError(f"{name} affine does not match the reference affine.")
    return reference_orientation


class UCSFPDGMVolumeDataset(Dataset):
    """Read complete UCSF-PDGM case/timepoint folders for BraTS-trained ANDi.

    Discovery is recursive because the local archive contains both direct case
    folders and a nested package tree. Only folders with all four core MRI
    modalities and the tumor segmentation are retained. The verified BraTS and
    UCSF arrays both use LPS voxel-axis semantics, so this adapter preserves the
    native layout and fails clearly if a future case violates that contract.
    """

    DEFAULT_MAPPING = {
        "flair": "FLAIR",
        "t1": "T1",
        "t1ce": "T1c",
        "t2": "T2",
    }

    def __init__(
        self,
        dataset_path: str | Path,
        image_size: int = 128,
        modalities: list[str] | None = None,
        modality_mapping: dict[str, str] | None = None,
        segmentation_suffix: str = "tumor_segmentation",
        reference_modality: str = "flair",
        model_orientation: str | None = "LPS",
        histogram_normalization: bool = False,
        return_metadata: bool = True,
        subject_limit: int | None = None,
        duplicate_policy: str = "error",
        csv_path: str | Path | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            raise FileNotFoundError(f"UCSF-PDGM dataset path does not exist: {self.dataset_path}")
        self.csv_path = Path(csv_path) if csv_path not in (None, "") else None
        self.requested_subject_ids = self._load_requested_subject_ids()
        self.image_size = int(image_size)
        self.modalities = list(modalities or ["flair", "t1", "t1ce", "t2"])
        mapping = dict(self.DEFAULT_MAPPING)
        mapping.update({str(key): str(value) for key, value in (modality_mapping or {}).items()})
        missing_mapping = [modality for modality in self.modalities if modality not in mapping]
        if missing_mapping:
            raise ValueError(f"Missing UCSF-PDGM modality mapping for: {missing_mapping}")
        self.modality_mapping = mapping
        self.segmentation_suffix = str(segmentation_suffix)
        self.reference_modality = str(reference_modality)
        if self.reference_modality not in self.modalities:
            raise ValueError(
                f"reference_modality={self.reference_modality!r} must appear in modalities={self.modalities}."
            )
        self.model_orientation = (
            str(model_orientation).strip().upper() if model_orientation not in (None, "") else None
        )
        self.histogram_normalization = bool(histogram_normalization)
        self.return_metadata = bool(return_metadata)
        self.subject_limit = int(subject_limit) if subject_limit not in (None, "", 0, False) else None
        self.duplicate_policy = str(duplicate_policy).strip().lower()
        if self.duplicate_policy not in {"error", "first"}:
            raise ValueError("duplicate_policy must be 'error' or 'first'.")
        self.candidate_folders: list[Path] = []
        self.incomplete_cases: list[dict[str, Any]] = []
        self.duplicate_subjects: dict[str, list[Path]] = {}
        self.subjects = self._discover_subjects()
        if not self.subjects:
            raise FileNotFoundError(
                f"No complete UCSF-PDGM case/timepoint folders found under {self.dataset_path}."
            )

    def __len__(self) -> int:
        return len(self.subjects)

    def _load_requested_subject_ids(self) -> list[str] | None:
        if self.csv_path is None:
            return None
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"UCSF-PDGM subject CSV does not exist: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        if "subject_id" not in frame.columns:
            raise ValueError(
                f"UCSF-PDGM subject CSV must contain a 'subject_id' column: {self.csv_path}"
            )
        invalid_rows: list[int] = []
        subject_ids: list[str] = []
        for row_number, value in enumerate(frame["subject_id"].tolist(), start=2):
            if pd.isna(value) or not str(value).strip():
                invalid_rows.append(row_number)
                continue
            subject_ids.append(str(value).strip())
        if invalid_rows:
            rows = ", ".join(str(row) for row in invalid_rows)
            raise ValueError(
                "UCSF-PDGM subject CSV contains blank subject_id values at "
                f"row(s): {rows}."
            )
        if not subject_ids:
            raise ValueError(f"UCSF-PDGM subject CSV contains no subject IDs: {self.csv_path}")

        seen: set[str] = set()
        duplicates: list[str] = []
        for subject_id in subject_ids:
            if subject_id in seen and subject_id not in duplicates:
                duplicates.append(subject_id)
            seen.add(subject_id)
        if duplicates:
            raise ValueError(
                "UCSF-PDGM subject CSV contains duplicate subject_id values: "
                + ", ".join(duplicates)
            )
        return subject_ids

    def _case_paths(self, folder: Path, subject_id: str) -> tuple[dict[str, Path], Path]:
        modality_paths = {
            modality: folder / f"{subject_id}_{self.modality_mapping[modality]}.nii.gz"
            for modality in self.modalities
        }
        segmentation_path = folder / f"{subject_id}_{self.segmentation_suffix}.nii.gz"
        return modality_paths, segmentation_path

    def _discover_subjects(self) -> list[UCSFPDGMSubject]:
        candidates = sorted(
            (path for path in self.dataset_path.rglob("*_nifti") if path.is_dir()),
            key=lambda path: (path.name.lower(), str(path).lower()),
        )
        self.candidate_folders = candidates
        complete_by_id: dict[str, list[UCSFPDGMSubject]] = {}
        for folder in candidates:
            subject_id = _ucsf_pdgm_file_stem(folder.name)
            modality_paths, segmentation_path = self._case_paths(folder, subject_id)
            required_paths = [*modality_paths.values(), segmentation_path]
            missing = [path.name for path in required_paths if not path.is_file()]
            if missing:
                self.incomplete_cases.append(
                    {"subject_id": subject_id, "folder": str(folder), "missing": missing}
                )
                continue
            subject = UCSFPDGMSubject(
                subject_id=subject_id,
                folder=folder,
                reference_path=modality_paths[self.reference_modality],
                modality_paths=modality_paths,
                segmentation_path=segmentation_path,
            )
            complete_by_id.setdefault(subject_id, []).append(subject)

        self.duplicate_subjects = {
            subject_id: [subject.folder for subject in subjects]
            for subject_id, subjects in complete_by_id.items()
            if len(subjects) > 1
        }
        if self.duplicate_subjects and self.duplicate_policy == "error":
            details = "; ".join(
                f"{subject_id}: {', '.join(str(path) for path in paths)}"
                for subject_id, paths in sorted(self.duplicate_subjects.items())
            )
            raise ValueError(f"Duplicate complete UCSF-PDGM case/timepoint IDs found: {details}")

        subjects_by_id: dict[str, UCSFPDGMSubject] = {}
        for subject_id in sorted(complete_by_id):
            matches = sorted(complete_by_id[subject_id], key=lambda subject: str(subject.folder).lower())
            if len(matches) > 1:
                warnings.warn(
                    f"Duplicate UCSF-PDGM ID {subject_id!r}; selecting {matches[0].folder} "
                    "by deterministic path order because duplicate_policy='first'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            subjects_by_id[subject_id] = matches[0]

        all_subjects = list(subjects_by_id.values())
        if self.requested_subject_ids is None:
            subjects = all_subjects
        else:
            missing_requested = [
                subject_id
                for subject_id in self.requested_subject_ids
                if subject_id not in subjects_by_id
            ]
            if missing_requested:
                raise FileNotFoundError(
                    "UCSF-PDGM subject CSV requested IDs that were not found as complete cases "
                    f"under {self.dataset_path}: {', '.join(missing_requested)}"
                )
            subjects = [subjects_by_id[subject_id] for subject_id in self.requested_subject_ids]

        complete_folders = sum(len(matches) for matches in complete_by_id.values())
        follow_up_cases = sum("_FU" in subject.subject_id.upper() for subject in all_subjects)
        self.discovery_summary = {
            "candidate_folders": len(candidates),
            "complete_folders": complete_folders,
            "complete_cases": len(all_subjects),
            "incomplete_cases": len(self.incomplete_cases),
            "duplicate_ids": len(self.duplicate_subjects),
            "follow_up_cases": int(follow_up_cases),
            "requested_cases": len(subjects),
            "selection_source": "csv" if self.requested_subject_ids is not None else "discovery",
        }
        if self.subject_limit is not None:
            subjects = subjects[: self.subject_limit]
        self.discovery_summary["selected_cases"] = len(subjects)
        return subjects

    @staticmethod
    def _load_nib(path: Path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("UCSFPDGMVolumeDataset requires the optional 'nibabel' package.") from exc
        return nib.load(str(path))

    def _load_case_arrays(
        self,
        subject: UCSFPDGMSubject,
    ) -> tuple[np.ndarray, np.ndarray, Any, str]:
        modality_images = {
            modality: self._load_nib(subject.modality_paths[modality])
            for modality in self.modalities
        }
        segmentation_image = self._load_nib(subject.segmentation_path)
        reference_image = modality_images[self.reference_modality]
        orientation = _validate_nifti_geometry(
            reference_image,
            {**modality_images, "segmentation": segmentation_image},
            self.model_orientation,
        )

        image_array = np.empty(
            (len(self.modalities), *reference_image.shape),
            dtype=np.float32,
        )
        for channel, modality in enumerate(self.modalities):
            image_array[channel] = modality_images[modality].get_fdata(
                dtype=np.float32,
                caching="unchanged",
            )
        segmentation = segmentation_image.get_fdata(
            dtype=np.float32,
            caching="unchanged",
        )
        return image_array, segmentation, reference_image, orientation

    def _resize_image_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if tuple(volume.shape[1:3]) == (self.image_size, self.image_size):
            return volume.contiguous()
        slices = volume.permute(3, 0, 1, 2).contiguous()
        resized = F.interpolate(
            slices,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.permute(1, 2, 3, 0).contiguous()

    def _resize_mask_volume(self, mask: torch.Tensor) -> torch.Tensor:
        if tuple(mask.shape[:2]) == (self.image_size, self.image_size):
            return mask.contiguous()
        slices = mask.permute(2, 0, 1).unsqueeze(1).float().contiguous()
        resized = F.interpolate(
            slices,
            size=(self.image_size, self.image_size),
            mode="nearest-exact",
        )
        return resized[:, 0].permute(1, 2, 0).bool().contiguous()

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        subject = self.subjects[idx]
        image_array, segmentation, reference_image, orientation = self._load_case_arrays(subject)
        if not np.isfinite(image_array).all():
            raise ValueError(f"Non-finite MRI intensity found for UCSF-PDGM subject {subject.subject_id}.")
        if not np.isfinite(segmentation).all():
            raise ValueError(f"Non-finite segmentation found for UCSF-PDGM subject {subject.subject_id}.")

        if self.histogram_normalization:
            volume = histogram_normalize_volume(image_array)
        else:
            volume = normalize_volume(torch.from_numpy(image_array).float())
        whole_tumor = torch.from_numpy((segmentation > 0).astype(np.uint8)).bool()
        resized = self._resize_image_volume(volume)
        resized_mask = self._resize_mask_volume(whole_tumor)

        if not self.return_metadata:
            return resized, resized_mask
        metadata = {
            "subject_id": subject.subject_id,
            "case_folder": str(subject.folder),
            "has_label": True,
            "reference_modality": self.reference_modality,
            "reference_path": str(subject.reference_path),
            "native_shape": _shape_text(tuple(reference_image.shape)),
            "model_shape": _shape_text(tuple(resized_mask.shape)),
            "native_spacing": _shape_text(tuple(reference_image.header.get_zooms()[:3])),
            "native_orientation": orientation,
            "model_orientation": self.model_orientation or orientation,
            "orientation_transform": "identity",
            "modality_mapping": {
                modality: self.modality_mapping[modality] for modality in self.modalities
            },
            "input_paths": {
                modality: str(subject.modality_paths[modality]) for modality in self.modalities
            },
            "segmentation_path": str(subject.segmentation_path),
        }
        return resized, resized_mask, metadata


def build_ucsf_pdgm_dataset(config: dict[str, Any]) -> UCSFPDGMVolumeDataset:
    image_size = int(config.get("image_size", 128))
    return UCSFPDGMVolumeDataset(
        dataset_path=config["dataset_path"],
        image_size=image_size,
        modalities=config.get("modalities"),
        modality_mapping=config.get("modality_mapping"),
        segmentation_suffix=str(config.get("segmentation_suffix", "tumor_segmentation")),
        reference_modality=str(config.get("reference_modality", "flair")),
        model_orientation=config.get("model_orientation", config.get("expected_orientation", "LPS")),
        histogram_normalization=bool(config.get("histogram_normalization", False)),
        return_metadata=bool(config.get("return_metadata", True)),
        subject_limit=config.get("subject_limit"),
        duplicate_policy=str(config.get("duplicate_policy", "error")),
        csv_path=config.get("path_to_csv"),
    )
