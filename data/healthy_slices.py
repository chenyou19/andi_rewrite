"""Healthy-slice eligibility, selection, and z-balanced sampling.

This module deliberately contains no LMDB concerns so the selection policy can be
reused by a future storage backend without changing its deterministic behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .imaging import DEFAULT_MODALITIES, load_subject_volume, resize_slices


@dataclass(frozen=True)
class HealthySliceCandidate:
    subject_id: str
    z: int
    modality_paths: tuple[str, ...]


@dataclass(frozen=True)
class BalancedSliceRecord:
    candidate: HealthySliceCandidate
    balance_z_count: int
    sampled_index_in_z: int
    duplicate_copy_id: int
    is_extra_sample: bool


def iter_healthy_slices(
    dataset_path: Path,
    subject_ids: Iterable[str],
    image_size: int,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> Iterable[tuple[str, int, torch.Tensor]]:
    """Yield every eligible healthy slice in the historical subject order."""

    for subject_id in subject_ids:
        yield from healthy_slices_for_subject(dataset_path, subject_id, image_size, modalities)


def healthy_slices_for_subject(
    dataset_path: Path,
    subject_id: str,
    image_size: int,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> list[tuple[str, int, torch.Tensor]]:
    images, mask = load_subject_volume(dataset_path, subject_id, modalities)
    subject_slices = []
    metadata = []
    for slice_index in range(images.shape[3]):
        first_modality = images[0, :, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        has_foreground = bool(torch.count_nonzero(first_modality).item())
        has_anomaly = bool(np.any(mask_slice >= 1))
        if has_foreground and not has_anomaly:
            subject_slices.append(images[:, :, :, slice_index])
            metadata.append((subject_id, slice_index))

    if not subject_slices:
        return []

    resized = resize_slices(torch.stack(subject_slices, dim=0), image_size)
    return [
        (current_subject_id, slice_index, slice_tensor)
        for (current_subject_id, slice_index), slice_tensor in zip(metadata, resized)
    ]


def healthy_slice_candidates_for_subject(
    dataset_path: Path,
    subject_id: str,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> tuple[list[HealthySliceCandidate], set[int]]:
    images, mask = load_subject_volume(dataset_path, subject_id, modalities)
    subject_dir = dataset_path / subject_id
    modality_paths = tuple(str(subject_dir / f"{subject_id}_{modality}.nii.gz") for modality in modalities)
    z_indices_seen = set(range(images.shape[3]))
    candidates = []
    for slice_index in range(images.shape[3]):
        first_modality = images[0, :, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        has_foreground = bool(torch.count_nonzero(first_modality).item())
        has_anomaly = bool(np.any(mask_slice > 0))
        if has_foreground and not has_anomaly:
            candidates.append(
                HealthySliceCandidate(
                    subject_id=subject_id,
                    z=slice_index,
                    modality_paths=modality_paths,
                )
            )
    return candidates, z_indices_seen


def balance_candidates_by_z(
    candidates_by_z: dict[int, list[HealthySliceCandidate]],
    per_z_count: int,
    balance_seed: int,
) -> tuple[list[BalancedSliceRecord], dict[int, dict[str, int]]]:
    if per_z_count <= 0:
        raise ValueError("--per-z-count must be a positive integer.")

    rng = np.random.default_rng(balance_seed)
    records = []
    z_summary = {}
    for z in sorted(candidates_by_z):
        candidates = candidates_by_z[z]
        original_count = len(candidates)
        if original_count == 0:
            continue

        z_records = []
        if original_count > per_z_count:
            sampled_indices = rng.choice(original_count, size=per_z_count, replace=False).tolist()
            z_records.extend(
                BalancedSliceRecord(
                    candidate=candidates[index],
                    balance_z_count=original_count,
                    sampled_index_in_z=int(index),
                    duplicate_copy_id=0,
                    is_extra_sample=False,
                )
                for index in sampled_indices
            )
        else:
            full_repeats = per_z_count // original_count
            remainder = per_z_count % original_count
            for repeat_id in range(full_repeats):
                z_records.extend(
                    BalancedSliceRecord(
                        candidate=candidate,
                        balance_z_count=original_count,
                        sampled_index_in_z=index,
                        duplicate_copy_id=repeat_id,
                        is_extra_sample=False,
                    )
                    for index, candidate in enumerate(candidates)
                )
            if remainder:
                sampled_indices = rng.choice(original_count, size=remainder, replace=False).tolist()
                z_records.extend(
                    BalancedSliceRecord(
                        candidate=candidates[index],
                        balance_z_count=original_count,
                        sampled_index_in_z=int(index),
                        duplicate_copy_id=full_repeats,
                        is_extra_sample=True,
                    )
                    for index in sampled_indices
                )

        records.extend(z_records)
        z_summary[z] = {"original": original_count, "output": len(z_records)}

    return records, z_summary


__all__ = [
    "BalancedSliceRecord",
    "HealthySliceCandidate",
    "balance_candidates_by_z",
    "healthy_slice_candidates_for_subject",
    "healthy_slices_for_subject",
    "iter_healthy_slices",
]
