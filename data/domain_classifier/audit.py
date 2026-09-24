"""Fail-closed audits for domain-classifier records and paired manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .matching import SPLITS, assert_participant_split_disjoint
from .records import SliceRecord


def audit_records(
    records: Sequence[SliceRecord],
    *,
    bins: int = 20,
    cap_per_bin: int = 2,
    require_pairs: bool = False,
    require_tumor_free: bool = False,
) -> dict[str, Any]:
    """Return counts and raise on identity, geometry, or pairing violations."""

    if bins < 1 or cap_per_bin < 1:
        raise ValueError("bins and cap_per_bin must be positive")
    rows = list(records)
    record_ids = [row.record_id for row in rows]
    duplicates = sorted(record_id for record_id, count in Counter(record_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate records: {duplicates[:8]}")
    split_counts = Counter(row.split for row in rows)
    label_counts = Counter(int(row.label) for row in rows)
    domain_counts = Counter(row.domain for row in rows)
    participants = assert_participant_split_disjoint(rows) if rows else {split: 0 for split in SPLITS}
    geometry_failures: list[str] = []
    tumor_failures: list[str] = []
    for row in rows:
        if tuple(row.model_shape) != (3, 128, 128):
            geometry_failures.append(f"{row.record_id}:model_shape={row.model_shape}")
        if not row.source_dataset or not row.participant_id or not row.case_id:
            geometry_failures.append(f"{row.record_id}:missing_identity")
        if not 0.0 <= row.z_norm <= 1.0 or not 0 <= row.z_bin < bins:
            geometry_failures.append(f"{row.record_id}:invalid_z")
        if int(row.z_bin) != min(bins - 1, int(float(row.z_norm) * bins)):
            geometry_failures.append(f"{row.record_id}:z_bin_mismatch")
        if row.source_dataset.lower() in {"brats", "brats21", "brats2021"}:
            if row.native_seg_voxels is not None and int(row.native_seg_voxels) != 0:
                tumor_failures.append(f"{row.record_id}:native_seg_voxels={row.native_seg_voxels}")
            if row.model_mask_voxels is not None and int(row.model_mask_voxels) != 0:
                tumor_failures.append(f"{row.record_id}:model_mask_voxels={row.model_mask_voxels}")
    if geometry_failures:
        raise ValueError(f"record geometry/identity audit failed: {geometry_failures[:8]}")
    if tumor_failures:
        raise ValueError(f"tumor-free audit failed: {tumor_failures[:8]}")
    if require_tumor_free:
        from .readers import validate_tumor_free

        for row in rows:
            if row.source_dataset.lower() in {"brats", "brats21", "brats2021"}:
                validate_tumor_free(row)

    pair_groups: dict[str, list[SliceRecord]] = defaultdict(list)
    for row in rows:
        if row.pair_id:
            pair_groups[row.pair_id].append(row)
    pair_failures: list[str] = []
    pair_summaries: list[dict[str, Any]] = []
    participant_pairs: dict[str, set[str]] = defaultdict(set)
    for pair_id, pair_rows in sorted(pair_groups.items()):
        labels = Counter(row.label for row in pair_rows)
        by_label: dict[int, list[SliceRecord]] = defaultdict(list)
        for row in pair_rows:
            by_label[row.label].append(row)
        if set(labels) != {0, 1}:
            pair_failures.append(f"{pair_id}:labels={dict(labels)}")
            continue
        left_hist = Counter(row.z_bin for row in by_label[0])
        right_hist = Counter(row.z_bin for row in by_label[1])
        if left_hist != right_hist:
            pair_failures.append(f"{pair_id}:z_histogram_mismatch")
        if any(count > cap_per_bin for count in left_hist.values()) or any(
            count > cap_per_bin for count in right_hist.values()
        ):
            pair_failures.append(f"{pair_id}:per_bin_cap")
        if len(by_label[0]) != len(by_label[1]):
            pair_failures.append(f"{pair_id}:slice_count_mismatch")
        for label, label_rows in by_label.items():
            participant_ids = {row.participant_id for row in label_rows}
            if len(participant_ids) != 1:
                pair_failures.append(f"{pair_id}:label_{label}_participant_count={len(participant_ids)}")
            else:
                participant = next(iter(participant_ids))
                participant_pairs[participant].add(pair_id)
        pair_splits = {row.split for row in pair_rows}
        if len(pair_splits) != 1:
            pair_failures.append(f"{pair_id}:split_mismatch")
        pair_summaries.append(
            {
                "pair_id": pair_id,
                "split": next(iter(pair_splits)) if len(pair_splits) == 1 else None,
                "records": len(pair_rows),
                "slice_count_per_domain": len(by_label[0]),
                "z_histogram": {str(key): int(value) for key, value in sorted(left_hist.items())},
                "participants": sorted({row.participant_id for row in pair_rows}),
            }
        )
    if require_pairs and not pair_groups:
        pair_failures.append("no_pair_ids")
    if require_pairs:
        empty_pair_rows = [row.record_id for row in rows if not row.pair_id]
        if empty_pair_rows:
            pair_failures.append(f"rows_without_pair_id={empty_pair_rows[:8]}")
    repeated_participants = {
        participant: sorted(pair_ids)
        for participant, pair_ids in participant_pairs.items()
        if len(pair_ids) > 1
    }
    if repeated_participants:
        pair_failures.append(f"participant_in_multiple_pairs={repeated_participants}")
    if pair_failures:
        raise ValueError(f"pair audit failed: {pair_failures[:8]}")
    return {
        "status": "PASS",
        "records": len(rows),
        "split_counts": {key: int(value) for key, value in sorted(split_counts.items())},
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "domain_counts": {key: int(value) for key, value in sorted(domain_counts.items())},
        "participant_counts": participants,
        "pair_count": len(pair_groups),
        "pairs": pair_summaries,
        "exclusions": [],
        "overlaps": [],
    }


def audit_source_provenance(records: Sequence[SliceRecord]) -> dict[str, Any]:
    """Summarize the source fingerprints embedded in a manifest."""

    hashes: Counter[str] = Counter()
    paths: set[str] = set()
    for row in records:
        for key, value in row.provenance.items():
            if "sha256" in str(key).lower() and value:
                hashes[str(value)] += 1
            if str(key).endswith("path") or str(key).endswith("_path"):
                paths.add(str(value))
        for value in row.image_paths.values():
            if value:
                paths.add(str(value))
    return {
        "unique_hashes": sorted(hashes),
        "hash_reference_counts": dict(hashes),
        "unique_paths": len(paths),
    }


__all__ = ["audit_records", "audit_source_provenance"]
