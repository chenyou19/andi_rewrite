"""Participant split and deterministic, z-balanced pair construction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .records import DEFAULT_Z_BINS, SliceRecord

SPLITS = ("train", "val", "test")


def participant_split_map(
    participant_ids: Iterable[str],
    *,
    seed: int = 73,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Assign participants to train/val/test with deterministic seed 73."""

    if len(fractions) != 3 or any(float(value) < 0 for value in fractions):
        raise ValueError("fractions must contain three non-negative values")
    total = float(sum(fractions))
    if not np.isclose(total, 1.0):
        raise ValueError(f"fractions must sum to one, found {total}")
    ids = sorted({str(value) for value in participant_ids})
    if not ids:
        return {}
    order = [ids[int(i)] for i in np.random.default_rng(int(seed)).permutation(len(ids))]
    n = len(order)
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    # Give leftover participants to the largest fractional remainders.  For
    # the common n>=3 case, ensure every requested split is represented.
    remainder = n - int(counts.sum())
    for index in np.argsort(-(raw - counts), kind="stable")[:remainder]:
        counts[int(index)] += 1
    if n >= 3:
        for index, fraction in enumerate(fractions):
            if fraction > 0 and counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[index] += 1
    mapping: dict[str, str] = {}
    cursor = 0
    for split, count in zip(SPLITS, counts):
        for participant in order[cursor : cursor + int(count)]:
            mapping[participant] = split
        cursor += int(count)
    if len(mapping) != len(ids):
        raise RuntimeError("participant split did not cover every participant")
    return mapping


def assert_participant_split_disjoint(records: Sequence[SliceRecord]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        if record.split not in by_split:
            raise ValueError(f"unknown split {record.split!r}")
        if not record.participant_id:
            raise ValueError(f"record has empty participant_id: {record.record_id}")
        by_split[record.split].add(record.participant_id)
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = by_split[left].intersection(by_split[right])
            if overlap:
                raise ValueError(f"participant overlap between {left} and {right}: {sorted(overlap)[:8]}")
    return {split: len(values) for split, values in by_split.items()}


def apply_participant_splits(
    records: Sequence[SliceRecord],
    *,
    seed: int = 73,
    split_map: Mapping[str, str] | None = None,
) -> tuple[list[SliceRecord], dict[str, str]]:
    """Apply one participant map to every session and slice."""

    mapping = dict(split_map) if split_map is not None else participant_split_map(
        (record.participant_id for record in records), seed=seed
    )
    missing = sorted({record.participant_id for record in records}.difference(mapping))
    if missing:
        raise ValueError(f"split map is missing participants: {missing[:8]}")
    updated = [record.with_updates(split=mapping[record.participant_id]) for record in records]
    assert_participant_split_disjoint(updated)
    return updated, mapping


def _primary_case(records: Sequence[SliceRecord]) -> tuple[str, str]:
    """Select one case per participant; all sessions remain audited in split."""

    cases = sorted({(record.case_id, record.session_id) for record in records})
    if not cases:
        raise ValueError("participant has no cases")
    return cases[0]


def _stable_pair_id(comparison: str, split: str, healthy: str, brats: str) -> str:
    text = f"{comparison}|{split}|{healthy}|{brats}".encode("utf-8")
    return f"{comparison}:{split}:{hashlib.sha256(text).hexdigest()[:12]}"


@dataclass(frozen=True)
class PairingResult:
    records: tuple[SliceRecord, ...]
    pairs: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "records": len(self.records),
            "pairs": len(self.pairs),
            "exclusions": len(self.exclusions),
        }


def _primary_by_participant(records: Sequence[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.participant_id].append(record)
    result: dict[str, list[SliceRecord]] = {}
    for participant, rows in grouped.items():
        case_id, session_id = _primary_case(rows)
        result[participant] = [
            row for row in rows if row.case_id == case_id and row.session_id == session_id
        ]
    return result


def _pair_participants(
    healthy_ids: Sequence[str],
    brats_ids: Sequence[str],
    *,
    seed: int,
    split: str,
) -> list[tuple[str, str]]:
    left = sorted(healthy_ids)
    right = sorted(brats_ids)
    split_offset = {name: index * 1009 for index, name in enumerate(SPLITS)}[split]
    left_order = [
        left[int(i)]
        for i in np.random.default_rng(seed + split_offset).permutation(len(left))
    ]
    right_order = [
        right[int(i)]
        for i in np.random.default_rng(seed + 17 + split_offset).permutation(len(right))
    ]
    return list(zip(left_order[: min(len(left_order), len(right_order))], right_order[: min(len(left_order), len(right_order))]))


def build_pairs(
    healthy_records: Sequence[SliceRecord],
    brats_records: Sequence[SliceRecord],
    *,
    comparison: str = "healthy_vs_brats",
    seed: int = 73,
    bins: int = DEFAULT_Z_BINS,
    cap_per_bin: int = 2,
) -> PairingResult:
    """Build fixed participant pairs with identical per-bin z histograms.

    Participant pairing is seeded and independent of image values.  Slices are
    then selected only from the primary case and only by normalized-z bin.
    """

    if bins < 1:
        raise ValueError("bins must be positive")
    if cap_per_bin not in (1, 2):
        raise ValueError("cap_per_bin must be 1 or 2")
    all_records = list(healthy_records) + list(brats_records)
    assert_participant_split_disjoint(all_records)
    by_split: dict[str, tuple[list[SliceRecord], list[SliceRecord]]] = {}
    for split in SPLITS:
        by_split[split] = (
            [record for record in healthy_records if record.split == split],
            [record for record in brats_records if record.split == split],
        )

    output: list[SliceRecord] = []
    summaries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for split in SPLITS:
        healthy, brats = by_split[split]
        h_primary = _primary_by_participant(healthy)
        b_primary = _primary_by_participant(brats)
        for h_id, b_id in _pair_participants(
            list(h_primary), list(b_primary), seed=seed, split=split
        ):
            h_rows = h_primary[h_id]
            b_rows = b_primary[b_id]
            h_by_bin: dict[int, list[SliceRecord]] = defaultdict(list)
            b_by_bin: dict[int, list[SliceRecord]] = defaultdict(list)
            for row in h_rows:
                h_by_bin[int(row.z_bin)].append(row)
            for row in b_rows:
                b_by_bin[int(row.z_bin)].append(row)
            selected_h: list[SliceRecord] = []
            selected_b: list[SliceRecord] = []
            histogram: dict[str, int] = {}
            for z_bin in range(bins):
                h_candidates = sorted(h_by_bin.get(z_bin, []), key=lambda row: (row.z, row.record_id))
                b_candidates = sorted(b_by_bin.get(z_bin, []), key=lambda row: (row.z, row.record_id))
                count = min(cap_per_bin, len(h_candidates), len(b_candidates))
                histogram[str(z_bin)] = int(count)
                selected_h.extend(h_candidates[:count])
                selected_b.extend(b_candidates[:count])
            if not selected_h:
                exclusions.append(
                    {
                        "split": split,
                        "healthy_participant": h_id,
                        "brats_participant": b_id,
                        "reason": "no_common_z_bin",
                    }
                )
                continue
            pair_id = _stable_pair_id(comparison, split, h_id, b_id)
            output.extend(row.with_updates(pair_id=pair_id) for row in selected_h)
            output.extend(row.with_updates(pair_id=pair_id) for row in selected_b)
            summaries.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "healthy_participant": h_id,
                    "brats_participant": b_id,
                    "healthy_case": selected_h[0].case_id,
                    "brats_case": selected_b[0].case_id,
                    "slice_count_per_domain": len(selected_h),
                    "z_histogram": histogram,
                    "cap_per_bin": cap_per_bin,
                }
            )
    output.sort(key=lambda row: (SPLITS.index(row.split), row.pair_id, row.label, row.z_bin, row.z, row.record_id))
    return PairingResult(tuple(output), tuple(summaries), tuple(exclusions))


__all__ = [
    "PairingResult",
    "SPLITS",
    "apply_participant_splits",
    "assert_participant_split_disjoint",
    "build_pairs",
    "participant_split_map",
]
