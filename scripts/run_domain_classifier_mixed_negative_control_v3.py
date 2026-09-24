"""Run the source-stratified Mixed same-cohort negative control.

The ordinary negative-control helper in ``run_domain_classifier_controls.py``
draws one global participant split.  That is appropriate for a standalone
cohort, but it changes the source composition of the Mixed cohort.  This
entry point keeps the Mixed control conditional on the source cohorts: every
underlying source receives its own fixed 40/10/50 participant split and one
fixed seed-73 balanced pseudo-label draw shared by the three initialization
fits.  Pairs are then matched on the existing 20-bin normalized-z coordinate
within source and target split.

The command is deliberately separate from the active controls runner.  It
uses the runner's immutable tensor materializer and fit/persistence helpers,
but never changes source fields, source split provenance, or the formal
70/15/15 manifests.  ``--dry-run`` builds and audits the planned mappings
without reading image tensors or starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# This helper is always a standalone subprocess.  Pin the BLAS/OpenMP policy
# before importing NumPy/PyTorch so the persisted runtime describes the child
# that actually performed the fit, rather than a separate probe process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

ANDI_PYTHON = Path(r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe")
SPLITS = ("train", "val", "test")
SOURCES = ("fomo45k", "mpi", "oasis3")
FRACTIONS = (0.40, 0.10, 0.50)
JOINT_MODALITIES = ("flair", "t1", "t2")
FAMILY_TO_MODEL = {"small_cnn": "small_cnn", "resnet18_scratch_gn": "resnet18"}


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if not callable(converter):
        raise TypeError(
            "Mixed negative records must be mappings or expose to_dict()."
        )
    return dict(converter())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _atomic_write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite immutable artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_or_verify_immutable_json(path: Path, value: Any) -> None:
    """Create an input-binding artifact once, or require byte-equivalent data."""

    normalized = _json_safe(value)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read existing immutable artifact: {path}") from exc
        if json.dumps(existing, sort_keys=True, separators=(",", ":")) != json.dumps(
            normalized, sort_keys=True, separators=(",", ":")
        ):
            raise RuntimeError(f"Existing immutable artifact differs: {path}")
        return
    _atomic_write_json(path, normalized)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def source_name_for_row(row: Mapping[str, Any]) -> str:
    """Resolve a Mixed row to its immutable standalone source namespace."""

    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    provenance = row.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source = _text(
        metadata.get("underlying_source_dataset")
        or provenance.get("underlying_source_dataset")
    ).lower()
    if not source:
        participant = _text(row.get("participant_id"))
        prefix = participant.split(":", 1)[0].lower() if ":" in participant else ""
        source = prefix
    aliases = {"fomo": "fomo45k", "fomo45k": "fomo45k", "brats": "brats21"}
    source = aliases.get(source, source)
    if source not in SOURCES:
        raise ValueError(
            "Mixed row is missing a recognized underlying source cohort: "
            f"participant={row.get('participant_id')!r}, source={source!r}."
        )
    participant = _text(row.get("participant_id"))
    if not participant:
        raise ValueError("Mixed negative rows require a non-empty participant_id.")
    prefix = participant.split(":", 1)[0].lower() if ":" in participant else ""
    if prefix in SOURCES and prefix != source:
        raise ValueError(
            f"Mixed participant namespace {prefix!r} disagrees with metadata source {source!r}."
        )
    return source


def _z_bin_for_row(row: Mapping[str, Any]) -> int:
    value = row.get("z_bin")
    if value is not None:
        integer = int(value)
        if integer < 0 or integer >= 20:
            raise ValueError(f"z_bin must lie in [0, 19], found {integer}.")
        return integer
    z_norm = row.get("z_norm")
    if z_norm is not None:
        normalized = float(z_norm)
        if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"z_norm must lie in [0, 1], found {normalized}.")
        return min(19, int(normalized * 20.0))
    z = int(row.get("z", 0))
    if z < 0:
        raise ValueError(f"z must be non-negative, found {z}.")
    return z


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row.get("z", 0)),
        _text(row.get("case_id")),
        _text(row.get("source_key")),
        _text(row.get("source_split")),
    )


def _seeded_rng(seed: int, source: str, split_index: int = 0) -> np.random.Generator:
    source_index = SOURCES.index(source) if source in SOURCES else 0
    sequence = np.random.SeedSequence([int(seed), int(source_index), int(split_index)])
    return np.random.default_rng(sequence)


def _split_counts(total: int, fractions: Sequence[float] = FRACTIONS) -> list[int]:
    if len(fractions) != 3 or any(float(value) <= 0.0 for value in fractions):
        raise ValueError("Negative split fractions must contain three positive values.")
    if not np.isclose(float(sum(fractions)), 1.0):
        raise ValueError("Negative split fractions must sum to one.")
    # Each pseudo-label group needs an even participant count.  Choose the
    # closest all-even allocation to the requested fractions, using all
    # participants for an even source and one fixed omission for an odd
    # source.  Brute force is tiny here (the largest source has only a few
    # hundred participants) and makes the rounding rule auditable.
    target = np.asarray([int(total) * float(value) for value in fractions], dtype=np.float64)
    usable_total = int(total) if int(total) % 2 == 0 else int(total) - 1
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for first in range(2, usable_total + 1, 2):
        for second in range(2, usable_total - first + 1, 2):
            third = usable_total - first - second
            if third < 2 or third % 2:
                continue
            values = (first, second, third)
            # A small deterministic tie-break preserves train/val/test order.
            error = float(np.sum((np.asarray(values, dtype=np.float64) - target) ** 2))
            candidates.append((error, values))
    if not candidates:
        raise ValueError(f"Source has too few participants for balanced 40/10/50 split: {total}.")
    counts = list(min(candidates, key=lambda item: (item[0], item[1]))[1])
    if any(count < 2 for count in counts):
        raise ValueError(f"Source has too few participants for 40/10/50 split: {counts!r}.")
    return counts


def _participant_profile(rows: Sequence[Mapping[str, Any]]) -> tuple[float, set[int]]:
    bins = {_z_bin_for_row(row) for row in rows}
    mean_bin = float(np.mean(sorted(bins))) if bins else 0.0
    return mean_bin, bins


def _build_one_source_split(
    *,
    source: str,
    split: str,
    participants: Sequence[str],
    participant_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    pseudo_seed: int,
    split_index: int,
    source_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign, pair, and z-match one source/target-split group."""

    assigned = list(participants)
    if len(assigned) < 2:
        raise ValueError(f"{source}/{split} has fewer than two participants.")
    # Odd-participant omission is part of the fixed source split assignment,
    # so it must not vary with a fit/initialization seed.  The final
    # participant in the deterministic split order is omitted before the
    # fixed seed-73 pseudo-label permutation is drawn.
    omitted: list[dict[str, Any]] = []
    if len(assigned) % 2:
        omitted_id = assigned[-1]
        assigned = assigned[:-1]
        omitted.append(
            {
                "participant_id": omitted_id,
                "reason": "ODD_PARTICIPANT_FOR_BALANCED_PSEUDO_GROUP",
            }
        )
    pseudo_rng = _seeded_rng(pseudo_seed, source, split_index + 11)
    permutation = pseudo_rng.permutation(len(assigned)).tolist()
    ordered = [assigned[int(index)] for index in permutation]
    midpoint = len(ordered) // 2
    group_zero = ordered[:midpoint]
    group_one = ordered[midpoint:]
    if len(group_zero) != len(group_one) or not group_zero:
        raise ValueError(f"{source}/{split} cannot form balanced pseudo groups.")

    profiles = {
        participant: _participant_profile(participant_rows[participant])
        for participant in ordered
    }
    remaining_one = list(group_one)
    output: list[dict[str, Any]] = []
    pair_audit: list[dict[str, Any]] = []
    pair_index = 0
    for participant_zero in group_zero:
        if not remaining_one:
            break
        mean_zero, bins_zero = profiles[participant_zero]
        candidate = min(
            remaining_one,
            key=lambda participant_one: (
                -len(bins_zero.intersection(profiles[participant_one][1])),
                abs(mean_zero - profiles[participant_one][0]),
                participant_one,
            ),
        )
        remaining_one.remove(candidate)
        bins_one = profiles[candidate][1]
        common_bins = sorted(bins_zero.intersection(bins_one))
        pair_info: dict[str, Any] = {
            "participants": [participant_zero, candidate],
            "common_z_bins": common_bins,
            "pair_id": None,
            "slices_per_label": 0,
            "status": "MATCHED" if common_bins else "OMITTED_NO_COMMON_Z_BIN",
        }
        if not common_bins:
            omitted.extend(
                [
                    {"participant_id": participant_zero, "reason": "NO_COMMON_Z_BIN"},
                    {"participant_id": candidate, "reason": "NO_COMMON_Z_BIN"},
                ]
            )
            pair_audit.append(pair_info)
            continue
        pair_id = (
            f"negative:mixed:{source}:{split}:seed{int(pseudo_seed)}:pair_{pair_index:04d}"
        )
        pair_index += 1
        pair_info["pair_id"] = pair_id
        rows_by_participant_bin: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for participant in (participant_zero, candidate):
            bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in participant_rows[participant]:
                bins[_z_bin_for_row(row)].append(row)
            for rows in bins.values():
                rows.sort(key=_row_sort_key)
            rows_by_participant_bin[participant] = dict(bins)
        for bin_value in common_bins:
            rows_zero = rows_by_participant_bin[participant_zero].get(bin_value, [])
            rows_one = rows_by_participant_bin[candidate].get(bin_value, [])
            keep_count = min(len(rows_zero), len(rows_one), 2)
            pair_info["slices_per_label"] += int(keep_count)
            for participant, label, selected in (
                (participant_zero, 0, rows_zero[:keep_count]),
                (candidate, 1, rows_one[:keep_count]),
            ):
                for original in selected:
                    updated = dict(original)
                    # ``split`` is the new negative-control split.  The
                    # original LMDB split is immutable provenance and must
                    # remain exactly as read.
                    updated["split"] = split
                    updated["label"] = int(label)
                    updated["pair_id"] = pair_id
                    updated["domain"] = "same_cohort_negative"
                    metadata = dict(updated.get("metadata", {}) or {})
                    metadata.update(
                        {
                            "negative_control": True,
                            "negative_control_type": "mixed_source_stratified",
                            "source_stratified": True,
                            "underlying_source_dataset": source,
                            "negative_source": source,
                            "negative_split": split,
                            "pseudo_label_seed": int(pseudo_seed),
                            "pseudo_label": int(label),
                            "original_label": 0,
                            "negative_pair_source": source,
                            "negative_pair_split": split,
                            "source_split_immutable": _text(original.get("source_split")),
                            "negative_common_z_bins": common_bins,
                        }
                    )
                    updated["metadata"] = metadata
                    output.append(updated)
        pair_audit.append(pair_info)
    audit = {
        "source": source,
        "split": split,
        "pseudo_seed": int(pseudo_seed),
        "assigned_participants": int(len(participants)),
        "used_pseudo_participants": int(len(ordered)),
        "pseudo_label_participants": {"0": len(group_zero), "1": len(group_one)},
        "omitted_participants": omitted,
        "matched_pair_count": int(sum(1 for pair in pair_audit if pair["status"] == "MATCHED")),
        "pair_audit": pair_audit,
        "output_slice_counts": {
            "0": sum(1 for row in output if int(row["label"]) == 0),
            "1": sum(1 for row in output if int(row["label"]) == 1),
        },
        "z_bin_histogram": {
            str(label): {
                str(bin_value): sum(
                    1
                    for row in output
                    if int(row["label"]) == label and _z_bin_for_row(row) == bin_value
                )
                for bin_value in range(20)
                if any(
                    int(row["label"]) == label and _z_bin_for_row(row) == bin_value
                    for row in output
                )
            }
            for label in (0, 1)
        },
    }
    return output, audit


def build_source_stratified_negative_records(
    records: Sequence[Mapping[str, Any] | Any],
    *,
    split_seed: int = 73,
    pseudo_seed: int = 73,
    fractions: Sequence[float] = FRACTIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Mixed negative rows with source-stratified pseudo groups.

The source participant split and pseudo-label pairing are both frozen at
seed 73 by the v3 control amendment.  The three requested seeds (73/173/273)
are model initialization seeds only; they must consume the same source-local
pair and slice selection.  This matches the existing FOMO negative control
and keeps the three fits a stability check rather than three data draws.
    """

    normalized = [_record_mapping(record) for record in records]
    if not normalized:
        raise ValueError("Mixed negative control cannot use an empty manifest.")
    if len(fractions) != 3 or not np.isclose(float(sum(fractions)), 1.0):
        raise ValueError("Negative split fractions must sum to one.")
    source_participant_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        source: {} for source in SOURCES
    }
    immutable_keys: set[tuple[str, str, str, str, str, int]] = set()
    original_source_splits: dict[str, set[str]] = {source: set() for source in SOURCES}
    for row in normalized:
        label = float(row.get("label", float("nan")))
        if not np.isfinite(label) or label not in (0.0, 1.0):
            raise ValueError("Mixed negative input labels must be finite binary values.")
        # The comparison manifest contains both healthy and BraTS rows.  The
        # negative control deliberately uses only the healthy (original label
        # 0) side; BraTS rows are excluded before source assignment and are
        # never relabeled into the pseudo cohort.
        if label != 0.0:
            continue
        if _text(row.get("source_dataset")).lower() != "mixed":
            raise ValueError(
                "Mixed source-stratified helper requires source_dataset='mixed' on every row."
            )
        source = source_name_for_row(row)
        participant = _text(row.get("participant_id"))
        source_split = _text(row.get("source_split"))
        if not source_split:
            raise ValueError("Mixed negative rows require immutable source_split provenance.")
        source_key = _text(row.get("source_key"))
        case_id = _text(row.get("case_id"))
        z = int(row.get("z", 0))
        key = (source, source_split, source_key, participant, case_id, z)
        if key in immutable_keys:
            raise ValueError(f"Duplicate immutable Mixed image identity: {key!r}.")
        immutable_keys.add(key)
        original_source_splits[source].add(source_split)
        source_participant_rows[source].setdefault(participant, []).append(row)

    source_assignment_audit: dict[str, Any] = {}
    output: list[dict[str, Any]] = []
    for source_index, source in enumerate(SOURCES):
        participant_rows = source_participant_rows[source]
        if len(participant_rows) < 6:
            raise ValueError(
                f"Mixed source {source!r} has too few participants: {len(participant_rows)}."
            )
        participants = sorted(participant_rows)
        counts = _split_counts(len(participants), fractions)
        split_assignment_rng = _seeded_rng(int(split_seed), source, source_index + 1)
        permutation = split_assignment_rng.permutation(len(participants)).tolist()
        shuffled = [participants[int(index)] for index in permutation]
        assigned: dict[str, list[str]] = {split: [] for split in SPLITS}
        cursor = 0
        for split, count in zip(SPLITS, counts):
            assigned[split] = shuffled[cursor : cursor + int(count)]
            cursor += int(count)
        assigned_total = sum(counts)
        unassigned = shuffled[assigned_total:]
        source_audit: dict[str, Any] = {
            "source": source,
            "participant_count": len(participants),
            "source_split_values": sorted(original_source_splits[source]),
            "target_split_fractions": list(fractions),
            "target_split_counts": {
                split: len(assigned[split]) for split in SPLITS
            },
            "split_assignment_omissions": [
                {
                    "participant_id": participant,
                    "reason": "ODD_SOURCE_TOTAL_FOR_BALANCED_PSEUDO_SPLITS",
                }
                for participant in unassigned
            ],
            "split_seed": int(split_seed),
            "pseudo_seed": int(pseudo_seed),
            "splits": {},
        }
        for split_index, split in enumerate(SPLITS):
            split_rows, split_audit = _build_one_source_split(
                source=source,
                split=split,
                participants=assigned[split],
                participant_rows=participant_rows,
                pseudo_seed=int(pseudo_seed),
                split_index=split_index,
                source_index=source_index,
            )
            output.extend(split_rows)
            source_audit["splits"][split] = split_audit
        source_assignment_audit[source] = source_audit

    output.sort(
        key=lambda row: (
            SOURCES.index(source_name_for_row(row)),
            SPLITS.index(_text(row.get("split"))),
            _text(row.get("pair_id")),
            int(row.get("label", -1)),
            _z_bin_for_row(row),
            _row_sort_key(row),
        )
    )
    # Every emitted pseudo pair is source-local and has exactly two
    # participant IDs.  This audit is deliberately strict because a later
    # subject bootstrap must never silently merge source groups.
    pair_members: dict[str, set[str]] = defaultdict(set)
    pair_sources: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for row in output:
        pair = _text(row.get("pair_id"))
        if not pair:
            raise ValueError("Mixed negative output contains an empty pair_id.")
        pair_members[pair].add(_text(row.get("participant_id")))
        pair_sources[pair].add(source_name_for_row(row))
        pair_splits[pair].add(_text(row.get("split")))
    for pair, members in pair_members.items():
        if len(members) != 2 or len(pair_sources[pair]) != 1 or len(pair_splits[pair]) != 1:
            raise ValueError(f"Invalid source-local negative pair {pair!r}: {members!r}.")
    global_audit = {
        "control_type": "mixed_source_stratified",
        "split_seed": int(split_seed),
        "pseudo_seed": int(pseudo_seed),
        "fractions": list(fractions),
        "input_rows": len(normalized),
        "healthy_input_rows": sum(
            1 for row in normalized if float(row.get("label", float("nan"))) == 0.0
        ),
        "output_rows": len(output),
        "input_participants_by_source": {
            source: len(source_participant_rows[source]) for source in SOURCES
        },
        "output_participants_by_source": {
            source: len(
                {
                    _text(row.get("participant_id"))
                    for row in output
                    if source_name_for_row(row) == source
                }
            )
            for source in SOURCES
        },
        "output_pair_count": len(pair_members),
        "pair_source_crossings": sum(1 for values in pair_sources.values() if len(values) != 1),
        "pair_split_crossings": sum(1 for values in pair_splits.values() if len(values) != 1),
        "source_assignment": source_assignment_audit,
    }
    return output, global_audit


def build_source_stratified_negative_datasets(
    datasets: Mapping[str, Any],
    *,
    split_seed: int = 73,
    pseudo_seed: int = 73,
    fractions: Sequence[float] = FRACTIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transform materialized Mixed datasets while retaining their loader."""

    missing = [split for split in SPLITS if split not in datasets]
    if missing:
        raise KeyError(f"Mixed negative datasets are missing splits: {missing!r}.")
    all_records: list[Any] = []
    for split in SPLITS:
        records = getattr(datasets[split], "records", None)
        if records is None:
            raise TypeError("Mixed negative datasets must expose manifest records.")
        all_records.extend(records)
    output_rows, audit = build_source_stratified_negative_records(
        all_records,
        split_seed=int(split_seed),
        pseudo_seed=int(pseudo_seed),
        fractions=fractions,
    )
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in output_rows:
        split = _text(row.get("split"))
        if split not in by_split:
            raise ValueError(f"Negative output has invalid target split {split!r}.")
        by_split[split].append(row)
    # Importing these adapters lazily keeps the pure mapping builder usable in
    # tests that do not have the full data package available.
    from andi_rewrite.domain_classifier.runner import subset_dataset

    transformed = {
        split: subset_dataset(datasets[split], by_split[split]) for split in SPLITS
    }
    return transformed, audit


def build_source_stratified_negative_cached_inputs(
    cached_inputs: Any,
    *,
    split_seed: int = 73,
    pseudo_seed: int = 73,
    fractions: Sequence[float] = FRACTIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a Mixed negative view over a validated v3 tensor cache.

    ``V3CachedInputs`` is the production binding boundary: it has already
    checked every selected healthy/BraTS tensor against the frozen ledger and
    keeps one canonical tensor object per source identity.  This adapter only
    copies row metadata and asks ``with_label_rows`` for a view, so no image is
    read again and every fit seed sees the same tensor and pair selection.
    """

    records_by_split = getattr(cached_inputs, "records_by_split", None)
    with_label_rows = getattr(cached_inputs, "with_label_rows", None)
    if not isinstance(records_by_split, Mapping) or not callable(with_label_rows):
        raise TypeError("cached_inputs must be a validated V3CachedInputs object.")
    all_records: list[Mapping[str, Any]] = []
    for split in SPLITS:
        if split not in records_by_split:
            raise KeyError(f"Validated cache is missing {split} records.")
        all_records.extend(_record_mapping(row) for row in records_by_split[split])
    output_rows, audit = build_source_stratified_negative_records(
        all_records,
        split_seed=int(split_seed),
        pseudo_seed=int(pseudo_seed),
        fractions=fractions,
    )
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in output_rows:
        split = _text(row.get("split"))
        if split not in by_split:
            raise ValueError(f"Negative output has invalid target split {split!r}.")
        by_split[split].append(row)
    return dict(with_label_rows(by_split)), audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_paths(manifest_root: Path) -> dict[str, Path]:
    root = manifest_root.resolve()
    paths = {split: root / f"{split}.jsonl" for split in SPLITS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Mixed manifest files are missing: " + "; ".join(missing))
    return paths


def _load_config(config_path: Path, *, model: str, device: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The Mixed negative helper requires PyYAML.") from exc
    from andi_rewrite.domain_classifier.runner import TrainConfig

    value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = value.get("training", value)
    if not isinstance(training, Mapping):
        training = {}
    allowed = set(TrainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: item for key, item in training.items() if key in allowed}
    if "widths" in kwargs:
        kwargs["widths"] = tuple(int(item) for item in kwargs["widths"])
    kwargs.update(
        {
            "model": model,
            "in_channels": 3,
            "modalities": JOINT_MODALITIES,
            "stage": "final",
            "device": device,
            "split_seed": 73,
            "max_epochs": 40,
            "patience": 8,
            "early_stopping": True,
            "dropout": 0.0,
            "weight_decay": 1.0e-4,
            "batch_size": 32,
            "num_workers": 0,
            "threshold": 0.5,
        }
    )
    if model == "resnet18":
        kwargs["learning_rate"] = 3.0e-4
    else:
        kwargs["learning_rate"] = 1.0e-3
    return TrainConfig(**kwargs)


def _runtime_thread_policy() -> dict[str, Any]:
    import torch

    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def _base_fingerprint(config: Any, paths: Mapping[str, Path], *, family: str) -> str:
    payload = {
        "config": asdict(config),
        "family": family,
        "paths": {
            split: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for split, path in paths.items()
        },
        "helper_sha256": _sha256(Path(__file__).resolve()),
        "protocol": "mixed_source_stratified_negative_v3",
    }
    return hashlib.sha256(
        json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _set_single_thread_runtime() -> None:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_control(
    *,
    config_path: Path,
    manifest_root: Path,
    output_root: Path,
    build_root: Path | None = None,
    ledger_path: Path | None = None,
    ledger_summary_path: Path | None = None,
    expected_ledger_sha256: str | None = None,
    family: str,
    device: str,
    split_seed: int,
    seeds: Sequence[int],
    dry_run: bool = False,
) -> dict[str, Any]:
    if family not in FAMILY_TO_MODEL:
        raise ValueError(f"Unsupported family {family!r}.")
    if Path(sys.executable).resolve() != ANDI_PYTHON.resolve():
        raise RuntimeError(f"Mixed negative execution requires {ANDI_PYTHON}.")
    if int(split_seed) != 73:
        raise ValueError("The frozen Mixed negative control requires split_seed=73.")
    paths = _manifest_paths(manifest_root)
    model = FAMILY_TO_MODEL[family]
    config = _load_config(config_path, model=model, device=device)
    from andi_rewrite.domain_classifier.runner import (
        atomic_write_json,
        dataset_from_manifest,
        evaluate_negative_gate,
        read_jsonl_manifest,
        write_prediction_rows,
    )

    manifest_counts = {
        split: len(read_jsonl_manifest(path)) for split, path in paths.items()
    }
    base_fp = _base_fingerprint(config, paths, family=family)
    common = {
        "control_type": "mixed_source_stratified",
        "comparison": "mixed",
        "mode": "negative",
        "family": family,
        "model": model,
        "modalities": list(JOINT_MODALITIES),
        "split_seed": int(split_seed),
        "pseudo_label_seed": 73,
        "fit_seeds": [int(seed) for seed in seeds],
        "fractions": list(FRACTIONS),
        "manifest_paths": {split: str(path) for split, path in paths.items()},
        "manifest_counts": manifest_counts,
        "base_fingerprint": base_fp,
        "protocol": "source-stratified participant split, pseudo labels, and z matching",
        "source_split_immutable": True,
        "cross_source_pairs_forbidden": True,
    }
    if dry_run:
        rows = [
            row
            for path in paths.values()
            for row in read_jsonl_manifest(path)
        ]
        planned_audits: dict[str, Any] = {}
        for seed in seeds:
            _, audit = build_source_stratified_negative_records(
                rows, split_seed=int(split_seed), pseudo_seed=73
            )
            planned_audits[str(int(seed))] = audit
        return {
            **common,
            "status": "PLANNED_NO_TRAINING",
            "training_started": False,
            "output_dir": str((output_root / "negative").resolve()),
            "planned_audits": planned_audits,
        }

    # The runner is imported only after the dry-run branch.  Materialize all
    # original Mixed rows exactly once, then reuse the cache for every
    # pseudo-label seed.  The helper is expected to run through the exact
    # ANDi interpreter; this check is also persisted in the result.
    _set_single_thread_runtime()
    destination = (output_root / "negative").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_build_root = (
        build_root.resolve()
        if build_root is not None
        else manifest_root.resolve().parent.parent
    )
    # Use the frozen v3 runtime for production input binding.  In addition to
    # enforcing the healthy tensor ledger and BraTS provenance hashes, this
    # produces an immutable per-canonical-identity digest map that is saved
    # beside the control output for later matrix/replay audits.
    from andi_rewrite.domain_classifier.v3_runtime import (
        DEFAULT_HEALTHY_LEDGER_SHA256,
        validate_and_materialize_v3_inputs,
    )

    validation_kwargs: dict[str, Any] = {
        "comparison": "mixed",
        "build_root": resolved_build_root,
        "config_path": config_path.resolve(),
    }
    if ledger_path is not None:
        validation_kwargs["ledger_path"] = ledger_path.resolve()
    if ledger_summary_path is not None:
        validation_kwargs["ledger_summary_path"] = ledger_summary_path.resolve()
    validation_kwargs["expected_ledger_sha256"] = (
        DEFAULT_HEALTHY_LEDGER_SHA256
        if expected_ledger_sha256 is None
        else str(expected_ledger_sha256)
    )
    cached_inputs = validate_and_materialize_v3_inputs(
        manifest_root.resolve(), **validation_kwargs
    )
    input_binding_audit = dict(cached_inputs.input_binding_audit)
    _write_or_verify_immutable_json(destination / "input_binding_audit.json", input_binding_audit)
    tensor_digest_rows = [
        {
            "identity": list(identity),
            "tensor_sha256": digest,
        }
        for identity, digest in sorted(
            cached_inputs.tensor_digests_by_identity.items(), key=lambda item: repr(item[0])
        )
    ]
    _write_or_verify_immutable_json(
        destination / "input_tensor_digest_ledger.json",
        {
            "status": "PASS",
            "comparison": "mixed",
            "shape": [3, 128, 128],
            "dtype": "torch.float32",
            "unique_canonical_tensors": len(tensor_digest_rows),
            "rows": tensor_digest_rows,
            "source": "v3_runtime.validate_and_materialize_v3_inputs",
            "input_binding_audit": str(destination / "input_binding_audit.json"),
        },
    )
    cached_metadata = {
        "input_binding_audit": str(destination / "input_binding_audit.json"),
        "input_tensor_digest_ledger": str(destination / "input_tensor_digest_ledger.json"),
        "input_binding_status": input_binding_audit.get("status"),
        "unique_canonical_tensors": len(tensor_digest_rows),
        "source_freeze": cached_inputs.source_freeze,
        "healthy_ledger_sha256": cached_inputs.ledger.ledger_sha256,
    }
    common.update(cached_metadata)
    identity = {
        **common,
        "run_fingerprint": base_fp,
        "output_dir": str(destination),
        "runtime": _runtime_thread_policy(),
    }
    identity_path = destination / "run_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != base_fp:
            raise RuntimeError(
                f"Existing Mixed negative output belongs to another run: {destination}"
            )
    else:
        atomic_write_json(identity_path, identity)

    from andi_rewrite.domain_classifier.runner import DomainClassifierRunner
    from scripts.run_domain_classifier_controls import (
        _atomic_torch_save,
        _json_safe as controls_json_safe,
        _persist_control_manifests,
        _read_existing_result,
    )

    runner = DomainClassifierRunner(config)
    results: list[dict[str, Any]] = []
    source_audits: dict[str, Any] = {}
    control_manifests: dict[str, Any] = {}
    # Build and persist one immutable source-local pair/slice selection.  All
    # three initialization seeds consume this exact view; rebuilding it in a
    # fit loop would make the control a data-draw comparison rather than a
    # model-initialization stability check.
    common_seed_rows, common_audit = build_source_stratified_negative_cached_inputs(
        cached_inputs, split_seed=int(split_seed), pseudo_seed=73
    )
    common_manifest = _persist_control_manifests(destination / "common", common_seed_rows)
    common_manifest_hashes = {
        split: _sha256(Path(common_manifest["manifest_dir"]) / f"{split}.jsonl")
        for split in SPLITS
    }
    common_manifest["manifest_sha256"] = common_manifest_hashes
    common_control_manifest = {
        **common_manifest,
        "manifest_dir": str(Path(common_manifest["manifest_dir"]).resolve()),
        "manifest_hashes": {
            split: {"status": "PASS", "sha256": digest}
            for split, digest in common_manifest_hashes.items()
        },
        "reused_across_fit_seeds": True,
    }
    _write_or_verify_immutable_json(
        destination / "common" / "source_stratified_audit.json", common_audit
    )
    for seed in [int(value) for value in seeds]:
        seed_rows = common_seed_rows
        audit = dict(common_audit)
        audit["fit_seed"] = seed
        audit["pairing_reused_across_fit_seeds"] = True
        source_audits[str(seed)] = audit
        control_manifests[str(seed)] = {
            **common_control_manifest,
            "fit_seed": seed,
        }
        artifact = destination / f"seed_{seed}.json"
        checkpoint = destination / f"model_seed_{seed}.pt"
        predictions_path = destination / f"seed_{seed}_test_predictions.jsonl"
        existing = _read_existing_result(artifact, base_fp)
        if existing is not None:
            if not checkpoint.is_file() or not predictions_path.is_file():
                raise RuntimeError(f"Incomplete Mixed negative seed artifacts: {artifact}")
            results.append(existing)
            continue
        if checkpoint.exists() or predictions_path.exists():
            raise RuntimeError(f"Partial Mixed negative seed artifacts exist: {artifact}")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        result = runner.train_one_seed(
            seed_rows["train"],
            seed_rows["val"],
            seed_rows["test"],
            seed=seed,
        )
        elapsed = float(time.perf_counter() - started)
        result = {
            "run_fingerprint": base_fp,
            "control_type": "mixed_source_stratified",
            "comparison": "mixed",
            "fit_seed": seed,
            "pseudo_label_seed": 73,
            "source_stratified_audit": audit,
            "control_manifest": common_control_manifest,
            "timing": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
            },
            **result,
        }
        _atomic_torch_save(result.get("state_dict"), checkpoint)
        write_prediction_rows(predictions_path, result.get("test_predictions", []))
        serializable = controls_json_safe(result)
        atomic_write_json(artifact, serializable)
        results.append(serializable)

    # Existing artifacts from a resumed run predate the in-memory audit
    # attachment; sidecars remain authoritative and are included in the
    # aggregate below.  The gate always sees exact per-seed metrics.
    gate = evaluate_negative_gate(results)
    summary = {
        **common,
        "mode": "negative",
        "run_fingerprint": base_fp,
        "status": gate.get("status"),
        "training_started": True,
        "output_dir": str(destination),
        "gate": gate,
        "results": results,
        "source_stratified_audit_by_seed": source_audits,
        "control_manifests_by_seed": control_manifests,
        "control_manifest": common_control_manifest,
        "common_control_manifest": common_control_manifest,
        "common_control_manifest_sha256": common_manifest_hashes,
        "pairing_and_slice_selection_reused_across_fit_seeds": True,
        "runtime": _runtime_thread_policy(),
        "cache_materialized_once": True,
        "shared_tensor_cache": True,
    }
    atomic_write_json(destination / "gate.json", controls_json_safe(summary), overwrite=True)
    return controls_json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--ledger-summary-path", type=Path, default=None)
    parser.add_argument("--expected-ledger-sha256", default=None)
    parser.add_argument("--family", choices=tuple(FAMILY_TO_MODEL), default="small_cnn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=73)
    parser.add_argument("--seeds", nargs="+", type=int, default=(73, 173, 273))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    manifest_root = (
        args.manifest_root
        if args.manifest_root.is_absolute()
        else REPO_ROOT / args.manifest_root
    )
    output_root = (
        args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    )
    result = run_control(
        config_path=config_path.resolve(),
        manifest_root=manifest_root.resolve(),
        output_root=output_root.resolve(),
        build_root=(
            None
            if args.build_root is None
            else (
                args.build_root
                if args.build_root.is_absolute()
                else REPO_ROOT / args.build_root
            ).resolve()
        ),
        ledger_path=(
            None
            if args.ledger_path is None
            else (
                args.ledger_path
                if args.ledger_path.is_absolute()
                else REPO_ROOT / args.ledger_path
            ).resolve()
        ),
        ledger_summary_path=(
            None
            if args.ledger_summary_path is None
            else (
                args.ledger_summary_path
                if args.ledger_summary_path.is_absolute()
                else REPO_ROOT / args.ledger_summary_path
            ).resolve()
        ),
        expected_ledger_sha256=args.expected_ledger_sha256,
        family=str(args.family),
        device=str(args.device),
        split_seed=int(args.split_seed),
        seeds=tuple(int(seed) for seed in args.seeds),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
