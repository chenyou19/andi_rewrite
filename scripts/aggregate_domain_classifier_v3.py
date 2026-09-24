"""Aggregate the frozen v3 cell matrix into the two preregistered families.

This command is deliberately a CPU-only reader of JSON/JSONL artifacts.  It
does not read MRI/LMDB/NPZ data and it never discovers a purported result by
glob.  The explicit ``matrix_plan.json`` supplies the 112 cell identities and
the result path for every seed.  The ordinary 48 groups are formed by
cohort/input/model; probabilities are averaged per subject across the fixed
seed family, then the frozen subject pairs are resampled for the 2,000-pair
bootstrap and 9,999 fixed-score swaps.  The four all-modality SmallCNN seed-73
primary cells are validated through their separate primary-null directories
and enter Holm-4 only when all 199 draws are complete and integrity checks
pass.

No p-value is accepted from an arbitrary result field.  Secondary p-values
are produced here from the fixed subject ensemble.  Primary p-values are
produced only from a complete generic ``run_domain_classifier_primary_null_v3``
artifact with finite, hash-verified draw results and deterministic label
reconstruction.  Legacy FOMO calibration output is adapted for descriptive
observed probabilities but remains explicitly ineligible for primary Holm-4
integrity when it lacks the generic draw/hash schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np  # noqa: E402

from andi_rewrite.domain_classifier.metrics import compute_dataset_metrics  # noqa: E402
from andi_rewrite.domain_classifier.report import (  # noqa: E402
    aggregate_fixed_seed_subject_ensemble,
    compute_secondary_ensemble_statistics,
    summarise_secondary_48,
)
from andi_rewrite.domain_classifier.v3_runtime import (  # noqa: E402
    DEFAULT_HEALTHY_LEDGER_SHA256,
    DEFAULT_LABEL_STREAM_ROOT,
    _label_stream_seeds,
)


PROTOCOL_ID = "domain_classifier_v3_statistics_v1"
COHORTS = ("fomo45k", "mpi", "oasis3", "mixed")
JOINT_MODALITIES = ("flair", "t1", "t2")
EXPECTED_MATRIX_CELLS = 112
EXPECTED_SECONDARY_GROUPS = 48
EXPECTED_PRIMARY_CELLS = 4
EXPECTED_NEURAL_SEEDS = (73, 173, 273)
EXPECTED_LOGISTIC_SEEDS = (73,)
PRIMARY_PROTOCOL_ID = "domain_classifier_primary_null_v3"


class StatisticsProtocolError(RuntimeError):
    """Raised when a frozen statistics input cannot be audited safely."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticsProtocolError(f"cannot read JSON artifact {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise StatisticsProtocolError(f"missing JSONL artifact {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise StatisticsProtocolError(f"JSONL row is not an object at {path}:{line_number}")
                rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticsProtocolError(f"cannot read JSONL artifact {path}") from exc
    return rows


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise StatisticsProtocolError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _normalise_model(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"logistic", "statistical", "statistical_logistic", "logreg", "logistic_regression"}:
        return "logistic"
    if text in {"small", "cnn", "small_cnn", "smallcnn"}:
        return "small_cnn"
    if text in {"resnet", "resnet18", "resnet_18", "resnet18_scratch_gn"}:
        return "resnet18_scratch_gn"
    return text


def _normalise_modalities(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = tuple(item.strip().lower() for item in value.replace(",", " ").split() if item.strip())
    elif isinstance(value, Sequence):
        values = tuple(str(item).strip().lower() for item in value)
    else:
        values = ()
    return values


def _is_primary_cell(cell: Mapping[str, Any]) -> bool:
    family = _normalise_model(cell.get("family", cell.get("model")))
    modalities = _normalise_modalities(cell.get("modalities", cell.get("input_modalities")))
    try:
        seed = int(cell.get("seed", -1))
    except (TypeError, ValueError):
        seed = -1
    return family == "small_cnn" and modalities == JOINT_MODALITIES and seed == 73


def _group_id(cohort: str, family: str, modalities: Sequence[str]) -> str:
    return f"{cohort}__{family}__{'_'.join(modalities)}"


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _result_path_for_cell(
    cell: Mapping[str, Any],
    *,
    build_root: Path,
    primary_roots: Mapping[str, Path],
) -> Path:
    """Resolve only an explicit plan path or the declared cell output path."""

    explicit = cell.get("result_path")
    if explicit not in (None, ""):
        return _resolve_path(str(explicit), base=REPO_ROOT)
    if _is_primary_cell(cell):
        cohort = str(cell.get("cohort", cell.get("comparison", ""))).strip().lower()
        root = primary_roots.get(cohort)
        if root is None:
            root = build_root / f"stage_a_{cohort}_observed_v3_20260917"
        # The generic primary runner has a fixed observed subdirectory.  A
        # missing path is retained in the generated manifest as PENDING.
        return (root / "observed" / "result.json").resolve()
    output = cell.get("output_dir")
    if output in (None, ""):
        return (build_root / "missing-cell-output" / "result.json").resolve()
    path = _resolve_path(str(output), base=REPO_ROOT)
    return path if path.name.lower() == "result.json" else path / "result.json"


def _load_plan(source: str | Path | Mapping[str, Any], *, build_root: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise StatisticsProtocolError("matrix plan must be a JSON object")
    return dict(payload), path


def _plan_groups(
    plan: Mapping[str, Any],
    *,
    build_root: Path,
    primary_roots: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
    raw_cells = plan.get("cells")
    errors: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes, bytearray)):
        return {}, ["matrix plan has no cells sequence"], []
    plan_cells: list[dict[str, Any]] = []
    seen_cell_ids: set[str] = set()
    for ordinal, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            errors.append(f"cells[{ordinal}] is not an object")
            continue
        cell = dict(raw)
        cell_id = str(cell.get("cell_id", "")).strip()
        if not cell_id:
            errors.append(f"cells[{ordinal}] has no cell_id")
            continue
        if cell_id in seen_cell_ids:
            errors.append(f"duplicate cell_id: {cell_id}")
        seen_cell_ids.add(cell_id)
        cohort = str(cell.get("cohort", cell.get("comparison", ""))).strip().lower()
        family = _normalise_model(cell.get("family", cell.get("model")))
        modalities = _normalise_modalities(cell.get("modalities", cell.get("input_modalities")))
        try:
            seed = int(cell.get("seed"))
        except (TypeError, ValueError):
            seed = -1
        if cohort not in COHORTS:
            errors.append(f"{cell_id}: unsupported cohort {cohort!r}")
        if family not in {"logistic", "small_cnn", "resnet18_scratch_gn"}:
            errors.append(f"{cell_id}: unsupported family {family!r}")
        if modalities not in {
            ("flair",),
            ("t1",),
            ("t2",),
            JOINT_MODALITIES,
        }:
            errors.append(f"{cell_id}: unsupported modalities {modalities!r}")
        expected_seeds = EXPECTED_LOGISTIC_SEEDS if family == "logistic" else EXPECTED_NEURAL_SEEDS
        if seed not in expected_seeds:
            errors.append(f"{cell_id}: seed {seed} is not in {expected_seeds}")
        group_key = _group_id(cohort, family, modalities)
        group = groups.setdefault(
            group_key,
            {
                "cell_id": group_key,
                "cohort": cohort,
                "classifier": family,
                "model": family,
                "modalities": list(modalities),
                "evaluation_split": "test",
                "expected_seeds": list(expected_seeds),
                "seed_cells": {},
            },
        )
        if seed in group["seed_cells"]:
            errors.append(f"{group_key}: duplicate seed {seed}")
        group["seed_cells"][seed] = {
            "seed": seed,
            "cell_id": cell_id,
            "cohort": cohort,
            "family": family,
            "modalities": list(modalities),
            "primary_observed": _is_primary_cell(cell),
            "result_path": str(_result_path_for_cell(cell, build_root=build_root, primary_roots=primary_roots)),
            "plan_cell": cell,
        }
        plan_cells.append(cell)
    if len(raw_cells) != EXPECTED_MATRIX_CELLS:
        errors.append(f"expected {EXPECTED_MATRIX_CELLS} matrix cells, observed {len(raw_cells)}")
    if len(groups) != EXPECTED_SECONDARY_GROUPS:
        errors.append(f"expected {EXPECTED_SECONDARY_GROUPS} secondary groups, observed {len(groups)}")
    for group_id, group in sorted(groups.items()):
        expected = set(int(value) for value in group["expected_seeds"])
        actual = set(int(value) for value in group["seed_cells"])
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{group_id}: missing seeds {missing}")
        if extra:
            errors.append(f"{group_id}: unexpected seeds {extra}")
    return groups, errors, plan_cells


def _result_identity_errors(
    path: Path,
    *,
    cell: Mapping[str, Any],
    primary_observed: bool,
) -> list[str]:
    if not path.is_file():
        return [f"missing result: {path}"]
    try:
        payload = _read_json(path)
    except StatisticsProtocolError as exc:
        return [str(exc)]
    if not isinstance(payload, Mapping):
        return [f"result is not an object: {path}"]
    errors: list[str] = []
    expected_model = _normalise_model(cell.get("family", cell.get("model")))
    result_config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    result_model = _normalise_model(result_config.get("model", payload.get("model")))
    if result_model != expected_model:
        errors.append(f"{path}: result model {result_model!r} != {expected_model!r}")
    result_modalities = _normalise_modalities(result_config.get("modalities", payload.get("modalities")))
    expected_modalities = _normalise_modalities(cell.get("modalities"))
    if result_modalities != expected_modalities:
        errors.append(f"{path}: result modalities {result_modalities!r} != {expected_modalities!r}")
    try:
        result_seed = int(payload.get("seed", result_config.get("seed", -1)))
        expected_seed = int(cell.get("seed", -2))
    except (TypeError, ValueError):
        result_seed, expected_seed = -1, -2
    if result_seed != expected_seed:
        errors.append(f"{path}: result seed {result_seed} != {expected_seed}")
    stage = str(payload.get("stage", result_config.get("stage", "final"))).strip().lower()
    if stage != "final":
        errors.append(f"{path}: result stage is {stage!r}")
    # Primary observed files intentionally use formal_final=false until their
    # separate null is complete.  They are allowed into the 48 observed
    # probability ensemble only by this explicit primary coordinate.
    if not primary_observed:
        if payload.get("formal_final") is False:
            errors.append(f"{path}: non-primary result is explicitly non-formal")
        for key in ("control_mode", "control_type", "control_name", "permutation_index", "shuffle", "is_control"):
            value = payload.get(key)
            if value not in (None, "", False, 0):
                errors.append(f"{path}: control marker {key}={value!r}")
        kind = str(payload.get("result_kind", "")).lower()
        if any(token in kind for token in ("control", "diagnostic", "null", "permutation", "shuffle")):
            errors.append(f"{path}: control/diagnostic result_kind {kind!r}")
    else:
        if payload.get("control_mode") not in (None, "", "formal_final"):
            errors.append(f"{path}: primary observed carries control_mode={payload.get('control_mode')!r}")
    return errors


def _compact_statistics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep report artifacts compact while retaining every inferential field."""

    output = {
        key: value.get(key)
        for key in (
            "status",
            "inference_family",
            "inference_unit",
            "conditional_on",
            "bootstrap_replicates",
            "swap_replicates",
            "seed",
            "errors",
            "subject_count",
            "pair_count",
            "subject_roc_auc",
            "subject_pr_auc",
            "secondary_p_value",
            "secondary_p_value_source",
        )
        if key in value
    }
    bootstrap = value.get("bootstrap")
    if isinstance(bootstrap, Mapping):
        output["bootstrap"] = {
            key: bootstrap.get(key)
            for key in ("observed_auc", "ci_low", "ci_high", "confidence", "n_bootstrap", "n_valid", "resampling_unit")
            if key in bootstrap
        }
    pair_swap = value.get("pair_swap")
    if isinstance(pair_swap, Mapping):
        output["pair_swap"] = {
            key: pair_swap.get(key)
            for key in (
                "observed_auc",
                "two_sided_deviation",
                "p_value",
                "n_swaps",
                "n_valid",
                "n_pairs",
                "status",
                "conditional_on_matched_pairs",
            )
            if key in pair_swap
        }
    return _json_safe(output)


def _manifest_rows_for_cell(cell: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    manifests = cell.get("manifests")
    if not isinstance(manifests, Mapping):
        raise StatisticsProtocolError(f"primary plan cell {cell.get('cell_id')} has no manifests")
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path_value = manifests.get(split)
        if path_value in (None, ""):
            raise StatisticsProtocolError(f"primary plan cell {cell.get('cell_id')} has no {split} manifest")
        rows[split] = _read_jsonl(Path(str(path_value)).resolve())
    return rows


def _finite_probability(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def _validate_prediction_rows(
    predictions: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_labels: Sequence[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    if len(predictions) != len(records):
        return [f"prediction count {len(predictions)} != manifest count {len(records)}"]
    if expected_labels is not None and len(expected_labels) != len(records):
        return ["expected label count differs from manifest"]
    seen: set[int] = set()
    for index, (prediction, record) in enumerate(zip(predictions, records)):
        try:
            record_index = int(prediction.get("record_index"))
        except (TypeError, ValueError):
            errors.append(f"prediction[{index}] has invalid record_index")
            continue
        if record_index != index or record_index in seen:
            errors.append(f"prediction[{index}] record_index join failed")
        seen.add(record_index)
        for field in ("participant_id", "pair_id", "case_id"):
            expected = record.get(field, index if field == "case_id" else None)
            if prediction.get(field) != expected:
                errors.append(f"prediction[{index}] {field} join failed")
        try:
            label = int(prediction.get("label"))
        except (TypeError, ValueError):
            errors.append(f"prediction[{index}] label invalid")
            continue
        probability = _finite_probability(prediction.get("probability"))
        if label not in (0, 1) or probability is None:
            errors.append(f"prediction[{index}] label/probability invalid")
        expected_label = int(record.get("label")) if expected_labels is None else int(expected_labels[index])
        if label != expected_label:
            errors.append(f"prediction[{index}] label differs from expected frozen label")
    return errors


def _expected_pair_labels(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    index: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Rebuild the v3 deterministic whole-pair label stream from manifests."""

    streams = _label_stream_seeds(int(index), root=DEFAULT_LABEL_STREAM_ROOT)
    output: list[dict[str, Any]] = []
    swap_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        records = [dict(row) for row in records_by_split[split]]
        by_pair: dict[str, dict[str, int]] = defaultdict(dict)
        for row in records:
            pair = str(row.get("pair_id", "")).strip()
            participant = str(row.get("participant_id", "")).strip()
            if not pair or not participant:
                raise StatisticsProtocolError(f"{split}: null pair/participant identity")
            label = int(row.get("label"))
            if label not in (0, 1):
                raise StatisticsProtocolError(f"{split}: non-binary baseline label")
            previous = by_pair[pair].get(participant)
            if previous is not None and previous != label:
                raise StatisticsProtocolError(f"{split}: conflicting baseline pair label")
            by_pair[pair][participant] = label
        rng = np.random.default_rng(int(streams[split]))
        swap_count = 0
        mapping: dict[tuple[str, str], int] = {}
        for pair in sorted(by_pair):
            members = by_pair[pair]
            if len(members) != 2 or sorted(members.values()) != [0, 1]:
                raise StatisticsProtocolError(f"{split}/{pair}: expected exactly one class per pair")
            do_swap = bool(rng.integers(0, 2))
            swap_count += int(do_swap)
            for participant, old_label in members.items():
                mapping[(pair, participant)] = 1 - old_label if do_swap else old_label
        swap_counts[split] = swap_count
        for row_index, row in enumerate(records):
            pair = str(row.get("pair_id"))
            participant = str(row.get("participant_id"))
            output.append(
                {
                    "split": split,
                    "record_index": row_index,
                    "participant_id": row.get("participant_id"),
                    "pair_id": row.get("pair_id"),
                    "case_id": row.get("case_id", row_index),
                    "source_dataset": row.get("source_dataset"),
                    "source_split": row.get("source_split"),
                    "source_key": row.get("source_key"),
                    "z": row.get("z"),
                    "label": mapping[(pair, participant)],
                }
            )
    return output, streams, swap_counts


def _validate_null_labels(
    rows: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    fingerprint: str,
    index: int,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != len(expected):
        return [f"labels row count {len(rows)} != expected {len(expected)}"]
    fields = (
        "split",
        "record_index",
        "participant_id",
        "pair_id",
        "case_id",
        "source_dataset",
        "source_split",
        "source_key",
        "z",
        "label",
    )
    for position, (actual, wanted) in enumerate(zip(rows, expected)):
        if actual.get("run_fingerprint") != fingerprint:
            errors.append(f"labels[{position}] fingerprint mismatch")
        if int(actual.get("permutation_index", -1)) != int(index):
            errors.append(f"labels[{position}] permutation index mismatch")
        for field in fields:
            if actual.get(field) != wanted.get(field):
                errors.append(f"labels[{position}] {field} mismatch")
    return errors


def _verify_hashes(result: Mapping[str, Any], paths: Mapping[str, Path], required: Sequence[str]) -> list[str]:
    recorded = result.get("artifact_hashes")
    if not isinstance(recorded, Mapping):
        return ["result has no artifact_hashes"]
    errors: list[str] = []
    for name in required:
        path = paths.get(name)
        expected = recorded.get(name)
        if path is None or not path.is_file():
            errors.append(f"missing artifact {name}")
            continue
        if not isinstance(expected, str) or not expected:
            errors.append(f"missing recorded hash {name}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"artifact hash mismatch {name}: {actual} != {expected}")
    return errors


def _subject_auc_from_predictions(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    labels = np.asarray([int(row.get("label")) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row.get("probability")) for row in rows], dtype=np.float64)
    participants = np.asarray([row.get("participant_id") for row in rows], dtype=object)
    pairs = np.asarray([row.get("pair_id") for row in rows], dtype=object)
    metrics = compute_dataset_metrics(labels, scores, participant_ids=participants, pair_ids=pairs)
    subject = metrics.get("subject") if isinstance(metrics.get("subject"), Mapping) else {}
    return _finite_probability(subject.get("roc_auc"))


def validate_primary_full_retrained_root(
    root: str | Path,
    *,
    plan_cell: Mapping[str, Any] | None = None,
    expected_replicates: int = 199,
) -> dict[str, Any]:
    """Validate one generic primary null root before admitting its p-value."""

    root_path = Path(root).resolve()
    result: dict[str, Any] = {
        "root": str(root_path),
        "status": "INCONCLUSIVE_PENDING",
        "format": "unknown",
        "expected_replicates": int(expected_replicates),
        "completed_replicates": 0,
        "errors": [],
        "warnings": [],
        "p_value": None,
        "p_value_source": None,
        "artifact_hashes": {},
    }
    observed_path = root_path / "observed" / "result.json"
    if not observed_path.is_file():
        result["errors"] = [f"missing observed result: {observed_path}"]
        return result
    try:
        observed = _read_json(observed_path)
    except StatisticsProtocolError as exc:
        result["errors"] = [str(exc)]
        return result
    if not isinstance(observed, Mapping):
        result["errors"] = ["observed result is not an object"]
        return result
    generic = str(observed.get("protocol_id", "")) == PRIMARY_PROTOCOL_ID
    generic = generic or (root_path / "retrained_null" / "permutation_0000" / "result.json").is_file()
    result["format"] = "generic_primary_v3" if generic else "legacy_fomo_calibration"
    errors: list[str] = []
    warnings: list[str] = []
    config = observed.get("config") if isinstance(observed.get("config"), Mapping) else {}
    if _normalise_model(config.get("model", observed.get("model"))) != "small_cnn":
        errors.append("observed model is not SmallCNN")
    if _normalise_modalities(config.get("modalities", observed.get("modalities"))) != JOINT_MODALITIES:
        errors.append("observed modalities are not FLAIR/T1/T2")
    try:
        if int(observed.get("seed", config.get("seed", -1))) != 73:
            errors.append("observed seed is not 73")
        if int(observed.get("split_seed", config.get("split_seed", -1))) != 73:
            errors.append("observed split_seed is not 73")
    except (TypeError, ValueError):
        errors.append("observed seed metadata is invalid")
    if str(config.get("stage", observed.get("stage", "final"))).lower() != "final":
        errors.append("observed stage is not final")
    observed_fingerprint = str(observed.get("run_fingerprint", ""))
    if not observed_fingerprint:
        errors.append("observed run_fingerprint is missing")
    manifest_rows: dict[str, list[dict[str, Any]]] | None = None
    if plan_cell is not None:
        try:
            manifest_rows = _manifest_rows_for_cell(plan_cell)
        except StatisticsProtocolError as exc:
            errors.append(str(exc))
    observed_predictions_path = root_path / "observed" / "test_predictions.jsonl"
    if not observed_predictions_path.is_file():
        errors.append(f"missing observed test predictions: {observed_predictions_path}")
    else:
        try:
            observed_predictions = _read_jsonl(observed_predictions_path)
            if manifest_rows is not None:
                errors.extend(_validate_prediction_rows(observed_predictions, manifest_rows["test"]))
            declared_auc = _finite_probability(
                ((observed.get("test") or {}).get("subject") or {}).get("roc_auc")
                if isinstance(observed.get("test"), Mapping)
                else None
            )
            recomputed_auc = _subject_auc_from_predictions(observed_predictions)
            if declared_auc is None or recomputed_auc is None or not np.isclose(declared_auc, recomputed_auc, rtol=0.0, atol=1.0e-12):
                errors.append(f"observed subject AUC is missing or differs from predictions: {declared_auc} != {recomputed_auc}")
            result["observed_subject_auc"] = recomputed_auc
        except (StatisticsProtocolError, TypeError, ValueError, FloatingPointError) as exc:
            errors.append(f"observed prediction validation failed: {exc}")
    observed_checkpoint = root_path / "observed" / "model_best.pt"
    observed_hash_errors: list[str] = []
    if generic:
        observed_hash_errors = _verify_hashes(
            observed,
            {
                "test_predictions": observed_predictions_path,
                "validation_predictions": root_path / "observed" / "validation_predictions.jsonl",
                "model_best": observed_checkpoint,
            },
            ("test_predictions", "validation_predictions", "model_best"),
        )
        errors.extend(observed_hash_errors)
        for name, path in {
            "observed_test_predictions": observed_predictions_path,
            "observed_validation_predictions": root_path / "observed" / "validation_predictions.jsonl",
            "observed_model_best": observed_checkpoint,
        }.items():
            if path.is_file():
                result["artifact_hashes"][name] = _sha256(path)
        binding_path = root_path / "input_binding_audit.json"
        if not binding_path.is_file():
            errors.append("generic primary input_binding_audit.json is missing")
        else:
            binding = _read_json(binding_path)
            if not isinstance(binding, Mapping) or str(binding.get("status", "")) != "PASS":
                errors.append("generic primary input binding is not PASS")
    else:
        # The old FOMO calibration runner has usable observed subject rows but
        # does not persist the generic per-draw labels/hash contract.  Keep it
        # available as a descriptive adapter and explicitly withhold Holm-4.
        warnings.append("legacy FOMO calibration format lacks generic per-draw artifact hashes/label reconstruction")
    null_root = root_path / "retrained_null"
    summary_path = null_root / "summary.json"
    status_path = null_root / "status.json"
    if not null_root.is_dir():
        errors.append(f"missing retrained_null directory: {null_root}")
    if generic:
        if manifest_rows is None:
            errors.append("generic primary null requires frozen manifests for label reconstruction")
        expected_indices = list(range(int(expected_replicates)))
        completed_indices: list[int] = []
        draw_errors: list[str] = []
        if manifest_rows is not None:
            try:
                expected_label_cache: dict[int, tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]] = {}
                for index in expected_indices:
                    draw_dir = null_root / f"permutation_{index:04d}"
                    draw_result_path = draw_dir / "result.json"
                    labels_path = draw_dir / "labels.jsonl"
                    predictions_path = draw_dir / "test_predictions.jsonl"
                    if not draw_result_path.is_file() or not labels_path.is_file() or not predictions_path.is_file():
                        continue
                    try:
                        draw = _read_json(draw_result_path)
                        if not isinstance(draw, Mapping):
                            raise StatisticsProtocolError("draw result is not an object")
                        if int(draw.get("permutation_index", -1)) != index:
                            raise StatisticsProtocolError("draw permutation index mismatch")
                        if int(draw.get("init_seed", -1)) != 73:
                            raise StatisticsProtocolError("draw init_seed mismatch")
                        if draw.get("run_fingerprint") != observed_fingerprint:
                            raise StatisticsProtocolError("draw run_fingerprint mismatch")
                        if draw.get("result_kind") != "v3_primary_full_retrained_pair_null":
                            raise StatisticsProtocolError("draw result_kind mismatch")
                        test = draw.get("test") if isinstance(draw.get("test"), Mapping) else {}
                        subject = test.get("subject") if isinstance(test.get("subject"), Mapping) else {}
                        auc = _finite_probability(subject.get("roc_auc"))
                        if auc is None:
                            raise StatisticsProtocolError("draw subject AUC is not finite in [0,1]")
                        expected_labels, streams, swap_counts = expected_label_cache.setdefault(
                            index,
                            _expected_pair_labels(manifest_rows, index=index),
                        )
                        actual_labels = _read_jsonl(labels_path)
                        draw_errors.extend(
                            f"index {index}: {message}"
                            for message in _validate_null_labels(
                                actual_labels,
                                expected_labels,
                                fingerprint=observed_fingerprint,
                                index=index,
                            )
                        )
                        if draw.get("label_stream_seeds") != streams:
                            raise StatisticsProtocolError("draw label_stream_seeds mismatch")
                        if draw.get("swap_counts") != swap_counts:
                            raise StatisticsProtocolError("draw swap_counts mismatch")
                        expected_test_labels = [
                            int(row["label"])
                            for row in expected_labels
                            if row["split"] == "test"
                        ]
                        actual_predictions = _read_jsonl(predictions_path)
                        draw_errors.extend(
                            f"index {index}: {message}"
                            for message in _validate_prediction_rows(
                                actual_predictions,
                                manifest_rows["test"],
                                expected_labels=expected_test_labels,
                            )
                        )
                        draw_hash_errors = _verify_hashes(
                            draw,
                            {
                                "labels": labels_path,
                                "test_predictions": predictions_path,
                                "model_best": draw_dir / "model_best.pt",
                            },
                            ("labels", "test_predictions", "model_best"),
                        )
                        draw_errors.extend(f"index {index}: {message}" for message in draw_hash_errors)
                        if not draw_hash_errors:
                            completed_indices.append(index)
                    except (StatisticsProtocolError, TypeError, ValueError, OSError) as exc:
                        draw_errors.append(f"index {index}: {exc}")
        result["completed_replicates"] = len(completed_indices)
        result["completed_indices"] = sorted(completed_indices)
        errors.extend(draw_errors)
        if status_path.is_file():
            status_payload = _read_json(status_path)
            if isinstance(status_payload, Mapping):
                if int(status_payload.get("requested", -1)) != int(expected_replicates):
                    errors.append("null status requested count mismatch")
                declared_completed = sorted(int(value) for value in status_payload.get("completed_indices", []))
                if declared_completed != sorted(completed_indices):
                    errors.append("null status completed_indices mismatch")
        if summary_path.is_file():
            summary = _read_json(summary_path)
            if not isinstance(summary, Mapping):
                errors.append("null summary is not an object")
            else:
                if str(summary.get("status", "")).lower() != "complete":
                    warnings.append(f"null summary status={summary.get('status')!r}")
                statistic = summary.get("statistics")
                if isinstance(statistic, Mapping):
                    p_value = _finite_probability(statistic.get("p_plus_one"))
                    if p_value is not None and int(statistic.get("n_null", -1)) == int(expected_replicates):
                        result["candidate_p_value"] = p_value
                    else:
                        warnings.append("null summary statistics are incomplete or use the wrong denominator")
                else:
                    errors.append("generic null summary has no statistics")
        else:
            warnings.append(f"null summary is pending: {summary_path}")
        if errors:
            result["status"] = "FAIL_CLOSED" if any("mismatch" in str(item).lower() or "hash" in str(item).lower() or "invalid" in str(item).lower() for item in errors) else "INCONCLUSIVE_PENDING"
        elif len(completed_indices) != int(expected_replicates):
            result["status"] = "INCONCLUSIVE_PENDING"
        else:
            summary_payload = _read_json(summary_path) if summary_path.is_file() else {}
            statistic = summary_payload.get("statistics") if isinstance(summary_payload, Mapping) else None
            p_value = _finite_probability(statistic.get("p_plus_one")) if isinstance(statistic, Mapping) else None
            if p_value is None or not isinstance(statistic, Mapping) or int(statistic.get("n_null", -1)) != int(expected_replicates):
                result["status"] = "INCONCLUSIVE_PENDING"
            else:
                result["status"] = "PASS"
                result["p_value"] = p_value
                result["p_value_source"] = "primary_full_retrained_holm4"
                result["p_denominator"] = int(expected_replicates) + 1
    else:
        # Legacy summary fields are retained as evidence but can never satisfy
        # the generic primary Holm-4 contract without draw-level hashes.
        result["status"] = "INCONCLUSIVE_LEGACY_FORMAT" if not errors else "FAIL_CLOSED"
    result["errors"] = sorted(set(str(item) for item in errors))
    result["warnings"] = sorted(set(str(item) for item in warnings))
    return result


def aggregate_v3_statistics(
    matrix_plan: str | Path | Mapping[str, Any],
    *,
    build_root: str | Path,
    output_root: str | Path | None = None,
    primary_roots: Mapping[str, str | Path] | None = None,
    bootstrap_replicates: int = 2000,
    swap_replicates: int = 9999,
    expected_matrix_cells: int = EXPECTED_MATRIX_CELLS,
    expected_secondary_groups: int = EXPECTED_SECONDARY_GROUPS,
    expected_primary_cells: int = EXPECTED_PRIMARY_CELLS,
    primary_replicates: int = 199,
) -> dict[str, Any]:
    """Build secondary/primary family summaries from one explicit matrix plan."""

    build = Path(build_root).resolve()
    plan, plan_path = _load_plan(matrix_plan, build_root=build)
    roots = {
        cohort: (Path(primary_roots[cohort]).resolve() if primary_roots and cohort in primary_roots else build / f"stage_a_{cohort}_observed_v3_20260917")
        for cohort in COHORTS
    }
    groups, plan_errors, plan_cells = _plan_groups(plan, build_root=build, primary_roots=roots)
    # Tests may use a deliberately small fixture; production defaults remain
    # the fixed 112/48/4 contract.
    if len(plan_cells) != int(expected_matrix_cells):
        plan_errors.append(f"expected {expected_matrix_cells} matrix cells, observed {len(plan_cells)}")
    if len(groups) != int(expected_secondary_groups):
        plan_errors.append(f"expected {expected_secondary_groups} secondary groups, observed {len(groups)}")
    manifest_cells: list[dict[str, Any]] = []
    group_audits: list[dict[str, Any]] = []
    for group_id, group in sorted(groups.items()):
        seed_cells = group.get("seed_cells", {})
        seed_entries: list[dict[str, Any]] = []
        identity_errors: list[str] = []
        for seed in sorted(int(value) for value in group.get("expected_seeds", [])):
            item = seed_cells.get(seed)
            if not isinstance(item, Mapping):
                continue
            path = Path(str(item["result_path"])).resolve()
            errors = _result_identity_errors(path, cell=item["plan_cell"], primary_observed=bool(item.get("primary_observed")))
            identity_errors.extend(errors)
            seed_entries.append({
                "seed": seed,
                "cell_id": item.get("cell_id"),
                "result_path": str(path),
                "primary_observed": bool(item.get("primary_observed")),
            })
        cell_payload: dict[str, Any] = {
            "cell_id": group_id,
            "cohort": group["cohort"],
            "classifier": group["classifier"],
            "model": group["model"],
            "modalities": group["modalities"],
            "evaluation_split": "test",
            "seed_results": [] if identity_errors else seed_entries,
            "expected_seeds": list(group.get("expected_seeds", [])),
            "inference_unit": "fixed_classifier_subject_probability_ensemble",
            "conditional_on": "frozen matched pairs and fixed classifier scores",
        }
        if identity_errors:
            cell_payload["audit_errors"] = sorted(set(identity_errors))
        ensemble = aggregate_fixed_seed_subject_ensemble(
            cell_payload,
            base_dir=plan_path.parent if plan_path is not None else REPO_ROOT,
            expected_seeds=group.get("expected_seeds"),
            evaluation_split="test",
        )
        statistics = compute_secondary_ensemble_statistics(
            ensemble,
            bootstrap_replicates=int(bootstrap_replicates),
            swap_replicates=int(swap_replicates),
            seed=73,
        )
        if str(statistics.get("status", "")).upper() == "PASS":
            cell_payload["secondary_p_value"] = statistics.get("secondary_p_value")
            cell_payload["p_value"] = statistics.get("secondary_p_value")
            cell_payload["p_value_source"] = statistics.get("secondary_p_value_source")
        cell_payload["ensemble_statistics"] = _compact_statistics(statistics)
        manifest_cells.append(cell_payload)
        group_audits.append({
            "cell_id": group_id,
            "identity_errors": sorted(set(identity_errors)),
            "ensemble": ensemble,
            "statistics": _compact_statistics(statistics),
        })
    primary_cells: list[dict[str, Any]] = []
    primary_audits: list[dict[str, Any]] = []
    for cell in plan_cells:
        if not _is_primary_cell(cell):
            continue
        cohort = str(cell.get("cohort", cell.get("comparison", ""))).strip().lower()
        root = roots.get(cohort, build / f"stage_a_{cohort}_observed_v3_20260917")
        audit = validate_primary_full_retrained_root(root, plan_cell=cell, expected_replicates=int(primary_replicates))
        primary_audits.append(audit)
        row = {
            "cell_id": str(cell.get("cell_id")),
            "cohort": cohort,
            "classifier": "small_cnn",
            "model": "small_cnn",
            "modalities": list(JOINT_MODALITIES),
            "seed": 73,
            "full_retrained_p_value": audit.get("p_value") if audit.get("status") == "PASS" else None,
            "p_value": audit.get("p_value") if audit.get("status") == "PASS" else None,
            "p_value_source": audit.get("p_value_source") if audit.get("status") == "PASS" else None,
            "integrity_status": audit.get("status"),
            "primary_audit": audit,
        }
        primary_cells.append(row)
    if len(primary_cells) != int(expected_primary_cells):
        plan_errors.append(f"expected {expected_primary_cells} primary cells, observed {len(primary_cells)}")
    manifest_payload = {
        "schema_version": 1,
        "family": "secondary_48",
        "protocol_id": PROTOCOL_ID,
        "matrix_plan_path": str(plan_path) if plan_path is not None else None,
        "cells": manifest_cells,
        "primary_full_retrained": {
            "family": "primary_full_retrained_holm4",
            "cells": primary_cells,
        },
        "fixed_contract": {
            "matrix_cells": int(expected_matrix_cells),
            "secondary_groups": int(expected_secondary_groups),
            "primary_cells": int(expected_primary_cells),
            "neural_seeds": list(EXPECTED_NEURAL_SEEDS),
            "logistic_seeds": list(EXPECTED_LOGISTIC_SEEDS),
            "bootstrap_replicates": int(bootstrap_replicates),
            "swap_replicates": int(swap_replicates),
            "primary_replicates": int(primary_replicates),
        },
    }
    # The report helper is the single Holm implementation.  It sees only the
    # explicit paths above and the namespaced p-value sources produced here.
    report_summary = summarise_secondary_48(
        manifest_payload,
        expected_cells=int(expected_secondary_groups),
        expected_seeds=EXPECTED_NEURAL_SEEDS,
        evaluation_split="test",
    )
    by_group_audit = {str(item["cell_id"]): item for item in group_audits}
    for cell in report_summary.get("cells", []):
        if isinstance(cell, Mapping) and str(cell.get("cell_id")) in by_group_audit:
            cell["statistics"] = by_group_audit[str(cell["cell_id"])].get("statistics")
            cell["audit_errors"] = by_group_audit[str(cell["cell_id"])].get("identity_errors", [])
    secondary_holm = report_summary.get("holm") if isinstance(report_summary.get("holm"), Mapping) else {}
    primary_holm = report_summary.get("primary_full_retrained") if isinstance(report_summary.get("primary_full_retrained"), Mapping) else {}
    status = "PASS" if not plan_errors and secondary_holm.get("status") == "PASS" and primary_holm.get("status") == "PASS" else "INCONCLUSIVE"
    output = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "formal_eligible": status == "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_plan_path": str(plan_path) if plan_path is not None else None,
        "matrix_plan_sha256": _sha256(plan_path) if plan_path is not None and plan_path.is_file() else None,
        "plan_errors": sorted(set(plan_errors)),
        "secondary_48": report_summary,
        "primary_full_retrained_holm4": primary_holm,
        "primary_audits": primary_audits,
        "group_audits": group_audits,
        "fixed_contract": manifest_payload["fixed_contract"],
        "notes": [
            "Secondary p-values are computed from fixed classifier subject-probability ensembles and conditional whole-pair swaps.",
            "Primary Holm-4 accepts only complete generic 199-draw full-retrained pair-null roots with deterministic labels and recorded artifact hashes.",
            "Legacy FOMO calibration observed rows may be descriptive ensemble inputs; their incomplete/legacy null cannot satisfy primary Holm-4.",
            "Missing results remain INCONCLUSIVE; no p-value or adjusted value is fabricated.",
        ],
    }
    if output_root is not None:
        destination = Path(output_root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "secondary_48_manifest.json", manifest_payload)
        _write_json(destination / "summary.json", output)
        source_artifacts: list[dict[str, Any]] = []
        if plan_path is not None:
            source_artifacts.append({"path": str(plan_path), "sha256": _sha256(plan_path), "kind": "matrix_plan"})
        for item in group_audits:
            ensemble = item.get("ensemble") if isinstance(item.get("ensemble"), Mapping) else {}
            for seed_entry in ensemble.get("seed_results", []) if isinstance(ensemble.get("seed_results"), Sequence) else []:
                path = Path(str(seed_entry.get("path", ""))) if seed_entry.get("path") else None
                if path is not None and path.is_file():
                    source_artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path), "kind": "secondary_result"})
        for audit in primary_audits:
            root = Path(str(audit.get("root", "")))
            for relative in ("observed/result.json", "retrained_null/summary.json", "retrained_null/status.json"):
                path = root / relative
                if path.is_file():
                    source_artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path), "kind": "primary_integrity_summary"})
        deduped: dict[str, dict[str, Any]] = {item["path"]: item for item in source_artifacts}
        provenance = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "source_artifacts": [deduped[key] for key in sorted(deduped)],
            "matrix_plan_path": str(plan_path) if plan_path is not None else None,
            "matrix_plan_sha256": output["matrix_plan_sha256"],
        }
        _write_json(destination / "provenance.json", provenance)
        _write_json(
            destination / "audit.json",
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "status": status,
                "plan_errors": sorted(set(plan_errors)),
                "secondary_group_count": len(groups),
                "secondary_holm_status": secondary_holm.get("status"),
                "primary_cell_count": len(primary_cells),
                "primary_holm_status": primary_holm.get("status"),
                "group_audits": group_audits,
                "primary_audits": primary_audits,
            },
        )
    return output


def _parse_primary_roots(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise StatisticsProtocolError("--primary-root must use cohort=PATH")
        cohort, path = value.split("=", 1)
        cohort = cohort.strip().lower()
        if cohort not in COHORTS or not path.strip():
            raise StatisticsProtocolError(f"invalid --primary-root {value!r}")
        output[cohort] = _resolve_path(path, base=REPO_ROOT)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-plan", type=Path, default=None)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_ROOT / "outputs/diagnostics/domain_classifier/model_grid_v3_fullcandidate_20260917_final",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--primary-root", action="append", default=[], metavar="COHORT=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_root = _resolve_path(args.build_root, base=REPO_ROOT)
    matrix_plan = args.matrix_plan or (build_root / "matrix_v3_orchestration" / "matrix_plan.json")
    output_root = args.output_root or (build_root / "secondary_48_statistics_v3")
    try:
        result = aggregate_v3_statistics(
            matrix_plan,
            build_root=build_root,
            output_root=_resolve_path(output_root, base=REPO_ROOT),
            primary_roots=_parse_primary_roots(args.primary_root),
        )
    except (StatisticsProtocolError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_MATRIX_CELLS",
    "EXPECTED_PRIMARY_CELLS",
    "EXPECTED_SECONDARY_GROUPS",
    "StatisticsProtocolError",
    "aggregate_v3_statistics",
    "validate_primary_full_retrained_root",
]
