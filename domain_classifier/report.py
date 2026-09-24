"""Auditable report aggregation for the ANDi domain-classifier experiment.

This module is intentionally a *reader* of experiment artifacts.  It does not
train a model and it never estimates a missing result from a neighbouring
run.  A run may provide per-slice predictions, a metrics JSON/CSV summary, or
both.  Predictions are reduced to one observation per participant before the
primary ROC-AUC, PR-AUC, bootstrap, and gate decisions are made.

The artifact reader is deliberately tolerant of small schema differences
between runner versions.  The output schema is stable and is documented by
``SUMMARY_FIELDS``.  Missing evidence is represented by ``None`` in Python,
an empty field in CSV, and an explicit ``INCONCLUSIVE`` status in Markdown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import (
    aggregate_subject_predictions,
    bootstrap_subject_auc,
    compute_dataset_metrics,
    holm_adjust,
    paired_heldout_swap_test,
)


SUMMARY_FIELDS = [
    "healthy_domain",
    "input_modalities",
    "classifier",
    "seed",
    "train_subjects",
    "val_subjects",
    "test_subjects",
    "train_slices",
    "val_slices",
    "test_slices",
    "slice_roc_auc",
    "subject_roc_auc",
    "subject_pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "permutation_p",
    "tiny_overfit_pass",
    "positive_control_pass",
    "negative_control_pass",
    "label_permutation_pass",
]

# Extra columns are useful for auditability while the first 24 columns retain
# the exact contract requested by the experiment design.
EXTENDED_FIELDS = [
    "run_id",
    "source_path",
    "source_artifact",
    "stage",
    "result_kind",
    "control_mode",
    "formal_final",
    "bce",
    "tp",
    "tn",
    "fp",
    "fn",
    "test_subject_count_observed",
    "subject_direction_invariant_auc",
    "direction_invariant_ci_low",
    "direction_invariant_ci_high",
    "control_gate_status",
    "low_separability_status",
    "evidence_status",
    "bootstrap_n",
    "bootstrap_valid_n",
    "bootstrap_resampling_unit",
    "bootstrap_source",
    "recorded_bootstrap_ci_low",
    "recorded_bootstrap_ci_high",
    "recorded_bootstrap_n",
    "recorded_bootstrap_valid_n",
    "recorded_bootstrap_resampling_unit",
    "permutation_status",
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_ci_low",
    "permutation_null_ci_high",
    "tiny_overfit_accuracy",
    "tiny_overfit_bce",
    "train_final_accuracy",
    "train_final_bce",
    "train_final_subject_accuracy",
    "train_final_subject_bce",
    "positive_control_auc",
    "positive_control_ci_low",
    "negative_control_auc",
    "negative_control_ci_low",
    "negative_control_ci_high",
    "label_permutation_auc",
    "label_permutation_ci_low",
    "label_permutation_ci_high",
]

ALL_FIELDS = SUMMARY_FIELDS + [field for field in EXTENDED_FIELDS if field not in SUMMARY_FIELDS]

CANONICAL_MODALITIES = ("FLAIR", "T1", "T2")

# Scientific conclusions are intentionally restricted to the final model-input
# run.  Controls and intermediate preprocessing stages can still be shown in
# ``summary.csv`` and used to populate the gate columns, but they must never
# become additional cohort/model observations in the nine-question narrative.
# The runner's registered stage vocabulary is ``raw``, ``final`` and
# ``registered``.  Keep this exact for formal conclusions: aliases such as
# ``final_input`` are audit-visible intermediate rows until explicitly mapped
# by the runner manifest.
_FINAL_STAGES = {"final"}
_CONTROL_MARKER_KEYS = (
    "control",
    "is_control",
    "control_type",
    "control_name",
    "control_kind",
    "control_id",
    "run_kind",
    "run_role",
    "result_kind",
    "analysis_kind",
    "evaluation_kind",
    "artifact_role",
    "mode",
    "purpose",
    "task",
    "permutation_mode",
    "permutation_index",
    "shuffle",
    "shuffled_labels",
)
_CONTROL_TOKENS = {
    "control",
    "controls",
    "tiny",
    "overfit",
    "positive",
    "negative",
    "registered",
    "permutation",
    "permutations",
    "shuffle",
    "shuffled",
    "null",
}

# These are source *locations*, not injected results.  ``load_andi_ap_points``
# only returns a point after reading the referenced file and recording its
# hash.  The values can therefore be checked against the current artifacts.
KNOWN_ANDI_AP_PATHS = {
    "FOMO": "outputs/runs/fomo45k_robust_iqr200_continue20_resume2/evaluation/brats21_test50_checkpoints40_20/epoch_0199/inference_metrics_summary.csv",
    "MPI": "outputs/runs/mpi_sri24_robust_iqr60_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv",
    "OASIS3": "outputs/runs/oasis3_sri24_robust_iqr20_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv",
    "Mixed": "outputs/runs/mixed_sri24_robust_iqr20_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv",
}


@dataclass(frozen=True)
class ReportConfig:
    """Reproducibility knobs used by report generation."""

    seed: int = 73
    bootstrap_replicates: int = 2000
    confidence: float = 0.95
    threshold: float = 0.5
    subject_method: str = "mean"
    low_separability_upper: float = 0.60
    positive_auc_threshold: float = 0.80
    negative_ci_low: float = 0.35
    negative_ci_high: float = 0.65
    tiny_accuracy_threshold: float = 0.99
    tiny_bce_threshold: float = 0.02


def _finite(value: Any) -> float | None:
    """Convert a scalar to float, returning ``None`` for missing/non-finite."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy values and NaN/Inf into JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalise_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _normalise_modalities(value: Any) -> str:
    """Return a deterministic ``FLAIR+T1+T2`` style modality label."""

    if value is None or value == "":
        return ""
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[+,/;_ ]+", value) if token]
    elif isinstance(value, Sequence):
        tokens = [str(token) for token in value]
    else:
        tokens = [str(value)]
    canonical = {token.strip().lower(): token.strip().upper() for token in tokens}
    ordered = [name for name in CANONICAL_MODALITIES if name.lower() in canonical]
    # Keep unknown modality labels visible for audit rather than silently
    # treating them as all three channels.
    unknown = sorted(
        {
            item.upper()
            for item in canonical.values()
            if item.upper() not in CANONICAL_MODALITIES
        }
    )
    return "+".join(ordered + unknown)


def _normalise_domain(value: Any, default: str = "unknown") -> str:
    text = _normalise_text(value, default).lower()
    aliases = {
        "fomo45k": "FOMO",
        "fomo": "FOMO",
        "mpi": "MPI",
        "oasis": "OASIS3",
        "oasis3": "OASIS3",
        "mixed": "Mixed",
        "brats": "BraTS21",
        "brats21": "BraTS21",
    }
    return aliases.get(text, default if text == "unknown" else _normalise_text(value, default))


def _infer_domain(value: Any, default: str = "unknown") -> str:
    """Infer cohort only from an explicit metadata/path label, never images."""

    text = _normalise_text(value, "").lower()
    # Mixed is checked first because its path contains mpi/oasis/fomo words.
    for token, domain in (("mixed", "Mixed"), ("oasis3", "OASIS3"), ("oasis", "OASIS3"), ("mpi", "MPI"), ("fomo45k", "FOMO"), ("fomo", "FOMO")):
        if token in text:
            return domain
    return default


def _normalise_classifier(value: Any, default: str = "unknown") -> str:
    text = _normalise_text(value, default)
    aliases = {
        "logreg": "logistic",
        "logistic_regression": "logistic",
        "cnn": "small_cnn",
        "smallcnn": "small_cnn",
        "resnet": "resnet18",
    }
    return aliases.get(text.lower(), text)


def _normalise_seed(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalise_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"pass", "passed", "true", "yes", "ok", "success", "complete"}:
        return True
    if text in {"fail", "failed", "false", "no", "error"}:
        return False
    if text in {"inconclusive", "missing", "unavailable", "unknown", "not_run", "not-run"}:
        return None
    return None


def _status_from_mapping(value: Any) -> bool | None:
    """Read a control pass/fail marker from common nested schemas."""

    if isinstance(value, Mapping):
        for key in ("pass", "passed", "ok", "success", "gate", "status"):
            if key in value:
                status = _normalise_bool(value[key])
                if status is not None or str(value[key]).strip().lower() in {
                    "inconclusive",
                    "missing",
                    "unavailable",
                }:
                    return status
        for key in ("result", "metrics", "control"):
            if key in value:
                status = _status_from_mapping(value[key])
                if status is not None:
                    return status
        return None
    return _normalise_bool(value)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_data_file(path: Path) -> Any:
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            return rows
        return _read_json(path)
    raise ValueError(f"Unsupported report artifact: {path}")


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract row-like objects without flattening metric sub-dictionaries.

    ``DomainClassifierRunner.train_one_seed`` returns a mapping with
    ``test_predictions``/``validation_predictions`` lists and nested metric
    dictionaries.  Those keys are expanded here so a caller can pass the
    result mapping directly to :func:`build_summary_rows`.
    """

    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    prediction_keys = (
        ("train_predictions", "train"),
        ("validation_predictions", "val"),
        ("val_predictions", "val"),
        ("test_predictions", "test"),
        ("heldout_predictions", "test"),
        ("train_final_predictions", "train"),
    )
    expanded: list[dict[str, Any]] = []
    inherited: dict[str, Any] = {}
    for key in (
        "healthy_domain",
        "healthy_cohort",
        "cohort",
        "comparison",
        "input_modalities",
        "modalities",
        "classifier",
        "model",
        "seed",
        "split_seed",
        "stage",
        "run_id",
        "experiment_id",
        "result_kind",
        "formal_final",
        "control_type",
        "control_name",
        "control_kind",
        "run_kind",
        "run_role",
        "purpose",
        "permutation_mode",
        "permutation_index",
    ):
        if key in payload:
            # Runner result JSONs also carry a serialised model state under
            # top-level ``model``.  That state is not classifier identity;
            # prefer the scalar architecture name in ``config.model`` below.
            if key == "model" and isinstance(payload[key], Mapping):
                continue
            inherited[key] = payload[key]
    config = _mapping(payload.get("config"))
    for key in (
        "seed",
        "split_seed",
        "stage",
        "model",
        "input_modalities",
        "modalities",
        "healthy_domain",
        "comparison",
        "result_kind",
        "formal_final",
        "control_type",
        "control_name",
        "control_kind",
        "run_kind",
        "run_role",
        "purpose",
        "permutation_mode",
    ):
        if key not in inherited or (key == "model" and isinstance(inherited.get(key), Mapping)):
            if key in config:
                inherited[key] = config[key]
    # Preserve the runner's recorded held-out subject bootstrap interval.  A
    # report-level recomputation is useful only when this evidence is absent;
    # silently replacing a saved interval changes the scientific result.
    test_statistics = _mapping(payload.get("test_statistics"))
    recorded_bootstrap = _mapping(test_statistics.get("subject_bootstrap"))
    if recorded_bootstrap:
        for source_key, target_key in (
            ("ci_low", "recorded_bootstrap_ci_low"),
            ("ci_high", "recorded_bootstrap_ci_high"),
            ("n_bootstrap", "recorded_bootstrap_n"),
            ("n_valid", "recorded_bootstrap_valid_n"),
            ("resampling_unit", "recorded_bootstrap_resampling_unit"),
        ):
            if source_key in recorded_bootstrap:
                inherited[target_key] = recorded_bootstrap[source_key]
        if "ci_low" in recorded_bootstrap:
            inherited["bootstrap_ci_low"] = recorded_bootstrap["ci_low"]
        if "ci_high" in recorded_bootstrap:
            inherited["bootstrap_ci_high"] = recorded_bootstrap["ci_high"]
        if "n_bootstrap" in recorded_bootstrap:
            inherited["bootstrap_n"] = recorded_bootstrap["n_bootstrap"]
        if "n_valid" in recorded_bootstrap:
            inherited["bootstrap_valid_n"] = recorded_bootstrap["n_valid"]
        if "resampling_unit" in recorded_bootstrap:
            inherited["bootstrap_resampling_unit"] = recorded_bootstrap["resampling_unit"]
    train_final = _mapping(payload.get("train_final"))
    train_final_slice = _mapping(train_final.get("slice"))
    train_final_subject = _mapping(train_final.get("subject"))
    for source_key, target_key in (
        ("accuracy", "train_final_accuracy"),
        ("bce", "train_final_bce"),
    ):
        if source_key in train_final_slice:
            inherited[target_key] = train_final_slice[source_key]
        if source_key in train_final_subject:
            inherited[f"train_final_subject_{source_key}"] = train_final_subject[source_key]
    for key, split in prediction_keys:
        values = payload.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, Mapping):
                    enriched = dict(inherited)
                    enriched.update(dict(item))
                    enriched.setdefault("split", split)
                    expanded.append(enriched)
    if expanded:
        return expanded
    # Summary-only runner results keep metrics under train/validation/test.
    # Preserve the split and inherited identity while exposing the metric
    # fields as one row.  This path is used only when no prediction list was
    # available, so it cannot duplicate a prediction-derived row.
    for key, split in (("train", "train"), ("validation", "val"), ("val", "val"), ("test", "test"), ("heldout", "test")):
        section = payload.get(key)
        if isinstance(section, Mapping) and any(
            name in section
            for name in ("roc_auc", "subject_roc_auc", "average_precision", "subject_pr_auc", "loss")
        ):
            summary = dict(inherited)
            summary.update(dict(section))
            summary["split"] = split
            return [summary]
    for key in ("predictions", "rows", "experiments", "results"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [dict(item) for item in candidate if isinstance(item, Mapping)]
    if any(
        key in payload
        for key in (
            "label",
            "y_true",
            "target",
            "score",
            "probability",
            "subject_roc_auc",
            "slice_roc_auc",
        )
    ):
        return [dict(payload)]
    return []


def _summarise_parity(payload: Any) -> dict[str, Any] | None:
    """Reduce the durable parity audit to concise human-readable evidence."""

    root = _mapping(payload)
    run = _mapping(root.get("latest") or root.get("run"))
    if not run and isinstance(root.get("runs"), list) and root["runs"]:
        run = _mapping(root["runs"][-1])
    if not run:
        return None
    cohorts: list[dict[str, Any]] = []
    for cohort in run.get("cohorts", []):
        if not isinstance(cohort, Mapping):
            continue
        errors: list[float] = []
        reasons: list[str] = []
        for pair in cohort.get("pairs", []):
            if not isinstance(pair, Mapping):
                continue
            for item in pair.get("slices", []):
                if not isinstance(item, Mapping):
                    continue
                if item.get("reason") not in (None, ""):
                    reasons.append(str(item["reason"]))
                checks = _mapping(item.get("checks"))
                for key, value in checks.items():
                    if "max_abs_error" in str(key):
                        number = _finite(value)
                        if number is not None:
                            errors.append(number)
        cohorts.append(
            {
                "cohort": cohort.get("cohort"),
                "status": cohort.get("status", "INCONCLUSIVE"),
                "selected_pairs": cohort.get("selected_pairs"),
                "actual_pair_count": cohort.get("actual_pair_count"),
                "actual_slice_count": cohort.get("actual_slice_count"),
                "pass_pairs": cohort.get("pass_pairs"),
                "incomplete_pairs": cohort.get("incomplete_pairs"),
                "fail_pairs": cohort.get("fail_pairs"),
                "max_abs_error": max(errors) if errors else None,
                "reasons": list(dict.fromkeys(reasons))[:2],
            }
        )
    return {
        "status": run.get("status", "INCONCLUSIVE"),
        "pairs_per_cohort": run.get("pairs_per_cohort"),
        "threads": run.get("threads"),
        "cohorts": cohorts,
    }


def _summarise_stage1(payload: Any) -> dict[str, Any] | None:
    """Reduce Stage1 audit evidence without copying its large inventories."""

    root = _mapping(payload)
    comparisons = _mapping(root.get("comparisons"))
    sources = _mapping(root.get("source_inventories"))
    brats = _mapping(root.get("brats_inventories"))
    comparison = _mapping(comparisons.get("fomo45k"))
    source = _mapping(sources.get("fomo45k"))
    brats_summary = _mapping(brats.get("fomo45k"))
    if not comparison and not source:
        return None
    return {
        "status": root.get("status", "INCONCLUSIVE"),
        "generated_at": root.get("generated_at"),
        "healthy_participants": _mapping(source.get("fomo45k")).get("participants", source.get("participants")),
        "healthy_records": _mapping(source.get("fomo45k")).get("records", source.get("records")),
        "brats_participants": brats_summary.get("participants"),
        "brats_records": brats_summary.get("records"),
        "pair_count": comparison.get("pair_count"),
        "comparison_records": comparison.get("records"),
        "native_tumor_free_records": comparison.get("native_tumor_free_records"),
        "model_mask_false_records": comparison.get("model_mask_false_records"),
    }


def _model_grid_candidate_coverage(payload: Mapping[str, Any], historical_reuse: Mapping[str, Any]) -> dict[str, Any]:
    """Classify candidate-pool coverage independently from reuse equivalence.

    ``selection_equivalence_proof`` answers whether historical controls can be
    reused.  It is not a candidate-inventory check.  Coverage therefore needs
    an explicit inventory/pool audit or scope status.  Missing evidence stays
    audit-pending; only an explicit superseded pretrain-pool marker is a block.
    """

    inventory_keys = (
        "candidate_inventory",
        "candidate_pool_audit",
        "brats_candidate_inventory",
        "source_candidate_inventory",
        "candidate_coverage",
        "pool_audit",
        "full_candidate_coverage",
    )
    inventory_maps = [_mapping(payload.get(key)) for key in inventory_keys if isinstance(payload.get(key), Mapping)]
    explicit_statuses: list[Any] = []
    for key in ("candidate_coverage_status", "candidate_scope_status", "coverage_status", "scope_status"):
        if key in payload:
            explicit_statuses.append(payload.get(key))
    # The v3 builder may emit this as either a boolean contract flag or a
    # nested audit mapping.  It is independent of historical selection
    # equivalence and is the authoritative full-pool coverage evidence.
    full_candidate_coverage = payload.get("full_candidate_coverage")
    if isinstance(full_candidate_coverage, bool):
        explicit_statuses.append("PASS" if full_candidate_coverage else "AUDIT_PENDING")
    for inventory in inventory_maps:
        for key in ("status", "candidate_coverage_status", "candidate_scope_status", "coverage_status", "scope_status"):
            if key in inventory:
                explicit_statuses.append(inventory.get(key))

    def status_text(value: Any) -> str:
        return str(value).strip().upper().replace("-", "_").replace(" ", "_")

    bug_fields = (
        "old_candidate_source",
        "candidate_source",
        "source_selection",
        "candidate_pool_source",
        "pool_source",
        "old_pool_attempt_status",
        "pretrain_pool_status",
        "selection_status",
    )
    bug_values: list[str] = []
    for mapping in [payload, *inventory_maps, historical_reuse]:
        for key in bug_fields:
            value = mapping.get(key)
            if value in (None, "") or value is False:
                continue
            if isinstance(value, bool):
                if value:
                    bug_values.append(f"{key}=true")
                continue
            text = status_text(value)
            if "SUPERSEDED_PRETRAIN_SELECTION_BUG" in text or (
                any(token in text for token in ("OLD", "HISTORICAL", "PRETRAIN"))
                and any(token in text for token in ("POOL", "CANDIDATE", "SELECTION", "CACHE"))
            ):
                bug_values.append(f"{key}={value}")
    for value in explicit_statuses:
        if "SUPERSEDED_PRETRAIN_SELECTION_BUG" in status_text(value):
            bug_values.append(str(value))

    verified_values: list[bool] = []
    verified_keys = (
        "source_full_inventory_verified",
        "full_inventory_verified",
        "candidate_pool_verified",
        "source_inventory_verified",
        "pool_verified",
        "verified",
        "full_candidate_coverage",
        "full_candidate_pool_verified",
    )
    count_keys = (
        "source_csv_count",
        "csv_count",
        "source_map_count",
        "map_count",
        "source_processed_count",
        "processed_count",
        "candidate_count",
        "candidate_pool_count",
        "pool_count",
        "candidate_rows",
        "csv_rows",
        "processed_subjects",
    )
    inventory_counts: dict[str, Any] = {}
    for mapping in [payload, *inventory_maps]:
        for key in verified_keys:
            if key in mapping:
                parsed = _normalise_bool(mapping.get(key))
                if parsed is not None:
                    verified_values.append(parsed)
        for key in count_keys:
            if key in mapping and mapping.get(key) not in (None, ""):
                inventory_counts[key] = mapping.get(key)

    if bug_values:
        status = "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG"
        basis = "explicit old/superseded pretrain candidate-pool source or status"
    else:
        normalized_statuses = [status_text(value) for value in explicit_statuses if value not in (None, "")]
        status_is_pass = any(
            value in {"PASS", "PASSED", "VERIFIED", "COMPLETE", "COVERAGE_PASS", "CANDIDATE_COVERAGE_VERIFIED"}
            or "COVERAGE_VERIFIED" in value
            for value in normalized_statuses
        )
        status_is_pending = any(
            value in {"UNKNOWN", "INCONCLUSIVE", "AUDIT_PENDING", "PENDING", "DESIGN_AUDIT_PENDING"}
            or "PENDING" in value
            for value in normalized_statuses
        )
        if status_is_pass or any(verified_values):
            status = "PASS"
            basis = "explicit candidate inventory/pool verification"
        elif status_is_pending or any(value is False for value in verified_values):
            status = "AUDIT_PENDING"
            basis = "candidate inventory/pool scope is explicit but not verified"
        else:
            status = "UNKNOWN"
            basis = "no explicit candidate inventory/pool coverage evidence"
    return {
        "candidate_coverage_status": status,
        "candidate_coverage_basis": basis,
        "source_full_inventory_verified": True if any(verified_values) else False if verified_values else None,
        "candidate_inventory_counts": inventory_counts,
        "old_candidate_source_markers": bug_values,
    }


def _summarise_model_grid_v3_audit(path: Path) -> dict[str, Any]:
    """Read compact v3 audit facts without treating them as model results.

    The v3 build audit can be several megabytes because it retains selected
    record metadata.  Reports only need the contract status, pair count, and
    source-composition counts.  In particular, ``canonical_parity=PASS`` is
    kept as a contract-level status; it is never promoted to exhaustive
    tensor parity evidence.
    """

    result: dict[str, Any] = {
        "audit_path": str(path.resolve()),
        "audit_status": "INCONCLUSIVE",
        "canonical_parity_status": "INCONCLUSIVE",
        "canonical_parity_scope": "contract-level metadata/shape/dtype/normalization checks; exhaustive tensor parity not established",
        "pair_z_histogram_status": "INCONCLUSIVE",
        "pair_count": None,
        "source_composition": {},
        "mixed_tensor_parity": {},
        "selected_brats_tensor_cache": {},
        "standalone_map_consistency": {},
        "mixed_source_join": {},
        "participant_map_info": {},
        "historical_control_reuse": {},
        "selection_equivalence_proof": None,
        "controls_reuse_status": "UNKNOWN",
        "candidate_coverage_status": "UNKNOWN",
        "candidate_coverage_basis": "no explicit candidate inventory/pool coverage evidence",
        "source_full_inventory_verified": None,
        "candidate_inventory_counts": {},
        "old_candidate_source_markers": [],
    }
    try:
        payload = _mapping(_read_json(path))
    except (OSError, ValueError, json.JSONDecodeError):
        result["audit_status"] = "UNREADABLE"
        return result
    canonical = _mapping(payload.get("canonical_parity"))
    pair_histogram = _mapping(payload.get("pair_z_histogram"))
    historical_reuse = _mapping(payload.get("historical_control_reuse"))
    mixed_tensor_parity = _mapping(payload.get("mixed_tensor_parity"))
    selected_brats_tensor_cache = _mapping(payload.get("brats_selected_tensor_cache"))
    standalone_map_consistency = _mapping(payload.get("standalone_map_consistency"))
    mixed_source_join = _mapping(payload.get("mixed_source_join"))
    canonical_status = str(canonical.get("status", "INCONCLUSIVE")).upper()
    pair_status = str(pair_histogram.get("status", "INCONCLUSIVE")).upper()
    historical_summary = {
        key: historical_reuse.get(key)
        for key in ("brats_candidate_tensors", "old_fomo_pair_assignments_reused", "results_reused", "selection_equivalence_proof")
        if key in historical_reuse
    }
    selection_equivalence = _normalise_bool(historical_reuse.get("selection_equivalence_proof"))
    result.update(
        {
            "canonical_parity_status": canonical_status,
            "pair_z_histogram_status": pair_status,
            "pair_count": pair_histogram.get("pair_count"),
            "audit_status": "PASS" if canonical_status == "PASS" and pair_status == "PASS" else "INCONCLUSIVE",
            "historical_control_reuse": historical_summary,
            "selection_equivalence_proof": selection_equivalence,
            "controls_reuse_status": "PASS" if selection_equivalence is True else "NOT_ALLOWED" if selection_equivalence is False else "UNKNOWN",
            "mixed_tensor_parity": {
                key: mixed_tensor_parity.get(key)
                for key in ("status", "selected_records", "checked_records", "mismatch_count", "aggregate_sha256")
                if key in mixed_tensor_parity
            },
            "selected_brats_tensor_cache": {
                key: selected_brats_tensor_cache.get(key)
                for key in ("status", "participants", "records", "cache_file_count", "mismatch_count")
                if key in selected_brats_tensor_cache
            },
            "standalone_map_consistency": {
                key: standalone_map_consistency.get(key)
                for key in ("status", "failure_count", "conflict_count")
                if key in standalone_map_consistency
            },
            "mixed_source_join": {
                key: mixed_source_join.get(key)
                for key in ("status", "failure_count", "conflict_count")
                if key in mixed_source_join
            },
        }
    )
    result.update(_model_grid_candidate_coverage(payload, historical_reuse))
    composition = _mapping(_mapping(payload.get("eligibility_composition")).get("by_underlying_cohort_split"))
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for key, item in composition.items():
        if not isinstance(item, Mapping):
            continue
        cohort, separator, split = str(key).partition(":")
        if not separator or cohort.lower() == "brats21":
            continue
        selected.setdefault(cohort, {})[split] = {
            "eligible_participants": item.get("eligible_participants"),
            "selected_participants": item.get("selected_participants"),
            "selected_records": item.get("selected_records"),
        }
    result["source_composition"] = selected
    participant_map = _mapping(payload.get("participant_map_info"))
    result["participant_map_info"] = {
        key: participant_map.get(key)
        for key in ("participant_count", "preserved_count", "extended_count", "seed", "split_counts")
        if key in participant_map
    }
    return result


def _summarise_model_grid_v3_sidecars(build_path: Path) -> dict[str, Any]:
    """Read small v3 execution sidecars without opening candidate JSONL rows.

    The builder and the later Stage-A audit write several immutable sidecars
    beside ``build_summary.json``.  They describe build/execution state, not
    classifier fits.  Keep only the fields needed for an interim report and
    retain their paths so :func:`build_report` fingerprints the exact revision
    that was read.
    """

    revision_root = build_path.parent
    sidecar_names = (
        "protocol.json",
        "source_fingerprints.json",
        "build_timing.json",
        "execution.json",
        "stage_a_commands.json",
        "stage_a_preflight.json",
        "training_protocol.json",
        "training_protocol_amendment_20260917.json",
        "training_protocol_manifest.json",
        "training_protocol_amendment_manifest.json",
        "final_external_audit.json",
        "final_external_audit_v2.json",
        "brats_candidate_inventory.json",
        "producer_code_provenance_audit.json",
    )
    payloads: dict[str, Mapping[str, Any]] = {}
    paths: list[str] = []
    parse_warnings: list[str] = []
    for name in sidecar_names:
        path = revision_root / name
        if not path.is_file():
            continue
        # The candidate inventory is compact metadata plus a subject audit;
        # do not load the 69 MB ``brats_candidate_rows.jsonl`` artifact here.
        try:
            payload = _mapping(_read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            # One runtime writer left a literal two-character ``\\n`` suffix
            # after an otherwise complete JSON object.  Recover that exact
            # serialization defect while retaining a visible warning; this
            # does not relax parsing of arbitrary trailing content.
            try:
                raw = path.read_text(encoding="utf-8")
                if raw.endswith("\\n"):
                    payload = _mapping(json.loads(raw[:-2]))
                    parse_warnings.append(f"{name}: stripped literal trailing \\n")
                else:
                    payload = {}
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
        payloads[name] = payload
        paths.append(str(path.resolve()))

    stage = payloads.get("stage_a_commands.json", {})
    training = payloads.get("training_protocol.json", {})
    # Prefer the newest explicitly versioned external audit.  A superseded
    # v1 may remain beside a corrected v2; selecting by filesystem order of
    # glob results would make the report depend on directory enumeration.
    external_names = [
        name for name in ("final_external_audit.json", "final_external_audit_v2.json")
        if name in payloads and (revision_root / name).is_file()
    ]
    external_name = max(
        external_names,
        key=lambda name: ((revision_root / name).stat().st_mtime_ns, name),
        default=None,
    )
    external = payloads.get(external_name, {}) if external_name else {}
    historical_external: dict[str, dict[str, Any]] = {}
    for name in external_names:
        if name == external_name:
            continue
        value = payloads.get(name, {})
        historical_external[name] = {
            "path": str((revision_root / name).resolve()),
            "status": value.get("status", "INCONCLUSIVE"),
            "failures": value.get("failures", []),
            "classification": "SUPERSEDED",
        }
    registered = _mapping(external.get("registered_materialization"))
    execution = payloads.get("execution.json", {})

    tiny_gate_status = "NOT_RECORDED_AS_OF_REPORT_GENERATION"
    tiny_gate_path: str | None = None
    tiny_root = revision_root / "stage_a_fomo45k" / "tiny"
    # This deliberately checks named gate files only.  It never recursively
    # scans prediction/checkpoint trees and therefore cannot mistake a partial
    # training artifact for a completed gate.
    for candidate in (
        tiny_root / "gate.json",
        tiny_root / "tiny_gate.json",
        tiny_root / "tiny" / "gate.json",
        tiny_root / "tiny" / "tiny_gate.json",
        revision_root / "stage_a_fomo45k" / "tiny_gate.json",
    ):
        if not candidate.is_file():
            continue
        tiny_gate_path = str(candidate.resolve())
        try:
            tiny_payload = _mapping(_read_json(candidate))
        except (OSError, ValueError, json.JSONDecodeError):
            tiny_payload = {}
        gate = _mapping(tiny_payload.get("gate"))
        tiny_gate_status = str(gate.get("status", tiny_payload.get("status", "RECORDED"))).upper()
        break

    control_specs = {
        "tiny_overfit": revision_root / "stage_a_fomo45k" / "tiny" / "tiny" / "gate.json",
        "registered_positive": revision_root / "stage_a_fomo45k_acl_retry_20260917" / "positive" / "gate.json",
        "same_cohort_negative": revision_root / "stage_a_fomo45k_negative_v3_20260917" / "negative" / "gate.json",
    }
    control_summaries: dict[str, dict[str, Any]] = {}
    for control_name, control_path in control_specs.items():
        if not control_path.is_file():
            continue
        paths.append(str(control_path.resolve()))
        try:
            control_payload = _mapping(_read_json(control_path))
        except (OSError, ValueError, json.JSONDecodeError):
            control_payload = {}
        gate = _mapping(control_payload.get("gate"))
        compact_rows: list[dict[str, Any]] = []
        raw_rows = gate.get("rows")
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, Mapping):
                    continue
                compact_rows.append(
                    {
                        key: row.get(key)
                        for key in (
                            "seed",
                            "accuracy",
                            "bce",
                            "subject_auc",
                            "bootstrap_ci_low",
                            "bootstrap_ci_high",
                            "subject_auc_ci",
                            "ci_lower",
                            "ci_upper",
                            "paired_swap_p_value",
                            "holm_adjusted_p_value",
                            "passed",
                        )
                        if key in row
                    }
                )
        control_summaries[control_name] = {
            "path": str(control_path.resolve()),
            "mode": control_payload.get("mode"),
            "status": str(gate.get("status", control_payload.get("status", "INCONCLUSIVE"))).upper(),
            "rows": compact_rows,
            "metadata": {
                key: gate.get(key)
                for key in ("name", "accuracy_threshold", "bce_threshold", "auc_threshold", "ci_lower_threshold", "n_seeds")
                if key in gate
            },
        }

    failure_reasons: list[str] = []
    raw_failures = external.get("failures")
    if isinstance(raw_failures, list):
        for item in raw_failures:
            if isinstance(item, Mapping):
                reason = item.get("reason") or item.get("detail") or item.get("status")
                if reason not in (None, ""):
                    failure_reasons.append(str(reason))
            elif item not in (None, ""):
                failure_reasons.append(str(item))

    materialization: dict[str, Any] = {
        "status": registered.get("status", "INCONCLUSIVE"),
        "protocol": registered.get("protocol"),
        "manifest_root": registered.get("manifest_root"),
        "total_records": registered.get("total_records"),
        "manifest_sha256": _mapping(registered.get("manifest_sha256")),
        "failure_count": sum(
            int(_finite(_mapping(value).get("failure_count")) or 0)
            for value in _mapping(registered.get("split_audits")).values()
            if isinstance(value, Mapping)
        ),
    }
    # The v3 Stage-A observed fit lives below the immutable build revision.
    # Read this one explicitly named path so it stays separate from the older
    # FOMO calibration directory that may be passed as another report source.
    v3_observed_root = revision_root / "stage_a_fomo45k_observed_v3_20260917"
    v3_observed = _summarise_calibration(
        v3_observed_root,
        classification="V3_STAGE_A_OBSERVED_EXCLUDED_FROM_CONFIRMATORY",
    )
    if v3_observed:
        paths.extend(
            str((v3_observed_root / relative).resolve())
            for relative in v3_observed.get("references", {})
        )
    explicit_mixed_review = _mapping(external.get("independent_full_mixed_review"))
    mixed_comparison = _mapping(_mapping(external.get("comparisons")).get("mixed"))
    mixed_review_status = str(explicit_mixed_review.get("status", "")).upper()
    if mixed_review_status in {"", "PENDING", "INCONCLUSIVE", "UNKNOWN"}:
        if (
            str(mixed_comparison.get("status", "")).upper() == "PASS"
            and str(mixed_comparison.get("mixed_tensor_parity", "")).upper() == "PASS"
        ):
            mixed_review_status = "PASS"
        else:
            mixed_review_status = "PENDING"

    return {
        "sidecar_paths": paths,
        "sidecar_parse_warnings": parse_warnings,
        "execution_status": execution.get("status"),
        "stage_a_status": stage.get("status", "NOT_RECORDED"),
        "stage_a_training_started": stage.get("training_started"),
        "tiny_status": stage.get("status", "NOT_RECORDED"),
        "tiny_gate_status": tiny_gate_status,
        "tiny_gate_path": tiny_gate_path,
        "control_summaries": control_summaries,
        "training_protocol_status": training.get("status", "NOT_RECORDED"),
        "training_protocol_training_started": training.get("training_started"),
        "formal_fit_count": 0,
        "v3_observed": v3_observed,
        "v3_observed_status": "COMPLETE" if v3_observed else "NOT_RECORDED",
        "v3_observed_null_status": _mapping(v3_observed.get("full_retrained_null")).get("status", "NOT_RECORDED") if v3_observed else "NOT_RECORDED",
        "v3_observed_null_completed": _mapping(v3_observed.get("full_retrained_null")).get("completed", 0) if v3_observed else 0,
        "v3_observed_null_requested": _mapping(v3_observed.get("full_retrained_null")).get("requested") if v3_observed else None,
        "v3_primary_observed_fit_count": 1 if v3_observed else 0,
        "v3_confirmatory_fit_count": 0,
        "v3_stage_a_gate_status": _v3_stage_a_gate_status(
            {"control_summaries": control_summaries, "v3_observed": v3_observed}
        ),
        "independent_full_mixed_review_status": mixed_review_status,
        "independent_full_mixed_review": {
            "status": mixed_review_status,
            "source_audit": str((revision_root / "audits" / "mixed.json").resolve()) if (revision_root / "audits" / "mixed.json").is_file() else None,
            "external_comparison": {
                key: mixed_comparison.get(key)
                for key in ("status", "records", "mixed_tensor_parity", "brats_selected_tensor_cache", "source_split_local_joins", "support_eligibility", "z_norm")
                if key in mixed_comparison
            },
        },
        "selected_external_audit": {
            "name": external_name,
            "path": str((revision_root / external_name).resolve()) if external_name else None,
            "status": external.get("status", "INCONCLUSIVE"),
        },
        "historical_external_audits": historical_external,
        "registered_materialization": materialization,
        "external_audit_status": external.get("status", "INCONCLUSIVE"),
        "external_audit_failure_reasons": failure_reasons,
    }


def _summarise_model_grid_v3(payload: Any, build_path: Path) -> dict[str, Any] | None:
    """Summarise a v3 model-grid build as build-only provenance.

    ``build_summary.json`` contains balanced record counts and references to
    per-cohort audits, but no classifier predictions.  Keep it in a dedicated
    context section so it cannot create formal rows or answer the nine
    scientific questions prematurely.
    """

    root = _mapping(payload)
    comparisons = _mapping(root.get("comparisons"))
    # ``build_summary.full_candidate_coverage`` is the authoritative full
    # inventory evidence for the final revision.  Per-comparison audits from
    # older builder revisions may omit this field; propagate the explicit
    # build-level evidence without treating selection equivalence as coverage.
    build_candidate_coverage = _model_grid_candidate_coverage(root, {})
    compact: list[dict[str, Any]] = []
    audit_paths: list[str] = []
    for cohort, item in comparisons.items():
        if not isinstance(item, Mapping):
            continue
        audit_value = item.get("audit")
        audit_path = Path(str(audit_value)) if audit_value not in (None, "") else None
        if audit_path is not None and not audit_path.is_absolute():
            audit_path = build_path.parent / audit_path
        audit_summary = (
            _summarise_model_grid_v3_audit(audit_path)
            if audit_path is not None and audit_path.is_file()
            else {
                "audit_path": str(audit_path) if audit_path is not None else None,
                "audit_status": "MISSING",
                "canonical_parity_status": "INCONCLUSIVE",
                "canonical_parity_scope": "contract-level metadata/shape/dtype/normalization checks; exhaustive tensor parity not established",
                "pair_z_histogram_status": "INCONCLUSIVE",
                "pair_count": None,
                "source_composition": {},
                "participant_map_info": {},
                "historical_control_reuse": {},
                "selection_equivalence_proof": None,
                "controls_reuse_status": "UNKNOWN",
                "candidate_coverage_status": "UNKNOWN",
                "candidate_coverage_basis": "no explicit candidate inventory/pool coverage evidence",
                "source_full_inventory_verified": None,
                "candidate_inventory_counts": {},
                "old_candidate_source_markers": [],
            }
        )
        if audit_path is not None:
            audit_paths.append(str(audit_path.resolve()))
        audit_coverage_status = str(audit_summary.get("candidate_coverage_status", "UNKNOWN"))
        if (
            build_candidate_coverage.get("candidate_coverage_status") == "PASS"
            and audit_coverage_status not in {"BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG"}
        ):
            audit_summary.update(
                {
                    "candidate_coverage_status": "PASS",
                    "candidate_coverage_basis": "build_summary.full_candidate_coverage",
                    "source_full_inventory_verified": build_candidate_coverage.get("source_full_inventory_verified"),
                    "candidate_inventory_counts": dict(build_candidate_coverage.get("candidate_inventory_counts", {})),
                }
            )
        compact.append(
            {
                "cohort": cohort,
                "status": item.get("status", "INCONCLUSIVE"),
                "records": item.get("records"),
                "domain_counts": _mapping(item.get("domain_counts")),
                "split_counts": _mapping(item.get("split_counts")),
                "audit": audit_summary,
            }
        )
    candidate_statuses = [
        str(_mapping(item.get("audit")).get("candidate_coverage_status", "UNKNOWN"))
        for item in compact
    ]
    candidate_coverage_status = (
        "PASS"
        if candidate_statuses and all(status == "PASS" for status in candidate_statuses)
        else "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG"
        if any(status == "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG" for status in candidate_statuses)
        else "AUDIT_PENDING"
    )
    scope_status = (
        "CANDIDATE_COVERAGE_VERIFIED_BUILD_ONLY"
        if candidate_coverage_status == "PASS"
        else "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG"
        if candidate_coverage_status == "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG"
        else "DESIGN_AUDIT_PENDING"
    )
    sidecars = _summarise_model_grid_v3_sidecars(build_path)
    mixed_item = next(
        (item for item in compact if str(item.get("cohort", "")).lower() == "mixed"),
        None,
    )
    mixed_audit = _mapping(mixed_item.get("audit")) if isinstance(mixed_item, Mapping) else {}
    mixed_parity = _mapping(mixed_audit.get("mixed_tensor_parity"))
    mixed_cache = _mapping(mixed_audit.get("selected_brats_tensor_cache"))
    mixed_map = _mapping(mixed_audit.get("standalone_map_consistency"))
    mixed_join = _mapping(mixed_audit.get("mixed_source_join"))
    mixed_review_evidence = {
        "status": "PASS"
        if (
            sidecars.get("external_audit_status", "").upper() == "PASS"
            and str(_mapping(sidecars.get("selected_external_audit")).get("name", "")).endswith("v2.json")
            and str(mixed_item.get("status", "")).upper() == "PASS" if isinstance(mixed_item, Mapping) else False
        )
        else "INCONCLUSIVE",
        "exact_tensor_records": mixed_parity.get("checked_records"),
        "tensor_mismatch_count": mixed_parity.get("mismatch_count"),
        "selected_cache_file_count": mixed_cache.get("cache_file_count"),
        "selected_cache_records": mixed_cache.get("records"),
        "map_failure_count": mixed_map.get("failure_count"),
        "source_join_status": mixed_join.get("status"),
        "source_audit_path": mixed_audit.get("audit_path"),
    }
    # The standalone audit carries the exhaustive Mixed tensor/map evidence;
    # preserve its compact counts while keeping the contract-level parity
    # statement above distinct from an unverified model-input claim.
    if (
        mixed_review_evidence["status"] == "PASS"
        and str(mixed_parity.get("status", "")).upper() == "PASS"
        and int(mixed_parity.get("mismatch_count") or 0) == 0
        and str(mixed_cache.get("status", "")).upper() == "PASS"
        and int(mixed_map.get("failure_count") or 0) == 0
        and str(mixed_join.get("status", "")).upper() == "PASS"
    ):
        sidecars["independent_full_mixed_review_status"] = "PASS"
        mixed_review_evidence["status"] = "PASS"
    elif sidecars.get("independent_full_mixed_review_status") == "PASS":
        # Retain an explicitly recorded external PASS even when the detailed
        # audit sidecar is unavailable, but expose that the compact evidence
        # could not be re-derived here.
        mixed_review_evidence["status"] = "PASS"
    sidecars["independent_full_mixed_review"] = mixed_review_evidence
    return {
        "classification": "BUILD_ONLY",
        "schema_version": root.get("schema_version"),
        "status": root.get("status", "INCONCLUSIVE"),
        "build_path": str(build_path.resolve()),
        "created_at_utc": root.get("created_at_utc"),
        "completed_at_utc": root.get("completed_at_utc"),
        "elapsed_seconds": root.get("elapsed_seconds"),
        "training_started": root.get("training_started"),
        "no_training": root.get("no_training"),
        "scope_status": scope_status,
        "candidate_coverage_status": candidate_coverage_status,
        "candidate_coverage_basis": build_candidate_coverage.get("candidate_coverage_basis"),
        "source_full_inventory_verified": build_candidate_coverage.get("source_full_inventory_verified"),
        "candidate_inventory_counts": build_candidate_coverage.get("candidate_inventory_counts", {}),
        "tensor_parity_claim": "NOT_ESTABLISHED_BY_BUILD_SUMMARY; canonical parity is contract-level only",
        "comparisons": compact,
        "audit_paths": audit_paths,
        **sidecars,
    }


ROSTER_FIELDS = [
    "comparison",
    "cohort",
    "split",
    "participant_id",
    "label",
    "domain",
    "pair_ids",
    "pair_count",
    "selected_slice_count",
    "selected_zs",
    "selected_z_bins",
    "case_ids",
    "session_ids",
    "source_datasets",
    "source_splits",
    "source_keys",
    "underlying_source_datasets",
    "underlying_source_splits",
    "underlying_source_keys",
    "source_lmdb_paths",
]


def _json_list(values: Iterable[Any]) -> str:
    """Encode a deterministic compact list for a subject-roster CSV cell."""

    return json.dumps(sorted({str(value) for value in values if value not in (None, "")}), ensure_ascii=False, separators=(",", ":"))


def _roster_row_key(row: Mapping[str, Any]) -> str:
    value = _first(row, "participant_id", "subject_id", "participant", "subject", "patient_id")
    return "" if value in (None, "") else str(value)


def export_model_grid_v3_rosters(
    build_root: str | Path,
    output_dir: str | Path,
    *,
    comparisons: Sequence[str] = ("fomo45k", "mpi", "oasis3", "mixed"),
    splits: Sequence[str] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Export participant-level v3 rosters from frozen manifests.

    The function reads only the explicitly named ``manifests/<comparison>/<split>.jsonl``
    files.  It writes subject-level CSVs and compact audit JSON into a separate
    output directory and refuses an output directory inside the frozen build
    root.  ``selected_slice_count`` remains a selected-record count; it does
    not imply a subject was healthy throughout its full volume.
    """

    build_path = Path(build_root).resolve()
    destination = Path(output_dir).resolve()
    if destination == build_path or build_path in destination.parents:
        raise ValueError("subject-roster output must be outside the frozen model-grid build root")
    requested_comparisons = tuple(str(value) for value in comparisons)
    requested_splits = tuple(str(value) for value in splits)
    source_paths: list[Path] = []
    missing: list[str] = []
    for comparison in requested_comparisons:
        for split in requested_splits:
            path = build_path / "manifests" / comparison / f"{split}.jsonl"
            if path.is_file():
                source_paths.append(path)
            else:
                missing.append(str(path))
    if missing:
        return {
            "status": "INCONCLUSIVE_MISSING_MANIFESTS",
            "build_root": str(build_path),
            "output_dir": str(destination),
            "missing_source_files": missing,
            "method": "explicit frozen manifests only; no recursive discovery",
        }

    destination.mkdir(parents=True, exist_ok=True)
    by_comparison_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    errors: list[dict[str, Any]] = []
    source_fingerprints = [_fingerprint(path) for path in source_paths]
    for comparison in requested_comparisons:
        by_comparison_split[comparison] = {}
        for split in requested_splits:
            path = build_path / "manifests" / comparison / f"{split}.jsonl"
            try:
                payload = _read_data_file(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"comparison": comparison, "split": split, "path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
                by_comparison_split[comparison][split] = []
                continue
            if not isinstance(payload, list):
                errors.append({"comparison": comparison, "split": split, "path": str(path), "reason": "manifest is not JSONL object rows"})
                by_comparison_split[comparison][split] = []
                continue
            by_comparison_split[comparison][split] = [dict(row) for row in payload if isinstance(row, Mapping)]
            if len(by_comparison_split[comparison][split]) != len(payload):
                errors.append({"comparison": comparison, "split": split, "path": str(path), "reason": "one or more manifest lines are not objects"})

    all_roster_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in requested_splits}
    comparison_summaries: dict[str, Any] = {}
    for comparison in requested_comparisons:
        participant_split: dict[str, str] = {}
        participant_pairs: dict[str, set[str]] = defaultdict(set)
        participant_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        pair_members: dict[tuple[str, str], dict[str, set[Any]]] = defaultdict(lambda: {"participants": set(), "labels": set(), "domains": set()})
        split_records: dict[str, int] = {}
        split_rosters: dict[str, list[dict[str, Any]]] = {}
        for split in requested_splits:
            aggregate: dict[str, dict[str, Any]] = {}
            rows = by_comparison_split[comparison].get(split, [])
            split_records[split] = len(rows)
            for row in rows:
                participant = _roster_row_key(row)
                if not participant:
                    errors.append({"comparison": comparison, "split": split, "reason": "manifest row has no participant_id"})
                    continue
                label_value = row.get("label")
                try:
                    label = int(label_value)
                except (TypeError, ValueError):
                    errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": "manifest row has non-integer label"})
                    continue
                if label not in (0, 1):
                    errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": "manifest row label is not 0/1"})
                    continue
                prior_split = participant_split.setdefault(participant, split)
                if prior_split != split:
                    errors.append({"comparison": comparison, "participant_id": participant, "reason": "participant occurs in multiple splits", "splits": sorted({prior_split, split})})
                item = aggregate.setdefault(
                    participant,
                    {
                        "participant_id": participant,
                        "labels": set(),
                        "domains": set(),
                        "pair_ids": set(),
                        "zs": set(),
                        "z_bins": set(),
                        "case_ids": set(),
                        "session_ids": set(),
                        "source_datasets": set(),
                        "source_splits": set(),
                        "source_keys": set(),
                        "underlying_source_datasets": set(),
                        "underlying_source_splits": set(),
                        "underlying_source_keys": set(),
                        "source_lmdb_paths": set(),
                        "record_count": 0,
                    },
                )
                metadata = _mapping(row.get("metadata"))
                provenance = _mapping(row.get("provenance"))
                pair_id = str(row.get("pair_id") or "")
                local_dataset = str(row.get("source_dataset") or row.get("domain") or comparison)
                local_split = str(row.get("source_split") or "")
                local_key = str(row.get("source_key") or "")
                underlying_dataset = str(metadata.get("underlying_source_dataset") or local_dataset)
                underlying_split = str(metadata.get("underlying_source_split") or local_split)
                underlying_key = str(metadata.get("underlying_source_key") or local_key)
                lmdb_path = str(metadata.get("lmdb_path") or provenance.get("source_lmdb_path") or "")
                item["labels"].add(label)
                item["domains"].add(str(row.get("domain") or comparison))
                if pair_id:
                    item["pair_ids"].add(pair_id)
                    participant_pairs[participant].add(pair_id)
                    member = pair_members[(split, pair_id)]
                    member["participants"].add(participant)
                    member["labels"].add(label)
                    member["domains"].add(str(row.get("domain") or comparison))
                else:
                    errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": "manifest row has no pair_id"})
                for field, target in (("z", item["zs"]), ("z_bin", item["z_bins"])):
                    raw_value = row.get(field)
                    if raw_value in (None, ""):
                        continue
                    try:
                        target.add(int(raw_value))
                    except (TypeError, ValueError):
                        errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": f"manifest row has non-integer {field}"})
                item["case_ids"].add(str(row.get("case_id") or ""))
                item["session_ids"].add(str(row.get("session_id") or ""))
                item["source_datasets"].add(local_dataset)
                item["source_splits"].add(local_split)
                item["source_keys"].add(local_key)
                item["underlying_source_datasets"].add(underlying_dataset)
                item["underlying_source_splits"].add(underlying_split)
                item["underlying_source_keys"].add(underlying_key)
                if lmdb_path:
                    item["source_lmdb_paths"].add(lmdb_path)
                item["record_count"] += 1
            roster_rows: list[dict[str, Any]] = []
            for participant in sorted(aggregate):
                item = aggregate[participant]
                if len(item["labels"]) != 1:
                    errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": "participant has mixed labels"})
                if len(participant_pairs.get(participant, set())) > 1:
                    errors.append({"comparison": comparison, "split": split, "participant_id": participant, "reason": "participant has multiple pair_ids", "pair_ids": sorted(participant_pairs[participant])})
                row_out = {
                    "comparison": comparison,
                    "cohort": comparison,
                    "split": split,
                    "participant_id": participant,
                    "label": next(iter(item["labels"])) if len(item["labels"]) == 1 else "",
                    "domain": "+".join(sorted(item["domains"])),
                    "pair_ids": _json_list(item["pair_ids"]),
                    "pair_count": len(item["pair_ids"]),
                    "selected_slice_count": item["record_count"],
                    "selected_zs": _json_list(item["zs"]),
                    "selected_z_bins": _json_list(item["z_bins"]),
                    "case_ids": _json_list(item["case_ids"]),
                    "session_ids": _json_list(item["session_ids"]),
                    "source_datasets": _json_list(item["source_datasets"]),
                    "source_splits": _json_list(item["source_splits"]),
                    "source_keys": _json_list(item["source_keys"]),
                    "underlying_source_datasets": _json_list(item["underlying_source_datasets"]),
                    "underlying_source_splits": _json_list(item["underlying_source_splits"]),
                    "underlying_source_keys": _json_list(item["underlying_source_keys"]),
                    "source_lmdb_paths": _json_list(item["source_lmdb_paths"]),
                }
                roster_rows.append(row_out)
                all_roster_rows[split].append(row_out)
            split_rosters[split] = roster_rows

        participant_by_split: dict[str, set[str]] = {
            split: {str(row["participant_id"]) for row in split_rosters.get(split, [])}
            for split in requested_splits
        }
        participant_overlap = {
            f"{left}:{right}": sorted(participant_by_split[left] & participant_by_split[right])
            for index, left in enumerate(requested_splits)
            for right in requested_splits[index + 1 :]
        }
        pair_by_split: dict[str, set[str]] = {
            split: {
                pair_id
                for (pair_split, pair_id) in pair_members
                if pair_split == split
            }
            for split in requested_splits
        }
        pair_overlap = {
            f"{left}:{right}": sorted(pair_by_split[left] & pair_by_split[right])
            for index, left in enumerate(requested_splits)
            for right in requested_splits[index + 1 :]
        }
        pair_errors: list[dict[str, Any]] = []
        for (split, pair_id), member in sorted(pair_members.items()):
            if len(member["participants"]) != 2 or member["labels"] != {0, 1}:
                pair_errors.append({"split": split, "pair_id": pair_id, "participants": sorted(member["participants"]), "labels": sorted(member["labels"])})
        overlap_pass = not errors and all(not values for values in participant_overlap.values()) and all(not values for values in pair_overlap.values()) and not pair_errors
        comparison_summaries[comparison] = {
            "status": "PASS" if overlap_pass else "FAIL",
            "split_records": split_records,
            "split_participants": {split: len(split_rosters.get(split, [])) for split in requested_splits},
            "split_pairs": {split: len(pair_by_split[split]) for split in requested_splits},
            "split_label_counts": {
                split: dict(Counter(str(row["label"]) for row in split_rosters.get(split, []) if row.get("label") != ""))
                for split in requested_splits
            },
            "participant_overlap": participant_overlap,
            "pair_overlap": pair_overlap,
            "invalid_pairs": pair_errors,
            "rosters": split_rosters,
        }

    output_paths: list[str] = []
    for comparison in requested_comparisons:
        comparison_dir = destination / comparison
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for split in requested_splits:
            path = comparison_dir / f"{split}_subjects.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROSTER_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(comparison_summaries[comparison]["rosters"].get(split, []))
            output_paths.append(str(path.resolve()))
    # The attachment explicitly names train_subjects.csv/val_subjects.csv/
    # test_subjects.csv.  Keep these as a combined, namespaced roster while
    # retaining one per-comparison file above for direct audit use.
    for split in requested_splits:
        path = destination / f"{split}_subjects.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROSTER_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(all_roster_rows[split], key=lambda row: (str(row.get("comparison")), str(row.get("participant_id")))))
        output_paths.append(str(path.resolve()))

    overlap_audit = {
        comparison: {
            key: value
            for key, value in summary.items()
            if key in {"status", "split_records", "split_participants", "split_pairs", "split_label_counts", "participant_overlap", "pair_overlap", "invalid_pairs"}
        }
        for comparison, summary in comparison_summaries.items()
    }
    status = "PASS" if not errors and all(summary["status"] == "PASS" for summary in comparison_summaries.values()) else "FAIL"
    manifest = {
        "schema_version": 1,
        "kind": "model_grid_v3_subject_roster_export",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_build_root": str(build_path),
        "source_manifest_paths": [str(path.resolve()) for path in source_paths],
        "source_manifest_fingerprints": source_fingerprints,
        "method": "read explicit frozen model_grid_v3 manifests; aggregate rows by participant_id within each manifest split; audit participant/pair disjointness",
        "comparisons": list(requested_comparisons),
        "splits": list(requested_splits),
        "output_paths": output_paths,
        "roster_fields": list(ROSTER_FIELDS),
        "comparison_summaries": comparison_summaries,
        "errors": errors,
    }
    # ``subject_roster_audit.json`` is deliberately separate from the source
    # build's audit files, so report generation cannot overwrite authoritative
    # preparation evidence.
    manifest_path = destination / "subject_roster_manifest.json"
    audit_path = destination / "subject_roster_audit.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "schema_version": 1,
                    "kind": "model_grid_v3_subject_roster_overlap_audit",
                    "status": status,
                    "source_build_root": str(build_path),
                    "source_manifest_fingerprints": source_fingerprints,
                    "comparison_overlap": overlap_audit,
                    "errors": errors,
                    "method": manifest["method"],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "status": status,
        "source_build_root": str(build_path),
        "output_dir": str(destination),
        "manifest_path": str(manifest_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "output_paths": output_paths,
        "source_manifest_paths": [str(path.resolve()) for path in source_paths],
        "source_manifest_fingerprints": source_fingerprints,
        "comparison_overlap": overlap_audit,
        "errors": errors,
        "method": manifest["method"],
    }


def _summarise_shuffle_diagnostic(root: Path) -> dict[str, Any] | None:
    """Read the fixed, post-gate shuffle addendum without copying payloads."""

    audit_path = root / "shuffle_diagnostic_audit.json"
    markdown_path = root / "shuffle_diagnostic_audit.md"
    orientation_path = root / "shuffle_saved_model_orientation.json"
    paths = [path for path in (audit_path, markdown_path, orientation_path) if path.is_file()]
    if not paths:
        return None
    audit: Mapping[str, Any] = {}
    orientation: Mapping[str, Any] = {}
    if audit_path.is_file():
        try:
            audit = _mapping(_read_json(audit_path))
        except (OSError, ValueError, json.JSONDecodeError):
            audit = {}
    if orientation_path.is_file():
        try:
            orientation = _mapping(_read_json(orientation_path))
        except (OSError, ValueError, json.JSONDecodeError):
            orientation = {}

    integrity_issues = 0
    seed_reports = _mapping(audit.get("seed_reports"))
    for report in seed_reports.values():
        split_integrity = _mapping(_mapping(report).get("split_integrity"))
        for key in ("invariant_field_mismatches", "missing_base_keys", "pair_id_mismatches", "test_true_label_mismatches", "unexpected_seed_keys"):
            value = _finite(split_integrity.get(key))
            if value is not None:
                integrity_issues += int(value)
        for key in ("duplicate_cache_keys", "participant_overlap", "pair_overlap"):
            value = split_integrity.get(key)
            if isinstance(value, Mapping):
                integrity_issues += sum(len(item) for item in value.values() if isinstance(item, list))
            elif isinstance(value, (list, tuple, set)):
                integrity_issues += len(value)
        prediction_checks = _mapping(_mapping(report).get("prediction_join_and_orientation"))
        for split in prediction_checks.values():
            integrity_issues += int(_finite(_mapping(split).get("join_error_count")) or 0)

    orientation_seeds = _mapping(orientation.get("seeds"))
    seed273_orientation = _mapping(orientation_seeds.get("273"))
    seed273_auc = {
        split: {
            "true_subject_auc": _finite(_mapping(seed273_orientation.get(split)).get("subject_true_auc")),
            "pseudo_subject_auc": _finite(_mapping(seed273_orientation.get(split)).get("subject_pseudo_auc")),
        }
        for split in ("train", "val", "test")
        if _mapping(seed273_orientation.get(split))
    }
    seed273_report = _mapping(seed_reports.get("273"))
    seed273_contingency = _mapping(_mapping(seed273_report.get("contingency")).get("train"))
    seed273_val_contingency = _mapping(_mapping(seed273_report.get("contingency")).get("val"))
    test_slice_counts: dict[str, int] = {}
    test_pair_counts: dict[str, int] = {}
    for seed, report in seed_reports.items():
        test_contingency = _mapping(_mapping(_mapping(report).get("contingency")).get("test"))
        test_slices = _finite(test_contingency.get("n_slices"))
        if test_slices is not None:
            test_slice_counts[str(seed)] = int(test_slices)
        domain_subjects = _mapping(test_contingency.get("subjects"))
        domain_counts = []
        for domain in ("fomo45k", "brats21"):
            counts = _mapping(domain_subjects.get(domain))
            if counts:
                domain_counts.append(sum(int(value) for value in counts.values() if _finite(value) is not None))
        if domain_counts and len(set(domain_counts)) == 1:
            test_pair_counts[str(seed)] = domain_counts[0]
    digest = _mapping(_mapping(audit.get("saved_model_orientation")).get("cache_tensor_digest"))
    references = {path.name: _fingerprint(path) for path in paths}
    return {
        "status": audit.get("status", "INCONCLUSIVE"),
        "orientation_status": orientation.get("status", _mapping(audit.get("saved_model_orientation")).get("status", "INCONCLUSIVE")),
        "integrity_issue_count": integrity_issues,
        "base_key_count": audit.get("base_key_count"),
        "cache_tensor_rows": digest.get("rows"),
        "cache_tensor_sha256": digest.get("sha256"),
        "tensor_hash_limitation": audit.get("tensor_cache_note", "per-row tensor hashes were not persisted by the in-memory cache"),
        "seed273_auc": seed273_auc,
        "seed273_train_domain_subjects": seed273_contingency.get("subjects"),
        "seed273_val_domain_subjects": seed273_val_contingency.get("subjects"),
        "v1_test_slice_counts": test_slice_counts,
        "v1_test_pair_counts": test_pair_counts,
        "v1_slice_selection_limitation": "powered-shuffle v1 test slice selection varies by seed; do not treat it as a fixed split/slice-order repeated-seed experiment",
        "interpretation": "consistent with finite-sample train-label/domain imbalance; no identity/split/prediction mismatch found; not uniquely proven cause",
        "next_protocol_status": "PENDING",
        "next_protocol": "preregistered protocol v2: fixed split seed 73, pair/slice-order hash invariant to labels across three runs, one-shot balanced-pair shuffle plus separate held-out-label sanity",
        "references": references,
    }


def _summarise_protocol_v2(root: Path) -> dict[str, Any] | None:
    """Read the supplementary balanced-shuffle v2 gate files compactly."""

    protocol_path = root / "protocol.json"
    preflight_path = root / "preflight.json"
    true_gate_path = root / "gate_true_test.json"
    random_gate_path = root / "gate_random_test_labels.json"
    aggregate_gate_path = root / "gate.json"
    summary_path = root / "supplementary_diagnostic_summary.json"
    summary_markdown_path = root / "supplementary_diagnostic_summary.md"
    # A regular run directory also has ``gate.json`` for tiny/positive/
    # negative/shuffle controls.  That file alone is not the supplementary v2
    # protocol and must not relabel every sibling row when several sources are
    # combined into one report.  Require one of the v2-specific artifacts;
    # ``gate.json`` is accepted only together with those artifacts below.
    if not any(path.is_file() for path in (preflight_path, true_gate_path, random_gate_path)):
        return None

    def read_mapping(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            return {}
        try:
            return _mapping(_read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    protocol = read_mapping(protocol_path)
    preflight = read_mapping(preflight_path)
    true_gate = read_mapping(true_gate_path)
    random_gate = read_mapping(random_gate_path)
    summary = read_mapping(summary_path)

    def compact_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
        compact_rows: list[dict[str, Any]] = []
        for item in gate.get("rows", []):
            if not isinstance(item, Mapping):
                continue
            interval = item.get("bootstrap_ci")
            low = high = None
            if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes, bytearray)) and len(interval) >= 2:
                low, high = _finite(interval[0]), _finite(interval[1])
            else:
                # The supplementary runner persists the interval as scalar
                # ``bootstrap_ci_low/high`` fields.  Accept the tuple form as
                # well because the report's synthetic fixtures and older
                # controls used that representation.
                low = _finite(item.get("bootstrap_ci_low"))
                high = _finite(item.get("bootstrap_ci_high"))
            paired_p = item.get("paired_swap_p")
            if paired_p is None:
                paired_p = item.get("paired_swap_p_value")
            holm_p = item.get("holm_p")
            if holm_p is None:
                holm_p = item.get("paired_swap_p_value_holm")
            compact_rows.append(
                {
                    "seed": item.get("seed"),
                    "subject_auc": _finite(item.get("subject_auc")),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "paired_swap_p": _finite(paired_p),
                    "holm_p": _finite(holm_p),
                }
            )
        return {
            "status": gate.get("status", "INCONCLUSIVE"),
            "closeness_status": gate.get("closeness_status", "INCONCLUSIVE"),
            "passed": _normalise_bool(gate.get("passed")),
            "failure_p_value_seeds": gate.get("failure_p_value_seeds", []),
            "failure_deviation_seeds": gate.get("failure_deviation_seeds", []),
            "rows": compact_rows,
            "reason": gate.get("reason"),
        }

    references: dict[str, dict[str, Any]] = {}
    reference_paths = [
        protocol_path,
        preflight_path,
        true_gate_path,
        random_gate_path,
        aggregate_gate_path,
        summary_path,
        summary_markdown_path,
    ]
    reference_paths.extend(sorted(root.glob("seed_*/result.json"), key=lambda path: str(path).lower()))
    for path in reference_paths:
        if path.is_file():
            references[path.relative_to(root).as_posix()] = _fingerprint(path)

    return {
        "protocol": protocol.get("protocol", summary.get("protocol", "balanced_powered_shuffle_v2")),
        "status": protocol.get("status", "INCONCLUSIVE"),
        "split_seed": protocol.get("split_seed", summary.get("split_seed")),
        "fit_seeds": protocol.get("fit_seeds", summary.get("fit_seeds", [])),
        "exact_pair_flips": protocol.get("exact_pair_flips"),
        "preflight_status": preflight.get("status", "INCONCLUSIVE"),
        "preflight_same_target_key_order": preflight.get("same_target_key_order_across_fit_seeds"),
        "preflight_same_target_pair_order": preflight.get("same_target_pair_order_across_fit_seeds"),
        "preflight_same_true_test_labels": preflight.get("same_true_test_labels_across_fit_seeds"),
        "retrained_permutation_null": protocol.get("retrained_permutation_null", "not_run"),
        "true_test_gate": compact_gate(true_gate),
        "random_test_label_gate": compact_gate(random_gate),
        "interpretation": summary.get(
            "interpretation",
            "Diagnostic v2 only; gate results are excluded from confirmatory formal rows.",
        ),
        "confirmatory_status": "EXCLUDED_DIAGNOSTIC",
        "next_stage2_status": "PENDING",
        "next_stage2": "one authorized FOMO Stage2 diagnostic fit with calibrated full train/val/test pair permutation null; not yet executed",
        "references": references,
    }


def _summarise_calibration(
    root: Path,
    *,
    classification: str = "DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY",
) -> dict[str, Any] | None:
    """Read the observed calibration fit without treating it as formal evidence.

    The calibration runner writes an observed fit beside a separately resumed
    full-retrain null.  The observed fit is useful for checking the calibrated
    model and recorded subject bootstrap, but it must not enter the
    confirmatory summary or ANDi/AP correlation while the preregistered null
    is incomplete (and while the historical controls remain failed).
    """

    observed_path = root / "observed" / "result.json"
    if not observed_path.is_file():
        return None

    def read_mapping(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            return {}
        try:
            return _mapping(_read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    observed = read_mapping(observed_path)
    config = _mapping(observed.get("config"))
    test = _mapping(observed.get("test"))
    test_slice = _mapping(test.get("slice"))
    test_subject = _mapping(test.get("subject"))
    validation = _mapping(observed.get("validation"))
    validation_subject = _mapping(validation.get("subject"))
    statistics = _mapping(observed.get("test_statistics"))
    pair_swap = _mapping(statistics.get("heldout_pair_swap"))
    bootstrap = _mapping(statistics.get("subject_bootstrap"))
    null_status_path = root / "retrained_null" / "status.json"
    null_summary_path = root / "retrained_null" / "summary.json"
    null_status = read_mapping(null_status_path)
    null_summary = read_mapping(null_summary_path)
    null_statistic = _mapping(null_summary.get("full_null_statistic"))
    if not null_status and null_summary:
        null_status = null_summary
    null_state = str(null_status.get("status", "not_started")).lower()
    if null_state in {"complete", "completed", "pass", "passed"}:
        null_state = "COMPLETE"
    elif null_state in {"incomplete", "running"}:
        null_state = "INCOMPLETE"
    elif null_state in {"failed", "fail"}:
        null_state = "FAIL"
    else:
        null_state = "NOT_STARTED"

    # The v3 observed run writes two small post-fit audits beside the result.
    # Keep their identity and gate facts, but never copy the large random-label
    # result payload into the report.  In particular, an INCONCLUSIVE sanity
    # gate is different from an incomplete audit artifact.
    binding_path = root / "input_binding_audit.json"
    binding_summary: dict[str, Any] | None = None
    if binding_path.is_file():
        binding = read_mapping(binding_path)

        def compact_digest(value: Any) -> dict[str, Any]:
            digest = _mapping(value)
            return {
                key: digest.get(key)
                for key in ("stage", "rows", "sha256", "materialized_once", "materialized_once_equivalent")
                if key in digest
            }

        frozen_ledger = _mapping(binding.get("frozen_healthy_ledger"))
        binding_errors = binding.get("errors")
        binding_summary = {
            "status": str(binding.get("status", "INCONCLUSIVE")).upper(),
            "audit_scope": binding.get("audit_scope"),
            "observed_rows_checked": binding.get("observed_rows_checked"),
            "brats_rows_checked_through_canonical_reader": binding.get("braTS_rows_checked_through_canonical_reader"),
            "split_counts": binding.get("split_counts", {}),
            "label_counts": binding.get("label_counts", {}),
            "shape_counts": binding.get("shape_counts", {}),
            "dtype_counts": binding.get("dtype_counts", {}),
            "digest_equal": _normalise_bool(binding.get("digest_equal")),
            "raw_dataset_bytes_copied": _normalise_bool(binding.get("raw_dataset_bytes_copied")),
            "tensor_cache_bytes_copied": _normalise_bool(binding.get("tensor_cache_bytes_copied")),
            "training_started_before_audit": _normalise_bool(binding.get("training_started_before_audit")),
            "observed_result_fingerprint": binding.get("observed_result_fingerprint"),
            "observed_source_cache_digest": compact_digest(binding.get("observed_source_cache_digest")),
            "recomputed_source_cache_digest": compact_digest(binding.get("recomputed_source_cache_digest")),
            "frozen_healthy_ledger": {
                key: frozen_ledger.get(key)
                for key in ("path", "summary_path", "sha256", "healthy_ledger_rows", "fomo_healthy_rows_checked", "fomo_healthy_keys_checked")
                if key in frozen_ledger
            },
            "error_count": len(binding_errors) if isinstance(binding_errors, list) else None,
        }

    sanity_path = root / "immutable_test_label_sanity" / "result.json"
    sanity_summary: dict[str, Any] | None = None
    if sanity_path.is_file():
        sanity = read_mapping(sanity_path)
        random_gate = _mapping(sanity.get("random_test_label_gate"))
        compact_seed_rows: list[dict[str, Any]] = []
        raw_seed_rows = random_gate.get("rows")
        if isinstance(raw_seed_rows, list):
            for item in raw_seed_rows:
                if not isinstance(item, Mapping):
                    continue
                compact_seed_rows.append(
                    {
                        key: _finite(item.get(key)) if key in {"subject_auc", "bootstrap_ci_low", "bootstrap_ci_high", "abs_auc_minus_half", "paired_swap_p_value", "paired_swap_p_value_holm", "holm_adjusted_p_value"} else item.get(key)
                        for key in (
                            "seed",
                            "subject_auc",
                            "bootstrap_ci_low",
                            "bootstrap_ci_high",
                            "abs_auc_minus_half",
                            "paired_swap_p_value",
                            "paired_swap_p_value_holm",
                            "holm_adjusted_p_value",
                            "paired_swap_status",
                        )
                        if key in item
                    }
                )
        holm_values = random_gate.get("holm_adjusted_p_values")
        sanity_summary = {
            "status": str(sanity.get("status", "INCONCLUSIVE")).upper(),
            "protocol": sanity.get("protocol"),
            "gate_status": str(random_gate.get("status", "INCONCLUSIVE")).upper(),
            "passed": _normalise_bool(random_gate.get("passed")),
            "closeness_status": str(random_gate.get("closeness_status", "INCONCLUSIVE")).upper(),
            "reason": random_gate.get("reason"),
            "probabilities_immutable": _normalise_bool(sanity.get("probabilities_immutable")),
            "true_test_labels_preserved": _normalise_bool(sanity.get("true_test_labels_preserved")),
            "fit_retrained": _normalise_bool(sanity.get("fit_retrained")),
            "n_test_participants": sanity.get("n_test_participants"),
            "n_test_slices": sanity.get("n_test_slices"),
            "swap_seeds": sanity.get("swap_seeds", []),
            "predictions_sha256": sanity.get("predictions_sha256"),
            "test_manifest_sha256": sanity.get("test_manifest_sha256"),
            "holm_adjusted_p_values": holm_values if isinstance(holm_values, Mapping) else {},
            "rows": compact_seed_rows,
        }

    reference_paths = [
        root / "protocol.json",
        root / "run_identity.json",
        observed_path,
        root / "observed" / "test_predictions.jsonl",
        root / "observed" / "model_best.pt",
        root / "source_cache_digest.json",
        null_status_path,
        null_summary_path,
        root / "retrained_null" / "statistic.json",
        root / "calibration_figures" / "plot_manifest.json",
        root / "calibration_figures" / "full_retrained_null.png",
        binding_path,
        sanity_path,
    ]
    references = {
        path.relative_to(root).as_posix(): _fingerprint(path)
        for path in reference_paths
        if path.is_file()
    }
    return {
        "status": "COMPLETE",
        "classification": classification,
        "cohort": "FOMO",
        "modalities": config.get("modalities", ["flair", "t1", "t2"]),
        "classifier": config.get("model", observed.get("model", {}).get("class", "unknown")),
        "seed": observed.get("seed", config.get("seed")),
        "split_seed": observed.get("split_seed", config.get("split_seed")),
        "best_epoch": observed.get("best_epoch"),
        "epochs_completed": observed.get("epochs_completed"),
        "subject_auc": _finite(test_subject.get("roc_auc")),
        "test_subject_count": test_subject.get("n"),
        "test_slice_count": test_slice.get("n"),
        "subject_bootstrap_ci_low": _finite(bootstrap.get("ci_low")),
        "subject_bootstrap_ci_high": _finite(bootstrap.get("ci_high")),
        "subject_bootstrap_n": bootstrap.get("n_bootstrap"),
        "subject_bootstrap_valid_n": bootstrap.get("n_valid"),
        "subject_bootstrap_unit": bootstrap.get("resampling_unit"),
        "slice_auc": _finite(test_slice.get("roc_auc")),
        "subject_bce": _finite(test_subject.get("bce")),
        "slice_bce": _finite(test_slice.get("bce")),
        "subject_accuracy": _finite(test_subject.get("accuracy")),
        "validation_subject_auc": _finite(validation_subject.get("roc_auc")),
        "validation_subject_bce": _finite(validation_subject.get("bce")),
        "heldout_pair_swap": {
            "status": pair_swap.get("status", "INCONCLUSIVE"),
            "conditional_on_matched_pairs": pair_swap.get("conditional_on_matched_pairs"),
            "n_pairs": pair_swap.get("n_pairs"),
            "n_swaps": pair_swap.get("n_swaps"),
            "observed_auc": _finite(pair_swap.get("observed_auc")),
            "p_value": _finite(pair_swap.get("p_value")),
            "null_ci_low": _finite(pair_swap.get("null_ci_low")),
            "null_ci_high": _finite(pair_swap.get("null_ci_high")),
        },
        "permutation_mode": config.get("permutation_mode"),
        "permutation_replicates_requested": config.get("permutation_replicates"),
        "full_retrained_null": {
            "status": null_state,
            "completed": null_status.get("completed", null_summary.get("completed", 0)),
            "requested": null_status.get("requested", null_summary.get("requested", config.get("permutation_replicates"))),
            "remaining": null_status.get("remaining"),
            "updated_at_utc": null_status.get("updated_at_utc"),
            # A resumed null writes a provisional statistic after each
            # prefix.  Never expose that p-value as a completed result; only
            # a terminal COMPLETE status can make it reportable.
            "p_plus_one": _finite(null_statistic.get("p_plus_one")) if null_state == "COMPLETE" else None,
            "statistic": null_statistic.get("statistic") if null_state == "COMPLETE" else None,
            "label_scope": null_statistic.get("label_scope", "train_val_test_complete_pair_swaps"),
        },
        "input_binding": binding_summary,
        "immutable_test_label_sanity": sanity_summary,
        "protocol": observed.get("protocol", read_mapping(root / "protocol.json").get("protocol")),
        "run_fingerprint": observed.get("run_fingerprint"),
        "calibration_plot_manifest": str((root / "calibration_figures" / "plot_manifest.json").resolve())
        if (root / "calibration_figures" / "plot_manifest.json").is_file()
        else None,
        "full_retrained_null_plot": str((root / "calibration_figures" / "full_retrained_null.png").resolve())
        if (root / "calibration_figures" / "full_retrained_null.png").is_file()
        else None,
        "references": references,
    }


def _prediction_fields(row: Mapping[str, Any]) -> tuple[int | None, float | None, str | None]:
    label_raw = _first(row, "label", "y_true", "target", "domain_label", "class")
    score_raw = _first(
        row,
        "score",
        "probability",
        "prob",
        "predicted_probability",
        "p_domain",
        "prediction",
    )
    participant_raw = _first(
        row,
        "participant_id",
        "subject_id",
        "participant",
        "subject",
        "patient_id",
        "case_id",
    )
    if label_raw is None or score_raw is None:
        return None, None, None
    if isinstance(label_raw, str):
        label_text = label_raw.strip().lower()
        if label_text in {"healthy", "control", "source", "a", "0"}:
            label = 0
        elif label_text in {"brats", "brats21", "target", "b", "1"}:
            label = 1
        else:
            return None, None, None
    else:
        try:
            label = int(float(label_raw))
        except (TypeError, ValueError):
            return None, None, None
    score = _finite(score_raw)
    if label not in (0, 1) or score is None:
        return None, None, None
    participant = None if participant_raw in (None, "") else str(participant_raw)
    return label, score, participant


def _meta_for_row(row: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    nested = _mapping(row.get("metadata"))
    merged: dict[str, Any] = dict(base)
    merged.update(nested)
    merged.update(row)
    return merged


def _path_has_control_marker(value: Any) -> bool:
    """Return whether a provenance/path value names a control artifact.

    Matching is done on path/name tokens instead of a raw substring so a
    parent directory such as ``domain_classifier`` is not mistaken for a
    control.  This is deliberately conservative for ``registered`` and
    permutation artifacts: they remain auditable rows, but cannot enter the
    formal final-input conclusions.
    """

    if value in (None, ""):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(value).lower()))
    return bool(tokens.intersection(_CONTROL_TOKENS))


def _value_has_control_marker(key: str, value: Any) -> bool:
    """Interpret explicit control/role metadata without inspecting metrics."""

    key_text = str(key).strip().lower()
    if key_text in {"is_control", "control", "shuffle", "shuffled_labels"}:
        if isinstance(value, Mapping):
            return True
        status = _normalise_bool(value)
        if status is not None:
            return bool(status)
        neutral = {"", "none", "null", "main", "formal", "final", "no", "false", "0"}
        return str(value).strip().lower() not in neutral
    if key_text in {"permutation_index"}:
        return value not in (None, "")
    if isinstance(value, Mapping):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(value).lower()))
    return bool(tokens.intersection(_CONTROL_TOKENS))


def _result_kind(row: Mapping[str, Any], base: Mapping[str, Any]) -> str:
    """Classify one input row for scientific reporting.

    Only an explicit ``final`` model-input stage with no control marker is a
    ``formal_final`` row.  Missing stage metadata is allowed for legacy metric
    artifacts because the report's identity layer defaults it to ``final``;
    explicit ``raw``/``registered``/control metadata always wins.
    """

    merged = _meta_for_row(row, base)
    explicit_kind = _normalise_text(merged.get("result_kind")).lower()
    if explicit_kind in {"diagnostic_formalfit", "diagnostic_formal_fit", "diagnostic_fit"}:
        return "diagnostic_formalfit"
    stage = _normalise_text(
        _first(merged, "stage", "preprocessing_stage", default="final"),
        "final",
    ).lower().replace(" ", "_")
    if stage not in _FINAL_STAGES:
        return "intermediate"

    for key in _CONTROL_MARKER_KEYS:
        if key not in merged:
            continue
        value = merged[key]
        # ``result_kind=formal_final`` is a positive identity marker; only
        # values that explicitly name a control should exclude the row.
        if key == "result_kind" and str(value).strip().lower() in {
            "formal_final",
            "formal",
            "final",
            "main",
        }:
            continue
        if _value_has_control_marker(key, value):
            return "control"

    # For rows loaded from a run directory, the loader records the exact
    # artifact file in ``_artifact_path``.  ``source_artifact`` and
    # ``source_path`` cover explicit records and older runner schemas.
    for key in ("_artifact_path", "artifact_path", "source_artifact", "source_path"):
        if _path_has_control_marker(merged.get(key)):
            return "control"
    return "formal_final"


def _control_mode_label(row: Mapping[str, Any], base: Mapping[str, Any]) -> str:
    """Return a human-readable control/run mode for the summary table."""

    merged = _meta_for_row(row, base)
    explicit_control_mode = _normalise_text(merged.get("control_mode"))
    if explicit_control_mode in {"balanced_shuffle_v2", "diagnostic_formalfit"}:
        return explicit_control_mode
    mode_values = [
        _normalise_text(merged.get(key))
        for key in ("mode", "control_type", "control_name", "run_kind", "run_role", "purpose")
        if merged.get(key) not in (None, "")
    ]
    mode = " ".join(mode_values)
    tokens = set(re.findall(r"[a-z0-9]+", mode.lower()))
    if tokens.intersection({"tiny", "overfit"}):
        return "tiny_overfit"
    if tokens.intersection({"positive", "registered"}):
        return "positive_registered"
    # Keep explicit label shuffles visibly separate from the completed
    # same-cohort negative control.  ``permutation_mode=smoke`` is a runner
    # setting for the recorded pair-swap sanity check, so it is not itself a
    # label-shuffle marker.
    if tokens.intersection({"shuffle", "shuffled", "permutation", "permuted"}):
        return "label_shuffle"
    permutation_mode = _normalise_text(merged.get("permutation_mode"))
    permutation_tokens = set(re.findall(r"[a-z0-9]+", permutation_mode.lower()))
    if permutation_tokens.intersection({"shuffle", "shuffled", "permutation", "permuted"}):
        return "label_shuffle"
    if "negative" in tokens or {"same", "cohort"}.issubset(tokens):
        return "same_cohort_negative"
    stage = _normalise_text(_first(merged, "stage", "preprocessing_stage", default="final"), "final").lower()
    if stage == "registered":
        return "positive_registered"
    return "formal_final"


def _is_formal_final_row(row: Mapping[str, Any], base: Mapping[str, Any] | None = None) -> bool:
    """Whether a summary row may contribute to the nine scientific answers."""

    classified = _result_kind(row, base or {})
    if "formal_final" in row:
        explicit = row.get("formal_final")
        if isinstance(explicit, str):
            if explicit.strip().lower() in {"false", "no", "fail", "control", "intermediate"}:
                return False
            if explicit.strip().lower() in {"true", "yes", "pass", "formal_final"}:
                return classified == "formal_final"
        if explicit is not None:
            return bool(explicit) and classified == "formal_final"
    if str(row.get("result_kind", "")).strip().lower() == "formal_final":
        return classified == "formal_final"
    return classified == "formal_final"


def _row_key(row: Mapping[str, Any], base: Mapping[str, Any], ordinal: int = 0) -> tuple[Any, ...]:
    merged = _meta_for_row(row, base)
    run_id = _normalise_text(_first(merged, "run_id", "experiment_id", default=base.get("run_id", "run")))
    domain = _normalise_domain(
        _first(merged, "healthy_domain", "healthy_cohort", "cohort", "source_dataset", "domain_a", default="unknown")
    )
    modalities = _normalise_modalities(
        _first(merged, "input_modalities", "modalities", "modality", "channels", default="")
    )
    classifier = _normalise_classifier(_first(merged, "classifier", "model", "architecture", default="unknown"))
    seed = _normalise_seed(_first(merged, "seed", "training_seed", default=None))
    # A metric collection can hold several seeds; an artifact with no explicit
    # run id still gets one stable group rather than being merged with another
    # caller's collection.
    return (run_id, domain, modalities, classifier, seed, ordinal if run_id == "run" else None)


def _candidate_files(root: Path, names: Sequence[str], *, under: str | None = None) -> list[Path]:
    directory = root / under if under else root
    if not directory.exists():
        return []
    found: list[Path] = []
    for name in names:
        direct = directory / name
        if direct.is_file():
            found.append(direct)
    # Do not walk cache/image trees.  ``rglob`` is still bounded to report
    # names and is fast for normal experiment directories.
    for path in directory.rglob("*"):
        if not path.is_file() or any(part.lower() in {"cache", "images", "checkpoints"} for part in path.parts):
            continue
        if path.name in names and path not in found:
            found.append(path)
    return sorted(found, key=lambda item: str(item).lower())


def _load_run_metadata(root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"run_id": root.name, "source_path": str(root.resolve())}
    names = (
        "experiment_manifest.json",
        "run_manifest.json",
        "manifest.json",
        "run_result.json",
        "metrics.json",
        "results.json",
        "config.json",
        "launch_metadata.json",
        "run_identity.json",
    )
    for path in _candidate_files(root, names):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, Mapping):
            # Keep the first value for core identity fields, while allowing
            # later result files to add metrics/controls.
            for key, value in payload.items():
                if key not in metadata or metadata[key] in (None, ""):
                    metadata[key] = value
            # ``run_identity.json`` stores model/input identity below
            # ``config``.  Flatten only scalar/list identity fields so rows
            # loaded from sibling prediction files retain the exact run
            # provenance without copying the full identity mapping into each
            # report row.
            config = _mapping(payload.get("config"))
            for key in (
                "model",
                "modalities",
                "input_modalities",
                "stage",
                "seed",
                "split_seed",
                "tiny",
                "permutation_mode",
            ):
                if key in config and (key not in metadata or metadata[key] in (None, "")):
                    metadata[key] = config[key]
            metadata.setdefault("artifact_paths", []).append(str(path.resolve()))
    return metadata


def _load_prediction_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    names = (
        "per_slice_predictions.csv",
        "predictions.csv",
        "per_slice_predictions.json",
        "predictions.json",
        "predictions.jsonl",
    )
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    candidates = _candidate_files(root, names, under="predictions") + _candidate_files(root, names)
    # The canonical runner writes one JSONL file per seed at the run root,
    # e.g. ``seed_73_test_predictions.jsonl``.  Include that pattern without
    # walking large cache directories.
    candidates.extend(
        path
        for path in root.rglob("*_test_predictions.jsonl")
        if path.is_file() and "cache" not in {part.lower() for part in path.parts}
    )
    # Supplementary protocol runs keep their complete runner result under
    # ``seed_<n>/result.json``.  Read this exact bounded pattern so the report
    # can fingerprint and classify those diagnostic fits explicitly.
    candidates.extend(path for path in root.glob("seed_*/result.json") if path.is_file())
    # A seed JSON contains the same predictions with the run/config identity;
    # reading it lets the report retain cohort/model/seed metadata even when
    # the separate JSONL file contains only label/probability/participant.
    candidates.extend(path for path in root.glob("seed_*.json") if path.is_file())
    seed_context: dict[str, dict[str, Any]] = {}
    for seed_path in sorted(root.glob("seed_*.json"), key=lambda item: str(item).lower()):
        try:
            payload = _read_json(seed_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            config = _mapping(payload.get("config"))
            inherited = {key: value for key, value in config.items()}
            inherited.update(
                {
                    key: value
                    for key, value in payload.items()
                    if key in {"seed", "run_id", "healthy_domain", "input_modalities", "modalities", "classifier", "stage"}
                    or (key == "model" and not isinstance(value, Mapping))
                }
            )
            seed_context[seed_path.stem.split("_", 1)[-1]] = inherited
    seen: set[str] = set()
    for path in candidates:
        if str(path) in seen:
            continue
        seen.add(str(path))
        if str(path) in paths:
            continue
        prediction_name = re.match(r"seed_(\d+)_test_predictions$", path.stem)
        if prediction_name:
            seed_json = root / f"seed_{prediction_name.group(1)}.json"
            if seed_json.is_file():
                try:
                    seed_payload = _read_json(seed_json)
                    # Prefer the seed JSON because it carries run/config
                    # identity.  The JSONL remains a fallback if its sibling
                    # contains no serialised prediction list.
                    if any(_prediction_fields(item)[0] is not None for item in _rows_from_payload(seed_payload)):
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        try:
            payload = _read_data_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidate = _rows_from_payload(payload)
        seed_match = re.match(r"seed_(\d+)", path.stem)
        if seed_match and seed_match.group(1) in seed_context:
            inherited = seed_context[seed_match.group(1)]
            candidate = [{**inherited, **row} for row in candidate]
        candidate = [
            {"_artifact_path": str(path.resolve()), **row}
            for row in candidate
        ]
        prediction_count = sum(_prediction_fields(row)[0] is not None for row in candidate)
        if prediction_count:
            rows.extend(candidate)
            paths.append(str(path.resolve()))
    return rows, paths


def _load_metric_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    names = (
        "metrics.json",
        "results.json",
        "run_result.json",
        "experiment_result.json",
        "summary.json",
        "metrics.csv",
        "results.csv",
        "summary.csv",
    )
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    candidates = _candidate_files(root, names)
    candidates.extend(path for path in root.glob("seed_*.json") if path.is_file())
    for path in candidates:
        try:
            payload = _read_data_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidate = _rows_from_payload(payload)
        # A seed JSON may contain the same prediction list as the dedicated
        # JSONL file.  The prediction reader owns that case; keeping it here
        # would produce duplicate rows and double-count a seed.
        if any(_prediction_fields(item)[0] is not None for item in candidate):
            continue
        candidate = [
            {"_artifact_path": str(path.resolve()), **row}
            for row in candidate
        ]
        if candidate:
            rows.extend(candidate)
            paths.append(str(path.resolve()))
    return rows, paths


def _control_files(root: Path) -> list[Path]:
    # Runner controls live either below ``controls/`` in synthetic fixtures or
    # as the run-local ``gate.json`` used by the domain-classifier runner.
    # Keep discovery bounded to control-named artifacts; ordinary metrics and
    # prediction files remain handled by their dedicated loaders.
    control_roots = [root / "controls"]
    result: list[Path] = []
    direct_gate = root / "gate.json"
    if direct_gate.is_file():
        result.append(direct_gate)
    for control_root in control_roots:
        if not control_root.exists():
            continue
        for path in control_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
                continue
            result.append(path)
    return sorted(result, key=lambda item: str(item).lower())


def _control_name(path: Path) -> str:
    text = str(path).lower().replace("-", "_")
    if "tiny" in text or "overfit" in text:
        return "tiny_overfit"
    if "positive" in text or "registered" in text:
        return "positive_control"
    if "negative" in text:
        return "negative_control"
    if "permutation" in text or "shuffle" in text or "shuffled" in text:
        return "label_permutation"
    return ""


def _control_payload_status(payload: Any) -> tuple[bool | None, Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            status, mapping = _control_payload_status(item)
            if status is not None:
                return status, mapping
        return None, {}
    mapping = _mapping(payload)
    return _status_from_mapping(mapping), mapping


def _compact_control_mapping(payload: Any, status: bool | None) -> dict[str, Any]:
    """Retain gate evidence without embedding runner prediction payloads.

    ``gate.json`` can contain the complete per-slice results for several
    seeds.  The report needs the gate decision, thresholds, and compact gate
    rows, but copying nested ``results``/``test_predictions`` into
    ``report_audit.json`` makes the audit needlessly huge and obscures which
    evidence was actually consumed.
    """

    root = _mapping(payload)
    gate = _mapping(root.get("gate"))
    selected: dict[str, Any] = {}
    for key in (
        "status",
        "mode",
        "permutation_mode",
        "permutation_replicates",
        "swap_replicates",
        "run_fingerprint",
        "subject_auc",
        "ci_low",
        "ci_high",
        "train_accuracy",
        "train_bce",
    ):
        if key in root and not isinstance(root[key], (Mapping, list)):
            selected[key] = root[key]
    if status is not None:
        selected["status"] = "PASS" if status else "FAIL"
    if gate:
        for key in (
            "name",
            "status",
            "passed",
            "reason",
            "alpha",
            "chance_ci_band",
            "deviation_threshold",
            "minimum_reproducible_seeds",
            "minimum_subject_auc",
            "minimum_bootstrap_ci_low",
            "accuracy_threshold",
            "bce_threshold",
            "closeness_status",
            "failure_deviation_seeds",
            "failure_p_value_seeds",
            "holm_adjusted_p_values",
            "direction_invariant_bootstrap_upper",
        ):
            if key in gate:
                selected[key] = gate[key]
        rows = gate.get("rows")
        if isinstance(rows, list):
            # Gate rows are small scalar summaries (one row per seed).  Drop
            # anything unexpectedly nested rather than serialising a large
            # result payload by accident.
            selected["rows"] = [
                {
                    str(key): value
                    for key, value in item.items()
                    if not isinstance(value, (Mapping, list))
                }
                for item in rows
                if isinstance(item, Mapping)
            ]
    retrained = _mapping(root.get("retrained_permutations"))
    if retrained:
        # The full-retrain null may be deliberately unrequested in a smoke
        # run.  Preserve its explicit incomplete/zero-count status without
        # serialising any hidden result vectors.
        selected["full_retrained_null"] = {
            key: retrained.get(key)
            for key in (
                "status",
                "completed",
                "requested",
                "remaining",
                "unit",
                "permutation_mode",
                "permutation_replicates",
                "selection_and_scaler_refit",
            )
            if key in retrained
        }
        if "permutation_mode" not in selected["full_retrained_null"] and selected.get("permutation_mode") not in (None, ""):
            selected["full_retrained_null"]["permutation_mode"] = selected["permutation_mode"]
        if "permutation_replicates" not in selected["full_retrained_null"] and selected.get("permutation_replicates") not in (None, ""):
            selected["full_retrained_null"]["permutation_replicates"] = selected["permutation_replicates"]
    if gate:
        pair_swap_rows = [item for item in gate.get("rows", []) if isinstance(item, Mapping) and item.get("paired_swap_status") not in (None, "")]
        if pair_swap_rows:
            statuses = sorted({str(item.get("paired_swap_status")) for item in pair_swap_rows})
            selected["pair_swap"] = {
                "status": statuses[0] if len(statuses) == 1 else "mixed",
                "completed_seeds": len(pair_swap_rows),
            }
    return selected


def _load_controls(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    run_metadata = _load_run_metadata(root)
    for path in _control_files(root):
        name = _control_name(path)
        if not name:
            continue
        try:
            payload = _read_data_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        status, mapping = _control_payload_status(payload)
        compact = _compact_control_mapping(payload, status)
        compact["source_path"] = str(path.resolve())
        for key in ("permutation_mode", "permutation_replicates", "swap_replicates"):
            if compact.get(key) in (None, "") and run_metadata.get(key) not in (None, ""):
                compact[key] = run_metadata[key]
        if compact.get("control_mode") in (None, ""):
            mode_label = _control_mode_label(compact, run_metadata)
            if mode_label != "formal_final":
                compact["control_mode"] = mode_label
        null = compact.get("full_retrained_null")
        if isinstance(null, Mapping):
            if null.get("permutation_mode") in (None, "") and compact.get("permutation_mode") not in (None, ""):
                null["permutation_mode"] = compact["permutation_mode"]
            if null.get("permutation_replicates") in (None, "") and compact.get("permutation_replicates") not in (None, ""):
                null["permutation_replicates"] = compact["permutation_replicates"]
        entry = result.setdefault(name, compact)
        if entry.get("status") is None and status is not None:
            entry["status"] = status
        for key, value in compact.items():
            if key not in entry:
                entry[key] = value
    return result


def _control_value(rows: Iterable[Mapping[str, Any]], controls: Mapping[str, Any], name: str) -> bool | None:
    if name in controls:
        status = _status_from_mapping(controls[name])
        if status is not None:
            return status
    aliases = {
        "tiny_overfit": ("tiny_overfit_pass", "tiny_overfit", "tiny_set_overfit"),
        "positive_control": ("positive_control_pass", "positive_control", "registered_positive_control"),
        "negative_control": ("negative_control_pass", "negative_control"),
        "label_permutation": ("label_permutation_pass", "label_permutation", "permutation_pass", "shuffle_pass"),
    }
    for row in rows:
        for key in aliases[name]:
            if key in row:
                status = _status_from_mapping(row[key])
                if status is not None or str(row[key]).strip().lower() in {"inconclusive", "missing"}:
                    return status
        nested = _mapping(row.get("controls"))
        if name in nested:
            status = _status_from_mapping(nested[name])
            if status is not None:
                return status
    return None


def _metric_from_mapping(row: Mapping[str, Any], name: str, *, level: str | None = None) -> float | None:
    aliases = {
        "slice_roc_auc": ("slice_roc_auc", "slice_auc", "roc_auc_slice"),
        "subject_roc_auc": ("subject_roc_auc", "subject_auc", "roc_auc_subject", "roc_auc"),
        "subject_pr_auc": ("subject_pr_auc", "subject_average_precision", "pr_auc_subject", "average_precision"),
        "accuracy": ("accuracy", "subject_accuracy"),
        "balanced_accuracy": ("balanced_accuracy", "subject_balanced_accuracy"),
        "sensitivity": ("sensitivity", "subject_sensitivity", "recall"),
        "specificity": ("specificity", "subject_specificity"),
        "bce": ("bce", "loss", "binary_cross_entropy"),
    }
    for alias in aliases.get(name, (name,)):
        value = _finite(row.get(alias))
        if value is not None:
            return value
    if level:
        nested = _mapping(row.get(level))
        for alias in aliases.get(name, (name,)):
            value = _finite(nested.get(alias))
            if value is not None:
                return value
    return None


def _nested_metric(row: Mapping[str, Any], section: str, *names: str) -> float | None:
    nested = _mapping(row.get(section))
    for name in names:
        value = _finite(nested.get(name))
        if value is not None:
            return value
    return None


def _count_for_split(rows: Sequence[Mapping[str, Any]], split: str, *, subjects: bool) -> int | None:
    aliases = {"val": {"val", "validation", "valid"}, "train": {"train", "training"}, "test": {"test", "heldout", "holdout"}}
    selected = [
        row
        for row in rows
        if _normalise_text(_first(row, "split", "partition", "set", default="")).lower() in aliases[split]
    ]
    if not selected:
        return None
    if not subjects:
        return len(selected)
    values = [
        _first(row, "participant_id", "subject_id", "participant", "subject", "patient_id", "case_id")
        for row in selected
    ]
    values = [str(value) for value in values if value not in (None, "")]
    return len(set(values)) if values else None


def _direct_count(row: Mapping[str, Any], split: str, subjects: bool) -> int | None:
    prefix = "val" if split == "val" else split
    names = (
        f"{prefix}_subjects",
        f"{prefix}_subject_count",
        f"n_{prefix}_subjects",
    ) if subjects else (
        f"{prefix}_slices",
        f"{prefix}_slice_count",
        f"n_{prefix}_slices",
    )
    for name in names:
        value = _finite(row.get(name))
        if value is not None:
            return int(value)
    nested = _mapping(row.get("counts"))
    for name in names:
        value = _finite(nested.get(name))
        if value is not None:
            return int(value)
    return None


def _ci_from_mapping(row: Mapping[str, Any], section: str, low_names: Sequence[str], high_names: Sequence[str]) -> tuple[float | None, float | None]:
    nested = _mapping(row.get(section))
    low = None
    high = None
    for key in low_names:
        low = _finite(nested.get(key))
        if low is not None:
            break
    for key in high_names:
        high = _finite(nested.get(key))
        if high is not None:
            break
    return low, high


def _direction_invariant_ci(auc: float | None, low: float | None, high: float | None) -> tuple[float | None, float | None, float | None]:
    if auc is None:
        return None, None, None
    direction = max(auc, 1.0 - auc)
    if low is None or high is None:
        return direction, None, None
    # AUC orientation is not selected after looking at the result.  For a
    # direction-invariant diagnostic, transform both orientations and retain a
    # conservative union interval.
    return direction, max(0.5, min(low, 1.0 - high)), max(high, 1.0 - low)


def _apply_gate_fields(row: dict[str, Any], config: ReportConfig) -> None:
    auc = _finite(row.get("subject_roc_auc"))
    ci_low = _finite(row.get("bootstrap_ci_low"))
    ci_high = _finite(row.get("bootstrap_ci_high"))
    direction, direction_low, direction_high = _direction_invariant_ci(auc, ci_low, ci_high)
    row["subject_direction_invariant_auc"] = direction
    row["direction_invariant_ci_low"] = direction_low
    row["direction_invariant_ci_high"] = direction_high

    statuses = [
        _normalise_bool(row.get("tiny_overfit_pass")),
        _normalise_bool(row.get("positive_control_pass")),
        _normalise_bool(row.get("negative_control_pass")),
        _normalise_bool(row.get("label_permutation_pass")),
    ]
    if all(status is True for status in statuses):
        row["control_gate_status"] = "PASS"
    elif any(status is False for status in statuses):
        row["control_gate_status"] = "FAIL"
    else:
        row["control_gate_status"] = "INCONCLUSIVE"

    if row["control_gate_status"] != "PASS" or direction_high is None:
        row["low_separability_status"] = "INCONCLUSIVE"
    elif direction_high < config.low_separability_upper:
        row["low_separability_status"] = "PASS"
    else:
        row["low_separability_status"] = "FAIL"
    row["evidence_status"] = (
        "COMPLETE"
        if auc is not None and row.get("control_gate_status") == "PASS"
        else "INCONCLUSIVE"
    )


def _base_identity(row: Mapping[str, Any], base: Mapping[str, Any], ordinal: int = 0) -> dict[str, Any]:
    merged = _meta_for_row(row, base)
    explicit_domain = _first(merged, "healthy_domain", "healthy_cohort", "cohort", "source_dataset", "domain_a", default=None)
    domain = _normalise_domain(explicit_domain, default="unknown") if explicit_domain is not None else "unknown"
    source_path = _normalise_text(_first(merged, "source_path", default=base.get("source_path", "")))
    if domain == "unknown":
        domain = _infer_domain(source_path or base.get("run_id", ""), default="unknown")
    result_kind = _result_kind(row, base)
    source_artifact = _first(
        merged,
        "_artifact_path",
        "artifact_path",
        "source_artifact",
        default=base.get("source_path", ""),
    )
    return {
        "healthy_domain": domain,
        "input_modalities": _normalise_modalities(
            _first(merged, "input_modalities", "modalities", "modality", "channels", default="")
        ),
        "classifier": _normalise_classifier(_first(merged, "classifier", "model", "architecture", default="unknown")),
        "seed": _normalise_seed(_first(merged, "seed", "training_seed", default=None)),
        "run_id": _normalise_text(_first(merged, "run_id", "experiment_id", default=base.get("run_id", f"run_{ordinal}"))),
        "source_path": source_path,
        "source_artifact": _normalise_text(source_artifact),
        "stage": _normalise_text(_first(merged, "stage", "preprocessing_stage", default="final"), "final"),
        "result_kind": result_kind,
        "control_mode": _control_mode_label(row, base),
        "formal_final": result_kind == "formal_final",
    }


def _row_from_predictions(
    rows: Sequence[Mapping[str, Any]],
    base: Mapping[str, Any],
    controls: Mapping[str, Any],
    config: ReportConfig,
    *,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _base_identity(rows[0] if rows else {}, base, ordinal)
    valid: list[dict[str, Any]] = []
    for item in rows:
        label, score, participant = _prediction_fields(item)
        if label is None or score is None:
            continue
        enriched = dict(item)
        enriched["_label"] = label
        enriched["_score"] = score
        enriched["_participant"] = participant
        valid.append(enriched)
    test_rows = [
        item
        for item in valid
        if _normalise_text(_first(item, "split", "partition", "set", default="")).lower()
        in {"test", "heldout", "holdout"}
    ]
    eval_rows = test_rows or valid
    labels = np.asarray([item["_label"] for item in eval_rows], dtype=np.int64)
    scores = np.asarray([item["_score"] for item in eval_rows], dtype=np.float64)
    participants = [item["_participant"] for item in eval_rows]
    have_subjects = bool(eval_rows) and all(value not in (None, "") for value in participants)
    pair_values = [_first(item, "pair_id", "match_id", default=None) for item in eval_rows]
    pair_ids = pair_values if any(value not in (None, "") for value in pair_values) else None

    result: dict[str, Any] = dict(identity)
    result.update({field: None for field in ALL_FIELDS if field not in result})
    result["healthy_domain"] = identity["healthy_domain"]
    result["input_modalities"] = identity["input_modalities"]
    result["classifier"] = identity["classifier"]
    result["seed"] = identity["seed"]
    result["run_id"] = identity["run_id"]
    result["source_path"] = identity["source_path"]
    result["stage"] = identity["stage"]
    metric_payload: dict[str, Any] = {}
    subject_payload: dict[str, Any] = {}
    subject_predictions: dict[str, np.ndarray] | None = None
    if labels.size:
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite prediction score in {identity['run_id']}")
        metric_payload = compute_dataset_metrics(
            labels,
            scores,
            participant_ids=participants if have_subjects else None,
            pair_ids=pair_ids if have_subjects else None,
            threshold=config.threshold,
            subject_method=config.subject_method,
        )
        subject_payload = _mapping(metric_payload.get("subject"))
        subject_predictions = metric_payload.get("subject_predictions")  # type: ignore[assignment]
        result["slice_roc_auc"] = _finite(_mapping(metric_payload.get("slice")).get("roc_auc"))
        if subject_payload:
            for key in (
                "subject_roc_auc",
                "subject_pr_auc",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "bce",
                "tp",
                "tn",
                "fp",
                "fn",
            ):
                source_key = {
                    "subject_roc_auc": "roc_auc",
                    "subject_pr_auc": "average_precision",
                }.get(key, key)
                result[key] = subject_payload.get(source_key)
            result["test_subject_count_observed"] = int(subject_payload.get("n", 0))

    # Direct metric artifacts may include bootstrap/permutation values beside
    # the prediction rows.  Merge them after recomputed held-out metrics.
    for item in rows:
        for key in (
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "bootstrap_n",
            "bootstrap_valid_n",
            "bootstrap_resampling_unit",
            "recorded_bootstrap_ci_low",
            "recorded_bootstrap_ci_high",
            "recorded_bootstrap_n",
            "recorded_bootstrap_valid_n",
            "recorded_bootstrap_resampling_unit",
            "permutation_p",
            "train_final_accuracy",
            "train_final_bce",
            "train_final_subject_accuracy",
            "train_final_subject_bce",
            "tiny_overfit_accuracy",
            "tiny_overfit_bce",
            "positive_control_auc",
            "positive_control_ci_low",
            "negative_control_auc",
            "negative_control_ci_low",
            "negative_control_ci_high",
            "label_permutation_auc",
            "label_permutation_ci_low",
            "label_permutation_ci_high",
        ):
            if result.get(key) is None:
                result[key] = _finite(item.get(key))
            if key in {"bootstrap_resampling_unit", "recorded_bootstrap_resampling_unit"} and result.get(key) is None:
                value = item.get(key)
                if value not in (None, ""):
                    result[key] = str(value)
    recorded_low = _finite(result.get("recorded_bootstrap_ci_low"))
    recorded_high = _finite(result.get("recorded_bootstrap_ci_high"))
    if recorded_low is not None or recorded_high is not None:
        result["bootstrap_source"] = "recorded_subject_bootstrap"
        if recorded_low is not None:
            result["bootstrap_ci_low"] = recorded_low
        if recorded_high is not None:
            result["bootstrap_ci_high"] = recorded_high
        for result_key, recorded_key in (
            ("bootstrap_n", "recorded_bootstrap_n"),
            ("bootstrap_valid_n", "recorded_bootstrap_valid_n"),
            ("bootstrap_resampling_unit", "recorded_bootstrap_resampling_unit"),
        ):
            if result.get(recorded_key) is not None:
                result[result_key] = result[recorded_key]
    if result.get("control_mode") == "tiny_overfit":
        if result.get("train_final_accuracy") is not None:
            result["tiny_overfit_accuracy"] = result["train_final_accuracy"]
        if result.get("train_final_bce") is not None:
            result["tiny_overfit_bce"] = result["train_final_bce"]
    for split in ("train", "val", "test"):
        result[f"{split}_slices"] = _count_for_split(valid, split, subjects=False)
        result[f"{split}_subjects"] = _count_for_split(valid, split, subjects=True)

    if subject_predictions is not None:
        subject_pair_ids = subject_predictions.get("pair_ids")
        if subject_pair_ids is not None:
            pair_values = list(np.asarray(subject_pair_ids, dtype=object).tolist())
            if not any(value not in (None, "") for value in pair_values):
                subject_pair_ids = None
        bootstrap = bootstrap_subject_auc(
            subject_predictions["labels"],
            subject_predictions["scores"],
            n_bootstrap=config.bootstrap_replicates,
            seed=config.seed,
            pair_ids=subject_pair_ids,
            confidence=config.confidence,
        )
        if result.get("bootstrap_source") is None:
            result["bootstrap_source"] = "recomputed_subject_bootstrap"
        if result.get("bootstrap_ci_low") is None:
            result["bootstrap_ci_low"] = bootstrap["ci_low"]
        if result.get("bootstrap_ci_high") is None:
            result["bootstrap_ci_high"] = bootstrap["ci_high"]
        if result.get("bootstrap_n") is None:
            result["bootstrap_n"] = bootstrap["n_bootstrap"]
        if result.get("bootstrap_valid_n") is None:
            result["bootstrap_valid_n"] = bootstrap["n_valid"]
        if result.get("bootstrap_resampling_unit") is None:
            result["bootstrap_resampling_unit"] = bootstrap["resampling_unit"]
    for control_name, field in (
        ("tiny_overfit", "tiny_overfit_pass"),
        ("positive_control", "positive_control_pass"),
        ("negative_control", "negative_control_pass"),
        ("label_permutation", "label_permutation_pass"),
    ):
        result[field] = _control_value(rows, controls, control_name)
    _apply_gate_fields(result, config)
    return result, {
        "prediction_rows": valid,
        "subject_predictions": subject_predictions,
        "training_files": [],
    }


def _row_from_metrics(
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    controls: Mapping[str, Any],
    config: ReportConfig,
    *,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _base_identity(row, base, ordinal)
    result.update({field: None for field in ALL_FIELDS if field not in result})
    for field in (
        "healthy_domain",
        "input_modalities",
        "classifier",
        "seed",
        "run_id",
        "source_path",
        "stage",
    ):
        result[field] = result.get(field)
    for name in (
        "slice_roc_auc",
        "subject_roc_auc",
        "subject_pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "bce",
    ):
        result[name] = _metric_from_mapping(row, name, level="slice" if name.startswith("slice_") else "subject")
    # Generic direct ``metrics`` and ``result`` sections are common in runner
    # outputs.  Preserve top-level values as the final fallback.
    for section_name in ("metrics", "result", "summary"):
        nested = _mapping(row.get(section_name))
        if not nested:
            continue
        for name in (
            "slice_roc_auc",
            "subject_roc_auc",
            "subject_pr_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "bce",
        ):
            if result.get(name) is None:
                result[name] = _metric_from_mapping(nested, name, level="slice" if name.startswith("slice_") else "subject")
    for split in ("train", "val", "test"):
        result[f"{split}_subjects"] = _direct_count(row, split, True)
        result[f"{split}_slices"] = _direct_count(row, split, False)
    for field in (
        "train_final_accuracy",
        "train_final_bce",
        "train_final_subject_accuracy",
        "train_final_subject_bce",
        "tiny_overfit_accuracy",
        "tiny_overfit_bce",
    ):
        result[field] = _finite(row.get(field))
    if result.get("control_mode") == "tiny_overfit":
        result["tiny_overfit_accuracy"] = result.get("train_final_accuracy")
        result["tiny_overfit_bce"] = result.get("train_final_bce")
    bootstrap_section = _mapping(row.get("bootstrap"))
    result["bootstrap_ci_low"] = _finite(
        _first(row, "bootstrap_ci_low", "subject_auc_ci_low", default=_first(bootstrap_section, "ci_low", "lower", "auc_ci_low"))
    )
    result["bootstrap_ci_high"] = _finite(
        _first(row, "bootstrap_ci_high", "subject_auc_ci_high", default=_first(bootstrap_section, "ci_high", "upper", "auc_ci_high"))
    )
    result["bootstrap_n"] = _finite(_first(row, "bootstrap_n", "n_bootstrap", default=_first(bootstrap_section, "n_bootstrap")))
    result["bootstrap_valid_n"] = _finite(_first(row, "bootstrap_valid_n", "n_valid", default=_first(bootstrap_section, "n_valid")))
    result["bootstrap_resampling_unit"] = _first(
        row,
        "bootstrap_resampling_unit",
        "resampling_unit",
        default=_first(bootstrap_section, "resampling_unit"),
    )
    result["recorded_bootstrap_ci_low"] = _finite(
        _first(row, "recorded_bootstrap_ci_low", default=result.get("bootstrap_ci_low"))
    )
    result["recorded_bootstrap_ci_high"] = _finite(
        _first(row, "recorded_bootstrap_ci_high", default=result.get("bootstrap_ci_high"))
    )
    result["recorded_bootstrap_n"] = _finite(
        _first(row, "recorded_bootstrap_n", default=result.get("bootstrap_n"))
    )
    result["recorded_bootstrap_valid_n"] = _finite(
        _first(row, "recorded_bootstrap_valid_n", default=result.get("bootstrap_valid_n"))
    )
    result["recorded_bootstrap_resampling_unit"] = _first(
        row,
        "recorded_bootstrap_resampling_unit",
        default=result.get("bootstrap_resampling_unit"),
    )
    if result["recorded_bootstrap_ci_low"] is not None or result["recorded_bootstrap_ci_high"] is not None:
        result["bootstrap_source"] = "recorded_subject_bootstrap"
    permutation_section = _mapping(row.get("permutation"))
    result["permutation_p"] = _finite(
        _first(row, "permutation_p", "permutation_p_value", "p_value", default=_first(permutation_section, "p_value", "p"))
    )
    result["permutation_n"] = _finite(_first(row, "permutation_n", "n_permutations", default=_first(permutation_section, "n_permutations")))
    result["permutation_null_mean"] = _finite(_first(row, "permutation_null_mean", default=_first(permutation_section, "null_mean", "null_auc_mean")))
    result["permutation_null_ci_low"] = _finite(_first(row, "permutation_null_ci_low", default=_first(permutation_section, "null_ci_low", "ci_low")))
    result["permutation_null_ci_high"] = _finite(_first(row, "permutation_null_ci_high", default=_first(permutation_section, "null_ci_high", "ci_high")))
    controls_section = _mapping(row.get("controls"))
    for control_name, field in (
        ("tiny_overfit", "tiny_overfit_pass"),
        ("positive_control", "positive_control_pass"),
        ("negative_control", "negative_control_pass"),
        ("label_permutation", "label_permutation_pass"),
    ):
        result[field] = _control_value([row], {**controls, **controls_section}, control_name)
    for field in (
        "tiny_overfit_accuracy",
        "tiny_overfit_bce",
        "positive_control_auc",
        "positive_control_ci_low",
        "negative_control_auc",
        "negative_control_ci_low",
        "negative_control_ci_high",
        "label_permutation_auc",
        "label_permutation_ci_low",
        "label_permutation_ci_high",
    ):
        result[field] = _finite(row.get(field))
    _apply_gate_fields(result, config)
    return result, {"prediction_rows": [], "subject_predictions": None, "training_files": []}


def _group_rows(rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        grouped[_row_key(row, base, ordinal)].append(dict(row))
    return list(grouped.values())


def _training_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    names = {"training_metrics.csv", "training_curve.csv", "training_curves.csv", "history.csv"}
    return [path for path in root.rglob("*") if path.is_file() and path.name in names]


def _authoritative_source_files(root: Path | None) -> list[str]:
    """Collect preparation audit files for reference, never for overwrite."""

    if root is None or not root.exists():
        return []
    names = {
        "audit.json",
        "experiment_manifest.json",
        "run_manifest.json",
        "manifest.json",
        "run_identity.json",
    }
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name in names
        and path.name not in {"report_audit.json", "report_manifest.json"}
        and not any(part.lower() in {"cache", "images", "checkpoints"} for part in path.parts)
    ]
    return [str(path.resolve()) for path in sorted(paths, key=lambda item: str(item).lower())]


def _load_source(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return metric/prediction rows, base metadata, and context."""

    # A report can intentionally combine several already-finished run
    # directories (for example tiny/positive/negative controls) while
    # excluding a still-running sibling.  Preserve each directory's identity
    # on its rows; flattening all children under one parent would merge their
    # predictions into a false single experiment.
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        if all(isinstance(item, (str, Path)) for item in source):
            combined_rows: list[dict[str, Any]] = []
            combined_context: dict[str, Any] = {
                "root": None,
                "prediction_paths": [],
                "metric_paths": [],
                "controls": {},
                "training_files": [],
                "authoritative_paths": [],
                "source_roots": [],
                "parity_summary": None,
                "stage1_summary": None,
                "shuffle_diagnostic_summary": None,
                "protocol_v2_summary": None,
                "calibration_summary": None,
                "model_grid_v3_summary": None,
                "model_grid_v3_paths": [],
                "calibration_only": False,
            }
            identity_keys = {
                "run_id",
                "source_path",
                "healthy_domain",
                "healthy_cohort",
                "cohort",
                "input_modalities",
                "modalities",
                "classifier",
                "model",
                "seed",
                "split_seed",
                "stage",
                "result_kind",
                "formal_final",
                "control_type",
                "control_name",
                "control_kind",
                "run_kind",
                "run_role",
                "purpose",
                "permutation_mode",
                "mode",
                "control_mode",
                "tiny",
            }
            for item in source:
                rows, base, context = _load_source(item)
                combined_context["source_roots"].append(str(item))
                # Calibration observed rows are carried in a dedicated
                # context section.  They are intentionally omitted from the
                # generic summary rows so they cannot be mistaken for a
                # confirmatory final fit or paired with ANDi AP.
                rows_to_append = [] if context.get("calibration_summary") is not None else rows
                for row in rows_to_append:
                    enriched = dict(row)
                    for key in identity_keys:
                        if key not in enriched and key in base:
                            enriched[key] = base[key]
                    combined_rows.append(enriched)
                for key in ("prediction_paths", "metric_paths", "training_files", "authoritative_paths"):
                    combined_context[key].extend(context.get(key, []))
                for key in ("parity_summary", "stage1_summary", "shuffle_diagnostic_summary", "protocol_v2_summary", "calibration_summary", "model_grid_v3_summary"):
                    if context.get(key) is not None:
                        combined_context[key] = context[key]
                combined_context["model_grid_v3_paths"].extend(context.get("model_grid_v3_paths", []))
                for name, value in context.get("controls", {}).items():
                    # A later duplicate control name is only a fallback; a
                    # finished explicit gate already present wins.
                    combined_context["controls"].setdefault(name, value)
            for key in ("prediction_paths", "metric_paths", "training_files", "authoritative_paths", "model_grid_v3_paths"):
                combined_context[key] = list(dict.fromkeys(combined_context[key]))
            return combined_rows, {"run_id": "combined", "source_path": "<combined>"}, combined_context

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            payload = _read_data_file(path)
            # Runtime keeps immutable, timestamped v3 revisions beside the
            # original directory (for example
            # ``model_grid_v3_fullcandidate_20260917``).  Only the explicitly
            # supplied build_summary is read; report never chooses an older
            # sibling revision by scanning the parent directory.
            is_model_grid_v3 = (
                path.name.lower() == "build_summary.json"
                and path.parent.name.lower().startswith("model_grid_v3")
            )
            if is_model_grid_v3:
                model_grid_v3_summary = _summarise_model_grid_v3(payload, path)
                return [], {"run_id": "model_grid_v3_build", "source_path": str(path.resolve())}, {
                    "root": path.parent,
                    "prediction_paths": [],
                    "metric_paths": [],
                    "controls": {},
                    "training_files": [],
                    "authoritative_paths": [],
                    "parity_summary": None,
                    "stage1_summary": None,
                    "shuffle_diagnostic_summary": None,
                    "protocol_v2_summary": None,
                    "calibration_summary": None,
                    "model_grid_v3_summary": model_grid_v3_summary,
                    "model_grid_v3_paths": (
                        [str(path.resolve())]
                        + list(model_grid_v3_summary.get("audit_paths", []))
                        + list(model_grid_v3_summary.get("sidecar_paths", []))
                    ) if model_grid_v3_summary else [str(path.resolve())],
                    "calibration_only": False,
                }
            rows = _rows_from_payload(payload)
            parity_summary = _summarise_parity(payload) if path.name.lower() == "parity.json" else None
            stage1_summary = _summarise_stage1(payload) if path.name.lower() == "audit.json" else None
            shuffle_diagnostic_summary = _summarise_shuffle_diagnostic(path.parent) if path.name.lower() in {
                "shuffle_diagnostic_audit.json",
                "shuffle_diagnostic_audit.md",
                "shuffle_saved_model_orientation.json",
            } else None
            protocol_v2_summary = _summarise_protocol_v2(path.parent) if path.name.lower() in {
                "protocol.json",
                "preflight.json",
                "gate_true_test.json",
                "gate_random_test_labels.json",
                "gate.json",
                "supplementary_diagnostic_summary.json",
                "supplementary_diagnostic_summary.md",
            } else None
            calibration_summary = (
                _summarise_calibration(path.parent.parent)
                if path.name.lower() == "result.json" and path.parent.name.lower() == "observed"
                else None
            )
            authoritative_names = {
                "audit.json",
                "experiment_manifest.json",
                "run_manifest.json",
                "manifest.json",
                "run_identity.json",
            }
            return rows, {"run_id": path.stem, "source_path": str(path.resolve())}, {
                "root": path.parent,
                "prediction_paths": [str(path.resolve())] if any(_prediction_fields(item)[0] is not None for item in rows) else [],
                "metric_paths": [str(path.resolve())],
                "controls": {},
                "training_files": [],
                "parity_summary": parity_summary,
                "stage1_summary": stage1_summary,
                "shuffle_diagnostic_summary": shuffle_diagnostic_summary,
                "protocol_v2_summary": protocol_v2_summary,
                "calibration_summary": calibration_summary,
                "model_grid_v3_summary": None,
                "model_grid_v3_paths": [],
                "calibration_only": calibration_summary is not None,
                # A directly supplied audit/manifest is authoritative by
                # itself.  Recursing through its parent would accidentally
                # reference unrelated sibling runs (including runs still in
                # progress) in the interim report.
                "authoritative_paths": [str(path.resolve())] if path.name in authoritative_names else [],
            }
        root = path
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(root)
        build_summary_path = root / "build_summary.json"
        if root.name.lower().startswith("model_grid_v3") and build_summary_path.is_file():
            model_grid_v3_summary = _summarise_model_grid_v3(_read_json(build_summary_path), build_summary_path)
            return [], {"run_id": "model_grid_v3_build", "source_path": str(root.resolve())}, {
                "root": root,
                "prediction_paths": [],
                "metric_paths": [],
                "controls": {},
                "training_files": [],
                "authoritative_paths": [],
                "parity_summary": None,
                "stage1_summary": None,
                "shuffle_diagnostic_summary": None,
                "protocol_v2_summary": None,
                "calibration_summary": None,
                "model_grid_v3_summary": model_grid_v3_summary,
                "model_grid_v3_paths": (
                    [str(build_summary_path.resolve())]
                    + list(model_grid_v3_summary.get("audit_paths", []))
                    + list(model_grid_v3_summary.get("sidecar_paths", []))
                ) if model_grid_v3_summary else [str(build_summary_path.resolve())],
                "calibration_only": False,
            }
        base = _load_run_metadata(root)
        prediction_rows, prediction_paths = _load_prediction_rows(root)
        metric_rows, metric_paths = _load_metric_rows(root)
        controls = _load_controls(root)
        shuffle_diagnostic_summary = _summarise_shuffle_diagnostic(root)
        protocol_v2_summary = _summarise_protocol_v2(root)
        calibration_summary = _summarise_calibration(root)
        if calibration_summary:
            # The calibration observed fit is diagnostic only.  Preserve its
            # metrics in the dedicated context table but never let it become
            # a generic formal/control summary row.
            prediction_rows = []
            metric_rows = []
            prediction_paths = []
            metric_paths = []
            base.update(
                {
                    "control_mode": "calibration_observed",
                    "result_kind": "diagnostic_calibration",
                    "formal_final": False,
                }
            )
        if protocol_v2_summary:
            # The v2 seed result files are diagnostic fits.  They are kept in
            # the summary for traceability but can never enter formal rows.
            base.update(
                {
                    "control_mode": "balanced_shuffle_v2",
                    "control_type": "balanced_shuffle_v2",
                    "result_kind": "diagnostic_formalfit",
                    "formal_final": False,
                }
            )
        return prediction_rows + metric_rows, base, {
            "root": root,
            "prediction_paths": prediction_paths,
            "metric_paths": metric_paths,
            "controls": controls,
            "training_files": [str(path.resolve()) for path in _training_files(root)],
            "authoritative_paths": _authoritative_source_files(root),
            "parity_summary": None,
            "stage1_summary": None,
            "shuffle_diagnostic_summary": shuffle_diagnostic_summary,
            "protocol_v2_summary": protocol_v2_summary,
            "calibration_summary": calibration_summary,
            "model_grid_v3_summary": None,
            "model_grid_v3_paths": [],
            "calibration_only": calibration_summary is not None,
        }
    if isinstance(source, Mapping):
        if isinstance(source.get("runs"), list):
            rows: list[dict[str, Any]] = []
            for run in source["runs"]:
                if isinstance(run, Mapping):
                    rows.extend(_rows_from_payload(run))
            return rows, {"run_id": "runs"}, {"root": None, "prediction_paths": [], "metric_paths": [], "controls": {}, "training_files": [], "authoritative_paths": [], "parity_summary": None, "stage1_summary": None, "shuffle_diagnostic_summary": None, "protocol_v2_summary": None, "calibration_summary": None, "model_grid_v3_summary": None, "model_grid_v3_paths": [], "calibration_only": False}
        config = _mapping(source.get("config"))
        base: dict[str, Any] = {"run_id": "mapping", "source_path": "<mapping>"}
        for key, value in config.items():
            if key not in base:
                base[key] = value
        for key in (
            "healthy_domain",
            "healthy_cohort",
            "cohort",
            "comparison",
            "input_modalities",
            "modalities",
            "classifier",
            "model",
            "seed",
            "stage",
            "run_id",
            "experiment_id",
            "result_kind",
            "formal_final",
            "control_type",
            "control_name",
            "control_kind",
            "run_kind",
            "run_role",
            "purpose",
            "permutation_mode",
            "permutation_index",
        ):
            if key in source:
                base[key] = source[key]
        controls = _mapping(source.get("controls"))
        return _rows_from_payload(source), base, {"root": None, "prediction_paths": [], "metric_paths": [], "controls": controls, "training_files": [], "authoritative_paths": [], "parity_summary": None, "stage1_summary": None, "shuffle_diagnostic_summary": None, "protocol_v2_summary": None, "calibration_summary": None, "model_grid_v3_summary": None, "model_grid_v3_paths": [], "calibration_only": False}
    if isinstance(source, Sequence):
        return [dict(item) for item in source if isinstance(item, Mapping)], {"run_id": "records", "source_path": "<records>"}, {"root": None, "prediction_paths": [], "metric_paths": [], "controls": {}, "training_files": [], "authoritative_paths": [], "parity_summary": None, "stage1_summary": None, "shuffle_diagnostic_summary": None, "protocol_v2_summary": None, "calibration_summary": None, "model_grid_v3_summary": None, "model_grid_v3_paths": [], "calibration_only": False}
    raise TypeError("source must be a run directory, artifact file, mapping, or sequence of mappings")


def build_summary_rows(source: Any, *, config: ReportConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate saved run artifacts into stable summary rows.

    The return value is ``(rows, context)``.  ``context`` contains only paths
    and prediction arrays needed by plotting; it is not written as raw arrays
    into the manifest.
    """

    cfg = config or ReportConfig()
    raw_rows, base, context = _load_source(source)
    if context.get("calibration_only"):
        # Observed calibration metrics are reported in their own auditable
        # section; they are not generic rows and cannot participate in formal
        # matrix/AP conclusions.
        raw_rows = []
    prediction_rows = [row for row in raw_rows if _prediction_fields(row)[0] is not None]
    metric_rows = [row for row in raw_rows if _prediction_fields(row)[0] is None]
    controls = context.get("controls", {})
    output_rows: list[dict[str, Any]] = []
    prediction_context: list[dict[str, Any]] = []
    if prediction_rows:
        for ordinal, group in enumerate(_group_rows(prediction_rows, base)):
            row, extra = _row_from_predictions(group, base, controls, cfg, ordinal=ordinal)
            output_rows.append(row)
            extra["row_index"] = len(output_rows) - 1
            prediction_context.append(extra)
    for ordinal, group in enumerate(_group_rows(metric_rows, base), start=len(output_rows)):
        row, extra = _row_from_metrics(group[0], base, controls, cfg, ordinal=ordinal)
        output_rows.append(row)
        extra["row_index"] = len(output_rows) - 1
        prediction_context.append(extra)
    context["prediction_context"] = prediction_context
    context["raw_row_count"] = len(raw_rows)
    context["formal_final_row_count"] = sum(1 for row in output_rows if _is_formal_final_row(row))
    context["excluded_row_count"] = sum(1 for row in output_rows if not _is_formal_final_row(row))
    return output_rows, context


def _csv_value(value: Any, *, field: str | None = None) -> Any:
    if value is None:
        return ""
    if field == "formal_final" and isinstance(value, bool):
        # ``formal_final`` is a predicate, not a gate status.  Keep false as
        # false in CSV instead of serialising it as the visually misleading
        # gate token ``FAIL``.
        return "true" if value else "false"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write summary rows with required fields first and missing values blank."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field), field=field) for field in ALL_FIELDS})
    return destination


def _fingerprint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    item: dict[str, Any] = {"path": str(target.resolve())}
    try:
        stat = target.stat()
    except OSError as exc:
        item["error"] = str(exc)
        return item
    item.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        item["sha256"] = digest.hexdigest()
    except OSError as exc:
        item["error"] = str(exc)
    return item


# The secondary grid is intentionally kept separate from the primary four-cell
# full-retrained null family.  In particular, a held-out fixed-classifier pair
# swap p-value is not interchangeable with either family-level p-value.
SECONDARY_48_FAMILY = "secondary_48"
PRIMARY_FULL_RETRAINED_HOLM4_FAMILY = "primary_full_retrained_holm4"
DEFAULT_SECONDARY_SEEDS = (73, 173, 273)


def _secondary_identity_token(value: Any) -> tuple[str, str]:
    """Return a type-preserving key for a subject identifier.

    JSON commonly serialises IDs as strings.  Keeping the original type in the
    key prevents an integer ``1`` and a string ``"1"`` from being silently
    treated as the same participant when two seed artifacts disagree.
    """

    if value is None or str(value).strip() == "":
        raise ValueError("subject identifier is missing or empty")
    if isinstance(value, bool):
        raise ValueError("subject identifier must not be boolean")
    return (type(value).__name__, str(value))


def _secondary_result_rows(payload: Any, *, evaluation_split: str = "test") -> list[Mapping[str, Any]]:
    """Read already aggregated subject rows from one runner result.

    The runner's stable output is ``test_subject_predictions``.  A mapping of
    arrays is accepted for compatibility with the in-memory metrics result,
    but slice-level ``test_predictions`` is deliberately not a fallback: the
    secondary analysis must aggregate identical subject observations across
    seeds, never silently substitute slices.
    """

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        candidates: list[Any] = [payload]
    elif isinstance(payload, Mapping):
        split = str(evaluation_split).strip().lower() or "test"
        candidates = [
            payload.get(f"{split}_subject_predictions"),
            payload.get("subject_prediction_rows"),
            payload.get("subject_predictions"),
        ]
        nested_split = _mapping(payload.get(split))
        if nested_split:
            candidates.extend([
                nested_split.get("subject_predictions"),
                nested_split.get("subject_prediction_rows"),
            ])
    else:
        candidates = []

    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            return [row for row in candidate if isinstance(row, Mapping)]
        if isinstance(candidate, Mapping):
            labels = candidate.get("labels", candidate.get("y_true"))
            scores = candidate.get("scores", candidate.get("probabilities", candidate.get("probability")))
            subjects = candidate.get("participant_ids", candidate.get("subject_ids", candidate.get("subjects")))
            if labels is None or scores is None or subjects is None:
                continue
            labels_list = list(np.asarray(labels, dtype=object).reshape(-1))
            scores_list = list(np.asarray(scores, dtype=object).reshape(-1))
            subjects_list = list(np.asarray(subjects, dtype=object).reshape(-1))
            if not (len(labels_list) == len(scores_list) == len(subjects_list)):
                raise ValueError("subject prediction arrays have incompatible lengths")
            pairs = candidate.get("pair_ids")
            pair_list = list(np.asarray(pairs, dtype=object).reshape(-1)) if pairs is not None else [None] * len(labels_list)
            if len(pair_list) != len(labels_list):
                raise ValueError("subject prediction pair_ids have incompatible length")
            return [
                {
                    "participant_id": subject,
                    "label": label,
                    "probability": score,
                    "pair_id": pair,
                }
                for subject, label, score, pair in zip(subjects_list, labels_list, scores_list, pair_list)
            ]
    raise ValueError(
        f"result does not contain {evaluation_split!r} subject predictions; "
        "slice-level predictions cannot be used for fixed-seed subject ensembles"
    )


def _secondary_subject_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalise one seed's subject prediction rows."""

    if not rows:
        raise ValueError("subject prediction rows are empty")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        subject = _first(row, "participant_id", "subject_id", "participant", "subject", "patient_id")
        token = _secondary_identity_token(subject)
        if token in seen:
            raise ValueError(f"duplicate subject identifier within seed: {subject!r}")
        seen.add(token)
        label_raw = _first(row, "label", "y_true", "target")
        try:
            label_float = float(label_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"subject {subject!r} has a non-numeric label") from exc
        if not np.isfinite(label_float) or label_float not in (0.0, 1.0):
            raise ValueError(f"subject {subject!r} has a non-binary label")
        probability = _finite(_first(row, "probability", "score", "prob", "p"))
        if probability is None or probability < 0.0 or probability > 1.0:
            raise ValueError(f"subject {subject!r} has an invalid probability")
        pair = _first(row, "pair_id", "pair", default=None)
        if pair in (None, ""):
            pair = None
        output.append(
            {
                "subject_id": subject,
                "subject_token": token,
                "label": int(label_float),
                "probability": float(probability),
                "pair_id": pair,
            }
        )
    return output


def _secondary_cell_id(cell: Mapping[str, Any], ordinal: int | None = None) -> str:
    explicit = _first(cell, "cell_id", "analysis_id", "comparison_id", default=None)
    if explicit not in (None, ""):
        return str(explicit)
    cohort = _first(cell, "cohort", "healthy_domain", "comparison", default="unknown")
    modalities = _normalise_modalities(_first(cell, "modalities", "input_modalities", default="")) or "unknown"
    classifier = _normalise_classifier(_first(cell, "classifier", "model", default="unknown"))
    split = _normalise_text(_first(cell, "evaluation_split", "split", default="test"), "test")
    base = f"{cohort}|{modalities}|{classifier}|{split}"
    return base if ordinal is None else f"{base}|{ordinal}"


def _secondary_seed_entries(cell: Mapping[str, Any]) -> list[Any]:
    for key in ("seed_results", "seeds", "runs"):
        value = cell.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        if isinstance(value, Mapping):
            entries: list[Any] = []
            for seed, item in value.items():
                if isinstance(item, Mapping):
                    entry = dict(item)
                    entry.setdefault("seed", seed)
                else:
                    entry = {"seed": seed, "result_path": item}
                entries.append(entry)
            return entries
    # ``result_paths`` is accepted only as an explicit list; report discovery
    # never glob-scans directories for a purported formal cell.
    value = cell.get("result_paths")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if isinstance(value, Mapping):
        return [{"seed": seed, "result_path": path} for seed, path in value.items()]
    return []


def _secondary_expected_seeds(
    cell: Mapping[str, Any],
    requested: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the frozen seed family for one secondary classifier cell."""

    classifier = _normalise_classifier(_first(cell, "classifier", "model", default="unknown")).lower()
    if classifier in {"logistic", "statistical", "statistical_logistic", "logreg", "logistic_regression"}:
        return (73,)
    values = tuple(int(seed) for seed in (DEFAULT_SECONDARY_SEEDS if requested is None else requested))
    if not values or len(set(values)) != len(values):
        raise ValueError("expected_seeds must be non-empty and unique")
    return values


def _secondary_load_seed_entry(entry: Any, *, base_dir: Path) -> tuple[int | None, Any, str | None]:
    """Resolve one explicit seed result path or inline payload."""

    path_text: str | None = None
    declared_seed: int | None = None
    payload: Any = None
    if isinstance(entry, (str, Path)):
        path_text = str(entry)
    elif isinstance(entry, Mapping):
        declared_seed = _normalise_seed(_first(entry, "seed", "random_seed", default=None))
        path_value = _first(entry, "result_path", "artifact_path", "path", default=None)
        if path_value not in (None, ""):
            path_text = str(path_value)
        inline = _first(entry, "result", "payload", default=None)
        if inline is not None:
            payload = inline
        elif any(key in entry for key in ("test_subject_predictions", "subject_predictions", "subject_prediction_rows")):
            payload = entry
    else:
        raise ValueError("seed entry must be a result path or mapping")

    resolved: Path | None = None
    if path_text is not None:
        resolved = Path(path_text)
        if not resolved.is_absolute():
            resolved = base_dir / resolved
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"secondary seed result not found: {resolved}")
        payload = _read_json(resolved)
    if payload is None:
        raise ValueError("seed entry has neither inline result payload nor result_path")
    payload_mapping = _mapping(payload)
    payload_seed = _normalise_seed(_first(payload_mapping, "seed", "random_seed", default=None))
    if declared_seed is not None and payload_seed is not None and declared_seed != payload_seed:
        raise ValueError(f"declared seed {declared_seed} disagrees with result seed {payload_seed}")
    return declared_seed if declared_seed is not None else payload_seed, payload, str(resolved) if resolved else None


def _secondary_p_value(cell: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:
    """Read a secondary-family p-value with an explicit inference contract.

    A fixed-classifier pair swap is valid for the secondary family only after
    the declared fixed-seed subject-probability ensemble has been formed.  An
    unqualified single-run/legacy ``heldout_pair_swap`` value remains
    excluded; the namespaced source below prevents it being confused with the
    primary full-retrained family.
    """

    source = _normalise_text(_first(cell, "p_value_source", "inference_source", default=""), "").lower()
    legal_conditional_sources = {
        "secondary_48_conditional_heldout_pair_swap",
        "secondary_ensemble_heldout_pair_swap",
        "fixed_classifier_subject_ensemble_pair_swap",
    }
    if source in legal_conditional_sources:
        unit = _normalise_text(
            _first(cell, "inference_unit", "conditional_on", "ensemble_unit", default=""),
            "",
        ).lower()
        if unit and not any(token in unit for token in ("ensemble", "fixed_classifier", "fixed-classifier")):
            return None, source, "conditional source does not declare a fixed subject ensemble"
        raw = _first(cell, "secondary_p_value", "p_value", default=None)
        if raw is None:
            return None, source, "secondary ensemble pair-swap p-value missing"
        value = _finite(raw)
        if value is None or value < 0.0 or value > 1.0:
            return None, source, "secondary p-value is not finite in [0, 1]"
        return float(value), source, None
    if source in {"heldout_pair_swap", "paired_swap", "conditional_heldout", "full_retrained", "primary_full_retrained_holm4"}:
        return None, source, "conditional or primary-family p-value cannot enter secondary_48"
    raw = _first(cell, "secondary_p_value", default=None)
    if raw is None and source in {"secondary_48", "secondary_subject_ensemble", "subject_ensemble", "ensemble"}:
        raw = _first(cell, "p_value", default=None)
    if raw is None:
        return None, source or None, "secondary p-value missing (expected secondary_p_value or explicit secondary p_value_source)"
    value = _finite(raw)
    if value is None or value < 0.0 or value > 1.0:
        return None, source or "secondary_p_value", "secondary p-value is not finite in [0, 1]"
    return float(value), source or "secondary_p_value", None


def _primary_full_retrained_p_value(cell: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:
    """Read only an explicitly labelled full-retrained primary p-value."""

    source = _normalise_text(_first(cell, "p_value_source", "inference_source", default=""), "").lower()
    if source in {"heldout_pair_swap", "paired_swap", "conditional_heldout", "secondary_48", "secondary_subject_ensemble"}:
        return None, source, "conditional or secondary-family p-value cannot enter primary full-retrained Holm4"
    raw = _first(cell, "full_retrained_p_value", default=None)
    if raw is None and source in {"full_retrained", "full_retrained_pair_null", "primary_full_retrained_holm4"}:
        raw = _first(cell, "p_value", default=None)
    if raw is None:
        return None, source or None, "full-retrained p-value missing (heldout swap p-values are not accepted)"
    value = _finite(raw)
    if value is None or value < 0.0 or value > 1.0:
        return None, source or "full_retrained_p_value", "full-retrained p-value is not finite in [0, 1]"
    return float(value), source or "full_retrained_p_value", None


def aggregate_fixed_seed_subject_ensemble(
    cell: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    expected_seeds: Sequence[int] | None = None,
    evaluation_split: str = "test",
) -> dict[str, Any]:
    """Aggregate one explicit fixed-seed cell at subject level.

    A valid cell contains one result for every expected seed and exactly the
    same subject identifiers and labels in every result.  Any missing seed,
    duplicate seed, subject-set mismatch, label mismatch, or malformed score
    returns ``FAIL_CLOSED`` with no ensemble metrics.  This is intentionally a
    report-side reader; it never searches for unlisted result artifacts.
    """

    expected = _secondary_expected_seeds(cell, expected_seeds)
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    entries = _secondary_seed_entries(cell)
    cell_id = _secondary_cell_id(cell)
    result: dict[str, Any] = {
        "cell_id": cell_id,
        "cohort": _normalise_domain(_first(cell, "cohort", "healthy_domain", "comparison", default="unknown")),
        "modalities": _normalise_modalities(_first(cell, "modalities", "input_modalities", default="")),
        "classifier": _normalise_classifier(_first(cell, "classifier", "model", default="unknown")),
        "evaluation_split": _normalise_text(_first(cell, "evaluation_split", "split", default=evaluation_split), evaluation_split),
        "family": SECONDARY_48_FAMILY,
        "expected_seeds": list(expected),
        "status": "FAIL_CLOSED",
        "formal_eligible": False,
        "errors": [],
        "seed_results": [],
    }
    if not entries:
        result["errors"] = ["cell has no explicit seed_results/result_paths"]
        return result
    observed: dict[int, tuple[Any, str | None]] = {}
    for entry in entries:
        try:
            seed, payload, path_text = _secondary_load_seed_entry(entry, base_dir=root)
            if seed is None:
                raise ValueError("seed is missing from entry and result payload")
            if seed in observed:
                raise ValueError(f"duplicate result for seed {seed}")
            observed[int(seed)] = (payload, path_text)
        except (OSError, TypeError, ValueError) as exc:
            result["errors"].append(str(exc))
    missing = [seed for seed in expected if seed not in observed]
    extra = sorted(seed for seed in observed if seed not in expected)
    if missing:
        result["errors"].append(f"missing expected seeds: {missing}")
    if extra:
        result["errors"].append(f"unexpected seeds: {extra}")
    if result["errors"]:
        result["missing_seeds"] = missing
        result["unexpected_seeds"] = extra
        return result

    per_seed: dict[int, list[dict[str, Any]]] = {}
    try:
        for seed in expected:
            payload, path_text = observed[seed]
            rows = _secondary_subject_rows(_secondary_result_rows(payload, evaluation_split=evaluation_split))
            per_seed[seed] = rows
            result["seed_results"].append({
                "seed": seed,
                "path": path_text,
                "fingerprint": _fingerprint(path_text) if path_text else None,
                "subject_count": len(rows),
            })
        reference = per_seed[expected[0]]
        reference_tokens = [row["subject_token"] for row in reference]
        reference_set = set(reference_tokens)
        labels_by_token = {row["subject_token"]: row["label"] for row in reference}
        pair_by_token = {row["subject_token"]: row["pair_id"] for row in reference}
        score_by_token: dict[tuple[str, str], list[float]] = {token: [row["probability"]] for token, row in ((r["subject_token"], r) for r in reference)}
        for seed in expected[1:]:
            rows = per_seed[seed]
            tokens = [row["subject_token"] for row in rows]
            if set(tokens) != reference_set or len(tokens) != len(reference_tokens):
                raise ValueError(f"subject identifier set mismatch for seed {seed}")
            for row in rows:
                token = row["subject_token"]
                if row["label"] != labels_by_token[token]:
                    raise ValueError(f"label mismatch for subject {row['subject_id']!r} at seed {seed}")
                reference_pair = pair_by_token[token]
                if reference_pair not in (None, "") or row["pair_id"] not in (None, ""):
                    if reference_pair != row["pair_id"]:
                        raise ValueError(f"pair_id mismatch for subject {row['subject_id']!r} at seed {seed}")
                score_by_token[token].append(row["probability"])
        if any(len(values) != len(expected) for values in score_by_token.values()):
            raise ValueError("not every subject has one probability per expected seed")
        mean_scores = np.asarray([float(np.mean(score_by_token[token])) for token in reference_tokens], dtype=np.float64)
        labels = np.asarray([labels_by_token[token] for token in reference_tokens], dtype=np.int64)
        participants = np.asarray([row["subject_id"] for row in reference], dtype=object)
        pairs = [pair_by_token[token] for token in reference_tokens]
        pair_values = np.asarray(pairs, dtype=object) if all(value not in (None, "") for value in pairs) else None
        metrics = compute_dataset_metrics(labels, mean_scores, participant_ids=participants, pair_ids=pair_values)
        subject_metrics = _mapping(metrics.get("subject"))
        result["subjects"] = [
            {
                "subject_id": row["subject_id"],
                "label": int(row["label"]),
                "mean_probability": float(np.mean(score_by_token[row["subject_token"]])),
                "seed_probabilities": [float(value) for value in score_by_token[row["subject_token"]]],
                "pair_id": pair_by_token[row["subject_token"]],
            }
            for row in reference
        ]
        result["n_subjects"] = len(reference)
        result["metrics"] = {
            "subject_roc_auc": _finite(subject_metrics.get("roc_auc")),
            "subject_pr_auc": _finite(subject_metrics.get("average_precision")),
            "subject_accuracy": _finite(subject_metrics.get("accuracy")),
            "subject_bce": _finite(subject_metrics.get("bce")),
            "subject_balanced_accuracy": _finite(subject_metrics.get("balanced_accuracy")),
            "subject_sensitivity": _finite(subject_metrics.get("sensitivity")),
            "subject_specificity": _finite(subject_metrics.get("specificity")),
            "tp": subject_metrics.get("tp"),
            "tn": subject_metrics.get("tn"),
            "fp": subject_metrics.get("fp"),
            "fn": subject_metrics.get("fn"),
        }
        result["status"] = "PASS"
        result["formal_eligible"] = True
    except (TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
        result.pop("subjects", None)
        result.pop("metrics", None)
    return result


def compute_secondary_ensemble_statistics(
    ensemble: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 2000,
    swap_replicates: int = 9999,
    seed: int = 73,
) -> dict[str, Any]:
    """Compute the preregistered statistics for one fixed seed ensemble.

    ``ensemble`` must already be the successful output of
    :func:`aggregate_fixed_seed_subject_ensemble`.  Scores are therefore
    fixed subject probabilities (the mean across the declared seeds); the
    pair-swap test is conditional on those scores and on the frozen matched
    pairs.  This helper deliberately refuses a single-seed payload or a
    subject-only resampling fallback, because the secondary family requires
    whole-pair bootstrap/swap units.
    """

    result: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "inference_family": SECONDARY_48_FAMILY,
        "inference_unit": "fixed_classifier_subject_probability_ensemble",
        "conditional_on": "frozen matched pairs and fixed classifier scores",
        "bootstrap_replicates": int(bootstrap_replicates),
        "swap_replicates": int(swap_replicates),
        "seed": int(seed),
        "errors": [],
    }
    if str(ensemble.get("status", "")).upper() != "PASS" or not ensemble.get("formal_eligible"):
        result["errors"] = ["fixed-seed subject ensemble is not complete"]
        return result
    subjects = ensemble.get("subjects")
    if not isinstance(subjects, Sequence) or isinstance(subjects, (str, bytes, bytearray)) or not subjects:
        result["errors"] = ["ensemble has no subject rows"]
        return result
    try:
        labels = np.asarray([int(row["label"]) for row in subjects], dtype=np.int64)
        scores = np.asarray([float(row["mean_probability"]) for row in subjects], dtype=np.float64)
        pair_ids = np.asarray([row.get("pair_id") for row in subjects], dtype=object)
    except (KeyError, TypeError, ValueError) as exc:
        result["errors"] = [f"invalid ensemble subject rows: {exc}"]
        return result
    if labels.size == 0 or labels.size != scores.size or labels.size != pair_ids.size:
        result["errors"] = ["ensemble labels/scores/pairs have incompatible lengths"]
        return result
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        result["errors"] = ["ensemble probabilities must be finite in [0,1]"]
        return result
    if any(value is None or str(value).strip() == "" for value in pair_ids.tolist()):
        result["errors"] = ["secondary ensemble requires non-empty pair IDs"]
        return result
    if int(bootstrap_replicates) <= 0 or int(swap_replicates) <= 0:
        result["errors"] = ["secondary bootstrap and pair-swap counts must be positive"]
        return result
    try:
        bootstrap = bootstrap_subject_auc(
            labels,
            scores,
            pair_ids=pair_ids,
            n_bootstrap=int(bootstrap_replicates),
            seed=int(seed),
        )
        pair_swap = paired_heldout_swap_test(
            labels,
            scores,
            pair_ids,
            n_swaps=int(swap_replicates),
            seed=int(seed),
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        result["errors"] = [f"secondary statistic computation failed: {exc}"]
        return result
    result.update(
        {
            "status": "PASS"
            if int(bootstrap.get("n_valid", 0)) == int(bootstrap_replicates)
            and str(pair_swap.get("status", "")).lower() == "complete"
            and int(pair_swap.get("n_valid", 0)) == int(swap_replicates)
            and _finite(pair_swap.get("p_value")) is not None
            else "INCONCLUSIVE",
            "subject_count": int(labels.size),
            "pair_count": int(pair_swap.get("n_pairs", 0)),
            "subject_roc_auc": _finite(_mapping(ensemble.get("metrics")).get("subject_roc_auc")),
            "subject_pr_auc": _finite(_mapping(ensemble.get("metrics")).get("subject_pr_auc")),
            "bootstrap": bootstrap,
            "pair_swap": pair_swap,
            "secondary_p_value": _finite(pair_swap.get("p_value")),
            "secondary_p_value_source": "secondary_48_conditional_heldout_pair_swap",
        }
    )
    if result["status"] != "PASS":
        result["errors"] = ["secondary bootstrap or pair-swap replicate set is incomplete"]
    return result


def _secondary_manifest_payload(source: Any) -> tuple[Mapping[str, Any], Path, list[str]]:
    """Load an explicit secondary manifest without directory discovery."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            for name in ("secondary_48_manifest.json", "secondary_manifest.json"):
                candidate = path / name
                if candidate.is_file():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "secondary manifest directory must contain secondary_48_manifest.json or secondary_manifest.json"
                )
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError("secondary manifest JSON must be an object")
        return payload, path.resolve().parent, [str(path.resolve())]
    if isinstance(source, Mapping):
        return source, Path.cwd(), []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return {"schema_version": 1, "family": SECONDARY_48_FAMILY, "cells": list(source)}, Path.cwd(), []
    raise TypeError("secondary source must be a manifest path, mapping, or sequence of cells")


def _secondary_cell_list(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("cells")
    if value is None:
        nested = _mapping(payload.get(SECONDARY_48_FAMILY))
        value = nested.get("cells")
    if isinstance(value, Mapping):
        output: list[Mapping[str, Any]] = []
        for key, item in value.items():
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("cell_id", str(key))
            output.append(row)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _holm_family_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    expected_count: int,
    p_reader: Any,
    alpha: float,
) -> dict[str, Any]:
    """Apply Holm only to a complete, explicitly identified p-value family."""

    p_values: dict[str, float] = {}
    missing: list[str] = []
    invalid: list[str] = []
    sources: dict[str, str] = {}
    for row in rows:
        cell_id = str(row.get("cell_id", "unknown"))
        value, source, error = p_reader(row)
        if value is None:
            if error and "not finite" in error:
                invalid.append(cell_id)
            else:
                missing.append(cell_id)
            continue
        p_values[cell_id] = float(value)
        if source:
            sources[cell_id] = source
    complete = len(rows) == int(expected_count) and not missing and not invalid and len(p_values) == int(expected_count)
    summary: dict[str, Any] = {
        "family": family,
        "expected_tests": int(expected_count),
        "observed_tests": len(rows),
        "status": "PASS" if complete else "INCONCLUSIVE",
        "alpha": float(alpha),
        "p_value_source": sources,
        "missing_p_value_cells": sorted(missing),
        "invalid_p_value_cells": sorted(invalid),
        "adjusted_p_values": {},
        "dependence_note": "Holm step-down is valid under arbitrary dependence; family membership is fixed before adjustment.",
    }
    if complete:
        adjusted = holm_adjust(p_values)
        summary["raw_p_values"] = p_values
        summary["adjusted_p_values"] = {
            key: (None if not np.isfinite(value) else float(value))
            for key, value in adjusted.items()
        }
    else:
        summary["raw_p_values"] = p_values
        summary["reason"] = "No adjusted family-level inference is emitted until every expected cell has a valid family-specific p-value."
    return summary


def summarise_secondary_48(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected_cells: int = 48,
    expected_seeds: Sequence[int] = DEFAULT_SECONDARY_SEEDS,
    evaluation_split: str = "test",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarise explicit secondary fixed-seed subject ensembles.

    The expected manifest has ``family='secondary_48'`` and ``cells``.  Each
    cell carries metadata plus ``seed_results`` entries with ``seed`` and an
    inline runner result or ``result_path``.  The optional
    ``primary_full_retrained`` section is a separate four-test family and
    accepts only explicit full-retrained p-values.  No control, calibration,
    or held-out pair-swap p-value is inferred or mixed into either family.
    """

    payload, base_dir, manifest_paths = _secondary_manifest_payload(source)
    cells = _secondary_cell_list(payload)
    summaries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for ordinal, cell in enumerate(cells):
        cell_id = _secondary_cell_id(cell, ordinal=ordinal if not _first(cell, "cell_id", "analysis_id", "comparison_id", default=None) else None)
        if cell_id in seen_ids:
            duplicate_ids.append(cell_id)
        seen_ids.add(cell_id)
        item = aggregate_fixed_seed_subject_ensemble(
            dict(cell, cell_id=cell_id),
            base_dir=base_dir,
            expected_seeds=_secondary_expected_seeds(cell, expected_seeds),
            evaluation_split=evaluation_split,
        )
        item["secondary_p_value"] = _secondary_p_value(cell)[0]
        item["secondary_p_value_source"] = _secondary_p_value(cell)[1]
        item["secondary_p_value_error"] = _secondary_p_value(cell)[2]
        summaries.append(item)
    family = _holm_family_summary(
        summaries,
        family=SECONDARY_48_FAMILY,
        expected_count=int(expected_cells),
        p_reader=lambda row: (
            _finite(row.get("secondary_p_value")),
            row.get("secondary_p_value_source"),
            row.get("secondary_p_value_error") or ("secondary p-value missing" if row.get("secondary_p_value") is None else None),
        ),
        alpha=alpha,
    )
    family_errors: list[str] = []
    if duplicate_ids:
        family_errors.append(f"duplicate cell IDs: {sorted(set(duplicate_ids))}")
    if len(cells) != int(expected_cells):
        family_errors.append(f"expected {int(expected_cells)} cells, observed {len(cells)}")
    if any(item.get("status") != "PASS" for item in summaries):
        family_errors.append("one or more cells failed closed; no formal ensemble result is claimed")
    if family["status"] != "PASS":
        family_errors.append(str(family.get("reason", "secondary Holm family is incomplete")))

    primary_payload = payload.get("primary_full_retrained")
    if primary_payload is None:
        primary_payload = payload.get(PRIMARY_FULL_RETRAINED_HOLM4_FAMILY)
    primary_cells = _secondary_cell_list(_mapping(primary_payload)) if isinstance(primary_payload, Mapping) else []
    primary_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(primary_cells):
        cell_id = _secondary_cell_id(row, ordinal=ordinal)
        primary_rows.append({
            "cell_id": cell_id,
            "full_retrained_p_value": _primary_full_retrained_p_value(row)[0],
            "full_retrained_p_value_source": _primary_full_retrained_p_value(row)[1],
            "full_retrained_p_value_error": _primary_full_retrained_p_value(row)[2],
        })
    primary_family = _holm_family_summary(
        primary_rows,
        family=PRIMARY_FULL_RETRAINED_HOLM4_FAMILY,
        expected_count=4,
        p_reader=lambda row: (
            _finite(row.get("full_retrained_p_value")),
            row.get("full_retrained_p_value_source"),
            row.get("full_retrained_p_value_error") or "full-retrained p-value missing",
        ),
        alpha=alpha,
    )
    primary_family["conditional_heldout_p_values_excluded"] = True
    return {
        "schema_version": 1,
        "family": SECONDARY_48_FAMILY,
        "status": "PASS" if not family_errors else "INCONCLUSIVE",
        "formal_eligible": not bool(family_errors),
        "expected_cells": int(expected_cells),
        "observed_cells": len(cells),
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "expected_seeds_by_cell": {
            str(item.get("cell_id")): list(item.get("expected_seeds", []))
            for item in summaries
        },
        "evaluation_split": evaluation_split,
        "cells": summaries,
        "holm": family,
        "primary_full_retrained": primary_family,
        "manifest_paths": manifest_paths,
        "errors": family_errors,
        "notes": [
            "Subject probabilities are averaged across exactly the declared fixed seeds after exact subject/label alignment.",
            "Missing seed or subject/label mismatch fails closed and removes that cell's metrics.",
            "Secondary Holm and primary full-retrained Holm4 are disjoint families; held-out conditional pair-swap p-values are excluded.",
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    if number is not None:
        return f"{number:.{digits}f}"
    if value in (None, ""):
        return "未提供"
    return str(value)


def compute_andi_ap_correlation(
    rows: Sequence[Mapping[str, Any]],
    andi_ap_points: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    auc_field: str = "subject_roc_auc",
    ap_field: str = "mf_ap",
) -> dict[str, Any]:
    """Compute the fixed four-point descriptive ANDi/AP association.

    The selector is deliberately exact: each point must come from the same
    v3 primary contract (``stage=final``, all three modalities, SmallCNN,
    seed/init seed 73).  A different modality, model, seed, or an incomplete
    selector is never used as a fallback.  This prevents input ordering from
    choosing an arbitrary formal row when a 112-cell matrix is present.
    """

    expected_domains = ("FOMO", "MPI", "OASIS3", "Mixed")
    selector = {
        "stage": "final",
        "input_modalities": "FLAIR+T1+T2",
        "classifier": "small_cnn",
        "seed": 73,
    }

    # Preserve duplicate AP points as an audit failure rather than allowing a
    # dict conversion to silently keep whichever duplicate happened to occur
    # last.  A Mapping cannot contain an identical key twice, but normalized
    # aliases such as ``oasis``/``OASIS3`` can still collide.
    point_candidates: dict[str, list[Any]] = defaultdict(list)
    if isinstance(andi_ap_points, Mapping):
        iterable = andi_ap_points.items()
    else:
        iterable = (
            (_first(point, "healthy_domain", "cohort", "domain"), point)
            for point in andi_ap_points
            if isinstance(point, Mapping)
        )
    for raw_name, point in iterable:
        if raw_name is None:
            continue
        normalized = _normalise_domain(raw_name).lower()
        point_candidates[normalized].append(point)

    duplicate_ap_domains = sorted(key for key, values in point_candidates.items() if len(values) > 1)
    point_map: dict[str, Any] = {
        key: values[0]
        for key, values in point_candidates.items()
        if values
    }
    ap_conflict_domains: list[str] = []
    for key, values in point_candidates.items():
        finite_values: list[float] = []
        for value in values:
            mapping = _mapping(value)
            ap = _finite(mapping.get(ap_field) if mapping else value)
            if ap is not None:
                finite_values.append(float(ap))
        if len({round(value, 15) for value in finite_values}) > 1:
            ap_conflict_domains.append(key)

    def is_exact_primary(row: Mapping[str, Any]) -> bool:
        stage = _normalise_text(_first(row, "stage", "preprocessing_stage", default="")).lower()
        modalities = _normalise_modalities(
            _first(row, "input_modalities", "modalities", "modality", "channels", default="")
        )
        classifier = _normalise_classifier(
            _first(row, "classifier", "model", "architecture", default="")
        )
        seed = _normalise_seed(_first(row, "seed", "training_seed", "init_seed", default=None))
        return (
            stage == selector["stage"]
            and modalities == selector["input_modalities"]
            and classifier == selector["classifier"]
            and seed == selector["seed"]
        )

    selected_by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _is_formal_final_row(row) or not is_exact_primary(row):
            continue
        domain = _normalise_domain(_first(row, "healthy_domain", "healthy_cohort", "cohort", default="unknown"))
        if domain in expected_domains:
            selected_by_domain[domain.lower()].append(row)

    duplicate_auc_domains = sorted(key for key, values in selected_by_domain.items() if len(values) > 1)
    missing_domains = sorted(
        domain for domain in expected_domains
        if len(selected_by_domain.get(domain.lower(), [])) != 1
    )
    missing_ap_domains = sorted(
        domain
        for domain in expected_domains
        if domain.lower() not in point_map
        or _finite(
            _mapping(point_map[domain.lower()]).get(ap_field)
            if isinstance(point_map.get(domain.lower()), Mapping)
            else point_map.get(domain.lower())
        ) is None
    )
    paired: list[dict[str, Any]] = []
    if not duplicate_auc_domains and not duplicate_ap_domains and not ap_conflict_domains:
        for domain in expected_domains:
            key = domain.lower()
            row_values = selected_by_domain.get(key, [])
            if len(row_values) != 1 or key not in point_map:
                continue
            row = row_values[0]
            point = point_map[key]
            point_mapping = _mapping(point)
            auc = _finite(row.get(auc_field))
            ap = _finite(point_mapping.get(ap_field) if point_mapping else point)
            if auc is None or ap is None:
                continue
            paired.append(
                {
                    "healthy_domain": domain,
                    "domain_auc": auc,
                    "andi_ap": ap,
                    "andi_ap_path": point_mapping.get("path") if point_mapping else None,
                    "andi_ap_sha256": point_mapping.get("sha256") if point_mapping else None,
                    "run_id": row.get("run_id"),
                    "source_path": row.get("source_path"),
                    "selector": dict(selector),
                }
            )
    x = np.asarray([item["domain_auc"] for item in paired], dtype=np.float64)
    y = np.asarray([item["andi_ap"] for item in paired], dtype=np.float64)
    if duplicate_auc_domains or duplicate_ap_domains or ap_conflict_domains:
        status = "FAIL_CLOSED_DUPLICATE_OR_CONFLICT"
    elif missing_domains or missing_ap_domains or len(paired) != len(expected_domains):
        status = "INCONCLUSIVE_SELECTOR_PENDING"
    elif len(paired) < 2:
        status = "INCONCLUSIVE"
    elif np.std(x) == 0 or np.std(y) == 0:
        status = "INCONCLUSIVE_CONSTANT_INPUT"
    else:
        status = "COMPLETE"
    result: dict[str, Any] = {
        "n_points": int(len(paired)),
        "points": paired,
        "auc_field": auc_field,
        "ap_field": ap_field,
        "selector": selector,
        "expected_domains": list(expected_domains),
        "selected_domains": [item["healthy_domain"] for item in paired],
        "missing_domains": missing_domains,
        "missing_ap_domains": missing_ap_domains,
        "duplicate_auc_domains": duplicate_auc_domains,
        "duplicate_ap_domains": duplicate_ap_domains,
        "ap_conflict_domains": ap_conflict_domains,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "caveat": "僅作描述性診斷；約 4 個 cohort points 時不能視為強統計證據。",
        "status": status,
    }
    if status == "INCONCLUSIVE_CONSTANT_INPUT":
        result["caveat"] = "至少一個 correlation 軸為 constant；Pearson/Spearman undefined，保留 scatter points 但不能計算相關係數。"
    elif status == "INCONCLUSIVE_SELECTOR_PENDING":
        result["caveat"] = "固定 primary selector 缺少至少一個 cohort 或 finite AP/AUC；不使用其他 model/modality/seed fallback。"
    elif status == "FAIL_CLOSED_DUPLICATE_OR_CONFLICT":
        result["caveat"] = "固定 primary selector 或 AP endpoint 有 duplicate/conflict；association fail-closed，不能選任意 row。"
    if status == "COMPLETE":
        try:
            from scipy.stats import pearsonr, spearmanr

            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            result["pearson_r"] = float(pearson.statistic)
            result["pearson_p"] = float(pearson.pvalue)
            result["spearman_rho"] = float(spearman.statistic)
            result["spearman_p"] = float(spearman.pvalue)
        except ImportError:  # pragma: no cover - SciPy is available in ANDi env
            result["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
            rank_x = np.argsort(np.argsort(x))
            rank_y = np.argsort(np.argsort(y))
            result["spearman_rho"] = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return result


def load_andi_ap_points(source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None, *, root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load MF AP points from an explicit design audit or known run paths.

    A missing path is retained with ``status='MISSING'`` and no AP value.  No
    number is copied from the request text or from a similarly named run.
    """

    base = Path(root) if root is not None else Path.cwd()
    if source is not None and not isinstance(source, (str, Path)):
        if isinstance(source, Mapping):
            return {str(key): dict(_mapping(value)) if isinstance(value, Mapping) else {"mf_ap": _finite(value)} for key, value in source.items()}
        return {str(_first(item, "healthy_domain", "cohort", "domain")): dict(item) for item in source if isinstance(item, Mapping)}
    paths: dict[str, Path] = {}
    if source is not None:
        paths["audit"] = Path(source)
    else:
        for domain, relative in KNOWN_ANDI_AP_PATHS.items():
            paths[domain] = base / relative
    result: dict[str, dict[str, Any]] = {}
    for domain, path in paths.items():
        if not path.is_absolute():
            path = base / path
        entry: dict[str, Any] = {"path": str(path.resolve()), "status": "MISSING"}
        if not path.is_file():
            result[domain] = entry
            continue
        try:
            payload = _read_data_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            entry["error"] = str(exc)
            result[domain] = entry
            continue
        candidate_rows = _rows_from_payload(payload)
        selected: Mapping[str, Any] | None = None
        for item in candidate_rows:
            version = _normalise_text(item.get("version", "")).lower()
            if version in {"median_filter", "mf", "medianfilter"} or selected is None:
                selected = item
        if selected is not None:
            ap = _finite(_first(selected, "mf_ap", "MF_AP", "AUPRC_mf", "AUPRC"))
            if ap is not None:
                entry["mf_ap"] = ap
                entry["status"] = "COMPLETE"
        entry.update(_fingerprint(path))
        result[domain] = entry
    return result


def _plot_context(rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    return {"rows": rows, "prediction_context": context.get("prediction_context", [])}


def generate_plots(
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    output_dir: str | Path,
    *,
    andi_correlation: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate available diagnostic figures and record skipped figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, dict[str, Any]] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {name: {"status": "MISSING", "reason": str(exc)} for name in (
            "roc_curves", "pr_curves", "confusion_matrices", "training_curves", "permutation_nulls",
            "bootstrap_auc", "modality_comparison", "cohort_comparison", "seed_summary", "andi_ap_scatter",
        )}

    prediction_context = context.get("prediction_context", [])
    formal_indices = {
        index for index, row in enumerate(rows) if _is_formal_final_row(row)
    }
    plotted_predictions = [
        item for item in prediction_context
        if item.get("subject_predictions") is not None
        and int(item.get("row_index", -1)) in formal_indices
    ]
    def finish(name: str, figure: Any, filename: str) -> None:
        figure.tight_layout()
        figure.savefig(destination / filename, dpi=160)
        plt.close(figure)
        statuses[name] = {"status": "COMPLETE", "path": str((destination / filename).resolve())}

    def skip(name: str, reason: str) -> None:
        statuses[name] = {"status": "MISSING", "reason": reason}

    # ROC and PR curves use the same subject-level held-out arrays that drive
    # the summary AUC, preventing a slice/subject denominator mix-up.
    if plotted_predictions:
        from sklearn.metrics import precision_recall_curve, roc_curve

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for item in plotted_predictions:
            subject = item["subject_predictions"]
            labels = subject["labels"]
            scores = subject["scores"]
            if np.unique(labels).size < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            index = int(item["row_index"])
            row = rows[index]
            ax.plot(fpr, tpr, label=f"{row.get('healthy_domain')} {row.get('input_modalities')} {row.get('classifier')}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Subject-level ROC")
        ax.legend(fontsize=7, loc="lower right")
        finish("roc_curves", fig, "roc_curves.png")

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for item in plotted_predictions:
            subject = item["subject_predictions"]
            labels = subject["labels"]
            scores = subject["scores"]
            if np.sum(labels) == 0:
                continue
            precision, recall, _ = precision_recall_curve(labels, scores)
            index = int(item["row_index"])
            row = rows[index]
            ax.plot(recall, precision, label=f"{row.get('healthy_domain')} {row.get('input_modalities')} {row.get('classifier')}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Subject-level precision-recall")
        ax.legend(fontsize=7, loc="lower left")
        finish("pr_curves", fig, "pr_curves.png")
    else:
        skip("roc_curves", "沒有可用的 subject-level predictions")
        skip("pr_curves", "沒有可用的 subject-level predictions")

    metric_rows = [
        row
        for index, row in enumerate(rows)
        if index in formal_indices and _finite(row.get("subject_roc_auc")) is not None
    ]
    if metric_rows:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        labels = [f"{row.get('healthy_domain')}\n{row.get('classifier')}" for row in metric_rows]
        values = [float(row["subject_direction_invariant_auc"] or row["subject_roc_auc"]) for row in metric_rows]
        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Direction-invariant subject AUC")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Classifier separability")
        finish("cohort_comparison", fig, "cohort_comparison.png")

        groups = sorted({str(row.get("input_modalities")) for row in metric_rows})
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for modality in groups:
            selected = [row for row in metric_rows if str(row.get("input_modalities")) == modality]
            ax.scatter([modality] * len(selected), [row["subject_direction_invariant_auc"] for row in selected], label=modality)
        ax.set_ylabel("Direction-invariant subject AUC")
        ax.set_title("Modality comparison")
        finish("modality_comparison", fig, "modality_comparison.png")
    else:
        skip("cohort_comparison", "沒有 finite subject-level AUC")
        skip("modality_comparison", "沒有 finite subject-level AUC")

    seed_values: dict[str, list[float]] = defaultdict(list)
    for row in metric_rows:
        key = f"{row.get('healthy_domain')} / {row.get('input_modalities')} / {row.get('classifier')}"
        value = _finite(row.get("subject_direction_invariant_auc"))
        if value is not None:
            seed_values[key].append(value)
    if seed_values:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        keys = list(seed_values)
        ax.boxplot([seed_values[key] for key in keys], labels=keys, vert=True)
        ax.set_ylabel("Direction-invariant subject AUC")
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)
        ax.set_title("Repeated-seed summary")
        finish("seed_summary", fig, "seed_summary.png")
    else:
        skip("seed_summary", "沒有可用 seed-level AUC")

    if andi_correlation and andi_correlation.get("points"):
        points = andi_correlation["points"]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        x = [point["domain_auc"] for point in points]
        y = [point["andi_ap"] for point in points]
        ax.scatter(x, y)
        for point in points:
            ax.annotate(str(point["healthy_domain"]), (point["domain_auc"], point["andi_ap"]), fontsize=8)
        ax.set(xlabel="Domain classifier subject ROC-AUC", ylabel="ANDi MF AP", title="Descriptive domain/AP association")
        finish("andi_ap_scatter", fig, "andi_ap_scatter.png")
    else:
        skip("andi_ap_scatter", "沒有兩者皆 finite 的 paired cohort points")

    # Confusion matrices, bootstrap and permutation figures need explicit
    # arrays in the run result.  A summary-only run is reported as missing.
    if metric_rows and all(all(row.get(key) is not None for key in ("tp", "tn", "fp", "fn")) for row in metric_rows):
        fig, axes = plt.subplots(1, len(metric_rows), figsize=(4 * len(metric_rows), 3.5), squeeze=False)
        for axis, row in zip(axes[0], metric_rows):
            matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
            axis.imshow(matrix, cmap="Blues")
            axis.set_xticks([0, 1], ["0", "1"])
            axis.set_yticks([0, 1], ["0", "1"])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(str(row.get("healthy_domain")))
            for (i, j), value in np.ndenumerate(matrix):
                axis.text(j, i, str(int(value)), ha="center", va="center")
        finish("confusion_matrices", fig, "confusion_matrices.png")
    else:
        skip("confusion_matrices", "沒有完整的 subject-level confusion counts")

    training_files = [Path(path) for path in context.get("training_files", [])]
    if training_files:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        plotted = False
        for path in training_files:
            try:
                curve_rows = _read_csv(path)
            except (OSError, ValueError):
                continue
            if not curve_rows:
                continue
            x_name = next((name for name in ("epoch", "step", "iteration") if name in curve_rows[0]), None)
            y_name = next((name for name in ("val_loss", "validation_loss", "train_loss", "loss") if name in curve_rows[0]), None)
            if not x_name or not y_name:
                continue
            x = [_finite(item.get(x_name)) for item in curve_rows]
            y = [_finite(item.get(y_name)) for item in curve_rows]
            if any(value is None for value in x + y):
                continue
            ax.plot(x, y, label=path.stem)
            plotted = True
        if plotted:
            ax.set(xlabel="Epoch/step", ylabel="Loss", title="Training curves")
            ax.legend(fontsize=7)
            finish("training_curves", fig, "training_curves.png")
        else:
            plt.close(fig)
            skip("training_curves", "training curve files lack recognised numeric columns")
    else:
        skip("training_curves", "沒有 training curve artifact")

    # A run may expose the full bootstrap/permutation vectors in JSON.  The
    # current runner's compact summary need not; in that case avoid drawing a
    # fake null/CI distribution.
    bootstrap_values: list[float] = []
    permutation_values: list[float] = []
    # Bootstrap samples are part of the formal final result.  Permutation
    # samples belong to control artifacts and are intentionally collected from
    # all rows for the separate null-control figure only.
    for index, row in enumerate(rows):
        if index not in formal_indices:
            continue
        for key, target in (("bootstrap_samples", bootstrap_values), ("permutation_null", permutation_values), ("permutation_samples", permutation_values)):
            values = row.get(key)
            if isinstance(values, Sequence) and not isinstance(values, str):
                target.extend(value for value in (_finite(item) for item in values) if value is not None)
    for index, row in enumerate(rows):
        if index in formal_indices:
            continue
        for key in ("permutation_null", "permutation_samples", "bootstrap_samples"):
            values = row.get(key)
            if isinstance(values, Sequence) and not isinstance(values, str):
                if key != "bootstrap_samples":
                    permutation_values.extend(value for value in (_finite(item) for item in values) if value is not None)
    if bootstrap_values:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.hist(bootstrap_values, bins=30)
        ax.set(xlabel="Subject AUC", ylabel="Replicates", title="Subject bootstrap AUC")
        finish("bootstrap_auc", fig, "bootstrap_auc.png")
    else:
        skip("bootstrap_auc", "沒有保存 bootstrap replicate vector")
    if permutation_values:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.hist(permutation_values, bins=30)
        ax.axvline(0.5, color="k", linestyle="--")
        ax.set(xlabel="Null subject AUC", ylabel="Replicates", title="Permutation null")
        finish("permutation_nulls", fig, "permutation_nulls.png")
    else:
        skip("permutation_nulls", "沒有保存 permutation null vector")
    (destination / "plot_manifest.json").write_text(json.dumps(_json_safe(statuses), indent=2, ensure_ascii=False), encoding="utf-8")
    return statuses


def generate_calibration_diagnostic_plots(
    result_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Plot one observed calibration fit without entering the formal matrix.

    This is deliberately separate from :func:`generate_plots`: an observed
    calibration fit is useful for checking capacity and recorded uncertainty,
    but it is excluded from formal rows while the preregistered full-retrain
    null is incomplete.  The null histogram is emitted only after the status
    artifact reports all requested replicates complete.
    """

    result_path = Path(result_path)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_root = result_path.parent.parent if result_path.parent.name.lower() == "observed" else result_path.parent
    null_root = run_root / "retrained_null"
    source_candidates = [
        result_path,
        result_path.parent / "test_predictions.jsonl",
        run_root / "protocol.json",
        run_root / "source_cache_digest.json",
        null_root / "status.json",
        null_root / "statistic.json",
        null_root / "summary.json",
    ]
    source_files: dict[str, dict[str, Any]] = {}
    for path in source_candidates:
        if not path.is_file():
            continue
        try:
            key = path.relative_to(run_root).as_posix()
        except ValueError:
            key = path.name
        source_files[key] = _fingerprint(path)

    payload = _mapping(_read_json(result_path))
    test = _mapping(payload.get("test"))
    test_subject = _mapping(test.get("subject"))
    statistics = _mapping(payload.get("test_statistics"))
    bootstrap = _mapping(statistics.get("subject_bootstrap"))
    statuses: dict[str, dict[str, Any]] = {}
    plot_source_keys = [key for key in source_files if key in {"observed/result.json", "observed/test_predictions.jsonl", "protocol.json"}]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        for name in ("observed_roc", "observed_pr", "observed_confusion", "observed_training_curves", "observed_subject_bootstrap", "full_retrained_null"):
            statuses[name] = {"status": "MISSING", "reason": str(exc)}
        manifest = {
            "schema_version": 1,
            "kind": "domain_classifier_calibration_diagnostic_plots",
            "status": "INCOMPLETE",
            "classification": "DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY",
            "formal_matrix_rows": 0,
            "source_files": source_files,
            "plots": statuses,
        }
        manifest_path = destination / "plot_manifest.json"
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
        return {"manifest_path": str(manifest_path.resolve()), "plots": statuses, "source_files": source_files}

    def finish(name: str, figure: Any, filename: str, description: str) -> None:
        figure.tight_layout()
        figure.savefig(destination / filename, dpi=180)
        plt.close(figure)
        statuses[name] = {
            "status": "COMPLETE",
            "path": str((destination / filename).resolve()),
            "description": description,
            "source_files": plot_source_keys,
        }

    def skip(name: str, reason: str) -> None:
        statuses[name] = {"status": "MISSING", "reason": reason, "source_files": plot_source_keys}

    subject_records = payload.get("test_subject_predictions")
    if not isinstance(subject_records, list):
        subject_records = test.get("subject_predictions")
    records = subject_records if isinstance(subject_records, list) else []
    labels: list[int] = []
    scores: list[float] = []
    for item in records:
        if not isinstance(item, Mapping):
            continue
        label = _finite(_first(item, "label", "y_true", "target"))
        score = _finite(_first(item, "probability", "score", "prediction"))
        if label is None or score is None:
            continue
        labels.append(int(label))
        scores.append(float(score))

    observed_auc = _finite(test_subject.get("roc_auc"))
    if len(set(labels)) >= 2 and len(labels) == len(scores):
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.plot(fpr, tpr, label=f"Observed subject ROC-AUC={auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Observed calibration diagnostic ROC (excluded from formal matrix)")
        ax.legend(loc="lower right", fontsize=8)
        finish("observed_roc", fig, "observed_roc.png", "Subject-level held-out ROC from observed calibration result")

        precision, recall, _ = precision_recall_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.plot(recall, precision, label=f"Average precision={ap:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Observed calibration diagnostic PR (excluded from formal matrix)")
        ax.legend(loc="lower left", fontsize=8)
        finish("observed_pr", fig, "observed_pr.png", "Subject-level held-out precision-recall from observed calibration result")
        observed_auc = auc
    else:
        skip("observed_roc", "observed test_subject_predictions lack both labels")
        skip("observed_pr", "observed test_subject_predictions lack both labels")

    confusion = {
        "tn": _finite(test_subject.get("tn")),
        "fp": _finite(test_subject.get("fp")),
        "fn": _finite(test_subject.get("fn")),
        "tp": _finite(test_subject.get("tp")),
    }
    if all(value is not None for value in confusion.values()):
        matrix = np.asarray([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]], dtype=float)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Observed calibration subject confusion (diagnostic)")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(int(value)), ha="center", va="center")
        finish("observed_confusion", fig, "observed_confusion.png", "Observed subject-level confusion counts")
    else:
        skip("observed_confusion", "observed test.subject lacks complete confusion counts")

    history = payload.get("history")
    history_rows = history if isinstance(history, list) else []
    epochs = [_finite(_first(item, "epoch")) or float(index + 1) for index, item in enumerate(history_rows) if isinstance(item, Mapping)]
    train_loss = [_finite(item.get("train_loss")) for item in history_rows if isinstance(item, Mapping)]
    val_loss = [_finite(item.get("val_loss")) for item in history_rows if isinstance(item, Mapping)]
    val_auc: list[float | None] = []
    for item in history_rows:
        subject = _mapping(_mapping(item).get("val_subject")) if isinstance(item, Mapping) else {}
        val_auc.append(_finite(subject.get("roc_auc")))
    if epochs and (any(value is not None for value in train_loss + val_loss) or any(value is not None for value in val_auc)):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        if any(value is not None for value in train_loss):
            axes[0].plot(epochs, [value if value is not None else np.nan for value in train_loss], label="train loss")
        if any(value is not None for value in val_loss):
            axes[0].plot(epochs, [value if value is not None else np.nan for value in val_loss], label="validation loss")
        axes[0].set(xlabel="Epoch", ylabel="BCE/loss", title="Observed calibration training curve")
        axes[0].legend(fontsize=8)
        if any(value is not None for value in val_auc):
            axes[1].plot(epochs, [value if value is not None else np.nan for value in val_auc], label="validation subject AUC")
        axes[1].set(xlabel="Epoch", ylabel="Subject ROC-AUC", title="Validation capacity diagnostic")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend(fontsize=8)
        finish("observed_training_curves", fig, "observed_training_curves.png", "Observed calibration history: loss and validation subject AUC")
    else:
        skip("observed_training_curves", "result history lacks numeric loss/AUC fields")

    bootstrap_samples = bootstrap.get("samples")
    if isinstance(bootstrap_samples, Sequence) and not isinstance(bootstrap_samples, (str, bytes, bytearray)):
        values = [value for value in (_finite(item) for item in bootstrap_samples) if value is not None]
    else:
        values = []
    if values:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.hist(values, bins=30, color="#4C78A8", alpha=0.85)
        if observed_auc is not None:
            ax.axvline(observed_auc, color="#D62728", linewidth=1.5, label=f"Observed AUC={observed_auc:.4f}")
        ci_low = _finite(bootstrap.get("ci_low"))
        ci_high = _finite(bootstrap.get("ci_high"))
        if ci_low is not None:
            ax.axvline(ci_low, color="k", linestyle=":", linewidth=1.0)
        if ci_high is not None:
            ax.axvline(ci_high, color="k", linestyle=":", linewidth=1.0, label=f"Recorded CI [{ci_low:.4f}, {ci_high:.4f}]")
        ax.set(xlabel="Subject ROC-AUC", ylabel="Matched-pair bootstrap count", title="Observed subject bootstrap (diagnostic)")
        ax.legend(fontsize=8)
        finish("observed_subject_bootstrap", fig, "observed_subject_bootstrap.png", "Recorded subject bootstrap samples; resampling unit is matched_pair")
    else:
        skip("observed_subject_bootstrap", "result lacks saved subject bootstrap samples")

    null_status = _mapping(_read_json(null_root / "status.json")) if (null_root / "status.json").is_file() else {}
    null_state = str(null_status.get("status", "not_started")).lower()
    completed = int(null_status.get("completed", 0) or 0)
    requested = int(null_status.get("requested", 0) or 0)
    null_complete = (
        null_state in {"complete", "completed", "pass", "passed"}
        and requested == 199
        and completed >= requested
    )
    if not null_complete:
        skip("full_retrained_null", f"deferred until terminal 199-run completion; status={null_state}, completed/requested={completed}/{requested}")
    else:
        statistic_path = null_root / "statistic.json"
        summary_path = null_root / "summary.json"
        statistic = _mapping(_read_json(statistic_path)) if statistic_path.is_file() else {}
        if not statistic and summary_path.is_file():
            statistic = _mapping(_mapping(_read_json(summary_path)).get("full_null_statistic"))
        null_values = [value for value in (_finite(item) for item in statistic.get("null_T_values", [])) if value is not None]
        observed_t = _finite(statistic.get("observed_T"))
        if observed_t is None and observed_auc is not None:
            observed_t = abs(float(observed_auc) - 0.5)
        if len(null_values) >= requested and observed_t is not None:
            n_null = len(null_values)
            p_plus_one = _finite(statistic.get("p_plus_one"))
            if p_plus_one is None:
                p_plus_one = (sum(value >= observed_t for value in null_values) + 1.0) / (n_null + 1.0)
            min_p = 1.0 / (n_null + 1.0)
            fig, ax = plt.subplots(figsize=(6.8, 4.5))
            ax.hist(null_values, bins=min(30, max(8, int(np.sqrt(n_null)))), color="#72B7B2", alpha=0.9)
            ax.axvline(observed_t, color="#D62728", linewidth=1.6, label=f"Observed T={observed_t:.4f}")
            ax.set(
                xlabel="T = abs(subject_mean_score_ROC-AUC − 0.5)",
                ylabel="Full-retrain null count",
                title="Full-retrained pair-swap null (conditional on matched pairs)",
            )
            ax.text(
                0.98,
                0.95,
                f"n={n_null}\nmin attainable p={min_p:.3f}\np+1={p_plus_one:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
            )
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
            finish("full_retrained_null", fig, "full_retrained_null.png", "Complete train/val/test pair-swap retrained null; conditional on matched pairs")
        else:
            skip("full_retrained_null", f"terminal null lacks {requested} null_T_values or observed_T")

    observed_plot_names = ("observed_roc", "observed_pr", "observed_confusion", "observed_training_curves", "observed_subject_bootstrap")
    observed_complete = all(statuses.get(name, {}).get("status") == "COMPLETE" for name in observed_plot_names)
    manifest = {
        "schema_version": 1,
        "kind": "domain_classifier_calibration_diagnostic_plots",
        "status": "COMPLETE" if observed_complete else "INCOMPLETE",
        "classification": "DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY",
        "formal_matrix_rows": 0,
        "observed_result": str(result_path.resolve()),
        "source_files": source_files,
        "plots": statuses,
        "null_policy": "full_retrained_null histogram is generated only after status=complete and completed=requested; partial p-values are not plotted",
        "null_status": {
            "status": null_state,
            "completed": completed,
            "requested": requested,
            "conditional_on": "matched pairs",
            "statistic": "T=abs(subject_mean_score_ROC-AUC - 0.5)",
            "minimum_attainable_p_for_199": 1.0 / 200.0,
        },
    }
    manifest_path = destination / "plot_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest_path": str(manifest_path.resolve()), "plots": statuses, "source_files": source_files, "null_status": manifest["null_status"]}


def _control_failure_present(context: Mapping[str, Any]) -> bool:
    """Return whether a recorded control gate explicitly failed."""

    controls = context.get("controls", {})
    if not isinstance(controls, Mapping):
        return False
    for item in controls.values():
        if isinstance(item, Mapping) and _status_from_mapping(item) is False:
            return True
    return False


def _v3_stage_a_gate_status(model_grid: Mapping[str, Any]) -> str:
    """Derive the v3 Stage-A status from v3-namespaced artifacts only.

    Legacy ``context["controls"]`` can contain the historical powered-shuffle
    failure.  It is intentionally not consulted here: that failure belongs to
    the old protocol and cannot change the v3 Stage-A control gate.
    """

    controls = _mapping(model_grid.get("control_summaries"))
    required = ("tiny_overfit", "registered_positive", "same_cohort_negative")
    statuses = [str(_mapping(controls.get(name)).get("status", "INCONCLUSIVE")).upper() for name in required]
    if any(status == "FAIL" for status in statuses):
        return "CONTROL_FAIL"
    if len(controls) < len(required) or any(status != "PASS" for status in statuses):
        return "INCONCLUSIVE_CONTROLS"
    observed = _mapping(model_grid.get("v3_observed"))
    if not observed:
        return "INCONCLUSIVE_OBSERVED_MISSING"
    null = _mapping(observed.get("full_retrained_null"))
    requested = int(null.get("requested", 0) or 0)
    completed = int(null.get("completed", 0) or 0)
    null_complete = str(null.get("status", "")).upper() == "COMPLETE" and requested == 199 and completed >= requested
    return "CONTROLS_PASS_OBSERVED_NULL_COMPLETE" if null_complete else "CONTROLS_PASS_OBSERVED_NULL_PENDING"


def _report_markdown(rows: Sequence[Mapping[str, Any]], correlation: Mapping[str, Any], context: Mapping[str, Any], config: ReportConfig) -> str:
    formal_rows = [row for row in rows if _is_formal_final_row(row)]
    excluded_rows = [row for row in rows if not _is_formal_final_row(row)]
    control_halt = _control_failure_present(context)
    matrix_status = "COMPLETE" if formal_rows else "INCONCLUSIVE / matrix112pending"
    matrix_note = "; rollout halted by control FAIL for the legacy v1/v2 protocol only (does not set the v3 Stage-A gate)" if control_halt else ""
    lines = [
        "# Domain Classifier / C2ST 稽核報告",
        "",
        "本文件只解讀已由 `report.py` 讀到的 run artifacts。沒有保存的 metric、control 或 predictions 會標示為 `INCONCLUSIVE`，不會補值。",
        "",
        "## 實驗狀態",
        "",
        f"- report rows: **{len(rows)}**；formal final rows used for scientific questions: **{len(formal_rows)}**；excluded control/intermediate rows: **{len(excluded_rows)}**",
        f"- formal result matrix status: **{matrix_status}{matrix_note}**",
        f"- bootstrap: subject-level、seed={config.seed}、replicates={config.bootstrap_replicates}（若 predictions 可用）",
        "- primary metric: subject-level ROC-AUC；slice-level AUC 僅為 secondary metric。",
        "- class 0 = Healthy，class 1 = BraTS21；classifier input 應只有 `[FLAIR,T1,T2]` model tensor。",
        "- 九個 scientific questions 僅使用 `result_kind=formal_final` 且 `stage=final` 的 rows；control、registered、negative、shuffle/permutation 與其他 preprocessing stages 只作 audit/control evidence。",
    ]
    stage1 = _mapping(context.get("stage1_summary"))
    if stage1:
        lines.append(
            "- Stage1 audit: "
            f"status={stage1.get('status', 'INCONCLUSIVE')}; "
            f"FOMO healthy participants={stage1.get('healthy_participants', '未提供')}, "
            f"matched pairs={stage1.get('pair_count', '未提供')}; "
            f"comparison records={stage1.get('comparison_records', '未提供')}; "
            f"native/model mask checks={stage1.get('native_tumor_free_records', '未提供')}/"
            f"{stage1.get('model_mask_false_records', '未提供')}."
        )
    parity = _mapping(context.get("parity_summary"))
    if parity:
        parity_parts: list[str] = []
        for item in parity.get("cohorts", []):
            if not isinstance(item, Mapping):
                continue
            cohort = _normalise_text(item.get("cohort"), "unknown")
            status = _normalise_text(item.get("status"), "INCONCLUSIVE").upper()
            if status == "PASS":
                slices = item.get("actual_slice_count", item.get("selected_pairs", "未提供"))
                max_error = _fmt(item.get("max_abs_error"), 0) if item.get("max_abs_error") is not None else "未提供"
                parity_parts.append(f"{cohort} PASS ({slices} slices, max_abs_error={max_error})")
            else:
                reasons = item.get("reasons")
                reason = _normalise_text(reasons[0] if isinstance(reasons, list) and reasons else "source/ACL unavailable", "source/ACL unavailable")
                reason = re.sub(r"\\s+", " ", reason).replace("|", "/")
                if "permissionerror" in reason.lower() and "winerror 5" in reason.lower():
                    reason = "ACL/PermissionError (WinError 5)"
                if len(reason) > 180:
                    reason = reason[:177] + "..."
                parity_parts.append(f"{cohort} {status} ({reason})")
        if parity_parts:
            lines.append("- Input parity: " + "; ".join(parity_parts) + ".")
    model_grid = _mapping(context.get("model_grid_v3_summary"))
    if model_grid:
        lines.extend([
            "",
            "## v3 model-grid build (build-only)",
            "",
            "這段的 model-grid build artifact 只含 candidate/manifest metadata，沒有 classifier predictions；同一 revision 的 Stage-A sidecars 另行保存 observed fit 與 controls。build metadata 本身不能產生 formal matrix rows 或 scientific question 結論。",
        ])
        lines.append(
            f"- Build artifact status=**{model_grid.get('status', 'INCONCLUSIVE')}**；scope status=**{model_grid.get('scope_status', 'DESIGN_AUDIT_PENDING')}**；"
            f"candidate coverage=**{model_grid.get('candidate_coverage_status', 'INCONCLUSIVE')}**；"
            f"elapsed={_fmt(model_grid.get('elapsed_seconds'))} s；training_started={model_grid.get('training_started', '未提供')}；"
            f"no_training={model_grid.get('no_training', '未提供')}。"
        )
        candidate_counts = _mapping(model_grid.get("candidate_inventory_counts"))
        if candidate_counts or model_grid.get("source_full_inventory_verified") is not None:
            lines.append(
                "- Full candidate inventory: "
                f"status=**{model_grid.get('candidate_coverage_status', 'INCONCLUSIVE')}**；"
                f"verified={model_grid.get('source_full_inventory_verified', '未提供')}；"
                f"CSV rows={candidate_counts.get('csv_rows', '未提供')}；"
                f"processed subjects={candidate_counts.get('processed_subjects', '未提供')}；"
                f"candidate rows={candidate_counts.get('candidate_rows', '未提供')}；"
                "old selected pool was not used as the candidate source。"
            )
        registered = _mapping(model_grid.get("registered_materialization"))
        v3_observed = _mapping(model_grid.get("v3_observed"))
        v3_null = _mapping(v3_observed.get("full_retrained_null"))
        observed_state = "COMPLETE (1/112 observed fit)" if v3_observed else "INCONCLUSIVE_OBSERVED_MISSING"
        null_state = (
            f"{str(v3_null.get('status', 'INCONCLUSIVE')).upper()} "
            f"({v3_null.get('completed', 0)}/{v3_null.get('requested', '未提供')})"
            if v3_null else "INCONCLUSIVE (not recorded)"
        )
        lines.append(
            "- v3 execution state: "
            f"Stage-A observed fit=**{observed_state}**；"
            f"primary conditional full-retrained null=**{null_state}**；"
            f"tiny gate=**{model_grid.get('tiny_gate_status', 'NOT_RECORDED_AS_OF_REPORT_GENERATION')}**；"
            f"registered materialization=**{registered.get('status', 'INCONCLUSIVE')}** "
            f"({registered.get('total_records', '未提供')} records)；"
            f"independent full Mixed review=**{model_grid.get('independent_full_mixed_review_status', 'PENDING')}**；"
            f"confirmatory fit count=**{model_grid.get('formal_fit_count', 0)}**。"
        )
        lines.append(
            "- v3 Stage-A gate（只看 v3 tiny/positive/negative 與 v3 observed/null，"
            "不受 legacy v1/v2 control FAIL 影響）："
            f"**{model_grid.get('v3_stage_a_gate_status', _v3_stage_a_gate_status(model_grid))}**；"
            f"primary observed fit count={model_grid.get('v3_primary_observed_fit_count', 0)}/112；"
            f"confirmatory formal fit count={model_grid.get('v3_confirmatory_fit_count', 0)}。"
        )
        lines.append(
            "- Legacy rollout gate（舊 v1/v2 controls，與 v3 gate 分開保存）："
            f"**{context.get('legacy_rollout_gate_status', '未提供')}**。"
        )
        control_summaries = _mapping(model_grid.get("control_summaries"))
        if control_summaries:
            control_parts: list[str] = []
            for control_name in ("tiny_overfit", "registered_positive", "same_cohort_negative"):
                control = _mapping(control_summaries.get(control_name))
                if not control:
                    continue
                row_parts: list[str] = []
                for row in control.get("rows", []):
                    if not isinstance(row, Mapping):
                        continue
                    seed = row.get("seed", "?")
                    if "accuracy" in row or "bce" in row:
                        row_parts.append(f"seed {seed}: acc={_fmt(row.get('accuracy'))}, BCE={_fmt(row.get('bce'))}")
                    elif "subject_auc" in row:
                        row_parts.append(
                            f"seed {seed}: subject AUC={_fmt(row.get('subject_auc'))}, "
                            f"CI=[{_fmt(row.get('bootstrap_ci_low'))},{_fmt(row.get('bootstrap_ci_high'))}]"
                        )
                detail = "; ".join(row_parts) if row_parts else "rows 未提供"
                control_parts.append(f"{control_name}={control.get('status', 'INCONCLUSIVE')} ({detail})")
            if control_parts:
                lines.append(
                    "- v3 controls（各自 gate，僅作 rollout evidence，不是 formal rows）："
                    + "；".join(control_parts)
                    + "。"
                )
        v3_observed = _mapping(model_grid.get("v3_observed"))
        if v3_observed:
            v3_null = _mapping(v3_observed.get("full_retrained_null"))
            v3_observed_ci = (
                f"[{_fmt(v3_observed.get('subject_bootstrap_ci_low'))},{_fmt(v3_observed.get('subject_bootstrap_ci_high'))}]"
            )
            lines.append(
                "- v3 FOMO Stage-A observed fit（獨立於舊 calibration、仍排除 confirmatory matrix）："
                f"SmallCNN seed={v3_observed.get('seed', '未提供')}；test subjects={v3_observed.get('test_subject_count', v3_observed.get('test_subjects', '未提供'))}；"
                f"subject AUC={_fmt(v3_observed.get('subject_auc'))}；recorded bootstrap CI={v3_observed_ci}；"
                f"slice AUC={_fmt(v3_observed.get('slice_auc'))}；"
                f"full-retrained conditional matched-pair null={v3_null.get('status', 'INCONCLUSIVE')} "
                f"({v3_null.get('completed', 0)}/{v3_null.get('requested', '未提供')})。"
            )
            lines.append(
                "  - 這個 AUC/CI 是目前 72 個 held-out subjects 的 empirical subject-bootstrap 結果；"
                "不能當作 population uncertainty，也不能外推到其他 seed、model 或 cohort。"
            )
            binding = _mapping(v3_observed.get("input_binding"))
            if binding:
                observed_cache = _mapping(binding.get("observed_source_cache_digest"))
                recomputed_cache = _mapping(binding.get("recomputed_source_cache_digest"))
                lines.append(
                    "  - Input binding audit: "
                    f"status=**{binding.get('status', 'INCONCLUSIVE')}**；"
                    f"rows={binding.get('observed_rows_checked', '未提供')} "
                    f"(BraTS canonical checks={binding.get('brats_rows_checked_through_canonical_reader', '未提供')})；"
                    f"shape={json.dumps(binding.get('shape_counts', {}), ensure_ascii=False, separators=(',', ':'))}；"
                    f"dtype={json.dumps(binding.get('dtype_counts', {}), ensure_ascii=False, separators=(',', ':'))}；"
                    f"digest_equal={binding.get('digest_equal', '未提供')}；"
                    f"source cache SHA={recomputed_cache.get('sha256', observed_cache.get('sha256', '未提供'))}。"
                )
            sanity = _mapping(v3_observed.get("immutable_test_label_sanity"))
            if sanity:
                sanity_cells = "; ".join(
                    f"s{row.get('seed')}: AUC={_fmt(row.get('subject_auc'))}"
                    for row in sanity.get("rows", [])
                    if isinstance(row, Mapping)
                ) or "未提供"
                lines.append(
                    "  - Immutable test-label sanity: "
                    f"artifact=**{sanity.get('status', 'INCONCLUSIVE')}**；"
                    f"random-label gate=**{sanity.get('gate_status', 'INCONCLUSIVE')}**；"
                    f"probabilities_immutable={sanity.get('probabilities_immutable', '未提供')}；"
                    f"seed rows: {sanity_cells}。這個 gate INCONCLUSIVE 不能被解讀為 low-separability 證據。"
                )
        if registered.get("manifest_root"):
            lines.append(
                f"- Registered FOMO materialization artifact: `{registered.get('manifest_root')}`；"
                f"protocol={registered.get('protocol', '未提供')}；"
                "its tensor-reader result is a materialization audit and is not a formal classifier fit。"
            )
        external_status = model_grid.get("external_audit_status")
        external_reasons = model_grid.get("external_audit_failure_reasons")
        if external_status:
            reason_text = ", ".join(str(value) for value in external_reasons or []) or "未提供"
            selected_external = _mapping(model_grid.get("selected_external_audit"))
            lines.append(
                f"- Selected final external audit ({selected_external.get('name', '未提供')}): **{external_status}**；failure reasons={reason_text}。"
            )
            for name, history in sorted(_mapping(model_grid.get("historical_external_audits")).items()):
                lines.append(
                    f"- Historical external audit ({name}) is **SUPERSEDED**；recorded status="
                    f"**{history.get('status', 'INCONCLUSIVE')}**；reasons="
                    f"{', '.join(str(value) for value in history.get('failures', []) or []) or '未提供'}。"
                )
        mixed_review = _mapping(model_grid.get("independent_full_mixed_review"))
        if mixed_review:
            lines.append(
                "- Independent full Mixed review evidence: "
                f"status=**{mixed_review.get('status', 'INCONCLUSIVE')}**；"
                f"selected healthy exact tensor records={mixed_review.get('exact_tensor_records', '未提供')}；"
                f"tensor mismatches={mixed_review.get('tensor_mismatch_count', '未提供')}；"
                f"selected BraTS cache files={mixed_review.get('selected_cache_file_count', '未提供')}；"
                f"map failures={mixed_review.get('map_failure_count', '未提供')}；"
                f"source join={mixed_review.get('source_join_status', '未提供')}。"
            )
        parse_warnings = model_grid.get("sidecar_parse_warnings")
        if parse_warnings:
            lines.append(
                "- v3 sidecar serialization warning: "
                + "; ".join(str(value) for value in parse_warnings)
                + "；payload was recovered only for this report and source hash is unchanged。"
            )
        lines.append(
            "- `canonical_parity=PASS` 的範圍是 contract-level metadata/shape/dtype/normalization checks；"
            "這個 build artifact 沒有建立全部 rows 的 exhaustive tensor parity。"
        )
        lines.append(
            "- `selection_equivalence_proof` 只控制歷史 controls/results 能否重用；它不代表 candidate coverage。"
            "candidate coverage 只使用 audit 明確提供的 inventory/pool verification 或 scope status；缺欄位維持 AUDIT_PENDING。"
        )
        lines.extend([
            "",
            "| v3 cohort | build status | records | domain counts | split counts | selected pairs | selected healthy source composition (participants selected/eligible; records) |",
            "|---|---|---:|---|---|---:|---|",
        ])
        for item in model_grid.get("comparisons", []):
            if not isinstance(item, Mapping):
                continue
            audit = _mapping(item.get("audit"))
            domain_counts = json.dumps(item.get("domain_counts", {}), ensure_ascii=False, separators=(",", ":"))
            split_counts = json.dumps(item.get("split_counts", {}), ensure_ascii=False, separators=(",", ":"))
            composition_parts: list[str] = []
            for source_name, split_map in sorted(_mapping(audit.get("source_composition")).items()):
                if not isinstance(split_map, Mapping):
                    continue
                split_parts: list[str] = []
                for split in ("train", "val", "test"):
                    values = _mapping(split_map.get(split))
                    if not values:
                        continue
                    split_parts.append(
                        f"{split}: {values.get('selected_participants', '未提供')}/{values.get('eligible_participants', '未提供')} p, "
                        f"{values.get('selected_records', '未提供')} records"
                    )
                if split_parts:
                    composition_parts.append(f"{source_name} ({'; '.join(split_parts)})")
            composition = "; ".join(composition_parts) if composition_parts else "未由 audit 提供"
            lines.append(
                f"| {item.get('cohort', 'unknown')} | {item.get('status', 'INCONCLUSIVE')} | {item.get('records', '未提供')} | "
                f"`{domain_counts}` | `{split_counts}` | {audit.get('pair_count', '未提供')} | {composition} |"
            )
        mixed_item = next(
            (item for item in model_grid.get("comparisons", []) if isinstance(item, Mapping) and str(item.get("cohort", "")).lower() == "mixed"),
            None,
        )
        if isinstance(mixed_item, Mapping) and _mapping(_mapping(mixed_item.get("audit")).get("source_composition")):
            lines.append(
                "- Mixed source selection 是 audit 所記錄的 participant-balanced subset；selected counts 不代表所有 eligible source participants，"
                "也不代表原始 ANDi slice-frequency mixture。"
            )
        subject_rosters = _mapping(model_grid.get("subject_rosters"))
        if subject_rosters:
            lines.append(
                "- Subject roster export: "
                f"status=**{subject_rosters.get('status', 'INCONCLUSIVE')}**；"
                "由 frozen manifests 逐 split、逐 participant 聚合 selected slice/pair/source 欄位，"
                f"overlap audit=`{subject_rosters.get('audit_path', '未提供')}`。"
            )
            if subject_rosters.get("status") == "PASS":
                lines.append(
                    f"  - train/val/test CSV 與各 comparison CSV 位於 `{subject_rosters.get('output_dir', '未提供')}`；"
                    "participant 與 pair split overlap audit 均為空。"
                )
    shuffle_diagnostic = _mapping(context.get("shuffle_diagnostic_summary"))
    if shuffle_diagnostic:
        auc_summary = _mapping(shuffle_diagnostic.get("seed273_auc"))
        train_auc = _mapping(auc_summary.get("train"))
        val_auc = _mapping(auc_summary.get("val"))
        test_auc = _mapping(auc_summary.get("test"))
        lines.append(
            "- Shuffle diagnostic addendum: "
            f"status={shuffle_diagnostic.get('status', 'INCONCLUSIVE')}; "
            f"saved-model orientation={shuffle_diagnostic.get('orientation_status', 'INCONCLUSIVE')}; "
            f"identity/split/prediction mismatch count={shuffle_diagnostic.get('integrity_issue_count', '未提供')}; "
            f"seed273 true-label subject AUC train/val/test="
            f"{_fmt(train_auc.get('true_subject_auc'))}/{_fmt(val_auc.get('true_subject_auc'))}/{_fmt(test_auc.get('true_subject_auc'))}; "
            f"pseudo train={_fmt(train_auc.get('pseudo_subject_auc'))}."
        )
        lines.append(
            "- Shuffle diagnostic interpretation: "
            f"{shuffle_diagnostic.get('interpretation', '未提供')}。 "
            f"seed273 train true-domain subject counts="
            f"{json.dumps(shuffle_diagnostic.get('seed273_train_domain_subjects'), ensure_ascii=False, separators=(',', ':'))}; "
            f"val={json.dumps(shuffle_diagnostic.get('seed273_val_domain_subjects'), ensure_ascii=False, separators=(',', ':'))}."
        )
        lines.append(
            "- Shuffle v1 limitation: "
            f"{shuffle_diagnostic.get('v1_slice_selection_limitation', '未提供')}; "
            f"test slice counts by seed={json.dumps(shuffle_diagnostic.get('v1_test_slice_counts'), ensure_ascii=False, separators=(',', ':'))}; "
            f"test pair counts={json.dumps(shuffle_diagnostic.get('v1_test_pair_counts'), ensure_ascii=False, separators=(',', ':'))}. "
            "Main formal split seed=73 is unchanged."
        )
        lines.append(
            "- Shuffle diagnostic limitation (historical v1 planning field): "
            f"{shuffle_diagnostic.get('tensor_hash_limitation', 'per-row tensor hashes were not persisted')}; "
            f"recorded next-protocol field `{shuffle_diagnostic.get('next_protocol', 'unrecorded')}` status="
            f"**{shuffle_diagnostic.get('next_protocol_status', 'PENDING')}**. "
            "Protocol v2 results are reported below; this historical field is not evidence of formal completion."
        )
        reference_parts = []
        for name, fingerprint in _mapping(shuffle_diagnostic.get("references")).items():
            if isinstance(fingerprint, Mapping):
                reference_parts.append(f"`{name}` sha256={fingerprint.get('sha256', 'unavailable')}")
        if reference_parts:
            lines.append("- Shuffle diagnostic references: " + "; ".join(reference_parts) + ".")
    # The v2 artifact retains its original next-stage planning fields for
    # provenance.  Once the separately authorized calibration null is
    # complete, the human report must describe that planning field as
    # historical rather than repeat its stale "not yet executed" text.
    calibration_for_planning = _mapping(context.get("calibration_summary"))
    planning_null = _mapping(calibration_for_planning.get("full_retrained_null"))
    planning_null_complete = (
        str(planning_null.get("status", "")).upper() == "COMPLETE"
        and int(planning_null.get("completed", 0) or 0) >= 199
        and int(planning_null.get("requested", 0) or 0) == 199
    )
    protocol_v2 = _mapping(context.get("protocol_v2_summary"))
    if protocol_v2:
        true_gate = _mapping(protocol_v2.get("true_test_gate"))
        random_gate = _mapping(protocol_v2.get("random_test_label_gate"))

        def gate_cells(gate: Mapping[str, Any]) -> str:
            cells: list[str] = []
            for item in gate.get("rows", []):
                if not isinstance(item, Mapping):
                    continue
                cells.append(
                    f"s{item.get('seed')}: AUC={_fmt(item.get('subject_auc'))} "
                    f"[{_fmt(item.get('bootstrap_ci_low'))}, {_fmt(item.get('bootstrap_ci_high'))}], "
                    f"Holm p={_fmt(item.get('holm_p'))}"
                )
            return "; ".join(cells) if cells else "未提供"

        lines.append(
            "- Supplementary protocol v2: "
            f"`{protocol_v2.get('protocol', 'balanced_powered_shuffle_v2')}`; "
            f"split_seed={protocol_v2.get('split_seed', '未提供')}; "
            f"fit_seeds={json.dumps(protocol_v2.get('fit_seeds', []), ensure_ascii=False)}; "
            f"preflight={protocol_v2.get('preflight_status', 'INCONCLUSIVE')}; "
            f"retrained_permutation_null={protocol_v2.get('retrained_permutation_null', '未提供')}; "
            f"classification={protocol_v2.get('confirmatory_status', 'EXCLUDED_DIAGNOSTIC')}."
        )
        lines.extend([
            "",
            "| protocol v2 gate | status | per-seed subject AUC / recorded CI / Holm p |",
            "|---|---|---|",
            f"| true test labels retained | {true_gate.get('status', 'INCONCLUSIVE')} | {gate_cells(true_gate)} |",
            f"| independent random held-out test labels | {random_gate.get('status', 'INCONCLUSIVE')} | {gate_cells(random_gate)} |",
        ])
        if planning_null_complete:
            lines.append(
                f"- Protocol v2 interpretation: {protocol_v2.get('interpretation', '未提供')} "
                "The recorded Stage2 planning field is historical/superseded for the authorized FOMO calibration "
                f"(recorded status=**{protocol_v2.get('next_stage2_status', 'PENDING')}**); "
                f"the full-retrained paired-label null is **COMPLETE ({planning_null.get('completed', 0)}/{planning_null.get('requested', 199)})** and is reported below. "
                "No matrix expansion is claimed from this legacy protocol-v2 field。"
            )
        else:
            lines.append(
                f"- Protocol v2 interpretation: {protocol_v2.get('interpretation', '未提供')} "
                f"Next Stage2 planning state (not a completed matrix result)=**{protocol_v2.get('next_stage2_status', 'PENDING')}**; "
                f"{protocol_v2.get('next_stage2', 'not executed')}。No matrix expansion is claimed from this legacy protocol-v2 planning field。"
            )
    calibration = calibration_for_planning
    if calibration:
        heldout_swap = _mapping(calibration.get("heldout_pair_swap"))
        null = _mapping(calibration.get("full_retrained_null"))
        ci = f"[{_fmt(calibration.get('subject_bootstrap_ci_low'))}, {_fmt(calibration.get('subject_bootstrap_ci_high'))}]"
        null_label = (
            f"{null.get('status', 'INCONCLUSIVE')} "
            f"({null.get('completed', 0)}/{null.get('requested', '未提供')})"
        )
        lines.extend([
            "",
            "## Observed calibration (diagnostic only)",
            "",
            "這個 calibrated FOMO SmallCNN observed fit 只作 capacity/calibration audit；它沒有併入 formal final rows、matrix 或 ANDi/AP correlation。",
            "",
            "| fit | seed/split | test subject AUC | recorded subject bootstrap CI | test slice AUC | val subject AUC | fixed-classifier heldout pair swap | full-retrained paired-label null (conditional matched sample) | classification |",
            "|---|---|---:|---|---:|---:|---|---|---|",
            f"| FOMO SmallCNN observed | {calibration.get('seed', '未提供')}/{calibration.get('split_seed', '未提供')} | "
            f"{_fmt(calibration.get('subject_auc'))} | {ci} (n={calibration.get('subject_bootstrap_valid_n', '未提供')}) | "
            f"{_fmt(calibration.get('slice_auc'))} | {_fmt(calibration.get('validation_subject_auc'))} | "
            f"{heldout_swap.get('status', 'INCONCLUSIVE')}, p={_fmt(heldout_swap.get('p_value'))}, n={heldout_swap.get('n_swaps', '未提供')} | "
            f"{null_label} | {calibration.get('classification', 'DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY')} |",
            f"- Calibration protocol=`{calibration.get('protocol', '未提供')}`; best_epoch={calibration.get('best_epoch', '未提供')}; "
            f"epochs_completed={calibration.get('epochs_completed', '未提供')}; permutation_mode={calibration.get('permutation_mode', '未提供')}; "
            f"requested full-retrained replicates={calibration.get('permutation_replicates_requested', '未提供')}。"
            f" null_status_updated={null.get('updated_at_utc', '未提供')}。"
            " Observed bootstrap CI is copied from result.json; it is not recomputed by this report.",
        ])
        reference_parts = []
        for name, fingerprint in _mapping(calibration.get("references")).items():
            if name in {"observed/result.json", "protocol.json", "retrained_null/status.json", "retrained_null/summary.json"} and isinstance(fingerprint, Mapping):
                reference_parts.append(f"`{name}` sha256={fingerprint.get('sha256', 'unavailable')}")
        if reference_parts:
            lines.append("- Calibration references: " + "; ".join(reference_parts) + ".")
        if calibration.get("full_retrained_null_plot"):
            lines.append(
                "- Full-retrained null histogram: "
                f"`{calibration.get('full_retrained_null_plot')}`; conditional on matched pairs; "
                f"T=abs(subject_mean_score_roc_auc-0.5), p+1={_fmt(null.get('p_plus_one'))}, "
                "minimum attainable p=0.005 for 199 replicates."
            )
        lines.append(
            "- Rollout state: the historical v1/v2 shuffle/control FAIL is retained as a halt for that legacy protocol only; "
            "it is not the v3 Stage-A control gate. The completed calibrated FOMO null closes only the old diagnostic "
            "sub-study. The v3 Stage-A observed fit and tiny/positive/negative controls are reported separately, while "
            "the v3 full-retrained null and the remaining matrix cells are still pending."
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| cohort | modalities | classifier | seed | control/run mode | result kind | slice AUC | subject AUC | train-final acc/BCE | bootstrap CI | bootstrap source | combined control gate | low-separability gate | evidence |",
        "|---|---|---|---:|---|---|---:|---:|---|---|---|---|---|---|",
    ])
    if rows:
        for row in rows:
            ci = f"[{_fmt(row.get('bootstrap_ci_low') )}, {_fmt(row.get('bootstrap_ci_high'))}]"
            lines.append(
                f"| {row.get('healthy_domain', 'unknown')} | {row.get('input_modalities', '未提供')} | {row.get('classifier', 'unknown')} | {_fmt(row.get('seed'), 0)} | {row.get('control_mode', 'formal_final')} | {row.get('result_kind', 'unknown')} | {_fmt(row.get('slice_roc_auc'))} | {_fmt(row.get('subject_roc_auc'))} | {_fmt(row.get('train_final_accuracy'))}/{_fmt(row.get('train_final_bce'))} | {ci} | {row.get('bootstrap_source', '未提供')} | {row.get('control_gate_status', 'INCONCLUSIVE')} | {row.get('low_separability_status', 'INCONCLUSIVE')} | {row.get('evidence_status', 'INCONCLUSIVE')} |"
            )
    else:
        lines.append("| 未執行/未找到 artifacts |  |  |  |  |  |  |  |  | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |")
    lines.extend([
        "",
        "## 九個問題",
        "",
    ])
    # Do not allow a positive/negative control or registered-stage metric to
    # become a fake cohort observation in any of the nine answers.
    finite_rows = [row for row in formal_rows if _finite(row.get("subject_roc_auc")) is not None]
    calibration_observed = (
        calibration
        if calibration
        and _finite(calibration.get("subject_auc")) is not None
        and str(calibration.get("classification", "")).startswith("DIAGNOSTIC_CALIBRATION")
        else {}
    )
    calibration_null = _mapping(calibration_observed.get("full_retrained_null"))
    calibration_null_complete = (
        str(calibration_null.get("status", "")).upper() == "COMPLETE"
        and int(calibration_null.get("completed", 0) or 0) >= 199
        and int(calibration_null.get("requested", 0) or 0) == 199
    )
    if finite_rows:
        best = max(finite_rows, key=lambda row: float(row.get("subject_direction_invariant_auc") or 0.0))
        worst = min(finite_rows, key=lambda row: float(row.get("subject_direction_invariant_auc") or 2.0))
        median = float(np.median([float(row["subject_direction_invariant_auc"]) for row in finite_rows]))
        lines.append(f"1. **Healthy 與 BraTS 在 final input 是否仍可辨識？** 已讀到 {len(finite_rows)} 個 finite subject-level result；direction-invariant AUC 中位數為 **{median:.4f}**。單一 AUC 不等同 scanner distance，gate 仍須看 controls 與 CI。")
        lines.append(f"2. **程度有多高？** 目前最高 cohort/model 為 `{best.get('healthy_domain')} / {best.get('classifier')}`（AUC={_fmt(best.get('subject_roc_auc'))}，CI={_fmt(best.get('bootstrap_ci_low'))}–{_fmt(best.get('bootstrap_ci_high'))}）；最低為 `{worst.get('healthy_domain')} / {worst.get('classifier')}`（AUC={_fmt(worst.get('subject_roc_auc'))}）。")
    elif calibration_observed and calibration_null_complete:
        calibration_auc = _fmt(calibration_observed.get("subject_auc"))
        calibration_ci = f"[{_fmt(calibration_observed.get('subject_bootstrap_ci_low'))}, {_fmt(calibration_observed.get('subject_bootstrap_ci_high'))}]"
        lines.append(
            "1. **Healthy 與 BraTS 在 final input 是否仍可辨識？** "
            f"在 FOMO 3-modal SmallCNN calibration setting，observed subject AUC={calibration_auc}、"
            f"recorded CI={calibration_ci}，且 199-replicate full-retrain matched-pair null 完成（p+1={_fmt(calibration_null.get('p_plus_one'))}）。"
            "這支持該 FOMO setting 存在 strong healthy/BraTS input fingerprint；它仍是 diagnostic sub-study，不能外推到其他 cohort 或宣稱 formal matrix 完成。"
        )
        lines.append(
            "2. **程度有多高？** "
            f"FOMO calibration 的 observed subject AUC={calibration_auc}（T=abs(AUC−0.5)=0.5000；full-retrain p+1={_fmt(calibration_null.get('p_plus_one'))}），"
            "是此 setting 的 maximal observed separability；不代表已完成跨 cohort/modalities 的程度比較。"
        )
    else:
        lines.append("1. **Healthy 與 BraTS 在 final input 是否仍可辨識？** formal final matrix 尚未完成（`INCONCLUSIVE / matrix112pending`）；目前只有 controls，不能回答。")
        lines.append("2. **程度有多高？** formal final matrix 尚未完成（`INCONCLUSIVE / matrix112pending`）；不能回答。")
    controls_for_capacity = _mapping(context.get("controls"))
    tiny_capacity = _mapping(controls_for_capacity.get("tiny_overfit"))
    positive_capacity = _mapping(controls_for_capacity.get("positive_control"))
    capacity_setting_pass = (
        bool(calibration_observed)
        and str(tiny_capacity.get("status", "")).upper() == "PASS"
        and str(positive_capacity.get("status", "")).upper() == "PASS"
        and _finite(calibration_observed.get("validation_subject_auc")) is not None
    )
    control_pass = bool(finite_rows) and all(row.get("control_gate_status") == "PASS" for row in finite_rows)
    if capacity_setting_pass:
        lines.append(
            "3. **AUC 接近 0.5 是否排除 classifier 太弱？** "
            f"在 FOMO calibration setting，tiny overfit 與 registered positive controls 都 PASS，且 validation subject AUC={_fmt(calibration_observed.get('validation_subject_auc'))}；"
            "因此 classifier capacity 在此 setting 有足夠的辨識能力。這不替代其他 classifier/cohort controls，也不表示 v3 full matrix 已完成；legacy shuffle FAIL 只屬舊 protocol。"
        )
    else:
        lines.append(f"3. **AUC 接近 0.5 是否排除 classifier 太弱？** {'所有已讀到的 rows 都有四項 controls PASS。' if control_pass else '不能排除；至少一項 control 缺失、FAIL 或沒有完整 evidence。'} AUC 接近 0.5 本身不能證明資料完全相同。")
    by_modality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in finite_rows:
        by_modality[str(row.get("input_modalities", "未提供"))].append(row)
    if by_modality:
        modality = max(by_modality, key=lambda key: np.mean([float(item.get("subject_direction_invariant_auc") or 0.0) for item in by_modality[key]]))
        lines.append(f"4. **哪個 modality 保留最多 domain information？** 在目前 rows 中，描述性最高為 `{modality}`；若缺少 modality-balanced runs，這只是目前 artifacts 的排序。")
    else:
        lines.append("4. **哪個 modality 保留最多 domain information？** 沒有 modality-comparable results，不能回答。")
    if finite_rows:
        lines.append(f"5. **哪個 healthy cohort 最接近/最遠？** 依 direction-invariant separability，最低 `{worst.get('healthy_domain')}`、最高 `{best.get('healthy_domain')}`；這描述 dataset/acquisition fingerprint，不直接量化 scanner 距離。")
    else:
        lines.append("5. **哪個 healthy cohort 最接近/最遠？** 沒有可比結果，不能回答。")
    classifiers = defaultdict(list)
    for row in finite_rows:
        classifiers[str(row.get("classifier"))].append(float(row.get("subject_direction_invariant_auc") or 0.0))
    if len(classifiers) >= 2:
        spread = {key: float(np.mean(value)) for key, value in classifiers.items()}
        lines.append(f"6. **Logistic、Small CNN、ResNet18 是否一致？** 目前 classifier means: `{json.dumps(spread, ensure_ascii=False)}`。差異代表 spatial/texture capacity 可能捕捉到 global statistics 看不到的 fingerprint；若某 capacity control FAIL，不能做這個解讀。")
    else:
        lines.append("6. **三種 classifier 是否一致？** 尚未同時讀到至少兩種 capacity 的結果。")
    if correlation.get("n_points", 0):
        lines.append(f"7. **Domain AUC 與 ANDi AP 是否有 descriptive relationship？** n={correlation.get('n_points')}，Pearson r={_fmt(correlation.get('pearson_r'))}、Spearman ρ={_fmt(correlation.get('spearman_rho'))}。{correlation.get('caveat')}")
    else:
        lines.append(
            "7. **Domain AUC 與 ANDi AP 是否有 descriptive relationship？** "
            f"目前有 {correlation.get('available_ap_points', '未提供')} 個 ANDi AP artifacts，但 paired formal domain-AUC/AP points n=0；不能計算 Pearson/Spearman。"
        )
    lines.append("8. **是否已有證據支持 cross-domain acquisition difference 是 ANDi performance gap 的重要來源？** Domain classifier 若成功只能證明影像存在 dataset/acquisition fingerprint；四點 scatter 仍是描述性診斷，不能證明 ANDi gap 的因果重要性。")
    lines.append("9. **目前能否說 scanner hardware 本身是主因？** **不能。** 沒有 scanner/protocol controlled analysis 時，只能說 dataset/acquisition/domain fingerprint 是否存在。")
    lines.extend([
        "",
        "## Controls 與判讀規則",
        "",
        "- Methods references： [Lopez-Paz & Oquab, C2ST](https://arxiv.org/abs/1610.06545) supports the held-out classification interpretation of a two-sample test；這不會自動驗證本研究的 pair selection 或 subject-level aggregation。Permutation p-values follow the +1 convention described by [Phipson & Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)；B=199 的 attainable minimum is 0.005。Random TRAIN-label assignment does not guarantee that the real TEST-domain AUC is approximately 0.5；therefore a deviation from chance alone does not prove leakage，需和完整 control/audit 一起判讀。",
        "- Tiny overfit gate：training accuracy ≥ 0.99 且 BCE ≤ 0.02；未達時正式 AUC 不可解讀。",
        "- Registered positive gate：AUC ≥ 0.80 且 bootstrap CI lower > 0.50。",
        "- Negative/shuffle controls：CI 必須整段落在 [0.35, 0.65]，並且 two-sided deviation 沒有顯著證據。FAIL 或 INCONCLUSIVE 都不能支持 low-separability claim。沒有顯著 p 值也不等於 CI 接近 0.5。",
        "- Fixed-classifier heldout pair swap 只是條件於 within-pair exchangeability 的 association test；目前 pairing 是各 split 兩側以獨立 seeded permutations 後 zip，再在 20 個 z bins 內按 cap 選 slice，不是 greedy matching。domain-specific lesion exclusion 仍限制其作為 unconditional C2ST 的解釋；full-retrained paired-label null 是另一個 conditional matched-sample procedure，必須另行保存。",
        "- low-separability gate 使用 direction-invariant max(AUC, 1−AUC) 的 conservative CI upper < 0.60，並要求所有 capacity/control evidence PASS。",
        "- Eligibility limitation：BraTS subjects are tumor patients；native/model mask zero 是選定 slice 的 tumor-free criterion，不是整個 subject 健康。lesion 若在其他 slice，仍會影響 full-volume robust IQR normalization；20-bin z matching 只控制粗略 slice-location 分布，不等於 exact anatomical alignment。",
        "- Interpretation limitation：AUC=1 alone 不能歸因為 scanner/acquisition、preprocessing quality 或 ANDi performance gap 的因果來源；這些需要另外的 controlled analysis。",
        "",
        "## Rollout protocol",
        "",
        "1. 先完成 manifest、participant split、20-bin z matching、native/model mask audit 與 tests。",
        "2. 只跑 FOMO 3-modal Small CNN 加四個 controls；任何 control FAIL 就停止擴張。",
        "3. 通過後才擴張單模態、Logistic、ResNet18、MPI/OASIS3/Mixed、bootstrap、permutation、multiple seeds。",
        "",
        "## Provenance",
        "",
        f"- source artifacts: {len(context.get('prediction_paths', []))} prediction files, {len(context.get('metric_paths', []))} metric files。",
        "- report does not overwrite source datasets, LMDBs, checkpoints, or original audits.",
    ])
    controls = context.get("controls", {})
    if isinstance(controls, Mapping) and controls:
        lines.extend(["", "### 已讀到的 control artifacts", ""])
        for name in ("tiny_overfit", "positive_control", "negative_control", "label_permutation"):
            item = controls.get(name)
            if not isinstance(item, Mapping):
                continue
            status = item.get("status", "INCONCLUSIVE")
            gate_name = item.get("name", name)
            mode_label = item.get("control_mode")
            if mode_label:
                gate_label = f"{mode_label} (gate={gate_name})"
            else:
                gate_label = str(gate_name)
            lines.append(f"- `{gate_label}`: **{status}**；source=`{item.get('source_path', '未提供')}`")
            if name == "label_permutation" and str(status).upper() == "FAIL":
                failure_seeds = item.get("failure_p_value_seeds", [])
                holm_values = item.get("holm_adjusted_p_values", {})
                lines.append(
                    "  - shuffle gate failure evidence: "
                    f"failure_p_value_seeds={json.dumps(failure_seeds, ensure_ascii=False)}; "
                    f"Holm_adjusted_p_values={json.dumps(holm_values, ensure_ascii=False)}。"
                )
                lines.append("  - 這個 FAIL 只會停止 legacy v1/v2 protocol 的 formal rollout；不單獨證明 information leakage 或指定成因，也不會改寫 v3 Stage-A gate。")
            pair_swap = item.get("pair_swap")
            if isinstance(pair_swap, Mapping):
                lines.append(
                    f"  - heldout pair-swap sanity (fixed-classifier association): **{pair_swap.get('status', 'INCONCLUSIVE')}**; "
                    f"completed_seeds={pair_swap.get('completed_seeds', '未提供')}。"
                )
            null = item.get("full_retrained_null")
            if isinstance(null, Mapping):
                lines.append(
                    f"  - full-retrained paired-label null (conditional matched sample): **{null.get('status', 'INCONCLUSIVE')}**；"
                    f"permutation_mode={null.get('permutation_mode', '未記錄')}；unit={null.get('unit', '未記錄')}；"
                    f"completed={null.get('completed', '未提供')}；requested={null.get('requested', '未提供')}。"
                )
    return "\n".join(lines) + "\n"


def build_report(
    source: Any,
    output_dir: str | Path,
    *,
    config: ReportConfig | None = None,
    andi_ap_source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    andi_ap_root: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Build summary CSV, manifest/audit JSON, Markdown, and figures."""

    cfg = config or ReportConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    rows, context = build_summary_rows(source, config=cfg)
    model_grid_context = _mapping(context.get("model_grid_v3_summary"))
    roster_export: dict[str, Any] | None = None
    build_root_text = model_grid_context.get("build_path")
    if build_root_text:
        try:
            build_root = Path(str(build_root_text))
            # ``model_grid_v3_summary.build_path`` is the explicit
            # build_summary.json path; the roster reader consumes the frozen
            # revision directory containing ``manifests/``.
            if build_root.is_file() and build_root.name.lower() == "build_summary.json":
                build_root = build_root.parent
            roster_export = export_model_grid_v3_rosters(
                build_root,
                destination / "v3_subject_rosters",
            )
        except (OSError, TypeError, ValueError) as exc:
            roster_export = {
                "status": "INCONCLUSIVE_ROSTER_EXPORT_ERROR",
                "source_build_root": str(build_root_text),
                "output_dir": str((destination / "v3_subject_rosters").resolve()),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        model_grid_context["subject_rosters"] = roster_export
        context["model_grid_v3_roster_export"] = roster_export
    formal_rows = [row for row in rows if _is_formal_final_row(row)]
    andi_points = load_andi_ap_points(andi_ap_source, root=andi_ap_root)
    finite_points = {
        key: value
        for key, value in andi_points.items()
        if _finite(value.get("mf_ap")) is not None
    }
    # ANDi/AP association is a formal-final cohort diagnostic.  Controls and
    # registered/intermediate rows may carry AUC-like numbers, but are never
    # paired with AP or counted as cohort points.
    correlation = compute_andi_ap_correlation(formal_rows, finite_points)
    correlation["available_ap_points"] = len(finite_points)
    plot_status = generate_plots(rows, context, destination / "figures", andi_correlation=correlation) if make_plots else {}
    summary_path = write_summary_csv(destination / "summary.csv", rows)
    legacy_rollout_gate_status = "HALTED_CONTROL_FAIL" if _control_failure_present(context) else None
    v3_stage_a_gate_status = (
        str(model_grid_context.get("v3_stage_a_gate_status"))
        if model_grid_context and model_grid_context.get("v3_stage_a_gate_status")
        else _v3_stage_a_gate_status(model_grid_context)
        if model_grid_context
        else None
    )
    # A v3 build has its own namespaced controls.  Use that status for the
    # report's primary rollout field and retain the legacy status separately;
    # a historical shuffle FAIL must not silently halt the v3 gate.
    rollout_gate_status = v3_stage_a_gate_status if model_grid_context else legacy_rollout_gate_status
    context["legacy_rollout_gate_status"] = legacy_rollout_gate_status
    context["v3_stage_a_gate_status"] = v3_stage_a_gate_status
    roster_source_paths = list(_mapping(roster_export).get("source_manifest_paths", [])) if roster_export else []
    roster_output_paths = list(_mapping(roster_export).get("output_paths", [])) if roster_export else []
    for path in (_mapping(roster_export).get("manifest_path"), _mapping(roster_export).get("audit_path")) if roster_export else ():
        if path:
            roster_output_paths.append(str(path))
    referenced_files = {
        "authoritative_source": list(context.get("authoritative_paths", [])),
        "prediction": list(context.get("prediction_paths", [])),
        "metrics": list(context.get("metric_paths", [])),
        "controls": sorted(
            str(value.get("source_path"))
            for value in context.get("controls", {}).values()
            if isinstance(value, Mapping) and value.get("source_path")
        ),
        "training": list(context.get("training_files", [])),
        "andi_ap": sorted(
            str(value.get("path"))
            for value in andi_points.values()
            if isinstance(value, Mapping) and value.get("path")
        ),
        "shuffle_diagnostic": sorted(
            str(value.get("path"))
            for value in _mapping(context.get("shuffle_diagnostic_summary", {})).get("references", {}).values()
            if isinstance(value, Mapping) and value.get("path")
        ),
        "protocol_v2": sorted(
            str(value.get("path"))
            for value in _mapping(context.get("protocol_v2_summary", {})).get("references", {}).values()
            if isinstance(value, Mapping) and value.get("path")
        ),
        "calibration": sorted(
            str(value.get("path"))
            for value in _mapping(context.get("calibration_summary", {})).get("references", {}).values()
            if isinstance(value, Mapping) and value.get("path")
        ),
        "model_grid_v3": sorted(set(str(path) for path in context.get("model_grid_v3_paths", []) if path) | set(roster_source_paths)),
        "generated_subject_rosters": sorted(set(str(path) for path in roster_output_paths if path)),
    }
    manifest = {
        "schema_version": 2,
        "kind": "domain_classifier_report",
        "status": "COMPLETE",
        "generated_at_utc": generated_at_utc,
        "scientific_status": "COMPLETE" if formal_rows else "INCONCLUSIVE",
        "formal_matrix_status": "COMPLETE" if formal_rows else "matrix112pending",
        "rollout_gate_status": rollout_gate_status,
        "legacy_rollout_gate_status": legacy_rollout_gate_status,
        "v3_stage_a_gate_status": v3_stage_a_gate_status,
        "v3_primary_observed_fit_count": model_grid_context.get("v3_primary_observed_fit_count", 0) if model_grid_context else 0,
        "v3_confirmatory_fit_count": model_grid_context.get("v3_confirmatory_fit_count", 0) if model_grid_context else 0,
        "formal_final_rows": len(formal_rows),
        "excluded_nonformal_rows": len(rows) - len(formal_rows),
        "config": _json_safe(cfg.__dict__),
        "summary_fields": ALL_FIELDS,
        "referenced_files": _json_safe(referenced_files),
        "stage1": _json_safe(context.get("stage1_summary")),
        "parity": _json_safe(context.get("parity_summary")),
        "shuffle_diagnostic": _json_safe(context.get("shuffle_diagnostic_summary")),
        "protocol_v2": _json_safe(context.get("protocol_v2_summary")),
        "calibration": _json_safe(context.get("calibration_summary")),
        "model_grid_v3": _json_safe(model_grid_context or context.get("model_grid_v3_summary")),
        "source": _json_safe({key: value for key, value in context.items() if key not in {"prediction_context", "root"}}),
        "rows": _json_safe(rows),
        "andi_ap_points": _json_safe(andi_points),
        "correlation": _json_safe(correlation),
        "plots": _json_safe(plot_status),
    }
    manifest_path = destination / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    fingerprint_paths: list[str] = []
    for path_group in referenced_files.values():
        for path_text in path_group:
            if path_text and path_text not in fingerprint_paths:
                fingerprint_paths.append(path_text)
    fingerprints = [_fingerprint(path_text) for path_text in fingerprint_paths]
    audit = {
        "schema_version": 2,
        "status": "COMPLETE",
        "generated_at_utc": generated_at_utc,
        "scientific_status": "COMPLETE" if formal_rows else "INCONCLUSIVE",
        "formal_matrix_status": "COMPLETE" if formal_rows else "matrix112pending",
        "rollout_gate_status": rollout_gate_status,
        "legacy_rollout_gate_status": legacy_rollout_gate_status,
        "v3_stage_a_gate_status": v3_stage_a_gate_status,
        "v3_primary_observed_fit_count": model_grid_context.get("v3_primary_observed_fit_count", 0) if model_grid_context else 0,
        "v3_confirmatory_fit_count": model_grid_context.get("v3_confirmatory_fit_count", 0) if model_grid_context else 0,
        "rows": len(rows),
        "formal_final_rows": len(formal_rows),
        "excluded_nonformal_rows": len(rows) - len(formal_rows),
        "raw_rows": context.get("raw_row_count", 0),
        "source_fingerprints": fingerprints,
        "referenced_files": _json_safe(referenced_files),
        "stage1": _json_safe(context.get("stage1_summary")),
        "parity": _json_safe(context.get("parity_summary")),
        "shuffle_diagnostic": _json_safe(context.get("shuffle_diagnostic_summary")),
        "protocol_v2": _json_safe(context.get("protocol_v2_summary")),
        "calibration": _json_safe(context.get("calibration_summary")),
        "model_grid_v3": _json_safe(model_grid_context or context.get("model_grid_v3_summary")),
        "controls": _json_safe(context.get("controls", {})),
        "notes": [
            "No empirical values are synthesized when artifacts are absent.",
            "Subject-level bootstrap is the primary uncertainty unit; slices are not iid bootstrap observations.",
            "Authoritative preparation audit/manifest files are referenced and fingerprinted; report generation does not overwrite them.",
            "Scientific questions and ANDi/AP correlation use formal final-input rows only; control/intermediate rows are excluded.",
            "v3 model-grid build artifacts are build-only provenance; their metadata/canonical contract checks do not establish exhaustive tensor parity or scientific readiness.",
        ],
    }
    audit_path = destination / "report_audit.json"
    audit_path.write_text(json.dumps(_json_safe(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = destination / "report.md"
    report_path.write_text(_report_markdown(rows, correlation, context, cfg), encoding="utf-8")
    return {
        "rows": rows,
        "summary_path": str(summary_path.resolve()),
        "report_path": str(report_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "andi_ap_points": andi_points,
        "correlation": correlation,
        "plots": plot_status,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Aggregate domain-classifier metrics into an auditable report")
    parser.add_argument("source", type=Path, help="Run directory or metrics/predictions artifact")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/domain_classifier"))
    parser.add_argument("--andi-ap", type=Path, default=None, help="Explicit design audit/AP artifact")
    parser.add_argument("--andi-ap-root", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = build_report(
        args.source,
        args.output_dir,
        config=ReportConfig(bootstrap_replicates=args.bootstrap),
        andi_ap_source=args.andi_ap,
        andi_ap_root=args.andi_ap_root,
        make_plots=not args.no_plots,
    )
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "ALL_FIELDS",
    "DEFAULT_SECONDARY_SEEDS",
    "EXTENDED_FIELDS",
    "KNOWN_ANDI_AP_PATHS",
    "PRIMARY_FULL_RETRAINED_HOLM4_FAMILY",
    "ReportConfig",
    "SECONDARY_48_FAMILY",
    "SUMMARY_FIELDS",
    "aggregate_fixed_seed_subject_ensemble",
    "build_report",
    "build_summary_rows",
    "compute_andi_ap_correlation",
    "compute_secondary_ensemble_statistics",
    "export_model_grid_v3_rosters",
    "generate_calibration_diagnostic_plots",
    "generate_plots",
    "load_andi_ap_points",
    "summarise_secondary_48",
    "write_summary_csv",
]
