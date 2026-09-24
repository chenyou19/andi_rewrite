"""Run a strict read-only audit over the completed model-grid v3 build."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import atomic_write_json, read_jsonl_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _subject_identity(value: Any) -> str:
    """Return the source subject key shared by namespaced and plain records.

    Candidate rows use the dataset-namespaced participant identifier (for
    example ``brats21:BraTS2021_00674``), while the durable per-subject
    inventory ledger stores the source subject key (``BraTS2021_00674``).
    This audit compares the two representations without changing either
    artifact.
    """

    text = str(value or "")
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        if prefix in {"brats21", "mpi", "oasis3", "healthy", "mixed"}:
            return suffix
    return text


def audit_build(output_root: Path) -> dict[str, Any]:
    summary = _load(output_root / "build_summary.json")
    inventory = _load(output_root / "brats_candidate_inventory.json")
    rows_path = output_root / "brats_candidate_rows.jsonl"
    ledger_path = output_root / "brats_candidate_completion.jsonl"
    rows = read_jsonl_manifest(rows_path)
    ledger = read_jsonl_manifest(ledger_path)
    candidate_keys = [(str(row.get("participant_id", "")), int(row.get("z", -1))) for row in rows]
    duplicate_keys = [list(key) for key, count in Counter(candidate_keys).items() if count > 1]
    candidate_subject_counts = Counter(_subject_identity(key[0]) for key in candidate_keys)
    expected_subject_counts = {
        _subject_identity(item["subject"]): int(item.get("support_pass_slices", 0))
        for item in inventory.get("subject_audit", [])
    }
    row_count_mismatches = [
        {"subject": subject, "rows": candidate_subject_counts.get(subject, 0), "expected": expected}
        for subject, expected in expected_subject_counts.items()
        if candidate_subject_counts.get(subject, 0) != expected
    ]
    ledger_subjects = [str(item.get("subject", "")) for item in ledger]
    csv_audit = inventory.get("source", {}).get("csv", {})
    required_audits = {}
    for comparison in ("fomo45k", "mpi", "oasis3", "mixed"):
        path = output_root / "audits" / f"{comparison}.json"
        audit = _load(path)
        required_audits[comparison] = {
            "status": audit.get("status"),
            "records": audit.get("records"),
            "source_split_local_joins": audit.get("source_split_local_joins", {}).get("status"),
            "participant_overlap": audit.get("participant_overlap", {}).get("status"),
            "support_eligibility": audit.get("support_eligibility", {}).get("status"),
            "z_norm": audit.get("z_norm", {}).get("status"),
            "mixed_tensor_parity": audit.get("mixed_tensor_parity", {}).get("status", "NOT_APPLICABLE"),
            "brats_selected_tensor_cache": audit.get("brats_selected_tensor_cache", {}).get("status"),
        }
    failures: list[str] = []
    if summary.get("status") != "PASS" or summary.get("training_started") is not False:
        failures.append("build_summary_not_pass_or_training_started")
    if inventory.get("status") != "PASS" or not inventory.get("full_candidate_coverage", {}).get("verified"):
        failures.append("candidate_inventory_not_verified")
    if int(csv_audit.get("csv_rows", -1)) != 938 or not bool(csv_audit.get("id_set_equal")):
        failures.append("csv_map_identity_not_938_exact")
    if len(rows) != int(inventory.get("model_grid_counts", {}).get("candidate_rows", -1)):
        failures.append("candidate_row_count_mismatch")
    if len(ledger_subjects) != 938 or len(set(ledger_subjects)) != 938:
        failures.append("completion_ledger_not_exact_938_unique")
    if duplicate_keys:
        failures.append("duplicate_candidate_participant_z")
    if row_count_mismatches:
        failures.append("candidate_rows_do_not_match_support_counts")
    for comparison, audit in required_audits.items():
        if audit["status"] != "PASS":
            failures.append(f"{comparison}_audit_not_pass")
        if audit["mixed_tensor_parity"] == "FAIL":
            failures.append(f"{comparison}_mixed_tensor_parity_fail")
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "output_root": str(output_root.resolve()),
        "training_started": False,
        "candidate": {
            "csv_rows": int(csv_audit.get("csv_rows", -1)),
            "csv_id_set_equal": bool(csv_audit.get("id_set_equal")),
            "candidate_rows": len(rows),
            "candidate_rows_sha256": _sha256(rows_path),
            "candidate_rows_inventory_sha256": inventory.get("candidate_rows_jsonl", {}).get("sha256"),
            "completion_ledger_rows": len(ledger),
            "completion_ledger_sha256": _sha256(ledger_path),
            "completion_ledger_inventory_sha256": inventory.get("candidate_rows_jsonl", {}).get("completion_ledger_sha256"),
            "unique_participant_z": len(set(candidate_keys)),
            "duplicate_participant_z_count": len(duplicate_keys),
            "row_count_mismatch_count": len(row_count_mismatches),
            "candidate_source": inventory.get("full_candidate_coverage", {}).get("candidate_source"),
            "old_selected_pool_used_as_source": inventory.get("full_candidate_coverage", {}).get("old_selected_pool_used_as_source"),
        },
        "comparisons": required_audits,
        "registered_fomo": (
            _load(output_root / "fomo45k_registered_paths_v1" / "registered_paths_audit.json")
            if (output_root / "fomo45k_registered_paths_v1" / "registered_paths_audit.json").is_file()
            else {"status": "MISSING"}
        ),
        "registered_materialization": (
            _load(output_root / "fomo45k_registered_paths_v1" / "registered_materialization_audit.json")
            if (output_root / "fomo45k_registered_paths_v1" / "registered_materialization_audit.json").is_file()
            else {"status": "MISSING"}
        ),
        "failures": failures,
    }
    atomic_write_json(output_root / "final_external_audit_v2.json", result, overwrite=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root).resolve()
    result = audit_build(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
