"""Read-only integrity audit for a calibration run and resumable null draws.

The audit does not load image tensors or retrain.  It validates the immutable
manifest/config/code identity, reconstructs each draw's independent pair-label
streams, checks validation and test prediction joins, and writes compact
pair-level label maps so a later resume can be checked against the same draw
indices.  Train labels are reconstructed from the deterministic stream and
source manifest because the training loop intentionally stores predictions for
held-out validation/test only; this limitation is recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import permute_pair_labels, read_jsonl_manifest  # noqa: E402
from scripts.run_domain_classifier_calibration import (  # noqa: E402
    SPLITS,
    _label_stream_seeds,
)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("participant_id", "")),
        str(row.get("pair_id", "")),
        str(row.get("case_id", "")),
        str(row.get("source_key", "")),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_rows(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    index: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    streams = _label_stream_seeds(index)
    expected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        expected[split] = permute_pair_labels(source_rows[split], seed=streams[split])
    return expected, streams


def _pair_label_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        pair = str(row.get("pair_id", ""))
        participant = str(row.get("participant_id", ""))
        if not pair or not participant:
            raise ValueError("Pair and participant identifiers must be non-empty.")
        output.setdefault(pair, {})[participant] = int(row["label"])
    return output


def _check_predictions(
    predictions: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[str]:
    errors: list[str] = []
    if len(predictions) != len(expected):
        errors.append(f"{split}: prediction count {len(predictions)} != expected {len(expected)}")
        return errors
    for index, (actual, target) in enumerate(zip(predictions, expected)):
        expected_join = _row_key(target)
        actual_join = (
            str(actual.get("participant_id", "")),
            str(actual.get("pair_id", "")),
            str(actual.get("case_id", "")),
            str(target.get("source_key", "")),
        )
        if actual_join[:3] != expected_join[:3] or int(actual.get("label", -1)) != int(target["label"]):
            errors.append(f"{split}: row {index} metadata/label mismatch")
            if len(errors) >= 5:
                break
    return errors


def audit_calibration(
    *,
    manifest_root: Path,
    output_root: Path,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    identity_path = output_root / "run_identity.json"
    protocol_path = output_root / "protocol.json"
    if not identity_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Calibration identity/protocol are required for an integrity audit.")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {split: manifest_root / f"{split}.jsonl" for split in SPLITS}
    source_rows = {split: read_jsonl_manifest(paths[split]) for split in SPLITS}
    errors: list[str] = []
    expected_hashes = (protocol.get("manifest_identity") or {}).get("sha256", {})
    for split, path in paths.items():
        if not path.is_file():
            errors.append(f"missing manifest {split}: {path}")
        elif expected_hashes.get(split) != _sha256(path):
            errors.append(f"manifest hash changed: {split}")
    snapshot_manifest = output_root / "code_snapshot" / "snapshot_manifest.json"
    snapshot_files: list[dict[str, Any]] = []
    if not snapshot_manifest.is_file():
        errors.append("missing code snapshot manifest")
    else:
        snapshot_files = json.loads(snapshot_manifest.read_text(encoding="utf-8")).get("files", [])
        for entry in snapshot_files:
            source = Path(entry["source"])
            if not source.is_file() or _sha256(source) != str(entry["sha256"]):
                errors.append(f"live code differs from snapshot: {source}")

    requested = int(protocol.get("requested_replicates", identity.get("requested_replicates", 0)))
    null_root = output_root / "retrained_null"
    completed: list[int] = []
    draw_audits: list[dict[str, Any]] = []
    label_audit_root = output_root / "label_audit"
    if write_artifacts:
        label_audit_root.mkdir(parents=True, exist_ok=True)
    for index in range(requested):
        result_path = null_root / f"permutation_{index:04d}.json"
        prediction_path = null_root / f"permutation_{index:04d}_test_predictions.jsonl"
        if not result_path.exists() and not prediction_path.exists():
            continue
        if not result_path.exists() or not prediction_path.exists():
            errors.append(f"partial draw artifacts: {index}")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_fingerprint") != identity.get("run_fingerprint"):
            errors.append(f"draw {index}: run fingerprint mismatch")
        if int(result.get("permutation_index", -1)) != index:
            errors.append(f"draw {index}: permutation index mismatch")
        if int(result.get("init_seed", -1)) != int(protocol.get("init_seed", 73)):
            errors.append(f"draw {index}: initialization seed changed")
        expected, streams = _expected_rows(source_rows, index=index)
        if result.get("label_stream_seeds") != streams:
            errors.append(f"draw {index}: label stream seed mismatch")
        expected_config = result.get("config", {})
        for field, target in {
            "seed": int(protocol.get("init_seed", 73)),
            "split_seed": 73,
            "max_epochs": 40,
            "patience": 8,
            "early_stopping": True,
            "bootstrap_replicates": 0,
            "swap_replicates": 0,
        }.items():
            if expected_config.get(field) != target:
                errors.append(f"draw {index}: config {field}={expected_config.get(field)!r}, expected {target!r}")
        predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        errors.extend(f"draw {index}: {message}" for message in _check_predictions(predictions, expected["test"], split="test"))
        validation_predictions = result.get("validation_predictions", [])
        errors.extend(f"draw {index}: {message}" for message in _check_predictions(validation_predictions, expected["val"], split="val"))
        pair_maps = {split: _pair_label_map(expected[split]) for split in SPLITS}
        if write_artifacts:
            (label_audit_root / f"permutation_{index:04d}.json").write_text(
                json.dumps({"index": index, "label_stream_seeds": streams, "pair_labels": pair_maps}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        draw_audits.append({
            "index": index,
            "init_seed": int(result.get("init_seed", -1)),
            "label_stream_seeds": streams,
            "validation_prediction_rows": len(validation_predictions),
            "test_prediction_rows": len(predictions),
            "train_labels_reconstructed": True,
            "train_labels_directly_persisted": False,
        })
        completed.append(index)
    status = "PASS" if not errors and len(completed) == requested else "INCOMPLETE" if not errors else "FAIL"
    audit = {
        "status": status,
        "output_root": str(output_root),
        "run_fingerprint": identity.get("run_fingerprint"),
        "requested_replicates": requested,
        "completed_replicates": len(completed),
        "completed_indices": completed,
        "checks": {
            "manifest_hashes": not any("manifest hash" in error for error in errors),
            "code_snapshot_hashes": not any("code differs" in error for error in errors),
            "draw_fingerprint_and_index": not any("fingerprint mismatch" in error or "permutation index" in error for error in errors),
            "independent_label_streams": not any("label stream seed" in error for error in errors),
            "validation_and_test_prediction_joins": not any("metadata/label mismatch" in error or "prediction count" in error for error in errors),
            "fixed_initialization_seed": not any("initialization seed" in error for error in errors),
            "null_statistics_zero_budget": not any("config bootstrap" in error or "config swap" in error for error in errors),
        },
        "train_label_audit": {
            "reconstructed_from_source_and_streams": True,
            "direct_training_rows_persisted": False,
            "limitation": "Training labels are reconstructed deterministically; the training loop stores validation/test predictions, not per-training-slice label rows.",
        },
        "draws": draw_audits,
        "errors": errors,
    }
    if write_artifacts:
        (output_root / "calibration_integrity_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    audit = audit_calibration(
        manifest_root=manifest_root.resolve(),
        output_root=output_root.resolve(),
        write_artifacts=not bool(args.no_write),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] in {"PASS", "INCOMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
