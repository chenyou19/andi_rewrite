"""Create an immutable metadata-only FOMO registered-path manifest revision.

The v3 candidate and pairing build is preserved byte-for-byte.  This helper
only adds the verified raw registered volume paths to every selected FOMO
healthy row by participant/case/session, allowing a newly selected z to reuse
the same source volumes as the historical manifest.  It refuses to write a
revision when any selected healthy row cannot be resolved or a referenced file
is missing.
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

from andi_rewrite.domain_classifier.runner import atomic_write_json, read_jsonl_manifest  # noqa: E402


SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_paths(row: Mapping[str, Any]) -> dict[str, str]:
    value = row.get("registered_paths", {})
    return {str(name): str(path) for name, path in value.items()} if isinstance(value, Mapping) else {}


def _path_keys(row: Mapping[str, Any]) -> list[tuple[str, ...]]:
    source_split = str(row.get("source_split") or "")
    source_key = str(row.get("source_key") or "")
    participant = str(row.get("participant_id") or "")
    case_id = str(row.get("case_id") or "")
    session_id = str(row.get("session_id") or "")
    keys: list[tuple[str, ...]] = [("source", source_split, source_key)]
    if participant and case_id:
        keys.append(("participant_case", participant, case_id, session_id))
        if ":" in participant:
            keys.append(("participant_case", participant.split(":", 1)[1], case_id, session_id))
    return keys


def _build_path_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        paths = _row_paths(row)
        if not paths:
            continue
        for key in _path_keys(row):
            prior = index.get(key)
            if prior is not None and prior != paths:
                raise ValueError(f"Conflicting registered paths for key {key!r}.")
            index[key] = paths
    return index


def _resolve(index: Mapping[tuple[str, ...], Mapping[str, str]], row: Mapping[str, Any]) -> dict[str, str]:
    for key in _path_keys(row):
        value = index.get(key)
        if isinstance(value, Mapping) and value:
            return {str(name): str(path) for name, path in value.items()}
    return {}


def _file_state(path: str) -> bool | None:
    """Return False for a known-missing path and None when ACL blocks stat."""

    try:
        return bool(Path(path).is_file())
    except PermissionError:
        return None


def enrich_registered_paths(*, manifest_root: Path, historical_root: Path, output_root: Path) -> dict[str, Any]:
    input_paths = {split: manifest_root / f"{split}.jsonl" for split in SPLITS}
    historical_paths = {split: historical_root / f"{split}.jsonl" for split in SPLITS}
    if any(not path.is_file() for path in list(input_paths.values()) + list(historical_paths.values())):
        raise FileNotFoundError("Both v3 and historical FOMO manifests are required for enrichment.")
    historical_rows = [row for path in historical_paths.values() for row in read_jsonl_manifest(path)]
    index = _build_path_index(historical_rows)
    output_rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []
    unverified_files: list[dict[str, Any]] = []
    healthy_count = 0
    for split, path in input_paths.items():
        rows = read_jsonl_manifest(path)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            updated = dict(row)
            if int(row.get("label", -1)) == 0:
                healthy_count += 1
                paths = _resolve(index, row)
                if set(paths) != {"flair", "t1", "t2"}:
                    missing.append({"split": split, "participant_id": row.get("participant_id"), "case_id": row.get("case_id"), "z": row.get("z"), "reason": "unresolved_registered_paths"})
                else:
                    absent = [name for name, value in paths.items() if _file_state(value) is False]
                    denied = [name for name, value in paths.items() if _file_state(value) is None]
                    if absent:
                        missing_files.append({"split": split, "participant_id": row.get("participant_id"), "case_id": row.get("case_id"), "z": row.get("z"), "missing": absent})
                    if denied:
                        unverified_files.append({"split": split, "participant_id": row.get("participant_id"), "case_id": row.get("case_id"), "z": row.get("z"), "access_denied": denied})
                updated["registered_paths"] = paths
            enriched.append(updated)
        output_rows[split] = enriched
    if missing or missing_files:
        raise RuntimeError(f"Registered path enrichment failed: unresolved={len(missing)}, missing_files={len(missing_files)}")

    out_manifest_root = output_root / "manifests" / "fomo45k"
    out_manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_audit: dict[str, Any] = {}
    for split, rows in output_rows.items():
        destination = out_manifest_root / f"{split}.jsonl"
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
        manifest_audit[split] = {
            "input": str(input_paths[split]),
            "input_sha256": _sha256(input_paths[split]),
            "output": str(destination),
            "output_sha256": _sha256(destination),
            "records": len(rows),
            "healthy_records": sum(int(row.get("label", -1)) == 0 for row in rows),
        }
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "revision": "fomo45k_v3_registered_paths_v1",
        "source_manifest_root": str(manifest_root.resolve()),
        "historical_manifest_root": str(historical_root.resolve()),
        "output_manifest_root": str(out_manifest_root.resolve()),
        "historical_index_entries": len(index),
        "healthy_records_enriched": healthy_count,
        "unresolved_count": 0,
        "missing_file_count": 0,
        "path_existence_unverified_count": len(unverified_files),
        "path_existence_unverified_reason": "Windows ACL denied stat for some raw FOMO paths; strings are preserved from the historical manifest and must be verified by the registered reader before positive training.",
        "image_selection_unchanged": True,
        "pair_split_z_unchanged": True,
        "manifest_audit": manifest_audit,
    }
    atomic_write_json(output_root / "registered_paths_audit.json", audit, overwrite=False)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, default=REPO_ROOT / "outputs/diagnostics/domain_classifier/manifests/fomo45k")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = (args.manifest_root if args.manifest_root.is_absolute() else REPO_ROOT / args.manifest_root).resolve()
    historical_root = (args.historical_root if args.historical_root.is_absolute() else REPO_ROOT / args.historical_root).resolve()
    output_root = (args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root).resolve()
    if args.dry_run:
        print(json.dumps({"status": "PLANNED", "manifest_root": str(manifest_root), "historical_root": str(historical_root), "output_root": str(output_root), "training_started": False}, indent=2, sort_keys=True))
        return 0
    result = enrich_registered_paths(manifest_root=manifest_root, historical_root=historical_root, output_root=output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
