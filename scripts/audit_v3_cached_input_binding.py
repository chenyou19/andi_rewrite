"""Materialize and audit one frozen v3 cohort without fitting a model.

This is an append-only, read-only data audit for the Stage-B controls.  It
uses the same v3 production reader as the fit entry points, checks the frozen
healthy ledger and manifest anchors, and writes a per-canonical-identity
tensor digest ledger.  No checkpoint, cache, manifest, or source artifact is
modified; only the requested audit output directory is created.

The command is intentionally fail-closed on a non-ANDi interpreter, a
pre-existing output directory, a manifest or ledger mismatch, a bad tensor,
or a thread-policy mismatch.  It is useful for Mixed and standalone cohorts;
the output records the comparison so that a Mixed audit cannot be mistaken
for the standalone source audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# Set process-level BLAS limits before importing NumPy/Torch.  The runner and
# v3 runtime also verify the PyTorch pools after import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

ANDI_PYTHON = Path(r"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe").resolve()
SPLITS = ("train", "val", "test")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    temporary.replace(path)
    return _sha256_file(path)


def _manifest_summary(cached: Any) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    label_counts: Counter[str] = Counter()
    participant_counts: dict[str, int] = {}
    row_keys: set[tuple[str, str, str, str]] = set()
    for split in SPLITS:
        rows = list(cached.records_by_split[split])
        split_counts[split] = len(rows)
        participants: set[str] = set()
        for row in rows:
            participant = str(row.get("participant_id", "")).strip()
            if not participant:
                raise ValueError(f"{split} contains an empty participant_id")
            participants.add(participant)
            label = str(row.get("label", ""))
            label_counts[label] += 1
            identity = tuple(str(value) for value in row.get("source_identity", ()))
            key = (
                str(row.get("source_dataset", "")),
                str(row.get("source_split", "")),
                str(row.get("source_key", "")),
                str(row.get("z", "")),
            )
            if len(identity) >= 4:
                key = tuple(identity[:4])  # type: ignore[assignment]
            if key in row_keys:
                raise ValueError(f"duplicate source identity in materialized rows: {key!r}")
            row_keys.add(key)
        participant_counts[split] = len(participants)
    return {
        "rows_by_split": split_counts,
        "total_rows": sum(split_counts.values()),
        "participants_by_split": participant_counts,
        "label_counts": dict(sorted(label_counts.items())),
        "unique_row_source_identities": len(row_keys),
    }


def _digest_rows(cached: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity, digest in sorted(cached.tensor_digests_by_identity.items(), key=lambda item: repr(item[0])):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid tensor digest for identity {identity!r}")
        rows.append({"identity": list(identity), "tensor_sha256": digest})
    return rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if Path(sys.executable).resolve() != ANDI_PYTHON:
        raise RuntimeError(f"This audit requires the exact ANDi interpreter: {ANDI_PYTHON}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    execution = {
        "schema_version": 1,
        "status": "RUNNING",
        "audit_type": "v3_actual_tensor_binding_preflight",
        "comparison": args.comparison,
        "training_started": False,
        "no_training": True,
        "executable": str(Path(sys.executable).resolve()),
        "command": " ".join(str(value) for value in args.command_tokens),
        "manifest_root": str(args.manifest_root.resolve()),
        "build_root": str(args.build_root.resolve()),
        "config_path": str(args.config_path.resolve()),
        "output_root": str(output_root),
        "started_at_utc": started_at,
        "environment_requested": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    _atomic_json(output_root / "execution.json", execution)

    try:
        from andi_rewrite.domain_classifier.v3_runtime import (
            DEFAULT_HEALTHY_LEDGER_SHA256,
            set_single_thread_runtime,
            validate_and_materialize_v3_inputs,
        )

        thread_policy = set_single_thread_runtime(strict=True)
        expected_ledger_sha = args.expected_ledger_sha256 or DEFAULT_HEALTHY_LEDGER_SHA256
        cached = validate_and_materialize_v3_inputs(
            args.manifest_root.resolve(),
            comparison=args.comparison,
            build_root=args.build_root.resolve(),
            ledger_path=args.ledger_path.resolve() if args.ledger_path else None,
            ledger_summary_path=args.ledger_summary_path.resolve() if args.ledger_summary_path else None,
            expected_ledger_sha256=expected_ledger_sha,
            config_path=args.config_path.resolve(),
        )
        audit = dict(cached.input_binding_audit)
        digest_rows = _digest_rows(cached)
        digest_path = output_root / "input_tensor_digest_ledger.jsonl"
        digest_sha = _write_jsonl(digest_path, digest_rows)
        manifest_summary = _manifest_summary(cached)
        _atomic_json(output_root / "input_binding_audit.json", audit)
        _atomic_json(
            output_root / "input_tensor_digest_ledger_summary.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "comparison": args.comparison,
                "rows": len(digest_rows),
                "path": str(digest_path),
                "sha256": digest_sha,
                "source": "V3CachedInputs.tensor_digests_by_identity",
            },
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        result = {
            "schema_version": 1,
            "status": "PASS",
            "audit_type": "v3_actual_tensor_binding_preflight",
            "comparison": args.comparison,
            "training_started": False,
            "no_training": True,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "elapsed_seconds": time.perf_counter() - started,
            "thread_policy": thread_policy,
            "manifest_summary": manifest_summary,
            "input_binding_audit": str((output_root / "input_binding_audit.json").resolve()),
            "input_binding_status": audit.get("status"),
            "tensor_digest_ledger": {
                "path": str(digest_path.resolve()),
                "sha256": digest_sha,
                "rows": len(digest_rows),
                "unique_canonical_tensors": len(cached.tensor_digests_by_identity),
            },
            "manifest_fingerprints": cached.manifest_fingerprints,
            "source_freeze": cached.source_freeze,
            "healthy_ledger": {
                "path": str(cached.ledger.path.resolve()),
                "summary_path": str(cached.ledger.summary_path.resolve()),
                "sha256": cached.ledger.ledger_sha256,
                "rows": cached.ledger.row_count,
            },
        }
        _atomic_json(output_root / "summary.json", result)
        execution.update(
            {
                "status": "PASS",
                "completed_at_utc": completed_at,
                "elapsed_seconds": result["elapsed_seconds"],
                "thread_policy_actual": thread_policy,
                "input_binding_status": audit.get("status"),
                "tensor_digest_ledger_sha256": digest_sha,
            }
        )
        _atomic_json(output_root / "execution.json", execution)
        return result
    except Exception as exc:
        execution.update(
            {
                "status": "FAIL_CLOSED",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _atomic_json(output_root / "execution.json", execution)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=("fomo45k", "mpi", "oasis3", "mixed"), required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--ledger-summary-path", type=Path, default=None)
    parser.add_argument("--expected-ledger-sha256", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.manifest_root = args.manifest_root.resolve()
    args.build_root = args.build_root.resolve()
    args.config_path = args.config_path.resolve()
    args.output_root = args.output_root.resolve()
    if args.ledger_path is not None:
        args.ledger_path = args.ledger_path.resolve()
    if args.ledger_summary_path is not None:
        args.ledger_summary_path = args.ledger_summary_path.resolve()
    args.command_tokens = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]]
    result = _run(args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
