"""Stage CLI for the LEMON T1/T2/high-resolution FLAIR MIP pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:  # pragma: no cover
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.lemon_mip.pipeline import (
    adjudicate_stage,
    clear_explicit_product_dir,
    finalize_stage,
    hdbet_stage,
    processing_stage,
    template_stage,
    validate_stage,
)
from andi_rewrite.utils import load_config


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "preprocess_lemon_mip.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("validate", "template", "hdbet", "pilot", "full", "review", "finalize", "status"),
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--scope", choices=("pilot", "full"))
    parser.add_argument(
        "--overwrite",
        action="append",
        metavar="EXACT_PRODUCT_DIRECTORY",
        help="Delete one inspected product directory below output_root; repeat for multiple exact directories.",
    )
    parser.add_argument("--decision", choices=("PASS", "FAIL", "EXCLUDE"))
    parser.add_argument("--reviewer")
    parser.add_argument("--session-id", action="append", dest="session_ids")
    parser.add_argument("--all-pending", action="store_true")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def _require_python(config: dict) -> None:
    required = config.get("required_python")
    if required is None:
        return
    actual = Path(sys.executable).resolve()
    expected = Path(required).resolve()
    if str(actual).casefold() != str(expected).casefold():
        raise RuntimeError(f"Wrong Python executable: actual={actual}, required={expected}")


def _status(config: dict) -> dict:
    root = Path(config["output_root"]).resolve()
    paths = {
        "validation": root / "metadata" / "validation_report.json",
        "template": root / "templates" / "mni2009c" / "template_metadata.json",
        "pilot_qc": root / "qc" / "pilot_qc_report.json",
        "full_qc": root / "qc" / "full_qc_report.json",
        "lmdb_build": root / "MIP_lmdb" / "build_report.json",
        "lmdb_audit": root / "MIP_lmdb" / "audit.building.json",
        "publication": root / "MIP_lmdb" / "publication.json",
    }
    result = {"output_root": str(root), "products": {}}
    for name, path in paths.items():
        if not path.is_file():
            result["products"][name] = {"exists": False}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result["products"][name] = {
            "exists": True,
            "path": str(path),
            "status": payload.get("status"),
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _require_python(config)
    for requested in args.overwrite or []:
        removed = clear_explicit_product_dir(config, requested)
        print(json.dumps({"removed_explicit_product_directory": str(removed)}, indent=2))

    if args.stage == "validate":
        result = validate_stage(config, config_path=args.config)
    elif args.stage == "template":
        result = template_stage(config)
    elif args.stage == "hdbet":
        if args.scope is None:
            raise ValueError("hdbet stage requires --scope pilot or --scope full.")
        result = hdbet_stage(config, scope=args.scope)
    elif args.stage in {"pilot", "full"}:
        result = processing_stage(config, scope=args.stage)
    elif args.stage == "review":
        if args.scope is None or args.decision is None or not args.reviewer:
            raise ValueError("review requires --scope, --decision, and --reviewer.")
        if not args.all_pending and not args.session_ids:
            raise ValueError("review requires --session-id (repeatable) or --all-pending.")
        result = adjudicate_stage(
            config,
            scope=args.scope,
            decision=args.decision,
            reviewer=args.reviewer,
            session_ids=args.session_ids,
            all_pending=args.all_pending,
            notes=args.notes,
        )
    elif args.stage == "finalize":
        result = finalize_stage(config)
    else:
        result = _status(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
