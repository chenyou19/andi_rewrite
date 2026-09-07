"""Validate and build the FOMO45K PT007_NIMH LPS training LMDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.fomo45k import (  # noqa: E402
    audit_output,
    build_output,
    dataset_status,
    validate_dataset,
)


DEFAULT_SOURCE = Path(r"C:\ML\data\FOMO45K_healthy_243\PT007_NIMH")
DEFAULT_TSV = Path(r"C:\ML\data\FOMO45K_healthy_T1_T2_FLAIR_243.tsv")
DEFAULT_OUTPUT = Path(r"C:\ML\data\FOMO45K_ANDi\PT007_NIMH")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-orientation",
        default="LPS",
        choices=["LPS"],
        help="Target voxel orientation. This pipeline is locked to BraTS21-compatible LPS.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate every source NIfTI and split count.")
    _common(validate)
    validate.add_argument("--report", type=Path, help="Optional JSON report path.")
    build = subparsers.add_parser("build", help="Validate, build, audit, and atomically publish LMDBs.")
    _common(build)
    audit = subparsers.add_parser("audit", help="Read and verify every published LMDB entry.")
    _common(audit)
    audit.add_argument("--no-write-report", action="store_true")
    status = subparsers.add_parser("status", help="Show publication, audit, and staging state.")
    _common(status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = tuple(args.target_orientation)
    if args.command == "validate":
        _records, result = validate_dataset(args.source_root, args.metadata_tsv)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
    elif args.command == "build":
        result = build_output(
            args.source_root,
            args.metadata_tsv,
            args.output_root,
            target_axcodes=target,
        )
    elif args.command == "audit":
        result = audit_output(args.output_root, write_report=not args.no_write_report)
    else:
        result = dataset_status(args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
