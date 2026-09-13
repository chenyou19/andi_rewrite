"""Build the FOMO45K SRI24 LMDB used by the three-channel ANDi model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.fomo45k.brats21_lmdb import (  # noqa: E402
    audit_output,
    build_output,
    dataset_status,
    read_source_manifests,
)


DEFAULT_SOURCE = Path(r"C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH")
DEFAULT_OUTPUT = Path(r"C:\ML\data\FOMO45K_SRI24_BraTS21_ANDi\PT007_NIMH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "build", "audit", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        _rows, _slices, result = read_source_manifests(args.source_root)
    elif args.command == "build":
        result = build_output(args.source_root, args.output_root, image_size=128, seed=73)
    elif args.command == "audit":
        result = audit_output(args.output_root)
    else:
        result = dataset_status(args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
