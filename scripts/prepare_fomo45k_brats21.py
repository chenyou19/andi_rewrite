"""Register FOMO45K PT007_NIMH to the BraTS21 SRI24 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.fomo45k.brats21 import (  # noqa: E402
    DEFAULT_ATLAS_ROOT,
    DEFAULT_TOOLCHAIN,
    RegistrationSettings,
    Toolchain,
    preflight_report,
    run_pipeline,
)


DEFAULT_SOURCE = Path(r"C:\ML\data\FOMO45K_healthy_243\PT007_NIMH")
DEFAULT_TSV = Path(r"C:\ML\data\FOMO45K_healthy_T1_T2_FLAIR_243.tsv")
DEFAULT_OUTPUT = Path(r"C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", help="Exact identifier, e.g. sub_10063/ses_1.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/config only.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--random-seed", type=int, default=73)
    parser.add_argument("--ants-registration", default=DEFAULT_TOOLCHAIN.ants_registration)
    parser.add_argument("--ants-apply-transforms", default=DEFAULT_TOOLCHAIN.ants_apply_transforms)
    parser.add_argument(
        "--n4-bias-field-correction", default=DEFAULT_TOOLCHAIN.n4_bias_field_correction
    )
    parser.add_argument("--synthstrip", default=DEFAULT_TOOLCHAIN.synthstrip)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--atlas-t1", type=Path)
    parser.add_argument("--atlas-brain", type=Path)
    parser.add_argument("--atlas-mask", type=Path)
    return parser


def _toolchain(args: argparse.Namespace) -> Toolchain:
    root = args.atlas_root
    return Toolchain(
        ants_registration=args.ants_registration,
        ants_apply_transforms=args.ants_apply_transforms,
        n4_bias_field_correction=args.n4_bias_field_correction,
        synthstrip=args.synthstrip,
        atlas_t1=str(args.atlas_t1 or root / "brats_sri24.nii"),
        atlas_brain=str(args.atlas_brain or root / "brats_sri24_skullstripped.nii"),
        atlas_mask=str(args.atlas_mask or root / "brats_sri24_mask.nii.gz"),
    )


def main() -> None:
    args = build_parser().parse_args()
    toolchain = _toolchain(args)
    if args.preflight_only:
        result = preflight_report(toolchain)
    else:
        result = run_pipeline(
            args.source_root,
            args.metadata_tsv,
            args.output_root,
            toolchain,
            RegistrationSettings(random_seed=args.random_seed),
            case_id=args.case_id,
            workers=args.workers,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            force=args.force,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
