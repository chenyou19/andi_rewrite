"""Windows entry point pinned to the verified local BraTS21 toolchain."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.fomo45k.brats21 import (  # noqa: E402
    RegistrationSettings,
    preflight_report,
    run_pipeline,
)
from andi_rewrite.scripts.prepare_fomo45k_brats21 import (  # noqa: E402
    _toolchain,
    build_parser,
)


ANTS_ROOT = Path(r"C:\ML\tools\ants-2.6.5\bin")
SYNTHSTRIP = Path(r"C:\ML\tools\synthstrip\mri_synthstrip.cmd")


def _write_provenance(output_root: Path, args) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("write_fomo45k_brats21_provenance.py")),
        "--output-root",
        str(output_root),
        "--ants-registration",
        str(args.ants_registration),
        "--ants-apply-transforms",
        str(args.ants_apply_transforms),
        "--n4",
        str(args.n4_bias_field_correction),
        "--synthstrip",
        str(args.synthstrip),
        "--random-seed",
        str(args.random_seed),
        "--force",
    ]
    subprocess.run(command, check=True)
    atlas_provenance = Path(args.atlas_root) / "atlas_provenance.json"
    if atlas_provenance.is_file():
        shutil.copy2(atlas_provenance, output_root / "atlas_provenance.json")


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        ants_registration=str(ANTS_ROOT / "antsRegistration.exe"),
        ants_apply_transforms=str(ANTS_ROOT / "antsApplyTransforms.exe"),
        n4_bias_field_correction=str(ANTS_ROOT / "N4BiasFieldCorrection.exe"),
        synthstrip=str(SYNTHSTRIP),
    )
    args = parser.parse_args()
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
        _write_provenance(args.output_root.resolve(), args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
