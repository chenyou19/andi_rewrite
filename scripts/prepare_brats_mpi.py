"""Build and inspect the offline BraTS-to-MPI inference cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_PARENT = REPOSITORY_ROOT.parent
if str(REPOSITORY_PARENT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_PARENT))

from andi_rewrite.data.brats_mpi.pipeline import (  # noqa: E402
    collect_status,
    load_config,
    run_processing,
    validate_dataset,
    write_manifests,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("validate", "pilot", "full", "status", "manifests"),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "preprocess_brats21_mpi.yaml"),
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("mni_affine", "ras_fixed_fov"),
        help="Repeat to run both modes. Defaults to the modes configured in YAML.",
    )
    parser.add_argument(
        "--overwrite-subject",
        action="append",
        help="Replace exactly this subject's cache; repeat for multiple explicit subjects.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    modes = args.mode or list(config.get("modes", ["mni_affine", "ras_fixed_fov"]))

    if args.stage == "validate":
        rows = validate_dataset(config)
        failed = sum(row["status"] != "PASS" for row in rows)
        print(f"validated={len(rows)} excluded={failed}")
        return int(failed > 0)
    if args.stage == "pilot":
        rows = run_processing(
            config,
            modes=modes,
            subject_ids=config.get("pilot_subjects", []),
            label="pilot",
            overwrite_subject=args.overwrite_subject,
        )
    elif args.stage == "full":
        rows = run_processing(
            config,
            modes=modes,
            label="full",
            overwrite_subject=args.overwrite_subject,
        )
    elif args.stage == "status":
        rows = collect_status(config, modes)
    else:
        outputs = write_manifests(config, modes)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    print(" ".join(f"{name.lower()}={count}" for name, count in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
