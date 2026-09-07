"""Write tool versions, checksums, and exact registration templates for a run."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nibabel
import numpy
import scipy
import surfa
import torch

from andi_rewrite.data.fomo45k.brats21 import (
    RegistrationSettings,
    build_apply_command,
    build_n4_command,
    build_registration_command,
    build_synthstrip_command,
)


DEFAULT_ANTS_ROOT = Path(r"C:\ML\tools\ants-2.6.5\bin")
DEFAULT_SYNTHSTRIP_ROOT = Path(r"C:\ML\tools\synthstrip")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(executable: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "path": str(executable.resolve()),
        "sha256": _sha256(executable),
        "version_output": output,
        "version_exit_code": completed.returncode,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ants-registration", type=Path, default=DEFAULT_ANTS_ROOT / "antsRegistration.exe")
    parser.add_argument(
        "--ants-apply-transforms", type=Path, default=DEFAULT_ANTS_ROOT / "antsApplyTransforms.exe"
    )
    parser.add_argument("--n4", type=Path, default=DEFAULT_ANTS_ROOT / "N4BiasFieldCorrection.exe")
    parser.add_argument(
        "--synthstrip", type=Path, default=DEFAULT_SYNTHSTRIP_ROOT / "mri_synthstrip.cmd"
    )
    parser.add_argument("--random-seed", type=int, default=73)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    destination = output_root / "tool_versions_and_parameters.json"
    if destination.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force explicitly")

    synthstrip_root = args.synthstrip.resolve().parent
    synthstrip_files = {}
    for name in (
        "mri_synthstrip.py",
        "mri_synthstrip_windows_entry_v2.py",
        "mri_synthstrip.cmd",
        "models/synthstrip.1.pt",
    ):
        path = synthstrip_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        synthstrip_files[name] = {"path": str(path), "sha256": _sha256(path)}

    settings = RegistrationSettings(random_seed=args.random_seed)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "nibabel": nibabel.__version__,
            "surfa": surfa.__version__,
        },
        "tools": {
            "antsRegistration": _version(args.ants_registration.resolve()),
            "antsApplyTransforms": _version(args.ants_apply_transforms.resolve()),
            "N4BiasFieldCorrection": _version(args.n4.resolve()),
            "SynthStrip": {
                "launcher": str(args.synthstrip.resolve()),
                "official_script_commit": "cf4bccf24875a47245b4df9fc9372d0f1c3d784f",
                "official_script_git_blob": "d89d19d878295c3b6bf7b957f839e3324bd23f6d",
                "model_sha256": "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
                "windows_compatibility": (
                    "surfa 0.6.3 Win64 Cython target_shape is np.intp; wrapper reproduces the "
                    "official interpolate wrapper with only the internal shape cast changed to np.intp"
                ),
                "files": synthstrip_files,
            },
        },
        "settings": dataclasses.asdict(settings),
        "command_templates": {
            "synthstrip": build_synthstrip_command(
                str(args.synthstrip.resolve()), "{moving}", "{brain_output}", "{mask_output}"
            ),
            "n4": build_n4_command(str(args.n4.resolve()), "{moving}", "{mask}", "{output}"),
            "t1_to_sri24_rigid_affine": build_registration_command(
                str(args.ants_registration.resolve()),
                "{atlas_brain}",
                "{t1_n4_brain}",
                "{atlas_mask}",
                "{t1_native_mask}",
                "{output_prefix}",
                affine=True,
                random_seed=args.random_seed,
            ),
            "modality_to_t1_rigid": build_registration_command(
                str(args.ants_registration.resolve()),
                "{t1_n4_brain}",
                "{modality_n4_brain}",
                "{t1_native_mask}",
                "{modality_native_mask}",
                "{output_prefix}",
                affine=False,
                random_seed=args.random_seed,
            ),
            "apply_same_grid": build_apply_command(
                str(args.ants_apply_transforms.resolve()),
                "{modality_original}",
                "{atlas_t1}",
                "{output}",
                ["{t1_to_sri24_affine}"],
            ),
            "apply_different_grid_composed_once": build_apply_command(
                str(args.ants_apply_transforms.resolve()),
                "{modality_original}",
                "{atlas_t1}",
                "{output}",
                ["{t1_to_sri24_affine}", "{modality_to_t1_rigid}"],
            ),
            "apply_mask": build_apply_command(
                str(args.ants_apply_transforms.resolve()),
                "{t1_native_mask}",
                "{atlas_t1}",
                "{output}",
                ["{t1_to_sri24_affine}"],
                nearest=True,
            ),
        },
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
