"""Shifts/MSSEG 資料整理、registration 與 normalization 的 CLI wrapper。"""

from __future__ import annotations

import argparse

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.prepare import ShiftsDataPreparer


def parse_bool(value) -> bool:
    """同時接受 flag-style 與原版 True/False 字串形式。"""

    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Shifts/MSSEG-style datasets for ANDi.")
    parser.add_argument("-d", "--data_set", required=True, help="Dataset folder to prepare.")
    parser.add_argument("-n", "--norm", nargs="?", const=True, default=False, type=parse_bool, help="Run histogram matching.")
    parser.add_argument("-i", "--input_volume", help="Source BraTS patient folder for histogram matching.")
    parser.add_argument("-r", "--register", nargs="?", const=True, default=False, type=parse_bool, help="Register patient volumes to a template.")
    parser.add_argument("-t", "--template", help="Template path, e.g. SRI T1_brain.nii.")
    parser.add_argument("--output-dir", default=None, help="Patient folder output for organization mode.")
    args = parser.parse_args()

    preparer = ShiftsDataPreparer(args.data_set)
    if args.register:
        if not args.template:
            raise ValueError("--template is required with --register.")
        preparer.register(args.template)
    elif args.norm:
        if not args.input_volume:
            raise ValueError("--input_volume is required with --norm.")
        preparer.histogram_matching(args.input_volume)
    else:
        output = preparer.prepare_patient_folders(args.output_dir)
        print(f"Prepared patient folders in {output}")


if __name__ == "__main__":
    main()
