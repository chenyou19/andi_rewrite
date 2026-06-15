"""Run one eval config over every checkpoint in a directory on a 50-volume CSV."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data import build_dataloader
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.scripts.eval import build_detector_from_config
from andi_rewrite.utils import load_config, print_config
from andi_rewrite.utils.reporting import save_inference_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every epoch_*.pt checkpoint on 50 volumes.")
    parser.add_argument(
        "--base-config",
        default="configs/eval_full_gaussian_from_empirical_spectrum233_lmdb_20260609.yaml",
        help="Eval config used as the template for model/data/metric settings.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints/empirical_spectrum233_lmdb_full_gaussian_20260609",
        help="Directory containing epoch_*.pt files.",
    )
    parser.add_argument(
        "--csv",
        default="splits/BraTS21/scans_test_50.csv",
        help="CSV containing the 50 evaluation volumes.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/metrics/empirical_spectrum233_lmdb_full_gaussian_20260609_50",
        help="Root directory for per-checkpoint metric outputs.",
    )
    parser.add_argument("--pattern", default="epoch_*.pt", help="Checkpoint filename glob.")
    parser.add_argument("--only", nargs="*", help="Optional checkpoint stems or filenames to run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip epochs whose ANDi.csv already exists.")
    parser.add_argument("--progress", action="store_true", help="Show per-volume and per-timestep progress bars.")
    return parser.parse_args()


def checkpoint_label(path: Path) -> str:
    return path.stem


def selected_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.pattern))
    if args.only:
        wanted = {Path(item).stem for item in args.only}
        checkpoints = [path for path in checkpoints if path.stem in wanted]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir / args.pattern}")
    return checkpoints


def config_for_checkpoint(base_config: dict, checkpoint: Path, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(base_config)
    label = checkpoint_label(checkpoint)
    output_dir = Path(args.output_root) / label

    config.setdefault("experiment", {})["name"] = f"{config.get('experiment', {}).get('name', 'eval')}_{label}_50"
    config.setdefault("data", {})["path_to_csv"] = str(Path(args.csv))
    config.setdefault("model", {})["checkpoint"] = str(checkpoint)
    config.setdefault("evaluation", {})["progress"] = bool(args.progress)
    metrics = config.setdefault("metrics", {})
    metrics["output_csv"] = str(output_dir / "ANDi.csv")
    metrics["output_mf_csv"] = str(output_dir / "ANDi_mf.csv")
    return config


def run_one(config: dict, checkpoint: Path, args: argparse.Namespace) -> dict:
    output_csv = Path(config["metrics"]["output_csv"])
    if args.skip_existing and output_csv.is_file():
        print(f"Skipping {checkpoint.name}; output already exists: {output_csv}")
        return {"checkpoint": str(checkpoint), "skipped": True, "output": str(output_csv)}

    print(f"Running evaluation for {checkpoint.name}")
    detector, accelerator = build_detector_from_config(config)
    dataloader = build_dataloader(config.get("data", {}))
    evaluator = VolumeEvaluator(
        detector=detector,
        config={**config.get("data", {}), **config.get("metrics", {}), **config.get("evaluation", {})},
        accelerator=accelerator,
    )
    dataloader = evaluator.prepare(dataloader)

    start = datetime.now().astimezone()
    result = evaluator.evaluate(dataloader)
    end = datetime.now().astimezone()
    if result and evaluator.is_main_process:
        save_inference_report(
            config=config,
            evaluator=evaluator,
            result=result,
            dataloader=dataloader,
            start_time=start,
            end_time=end,
            config_path=args.base_config,
            cli_args={
                "base_config": args.base_config,
                "checkpoint": str(checkpoint),
                "csv": args.csv,
                "output_root": args.output_root,
            },
        )
    return {"checkpoint": str(checkpoint), **result}


def main() -> None:
    args = parse_args()
    base_config = load_config(args.base_config)
    base_config.pop("_config_path", None)
    checkpoints = selected_checkpoints(args)
    print("Batch evaluation checkpoints:")
    print_config({"count": len(checkpoints), "checkpoints": [str(path) for path in checkpoints]})

    results = []
    for checkpoint in checkpoints:
        config = config_for_checkpoint(base_config, checkpoint, args)
        result = run_one(config, checkpoint, args)
        results.append(result)
        print("Checkpoint result:")
        print_config(result)

    print("Batch evaluation complete:")
    print_config({"count": len(results), "results": results})


if __name__ == "__main__":
    main()
