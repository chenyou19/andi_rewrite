"""Run one eval config over every checkpoint in a directory on a 50-volume CSV."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import torch

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
        default=None,
        help="CSV containing the evaluation volumes; defaults to data.path_to_csv in base config.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/metrics/empirical_spectrum233_lmdb_full_gaussian_20260609_50",
        help="Root directory for per-checkpoint metric outputs.",
    )
    parser.add_argument("--pattern", default="epoch_*.pt", help="Checkpoint filename glob.")
    parser.add_argument("--only", nargs="*", help="Optional checkpoint stems or filenames to run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip epochs whose ANDi.csv already exists.")
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=50,
        help="Require this many unique CSV subjects and completed-report subjects.",
    )
    parser.add_argument("--progress", action="store_true", help="Show per-volume and per-timestep progress bars.")
    parser.add_argument(
        "--label-completed-epoch",
        action="store_true",
        help="Name each output directory from payload epoch + 1 (for example completed_epoch_20).",
    )
    return parser.parse_args()


def checkpoint_label(path: Path, completed_epoch: bool = False) -> str:
    if not completed_epoch:
        return path.stem
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "epoch" not in payload:
        raise ValueError(f"Checkpoint does not contain a stored epoch: {path}")
    return f"completed_epoch_{int(payload['epoch']) + 1}"


_EPOCH_STEM = re.compile(r"^epoch_(\d+)$")


def checkpoint_epoch(path: Path) -> int:
    match = _EPOCH_STEM.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Checkpoint is not an epoch snapshot: {path}")
    return int(match.group(1))


def selected_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = [path for path in checkpoint_dir.glob(args.pattern) if path.is_file()]
    checkpoints = [path for path in checkpoints if _EPOCH_STEM.fullmatch(path.stem)]
    if args.only:
        wanted = {Path(item).stem for item in args.only}
        checkpoints = [path for path in checkpoints if path.stem in wanted]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir / args.pattern}")

    by_epoch: dict[int, Path] = {}
    duplicates: list[tuple[int, Path, Path]] = []
    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        previous = by_epoch.get(epoch)
        if previous is not None:
            duplicates.append((epoch, previous, checkpoint))
        else:
            by_epoch[epoch] = checkpoint
    if duplicates:
        details = "; ".join(
            f"epoch {epoch}: {first} and {second}" for epoch, first, second in duplicates
        )
        raise ValueError(f"Duplicate epoch checkpoints detected: {details}")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def validate_subject_csv(csv_path: str | Path, expected_subjects: int) -> list[str]:
    if expected_subjects <= 0:
        raise ValueError("--expected-subjects must be positive.")
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0]:
        raise ValueError(f"Evaluation CSV has no header: {path}")
    subject_ids = [row[0].strip() for row in rows[1:] if row]
    if len(subject_ids) != expected_subjects:
        raise ValueError(
            f"Evaluation CSV must contain exactly {expected_subjects} subjects; "
            f"found {len(subject_ids)} in {path}."
        )
    if any(not subject_id for subject_id in subject_ids):
        raise ValueError(f"Evaluation CSV contains a blank subject id: {path}")
    duplicates = sorted({item for item in subject_ids if subject_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Evaluation CSV contains duplicate subject ids: {duplicates}")
    return subject_ids


def validate_base_config(base_config: dict, args: argparse.Namespace) -> None:
    data = base_config.get("data", {})
    model = base_config.get("model", {})
    modalities = list(data.get("modalities", []))
    channels = int(data.get("channels", len(modalities)))
    in_channels = int(model.get("in_channels", model.get("channels", channels)))
    out_channels = int(model.get("out_channels", model.get("channels", channels)))
    if not modalities:
        raise ValueError("data.modalities must be a non-empty list.")
    if not (len(modalities) == channels == in_channels == out_channels):
        raise ValueError(
            "Configured channel mismatch: "
            f"modalities={len(modalities)}, data.channels={channels}, "
            f"model.in_channels={in_channels}, model.out_channels={out_channels}."
        )
    dataset_path = Path(str(data.get("dataset_path", "")))
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    csv_path = args.csv or data.get("path_to_csv")
    if not csv_path:
        raise ValueError("Provide --csv or configure data.path_to_csv.")
    validate_subject_csv(csv_path, args.expected_subjects)

    schedule = base_config.get("noise", {}).get("schedule", {})
    sampler = schedule.get("sampler", schedule)
    if str(sampler.get("type", "")).lower() == "empirical_spectrum":
        stats_path = Path(str(sampler.get("stats_path", "")))
        if not stats_path.is_file():
            raise FileNotFoundError(f"Empirical-spectrum statistics do not exist: {stats_path}")


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def completed_output(config: dict, checkpoint: Path, expected_subjects: int) -> bool:
    output_dir = Path(config["metrics"]["output_csv"]).parent
    required = [
        output_dir / "ANDi.csv",
        output_dir / "ANDi_mf.csv",
        output_dir / "inference_metrics_summary.csv",
        output_dir / "inference_report.json",
        output_dir / "inference_report.md",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        report = json.loads((output_dir / "inference_report.json").read_text(encoding="utf-8"))
        reported_checkpoint = report["inference_settings"]["checkpoint_path"]
        subjects = int(report["evaluation_result"]["subjects"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return _resolved_path(reported_checkpoint) == _resolved_path(checkpoint) and subjects == expected_subjects


def config_for_checkpoint(base_config: dict, checkpoint: Path, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(base_config)
    label = checkpoint_label(checkpoint, completed_epoch=args.label_completed_epoch)
    output_dir = Path(args.output_root) / label

    config.setdefault("experiment", {})["name"] = f"{config.get('experiment', {}).get('name', 'eval')}_{label}_50"
    data_config = config.setdefault("data", {})
    csv_path = args.csv or data_config.get("path_to_csv")
    if not csv_path:
        raise ValueError("Provide --csv or configure data.path_to_csv.")
    data_config["path_to_csv"] = str(Path(csv_path))
    config.setdefault("model", {})["checkpoint"] = str(checkpoint)
    config.setdefault("evaluation", {})["progress"] = bool(args.progress)
    metrics = config.setdefault("metrics", {})
    metrics["output_csv"] = str(output_dir / "ANDi.csv")
    metrics["output_mf_csv"] = str(output_dir / "ANDi_mf.csv")
    return config


def run_one(config: dict, checkpoint: Path, args: argparse.Namespace) -> dict:
    output_csv = Path(config["metrics"]["output_csv"])
    if args.skip_existing and completed_output(config, checkpoint, args.expected_subjects):
        print(f"Skipping {checkpoint.name}; complete output already exists: {output_csv.parent}")
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
    validate_base_config(base_config, args)
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
