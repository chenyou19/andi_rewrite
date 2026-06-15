"""Generate and run 5-fold ANDi experiments from one command."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import yaml

from andi_rewrite.data.preprocess import split_healthy_kfold_to_lmdb
from andi_rewrite.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare combined train/test 5-fold LMDBs, generate fold configs, "
            "and run each fold sequentially."
        )
    )
    parser.add_argument("--dataset", default="C:/ML/data/BraTS_2021", help="BraTS dataset root.")
    parser.add_argument(
        "--train-csv",
        default="C:/ML/ANDi/splits/BraTS21/scans_train.csv",
        help="Original training subject CSV.",
    )
    parser.add_argument(
        "--test-csv",
        default="C:/ML/ANDi/splits/BraTS21/scans_test.csv",
        help="Original test subject CSV.",
    )
    parser.add_argument(
        "--split-root",
        default="C:/ML/ANDi/splits/BraTS21/5fold",
        help="Output directory for per-fold scans_train.csv/scans_test.csv.",
    )
    parser.add_argument(
        "--lmdb-root",
        default="C:/ML/data/BraTS_2021_healthy_lmdb_5fold",
        help="Output directory for per-fold training LMDBs.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to run.")
    parser.add_argument(
        "--combined-test-size",
        type=int,
        default=251,
        help="Test subject count per fold after combining train/test CSVs.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic fold splits.")
    parser.add_argument(
        "--base-train-config",
        default="configs/train_pyramid20_lmdb.yaml",
        help="Training config template.",
    )
    parser.add_argument(
        "--base-eval-config",
        default="configs/eval_gaussian50_pyramid20.yaml",
        help="Evaluation config template.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/5fold",
        help="Directory where generated fold train/eval configs are written.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional run name prefix. Defaults to base training run_name plus '_5fold'.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints/5fold",
        help="Fold checkpoint root.",
    )
    parser.add_argument(
        "--sample-dir",
        default="outputs/samples/5fold",
        help="Fold sample image root.",
    )
    parser.add_argument(
        "--metric-dir",
        default="outputs/metrics/5fold",
        help="Fold metric output root.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch scripts/train.py.",
    )
    parser.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help="Do not generate split CSVs or LMDBs; only generate configs and run training.",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Overwrite existing per-fold LMDB directories when preparing data.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable preprocessing progress bars.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["healthy", "z_balanced"],
        default="healthy",
        help="LMDB slice sampling mode.",
    )
    parser.add_argument("--z-balanced", action="store_true", help="Alias for --sampling-mode z_balanced.")
    parser.add_argument("--per-z-count", type=int, default=447, help="Per-z count for z-balanced mode.")
    parser.add_argument("--balance-seed", type=int, default=42, help="Base seed for z-balanced sampling.")
    parser.add_argument(
        "--only-folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional fold numbers to run, e.g. --only-folds 0 2 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data/configs and print train commands without launching training.",
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Only generate fold train/eval configs; do not prepare data or launch training.",
    )
    return parser.parse_args()


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def get_run_prefix(base_train_config: dict[str, Any], explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix
    training = base_train_config.get("training", {})
    experiment = base_train_config.get("experiment", {})
    base_name = str(training.get("run_name") or experiment.get("name") or "andi_5fold")
    return f"{base_name}_5fold"


def get_fold_numbers(args: argparse.Namespace) -> list[int]:
    return args.only_folds or list(range(args.folds))


def validate_existing_fold_inputs(args: argparse.Namespace) -> None:
    missing: list[str] = []
    for fold in get_fold_numbers(args):
        fold_name = f"fold_{fold}"
        split_csv = Path(args.split_root) / fold_name / "scans_test.csv"
        train_lmdb = Path(args.lmdb_root) / fold_name
        if not split_csv.is_file():
            missing.append(f"{fold_name}: missing split CSV {split_csv}")
        if not train_lmdb.is_dir() or not any(train_lmdb.iterdir()):
            missing.append(f"{fold_name}: missing or empty LMDB directory {train_lmdb}")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Existing fold inputs are incomplete. Because --skip-prepare-data was used, "
            "run_5fold.py will not create missing split CSVs or LMDBs.\n"
            f"  - {details}\n"
            "Use the split root that was used when preparing the LMDBs, or rerun without "
            "--skip-prepare-data and add --overwrite-data if the fold LMDBs should be rebuilt."
        )


def final_checkpoint_path(train_config: dict[str, Any], run_name: str) -> str:
    training = train_config.get("training", {})
    checkpoint = training.get("checkpoint", {})
    epochs = int(training.get("epochs", 1))
    epoch = max(epochs - 1, 0)
    checkpoint_dir = Path(checkpoint.get("dir", "outputs/checkpoints"))
    return str(checkpoint_dir / run_name / f"epoch_{epoch:04d}.pt")


def iter_empirical_spectrum_samplers(node: Any) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if str(node.get("type", "")).lower() == "empirical_spectrum":
            matches.append(node)
        for value in node.values():
            matches.extend(iter_empirical_spectrum_samplers(value))
    elif isinstance(node, list):
        for item in node:
            matches.extend(iter_empirical_spectrum_samplers(item))
    return matches


def compute_fold_empirical_spectrum(
    args: argparse.Namespace,
    fold: int,
    fold_lmdb_path: Path,
    out_path: Path,
) -> None:
    command = [
        args.python,
        "scripts/compute_lmdb_spectrum.py",
        "--lmdb-path",
        str(fold_lmdb_path),
        "--out",
        str(out_path),
        "--overwrite",
    ]
    if args.no_progress:
        command.append("--no-progress")
    print(f"Fold {fold} empirical spectrum command:")
    print(" ".join(command))
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


def apply_fold_empirical_spectrum(
    config: dict[str, Any],
    stats_path: Path,
) -> bool:
    samplers = iter_empirical_spectrum_samplers(config.get("noise", {}))
    for sampler in samplers:
        sampler["stats_path"] = str(stats_path)
    return bool(samplers)


def remove_yen_threshold_from_metric_postprocess(config: dict[str, Any]) -> None:
    postprocess = config.get("metrics", {}).get("postprocess", {})
    if not isinstance(postprocess, dict):
        return
    for postprocess_config in postprocess.values():
        if not isinstance(postprocess_config, dict):
            continue
        pipeline = postprocess_config.get("pipeline")
        if isinstance(pipeline, dict):
            pipeline_items = [pipeline]
        elif isinstance(pipeline, list):
            pipeline_items = pipeline
        else:
            continue
        postprocess_config["pipeline"] = [
            step
            for step in pipeline_items
            if not (isinstance(step, dict) and str(step.get("type", "")).lower() == "yen_threshold")
        ]


def build_fold_configs(args: argparse.Namespace) -> list[tuple[int, Path, Path]]:
    base_train = load_config(args.base_train_config)
    base_eval = load_config(args.base_eval_config)
    base_train.pop("_config_path", None)
    base_eval.pop("_config_path", None)

    config_dir = Path(args.config_dir)
    run_prefix = get_run_prefix(base_train, args.run_prefix)
    fold_numbers = get_fold_numbers(args)
    generated = []

    for fold in fold_numbers:
        fold_name = f"fold_{fold}"
        run_name = f"{run_prefix}_{fold_name}"
        fold_lmdb_path = Path(args.lmdb_root) / fold_name
        train_config = copy.deepcopy(base_train)
        eval_config = copy.deepcopy(base_eval)

        train_config.setdefault("experiment", {})["name"] = run_name
        train_config.setdefault("data", {})["path"] = str(fold_lmdb_path)
        training = train_config.setdefault("training", {})
        training["run_name"] = run_name
        training.setdefault("checkpoint", {})["dir"] = args.checkpoint_dir
        training["checkpoint"]["start_epoch"] = max(int(training.get("epochs", 1)) - 1, 0)
        training["checkpoint"]["save_every_epochs"] = 1
        training.setdefault("samples", {})["output_dir"] = args.sample_dir
        training["samples"]["start_epoch"] = max(int(training.get("epochs", 1)) - 1, 0)
        training["samples"]["every_epochs"] = 1

        train_config_path = config_dir / f"train_{fold_name}.yaml"
        eval_config_path = config_dir / f"eval_{fold_name}.yaml"
        training["eval_after_fit"] = {"enabled": True, "config": str(eval_config_path)}

        eval_config.setdefault("experiment", {})["name"] = f"{run_name}_eval"
        eval_config.setdefault("data", {})["path_to_csv"] = str(Path(args.split_root) / fold_name / "scans_test.csv")
        eval_config["data"]["dataset_path"] = args.dataset
        eval_config.setdefault("model", {})["checkpoint"] = final_checkpoint_path(train_config, run_name)
        metrics = eval_config.setdefault("metrics", {})
        fold_metric_dir = Path(args.metric_dir) / fold_name
        metrics["output_csv"] = str(fold_metric_dir / "ANDi.csv")
        metrics["output_mf_csv"] = str(fold_metric_dir / "ANDi_mf.csv")
        remove_yen_threshold_from_metric_postprocess(eval_config)

        train_uses_empirical = bool(iter_empirical_spectrum_samplers(train_config.get("noise", {})))
        eval_uses_empirical = bool(iter_empirical_spectrum_samplers(eval_config.get("noise", {})))
        if train_uses_empirical or eval_uses_empirical:
            stats_path = Path("outputs") / "spectrum" / "5fold" / f"{run_name}.npz"
            compute_fold_empirical_spectrum(args, fold, fold_lmdb_path, stats_path)
            apply_fold_empirical_spectrum(train_config, stats_path)
            apply_fold_empirical_spectrum(eval_config, stats_path)

        write_yaml(train_config_path, train_config)
        write_yaml(eval_config_path, eval_config)
        generated.append((fold, train_config_path, eval_config_path))

    return generated


def prepare_data(args: argparse.Namespace) -> None:
    sampling_mode = "z_balanced" if args.z_balanced else args.sampling_mode
    summary = split_healthy_kfold_to_lmdb(
        dataset_path=args.dataset,
        input_csv=args.train_csv,
        output_root=args.split_root,
        lmdb_root=args.lmdb_root,
        test_csv=args.test_csv,
        combine_train_test=True,
        combined_test_size=args.combined_test_size,
        folds=args.folds,
        overwrite=args.overwrite_data,
        progress=not args.no_progress,
        sampling_mode=sampling_mode,
        per_z_count=args.per_z_count,
        split_seed=args.split_seed,
        balance_seed=args.balance_seed,
    )
    print("Prepared 5-fold data:")
    print(yaml.safe_dump(summary.as_dict(), sort_keys=False, allow_unicode=True))


def run_training(args: argparse.Namespace, generated_configs: list[tuple[int, Path, Path]]) -> None:
    for fold, train_config_path, _ in generated_configs:
        command = [args.python, "scripts/train.py", "--config", str(train_config_path), "--fit"]
        print(f"Fold {fold} command:")
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


def main() -> None:
    args = parse_args()
    if args.configs_only:
        args.skip_prepare_data = True
        args.dry_run = True
    if not args.skip_prepare_data:
        prepare_data(args)
    else:
        validate_existing_fold_inputs(args)
    generated_configs = build_fold_configs(args)
    print("Generated fold configs:")
    for fold, train_path, eval_path in generated_configs:
        print(f"  fold_{fold}: train={train_path} eval={eval_path}")
    if args.configs_only:
        return
    run_training(args, generated_configs)


if __name__ == "__main__":
    main()
