"""Run a fixed-split domain-classifier audit.

All Python invocations for this repository should use the ANDi environment,
for example::

    C:\\Users\\E-118-3\\miniconda3\\envs\\ANDi\\python.exe \\
      scripts\\run_domain_classifier.py --config configs\\domain_classifier.yaml --dry-run

The command does not create a report or overwrite an audit file.  It writes
one resumable ``seed_*.json`` and one best-model state dict per requested seed
under the explicitly supplied output directory; report rendering is a separate
step so that provenance remains immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.domain_classifier.runner import (  # noqa: E402
    DomainClassifierRunner,
    TrainConfig,
    atomic_write_json,
    dataset_from_manifest,
    fit_shared_train_scalar,
    iter_retrained_pair_permutation_control,
    materialize_dataset,
    permutation_control_status,
    read_jsonl_manifest,
    run_fingerprint,
    run_statistical_control,
    subset_dataset,
    write_prediction_rows,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - config environment includes PyYAML
        raise RuntimeError("The CLI requires PyYAML to load the YAML config.") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config {path} must contain a mapping at its root.")
    return value


def _config_from_yaml(value: dict[str, Any], *, stage: str | None, device: str | None) -> TrainConfig:
    training = value.get("training", value)
    if not isinstance(training, dict):
        training = {}
    allowed = set(TrainConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: item for key, item in training.items() if key in allowed}
    if "widths" in kwargs:
        kwargs["widths"] = tuple(int(item) for item in kwargs["widths"])
    if "modalities" in kwargs:
        kwargs["modalities"] = tuple(str(item).strip().lower() for item in kwargs["modalities"])
    if stage is not None:
        kwargs["stage"] = stage
    if device is not None:
        kwargs["device"] = device
    return TrainConfig(**kwargs)


def _manifest_paths(value: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    manifests = value.get("manifests", {})
    if not isinstance(manifests, dict):
        manifests = {}
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        command_value = getattr(args, f"{split}_manifest")
        candidate = command_value or manifests.get(split)
        if candidate is None and args.manifest_root:
            candidate = Path(args.manifest_root) / f"{split}.jsonl"
        if candidate is None:
            raise ValueError(f"Missing {split} manifest; pass --{split}-manifest or set manifests.{split}.")
        path = Path(candidate)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        paths[split] = path
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "state_dict"}
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def _existing_seed_is_compatible(path: Path, *, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot validate existing seed artifact {path}.") from exc
    if value.get("run_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing artifact {path} belongs to another run; choose a new output directory."
        )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--comparison", choices=("fomo", "fomo45k", "mpi", "oasis3", "mixed"), default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=("raw", "final", "registered"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modalities", nargs="+", choices=("flair", "t1", "t2"), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(73, 173, 273))
    parser.add_argument("--tiny", action="store_true", help="Use the no-early-stopping tiny smoke settings.")
    parser.add_argument("--run-permutations", action="store_true", help="Run configured retrained whole-pair controls.")
    parser.add_argument("--permute-test-labels", action="store_true", help="Also swap test labels in retrained controls.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and print counts without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config_value = _load_yaml(config_path.resolve())
    if args.comparison is not None:
        comparison_names = {"fomo": "fomo45k", "fomo45k": "fomo45k", "mpi": "mpi", "oasis3": "oasis3", "mixed": "mixed"}
        comparison = comparison_names[args.comparison]
        manifest_root = args.manifest_root or (REPO_ROOT / "outputs" / "diagnostics" / "domain_classifier" / "manifests" / comparison)
        config_value = dict(config_value)
        config_value["manifests"] = {split: str(manifest_root / f"{split}.jsonl") for split in ("train", "val", "test")}
        config_value["comparison"] = comparison
    cfg = _config_from_yaml(config_value, stage=args.stage, device=args.device)
    if args.modalities:
        cfg = replace(
            cfg,
            modalities=tuple(args.modalities),
            in_channels=len(args.modalities),
        )
    if args.tiny:
        cfg = replace(
            cfg,
            tiny=True,
            early_stopping=False,
            dropout=0.0,
            weight_decay=0.0,
            patience=max(1, cfg.max_epochs + 1),
        )
    paths = _manifest_paths(config_value, args)
    counts = {split: len(read_jsonl_manifest(path)) for split, path in paths.items()}
    summary = {
        "config": asdict(cfg),
        "manifests": {split: str(path) for split, path in paths.items()},
        "counts": counts,
        "seeds": [int(seed) for seed in args.seeds],
    }
    if args.dry_run:
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
        return 0
    output_dir = args.output_dir or Path(config_value.get("output_dir", "outputs/domain_classifier"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_paths = [
        REPO_ROOT / "domain_classifier" / "models.py",
        REPO_ROOT / "domain_classifier" / "metrics.py",
        REPO_ROOT / "domain_classifier" / "runner.py",
        Path(__file__).resolve(),
        config_path.resolve(),
    ]
    fingerprint = run_fingerprint(cfg, paths, code_paths=code_paths)
    identity = {
        "run_fingerprint": fingerprint,
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifests": {split: str(path) for split, path in paths.items()},
        "code_paths": [str(path) for path in code_paths],
    }
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        with identity_path.open("r", encoding="utf-8") as handle:
            existing_identity = json.load(handle)
        if existing_identity.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing output directory {output_dir} has a different run identity; choose a new directory."
            )
    else:
        atomic_write_json(identity_path, identity)
    datasets = {
        split: dataset_from_manifest(
            path,
            stage=cfg.stage,
            modalities=cfg.modalities,
            shared_train_scalar=cfg.shared_train_scalar,
        )
        for split, path in paths.items()
    }
    runner = DomainClassifierRunner(cfg)
    statistical_model = str(cfg.model).strip().lower().replace("-", "_") in {
        "statistical",
        "statistical_logistic",
        "logistic",
        "logistic_regression",
    }
    materialized = (
        {split: materialize_dataset(dataset) for split, dataset in datasets.items()}
        if statistical_model
        else None
    )
    completed: list[int] = []
    for seed in args.seeds:
        seed_value = int(seed)
        seed_json = output_dir / f"seed_{seed_value}.json"
        checkpoint = output_dir / f"best_seed_{seed_value}.pt"
        prediction_jsonl = output_dir / f"seed_{seed_value}_test_predictions.jsonl"
        if _existing_seed_is_compatible(seed_json, fingerprint):
            if not checkpoint.exists() or not prediction_jsonl.exists():
                raise RuntimeError(
                    f"Seed {seed_value} JSON exists but checkpoint/predictions are missing; "
                    "refusing to overwrite a partial run."
                )
            completed.append(seed_value)
            continue
        if checkpoint.exists() or prediction_jsonl.exists():
            raise RuntimeError(
                f"Partial artifacts exist for seed {seed_value}; choose a new output directory "
                "or remove only that incomplete seed after inspection."
            )
        if statistical_model:
            assert materialized is not None
            train_values = materialized["train"]
            test_values = materialized["test"]
            control = run_statistical_control(
                train_values["images"],
                train_values["labels"],
                test_values["images"],
                test_values["labels"],
                train_participant_ids=train_values["participant_ids"],
                eval_participant_ids=test_values["participant_ids"],
                eval_pair_ids=test_values["pair_ids"],
                eval_case_ids=test_values["case_ids"],
                background_value=0.0 if cfg.stage == "registered" else -1.0,
                threshold=cfg.threshold,
                seed=seed_value,
                bootstrap_replicates=cfg.bootstrap_replicates,
                swap_replicates=cfg.swap_replicates,
            )
            result = {
                "seed": seed_value,
                "split_seed": int(cfg.split_seed),
                "config": asdict(cfg),
                "device": "cpu",
                "model": "statistical_logistic",
                "test": control["metrics"],
                "test_statistics": {
                    key: control[key]
                    for key in ("subject_bootstrap", "heldout_pair_swap")
                    if key in control
                },
                "test_predictions": control.get("prediction_rows", []),
                "test_subject_predictions": control.get("subject_prediction_rows", []),
                "statistical_control": control,
                "state_dict": {
                    "coef": control.get("model_coef"),
                    "intercept": control.get("model_intercept"),
                },
            }
        else:
            result = runner.train_one_seed(
                datasets["train"],
                datasets["val"],
                datasets["test"],
                seed=seed_value,
            )
        _atomic_torch_save(result.get("state_dict"), checkpoint)
        write_prediction_rows(prediction_jsonl, result.get("test_predictions", []))
        serializable = _json_safe({"run_fingerprint": fingerprint, **result})
        atomic_write_json(seed_json, serializable)
        completed.append(seed_value)
    permutation_summary = permutation_control_status(
        requested=cfg.permutation_replicates,
        completed=0,
        mode=cfg.permutation_mode,
    )
    if args.run_permutations and int(cfg.permutation_replicates) > 0:
        permutation_dir = output_dir / "permutations"
        permutation_dir.mkdir(parents=True, exist_ok=True)
        completed_permutations = 0
        for index, result in iter_retrained_pair_permutation_control(
            datasets["train"],
            datasets["val"],
            datasets["test"],
            config=cfg,
            replicates=cfg.permutation_replicates,
            seed=cfg.seed,
            permute_test_labels=args.permute_test_labels,
        ):
            destination = permutation_dir / f"permutation_{index:04d}.json"
            if destination.exists():
                raise RuntimeError(
                    f"Permutation artifact {destination} already exists; choose a new output directory "
                    "or run a dedicated resume implementation after inspecting it."
                )
            atomic_write_json(
                destination,
                _json_safe({"run_fingerprint": fingerprint, "permutation_index": index, **result}),
            )
            write_prediction_rows(
                permutation_dir / f"permutation_{index:04d}_test_predictions.jsonl",
                result.get("test_predictions", []),
            )
            completed_permutations += 1
        permutation_summary = permutation_control_status(
            requested=cfg.permutation_replicates,
            completed=completed_permutations,
            mode=cfg.permutation_mode,
        )
        atomic_write_json(permutation_dir / "status.json", permutation_summary)
    print(json.dumps(_json_safe({**summary, "output_dir": str(output_dir), "run_fingerprint": fingerprint, "completed_seeds": completed, "permutation_controls": permutation_summary}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
