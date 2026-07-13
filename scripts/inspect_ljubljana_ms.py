"""Inspect Shifts MS / Ljubljana-style NIfTI volumes and run dataset smoke checks."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch

from andi_rewrite.data import build_dataloader, build_dataset
from andi_rewrite.data.datasets import ShiftsMSVolumeDataset
from andi_rewrite.utils import load_config, print_config


DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "type": "ljubljana_ms_volume",
    "image_size": 128,
    "modalities": ["flair", "t1", "t1ce", "t2"],
    "preferred_locations": ["ljubljana", "best", "msseg"],
    "splits": ["dev_out", "dev_in", "eval_in"],
    "reference_modality": "flair",
    "require_segmentation": False,
    "require_modalities": True,
    "resample_to_reference": True,
    "histogram_normalization": False,
    "return_metadata": True,
    "modality_mapping": {
        "flair": ["flair", "FLAIR"],
        "t1": ["t1", "T1"],
        "t1ce": ["t1ce", "T1Post", "t1post", "pd", "PD"],
        "t2": ["t2", "T2"],
        "segmentation": ["Gold_Standard", "gold_standard", "gt"],
        "brain_mask": ["fg_mask", "brain_mask", "mask"],
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_nifti_info(path: Path) -> dict[str, Any]:
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    finite = data[np.isfinite(data)]
    return {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(data.dtype),
        "header_dtype": str(image.header.get_data_dtype()),
        "voxel_spacing": [float(item) for item in image.header.get_zooms()[:3]],
        "orientation": "".join(nib.orientations.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "nonzero_voxels": int(np.count_nonzero(data)),
        "nan_or_inf_voxels": int(data.size - finite.size),
    }


def _same_affine(left: list[list[float]], right: list[list[float]]) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), atol=1.0e-4))


def _inventory_config(config: dict[str, Any] | None, dataset_path: str) -> dict[str, Any]:
    data_config = copy.deepcopy(DEFAULT_DATA_CONFIG)
    if config:
        data_config.update(copy.deepcopy(config.get("data", {})))
    data_config["dataset_path"] = dataset_path
    data_config["require_segmentation"] = bool(data_config.get("require_segmentation", False))
    data_config.pop("subject_limit", None)
    return data_config


def build_inventory(dataset_path: str, config: dict[str, Any] | None) -> dict[str, Any]:
    data_config = _inventory_config(config, dataset_path)
    dataset = build_dataset(data_config)
    if not isinstance(dataset, ShiftsMSVolumeDataset):
        raise TypeError(f"Inventory requires ShiftsMSVolumeDataset, got {type(dataset)!r}")

    subject_rows = []
    location_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    warnings: list[str] = []

    for subject in dataset.subjects:
        location_counts[subject.location] += 1
        split_counts[f"{subject.location}/{subject.split}"] += 1
        ref_info = _load_nifti_info(subject.reference_path)
        modality_info = {}
        same_shape = True
        same_affine = True
        for modality, path in subject.modality_paths.items():
            source_counts[modality][subject.modality_sources.get(modality, "missing")] += 1
            if path is None:
                modality_info[modality] = {"path": "", "missing": True}
                same_shape = False
                same_affine = False
                continue
            info = _load_nifti_info(path)
            modality_info[modality] = info
            same_shape = same_shape and info["shape"] == ref_info["shape"]
            same_affine = same_affine and _same_affine(info["affine"], ref_info["affine"])

        segmentation_info = None
        if subject.segmentation_path is not None:
            segmentation_info = _load_nifti_info(subject.segmentation_path)
            same_shape = same_shape and segmentation_info["shape"] == ref_info["shape"]
            same_affine = same_affine and _same_affine(segmentation_info["affine"], ref_info["affine"])

        brain_mask_info = None
        if subject.brain_mask_path is not None:
            brain_mask_info = _load_nifti_info(subject.brain_mask_path)
            same_shape = same_shape and brain_mask_info["shape"] == ref_info["shape"]
            same_affine = same_affine and _same_affine(brain_mask_info["affine"], ref_info["affine"])

        if not same_shape:
            warnings.append(f"{subject.location}/{subject.split}/{subject.subject_id}: shape mismatch")
        if not same_affine:
            warnings.append(f"{subject.location}/{subject.split}/{subject.subject_id}: affine mismatch")

        subject_rows.append(
            {
                "subject_id": subject.subject_id,
                "location": subject.location,
                "split": subject.split,
                "reference_modality": dataset.reference_modality,
                "reference": ref_info,
                "modalities": modality_info,
                "modality_mapping": subject.modality_sources,
                "segmentation": segmentation_info,
                "brain_mask": brain_mask_info,
                "has_segmentation": subject.segmentation_path is not None,
                "has_brain_mask": subject.brain_mask_path is not None,
                "same_shape_as_reference": same_shape,
                "same_affine_as_reference": same_affine,
            }
        )

    available_locations = sorted(location_counts)
    if "ljubljana" not in {name.lower() for name in available_locations}:
        warnings.append(
            "No Ljubljana location directory was found in this local dataset path; "
            "inventory reflects the available Shifts MS locations."
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(Path(dataset_path)),
        "resolved_root": str(dataset.root),
        "data_config": data_config,
        "summary": {
            "subjects": len(subject_rows),
            "locations": dict(location_counts),
            "splits": dict(split_counts),
            "modality_sources": {key: dict(value) for key, value in source_counts.items()},
            "all_subjects_same_shape_as_reference": all(row["same_shape_as_reference"] for row in subject_rows),
            "all_subjects_same_affine_as_reference": all(row["same_affine_as_reference"] for row in subject_rows),
            "subjects_with_segmentation": sum(1 for row in subject_rows if row["has_segmentation"]),
            "subjects_with_brain_mask": sum(1 for row in subject_rows if row["has_brain_mask"]),
        },
        "warnings": warnings,
        "subjects": subject_rows,
    }


def inventory_markdown(inventory: dict[str, Any]) -> str:
    summary = inventory["summary"]
    lines = [
        "# Ljubljana / Shifts MS Dataset Inventory",
        "",
        f"- Dataset path: `{inventory['dataset_path']}`",
        f"- Resolved root: `{inventory['resolved_root']}`",
        f"- Generated at: `{inventory['generated_at']}`",
        f"- Subjects: {summary['subjects']}",
        f"- Locations: `{json.dumps(summary['locations'], ensure_ascii=False)}`",
        f"- Splits: `{json.dumps(summary['splits'], ensure_ascii=False)}`",
        f"- Modality sources: `{json.dumps(summary['modality_sources'], ensure_ascii=False)}`",
        f"- All shapes match reference: {summary['all_subjects_same_shape_as_reference']}",
        f"- All affines match reference: {summary['all_subjects_same_affine_as_reference']}",
        f"- Subjects with segmentation: {summary['subjects_with_segmentation']}",
        f"- Subjects with brain mask: {summary['subjects_with_brain_mask']}",
        "",
    ]
    if inventory["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in inventory["warnings"])
        lines.append("")

    lines.extend(
        [
            "## Subjects",
            "",
            "| subject | location | split | native shape | spacing | orientation | mapping | label | brain mask | same grid |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in inventory["subjects"]:
        ref = row["reference"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["subject_id"]),
                    str(row["location"]),
                    str(row["split"]),
                    "x".join(str(item) for item in ref["shape"]),
                    "x".join(f"{item:g}" for item in ref["voxel_spacing"]),
                    ref["orientation"],
                    json.dumps(row["modality_mapping"], ensure_ascii=False),
                    "yes" if row["has_segmentation"] else "no",
                    "yes" if row["has_brain_mask"] else "no",
                    "yes" if row["same_shape_as_reference"] and row["same_affine_as_reference"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_inventory(inventory: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ljubljana_dataset_inventory.json"
    md_path = output_dir / "ljubljana_dataset_inventory.md"
    json_path.write_text(json.dumps(_json_safe(inventory), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(inventory_markdown(inventory), encoding="utf-8")
    return json_path, md_path


def _smoke_data_config(config: dict[str, Any] | None, dataset_path: str, subject_limit: int | None) -> dict[str, Any]:
    data_config = _inventory_config(config, dataset_path)
    data_config["batch_size"] = 1
    data_config["workers"] = 0
    data_config["return_metadata"] = True
    if subject_limit is not None:
        data_config["subject_limit"] = subject_limit
    return data_config


def run_dataset_smoke(config: dict[str, Any] | None, dataset_path: str, subject_limit: int | None) -> dict[str, Any]:
    data_config = _smoke_data_config(config, dataset_path, subject_limit)
    dataloader = build_dataloader(data_config)
    image, label, metadata = next(iter(dataloader))
    result = {
        "dataset_length": len(dataloader.dataset),  # type: ignore[arg-type]
        "image_shape": list(image.shape),
        "label_shape": list(label.shape) if label is not None else None,
        "label_dtype": str(label.dtype) if label is not None else None,
        "label_any": bool(label.any().item()) if label is not None else False,
        "metadata": metadata,
    }
    if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 4:
        raise AssertionError(f"Expected [1, 4, H, W, Z] image tensor, got {tuple(image.shape)}")
    if label is not None and tuple(label.shape[-3:]) != tuple(image.shape[-3:]):
        raise AssertionError(f"Label shape {tuple(label.shape)} does not match image volume shape {tuple(image.shape)}")
    if label is not None and label.dtype != torch.bool:
        raise AssertionError(f"Label tensor must be bool, got {label.dtype}")
    return result


def run_inference_smoke(config: dict[str, Any], dataset_path: str, subject_limit: int | None) -> dict[str, Any]:
    from andi_rewrite.engine import VolumeEvaluator
    from andi_rewrite.scripts.eval import build_detector_from_config

    smoke_config = copy.deepcopy(config)
    smoke_config.setdefault("data", {})
    smoke_config["data"].update(_smoke_data_config(config, dataset_path, subject_limit or 1))
    smoke_config["data"]["subject_limit"] = subject_limit or 1
    detector, accelerator = build_detector_from_config(smoke_config)
    dataloader = build_dataloader(smoke_config.get("data", {}))
    evaluator = VolumeEvaluator(
        detector=detector,
        config={
            **smoke_config.get("data", {}),
            **smoke_config.get("metrics", {}),
            **smoke_config.get("evaluation", {}),
            "prediction_output": smoke_config.get("prediction_output", {}),
            "model": smoke_config.get("model", {}),
            "anomaly": smoke_config.get("anomaly", {}),
        },
        accelerator=accelerator,
    )
    dataloader = evaluator.prepare(dataloader)
    return evaluator.evaluate(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Shifts MS / Ljubljana NIfTI data and smoke-test the loader.")
    parser.add_argument("--dataset", required=True, help="Path to the Shifts MS dataset root or archive root.")
    parser.add_argument("--config", help="Optional eval YAML config to use for dataset smoke checks.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Limit smoke dataset subjects; inventory still scans all.")
    parser.add_argument("--diagnostics-dir", default="outputs/diagnostics", help="Directory for inventory JSON/Markdown.")
    parser.add_argument("--run-inference", action="store_true", help="Also run one-subject ANDi inference/export using --config.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    inventory = build_inventory(args.dataset, config)
    json_path, md_path = write_inventory(inventory, Path(args.diagnostics_dir))
    print(f"Wrote inventory JSON: {json_path}")
    print(f"Wrote inventory Markdown: {md_path}")
    print("Inventory summary:")
    print_config(inventory["summary"])
    if inventory["warnings"]:
        print("Inventory warnings:")
        for warning in inventory["warnings"]:
            print(f"- {warning}")

    smoke = run_dataset_smoke(config, args.dataset, args.subject_limit)
    print("Dataset smoke result:")
    print_config(_json_safe(smoke))

    if args.run_inference:
        if config is None:
            raise ValueError("--run-inference requires --config.")
        result = run_inference_smoke(config, args.dataset, args.subject_limit)
        print("Inference smoke result:")
        print_config(_json_safe(result))


if __name__ == "__main__":
    main()
