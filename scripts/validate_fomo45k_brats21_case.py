"""Independently validate a published FOMO45K BraTS21-like case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from andi_rewrite.data.datasets.fomo45k_brats21 import FOMO45KBraTS21VolumeDataset
from andi_rewrite.data.fomo45k.brats21 import (
    BRATS_SHAPE,
    BRATS_SPACING_MM,
    MODEL_CHANNEL_ORDER,
    _read_itk_affine_linear,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path(r"C:\ML\data\atlases\brats_sri24\brats_sri24.nii"),
    )
    return parser


def _spatial_header(image: nib.spatialimages.SpatialImage) -> dict[str, object]:
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return {
        "affine": np.asarray(image.affine),
        "qform": np.asarray(qform),
        "qform_code": int(qcode),
        "sform": np.asarray(sform),
        "sform_code": int(scode),
        "zooms": np.asarray(image.header.get_zooms()[:3]),
    }


def main() -> int:
    args = _parser().parse_args()
    case_dir = args.case_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    atlas = nib.load(str(args.atlas.resolve()))
    atlas_header = _spatial_header(atlas)
    failures: list[str] = []
    checks: dict[str, object] = {
        "case_id": status["case_id"],
        "channel_order": list(MODEL_CHANNEL_ORDER),
        "modalities": {},
    }

    mask_path = case_dir / status["brain_mask"]
    mask_image = nib.load(str(mask_path))
    mask = np.asarray(mask_image.dataobj) > 0.5
    if tuple(mask_image.shape) != BRATS_SHAPE or not mask.any():
        failures.append("invalid_brain_mask")

    for name in MODEL_CHANNEL_ORDER:
        path = case_dir / status["outputs"][name]
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
        spatial = _spatial_header(image)
        header_match = all(
            np.array_equal(spatial[key], atlas_header[key])
            for key in ("affine", "qform", "sform", "zooms")
        ) and all(
            spatial[key] == atlas_header[key] for key in ("qform_code", "sform_code")
        )
        outside_nonzero = int(np.count_nonzero(data[~mask]))
        item = {
            "path": str(path),
            "shape": list(image.shape),
            "spacing_mm": list(image.header.get_zooms()[:3]),
            "dtype": str(image.get_data_dtype()),
            "spatial_header_exact": bool(header_match),
            "outside_mask_nonzero_voxels": outside_nonzero,
        }
        checks["modalities"][name] = item
        if tuple(image.shape) != BRATS_SHAPE:
            failures.append(f"{name}_shape")
        if not np.array_equal(np.asarray(item["spacing_mm"]), np.asarray(BRATS_SPACING_MM)):
            failures.append(f"{name}_spacing")
        if image.get_data_dtype() != np.dtype(np.float32):
            failures.append(f"{name}_dtype")
        if not header_match:
            failures.append(f"{name}_spatial_header")
        if outside_nonzero:
            failures.append(f"{name}_background")

    source_checks: dict[str, object] = {}
    for name, source in status["source_state"].items():
        path = Path(source["path"])
        current = {"size": path.stat().st_size, "sha256": _sha256(path)}
        current["unchanged"] = bool(
            current["size"] == source["size"] and current["sha256"] == source["sha256"]
        )
        source_checks[name] = current
        if not current["unchanged"]:
            failures.append(f"{name}_source_changed")
    checks["source_checks"] = source_checks

    transform_path = case_dir / "transforms" / "t1_to_sri24_0GenericAffine.mat"
    linear = _read_itk_affine_linear(transform_path)
    determinant = float(np.linalg.det(linear))
    checks["affine"] = {
        "finite": bool(np.all(np.isfinite(linear))),
        "determinant": determinant,
        "invertible": bool(np.isfinite(determinant) and abs(determinant) > 1.0e-12),
    }
    if not checks["affine"]["finite"] or not checks["affine"]["invertible"]:
        failures.append("affine_invalid")
    if not 0.5 <= abs(determinant) <= 2.0:
        failures.append("affine_determinant")

    dataset = FOMO45KBraTS21VolumeDataset(
        dataset_root,
        return_dict=True,
        return_metadata=True,
    )
    index = next(i for i, row in enumerate(dataset.rows) if row["case_id"] == status["case_id"])
    tensors, metadata = dataset[index]
    loader_stats: dict[str, object] = {
        "image_shape": list(tensors["image"].shape),
        "dtype": str(tensors["image"].dtype),
        "channel_order": metadata["channel_order"],
        "modalities": {},
    }
    tensor_mask = tensors["brain_mask"]
    for name in MODEL_CHANNEL_ORDER:
        tensor = tensors[name]
        foreground = tensor[tensor_mask]
        background = tensor[~tensor_mask]
        loader_stats["modalities"][name] = {
            "mean": float(foreground.mean()),
            "std": float(foreground.std(unbiased=False)),
            "background_max_abs": float(background.abs().max()),
        }
        if abs(float(foreground.mean())) > 1.0e-5:
            failures.append(f"{name}_loader_mean")
        if abs(float(foreground.std(unbiased=False)) - 1.0) > 1.0e-5:
            failures.append(f"{name}_loader_std")
        if bool(torch_any_nonzero(background)):
            failures.append(f"{name}_loader_background")
    checks["loader"] = loader_stats
    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    print(json.dumps(checks, indent=2))
    return 0 if not failures else 1


def torch_any_nonzero(tensor) -> bool:
    return bool((tensor != 0).any().item())


if __name__ == "__main__":
    raise SystemExit(main())
