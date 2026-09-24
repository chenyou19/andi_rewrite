"""Create robust-IQR model-input montages for OASIS3/MPI versus tumor-free BraTS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize

from andi_rewrite.data import build_dataloader
from andi_rewrite.data.robust_normalization import robust_normalize_volume
from andi_rewrite.scripts.prepare_robust_b import load_normalized_case
from andi_rewrite.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
BRAINS_CONFIG = ROOT / "configs/eval_brats_val50_robust_iqr20_b.yaml"
MODS = ["FLAIR", "T1", "T2"]
TARGET_Z = [50, 74, 100]
MIN_FOREGROUND_FRACTION = 0.10

COHORTS = {
    "oasis3": {
        "label": "OASIS3",
        "data_root": ROOT / "outputs/datasets/oasis3_sri24_robust_iqr",
        "config": ROOT / "configs/train_oasis3_sri24_robust_iqr20_own_spectrum.yaml",
        "output": ROOT / "outputs/diagnostics/robust_b/input_comparison_oasis3_tumor_free_20",
    },
    "mpi": {
        "label": "MPI",
        "data_root": ROOT / "outputs/datasets/mpi_sri24_robust_iqr",
        "config": ROOT / "configs/train_mpi_sri24_robust_iqr60_own_spectrum.yaml",
        "output": ROOT / "outputs/diagnostics/robust_b/input_comparison_mpi_tumor_free_20",
    },
}


def load_registered_case(data_root: Path, case_id: str):
    case_root = data_root / "volumes" / Path(case_id)
    status_path = case_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        raise ValueError(f"Source case is not PASS: {case_id}")
    paths = {
        modality.upper(): case_root / status["outputs"][modality]
        for modality in ("flair", "t1", "t2")
    }
    brain_mask = case_root / status["brain_mask"]
    return paths, brain_mask, status


def raw_case(paths: dict[str, Path]):
    images = [nib.load(str(paths[modality])) for modality in MODS]
    reference = images[0]
    for image in images:
        if image.shape != (240, 240, 155):
            raise ValueError(f"Unexpected registered shape: {image.shape}")
        np.testing.assert_allclose(image.affine, reference.affine, atol=1e-6, rtol=0)
    values = torch.from_numpy(
        np.stack([np.asarray(image.dataobj, dtype=np.float32) for image in images])
    )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Registered source contains NaN/Inf")
    return values, reference


def render_pair(pair: dict, destination: Path, source_label: str):
    figure, axes = plt.subplots(3, 6, figsize=(15, 8.6), facecolor="#111111")
    image_artist = None
    for row, z in enumerate(pair["z"]):
        for column in range(6):
            axis = axes[row, column]
            cohort, channel = column // 3, column % 3
            value = pair["arrays"][cohort][row, channel]
            image_artist = axis.imshow(
                value.T,
                origin="lower",
                cmap="gray",
                vmin=-1,
                vmax=3,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    (source_label if cohort == 0 else "BraTS21") + " " + MODS[channel],
                    color="white",
                    fontsize=11,
                )
            if column == 0:
                axis.set_ylabel(f"z={z}", color="white")
    figure.suptitle(
        f"Pair {pair['number']:02d} | {source_label}: {pair['source_case']}  vs  "
        f"BraTS21: {pair['brats_case']} | zero labeled lesion voxels",
        color="white",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.035, right=0.94, top=0.92, bottom=0.065, wspace=0.025, hspace=0.06)
    colorbar = figure.colorbar(image_artist, cax=figure.add_axes([0.955, 0.15, 0.012, 0.65]))
    colorbar.ax.tick_params(colors="white")
    colorbar.set_label("Model intensity", color="white")
    figure.text(
        0.5,
        0.02,
        "128 x 128 | median/IQR | fixed window [-1, 3] | same z is not exact anatomical correspondence",
        ha="center",
        color="white",
        fontsize=10,
    )
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def source_cases(data_root: Path):
    entries_by_case: dict[str, dict[int, int]] = defaultdict(dict)
    cases_by_participant: dict[str, set[str]] = defaultdict(set)
    with (data_root / "entries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            case_id = str(row["case_id"])
            entries_by_case[case_id][int(row["z"])] = int(row["key"])
            cases_by_participant[str(row["participant_id"])].add(case_id)
    cases = {}
    for participant in sorted(cases_by_participant):
        case_id = sorted(cases_by_participant[participant])[0]
        cases[participant] = case_id
    return entries_by_case, cases


def main(dataset: str, count: int):
    if count < 1:
        raise ValueError("count must be positive")
    cohort = COHORTS[dataset]
    data_root = cohort["data_root"]
    output = cohort["output"]
    output.mkdir(parents=True, exist_ok=True)

    train_config = load_config(cohort["config"])
    evaluation_config = load_config(BRAINS_CONFIG)
    if train_config["training"]["normalize_input"] is not False:
        raise ValueError("Healthy model input must already be robust-IQR model space")
    if evaluation_config["data"]["normalize_input"] is not False:
        raise ValueError("BraTS model input must already be robust-IQR model space")
    if evaluation_config["data"]["modalities"] != ["flair", "t1", "t2"]:
        raise ValueError("BraTS channel order is not FLAIR,T1,T2")

    source_dataset = build_dataloader({**train_config["data"], "shuffle": False}).dataset
    brats_dataset = build_dataloader(evaluation_config["data"]).dataset
    source_entries, participants = source_cases(data_root)
    if count > len(participants):
        raise ValueError(f"{dataset} has only {len(participants)} train participants")
    if count > len(brats_dataset):
        raise ValueError(f"BraTS validation has only {len(brats_dataset)} subjects")

    report = json.loads((data_root / "build_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError(f"{dataset} build report is not PASS")

    rng = np.random.default_rng(73)
    selected_participants = rng.choice(sorted(participants), count, replace=False)
    selected_cases = [participants[str(participant)] for participant in selected_participants]
    selected_brats = rng.choice(len(brats_dataset), count, replace=False)
    resize = Resize(128, antialias=True)
    records = []

    for number, (participant, source_case, brats_index) in enumerate(
        zip(selected_participants, selected_cases, selected_brats), 1
    ):
        source_paths, source_mask_path, source_status = load_registered_case(data_root, source_case)
        source_raw, source_reference = raw_case(source_paths)
        source_normalized, _ = load_normalized_case(source_paths)
        source_mask_image = nib.load(str(source_mask_path))
        source_mask = np.asarray(source_mask_image.dataobj) > 0
        np.testing.assert_allclose(source_mask_image.affine, source_reference.affine, atol=1e-6, rtol=0)
        if source_mask.shape != (240, 240, 155):
            raise ValueError(f"Unexpected source mask shape: {source_mask.shape}")

        brats_volume, brats_mask, brats_metadata = brats_dataset[int(brats_index)]
        brats_paths = {modality: Path(brats_metadata["input_paths"][modality.lower()]) for modality in MODS}
        brats_raw, brats_reference = raw_case(brats_paths)
        brats_normalized = robust_normalize_volume(brats_raw)
        segmentation_image = nib.load(brats_metadata["segmentation_path"])
        segmentation = torch.from_numpy(np.asarray(segmentation_image.dataobj, dtype=np.float32))
        np.testing.assert_allclose(segmentation_image.affine, brats_reference.affine, atol=1e-6, rtol=0)
        if segmentation.shape != (240, 240, 155):
            raise ValueError(f"Unexpected BraTS segmentation shape: {segmentation.shape}")

        common_z = source_entries[source_case]
        lesion_counts = (segmentation > 0).sum(dim=(0, 1))
        available = sorted(
            z
            for z in common_z
            if 0 <= z < 155
            and bool(source_mask[:, :, z].any())
            and lesion_counts[z] == 0
            and bool(
                (
                    (source_raw[..., z] > 0).flatten(1).float().mean(dim=1)
                    >= MIN_FOREGROUND_FRACTION
                ).all()
            )
            and bool(
                (
                    (brats_raw[..., z] > 0).flatten(1).float().mean(dim=1)
                    >= MIN_FOREGROUND_FRACTION
                ).all()
            )
        )
        if len(available) < len(TARGET_Z):
            raise ValueError(f"Insufficient common tumor-free slices for {source_case} / {brats_metadata['subject_id']}")
        selected_z = []
        for target in TARGET_Z:
            selected_z.append(min((z for z in available if z not in selected_z), key=lambda z: (abs(z - target), z)))

        assert not bool((segmentation[..., selected_z] > 0).any())
        assert not bool(brats_mask[..., selected_z].any())
        source_entries_for_z = [source_entries[source_case][z] for z in selected_z]
        source_inputs = torch.stack([source_dataset[key] for key in source_entries_for_z])
        brats_inputs = torch.stack([brats_volume[..., z] for z in selected_z])
        verification_errors = []
        for row, z in enumerate(selected_z):
            expected_source = resize(source_normalized[..., z][None])[0]
            expected_brats = resize(brats_normalized[..., z][None])[0]
            for actual, expected in ((source_inputs[row], expected_source), (brats_inputs[row], expected_brats)):
                if actual.shape != (3, 128, 128) or not bool(torch.isfinite(actual).all()):
                    raise ValueError("Invalid model input shape or finite-value check")
                verification_errors.append(float((actual - expected).abs().max()))
                assert torch.equal(actual, expected)

        source_core = resize((source_raw[..., selected_z] > 0).float().permute(3, 0, 1, 2)).numpy() > 0.999
        brats_core = resize((brats_raw[..., selected_z] > 0).float().permute(3, 0, 1, 2)).numpy() > 0.999
        arrays = [source_inputs.numpy(), brats_inputs.numpy()]
        window_statistics = {}
        for label, values, core in (
            (f"{cohort['label']} foreground", arrays[0], source_core),
            ("BraTS tumor-free foreground", arrays[1], brats_core),
        ):
            window_statistics[label] = {
                modality: {
                    "pixels": int(values[:, index][core[:, index]].size),
                    "below_window_fraction": float(np.mean(values[:, index][core[:, index]] < -1)),
                    "above_window_fraction": float(np.mean(values[:, index][core[:, index]] > 3)),
                }
                for index, modality in enumerate(MODS)
            }

        pair = {
            "number": number,
            "source_case": source_case,
            "brats_case": brats_metadata["subject_id"],
            "z": selected_z,
            "arrays": arrays,
        }
        render_pair(pair, output / f"pair_{number:02d}.png", cohort["label"])
        records.append(
            {
                "pair": number,
                "source_dataset": dataset,
                "source_participant": str(participant),
                "source_case": source_case,
                "brats_case": brats_metadata["subject_id"],
                "z": selected_z,
                "source_lmdb_keys": source_entries_for_z,
                "source_paths": {key: str(value) for key, value in source_paths.items()},
                "brats_paths": {key: str(value) for key, value in brats_paths.items()},
                "brats_segmentation": brats_metadata["segmentation_path"],
                "source_status": str(data_root / "volumes" / Path(source_case) / "status.json"),
                "orientation": list(nib.aff2axcodes(source_reference.affine)),
                "affine": source_reference.affine.tolist(),
                "native_lesion_voxels": [int(lesion_counts[z]) for z in selected_z],
                "model_lesion_voxels": [int(brats_mask[..., z].sum()) for z in selected_z],
                "source_foreground_fraction": [
                    [float((source_raw[channel, ..., z] > 0).float().mean()) for channel in range(3)]
                    for z in selected_z
                ],
                "brats_foreground_fraction": [
                    [float((brats_raw[channel, ..., z] > 0).float().mean()) for channel in range(3)]
                    for z in selected_z
                ],
                "max_input_verification_error": max(verification_errors),
                "sha256": [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays],
                "window_statistics": window_statistics,
            }
        )
        print(
            f"{number}/{count}: {cohort['label']} {source_case} vs {brats_metadata['subject_id']}; "
            f"z={selected_z}; max error={max(verification_errors)}",
            flush=True,
        )

    result = {
        "seed": 73,
        "source_dataset": dataset,
        "source_label": cohort["label"],
        "modalities": MODS,
        "shape": [3, 128, 128],
        "window": [-1, 3],
        "normalization": report["normalization"],
        "min_foreground_fraction_per_modality": MIN_FOREGROUND_FRACTION,
        "selection": "Twenty distinct source participants (one sorted train case each) and twenty distinct BraTS validation subjects. For targets 50, 74, 100, choose nearest unused common z with zero native segmentation voxels and sufficient foreground; ties choose lower z.",
        "display": "Transpose axes, origin lower, nearest interpolation; identical registered affine checked; no intensity changes.",
        "distribution": "Not generated in this exact-20 output mode. Each selected source and BraTS model input was checked against full-volume robust-IQR normalization followed by the existing resize.",
        "pairs": records,
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(COHORTS), required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    main(args.dataset, args.count)
