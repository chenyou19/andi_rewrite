"""Render one exact final-input FOMO/BraTS21 pair from the v3 test manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from andi_rewrite.data.domain_classifier.readers import load_records, load_slice  # noqa: E402


MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "diagnostics"
    / "domain_classifier"
    / "model_grid_v3_fullcandidate_20260917_final"
    / "manifests"
    / "fomo45k"
    / "test.jsonl"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "diagnostics"
    / "domain_classifier"
    / "input_examples_final_test"
)
CHANNELS = ("FLAIR", "T1", "T2")
DISPLAY_VMIN = -1.0
DISPLAY_VMAX = 3.0


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def probability_lookup() -> dict[tuple[str, str, int], float]:
    result_path = (
        REPO_ROOT
        / "outputs"
        / "diagnostics"
        / "domain_classifier"
        / "model_grid_v3_fullcandidate_20260917_final"
        / "stage_a_fomo45k_observed_v3_20260917"
        / "observed"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str, int], float] = {}
    for row in result["test_predictions"]:
        lookup[(str(row["participant_id"]), str(row["pair_id"]), int(row["record_index"]))] = float(
            row["probability"]
        )
    return lookup


def select_pair(records: list) -> tuple[object, object]:
    fomo = next(record for record in records if int(record.label) == 0)
    brats = next(
        record
        for record in records
        if int(record.label) == 1 and str(record.pair_id) == str(fomo.pair_id)
    )
    return fomo, brats


def support_fraction(tensor: torch.Tensor) -> list[float]:
    return [
        float(value)
        for value in (torch.abs(tensor + 1.0) > 1.0e-6).float().mean(dim=(1, 2)).tolist()
    ]


def render_single(name: str, record: object, tensor: torch.Tensor, probability: float) -> Path:
    destination = OUTPUT_DIR / f"{name}.png"
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), constrained_layout=True)
    for channel_index, (axis, channel_name) in enumerate(zip(axes, CHANNELS)):
        image = tensor[channel_index].numpy()
        axis.imshow(
            image,
            cmap="gray",
            vmin=DISPLAY_VMIN,
            vmax=DISPLAY_VMAX,
            interpolation="nearest",
        )
        axis.set_title(channel_name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{name} | subject={record.participant_id} | z={record.z} | "
        f"P(BraTS21 domain)={probability:.6f}",
        fontsize=11,
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def render_combined(
    fomo_record: object,
    fomo_tensor: torch.Tensor,
    fomo_probability: float,
    brats_record: object,
    brats_tensor: torch.Tensor,
    brats_probability: float,
) -> Path:
    destination = OUTPUT_DIR / "fomo_vs_brats21_input_examples.png"
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
    for row_index, (name, record, tensor, probability) in enumerate(
        (
            ("FOMO", fomo_record, fomo_tensor, fomo_probability),
            ("BraTS21", brats_record, brats_tensor, brats_probability),
        )
    ):
        for channel_index, (axis, channel_name) in enumerate(zip(axes[row_index], CHANNELS)):
            axis.imshow(
                tensor[channel_index].numpy(),
                cmap="gray",
                vmin=DISPLAY_VMIN,
                vmax=DISPLAY_VMAX,
                interpolation="nearest",
            )
            if channel_index == 0:
                subject_text = str(record.participant_id).split(":", 1)[-1]
                axis.set_title(
                    f"{name} — {channel_name}\n"
                    f"{subject_text} | z={record.z} | P={probability:.6f}",
                    fontsize=10,
                )
            else:
                axis.set_title(f"{name} — {channel_name}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Exact final model inputs from the observed FOMO-vs-BraTS21 test manifest\n"
        "display range clipped to [-1, 3] for visibility; tensors are unchanged",
        fontsize=12,
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def main() -> None:
    records = load_records(MANIFEST)
    fomo_record, brats_record = select_pair(records)
    fomo_tensor = load_slice(fomo_record, stage="final", image_size=128, base_dir=MANIFEST.parent)
    brats_tensor = load_slice(brats_record, stage="final", image_size=128, base_dir=MANIFEST.parent)
    if tuple(fomo_tensor.shape) != (3, 128, 128) or tuple(brats_tensor.shape) != (3, 128, 128):
        raise RuntimeError("unexpected model input shape")
    lookup = probability_lookup()
    # The result JSON stores record_index in test_predictions.  Find the exact
    # row indices in the manifest to bind the shown scores to the displayed inputs.
    fomo_index = next(index for index, record in enumerate(records) if record is fomo_record)
    brats_index = next(index for index, record in enumerate(records) if record is brats_record)
    fomo_probability = next(
        float(row["probability"])
        for row in json.loads(
            (
                REPO_ROOT
                / "outputs"
                / "diagnostics"
                / "domain_classifier"
                / "model_grid_v3_fullcandidate_20260917_final"
                / "stage_a_fomo45k_observed_v3_20260917"
                / "observed"
                / "result.json"
            ).read_text(encoding="utf-8")
        )["test_predictions"]
        if int(row["record_index"]) == fomo_index
    )
    brats_probability = next(
        float(row["probability"])
        for row in json.loads(
            (
                REPO_ROOT
                / "outputs"
                / "diagnostics"
                / "domain_classifier"
                / "model_grid_v3_fullcandidate_20260917_final"
                / "stage_a_fomo45k_observed_v3_20260917"
                / "observed"
                / "result.json"
            ).read_text(encoding="utf-8")
        )["test_predictions"]
        if int(row["record_index"]) == brats_index
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": str(MANIFEST.resolve()),
        "fomo": {
            "record_index": fomo_index,
            "participant_id": fomo_record.participant_id,
            "case_id": fomo_record.case_id,
            "z": int(fomo_record.z),
            "pair_id": fomo_record.pair_id,
            "probability_brats21_domain": fomo_probability,
            "tensor_shape": list(fomo_tensor.shape),
            "tensor_dtype": str(fomo_tensor.dtype).replace("torch.", ""),
            "tensor_sha256": tensor_sha256(fomo_tensor),
            "support_fraction": dict(zip(CHANNELS, support_fraction(fomo_tensor))),
            "image": str((OUTPUT_DIR / "fomo_input.png").resolve()),
        },
        "brats21": {
            "record_index": brats_index,
            "participant_id": brats_record.participant_id,
            "case_id": brats_record.case_id,
            "z": int(brats_record.z),
            "pair_id": brats_record.pair_id,
            "probability_brats21_domain": brats_probability,
            "tensor_shape": list(brats_tensor.shape),
            "tensor_dtype": str(brats_tensor.dtype).replace("torch.", ""),
            "tensor_sha256": tensor_sha256(brats_tensor),
            "support_fraction": dict(zip(CHANNELS, support_fraction(brats_tensor))),
            "image": str((OUTPUT_DIR / "brats21_input.png").resolve()),
        },
        "display": {"vmin": DISPLAY_VMIN, "vmax": DISPLAY_VMAX},
    }
    render_single("fomo_input", fomo_record, fomo_tensor, fomo_probability)
    render_single("brats21_input", brats_record, brats_tensor, brats_probability)
    combined = render_combined(
        fomo_record,
        fomo_tensor,
        fomo_probability,
        brats_record,
        brats_tensor,
        brats_probability,
    )
    outputs["combined_image"] = str(combined.resolve())
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
