from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from andi_rewrite.domain_classifier.v3_runtime import (
    CANONICAL_MODALITIES,
    MODEL_SHAPE,
    SPLITS,
    CachedV3Dataset,
    V3CachedInputs,
    canonical_source_identity,
)
from andi_rewrite.domain_classifier.runner import ManifestDataset, TrainConfig
from scripts import run_domain_classifier_controls_v3 as controls_v3


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _row(
    *,
    split: str,
    label: int,
    participant: str,
    pair: str,
    key: str,
    z: int,
) -> dict[str, Any]:
    if label == 0:
        source_dataset = "fomo45k"
        source_split = split
        source_key = key
        provenance = {
            "source_split_immutable": source_split,
            "source_key_immutable": source_key,
        }
        metadata = {"source_participant_id": participant.split(":", 1)[1]}
    else:
        source_dataset = "brats21"
        source_split = "train"
        source_key = key
        provenance = {"model_input_sha256": "placeholder"}
        metadata = {"source_participant_id": participant}
    return {
        "split": split,
        "label": label,
        "domain": "fomo45k" if label == 0 else "brats21",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_key": source_key,
        "participant_id": participant,
        "pair_id": pair,
        "case_id": f"case-{key}",
        "session_id": "ses-1",
        "z": z,
        "z_bin": z,
        "model_shape": list(MODEL_SHAPE),
        "metadata": metadata,
        "provenance": provenance,
    }


def _cached_fixture(*, participants_per_split: int = 20, rows_per_participant: int = 4) -> V3CachedInputs:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    tensors_by_identity: dict[tuple[str, str, str, str], torch.Tensor] = {}
    for split_index, split in enumerate(SPLITS):
        rows: list[dict[str, Any]] = []
        for pair_index in range(participants_per_split // 2):
            healthy = f"fomo45k:healthy-{split}-{pair_index}"
            brats = f"brats21:case-{split}-{pair_index}"
            pair = f"fomo45k:{split}:pair-{pair_index}"
            for slice_index in range(rows_per_participant):
                # Every healthy participant has the same z-bin support so the
                # existing same-cohort matcher can form a non-empty pair.
                z = slice_index
                healthy_row = _row(
                    split=split,
                    label=0,
                    participant=healthy,
                    pair=pair,
                    key=f"healthy-{split}-{pair_index}-{slice_index}",
                    z=z,
                )
                brats_row = _row(
                    split=split,
                    label=1,
                    participant=brats,
                    pair=pair,
                    key=f"brats-{split}-{pair_index}-{slice_index}",
                    z=z,
                )
                healthy_tensor = torch.full(
                    MODEL_SHAPE,
                    float(10 + split_index + pair_index + slice_index / 10.0),
                    dtype=torch.float32,
                )
                brats_tensor = torch.full(
                    MODEL_SHAPE,
                    float(20 + split_index + pair_index + slice_index / 10.0),
                    dtype=torch.float32,
                )
                healthy_identity = ("healthy", "fomo45k", split, healthy_row["source_key"])
                brats_identity = ("brats21", f"case-{split}-{pair_index}", "z", str(z))
                tensors_by_identity[healthy_identity] = healthy_tensor
                tensors_by_identity[brats_identity] = brats_tensor
                healthy_row["provenance"]["model_input_sha256"] = _digest(healthy_tensor)
                brats_row["provenance"]["model_input_sha256"] = _digest(brats_tensor)
                rows.extend((healthy_row, brats_row))
        records_by_split[split] = rows
    datasets = {
        split: CachedV3Dataset(
            records_by_split[split],
            [tensors_by_identity[canonical_source_identity(row)] for row in records_by_split[split]],
        )
        for split in SPLITS
    }
    digest_map = {
        identity: _digest(tensor)
        for identity, tensor in tensors_by_identity.items()
    }
    return V3CachedInputs(
        comparison="fomo45k",
        build_root=Path("."),
        manifest_root=Path("."),
        manifest_paths={},
        records_by_split=records_by_split,
        datasets=datasets,
        tensors_by_identity=tensors_by_identity,
        tensor_digests_by_identity=digest_map,
        input_binding_audit={
            "status": "PASS",
            "manifest_identity": {},
            "rows_checked": sum(len(rows) for rows in records_by_split.values()),
        },
        source_freeze={"status": "PASS", "files": []},
        ledger=SimpleNamespace(ledger_sha256="fixture-ledger"),
        manifest_fingerprints={split: f"sha-{split}" for split in SPLITS},
    )


def test_tiny_selector_projects_same_canonical_tensor_objects() -> None:
    cached = _cached_fixture()
    selected = controls_v3.build_cached_tiny_datasets(
        cached,
        subjects_per_label=10,
        max_slices=128,
        min_slices=64,
        seed=73,
        modalities=("t1",),
    )
    assert all(len(selected[split]) == 80 for split in SPLITS)
    assert tuple(selected["test"][0]["image"].shape) == (1, 128, 128)
    selected_identity = canonical_source_identity(selected["test"].records[0])
    selected_index = next(
        index
        for index, row in enumerate(cached.records_by_split["test"])
        if canonical_source_identity(row) == selected_identity
    )
    assert selected["test"].tensors[0].data_ptr() == cached.datasets["test"].tensors[selected_index].data_ptr()


def test_negative_selector_maps_transformed_rows_without_an_image_reader() -> None:
    cached = _cached_fixture()
    selected = controls_v3.build_cached_negative_datasets(cached, seed=73)
    assert all(len(selected[split]) > 0 for split in SPLITS)
    assert all(tuple(selected[split][0]["image"].shape) == (3, 128, 128) for split in SPLITS)
    assert all(
        selected[split].tensors[index].data_ptr()
        == cached.tensors_by_identity[canonical_source_identity(row)].data_ptr()
        for split in SPLITS
        for index, row in enumerate(selected[split].records)
    )
    assert all(int(row["label"]) in (0, 1) for split in SPLITS for row in selected[split].records)


def test_registered_digest_audit_records_train_only_scalar_and_no_final_ledger_join() -> None:
    records = {
        split: [
            {
                "source_dataset": "fomo45k",
                "source_split": split,
                "source_key": f"key-{split}",
                "participant_id": f"fomo45k:{split}",
                "case_id": f"case-{split}",
                "z": 2,
                "label": 0,
                "image": np.full((3, 4, 4), 4.0, dtype=np.float32),
            }
        ]
        for split in SPLITS
    }
    raw = {split: ManifestDataset(rows) for split, rows in records.items()}
    scaled = {split: controls_v3.dataset_with_shared_train_scalar(raw[split], 2.0) for split in SPLITS}
    audit = controls_v3._registered_digest_audit(raw, scaled, scalar=2.0, modalities=CANONICAL_MODALITIES)
    assert audit["scalar_fit"] == {"source_split": "train", "quantile": 0.995, "train_only": True}
    assert audit["pre_scalar_aggregate_sha256"] != audit["post_scalar_aggregate_sha256"]
    assert audit["rows_by_split"]["train"]["rows"][0]["pre_scalar_sha256"] != audit["rows_by_split"]["train"]["rows"][0]["post_scalar_sha256"]
    assert audit["final_ledger_comparison"].startswith("NOT_PERFORMED")


def test_dry_run_writes_binding_and_manifest_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = _cached_fixture()
    build_root = tmp_path / "build"
    manifest_root = build_root / "manifests" / "fomo45k"
    manifest_root.mkdir(parents=True)
    for split in SPLITS:
        (manifest_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in cached.records_by_split[split]),
            encoding="utf-8",
        )
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  model: small_cnn\n", encoding="utf-8")
    monkeypatch.setattr(controls_v3, "validate_and_materialize_v3_inputs", lambda *args, **kwargs: cached)
    monkeypatch.setattr(
        controls_v3,
        "set_single_thread_runtime",
        lambda: {
            "status": "PASS",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        },
    )
    result = controls_v3.run_control(
        config_path=config,
        manifest_root=manifest_root,
        output_dir=tmp_path / "out",
        mode="tiny",
        comparison="fomo45k",
        build_root=build_root,
        device="cpu",
        modalities=("flair", "t1", "t2"),
        seeds=(73,),
        dry_run=True,
    )
    assert result["status"] == "PLANNED_NO_TRAINING"
    assert (tmp_path / "out" / "input_binding_audit.json").is_file()
    assert (tmp_path / "out" / "source_freeze.json").is_file()
    assert (tmp_path / "out" / "tiny" / "manifests" / "train.jsonl").is_file()
    dry_run = json.loads((tmp_path / "out" / "tiny" / "dry_run.json").read_text(encoding="utf-8"))
    assert dry_run["training_started"] is False
    assert dry_run["control_manifest"]["manifest_hashes"]["train"]["status"] == "PASS"


def test_cli_keeps_v4_tiny_compatibility_flag_and_rejects_mixed_adapter() -> None:
    parser = controls_v3.build_parser()
    args = parser.parse_args(
        [
            "--config", "config.yaml",
            "--comparison", "fomo",
            "--manifest-root", "manifests",
            "--output-dir", "out",
            "--mode", "tiny",
            "--tiny",
            "--tiny-seed", "73",
            "--tiny-max-slices", "128",
            "--tiny-min-slices", "64",
        ]
    )
    assert args.tiny is True
    assert args.mode == "tiny"
    with pytest.raises(controls_v3.ControlsV3Error, match="dedicated source-stratified"):
        controls_v3._validate_contract(
            mode="negative",
            comparison="mixed",
            config=TrainConfig(model="small_cnn"),
            modalities=CANONICAL_MODALITIES,
            seeds=(73, 173, 273),
            tiny_seed=73,
            tiny_subjects_per_label=10,
            tiny_max_slices=128,
            tiny_min_slices=64,
            stage="final",
            fit_positive_shared_scalar=False,
            retrained_permutations=0,
        )
