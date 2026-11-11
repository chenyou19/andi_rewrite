from __future__ import annotations

import importlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from torchvision.transforms import Resize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))
from andi_rewrite.data.mean_std_normalization import mean_std_normalize_volume
from andi_rewrite.data.datasets.brats import MRIDataVolume, build_mri_volume_dataset
from andi_rewrite.engine.evaluation.fingerprints import cache_fingerprints
from andi_rewrite.anomaly import build_postprocess_policy


def volume():
    x = torch.arange(1, 3*8*8*4+1, dtype=torch.float32).reshape(3,8,8,4)
    x[:, :2] = 0
    return x


def test_population_statistics_and_background_without_input_mutation():
    x = volume()
    before = x.clone()
    result, stats = mean_std_normalize_volume(x, return_statistics=True)
    assert torch.equal(x, before)
    assert result.dtype == torch.float32
    assert (result[x == 0] == -1).all()
    for c in range(3):
        fg = result[c][x[c] > 0].double()
        assert float(fg.mean()) == pytest.approx(0, abs=1e-7)
        assert float(fg.std(correction=0)) == pytest.approx(1, abs=1e-7)
        assert stats[c]["std"] == pytest.approx(float(x[c][x[c]>0].double().std(correction=0)))
    # Population std on two values is 1, whereas the sample std is sqrt(2).
    pair = torch.tensor([0., 1., 3.]).reshape(1,1,1,3)
    assert torch.equal(mean_std_normalize_volume(pair), torch.tensor([-1.,-1.,1.]).reshape_as(pair))


def test_no_clipping_and_affine_invariance():
    x = volume()
    x[0,-1,-1,-1] = 100000
    changed = torch.where(x > 0, x*7+120, x)
    assert torch.allclose(mean_std_normalize_volume(x), mean_std_normalize_volume(changed), atol=1e-6)
    assert float(mean_std_normalize_volume(x).max()) > 10


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(value):
    x = volume(); x[0,0,0,0] = value
    with pytest.raises(ValueError, match="NaN/Inf"):
        mean_std_normalize_volume(x)


def test_empty_degenerate_and_dimensions():
    assert (mean_std_normalize_volume(torch.zeros(3,8,8,4)) == -1).all()
    with pytest.raises(ValueError, match="degenerate"):
        mean_std_normalize_volume(torch.ones(3,8,8,4))
    with pytest.raises(ValueError, match="C,H,W,Z"):
        mean_std_normalize_volume(torch.ones(3,8,8))


def test_loader_matches_offline_preprocessing_and_does_not_use_labels(tmp_path):
    x = volume()
    case = tmp_path / "subject"; case.mkdir()
    for c, m in enumerate(("flair", "t1", "t2")):
        nib.save(nib.Nifti1Image(x[c].numpy(), np.eye(4)), case / f"subject_{m}.nii.gz")
    seg = case / "subject_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8,8,4), dtype=np.uint8), np.eye(4)), seg)
    ds = MRIDataVolume(None, tmp_path, image_size=4, modalities=["flair","t1","t2"], intensity_normalization="mean_std")
    before, labels = ds[0]
    expected = Resize(4, antialias=True)(mean_std_normalize_volume(x).permute(3,0,1,2)).permute(1,2,3,0)
    assert torch.equal(before, expected)
    nib.save(nib.Nifti1Image(np.ones((8,8,4), dtype=np.uint8), np.eye(4)), seg)
    after, after_labels = ds[0]
    assert torch.equal(before, after)
    assert not torch.equal(labels, after_labels)
    with pytest.raises(ValueError, match="Histogram equalization"):
        MRIDataVolume(None, tmp_path, histogram_normalization=True, intensity_normalization="mean_std")


def test_reject_double_normalization_and_distinguish_cache():
    with pytest.raises(ValueError, match="normalize_input=false"):
        build_mri_volume_dataset({"intensity_normalization": "mean_std"})
    policy = build_postprocess_policy({})
    def fingerprint(method):
        return cache_fingerprints({"dataset_path":"unused", "normalize_input":False, "intensity_normalization":method},
                                 model_config={}, anomaly_config={}, postprocess_policy=policy)[0]
    assert fingerprint("mean_std") != fingerprint("robust_iqr")


def test_full_config_contract_and_isolated_outputs():
    pipeline = importlib.import_module("run_mixed_meanstd_histmatch200_pipeline")
    train, evaluation = pipeline.validate_configs()
    assert train["training"]["epochs"] == 200
    assert train["training"]["checkpoint"]["resume"] is None
    assert train["data"]["batch_size"] == 52
    assert train["noise"] == evaluation["noise"]
    assert "meanstd" in train["data"]["path"]
    assert "meanstd" in train["noise"]["schedule"]["sampler"]["stats_path"]
    assert evaluation["data"]["intensity_normalization"] == "mean_std"
    assert evaluation["model"]["checkpoint"].endswith("epoch_0199.pt")
    assert evaluation["model"]["use_ema"]
    assert not train["training"]["normalize_input"] and not evaluation["data"]["normalize_input"]
    assert not evaluation["prediction_output"]["enabled"]


def test_fixed_matching_agrees_with_legacy_simpleitk_defaults():
    builder = importlib.import_module("prepare_mixed_meanstd_andi_histmatch")
    original = builder.base
    mask = np.zeros((3,8,8), dtype=bool); mask[:,2:6,2:6] = True
    x = np.full(mask.shape, -1, dtype=np.float32)
    x[mask] = np.linspace(-2,4,mask.sum())
    ref = original.synthetic_references(np.tile(np.linspace(-2,3,4097),(3,1)), np.ones(3)*5)[0]
    np.testing.assert_array_equal(builder.match_volume(x, mask, ref), original.match_volume(x, mask, ref))


def test_final_checkpoint_requires_finite_weights_identity_and_steps(tmp_path, monkeypatch):
    pipeline = importlib.import_module("run_mixed_meanstd_histmatch200_pipeline")
    payload = {"epoch":199, "config":{"run_name":pipeline.NAME},
               "model":{"a":torch.ones(1)}, "ema_model":{"a":torch.ones(1)},
               "ema":{"step":324400}, "scheduler":{"last_epoch":324400}}
    path = tmp_path / "epoch_0199.pt"
    torch.save(payload, path)
    assert pipeline.verify_checkpoint(path)["finite_model_and_ema"]
    payload["model"]["a"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonfinite final model"):
        pipeline.verify_checkpoint(path)
    payload["model"]["a"][0] = 1
    payload["ema"]["step"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="step count"):
        pipeline.verify_checkpoint(path)
    payload["config"]["run_name"] = "other"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity"):
        pipeline.verify_checkpoint(path)
