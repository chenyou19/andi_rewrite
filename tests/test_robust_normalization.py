from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from torchvision.transforms import Resize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from andi_rewrite.data.imaging import normalize_volume
from andi_rewrite.data.robust_normalization import robust_normalize_volume
from andi_rewrite.data.datasets.brats import MRIDataVolume, build_mri_volume_dataset
from andi_rewrite.data.datasets.lmdb import build_lmdb_dataset
from andi_rewrite.engine.evaluation.fingerprints import cache_fingerprints
from andi_rewrite.anomaly import build_postprocess_policy

ROOT = Path(__file__).resolve().parents[1]


def volume():
    x = torch.arange(1, 3*8*8*4+1, dtype=torch.float32).reshape(3,8,8,4)
    x[:, :2] = 0
    return x


def test_foreground_is_centered_scaled_and_background_preserved():
    x = volume(); before = x.clone()
    result = robust_normalize_volume(x)
    assert torch.equal(x, before)
    assert (result[x == 0] == -1).all()
    for c in range(3):
        q = torch.quantile(result[c][x[c] > 0], torch.tensor([.25,.5,.75]))
        assert float(q[1]) == pytest.approx(0,abs=1e-6)
        assert float(q[2]-q[0]) == pytest.approx(1,abs=1e-6)


def test_affine_foreground_scaling_invariant_and_tails_not_clipped():
    x=volume(); x[0,-1,-1,-1]=100000
    mask=x>0
    shifted=torch.where(mask,x*7+120,x)
    assert torch.allclose(robust_normalize_volume(x),robust_normalize_volume(shifted),atol=2e-5)
    assert robust_normalize_volume(x).max() > 10


def test_empty_and_degenerate_and_nonfinite():
    assert (robust_normalize_volume(torch.zeros(3,8,8,4)) == -1).all()
    with pytest.raises(ValueError,match='degenerate'):
        robust_normalize_volume(torch.ones(3,8,8,4))
    x=volume(); x[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='NaN'):
        robust_normalize_volume(x)
    with pytest.raises(ValueError,match='C,H,W,Z'):
        robust_normalize_volume(torch.ones(3,8,8))


def test_p99_default_unchanged():
    x=volume(); expected=x.clone()
    for c in range(3): expected[c] /= torch.quantile(x[c][x[c]>0],.99)
    assert torch.equal(normalize_volume(x.clone()),expected)


def test_volume_adapter_matches_offline_path_and_ignores_lesion_labels(tmp_path):
    x=volume(); case=tmp_path/'subject';case.mkdir()
    for c,m in enumerate(['flair','t1','t2']):
        nib.save(nib.Nifti1Image(x[c].numpy(),np.eye(4)),case/f'subject_{m}.nii.gz')
    seg_path=case/'subject_seg.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((8,8,4),dtype=np.uint8),np.eye(4)),seg_path)
    ds=MRIDataVolume(None,tmp_path,image_size=4,modalities=['flair','t1','t2'],intensity_normalization='robust_iqr')
    before,labels=ds[0]
    expected=Resize(4,antialias=True)(robust_normalize_volume(x).permute(3,0,1,2)).permute(1,2,3,0)
    assert torch.equal(before,expected)
    nib.save(nib.Nifti1Image(np.ones((8,8,4),dtype=np.uint8),np.eye(4)),seg_path)
    after,labels_after=ds[0]
    assert torch.equal(before,after)
    assert not torch.equal(labels,labels_after)


def test_reject_accidental_second_scaling_and_wrong_lmdb(tmp_path):
    with pytest.raises(ValueError,match='normalize_input=false'):
        build_mri_volume_dataset({'intensity_normalization':'robust_iqr'})
    with pytest.raises(ValueError,match='normalization.json'):
        build_lmdb_dataset({'path':str(tmp_path),'intensity_normalization':'robust_iqr'})


def test_normalization_changes_cache_fingerprint():
    base={'dataset_path':'unused','normalize_input':False}
    policy=build_postprocess_policy({})
    def fingerprint(config):
        return cache_fingerprints(config,model_config={},anomaly_config={},postprocess_policy=policy)[0]
    assert fingerprint(base) != fingerprint({**base,'intensity_normalization':'robust_iqr'})


def test_b_train_eval_config_contract():
    train=yaml.safe_load((ROOT/'configs/train_fomo45k_robust_iqr20_b.yaml').read_text())
    evaluation=yaml.safe_load((ROOT/'configs/eval_brats_val50_robust_iqr20_b.yaml').read_text())
    assert train['training']['epochs']==20
    assert not train['training']['normalize_input']
    assert not evaluation['data']['normalize_input']
    assert train['data']['intensity_normalization']==evaluation['data']['intensity_normalization']=='robust_iqr'
    assert train['noise']==evaluation['noise']
    assert train['validation']['use_ema'] and evaluation['model']['use_ema']
    assert 'validation50' in evaluation['data']['path_to_csv']
    assert 'test' not in evaluation['data']['path_to_csv']
