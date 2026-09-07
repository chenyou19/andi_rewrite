"""Verify rebuilt data and run one disposable GPU optimizer step before launch."""
import argparse
import csv
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import torch
from torchvision.transforms import Resize
from andi_rewrite.utils import load_config
from andi_rewrite.data import build_dataloader
from andi_rewrite.scripts.train import build_trainer_from_config
from andi_rewrite.scripts.prepare_robust_b import load_normalized_case

ROOT=Path(__file__).resolve().parents[1]


def check(gpu_step=True):
    torch.set_num_threads(4)
    train=load_config(ROOT/'configs/train_fomo45k_robust_iqr20_b.yaml')
    evaluation=load_config(ROOT/'configs/eval_brats_val50_robust_iqr20_b.yaml')
    data_root=ROOT/'outputs/datasets/fomo45k_sri24_robust_iqr'
    report=json.loads((data_root/'build_report.json').read_text())
    assert report['status']=='PASS' and not report['smoke_only']
    assert report['entries']=={'train':30784,'val':3369}
    def ids(path):
        with path.open() as f: return {r['BraTS21ID'] for r in csv.DictReader(f)}
    validation=ids(data_root/'brats_validation50.csv')
    assert len(validation)==50
    assert not validation & ids(ROOT/'splits/BraTS21/scans_test.csv')
    loader=build_dataloader({**train['data'],'shuffle':False})
    first_session=report['sessions'][0]
    volume,_=load_normalized_case(first_session['input_paths'])
    with (data_root/'manifests/train_entries.csv').open() as f: first=next(csv.DictReader(f))
    expected=Resize(128,antialias=True)(volume[:,:,:,int(first['z'])].unsqueeze(0))[0]
    assert torch.equal(loader.dataset[0],expected)
    target_loader=build_dataloader(evaluation['data'])
    target_image,_,_=next(iter(target_loader))
    assert torch.isfinite(target_image).all()
    result={'data_counts':report['entries'],'target_validation_subjects':50,'test_overlap':0,
            'fomo_source_to_lmdb_max_error':float((loader.dataset[0]-expected).abs().max()),
            'target_input_min':float(target_image.min()),'target_input_max':float(target_image.max())}
    if gpu_step:
        train['training']['run_name']='fomo45k_robust_iqr20_b_smoke'
        train['training']['eval_after_fit']={'enabled':False}
        trainer,dataloader=build_trainer_from_config(train)
        result['training_step']=trainer.run_one_step_diagnostics(next(iter(dataloader)))
    destination=ROOT/'outputs/diagnostics/robust_b'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    check(not args.cpu_only)
