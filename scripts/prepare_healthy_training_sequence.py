"""Lock data/spectrum provenance and write the three independent training jobs."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import yaml
import lmdb

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'outputs/reports/healthy_training_sequence'
JOBS=[('mpi',60,'mpi_sri24_robust_iqr'),('oasis3',20,'oasis3_sri24_robust_iqr'),('mixed',20,'mpi_oasis3_fomo45k_sri24_robust_iqr')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'plan.json').exists():raise FileExistsError('Sequence plan already exists')
    train_base=yaml.safe_load((ROOT/'configs/train_fomo45k_robust_iqr20_b.yaml').read_text())
    eval_base=yaml.safe_load((ROOT/'configs/eval_brats_val50_robust_iqr20_b.yaml').read_text())
    split_report={}
    for dataset in ('mpi','oasis3','fomo45k'):
        d=ROOT/'outputs/datasets'/f'{dataset}_sri24_robust_iqr'
        if dataset=='fomo45k':
            sessions=json.loads((d/'build_report.json').read_text())['sessions']
            groups={k:{s['case_id'].split('/')[0] for s in sessions if s['split']==k} for k in ('train','val')}
        else:
            split=json.loads((d/'split.json').read_text());groups={k:{p for p,s in split.items() if s==k} for k in ('train','val')}
        assert not groups['train'] & groups['val']
        split_report[dataset]={k:len(v) for k,v in groups.items()}
    test_csv=ROOT/'splits/BraTS21/scans_test_50.csv'
    ids=[r['BraTS21ID'] for r in csv.DictReader(test_csv.open(encoding='utf-8-sig'))]
    assert len(ids)==len(set(ids))==50
    for sid in ids:
        for m in ('flair','t1','t2','seg'):
            assert (Path(eval_base['data']['dataset_path'])/sid/f'{sid}_{m}.nii.gz').is_file()
    jobs=[]
    for label,epochs,folder in JOBS:
        d=ROOT/'outputs/datasets'/folder
        val=d/'val' if label!='mixed' else d/'validation/val'
        spectrum=d/'spectrum'/f'{label}_train_robust_iqr_empirical_spectrum.npz'
        assert spectrum.is_file()
        counts={}
        specs=[]
        for name,path in [('train',d/'train'),('val',val)]:
            specs.append(json.loads((path/'normalization.json').read_text()))
            with lmdb.open(str(path),readonly=True,lock=False) as env:
                counts[name]=env.stat()['entries']
        assert specs[0]==specs[1]
        expected={'mpi':(14475,1686),'oasis3':(39082,4514),'mixed':(84341,9569)}[label]
        assert (counts['train'],counts['val'])==expected
        run=f'{label}_sri24_robust_iqr{epochs}_own_spectrum'
        assert not (ROOT/'outputs/runs'/run).exists()
        train=copy.deepcopy(train_base);evaluation=copy.deepcopy(eval_base)
        train['experiment']['name']=run
        train['runtime']['device']='cuda'
        train['data']['path']=str(d/'train');train['validation']['data']['path']=str(val)
        train['noise']['schedule']['sampler']['stats_path']=str(spectrum)
        train['training'].update(run_name=run,epochs=epochs,eval_after_fit={'enabled':False})
        train['training']['checkpoint'].update(start_epoch=19,save_every_epochs=20,save_last=True)
        evaluation['runtime']['device']='cuda'
        evaluation['experiment']['name']=run+'_brats21_test50'
        evaluation['noise']=copy.deepcopy(train['noise'])
        evaluation['data']['path_to_csv']=str(test_csv)
        evaluation['model']['checkpoint']=str(ROOT/'outputs/runs'/run/f'epoch_{epochs-1:04d}.pt')
        eroot=ROOT/'outputs/runs'/run/'evaluation/brats21_test50'
        evaluation['metrics']['output_csv']=str(eroot/'ANDi.csv')
        evaluation['metrics']['output_mf_csv']=str(eroot/'ANDi_mf.csv')
        evaluation['evaluation']['cache']['directory']=str(eroot/'cache')
        evaluation['prediction_output']=dict(enabled=True,directory=str(eroot/'predictions'),normalization_scope='dataset',
            binary_mask_source='score_mf',save_raw_score=True,save_median_filtered_score=True,save_binary_mask=True,
            save_threshold_mask=False,restore_native_grid=True,save_model_grid=False)
        tp=ROOT/'configs'/f'train_{run}.yaml';ep=ROOT/'configs'/f'eval_{run}_brats21_test50.yaml'
        for path,cfg in ((tp,train),(ep,evaluation)):
            if path.exists():raise FileExistsError(path)
            path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        smoke=copy.deepcopy(train);smoke['training']['run_name']=run+'_smoke'
        sp=REPORT/f'{label}_smoke.yaml';sp.write_text(yaml.safe_dump(smoke,sort_keys=False))
        jobs.append(dict(dataset=label,run_name=run,epochs=epochs,train_config=str(tp),eval_config=str(ep),smoke_config=str(sp),
            train_config_sha256=sha(tp),eval_config_sha256=sha(ep),spectrum=str(spectrum),spectrum_sha256=sha(spectrum),
            counts=counts,checkpoint=evaluation['model']['checkpoint'],evaluation_root=str(eroot)))
    plan=dict(status='PREPARED',jobs=jobs,split_integrity=split_report,test50_ids=ids,test50_sha256=sha(test_csv),batch_size=52)
    (REPORT/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
