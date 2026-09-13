"""Run independent training/evaluation stages sequentially with durable state."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'outputs/reports/healthy_training_sequence'

def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)

def verify_checkpoint(job):
    import torch
    p=Path(job['checkpoint'])
    payload=torch.load(p,map_location='cpu',weights_only=False)
    assert payload['epoch']==job['epochs']-1 and 'ema_model' in payload
    assert payload['config']['run_name']==job['run_name']
    assert all(torch.isfinite(v).all() for v in payload['ema_model'].values() if v.is_floating_point())
    return dict(path=str(p),sha256=sha(p),epoch=payload['epoch'])

def verify_evaluation(job,expected):
    import nibabel as nib
    root=Path(job['evaluation_root']);manifest=json.loads((root/'cache/manifest.json').read_text())
    assert manifest['collection_complete'] and manifest['subjects']==len(expected)
    assert len(manifest['entries'])==len(expected) and {s['subject_id'] for s in manifest['entries']}==set(expected)
    for f in ('ANDi.csv','ANDi_mf.csv'):
        assert (root/f).stat().st_size>0
    for sid in expected:
        d=root/'predictions'/sid
        for name in ('anomaly_score_raw.nii.gz','anomaly_score_mf.nii.gz'):
            paths=list(d.rglob(name));assert len(paths)==1,(sid,name)
            image=nib.load(paths[0]);assert image.shape==(240,240,155)
        assert list(d.rglob('lesion_mask_*.nii.gz')),sid
    return dict(subjects=len(expected),predictions=str(root/'predictions'),metrics=str(root/'ANDi.csv'))

def main():
    global REPORT
    import msvcrt
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-dir', type=Path, default=REPORT);args=parser.parse_args()
    REPORT=args.report_dir.resolve()
    plan=json.loads((REPORT/'plan.json').read_text())
    state_path=REPORT/('smoke_status.json' if args.smoke else 'status.json')
    with (REPORT/'sequence.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(stages={})
        for key in ('error', 'traceback', 'finished_at'):
            state.pop(key, None)
        state.update(status='RUNNING',pid=os.getpid(),started_at=now());save(state_path,state)
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        def run(stage,script,config,flag):
            log=REPORT/(stage+'.log')
            state['active_stage']=stage
            state['stages'][stage]=dict(status='RUNNING',started_at=now(),log=str(log));save(state_path,state)
            with log.open('a',encoding='utf-8') as handle:
                handle.write(f'\nAttempt {now()}\n');handle.flush()
                process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),'--config',str(config),flag],
                    cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                state['stages'][stage]['pid']=process.pid;save(state_path,state)
                code=process.wait()
            if code:raise RuntimeError(f'{stage} exited {code}; inspect {log}')
        try:
            if not args.smoke:
                assert json.loads((REPORT/'smoke_status.json').read_text())['status']=='COMPLETE'
            assert sha(plan.get('test_csv', ROOT/'splits/BraTS21/scans_test_50.csv'))==plan['test50_sha256']
            for job in plan['jobs']:
                for key in ('train_config','eval_config','spectrum'):
                    assert sha(job[key])==job[key+'_sha256'],f'Changed {key}'
                label=job['dataset']
                if args.smoke:
                    stage=label+'_smoke';run(stage,'train.py',job['smoke_config'],'--run-one-step')
                    state['stages'][stage].update(status='PASS',finished_at=now());save(state_path,state)
                    continue
                train_stage=label+'_train';eval_stage=label+'_eval'
                if not Path(job['checkpoint']).exists():
                    cfg=job['train_config']
                    existing=sorted((ROOT/'outputs/runs'/job['run_name']).glob('epoch_*.pt'))
                    if existing:
                        import yaml
                        config=yaml.safe_load(Path(cfg).read_text())
                        config['training']['checkpoint']['resume']=str(existing[-1])
                        cfg=REPORT/(label+'_resume.yaml');cfg.write_text(yaml.safe_dump(config,sort_keys=False))
                    run(train_stage,'train.py',cfg,'--fit')
                state['stages'].setdefault(train_stage,{})
                state['stages'][train_stage].update(status='PASS',finished_at=now(),checkpoint=verify_checkpoint(job));save(state_path,state)
                if state['stages'].get(eval_stage,{}).get('status')!='PASS':
                    run(eval_stage,'eval.py',job['eval_config'],'--run-eval')
                state['stages'].setdefault(eval_stage,{})
                state['stages'][eval_stage].update(status='PASS',finished_at=now(),outputs=verify_evaluation(job,plan['test50_ids']));save(state_path,state)
            state.update(status='COMPLETE',finished_at=now(),active_stage=None);save(state_path,state)
        except Exception as exc:
            state.update(status='FAILED',error=str(exc),traceback=traceback.format_exc(),finished_at=now());save(state_path,state)
            raise

if __name__=='__main__':main()
