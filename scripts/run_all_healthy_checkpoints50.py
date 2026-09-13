"""Evaluate the 10 mixed checkpoints on the same 50 cases, resumably."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import yaml
from run_healthy_training_sequence import ROOT, now, save, sha, verify_checkpoint, verify_evaluation

REPORT = ROOT/'outputs/reports/mixed_checkpoints50'

def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT/'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    base = json.loads((ROOT/'outputs/reports/healthy_training_sequence/plan.json').read_text())
    test = ROOT/'splits/BraTS21/scans_test_50.csv'
    assert sha(test) == base['test50_sha256']
    jobs = []
    for source in base['jobs']:
        if source['dataset'] != 'mixed':
            continue
        runs = [source['run_name']]
        if source['dataset'] == 'mixed':
            runs.append('mixed_sri24_robust_iqr200_continue20')
        for run in runs:
            for checkpoint in sorted((ROOT/'outputs/runs'/run).glob('epoch_*.pt')):
                epoch = int(checkpoint.stem.split('_')[1])+1
                label = source['dataset']+f'_epoch{epoch:03d}'
                reuse = str(checkpoint) == source['checkpoint']
                cfg = yaml.safe_load(Path(source['eval_config']).read_text())
                out = Path(source['evaluation_root']) if reuse else checkpoint.parent/'evaluation'/f'brats21_test50_epoch{epoch:03d}'
                cfg['experiment']['name'] = label+'_brats21_test50'
                cfg['model']['checkpoint'] = str(checkpoint)
                cfg['metrics']['output_csv'] = str(out/'ANDi.csv')
                cfg['metrics']['output_mf_csv'] = str(out/'ANDi_mf.csv')
                cfg['prediction_output']['directory'] = str(out/'predictions')
                cfg['evaluation']['cache']['directory'] = str(out/'cache')
                config = Path(source['eval_config']) if reuse else REPORT/(label+'.yaml')
                if not reuse:
                    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
                job = dict(label=label, run_name=run, epochs=epoch, checkpoint=str(checkpoint), evaluation_root=str(out), config=str(config), reuse=reuse, spectrum=source['spectrum'])
                for key in ('checkpoint', 'config', 'spectrum'):
                    job[key+'_sha256'] = sha(job[key])
                verify_checkpoint(job)
                if reuse:
                    verify_evaluation(job, base['test50_ids'])
                jobs.append(job)
    assert len(jobs) == 10
    plan = dict(jobs=jobs, test_csv=str(test), test_sha256=sha(test), ids=base['test50_ids'])
    save(path, plan)
    return plan

def main():
    import msvcrt
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT/'sequence.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        plan = prepare()
        path = REPORT/'status.json'
        state = json.loads(path.read_text()) if path.exists() else dict(stages={})
        state.update(status='RUNNING', pid=os.getpid(), started_at=now())
        state.pop('error', None)
        save(path, state)
        try:
            assert sha(plan['test_csv']) == plan['test_sha256']
            for job in plan['jobs']:
                label = job['label']
                for key in ('checkpoint', 'config', 'spectrum'):
                    assert sha(job[key]) == job[key+'_sha256'], (label, key)
                if not job['reuse'] and state['stages'].get(label, {}).get('status') != 'PASS':
                    env = os.environ.copy()
                    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    log = REPORT/(label+'.log')
                    with log.open('a') as handle:
                        p = subprocess.Popen([sys.executable, str(ROOT/'scripts/eval.py'), '--config', job['config'], '--run-eval'], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                        state.update(active_stage=label)
                        state['stages'][label] = dict(status='RUNNING', pid=p.pid, log=str(log), started_at=now())
                        save(path, state)
                        code = p.wait()
                    if code:
                        raise RuntimeError(f'{label} exited {code}; see {log}')
                outputs = verify_evaluation(job, plan['ids'])
                state['stages'][label] = dict(status='PASS', reused=job['reuse'], finished_at=now(), outputs=outputs)
                save(path, state)
            rows = []
            for job in plan['jobs']:
                with (Path(job['evaluation_root'])/'inference_metrics_summary.csv').open() as f:
                    for row in csv.DictReader(f):
                        rows.append(dict(checkpoint=job['label'], **row))
            with (REPORT/'metrics_comparison.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            state.update(status='COMPLETE', active_stage=None, finished_at=now()); save(path, state)
        except Exception:
            state.update(status='FAILED', error=traceback.format_exc(), finished_at=now()); save(path, state)
            raise

if __name__ == '__main__':
    main()
