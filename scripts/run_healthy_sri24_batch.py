"""Run both reviewed cohorts sequentially, with durable logs and final audit status."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

def now():
    return datetime.now().astimezone().isoformat()

def main():
    import msvcrt
    report_root = ROOT/'outputs/reports'
    report_root.mkdir(parents=True,exist_ok=True)
    # OS lock is released if the controller terminates; the lock file is retained.
    with (report_root/'healthy_sri24_batch.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0');lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        report = dict(status='RUNNING',pid=os.getpid(),started_at=now(),datasets={})
        path = report_root/'healthy_sri24_batch.json'
        def save():
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
            temp.replace(path)
        save()
        for dataset in ('mpi','oasis3'):
            log = report_root/f'{dataset}_sri24_full.log'
            report['datasets'][dataset] = dict(status='RUNNING',started_at=now(),log=str(log))
            save()
            with log.open('a',encoding='utf-8') as handle:
                handle.write(f'\nBatch attempt started {now()}\n');handle.flush()
                process = subprocess.Popen([sys.executable,str(ROOT/'scripts/prepare_healthy_sri24.py'),
                                         'full','--dataset',dataset],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                report['datasets'][dataset]['pid'] = process.pid
                save()
                returncode = process.wait()
            item = report['datasets'][dataset]
            item.update(status='PASS' if returncode == 0 else 'FAILED',exit_code=returncode,finished_at=now())
            build_report = ROOT/'outputs/datasets'/f'{dataset}_sri24_robust_iqr'/'build_report.json'
            if returncode == 0 and build_report.exists():
                built = json.loads(build_report.read_text())
                item.update(entries=built['entries'],participants=built['participants'],build_report=str(build_report))
            elif returncode == 0:
                item['status']='FAILED'
                item['reason']='Full command returned without a published build report'
            save()
        report.update(status='COMPLETE' if all(v['status']=='PASS' for v in report['datasets'].values()) else 'COMPLETED_WITH_FAILURES',finished_at=now())
        save()

if __name__ == '__main__':
    main()
