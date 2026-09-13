"""MPI/OASIS3 adapter for the existing FoMo SRI24 and robust-IQR contracts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap
bootstrap()
import lmdb
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize
from andi_rewrite.data.fomo45k import brats21 as spatial
from andi_rewrite.data.robust_normalization import ROBUST_SPEC, robust_normalize_volume

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'mpi': Path('F:/ML/DATA/MPI'), 'oasis3': Path('F:/ML/DATA/OASIS/OASIS3_healthy_447')}
ATLAS = Path('C:/ML/data/atlases/brats_sri24')
ANTS = Path('C:/ML/tools/ants-2.6.5/bin')
TOOLS = spatial.Toolchain(str(ANTS/'antsRegistration.exe'), str(ANTS/'antsApplyTransforms.exe'),
    str(ANTS/'N4BiasFieldCorrection.exe'), str(ROOT/'scripts/synthstrip_healthy.cmd'),
    str(ATLAS/'brats_sri24.nii'), str(ATLAS/'brats_sri24_skullstripped.nii'), str(ATLAS/'brats_sri24_mask.nii.gz'))

def dump(path, value):
    spatial._json_dump(path, value)

def discover(dataset, source):
    records, inventory = [], []
    visits = sorted(source.glob('sub-*/ses-*')) if dataset == 'mpi' else sorted(p for p in source.iterdir() if p.is_dir())
    for visit in visits:
        participant, session = (visit.parent.name, visit.name) if dataset == 'mpi' else visit.name.split('_MR_', 1)
        candidates = {m: [] for m in spatial.MODEL_CHANNEL_ORDER}
        for p in sorted(visit.rglob('*.nii*')):
            if dataset == 'mpi' and p.parent.name != 'anat':
                continue
            mod = spatial.classify_modality(p)
            if mod is None or (dataset == 'mpi' and mod == 'flair' and 'acq-highres_' not in p.name):
                continue
            candidates[mod].append(p)
        reasons = [f'{m}_candidates:{len(ps)}' for m, ps in candidates.items() if len(ps) != 1]
        headers = {}
        if not reasons:
            for mod, ps in candidates.items():
                try:
                    im = spatial._validate_nifti(ps[0])
                    if len(im.shape) != 3:
                        raise ValueError('Expected 3D image')
                    q, qc = im.get_qform(coded=True)
                    s, sc = im.get_sform(coded=True)
                    if not qc and not sc:
                        raise ValueError('No coded qform or sform')
                    # Compare physical corner positions, not rounded orientation labels.
                    if qc and sc:
                        corners = np.array([[x,y,z] for x in (0,im.shape[0]-1) for y in (0,im.shape[1]-1) for z in (0,im.shape[2]-1)])
                        delta = np.linalg.norm(nib.affines.apply_affine(q,corners)-nib.affines.apply_affine(s,corners),axis=1).max()
                        if delta > 0.1:
                            raise ValueError(f'qform/sform corner disagreement {delta:.4f} mm')
                    headers[mod] = dict(shape=list(im.shape), spacing=list(map(float,im.header.get_zooms())), orientation=''.join(nib.aff2axcodes(im.affine)))
                except Exception as exc:
                    reasons.append(f'{mod}_header:{exc}')
        record = spatial.SessionRecord(participant, session, f'{participant}/{session}',
            *[str(candidates[m][0]) if len(candidates[m]) == 1 else None for m in ('t1','t2','flair')],
            'EXCLUDED' if reasons else 'READY', tuple(reasons))
        records.append(record)
        inventory.append(dict(**record.as_row(), candidates={m:list(map(str,ps)) for m,ps in candidates.items()}, headers=headers))
    return records, inventory

def split_subjects(ids):
    ids = sorted(set(ids))
    if len(ids) < 2:
        raise ValueError('Need at least two PASS participants')
    np.random.default_rng(73).shuffle(ids)
    val = set(ids[:math.ceil(len(ids)*0.1)])
    return {p: 'val' if p in val else 'train' for p in sorted(ids)}

def normalized_case(base, status):
    ims = [nib.load(base/status['outputs'][m]) for m in spatial.MODEL_CHANNEL_ORDER]
    atlas = nib.load(TOOLS.atlas_t1)
    if not all(im.shape == (240,240,155) and np.allclose(im.affine,atlas.affine,atol=1e-6,rtol=0) for im in ims):
        raise ValueError('Output geometry does not match SRI24')
    data = torch.from_numpy(np.stack([np.asarray(im.dataobj,dtype=np.float32) for im in ims]))
    if any(not bool((channel>0).any()) for channel in data):
        raise ValueError('Empty modality')
    return robust_normalize_volume(data,return_statistics=True)

def audit(folder):
    entries = [json.loads(s) for s in (folder/'entries.jsonl').read_text().splitlines()]
    counts = Counter()
    subjects = {'train':set(), 'val':set()}
    for split in subjects:
        assert json.loads((folder/split/'normalization.json').read_text()) == ROBUST_SPEC
        rows = [r for r in entries if r['split'] == split]
        with lmdb.open(str(folder/split), readonly=True, lock=False) as env:
            with env.begin() as txn:
                assert txn.stat()['entries'] == len(rows)
                for index, row in enumerate(rows):
                    assert row['key'] == f'{index:08d}'
                    a = pickle.loads(txn.get(row['key'].encode()))
                    assert a.shape == (3,128,128) and a.dtype == np.float32 and np.isfinite(a).all()
                    subjects[split].add(row['participant_id'])
                    counts[split] += 1
    assert not subjects['train'] & subjects['val']
    return dict(status='PASS', entries=dict(counts), participants={k:len(v) for k,v in subjects.items()})

def build(output, records):
    if (output/'train').exists():
        return audit(output)
    passed = []; intensity_excluded = []
    for record in records:
        if record.status != 'READY':
            continue
        path = output/'volumes'/record.relative_dir/'status.json'
        if path.exists():
            s = json.loads(path.read_text())
            if s['status'] == 'PASS':
                if spatial._source_state(record) != s['source_state']:
                    raise ValueError(f'Source changed after spatial preprocessing: {record.case_id}')
                try:
                    normalized_case(path.parent,s)
                    passed.append((record,s))
                except ValueError as exc:
                    intensity_excluded.append(dict(case_id=record.case_id,reason=str(exc)))
    dump(output/'intensity_exclusions.json',dict(cases=intensity_excluded))
    split = split_subjects(r.participant_id for r,s in passed)
    stage = output/'lmdb.staging'
    if stage.exists():
        raise FileExistsError(f'Interrupted staging must be inspected: {stage}')
    estimate = len(passed)*155*3*128*128*4
    if shutil.disk_usage(output).free < estimate*1.3+2*1024**3:
        raise RuntimeError('Insufficient LMDB disk space')
    stage.mkdir()
    dump(stage/'split.json', split)
    counts = Counter(); stats = []; envs = {}
    try:
        for name in ('train','val'):
            (stage/name).mkdir()
            envs[name] = lmdb.open(str(stage/name), map_size=max(128*1024**2,int(estimate*1.3)))
            dump(stage/name/'normalization.json', ROBUST_SPEC)
        with (stage/'entries.jsonl').open('w') as manifest:
            for record, status in passed:
                base = output/'volumes'/record.relative_dir
                volume, statistics = normalized_case(base,status)
                slices = Resize(128,antialias=True)(volume.permute(3,0,1,2))
                mask = np.asarray(nib.load(base/status['brain_mask']).dataobj)>0
                name = split[record.participant_id]
                with envs[name].begin(write=True) as txn:
                    for z in np.flatnonzero(mask.any(axis=(0,1))):
                        key = f'{counts[name]:08d}'
                        value = slices[int(z)].contiguous().numpy()
                        assert txn.put(key.encode(),pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL),overwrite=False)
                        manifest.write(json.dumps(dict(key=key,split=name,case_id=record.case_id,participant_id=record.participant_id,z=int(z)))+'\n')
                        counts[name]+=1
                stats.append(dict(case_id=record.case_id,statistics=statistics))
    finally:
        for env in envs.values():
            env.sync(); env.close()
    report = audit(stage)
    dump(stage/'build_report.json',dict(**report,normalization=ROBUST_SPEC,sessions=stats))
    for path in stage.iterdir():
        path.rename(output/path.name)
    stage.rmdir()
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory','preflight','pilot','full','build','audit','status'])
    parser.add_argument('--dataset', choices=list(SOURCES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = ROOT/'outputs/datasets'/f'{args.dataset}_sri24_robust_iqr'
    output.mkdir(parents=True,exist_ok=True)
    if args.command == 'audit':
        print(json.dumps(audit(output))); return
    if args.command == 'status':
        print(json.dumps(dict(Counter(json.loads(p.read_text())['status'] for p in (output/'volumes').glob('*/*/status.json'))))); return
    records, inventory = discover(args.dataset,SOURCES[args.dataset])
    dump(output/'inventory.json',dict(records=inventory,counts=dict(Counter(r.status for r in records))))
    if args.command == 'inventory':
        print(Counter(r.status for r in records)); return
    preflight = spatial.preflight_report(TOOLS)
    dump(output/'configuration.json',dict(toolchain=asdict(TOOLS),settings=asdict(spatial.RegistrationSettings()),preflight=preflight,
        adapter_sha256=spatial.sha256_file(__file__),spatial_sha256=spatial.sha256_file(spatial.__file__),
        synthstrip_launcher_sha256=spatial.sha256_file(TOOLS.synthstrip),
        synthstrip_launcher_note='Direct pinned Python invocation; avoids BOM parsing in external python_path.txt. Same official model and compatibility entrypoint.'))
    if preflight['status'] != 'PASS':
        raise RuntimeError(preflight)
    provenance = output/'tool_versions_and_parameters.json'
    if not provenance.exists():
        subprocess.run([sys.executable,'-m','andi_rewrite.scripts.write_fomo45k_brats21_provenance',
                        '--output-root',str(output)],cwd=ROOT.parent,check=True)
    tool_info = json.loads(provenance.read_text())['tools']
    for name in ('antsRegistration','antsApplyTransforms','N4BiasFieldCorrection'):
        if spatial.sha256_file(tool_info[name]['path']) != tool_info[name]['sha256']:
            raise RuntimeError(f'Tool fingerprint changed: {name}')
    for name, info in tool_info['SynthStrip']['files'].items():
        if spatial.sha256_file(info['path']) != info['sha256']:
            raise RuntimeError(f'SynthStrip fingerprint changed: {name}')
    if (ATLAS/'atlas_provenance.json').exists() and not (output/'atlas_provenance.json').exists():
        shutil.copy2(ATLAS/'atlas_provenance.json',output/'atlas_provenance.json')
    if args.command == 'preflight':
        print(json.dumps(preflight)); return
    if args.command == 'build':
        print(json.dumps(build(output,records))); return
    ready = [r for r in records if r.status == 'READY']
    if args.command == 'pilot':
        ready = [ready[i] for i in sorted(set(np.linspace(0,len(ready)-1,min(5,len(ready)),dtype=int))) ]
        dump(output/'pilot.json',dict(cases=[r.case_id for r in ready]))
    else:
        review_path = output/'pilot_review.json'
        if not review_path.exists() or json.loads(review_path.read_text()).get('status') != 'PASS':
            raise RuntimeError('Pilot montage review must pass before full processing')
        completed_sizes = [sum(p.stat().st_size for p in s.parent.rglob('*') if p.is_file()) for s in (output/'volumes').glob('*/*/status.json')]
        estimated_bytes = max(completed_sizes, default=512*1024**2)*len(ready)*1.3 + len(ready)*155*3*128*128*4*1.3
        if shutil.disk_usage(output).free < estimated_bytes:
            raise RuntimeError(f'Insufficient free space: conservative estimate {estimated_bytes/1024**3:.1f} GiB')
    failures = []
    for i, record in enumerate(ready,1):
        print(f'{i}/{len(ready)} START {record.case_id}',flush=True)
        try:
            target = output/'volumes'/record.relative_dir/'status.json'
            if target.exists():
                old = json.loads(target.read_text())
                signature = spatial._signature(spatial._source_state(record),preflight,spatial.RegistrationSettings())
                if old['signature'] != signature:
                    raise RuntimeError('Stale spatial cache; refusing replacement')
                result = old
            else:
                result = spatial.process_session(record,output/'volumes',TOOLS,spatial.RegistrationSettings(),preflight)
            print(f"{record.case_id} {result['status']}",flush=True)
        except Exception as exc:
            failures.append(dict(case_id=record.case_id,error=str(exc),traceback=traceback.format_exc()))
            print(f'FAILED {record.case_id}: {exc}',flush=True)
        dump(output/'run_report.json',dict(command=args.command,finished=i,total=len(ready),failures=failures))
    spatial_results = []
    for record in records:
        path = output/'volumes'/record.relative_dir/'status.json'
        if path.exists():
            spatial_results.append(json.loads(path.read_text()))
    dump(output/'spatial_summary.json',dict(counts=dict(Counter(s['status'] for s in spatial_results)),cases=spatial_results))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; inspect run_report.json before building LMDB')
    if args.command == 'full' and not failures:
        print(json.dumps(build(output,records)),flush=True)

if __name__ == '__main__':
    main()
