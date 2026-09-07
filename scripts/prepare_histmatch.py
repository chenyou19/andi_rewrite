"""Build label-free session histogram matching against training-only BraTS tissue."""
import csv
import hashlib
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH')
BRATS = Path('C:/ML/data/BraTS_2021')
SPLITS = ROOT/'outputs/datasets/fomo45k_sri24_robust_iqr'
OUT = ROOT/'outputs/datasets/fomo45k_sri24_brats_histmatch'
WORK = ROOT/'outputs/diagnostics/fomo_brats_histmatch'
P = np.linspace(0, 1, 4097)
MODS = ['flair', 't1', 't2']


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    import os, time
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9: raise
            time.sleep(.2*(attempt+1))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landmarks(values):
    if values.size < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Expected finite positive nonempty foreground')
    q = np.quantile(values, P)
    if q[-1] <= q[0]: raise ValueError('Degenerate foreground')
    return q


def mapping(source_q, target_q):
    x, start, counts = np.unique(source_q, return_index=True, return_counts=True)
    y = np.add.reduceat(target_q, start)/counts
    if len(x) < 2 or np.any(np.diff(y) < 0) or np.any(y <= 0):
        raise ValueError('Invalid monotone mapping')
    return x, y


def transform(a, target):
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Source must be finite and nonnegative')
    result = np.zeros_like(a)
    maps = []
    for c in range(3):
        mask = a[:, c] > 0
        q = landmarks(a[:, c][mask])
        x, y = mapping(q, target[c])
        result[:, c][mask] = np.interp(a[:, c][mask], x, y)
        maps.append({'source_quantiles': q.tolist(), 'x': x.tolist(), 'y': y.tolist()})
    if not np.isfinite(result).all() or np.any(result[a > 0] <= 0):
        raise ValueError('Invalid mapped foreground')
    assert np.array_equal(result[a == 0], a[a == 0])
    return result, maps


def brats_quantiles(sid, folder):
    cache = folder/f'{sid}.npz'
    paths = [BRATS/sid/f'{sid}_{m}.nii.gz' for m in MODS+['seg']]
    signature = json.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths])
    if cache.exists():
        with np.load(cache) as d:
            if str(d['signature']) != signature: raise ValueError('Reference source changed')
            return d['q']
    seg = np.asarray(nib.load(paths[-1]).dataobj, dtype=np.float32)
    resize = Resize(128, antialias=True)
    excluded = resize(torch.from_numpy(seg > 0).permute(2,0,1).unsqueeze(1).float()).numpy()[:,0] > 0
    qq = []
    for path in paths[:3]:
        a = torch.from_numpy(np.asarray(nib.load(path).dataobj, dtype=np.float32))
        if not torch.isfinite(a).all(): raise ValueError(f'Nonfinite input {path}')
        a = a/torch.quantile(a[a > 0], .99).clamp_min(1e-8)
        a = resize(a.permute(2,0,1).unsqueeze(1)).numpy()[:,0]
        qq.append(landmarks(a[(a > 0)&~excluded]))
    q = np.stack(qq)
    with cache.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, q=q, signature=signature)
    cache.with_suffix('.tmp').replace(cache)
    return q


def cdf(values, grid):
    return np.searchsorted(np.sort(values), grid, side='right')


def distances(f, target):
    iqr = target[3072]-target[1024]
    return {'cdf_max_grid_error': float(np.max(np.abs(f-P))),
            'wasserstein_grid_over_reference_iqr': float(np.trapz(np.abs(f-P), target)/iqr)}


def figure(a, b, name):
    z = int(np.argmax((a[:,0] > 0).sum(axis=(1,2))))
    fig, ax = plt.subplots(3,2,figsize=(6,9))
    for c,m in enumerate(MODS):
        for j,v in enumerate([a,b]):
            ax[c,j].imshow(v[z,c], cmap='gray', vmin=0,vmax=1.5)
            ax[c,j].set_title(m+' '+['p99','matched'][j]); ax[c,j].axis('off')
    fig.suptitle(name+'; identical display window [0,1.5]')
    fig.tight_layout(); fig.savefig(WORK/'montages'/f'{name.replace("/","_")}.png',dpi=120); plt.close(fig)


def main():
    torch.set_num_threads(4)
    if OUT.exists(): raise FileExistsError(f'Refusing overwrite: {OUT}')
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK/'reference_cases').mkdir(exist_ok=True)
    (WORK/'montages').mkdir(exist_ok=True)
    train_ids = [r['BraTS21ID'] for r in rows(SPLITS/'brats_remaining_train.csv')]
    val_ids = [r['BraTS21ID'] for r in rows(SPLITS/'brats_validation50.csv')]
    test_ids = {r['BraTS21ID'] for r in rows(ROOT/'splits/BraTS21/scans_test.csv')}
    assert len(set(train_ids)) == 888 and len(set(val_ids)) == 50
    assert not set(train_ids)&(set(val_ids)|test_ids) and not set(val_ids)&test_ids
    original_train = {r['BraTS21ID'] for r in rows(ROOT/'splits/BraTS21/scans_train.csv')}
    assert set(train_ids)|set(val_ids) == original_train
    status = {'status':'BUILDING_REFERENCE','reference_subjects':train_ids,'validation_subjects':val_ids,
              'source_manifest_sha256':fingerprint(SOURCE/'build_manifest.json'),
              'script_sha256':fingerprint(__file__),
              'method':'4097 quantiles; equal-case mean target; per-session mapping after resize; positive foreground; no lesion coverage'}
    dump(WORK/'status.json',status)
    qgroups = {}
    for group,ids in [('reference',train_ids),('validation',val_ids)]:
        values=[]
        for i,sid in enumerate(ids):
            values.append(brats_quantiles(sid,WORK/'reference_cases'))
            print(f'{group} {i+1}/{len(ids)} {sid}',flush=True)
        qgroups[group] = np.mean(values,axis=0)
    target = qgroups['reference']
    np.savez_compressed(WORK/'reference.npz',p=P,reference=target,validation=qgroups['validation'])
    entries = {s:rows(SOURCE/f'manifests/{s}_entries.csv') for s in ['train','val']}
    assert {s:len(v) for s,v in entries.items()} == {'train':30784,'val':3369}
    assert not {r['participant_id'] for r in entries['train']}&{r['participant_id'] for r in entries['val']}
    groups = {}
    for split,rr in entries.items():
        for r in rr: groups.setdefault((split,r['case_id']),[]).append(r)
    rng = np.random.default_rng(73)
    cases = list(groups)
    pilots = [cases[i] for i in rng.choice(len(cases),12,replace=False)]
    envs = {s:lmdb.open(str(SOURCE/s),readonly=True,lock=False,readahead=True) for s in entries}
    def load_case(key):
        with envs[key[0]].begin() as t:
            return np.stack([pickle.loads(t.get(r['key'].encode())) for r in groups[key]])
    for key in pilots:
        a=load_case(key);b,_=transform(a,target);figure(a,b,key[1])
    status['status']='PILOT_PASSED_BUILDING_LMDB'; dump(WORK/'status.json',status)
    stage = OUT.with_name(OUT.name+'.staging')
    if stage.exists(): raise FileExistsError(f'Inspect existing incomplete output: {stage}')
    if shutil.disk_usage(OUT.parent).free < 10*1024**3: raise RuntimeError('Need 10 GiB free')
    stage.mkdir(); (stage/'manifests').mkdir(); (stage/'mappings').mkdir()
    accum = {s:{g:np.zeros((3,len(P)),dtype=np.int64) for g in ['before','after']} for s in entries}
    counts = {s:np.zeros(3,dtype=np.int64) for s in entries}
    output_envs={s:lmdb.open(str(stage/s),map_size=10*1024**3) for s in entries}
    try:
        for i,key in enumerate(cases):
            split,sid=key; a=load_case(key);b,maps=transform(a,target)
            dump(stage/'mappings'/f'{sid.replace("/","_")}.json',{'case_id':sid,'split':split,'modalities':MODS,'maps':maps})
            with output_envs[split].begin(write=True) as t:
                for r,v in zip(groups[key],b):
                    assert v.shape==(3,128,128)
                    if not t.put(r['key'].encode(),pickle.dumps(v,protocol=pickle.HIGHEST_PROTOCOL),overwrite=False): raise ValueError('Duplicate key')
            for c in range(3):
                mask=a[:,c]>0;counts[split][c]+=int(mask.sum())
                for g,v in [('before',a),('after',b)]:accum[split][g][c]+=cdf(v[:,c][mask],target[c])
            print(f'FOMO {i+1}/{len(cases)} {sid}',flush=True)
        result={};passed=True
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ri,s in enumerate(entries):
            result[s]={}
            assert output_envs[s].stat()['entries']==len(entries[s])
            output_envs[s].sync()
            shutil.copyfile(SOURCE/f'manifests/{s}_entries.csv',stage/f'manifests/{s}_entries.csv')
            for c,m in enumerate(MODS):
                result[s][m]={}
                for g in ['before','after']:
                    f=accum[s][g][c]/counts[s][c]
                    result[s][m][g]=distances(f,target[c])
                    axes[ri,c].plot(target[c],f,label=g)
                    if g=='after':passed &= result[s][m][g]['cdf_max_grid_error']<=.01 and result[s][m][g]['wasserstein_grid_over_reference_iqr']<=.02
                axes[ri,c].plot(target[c],P,'k--',label='BraTS reference')
                axes[ri,c].plot(qgroups['validation'][c],P,':',label='BraTS val50')
                axes[ri,c].set_title(s+' '+m);axes[ri,c].set_xlim(0,1.5);axes[ri,c].legend(fontsize=7)
            dump(stage/s/'normalization.json',{'type':'brats_histmatch','version':1,'background':0,'model_normalize_input':True,'reference_sha256':fingerprint(WORK/'reference.npz')})
        fig.tight_layout();fig.savefig(WORK/'cdf_comparison.png',dpi=150);plt.close(fig)
        np.savez_compressed(WORK/'cdf_counts.npz',p=P,reference=target,**{s+'_'+g:a for s,v in accum.items() for g,a in v.items()},**{s+'_counts':a for s,a in counts.items()})
        status.update(status='PASS' if passed else 'FAILED_ALIGNMENT',metrics=result,entries={s:len(v) for s,v in entries.items()},pilots=[k[1] for k in pilots],metric_note='CDF and Wasserstein evaluated on fixed reference quantile grid; Wasserstein uses trapezoid integration; not an exact empirical OT distance')
        dump(stage/'build_report.json',status);dump(WORK/'status.json',status)
        shutil.copyfile(WORK/'reference.npz',stage/'reference.npz')
    finally:
        for e in list(envs.values())+list(output_envs.values()): e.close()
    if passed:stage.rename(OUT)
    print(json.dumps(status['metrics'],indent=2),flush=True)
    print(status['status'],flush=True)


if __name__=='__main__':main()
