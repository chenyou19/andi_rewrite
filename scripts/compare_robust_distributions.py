"""Diagnostic comparison; labels are used only after normalization for summaries."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import nibabel as nib
import torch
from torchvision.transforms import Resize
from andi_rewrite.data.robust_normalization import robust_normalize_volume
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'outputs/datasets/fomo45k_sri24_robust_iqr'
OUT = ROOT/'outputs/diagnostics/robust_b/distribution'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
rng = np.random.default_rng(73)
report = json.loads((DATA/'build_report.json').read_text())
sessions = [s for s in report['sessions'] if s['split']=='train']
chosen = rng.choice(len(sessions),20,replace=False)
zs = defaultdict(list)
with (DATA/'manifests/train_entries.csv').open() as f:
    for r in csv.DictReader(f): zs[r['case_id']].append(int(r['z']))
jobs = [('FOMO', sessions[i]['case_id'], sessions[i]['input_paths'], None, zs[sessions[i]['case_id']]) for i in chosen]
with (DATA/'brats_validation50.csv').open() as f:
    for r in csv.DictReader(f):
        sid=r['BraTS21ID']; base=Path('C:/ML/data/BraTS_2021')/sid
        jobs.append(('BraTS',sid,{m:str(base/f'{sid}_{m.lower()}.nii.gz') for m in ['FLAIR','T1','T2']},base/f'{sid}_seg.nii.gz',None))
samples=defaultdict(lambda:[[],[],[]])
resize=Resize(128,antialias=True)
for i,(cohort,sid,paths,segpath,zselect) in enumerate(jobs):
    raw=torch.from_numpy(np.stack([np.asarray(nib.load(paths[m]).dataobj,dtype=np.float32) for m in ['FLAIR','T1','T2']]))
    normalized=robust_normalize_volume(raw)
    image=resize(normalized.permute(3,0,1,2)).numpy()
    core=resize((raw>0).float().permute(3,0,1,2)).numpy()>.999
    lesion=None
    if segpath is not None:
        seg=torch.from_numpy(np.asarray(nib.load(segpath).dataobj,dtype=np.float32))
        lesion=resize((seg>0).float().permute(2,0,1).unsqueeze(1)).numpy()[:,0]>0
    if zselect is not None: image=image[zselect]; core=core[zselect]
    for c in range(3):
        groups={cohort+' all':image[:,c].ravel(),cohort+' foreground':image[:,c][core[:,c]]}
        if lesion is not None: groups['BraTS non-lesion foreground']=image[:,c][core[:,c]&~lesion]
        for group,values in groups.items():
            samples[group][c].append(rng.choice(values,20000,replace=values.size<20000))
    print(f'{i+1}/{len(jobs)} {cohort} {sid}',flush=True)
result={'seed':73,'sampling':'20,000 pixels per case per modality per group; equal case weighting; FOMO 20 training sessions selected slices; BraTS 50 validation subjects all slices','foreground':'raw positive mask resized with antialiasing > 0.999; excludes mixed background boundary','non_lesion':'any positive resized lesion-mask coverage excluded; diagnostic only after label-free normalization','cases':{'FOMO':[j[1] for j in jobs if j[0]=='FOMO'],'BraTS':[j[1] for j in jobs if j[0]=='BraTS']},'groups':{}}
pooled={}
for group,channels in samples.items():
    pooled[group]=[np.concatenate(x) for x in channels]
    result['groups'][group]={}
    for c,mod in enumerate(['FLAIR','T1','T2']):
        a=pooled[group][c]; q=np.quantile(a,[.01,.25,.5,.75,.95,.99])
        result['groups'][group][mod]={'mean':float(a.mean()),'std':float(a.std()),**dict(zip(['p01','p25','median','p75','p95','p99'],q.tolist()))}
(OUT/'fomo_brats_robust_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,3,figsize=(14,8))
colors=['#2670ba','#dc8337','#299578']
for row,groups in enumerate([['FOMO all','BraTS all'],['FOMO foreground','BraTS foreground','BraTS non-lesion foreground']]):
    for c,mod in enumerate(['FLAIR','T1','T2']):
        for k,g in enumerate(groups):
            a=pooled[g][c]; h,e=np.histogram(a,bins=np.linspace(-6,10,321)); axs[row,c].stairs(h/len(a)/(e[1]-e[0]),e,label=g,color=colors[k],linestyle='--' if k==2 else '-')
        axs[row,c].set_yscale('log'); axs[row,c].set_ylim(1e-4,30); axs[row,c].set_xlim(-5,8); axs[row,c].set_title(mod+(' | including background' if row==0 else ' | foreground')); axs[row,c].set_xlabel('Robust-normalized model intensity'); axs[row,c].grid(alpha=.2)
        if c==0: axs[row,c].set_ylabel('Density (log scale)'); axs[row,c].legend(fontsize=8)
fig.suptitle('FOMO vs BraTS21: identical median/IQR preprocessing, case-balanced pixel samples')
fig.tight_layout(); fig.savefig(OUT/'fomo_brats_robust_comparison.png',dpi=160)
print(json.dumps(result['groups'],indent=2),flush=True)
