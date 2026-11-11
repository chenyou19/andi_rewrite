"""Paired MPI intensity-only diagnostic; no dataset or model is modified."""
import os,sys,json,pickle,hashlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd,lmdb,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from andi_rewrite.scripts.run_healthy_domain_classifier_inference import _load_observed_model,_predict,DEFAULT_CHECKPOINT,file_fingerprint
BASE=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis'
OUT=BASE/'intensity_only_experiment'

def transform(a,b,method):
    mask=np.abs(a+1)>1e-6
    v=a[mask].astype(np.float64); target=b[mask].astype(np.float64)
    if not len(v):return a.copy(),True
    q=np.quantile(v,[.25,.5,.75]); t=np.quantile(target,[.25,.5,.75])
    degenerate=q[2]-q[0]<=1e-8
    scale=1.0 if degenerate else (t[2]-t[0])/(q[2]-q[0])
    if method=='median':w=v+t[1]-q[1]
    elif method=='iqr':w=(v-q[1])*scale+q[1]
    elif method=='median_iqr':w=(v-q[1])*scale+t[1]
    else:
        grid=np.linspace(0,1,101)
        x=np.quantile(v,grid);y=np.quantile(target,grid)
        # Collapse tied source knots deterministically; mapping remains monotone.
        ux,inv=np.unique(x,return_inverse=True)
        uy=np.bincount(inv,weights=y)/np.bincount(inv)
        w=np.interp(v,ux,uy)
    result=a.copy();result[mask]=w.astype(np.float32)
    assert np.isfinite(result).all() and np.array_equal(result[~mask],a[~mask])
    return result,degenerate

def main():
    if OUT.exists():raise RuntimeError('Output already exists; preserve prior experiment')
    OUT.mkdir()
    d=pd.read_csv(BASE/'paired_slice_analysis.csv',dtype={'key':str})
    sample=d[d.source_dataset=='mpi'].sample(n=1600,random_state=73).sort_values('key')
    previous=pd.read_csv(BASE/'channel_swap_predictions.csv',dtype={'key':str,'iqr_channels':str})
    assert set(sample.key)==set(previous.key)
    sample.to_csv(OUT/'sample_manifest.csv',index=False)
    device=torch.device('cuda'); model,contract=_load_observed_model(DEFAULT_CHECKPOINT,device)
    paths=[ROOT/'outputs/datasets'/n/'train' for n in ['mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888','mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888']]
    stats=[file_fingerprint(p/'data.mdb',content=False) for p in paths]
    envs=[lmdb.open(str(p),readonly=True,lock=False,readahead=False) for p in paths]
    records=[];metrics=[];degenerate_count=0
    with envs[0].begin() as tr,envs[1].begin() as to:
        for start in range(0,len(sample),64):
            sub=sample.iloc[start:start+64]
            raw=np.stack([pickle.loads(tr.get(k.encode())) for k in sub.key]);old=np.stack([pickle.loads(to.get(k.encode())) for k in sub.key])
            variants={'raw':raw,'iqr_match_all':old}
            for label,channels in [('FLAIR',[0]),('T2',[2]),('FLAIR_T2',[0,2])]:
                swap=raw.copy();swap[:,channels]=old[:,channels];variants['swap_'+label]=swap
                for method in ['median','iqr','median_iqr','quantile101']:
                    x=raw.copy()
                    for i,k in enumerate(sub.key):
                        for c in channels:
                            x[i,c],deg=transform(raw[i,c],old[i,c],method);degenerate_count+=int(deg)
                            mask=np.abs(raw[i,c]+1)>1e-6
                            metrics.append(dict(key=k,method=method,channels=label,channel=c,mae_to_target=float(np.mean(abs(x[i,c][mask]-old[i,c][mask])))))
                    untouched=[c for c in range(3) if c not in channels]
                    assert np.array_equal(x[:,untouched],raw[:,untouched])
                    variants[method+'_'+label]=x
            for name,x in variants.items():
                logits,probs=_predict(model,[torch.from_numpy(v) for v in x],device)
                for k,case,z,l,p in zip(sub.key,sub['case'],sub.z,logits,probs):records.append(dict(key=k,case=case,z=int(z),variant=name,logit=l,p=p))
            print(f'INFER {start+len(sub)}/1600',flush=True)
    for e in envs:e.close()
    assert stats==[file_fingerprint(p/'data.mdb',content=False) for p in paths]
    r=pd.DataFrame(records);r.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'mapping_error.csv',index=False)
    baseline=r[r.variant=='raw'].set_index('key');r['baseline_p']=r.key.map(baseline.p);r['baseline_logit']=r.key.map(baseline.logit);r['delta_p']=r.p-r.baseline_p;r['delta_logit']=r.logit-r.baseline_logit
    summaries=[]
    for group,frame in [('all',r),('baseline_low',r[r.baseline_p<.5]),('baseline_high',r[r.baseline_p>=.99])]:
        for name,f in frame.groupby('variant'):
            case_deltas=f.groupby('case').delta_logit.mean().to_numpy();rng=np.random.default_rng(73)
            boot=np.mean(rng.choice(case_deltas,size=(2000,len(case_deltas)),replace=True),axis=1)
            summaries.append(dict(group=group,variant=name,n=len(f),cases=f['case'].nunique(),mean_p=f.p.mean(),median_p=f.p.median(),brats_fraction=(f.p>=.5).mean(),mean_delta_p=f.delta_p.mean(),mean_delta_logit=f.delta_logit.mean(),case_mean_delta_logit=case_deltas.mean(),case_delta_logit_ci_lo=np.quantile(boot,.025),case_delta_logit_ci_hi=np.quantile(boot,.975)))
    s=pd.DataFrame(summaries);s.to_csv(OUT/'summary.csv',index=False)
    # Reproduce the prior inference controls before interpreting modifications.
    controls=[]
    for variant,bits in [('raw','000'),('swap_FLAIR','100'),('swap_T2','001'),('swap_FLAIR_T2','101'),('iqr_match_all','111')]:
        a=r[r.variant==variant].set_index('key').p;b=previous[previous.iqr_channels==bits].set_index('key').p
        error=float(np.max(abs(a-b)));assert error<1e-5;controls.append({'variant':variant,'max_probability_error':error})
    audit={'status':'PASS','n':1600,'seed':73,'checkpoint':file_fingerprint(DEFAULT_CHECKPOINT),'controls':controls,'input_lmdb_stat_unchanged':True,'background_and_untouched_channels_preserved':True,'degenerate_channel_transform_events':degenerate_count,'quantiles':101,'scope':'Per-slice foreground intensity-only diagnostic against same-slice IQR+matching target; target quantiles evaluated at raw foreground positions. No spatial resampling or clipping. IQR-only scaling is centered at original median. Tied quantile knots averaged.','uncertainty':'CI uses bootstrap of per-case mean paired logit change, not independent-slice bootstrap. Exploratory within-training-set experiment; not held-out validation.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    order=['raw','median_FLAIR','iqr_FLAIR','median_iqr_FLAIR','quantile101_FLAIR','median_T2','iqr_T2','median_iqr_T2','quantile101_T2','median_FLAIR_T2','iqr_FLAIR_T2','median_iqr_FLAIR_T2','quantile101_FLAIR_T2','swap_FLAIR_T2','iqr_match_all']
    for ax,group in zip(axs,['all','baseline_low']):
        t=s[s.group==group].set_index('variant').loc[order];ax.barh(order,t.mean_p*100);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Mean BraTS21 probability (%)');ax.set_title(f'{group} (n={int(t.n.iloc[0])})');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'probability_comparison.png',dpi=150)
    print(s[['group','variant','n','mean_p','brats_fraction','mean_delta_logit']].to_string(index=False))
if __name__=='__main__':main()
