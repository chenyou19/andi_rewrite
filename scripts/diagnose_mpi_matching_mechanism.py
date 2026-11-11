"""Case-volume matching mechanism audit on the frozen 1600-slice MPI sample."""
import os,sys,json,pickle,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd,lmdb,torch,SimpleITK as sitk
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT.parent))
import compare_mixed_source_no_iqr as comp
from andi_rewrite.scripts.run_healthy_domain_classifier_inference import _load_observed_model,_predict,DEFAULT_CHECKPOINT,file_fingerprint
BASE=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis';OUT=BASE/'matching_mechanism'
OUT.mkdir(exist_ok=False)
sample=pd.read_csv(BASE/'intensity_only_experiment/sample_manifest.csv',dtype={'key':str}); sample.to_csv(OUT/'sample_manifest.csv',index=False)
source_report=json.loads((ROOT/'outputs/datasets/mpi_sri24_robust_iqr/build_report.json').read_text());stats={r['case_id']:r['statistics'] for r in source_report['sessions']}
cases=[json.loads(l) for l in (comp.OUT/'case_results_mpi.jsonl').open()];cases=[r for r in cases if r['split']=='train' and r['case_id'] in set(sample['case'])]
with np.load(comp.BASE/'reference.npz') as f:curves=f['quantiles'];ratios=f['background_ratio']
refs=comp.original.synthetic_references(curves,ratios);fgrefs=[sitk.GetImageFromArray(q.reshape(1,1,-1).astype(np.float64)) for q in curves]
model,_=_load_observed_model(DEFAULT_CHECKPOINT,torch.device('cuda')); sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
paths=[ROOT/'outputs/datasets'/n/'train' for n in ['mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888','mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888']];envs=[lmdb.open(str(p),readonly=True,lock=False,readahead=False) for p in paths]
variants=[('full_mean_on_p1',False,True,1),('full_mean_off_p1',False,False,1),('full_mean_on_p7',False,True,7),('full_mean_on_p31',False,True,31),('fg_mean_on_p1',True,True,1),('fg_mean_off_p1',True,False,1),('fg_mean_off_p7',True,False,7),('fg_mean_off_p31',True,False,31)]
records=[];thresholds=[];errors=[];start=time.time()
for ix,case in enumerate(cases):
    cid=case['case_id'];z=case['z'];sub=sample[sample['case']==cid];index={k:i for i,k in enumerate(case['mixed_keys'])};indices=[index[k] for k in sub.key]
    raw,robust,mask=comp.raw_slices([comp.original.case_paths('mpi',cid)[m] for m in comp.original.MODALITIES],stats[cid],z)
    with envs[0].begin() as t:stored=np.stack([pickle.loads(t.get(k.encode())) for k in sub.key])
    with envs[1].begin() as t:old=np.stack([pickle.loads(t.get(k.encode())) for k in sub.key])
    outputs={name:np.empty_like(stored) for name,_,_,_ in variants}
    for c,mod in enumerate(comp.original.MODALITIES):
        a=raw[:,c];b=robust[:,c];m=mask[:,c]
        for source,v in [('raw',a),('robust',b)]:
            gate=v>v.mean();boundary=m & (np.abs(v+1)>0)
            thresholds.append(dict(case=cid,modality=mod,source=source,full_mean=float(v.mean()),foreground_mean=float(v[m].mean()),foreground_above_full_mean=float(gate[m].mean()),background_above_full_mean=float(gate[~m].mean()),foreground_fraction=float(m.mean())))
        # Globally affine-transform every voxel, including background, to isolate fixed-background effects.
        affine=b.astype(np.float64)*stats[cid][c]['iqr']+stats[cid][c]['median']
        control=sitk.GetArrayFromImage(sitk.HistogramMatching(sitk.GetImageFromArray(affine),refs[c],256,1,True)).astype(np.float32);control[~m]=-1
        err=float(np.max(abs(control[indices]-old[:,c])));assert err<2e-5
        errors.append(dict(case=cid,modality=mod,affine_control_max_error=err))
        for name,fg,on,points in variants:
            src=a[m].reshape(1,1,-1) if fg else a
            target=fgrefs[c] if fg else refs[c]
            v=sitk.GetArrayFromImage(sitk.HistogramMatching(sitk.GetImageFromArray(src.astype(np.float64)),target,256,points,on)).astype(np.float32)
            if fg:
                result=np.full_like(a,-1);result[m]=v.reshape(-1)
            else:result=v;result[~m]=-1
            assert np.isfinite(result).all() and np.all(result[~m]==-1)
            outputs[name][:,c]=result[indices]
    baseline_error=float(np.max(abs(outputs['full_mean_on_p1']-stored)));assert baseline_error<2e-5
    outputs['stored_raw']=stored;outputs['stored_iqr_match']=old
    for name,value in list(outputs.items()):
        # Main test varies only FLAIR/T2; also record all-channel application.
        for scope in (['all'] if name.startswith('stored') else ['FLAIR_T2','all']):
            x=value.copy()
            if scope=='FLAIR_T2':x[:,1]=stored[:,1]
            logits,p=_predict(model,[torch.from_numpy(v) for v in x],torch.device('cuda'))
            for k,l,prob in zip(sub.key,logits,p):records.append(dict(key=k,case=cid,variant=name,scope=scope,logit=l,p=prob))
    pd.DataFrame(records).to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False);pd.DataFrame(errors).to_csv(OUT/'affine_controls.csv',index=False)
    (OUT/'status.json').write_text(json.dumps({'status':'RUNNING','cases':ix+1,'total_cases':len(cases),'baseline_error':baseline_error,'elapsed_seconds':time.time()-start}))
    print(f'CASE {ix+1}/{len(cases)} slices={len(sub)} elapsed={time.time()-start:.1f}s',flush=True)
for e in envs:e.close()
r=pd.DataFrame(records);base=r[r.variant=='stored_raw'].set_index('key');r['baseline_p']=r.key.map(base.p);r['delta_logit']=r.logit-r.key.map(base.logit)
previous=pd.read_csv(BASE/'intensity_only_experiment/predictions.csv',dtype={'key':str});pr=previous[previous.variant=='raw'].set_index('key').p;assert np.max(abs(base.p-pr))<1e-5
summaries=[]
for group,frame in [('all',r),('baseline_low',r[r.baseline_p<.5])]:
 for (variant,scope),f in frame.groupby(['variant','scope']):
  summaries.append(dict(group=group,variant=variant,scope=scope,n=len(f),mean_p=f.p.mean(),brats_fraction=(f.p>=.5).mean(),mean_delta_logit=f.delta_logit.mean()))
s=pd.DataFrame(summaries);s.to_csv(OUT/'summary.csv',index=False)
audit={'status':'PASS','cases':len(cases),'slices':len(sample),'checkpoint':file_fingerprint(DEFAULT_CHECKPOINT),'reference':file_fingerprint(comp.BASE/'reference.npz'),'SimpleITK':sitk.Version_VersionString(),'histogram_levels':256,'baseline_reproduction_tolerance':2e-5,'affine_control_max_error':max(e['affine_control_max_error'] for e in errors),'explicit_mask_method':'Pack known pre-resize foreground support values into image; reference contains frozen foreground quantiles only; restore background to -1. Keep per-case/per-modality matching.','foreground_support':'resize(raw>0)>0 as original builder','status_note':'No published dataset or model changed.'}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));(OUT/'status.json').write_text(json.dumps({'status':'PASS','elapsed_seconds':time.time()-start}));print(s.to_string(index=False))
