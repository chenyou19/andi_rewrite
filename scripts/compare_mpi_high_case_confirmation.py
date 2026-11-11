"""Retrospective case-disjoint comparison: MPI low vs MPI/OASIS/FOMO high."""
from pathlib import Path
import json,pickle,hashlib
import numpy as np,pandas as pd,lmdb
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis'
OUT=BASE/'mpi_high_case_confirmation';OUT.mkdir(exist_ok=False)
d=pd.read_csv(BASE/'paired_slice_analysis.csv',dtype={'key':str});d['case_uid']=d.source_dataset+':'+d['case']
rng=np.random.default_rng(73);assign={}
for source,g in d.groupby('source_dataset'):
 ids=np.array(sorted(g.case_uid.unique()));rng.shuffle(ids);cut=int(len(ids)*.67)
 assign.update({v:('explore' if i<cut else 'confirm') for i,v in enumerate(ids)})
d['partition']=d.case_uid.map(assign)
pd.DataFrame([{'case_uid':k,'partition':v} for k,v in assign.items()]).to_csv(OUT/'case_partition.csv',index=False)
# Freeze protocol before computing image features. Earlier analyses used these train cases.
protocol={'seed':73,'partition':'67% exploration / remainder confirmation by source case; retrospective, not untouched holdout','low':'MPI p<0.5, support>=0.1','high':'p>=0.99, support>=0.1','matching':'abs z<=2, abs support<=0.03; unique control slices; MPI control must be from different case; all four slices same partition','sampling':'up to 600 uniformly shuffled low slices per partition; no tuning after image feature inspection','feature_mask':'raw support abs(x+1)>1e-6, eroded 2 pixels','bootstrap':'2000 case-level multinomial resamples; same MPI case weights across low/high groups; equal-case mean features, contrast low minus control; exploratory 95% intervals, no multiple-testing correction'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
allrows=[];coverage=[]
for part in ['explore','confirm']:
 f=d[d.partition==part];low=f[(f.source_dataset=='mpi')&(f.p<.5)&(f.support>=.1)]
 pools={label:f[(f.source_dataset==s)&(f.p>=.99)&(f.support>=.1)] for label,s in [('MPI_high','mpi'),('OASIS_high','oasis3'),('FOMO_high','fomo45k')]}
 used={k:set() for k in pools};n=0
 for _,r in low.sample(n=min(600,len(low)),random_state=73).iterrows():
  chosen={}
  for label,pool in pools.items():
   c=pool[(abs(pool.z-r.z)<=2)&(abs(pool.support-r.support)<=.03)&(~pool.key.isin(used[label]))&(pool.case_uid!=r.case_uid)].copy()
   if c.empty:break
   c['distance']=abs(c.z-r.z)/2+abs(c.support-r.support)/.03;chosen[label]=c.sort_values(['distance','key']).iloc[0]
  if len(chosen)!=3:continue
  allrows.append(dict(r,group='MPI_low',pair=f'{part}_{n}'))
  for label,row in chosen.items():allrows.append(dict(row,group=label,pair=f'{part}_{n}'));used[label].add(row.key)
  n+=1
 coverage.append({'partition':part,'eligible_low_slices':len(low),'attempted':min(600,len(low)),'matched_sets':n})
m=pd.DataFrame(allrows);m.to_csv(OUT/'matched_slices.csv',index=False)
assert not(set(m[m.partition=='explore'].case_uid)&set(m[m.partition=='confirm'].case_uid))
env=lmdb.open(str(ROOT/'outputs/datasets/mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888/train'),readonly=True,lock=False,readahead=False)
features=[];cache={}
with env.begin() as txn:
 for key in m.key.unique():
  x=pickle.loads(txn.get(key.encode()));cache[key]=x
  for c,mod in enumerate(['FLAIR','T1','T2']):
   a=x[c];mask=binary_erosion(abs(a+1)>1e-6,iterations=2);v=a[mask];q=np.quantile(v,[.05,.25,.5,.75,.95]);dy,dx=np.gradient(a);grad=np.hypot(dx,dy)[mask].mean()
   features.append(dict(key=key,modality=mod,p50=q[2],iqr=q[3]-q[1],p95=q[4],negative_fraction=(v<0).mean(),gradient_mean=grad,relative_gradient=grad/max(q[3]-q[1],1e-8)))
env.close();f=m.merge(pd.DataFrame(features),on='key',validate='one_to_many');f.to_csv(OUT/'features.csv',index=False)
metrics=['p50','iqr','p95','negative_fraction','gradient_mean','relative_gradient']
c=f.groupby(['partition','case_uid','group','modality'])[metrics].mean().reset_index();c.to_csv(OUT/'case_features.csv',index=False)
s=c.groupby(['partition','group','modality'])[metrics].mean();s.to_csv(OUT/'case_balanced_means.csv')
contrasts=[]
for part in ['explore','confirm']:
 frame=c[c.partition==part];ids=sorted(frame.case_uid.unique());rng=np.random.default_rng(73)
 weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
 for mod in ['FLAIR','T1','T2']:
  for metric in metrics:
   values={}
   for group in ['MPI_low','MPI_high','OASIS_high','FOMO_high']:
    v=frame[(frame.group==group)&(frame.modality==mod)].set_index('case_uid')[metric].reindex(ids).to_numpy();present=np.isfinite(v);den=weights@present.astype(float);boot=(weights@np.nan_to_num(v))/np.maximum(den,1);values[group]=(np.nanmean(v),boot)
   for a,b in [('MPI_low','MPI_high'),('MPI_high','OASIS_high'),('MPI_high','FOMO_high')]:
    mean=values[a][0]-values[b][0];boot=values[a][1]-values[b][1]
    contrasts.append(dict(partition=part,modality=mod,metric=metric,contrast=a+' minus '+b,difference=mean,ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975)))
pd.DataFrame(contrasts).to_csv(OUT/'case_bootstrap_contrasts.csv',index=False)
summary={'status':'PASS','coverage':coverage,'matched_cases':m.groupby(['partition','group']).case_uid.nunique().to_dict().__str__(),'mean_scores':m.groupby(['partition','group']).p.mean().to_dict().__str__(),'limitations':protocol['partition']}
(OUT/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(s.to_string());co=pd.DataFrame(contrasts);print(co[(co.contrast=='MPI_low minus MPI_high')&co.metric.isin(['iqr','gradient_mean','relative_gradient'])].to_string(index=False))
# Representative matched confirmation set near median low score.
l=m[(m.partition=='confirm')&(m.group=='MPI_low')];row=l.iloc[np.argmin(abs(l.p-l.p.median()))];g=m[m.pair==row.pair].set_index('group')
fig,axs=plt.subplots(4,3,figsize=(10,11))
for i,group in enumerate(['MPI_low','MPI_high','OASIS_high','FOMO_high']):
 r=g.loc[group]
 for j,mod in enumerate(['FLAIR','T1','T2']):
  ax=axs[i,j];ax.imshow(cache[r.key][j],cmap='gray',vmin=-1,vmax=2,origin='lower');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{group} {mod} p={r.p:.4f}',fontsize=10)
  if j==0:ax.set_ylabel(f'z={r.z}, support={r.support:.3f}\n{r["case"]}',fontsize=9)
fig.suptitle('Case-disjoint confirmation subset: matched example, display [-1, 2]');fig.tight_layout();fig.savefig(OUT/'confirmation_example.png',dpi=150)
fig,axs=plt.subplots(2,3,figsize=(13,7))
for i,metric in enumerate(['iqr','relative_gradient']):
 for j,mod in enumerate(['FLAIR','T1','T2']):
  a=axs[i,j]
  for part,offset in [('explore',-.17),('confirm',.17)]:
   vals=[s.loc[(part,g,mod),metric] for g in ['MPI_low','MPI_high','OASIS_high','FOMO_high']];a.bar(np.arange(4)+offset,vals,width=.34,label=part)
  a.set_xticks(range(4),['MPI low','MPI high','OAS high','FOMO high']);a.set_title(mod+' '+metric);a.grid(axis='y',alpha=.2)
axs[0,0].legend();fig.suptitle('Equal-case mean features: exploration vs retrospective confirmation');fig.tight_layout();fig.savefig(OUT/'case_feature_comparison.png',dpi=150)
