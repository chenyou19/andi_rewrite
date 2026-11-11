from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis/matching_mechanism')
assert json.loads((out/'status.json').read_text())['status']=='PASS'
s=pd.read_csv(out/'summary.csv');t=pd.read_csv(out/'thresholds.csv')
g=t.groupby(['modality','source'])[['foreground_above_full_mean','background_above_full_mean','foreground_fraction']].mean();g.to_csv(out/'threshold_summary.csv');print(g.to_string())
r=pd.read_csv(out/'predictions.csv',dtype={'key':str});base=r[r.variant=='stored_raw'].set_index('key');r['baseline_p']=r.key.map(base.p);r['delta_logit']=r.logit-r.key.map(base.logit)
ci=[]
for (v,scope),f in r[r.baseline_p<.5].groupby(['variant','scope']):
 c=f.groupby('case').delta_logit.mean().to_numpy();rng=np.random.default_rng(73);b=rng.choice(c,size=(2000,len(c)),replace=True).mean(axis=1)
 ci.append(dict(variant=v,scope=scope,case_mean_delta_logit=c.mean(),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975)))
pd.DataFrame(ci).to_csv(out/'case_bootstrap_low.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(13,6))
order=['full_mean_on_p1','full_mean_off_p1','full_mean_on_p7','full_mean_on_p31','fg_mean_on_p1','fg_mean_off_p1','fg_mean_off_p7','fg_mean_off_p31']
for ax,group in zip(axs,['all','baseline_low']):
 for i,scope in enumerate(['FLAIR_T2','all']):
  f=s[(s.group==group)&(s.scope==scope)].set_index('variant').loc[order];ax.barh(np.arange(8)+(i-.5)*.35,f.mean_p*100,height=.35,label=scope)
 ax.set_yticks(range(8),order);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_title(group);ax.set_xlabel('Mean BraTS21 probability (%)');ax.legend();ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(out/'matching_comparison.png',dpi=150)
print(s[(s.scope=='FLAIR_T2')|s.variant.str.startswith('stored')].to_string(index=False))
