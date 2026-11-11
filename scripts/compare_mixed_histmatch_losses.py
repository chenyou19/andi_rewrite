from pathlib import Path
import json
import pandas as pd,numpy as np,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager
root=Path.cwd();out=root/'outputs/diagnostics/mixed_raw_vs_iqr_histmatch_loss200';out.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msjh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msjh.ttc').get_name(),'axes.unicode_minus':False})
names={'raw':'mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh','iqr':'mixed_sri24_robust_iqr_andi_histmatch_mean888_200ep_fresh'}
frames={};summary={};configs={}
for k,n in names.items():
 p=root/'outputs/runs'/n;d=pd.read_csv(p/'training_metrics.csv');assert d.epoch_index.is_unique and set(d.epoch_index)==set(range(200));assert np.isfinite(d[['train_loss','validation_loss']]).all().all();d=d.sort_values('epoch_index');frames[k]=d
 last=d.iloc[-1];best=d.loc[d.validation_loss.idxmin()];summary[k]={'final_train':last.train_loss,'final_val':last.validation_loss,'final_gap':last.validation_loss-last.train_loss,'best_val':best.validation_loss,'best_val_completed_epoch':int(best.completed_epoch),'last20_train':d.tail(20).train_loss.mean(),'last20_val':d.tail(20).validation_loss.mean()};configs[k]=yaml.safe_load((p/'config_snapshot.yaml').read_text())
summary['raw_reduction_percent']={key:100*(1-summary['raw'][key]/summary['iqr'][key]) for key in ['final_train','final_val','last20_train','last20_val']}
def diffs(a,b,path=''):
 result=[]
 for k in a.keys()|b.keys():
  x,y=a.get(k),b.get(k);s=path+'.'+k if path else k
  if isinstance(x,dict) and isinstance(y,dict):result+=diffs(x,y,s)
  elif x!=y:result.append({'field':s,'raw':x,'iqr':y})
 return result
summary['config_differences']=diffs(configs['raw'],configs['iqr']);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(12,8))
for j,(metric,title) in enumerate([('train_loss','Train loss'),('validation_loss','Validation loss（EMA）')]):
 for k,label,color in [('raw','無 IQR＋matching','#157f91'),('iqr','IQR＋matching','#ce7937')]:
  d=frames[k];axs[0,j].plot(d.completed_epoch,d[metric],color=color,label=label,lw=1.6);tail=d[d.completed_epoch>=100];axs[1,j].plot(tail.completed_epoch,tail[metric],color=color,label=label,lw=1.6)
 axs[0,j].set_title(title+'｜1–200 epochs');axs[0,j].set_yscale('log');axs[0,j].yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'));axs[0,j].yaxis.set_minor_formatter(FuncFormatter(lambda v,pos: f'{v:g}'));axs[0,j].set_ylabel('Loss（log scale）');axs[0,j].legend()
 axs[1,j].set_title(title+'｜後半段放大');axs[1,j].set_ylabel('Loss')
 for i in range(2):axs[i,j].set_xlabel('完成 epoch 數（1-based）');axs[i,j].grid(alpha=.2)
fig.suptitle('Mixed：無 IQR＋matching vs IQR＋matching',fontsize=16);fig.tight_layout();fig.savefig(out/'loss_comparison.png',dpi=160)
joined=frames['raw'][['completed_epoch','train_loss','validation_loss']].merge(frames['iqr'][['completed_epoch','train_loss','validation_loss']],on='completed_epoch',suffixes=('_raw','_iqr'));joined.to_csv(out/'paired_losses.csv',index=False)
print(json.dumps(summary,indent=2))
