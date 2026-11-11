import os,sys,json,pickle,hashlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,lmdb,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path.cwd();sys.path.insert(0,str(ROOT.parent))
from andi_rewrite.scripts.run_healthy_domain_classifier_inference import _load_observed_model,_predict,DEFAULT_CHECKPOINT,file_fingerprint
out=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis/three_preprocessing_states';out.mkdir(exist_ok=True)
key='00026006';names=['mpi_oasis3_fomo45k_sri24_robust_iqr','mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888','mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888'];folders=['mixed_robust_iqr_train_all84341','mixed_histmatch_mean888_train_all84341','mixed_raw_histmatch_mean888_train_all84341'];labels=['只有 IQR','IQR＋matching','只有 matching（無 IQR）'];arrays=[];prior=[];manifests=[]
for name,folder in zip(names,folders):
 p=ROOT/'outputs/datasets'/name
 with (p/'source_entries.jsonl').open() as h:r=next(json.loads(l) for l in h if json.loads(l)['key']==key)
 manifests.append(r)
 env=lmdb.open(str(p/'train'),readonly=True,lock=False,readahead=False)
 with env.begin() as t:
  raw=t.get(key.encode());x=pickle.loads(raw)
 env.close();arrays.append(x)
 with (ROOT/'outputs/diagnostics/domain_classifier'/folder/'slice_predictions.jsonl').open() as h:r=next(json.loads(l) for l in h if json.loads(l)['key']==key)
 assert r['lmdb_value_sha256']==hashlib.sha256(raw).hexdigest();prior.append(r['probability_brats21_domain'])
assert manifests[0]==manifests[1]==manifests[2]
model,_=_load_observed_model(DEFAULT_CHECKPOINT,torch.device('cuda'));logits,probs=_predict(model,[torch.from_numpy(a) for a in arrays],torch.device('cuda'));assert np.max(abs(np.array(probs)-prior))<1e-5
font_manager.fontManager.addfont('C:/Windows/Fonts/msjh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msjh.ttc').get_name(),'axes.unicode_minus':False})
for tag,lo,hi in [('standard',-1,2),('wide',-3,3)]:
 fig,axs=plt.subplots(3,3,figsize=(12,10))
 for j,(label,x,p) in enumerate(zip(labels,arrays,probs)):
  for c,mod in enumerate(['FLAIR','T1','T2']):
   ax=axs[c,j];im=ax.imshow(x[c],cmap='gray',vmin=lo,vmax=hi,origin='lower',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if c==0:ax.set_title(label+f'\np(BraTS21) = {p*100:.4f}%',fontsize=13,pad=12)
   if j==0:ax.set_ylabel(mod,fontsize=14)
 fig.suptitle('同一張 MPI 切片的三種前處理\nsub-010136/ses-01 | z=100 | key=00026006',fontsize=16,y=.985)
 fig.subplots_adjust(top=.85,bottom=.075,left=.06,right=.91,wspace=.05,hspace=.07)
 cb=fig.add_axes([.925,.2,.015,.55]);fig.colorbar(im,cax=cb,label='模型輸入強度')
 fig.text(.5,.025,f'三欄皆為實際訓練 LMDB 值；所有通道共用灰階 [{lo}, {hi}]。\n每欄機率由 FLAIR＋T1＋T2 三通道共同輸入分類器得到。',ha='center',fontsize=11)
 fig.savefig(out/f'three_states_{tag}.png',dpi=160);plt.close(fig)
results=[dict(state=l,dataset=n,probability_brats21=p,probability_fomo=1-p,logit=z) for l,n,p,z in zip(labels,names,probs,logits)]
(out/'results.json').write_text(json.dumps({'status':'PASS','key':key,'source':manifests[0],'checkpoint':file_fingerprint(DEFAULT_CHECKPOINT),'results':results,'checks':'same source manifest identity; LMDB hashes match archived predictions; re-inference matches within 1e-5'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=True,indent=2))
