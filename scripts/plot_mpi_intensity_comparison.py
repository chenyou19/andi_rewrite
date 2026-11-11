"""Render and revalidate intensity-only comparison figures from frozen experiment."""
import os,sys,json,pickle
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,lmdb,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT.parent))
from diagnose_mpi_intensity_only import transform
from andi_rewrite.scripts.run_healthy_domain_classifier_inference import _load_observed_model,_predict,DEFAULT_CHECKPOINT
font_manager.fontManager.addfont('C:/Windows/Fonts/msjh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msjh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white'})
BASE=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis/intensity_only_experiment';OUT=BASE/'comparison_figures';OUT.mkdir(exist_ok=True)
d=pd.read_csv(BASE/'predictions.csv',dtype={'key':str});w=d.pivot(index='key',columns='variant',values='p');low=w[w.raw<.5].copy();low['gain']=low.quantile101_FLAIR_T2-low.raw
key=(low.gain-low.gain.median()).abs().idxmin();fail=low[low.quantile101_FLAIR_T2<.5];fkey=(fail.quantile101_FLAIR_T2-fail.quantile101_FLAIR_T2.median()).abs().idxmin()
names=['raw','median_iqr_FLAIR_T2','quantile101_FLAIR_T2','swap_FLAIR_T2'];labels=['原始無 IQR＋matching','線性：中位數＋四分位距','非線性：101 分位數映射','對照：換回 IQR＋matching']
colors=['#697586','#d79237','#147d92','#7b69ab']
paths=[ROOT/'outputs/datasets'/n/'train' for n in ['mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888','mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888']];envs=[lmdb.open(str(p),readonly=True,lock=False,readahead=False) for p in paths]
model,_=_load_observed_model(DEFAULT_CHECKPOINT,torch.device('cuda'));audit=[]
for k,stem in [(key,'representative_slice'),(fkey,'unrecovered_slice')]:
 with envs[0].begin() as t:a=pickle.loads(t.get(k.encode()))
 with envs[1].begin() as t:b=pickle.loads(t.get(k.encode()))
 variants=[a.copy() for _ in names]
 for c in [0,2]:
  variants[1][c],_=transform(a[c],b[c],'median_iqr');variants[2][c],_=transform(a[c],b[c],'quantile101');variants[3][c]=b[c]
 logits,prob=_predict(model,[torch.from_numpy(x) for x in variants],torch.device('cuda'))
 assert np.max(abs(np.array(prob)-w.loc[k,names].to_numpy()))<1e-5
 for x in variants:assert np.array_equal(x[1],a[1])
 info=d[d.key==k].iloc[0];audit.append({'key':k,'case':info['case'],'z':int(info.z),'probabilities':dict(zip(names,prob))})
 fig,axs=plt.subplots(3,4,figsize=(15,10.7))
 for j,(name,x) in enumerate(zip(names,variants)):
  for c,mod in enumerate(['FLAIR','T1（完全相同）','T2']):
   ax=axs[c,j];im=ax.imshow(x[c],cmap='gray',vmin=-1,vmax=2,origin='lower',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if c==0:ax.set_title(labels[j]+f'\np(BraTS21) = {prob[j]*100:.4f}%',fontsize=12,color=colors[j],pad=12)
   if j==0:ax.set_ylabel(mod,fontsize=13)
 fig.suptitle(f'同一張 MPI 切片｜{info["case"]}｜z={info.z}｜key={k}',fontsize=17,y=.98)
 fig.text(.5,.025,'只處理 FLAIR、T2；T1 保持原值。所有影像固定灰階範圍 [-1, 2]，沒有各自調整亮度。\n非線性映射保留像素位置與背景；目標為同切片 IQR＋matching 的前景分布。',ha='center',fontsize=11)
 fig.subplots_adjust(top=.88,bottom=.11,left=.06,right=.92,wspace=.08,hspace=.08)
 cb=fig.add_axes([.935,.2,.012,.56]);fig.colorbar(im,cax=cb,label='模型輸入強度')
 fig.savefig(OUT/(stem+'.png'),dpi=150);plt.close(fig)
 if k==key:
  fig,axs=plt.subplots(1,2,figsize=(12,4.7))
  for ax,c,mod in zip(axs,[0,2],['FLAIR','T2']):
   mask=abs(a[c]+1)>1e-6;v=a[c][mask];target=b[c][mask];q=np.linspace(0,1,101);x=np.quantile(v,q);y=np.quantile(target,q);ux,inv=np.unique(x,return_inverse=True);uy=np.bincount(inv,weights=y)/np.bincount(inv)
   bounds=np.quantile(v,[.01,.99]);xx=np.linspace(*bounds,400);qa=np.quantile(v,[.25,.5,.75]);qb=np.quantile(target,[.25,.5,.75]);linear=(xx-qa[1])*(qb[2]-qb[0])/(qa[2]-qa[0])+qb[1]
   ax.plot(xx,xx,'--',color=colors[0],label='不變');ax.plot(xx,linear,color=colors[1],label='線性映射');ax.plot(xx,np.interp(xx,ux,uy),color=colors[2],lw=2.5,label='101 分位數映射');ax.set_title(mod);ax.set_xlabel('原始前景強度');ax.set_ylabel('映射後強度');ax.grid(alpha=.2);ax.legend()
  fig.suptitle('代表切片的強度映射曲線｜顯示原始前景第 1–99 百分位範圍',fontsize=14);fig.tight_layout();fig.savefig(OUT/'mapping_curves.png',dpi=150);plt.close(fig)
for e in envs:e.close()
fig,axs=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0];rescued=low.quantile101_FLAIR_T2>=.5
ax.scatter(low.raw[~rescued]*100,low.quantile101_FLAIR_T2[~rescued]*100,s=17,alpha=.65,color=colors[0],label=f'映射後仍低分：{int((~rescued).sum())} 張')
ax.scatter(low.raw[rescued]*100,low.quantile101_FLAIR_T2[rescued]*100,s=17,alpha=.6,color=colors[2],label=f'映射後 >= 0.5：{int(rescued.sum())} 張')
ax.plot([0,100],[0,100],'--',color='#aaaaaa');ax.axhline(50,ls=':',color='#999999');ax.set_xlim(0,50);ax.set_ylim(0,102);ax.set_xlabel('映射前 p(BraTS21)（%）');ax.set_ylabel('非線性映射後 p(BraTS21)（%）');ax.legend(fontsize=9,loc='center right');ax.set_title('每個點代表同一張切片')
ax=axs[1];means=[low[n].mean()*100 for n in names];bars=ax.barh(range(4),means,color=colors);ax.set_yticks(range(4),['原始','線性調整','非線性映射','通道替換對照']);ax.invert_yaxis();ax.set_xlim(0,106);ax.set_xlabel('平均 p(BraTS21)（%）');ax.set_title('原本低分 538 張的平均機率');ax.grid(axis='x',alpha=.15)
for bar,v in zip(bars,means):ax.text(v+1,bar.get_y()+bar.get_height()/2,f'{v:.3f}%',va='center')
fig.suptitle('只改 FLAIR＋T2 強度映射：低分 MPI 切片的分類器反應',fontsize=15);fig.tight_layout();fig.savefig(OUT/'cohort_comparison.png',dpi=160);plt.close(fig)
(OUT/'figure_audit.json').write_text(json.dumps({'status':'PASS','sample_size':len(low),'rescued':int(rescued.sum()),'selection':'Representative slice closest to median paired probability gain among all 538 baseline-low slices; supplementary failure closest to median post-map probability among unrecovered slices.','examples':audit,'inference_reproduced':True,'display_range':[-1,2]},indent=2))
print(json.dumps(audit,indent=2));print('PASS',OUT)
