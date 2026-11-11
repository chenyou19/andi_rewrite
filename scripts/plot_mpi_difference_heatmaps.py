"""Plot numerical before/after differences from original float32 tensors."""
import sys,json,pickle
from pathlib import Path
import numpy as np,lmdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_mpi_intensity_only import transform
OUT=ROOT/'outputs/diagnostics/domain_classifier/mixed_raw_histmatch_mean888_train_all84341/mpi_diagnosis/intensity_only_experiment/comparison_figures'
audit=json.loads((OUT/'figure_audit.json').read_text())
font_manager.fontManager.addfont('C:/Windows/Fonts/msjh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msjh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
envs=[lmdb.open(str(ROOT/'outputs/datasets'/n/'train'),readonly=True,lock=False,readahead=False) for n in ['mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888','mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888']]
items=[]
for info in audit['examples']:
 k=info['key']
 with envs[0].begin() as t:a=pickle.loads(t.get(k.encode()))
 with envs[1].begin() as t:b=pickle.loads(t.get(k.encode()))
 after=a.copy()
 for c in [0,2]:after[c],_=transform(a[c],b[c],'quantile101')
 delta=after.astype(np.float64)-a.astype(np.float64)
 assert np.array_equal(after[1],a[1])
 assert np.all(delta[np.abs(a+1)<=1e-6]==0)
 items.append((info,a,after,delta))
for e in envs:e.close()
limit=float(np.ceil(max(np.max(abs(v[3])) for v in items)*10)/10)
rows=[]
for idx,(info,a,after,delta) in enumerate(items):
 fig,axs=plt.subplots(3,3,figsize=(12,10))
 for c,mod in enumerate(['FLAIR','T1','T2']):
  mask=abs(a[c]+1)>1e-6;mae=float(abs(delta[c][mask]).mean());maximum=float(abs(delta[c]).max())
  rows.append(dict(key=info['key'],modality=mod,foreground_mae=mae,max_abs_difference=maximum))
  for j,v in enumerate([a[c],after[c],delta[c]]):
   ax=axs[c,j];im=ax.imshow(v,cmap='RdBu_r' if j==2 else 'gray',vmin=-limit if j==2 else -1,vmax=limit if j==2 else 2,origin='lower',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if j==0:ax.set_ylabel(mod,fontsize=14)
   if j==2:ax.set_xlabel(f'前景平均 |差值|={mae:.3f}；最大 |差值|={maximum:.3f}',fontsize=10)
   if c==0:
    title=['原始\np(BraTS21)='+f'{info["probabilities"]["raw"]*100:.2f}%','非線性映射後\np(BraTS21)='+f'{info["probabilities"]["quantile101_FLAIR_T2"]*100:.2f}%','差值＝映射後 - 原始\n紅：增加　藍：減少　白：不變'][j];ax.set_title(title,fontsize=13,pad=10)
 fig.suptitle(f'同切片差值熱圖｜{info["case"]}｜z={info["z"]}｜key={info["key"]}',fontsize=16,y=.985)
 fig.subplots_adjust(left=.055,right=.91,top=.875,bottom=.09,wspace=.06,hspace=.16)
 cb=fig.add_axes([.925,.23,.017,.50]);fig.colorbar(im,cax=cb,label='強度差（所有通道／兩個例子使用相同色階）')
 fig.text(.5,.024,f'差值直接由原始數值計算，未受灰階顯示 [-1, 2] 截限影響。差值色階 +/-{limit:.1f}，無飽和截限。\nT1 與背景差值皆為 0；本圖顯示修改位置，不是分類器重要性／注意力圖。',ha='center',fontsize=10)
 name='representative_difference_heatmap' if idx==0 else 'unrecovered_difference_heatmap';fig.savefig(OUT/(name+'.png'),dpi=160);plt.close(fig)
(OUT/'difference_statistics.json').write_text(json.dumps({'shared_difference_limit':limit,'checks':'PASS: background and T1 unchanged; float64 subtraction of original float32 values','statistics':rows},indent=2))
print(json.dumps(rows,indent=2))
