"""Independent read-back of every output slice and sampled distribution summaries."""
import csv,json,pickle,hashlib
from pathlib import Path
import lmdb,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/datasets/fomo45k_sri24_brats_histmatch'
WORK=ROOT/'outputs/diagnostics/fomo_brats_histmatch'
SOURCE=Path('C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH')

def main():
    if not OUT.exists(): raise RuntimeError('Published output is absent; inspect build alignment status')
    reference=np.load(OUT/'reference.npz');target=reference['reference'];val=reference['validation'];p=reference['p']
    rng=np.random.default_rng(73);result={'status':'PASS','sampling':'128 uniform spatial pixels per stored slice; positive source pixels retained; seed 73','splits':{}}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ri,split in enumerate(['train','val']):
        aenv=lmdb.open(str(SOURCE/split),readonly=True,lock=False,readahead=True)
        benv=lmdb.open(str(OUT/split),readonly=True,lock=False,readahead=True)
        assert (SOURCE/f'manifests/{split}_entries.csv').read_bytes()==(OUT/f'manifests/{split}_entries.csv').read_bytes()
        bags={g:[[],[],[]] for g in ['before','after']};count=0
        cdfs={g:np.zeros((3,len(p)),dtype=np.int64) for g in bags};pixels=np.zeros(3,dtype=np.int64)
        with aenv.begin() as ta,benv.begin() as tb:
            assert ta.stat()['entries']==tb.stat()['entries']
            for key,blob in tb.cursor():
                a=pickle.loads(ta.get(key));b=pickle.loads(blob)
                assert b.shape==a.shape==(3,128,128) and b.dtype==np.float32
                assert np.isfinite(b).all() and np.array_equal(a==0,b==0) and np.all(b[a>0]>0)
                idx=rng.choice(128*128,128,replace=False)
                for c in range(3):
                    mask=a[c]>0;pixels[c]+=mask.sum()
                    samplemask=mask.ravel()[idx]
                    for g,v in [('before',a),('after',b)]:
                        cdfs[g][c]+=np.searchsorted(np.sort(v[c][mask]),target[c],side='right')
                        bags[g][c].append(v[c].ravel()[idx][samplemask])
                count+=1
        aenv.close();benv.close();result['splits'][split]={'verified_slices':count,'modalities':{}}
        for c,m in enumerate(['FLAIR','T1','T2']):
            metrics={}
            for g in bags:
                v=np.concatenate(bags[g][c]);q=np.quantile(v,[.25,.5,.75,.95,.99]);f=cdfs[g][c]/pixels[c]
                iqr=target[c,3072]-target[c,1024]
                metrics[g]={'mean':float(v.mean()),'std':float(v.std()),'median':float(q[1]),'iqr':float(q[2]-q[0]),'p95':float(q[3]),'p99':float(q[4]),'cdf_max_grid_error':float(np.max(np.abs(f-p))),'wasserstein_grid_over_reference_iqr':float(np.trapz(np.abs(f-p),target[c])/iqr)}
                h,e=np.histogram(v,bins=np.linspace(0,3,241));axes[ri,c].stairs(h/len(v)/(e[1]-e[0]),e,label=g)
            # Quantile-curve densities are the fixed reference distributions.
            for name,qq in [('reference',target[c]),('BraTS val50',val[c])]:
                v=np.interp(np.linspace(0,1,100001),p,qq);h,e=np.histogram(v,bins=np.linspace(0,3,241));axes[ri,c].stairs(h/len(v)/(e[1]-e[0]),e,label=name,linestyle='--')
                fq=np.interp(target[c],qq,p);metrics[name]={'median':float(qq[2048]),'iqr':float(qq[3072]-qq[1024]),'p95':float(np.interp(.95,p,qq)),'p99':float(np.interp(.99,p,qq))}
                if name=='BraTS val50':
                    metrics[name]['cdf_max_grid_error_to_reference']=float(np.max(np.abs(fq-p)))
                    after_f=cdfs['after'][c]/pixels[c]
                    metrics[name]['cdf_max_grid_error_to_output']=float(np.max(np.abs(fq-after_f)))
                    metrics[name]['wasserstein_grid_to_output_over_reference_iqr']=float(np.trapz(np.abs(fq-after_f),target[c])/iqr)
            assert metrics['after']['cdf_max_grid_error']<=.01
            assert metrics['after']['wasserstein_grid_over_reference_iqr']<=.02
            result['splits'][split]['modalities'][m]=metrics
            axes[ri,c].set_title(split+' '+m);axes[ri,c].set_yscale('log');axes[ri,c].set_ylim(1e-4,30);axes[ri,c].set_xlim(0,2);axes[ri,c].legend(fontsize=7)
        print(split,count,'verified',flush=True)
    fig.tight_layout();fig.savefig(WORK/'histogram_comparison.png',dpi=150)
    (WORK/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (WORK/'distribution_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['split','modality','version','median','iqr','p95','p99'])
        for split,s in result['splits'].items():
            for mod,versions in s['modalities'].items():
                for name,v in versions.items():w.writerow([split,mod,name,*[v[k] for k in ['median','iqr','p95','p99']]])
    print('PASS',flush=True)

if __name__=='__main__':main()
