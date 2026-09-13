"""Deterministic bucket-balanced resampling without modifying image bytes."""
import csv,hashlib,json,pickle,shutil
from collections import Counter,defaultdict
from pathlib import Path
import lmdb,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/datasets/fomo45k_sri24_brats_histmatch'
OUT=SOURCE.with_name(SOURCE.name+'_zbucket29_69k')
REF=ROOT/'outputs/diagnostics/brats_healthy_train_z_distribution/slice_counts_by_z.csv'
BINS=[(i,i+4) for i in range(0,140,5)]+[(140,145)]

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def allocate(weights,total):
    q=total*np.asarray(weights,dtype=float)/sum(weights)
    n=np.floor(q).astype(int)
    order=np.lexsort((np.arange(len(q)),-(q-n)))
    n[order[:total-int(n.sum())]]+=1
    return n

def draw(indices,n,rng):
    if not indices:raise ValueError('Empty source bucket')
    copies,remainder=divmod(int(n),len(indices))
    return list(indices)*copies+rng.choice(indices,remainder,replace=False).tolist()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    stage=OUT.with_name(OUT.name+'.staging')
    if stage.exists():raise FileExistsError(stage)
    records=rows(SOURCE/'manifests/train_entries.csv');validation=rows(SOURCE/'manifests/val_entries.csv')
    assert len(records)==30784 and len(validation)==3369
    assert not {r['participant_id'] for r in records}&{r['participant_id'] for r in validation}
    reference={int(r['z_index']):int(r['slice_count']) for r in rows(REF)}
    weights=[sum(reference[z] for z in range(a,b+1)) for a,b in BINS]
    quota=allocate(weights,69000);rng=np.random.default_rng(73)
    buckets=[[] for _ in BINS]
    for i,r in enumerate(records):
        z=int(r['z']);assert 0<=z<=145;buckets[min(z//5,28)].append(i)
    selected=[]
    for bucket,(ids,n) in enumerate(zip(buckets,quota)):
        selected.extend((i,bucket) for i in draw(ids,n,rng))
    rng.shuffle(selected)
    assert len(selected)==69000
    if shutil.disk_usage(OUT.parent).free<17*1024**3:raise RuntimeError('Need 17 GiB free')
    stage.mkdir();(stage/'manifests').mkdir()
    report={'status':'BUILDING','seed':73,'source':str(SOURCE),'reference':str(REF),'source_manifest_sha256':sha(SOURCE/'manifests/train_entries.csv'),'reference_sha256':sha(REF),'script_sha256':sha(__file__),'bins':BINS,'quotas':quota.tolist()}
    (stage/'build_report.json').write_text(json.dumps(report,indent=2))
    source=lmdb.open(str(SOURCE/'train'),readonly=True,lock=False,readahead=True)
    output=lmdb.open(str(stage/'train'),map_size=16*1024**3)
    copies=Counter();zcounts=Counter();bucketcounts=Counter()
    fieldnames=list(records[0])+['source_key','bucket_id','duplicate_sequence']
    with source.begin() as ts,(stage/'manifests/train_entries.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for start in range(0,len(selected),512):
            with output.begin(write=True) as t:
                for index,(i,bucket) in enumerate(selected[start:start+512],start):
                    r=records[i];key=f'{index:08d}';blob=ts.get(r['key'].encode());assert blob is not None
                    assert t.put(key.encode(),blob,overwrite=False)
                    copies[r['key']]+=1;zcounts[int(r['z'])]+=1;bucketcounts[bucket]+=1
                    w.writerow({**r,'key':key,'source_key':r['key'],'bucket_id':bucket,'duplicate_sequence':copies[r['key']]})
            if start%5120==0:print('written',min(start+512,69000),flush=True)
    output.sync();output.close();source.close()
    shutil.copytree(SOURCE/'val',stage/'val')
    shutil.copyfile(SOURCE/'manifests/val_entries.csv',stage/'manifests/val_entries.csv')
    shutil.copyfile(SOURCE/'train/normalization.json',stage/'train/normalization.json')
    shutil.copyfile(SOURCE/'reference.npz',stage/'reference.npz')
    # Independent byte-for-byte verification of every selected train entry and all validation entries.
    newrows=rows(stage/'manifests/train_entries.csv')
    for split in ['train','val']:
        a=lmdb.open(str(SOURCE/split),readonly=True,lock=False,readahead=True)
        b=lmdb.open(str(stage/split),readonly=True,lock=False,readahead=True)
        rr=newrows if split=='train' else validation
        with a.begin() as ta,b.begin() as tb:
            assert tb.stat()['entries']==len(rr)
            for r in rr:
                assert tb.get(r['key'].encode())==ta.get(r.get('source_key',r['key']).encode())
        a.close();b.close();print('verified',split,len(rr),flush=True)
    assert [bucketcounts[i] for i in range(29)]==quota.tolist()
    assert max(copies.values())==7 and len(copies)==28309
    with (stage/'bucket_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['bucket','z_start','z_end','source_count','reference_count','target_count','actual_count'])
        for i,(a,b) in enumerate(BINS):w.writerow([i,a,b,len(buckets[i]),weights[i],int(quota[i]),bucketcounts[i]])
    with (stage/'z_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['z','slice_count'])
        for z in range(146):w.writerow([z,zcounts[z]])
    with (stage/'repetition_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['source_key','case_id','z','uses'])
        for r in records:w.writerow([r['key'],r['case_id'],r['z'],copies[r['key']]])
    fig,ax=plt.subplots(2,1,figsize=(12,7))
    ax[0].bar(range(29),[bucketcounts[i] for i in range(29)],label='actual')
    ax[0].plot(range(29),quota,'o-',color='orange',label='target');ax[0].legend();ax[0].set_xlabel('Bucket index');ax[0].set_ylabel('Slice count')
    ax[1].bar(range(146),[zcounts[z] for z in range(146)]);ax[1].set_xlabel('Original z index');ax[1].set_ylabel('Slice count')
    fig.suptitle('FOMO matched: 69,000 slices; 29 mixed-z buckets');fig.tight_layout();fig.savefig(stage/'z_distribution.png',dpi=150);plt.close(fig)
    # Read each original slice once and weight by multiplicity; sampling uses uniform spatial pixels.
    rng=np.random.default_rng(73);bags=[[] for _ in range(3)];multiplicity=[[] for _ in range(3)]
    env=lmdb.open(str(SOURCE/'train'),readonly=True,lock=False,readahead=True)
    with env.begin() as t:
        for r in records:
            a=pickle.loads(t.get(r['key'].encode())).reshape(3,-1);idx=rng.choice(a.shape[1],128,replace=False)
            for c in range(3):
                v=a[c,idx];v=v[v>0];bags[c].append(v);multiplicity[c].append(np.full(len(v),copies[r['key']],dtype=np.int16))
    env.close();ref=np.load(SOURCE/'reference.npz');target=ref['reference'];p=ref['p'];summary={}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for c,m in enumerate(['FLAIR','T1','T2']):
        v=np.concatenate(bags[c]);mult=np.concatenate(multiplicity[c]).astype(np.int64);order=np.argsort(v);v=v[order];mult=mult[order];summary[m]={}
        for label,w in [('source',np.ones(len(v))),('resampled',mult)]:
            cum=np.cumsum(w);cum=cum/cum[-1];q=np.interp([.25,.5,.75,.95,.99],cum,v)
            pos=np.searchsorted(v,target[c],side='right');cdf=np.where(pos>0,cum[np.maximum(pos-1,0)],0)
            summary[m][label]={'median':float(q[1]),'iqr':float(q[2]-q[0]),'p95':float(q[3]),'p99':float(q[4]),'cdf_grid_error_to_reference':float(np.max(np.abs(cdf-p)))}
            h,e=np.histogram(v,bins=np.linspace(0,2,161),weights=w);axes[c].stairs(h/w.sum()/(e[1]-e[0]),e,label=label)
        rv=np.interp(np.linspace(0,1,100001),p,target[c]);h,e=np.histogram(rv,bins=np.linspace(0,2,161));axes[c].stairs(h/len(rv)/(e[1]-e[0]),e,label='BraTS reference',linestyle='--')
        axes[c].set_title(m);axes[c].set_yscale('log');axes[c].set_ylim(1e-4,30);axes[c].legend(fontsize=8)
    fig.tight_layout();fig.savefig(stage/'brightness_distribution.png',dpi=150);plt.close(fig)
    report.update(status='PASS',train_entries=69000,val_entries=3369,unique_source_slices=len(copies),max_uses=max(copies.values()),repetition_histogram=dict(Counter(copies.get(r['key'],0) for r in records)),byte_verification=True,brightness_method='128 uniformly sampled spatial pixels per original slice, positive foreground only, weighted by exact multiplicity; seed 73',brightness=summary)
    (stage/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');stage.rename(OUT)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
