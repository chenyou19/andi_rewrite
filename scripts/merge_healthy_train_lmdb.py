"""Mix the three robust-IQR training LMDBs without changing stored values."""
import hashlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'outputs/datasets'
NAMES = ('mpi', 'oasis3', 'fomo45k')
DEST = ROOT / 'mpi_oasis3_fomo45k_sri24_robust_iqr'

def main(split='train'):
    destination = DEST if split == 'train' else DEST/'validation'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    stage = destination.with_name(destination.name + '.staging')
    if stage.exists():
        raise FileExistsError(f'Inspect existing staging before retry: {stage}')
    sources = []
    environments = []
    normalization = None
    try:
        for name in NAMES:
            folder = ROOT / f'{name}_sri24_robust_iqr' / split
            spec = json.loads((folder/'normalization.json').read_text())
            if normalization is None:
                normalization = spec
            if spec != normalization:
                raise ValueError(f'Normalization mismatch: {name}')
            env = lmdb.open(str(folder),readonly=True,lock=False,readahead=False,max_readers=4)
            environments.append(env)
            with env.begin() as txn:
                count = txn.stat()['entries']
                keys = list(txn.cursor().iternext(values=False))
                if keys != [f'{i:08d}'.encode() for i in range(count)]:
                    raise ValueError(f'Non-contiguous source keys: {name}')
            stat = (folder/'data.mdb').stat()
            sources.append(dict(name=name,path=str(folder),count=count,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        total = sum(s['count'] for s in sources)
        estimate = sum(s['size'] for s in sources)
        if shutil.disk_usage(ROOT).free < estimate*1.3 + 2*1024**3:
            raise RuntimeError('Insufficient free space')
        print(json.dumps({'sources':sources,'total':total,'seed':73}),flush=True)
        order = [(i,k) for i,s in enumerate(sources) for k in range(s['count'])]
        np.random.default_rng(73).shuffle(order)
        (stage/split).mkdir(parents=True)
        output = lmdb.open(str(stage/split),map_size=int(estimate*1.3)+1024**3)
        transactions = [env.begin() for env in environments]
        digest = hashlib.sha256()
        try:
            with (stage/'source_entries.jsonl').open('w',encoding='utf-8') as manifest:
                for start in range(0,total,256):
                    with output.begin(write=True) as txn:
                        for index in range(start,min(start+256,total)):
                            source_index, source_key = order[index]
                            value = transactions[source_index].get(f'{source_key:08d}'.encode())
                            array = pickle.loads(value)
                            if not isinstance(array,np.ndarray) or array.shape!=(3,128,128) or array.dtype!=np.float32 or not np.isfinite(array).all():
                                raise ValueError(f'Invalid source entry {source_index}:{source_key}')
                            key = f'{index:08d}'.encode()
                            if not txn.put(key,value,overwrite=False):
                                raise ValueError('Duplicate output key')
                            digest.update(key);digest.update(value)
                            manifest.write(json.dumps(dict(key=key.decode(),source_dataset=sources[source_index]['name'],source_key=f'{source_key:08d}',source_split=split))+'\n')
                    if start % 4096 == 0:
                        print(f'Written {min(start+256,total)}/{total}',flush=True)
            output.sync()
            # Read back every value and compare its exact serialized bytes to its source.
            with output.begin() as txn:
                if txn.stat()['entries'] != total:
                    raise ValueError('Output count mismatch')
                check = hashlib.sha256()
                for index,(source_index,source_key) in enumerate(order):
                    key=f'{index:08d}'.encode();value=txn.get(key)
                    if value != transactions[source_index].get(f'{source_key:08d}'.encode()):
                        raise ValueError(f'Read-back mismatch at {index}')
                    check.update(key);check.update(value)
                if check.digest()!=digest.digest():
                    raise ValueError('Read-back digest mismatch')
        finally:
            for txn in transactions:
                txn.abort()
            output.close()
        for source in sources:
            stat=(Path(source['path'])/'data.mdb').stat()
            if (stat.st_size,stat.st_mtime_ns)!=(source['size'],source['mtime_ns']):
                raise RuntimeError('Source changed during merge')
        (stage/split/'normalization.json').write_text(json.dumps(normalization,indent=2))
        report=dict(status='PASS',completed_at=datetime.now().astimezone().isoformat(),sources=sources,total_entries=total,
                    channel_order=['FLAIR','T1','T2'],shape=[3,128,128],dtype='float32',seed=73,
                    mixing=f'Deterministic shuffle of all source {split} entries; each included exactly once; no oversampling',
                    verification='Every output value read back and byte-compared with source',ordered_content_sha256=digest.hexdigest())
        (stage/'build_report.json').write_text(json.dumps(report,indent=2))
        stage.rename(destination)
        print(json.dumps(dict(status='PASS',path=str(destination/split/'data.mdb'),total_entries=total)),flush=True)
    finally:
        for env in environments:
            env.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train','val'],default='train')
    main(parser.parse_args().split)
