"""Build experiment B from source volumes, preserving original FOMO splits.

Only a new output directory is written. Source LMDBs/NIfTIs are read-only.
An interrupted staging directory is kept for diagnosis and never published.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap
bootstrap()

import lmdb
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize

from andi_rewrite.data.robust_normalization import ROBUST_SPEC, robust_normalize_volume

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH')
DEFAULT_OUTPUT = ROOT / 'outputs/datasets/fomo45k_sri24_robust_iqr'


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def load_normalized_case(paths):
    arrays, reference = [], None
    for mod in ('FLAIR', 'T1', 'T2'):
        image = nib.load(paths[mod])
        if reference is not None and (image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-6, rtol=0)):
            raise ValueError('Modality geometry mismatch')
        reference = image
        arrays.append(np.asarray(image.dataobj, dtype=np.float32))
    volume = torch.from_numpy(np.stack(arrays))
    return robust_normalize_volume(volume, return_statistics=True)


def write_brats_validation(stage):
    train = csv_rows(ROOT / 'splits/BraTS21/scans_train.csv')
    test = csv_rows(ROOT / 'splits/BraTS21/scans_test.csv')
    ids = [r['BraTS21ID'] for r in train]
    test_ids = {r['BraTS21ID'] for r in test}
    if len(ids) != len(set(ids)) or set(ids) & test_ids:
        raise ValueError('BraTS train/test identity integrity failed')
    chosen = set(np.random.default_rng(73).choice(ids, size=50, replace=False).tolist())
    for filename, selected in [('brats_validation50.csv', [s for s in ids if s in chosen]),
                               ('brats_remaining_train.csv', [s for s in ids if s not in chosen])]:
        with (stage / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle); writer.writerow(['BraTS21ID'])
            writer.writerows([[s] for s in selected])
    return {'validation_subjects': 50, 'remaining_train_subjects': len(ids)-50,
            'test_overlap': 0, 'seed': 73,
            'purpose': 'Target-domain validation only; no BraTS model training or distribution fitting.'}


def build(output, max_sessions=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be a NEW directory inside this workspace outputs/')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    sessions_path = SOURCE / 'manifests/sessions.jsonl'
    sessions = [json.loads(line) for line in sessions_path.read_text(encoding='utf-8').splitlines()]
    entries = {}
    for split in ('train','val'):
        entries[split] = csv_rows(SOURCE / f'manifests/{split}_entries.csv')
    by_case = defaultdict(list)
    for split, rows in entries.items():
        for row in rows:
            if row['split'] != split:
                raise ValueError('Source split metadata mismatch')
            by_case[row['case_id']].append(row)
    participants = {split: {r['participant_id'] for r in rows} for split,rows in entries.items()}
    if participants['train'] & participants['val']:
        raise ValueError('FOMO subject leakage')
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    expected = {split: sum(len(by_case[s['case_id']]) for s in sessions if s['split']==split) for split in entries}
    estimate = sum(expected.values()) * 3 * 128 * 128 * 4
    if shutil.disk_usage(output.parent).free < estimate * 1.3 + 2*1024**3:
        raise RuntimeError('Insufficient free space for the new dataset')
    stage = output.with_name(output.name + '.staging-' + uuid.uuid4().hex[:8])
    stage.mkdir()
    (stage / 'manifests').mkdir()
    environments, handles, writers = {}, {}, {}
    counts = {'train': 0, 'val': 0}
    result = {'status': 'BUILDING', 'normalization': ROBUST_SPEC, 'source': str(SOURCE),
              'source_sessions_sha256': hashlib.sha256(sessions_path.read_bytes()).hexdigest(),
              'started_at': datetime.now().astimezone().isoformat(), 'sessions': [],
              'expected_entries': expected, 'smoke_only': max_sessions is not None}
    try:
        for split in counts:
            folder = stage / split; folder.mkdir()
            environments[split] = lmdb.open(str(folder), map_size=max(128*1024**2, int(expected[split]*3*128*128*4*1.3)))
            handles[split] = (stage / f'manifests/{split}_entries.csv').open('w', encoding='utf-8', newline='')
            writers[split] = csv.DictWriter(handles[split],fieldnames=list(entries[split][0]))
            writers[split].writeheader()
        for index, session in enumerate(sessions, 1):
            split = session['split']
            volume, statistics = load_normalized_case(session['input_paths'])
            slices = Resize(128, antialias=True)(volume.permute(3,0,1,2))
            if tuple(slices.shape[1:]) != (3,128,128):
                raise ValueError('Expected [Z,3,128,128] after resizing')
            with environments[split].begin(write=True) as txn:
                for source_row in by_case[session['case_id']]:
                    value = slices[int(source_row['z'])].contiguous().numpy()
                    key = f'{counts[split]:08d}'
                    if not txn.put(key.encode(), pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL),overwrite=False):
                        raise ValueError('Duplicate LMDB key')
                    writers[split].writerow({**source_row,'key':key})
                    counts[split] += 1
            result['sessions'].append({'case_id':session['case_id'],'split':split,
                                       'statistics':statistics,'input_paths':session['input_paths']})
            print(f'{index}/{len(sessions)} {session["case_id"]} {split} {counts}',flush=True)
        for split,env in environments.items():
            with env.begin(write=False) as txn:
                if txn.stat()['entries'] != expected[split] or counts[split] != expected[split]:
                    raise ValueError('LMDB count mismatch')
            env.sync()
            dump(stage / split / 'normalization.json',ROBUST_SPEC)
        result['brats_split'] = write_brats_validation(stage)
        result.update(status='PASS', entries=counts, finished_at=datetime.now().astimezone().isoformat())
        dump(stage / 'build_report.json', result)
    except Exception as exc:
        result.update(status='FAILED', error=str(exc))
        dump(stage / 'build_report.json', result)
        raise
    finally:
        for env in environments.values(): env.close()
        for handle in handles.values(): handle.close()
    stage.rename(output)
    print(f'Published {output}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--max-sessions',type=int)
    args=parser.parse_args()
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error('--max-sessions must be positive')
    torch.set_num_threads(4)
    build(args.output,args.max_sessions)
