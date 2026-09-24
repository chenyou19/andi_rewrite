"""Reproducible, read-only dataset audit of actual robust-IQR model inputs."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import Resize
from andi_rewrite.data import build_dataloader
from andi_rewrite.scripts.prepare_robust_b import load_normalized_case
from andi_rewrite.data.robust_normalization import robust_normalize_volume
from andi_rewrite.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/datasets/fomo45k_sri24_robust_iqr'
OUT = ROOT / 'outputs/diagnostics/robust_b/input_comparison_tumor_free'
MODS = ['FLAIR', 'T1', 'T2']
MIN_FOREGROUND_FRACTION = 0.10


def raw_case(paths):
    images = [nib.load(str(paths[m])) for m in MODS]
    ref = images[0]
    for im in images:
        assert im.shape == (240, 240, 155)
        np.testing.assert_allclose(im.affine, ref.affine, atol=1e-6, rtol=0)
    return torch.from_numpy(np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in images])), ref


def render_pair(pair, destination, contours=False):
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.6), facecolor='#111111')
    for row, z in enumerate(pair['z']):
        for col in range(6):
            ax = axes[row, col]
            cohort, channel = col // 3, col % 3
            a = pair['arrays'][cohort][row, channel]
            im = ax.imshow(a.T, origin='lower', cmap='gray', vmin=-1, vmax=3, interpolation='nearest')
            if contours and cohort and pair['lesion'][row].any():
                ax.contour(pair['lesion'][row].T, levels=[.5], colors=['#ff735e'], linewidths=.6, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(('FOMO ' if not cohort else 'BraTS ') + MODS[channel], color='white', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'z={z}', color='white')
    fig.suptitle(f"Pair {pair['number']:02d} | {pair['fomo']}  vs  {pair['brats']} | BraTS: zero labeled lesion voxels", color='white', fontsize=13)
    fig.subplots_adjust(left=.035, right=.94, top=.92, bottom=.065, wspace=.025, hspace=.06)
    bar = fig.colorbar(im, cax=fig.add_axes([.955, .15, .012, .65]))
    bar.ax.tick_params(colors='white'); bar.set_label('Model intensity', color='white')
    fig.text(.5, .02, '128 x 128 | median/IQR | fixed window [-1, 3] | same z is not exact anatomical correspondence', ha='center', color='white', fontsize=10)
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(fig)


def main(count=6, output_dir=OUT, include_summary=True):
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if count < 1:
        raise ValueError('count must be positive')
    train = load_config(ROOT / 'configs/train_fomo45k_robust_iqr20_b.yaml')
    evaluation = load_config(ROOT / 'configs/eval_brats_val50_robust_iqr20_b.yaml')
    assert train['training']['normalize_input'] is False
    assert evaluation['data']['normalize_input'] is False
    assert evaluation['data']['modalities'] == ['flair', 't1', 't2']
    fds = build_dataloader({**train['data'], 'shuffle': False}).dataset
    bds = build_dataloader(evaluation['data']).dataset
    report = json.loads((DATA / 'build_report.json').read_text())
    assert report['status'] == 'PASS' and not report['smoke_only']
    entries = defaultdict(dict)
    with (DATA / 'manifests/train_entries.csv').open() as f:
        for r in csv.DictReader(f):
            entries[r['case_id']][int(r['z'])] = int(r['key'])
    subjects = defaultdict(list)
    for s in report['sessions']:
        if s['split'] == 'train':
            subjects[s['case_id'].split('/')[0]].append(s)
    rng = np.random.default_rng(73)
    if count > len(subjects):
        raise ValueError(f'count={count} exceeds available FOMO subjects={len(subjects)}')
    if count > len(bds):
        raise ValueError(f'count={count} exceeds available BraTS subjects={len(bds)}')
    chosen = rng.choice(sorted(subjects), count, replace=False)
    sessions = [sorted(subjects[s], key=lambda v: v['case_id'])[0] for s in chosen]
    bchosen = rng.choice(len(bds), count, replace=False)
    resize = Resize(128, antialias=True)
    pairs, records = [], []
    samples = {g: [[] for _ in MODS] for g in ['FOMO foreground', 'BraTS non-lesion foreground']}
    for number, (session, bi) in enumerate(zip(sessions, bchosen), 1):
        sid = session['case_id']
        bv, bm, meta = bds[int(bi)]
        bpaths = {m: meta['input_paths'][m.lower()] for m in MODS}
        fr, fi = raw_case(session['input_paths'])
        br, bri = raw_case(bpaths)
        np.testing.assert_allclose(fi.affine, bri.affine, atol=1e-6, rtol=0)
        segimg = nib.load(meta['segmentation_path'])
        np.testing.assert_allclose(segimg.affine, bri.affine, atol=1e-6, rtol=0)
        assert segimg.shape == bri.shape
        seg = torch.from_numpy(np.asarray(segimg.dataobj, dtype=np.float32))
        assert torch.isfinite(seg).all()
        lesion_counts = (seg > 0).sum(dim=(0, 1))
        available = sorted(z for z in entries[sid] if 0 <= z < bv.shape[-1]
                           and lesion_counts[z] == 0
                           and bool(((br[..., z] > 0).flatten(1).float().mean(dim=1) >= MIN_FOREGROUND_FRACTION).all())
                           and bool(((fr[..., z] > 0).flatten(1).float().mean(dim=1) >= MIN_FOREGROUND_FRACTION).all()))
        assert len(available) >= 3, f'Insufficient tumor-free common slices: {sid}'
        zs = []
        for target in [50, 74, 100]:
            zs.append(min((z for z in available if z not in zs), key=lambda z: (abs(z-target), z)))
        assert len(set(zs)) == 3
        assert not bool((seg[..., zs] > 0).any())
        assert not bool(bm[..., zs].any())
        fv = torch.stack([fds[entries[sid][z]] for z in zs])
        bs = torch.stack([bv[..., z] for z in zs])
        fn, _ = load_normalized_case(session['input_paths'])
        bn = robust_normalize_volume(br)
        errors = []
        for i, z in enumerate(zs):
            for actual, expected in [(fv[i], resize(fn[..., z][None])[0]), (bs[i], resize(bn[..., z][None])[0])]:
                assert actual.shape == (3, 128, 128) and torch.isfinite(actual).all()
                errors.append(float((actual-expected).abs().max()))
                assert torch.equal(actual, expected)
        fcore = resize((fr[..., zs] > 0).float().permute(3, 0, 1, 2)).numpy() > .999
        bcore = resize((br[..., zs] > 0).float().permute(3, 0, 1, 2)).numpy() > .999
        lesion = resize((seg[..., zs] > 0).float().permute(2, 0, 1)[:, None]).numpy()[:, 0] > 0
        assert not lesion.any()
        arrays = [fv.numpy(), bs.numpy()]
        stats = {}
        for group, a, core in [('FOMO foreground', arrays[0], fcore), ('BraTS non-lesion foreground', arrays[1], bcore & ~lesion[:, None])]:
            stats[group] = {}
            for c, mod in enumerate(MODS):
                values = a[:, c][core[:, c]]
                assert values.size
                samples[group][c].append(values)
                stats[group][mod] = {'pixels': int(values.size), 'below_window_fraction': float(np.mean(values < -1)), 'above_window_fraction': float(np.mean(values > 3))}
        pair = {'number': number, 'fomo': sid, 'brats': meta['subject_id'], 'z': zs, 'arrays': arrays, 'lesion': bm[..., zs].permute(2, 0, 1).numpy()}
        pairs.append(pair)
        render_pair(pair, output_dir / f'pair_{number:02d}.png')
        records.append({'pair': number, 'fomo': sid, 'brats': meta['subject_id'], 'z': zs, 'lmdb_indices': [entries[sid][z] for z in zs], 'fomo_paths': session['input_paths'], 'brats_paths': bpaths, 'segmentation': meta['segmentation_path'], 'orientation': list(nib.aff2axcodes(fi.affine)), 'affine': fi.affine.tolist(), 'max_input_verification_error': max(errors), 'sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays], 'window_statistics': stats})
        records[-1].update({'requested_z': [50, 74, 100], 'native_lesion_voxels': [int(lesion_counts[z]) for z in zs], 'model_lesion_voxels': [int(bm[..., z].sum()) for z in zs], 'available_tumor_free_z': available})
        print(f'{number}/{count}: {sid} vs {meta["subject_id"]}; z={zs}; max error={max(errors)}', flush=True)
    if include_summary:
        fig, axes = plt.subplots(count, 6, figsize=(12, max(13, count * 2.2)), facecolor='#111111', squeeze=False)
        for row, pair in enumerate(pairs):
            for col in range(6):
                ax = axes[row, col]
                im = ax.imshow(pair['arrays'][col // 3][1, col % 3].T, origin='lower', cmap='gray', vmin=-1, vmax=3, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0: ax.set_title(('FOMO ' if col < 3 else 'BraTS ') + MODS[col % 3], color='white')
                if col == 0: ax.set_ylabel(f"Pair {row+1} / z={pair['z'][1]}", color='white')
        fig.suptitle(f'Robust IQR | {count} pairs | BraTS slices: zero labeled lesion voxels', color='white')
        fig.subplots_adjust(left=.06, right=.94, top=.94, bottom=.035, wspace=.025, hspace=.04)
        cb = fig.colorbar(im, cax=fig.add_axes([.955, .2, .012, .6])); cb.ax.tick_params(colors='white')
        fig.savefig(output_dir / 'overview.png', dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(fig)
    summary = None
    if include_summary:
        summary = {}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        bins = np.linspace(-6, 10, 321)
        for group, channels in samples.items():
            summary[group] = {}
            for c, mod in enumerate(MODS):
                values = channels[c]
                density = np.mean([np.histogram(a, bins=bins)[0] / len(a) / np.diff(bins) for a in values], axis=0)
                axes[c].stairs(density, bins, label=group)
                summary[group][mod] = {'per_case_quantiles_p01_p25_p50_p75_p95_p99': [np.quantile(a, [.01,.25,.5,.75,.95,.99]).tolist() for a in values], 'mean_case_fraction_outside_plot': float(np.mean([np.mean((a < bins[0]) | (a > bins[-1])) for a in values]))}
        for ax, mod in zip(axes, MODS):
            ax.set_title(mod); ax.set_xlabel('Model intensity'); ax.set_yscale('log'); ax.set_ylim(1e-4, 10); ax.grid(alpha=.2)
        axes[0].set_ylabel('Case-balanced density'); axes[0].legend(fontsize=8)
        fig.suptitle(f'Selected {count} pairs, 3 slices each | positive foreground | BraTS tumor-free slices')
        fig.tight_layout(); fig.savefig(output_dir / 'foreground_distribution.png', dpi=180); plt.close(fig)
    result = {'seed': 73, 'modalities': MODS, 'shape': [3,128,128], 'window': [-1,3], 'normalization': report['normalization'], 'min_foreground_fraction_per_modality': MIN_FOREGROUND_FRACTION, 'selection': f'{count} distinct FOMO training subjects and {count} distinct BraTS validation subjects; first sorted session per FOMO subject.', 'display': 'Transpose axes, origin lower, nearest interpolation; identical affine verified; no intensity changes.', 'distribution': 'Equal case weighting over displayed slices; raw positive foreground resized >0.999. BraTS selected slices have zero lesion voxels in native and model segmentation.', 'pairs': records, 'distribution_summary': summary if include_summary else None}
    result['selection'] += ' For targets 50,74,100 in order, choose nearest unused common z with zero native positive segmentation voxels and nonempty positive foreground in all modalities of both cohorts; ties choose lower z.'
    result['distribution'] = 'Equal case weighting over displayed slices; raw positive foreground resized >0.999. All BraTS selected slices have zero lesion voxels in both native and model segmentation. Full-volume normalization is unchanged and can still be influenced by tumors elsewhere in the volume.'
    (output_dir / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(output_dir, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=6, help='number of distinct FOMO/BraTS subject pairs')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--skip-summary', action='store_true', help='write only pair PNGs and comparison.json')
    args = parser.parse_args()
    main(args.count, args.output_dir, include_summary=not args.skip_summary)
