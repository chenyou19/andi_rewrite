"""Compare the three robust-IQR spectra with the existing BraTS21 reference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from compute_lmdb_spectrum import build_centered_radius_bins, radial_mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/reports/healthy_noise_spectra'
PATHS = {
    'MPI': ROOT/'outputs/datasets/mpi_sri24_robust_iqr/spectrum/mpi_train_robust_iqr_empirical_spectrum.npz',
    'OASIS3': ROOT/'outputs/datasets/oasis3_sri24_robust_iqr/spectrum/oasis3_train_robust_iqr_empirical_spectrum.npz',
    'Mixed': ROOT/'outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr/spectrum/mixed_train_robust_iqr_empirical_spectrum.npz',
    'FoMo45k': ROOT/'outputs/datasets/fomo45k_sri24_robust_iqr/spectrum/fomo45k_train_robust_iqr_empirical_spectrum.npz',
    'BraTS21 legacy': Path('C:/ML/data/spectrum/brats21_healthy_empirical_spectrum.npz'),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bins,counts=build_centered_radius_bins(128,128,64)
    radial={};mass={};provenance={};raw_power={}
    for name,path in PATHS.items():
        with np.load(path,allow_pickle=False) as s:
            power=s['mean_power'].astype(np.float64)
            if name=='BraTS21 legacy':
                power=power[[0,1,3]]
            else:
                assert s['channel_order'].tolist()==['FLAIR','T1','T2']
                assert str(s['mask_mode'])=='robust_iqr_background'
                assert int(s['radial_bins'])==64
                assert int(s['num_slices_used'])+int(s['num_slices_skipped'])=={'MPI':14475,'OASIS3':39082,'Mixed':84341,'FoMo45k':30784}[name]
            assert power.shape==(3,128,128) and np.isfinite(power).all() and (power>=0).all()
            raw_power[name]=power
            values=np.stack([radial_mean(v,bins,counts,64) for v in power])
            radial[name]=values/np.maximum((values*counts).sum(axis=1,keepdims=True),1e-30)
            mass[name]=radial[name]*counts
            provenance[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                slices_used=int(s['num_slices_used']),slices_skipped=int(s['num_slices_skipped']),
                original_radial_bins=int(s['radial_bins']),mask_mode=str(s['mask_mode']),
                channel_indices=[0,1,3] if name=='BraTS21 legacy' else [0,1,2])
            if name=='BraTS21 legacy':
                amplitude=s['mean_amplitude'][[0,1,3]]
                np.savez_compressed(OUT/'brats21_legacy_reference_rebinned64.npz',
                    mean_power=power.astype(np.float32),mean_amplitude=amplitude,
                    radial_power=values.astype(np.float32),radial_amplitude=np.stack([radial_mean(v,bins,counts,64) for v in amplitude]).astype(np.float32),
                    radial_counts=counts.astype(np.int64),channels=3,height=128,width=128,radial_bins=64,
                    channel_order=np.array(['FLAIR','T1','T2']),num_slices_used=int(s['num_slices_used']),
                    mask_mode=np.array('union_nonzero'),window=np.array('hann'),crop_margin=4)
    total_used=sum(provenance[name]['slices_used'] for name in ('MPI','OASIS3','FoMo45k'))
    assert total_used==provenance['Mixed']['slices_used']
    expected=sum(raw_power[name]*provenance[name]['slices_used'] for name in ('MPI','OASIS3','FoMo45k'))/total_used
    if not np.allclose(raw_power['Mixed'],expected,rtol=2e-6,atol=1e-9):
        raise ValueError('Mixed spectrum does not equal source-count-weighted spectra')
    mixture_error=float(np.max(np.abs(raw_power['Mixed']-expected)/np.maximum(expected,1e-30)))
    freq=(np.arange(64)+.5)*np.sqrt(.5)/64
    rows=[]
    for name,prob in mass.items():
        for c,mod in enumerate(('FLAIR','T1','T2')):
            p=prob[c];q=mass['BraTS21 legacy'][c];m=(p+q)/2
            js=.5*np.sum(p*np.log2(np.maximum(p,1e-30)/np.maximum(m,1e-30)))+.5*np.sum(q*np.log2(np.maximum(q,1e-30)/np.maximum(m,1e-30)))
            rows.append(dict(dataset=name,modality=mod,js_divergence_bits=float(js),
                cosine_similarity=float(np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))),
                centroid_cycles_per_pixel=float(p@freq),low_power_fraction_le_0p1=float(p[freq<=.1].sum()),
                high_power_fraction_ge_0p25=float(p[freq>=.25].sum())))
            f=mass['FoMo45k'][c];mid=(p+f)/2
            rows[-1]['js_vs_fomo45k_bits']=float(.5*np.sum(p*np.log2(np.maximum(p,1e-30)/np.maximum(mid,1e-30)))+.5*np.sum(f*np.log2(np.maximum(f,1e-30)/np.maximum(mid,1e-30))))
    with (OUT/'comparison_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    colors={'MPI':'#1671b7','OASIS3':'#e28324','Mixed':'#23966c','FoMo45k':'#a448b8','BraTS21 legacy':'#272b35'}
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for c,mod in enumerate(('FLAIR','T1','T2')):
        for name in PATHS:
            style='--' if name=='BraTS21 legacy' else '-'
            axs[0,c].semilogy(freq,np.maximum(radial[name][c],1e-14),style,color=colors[name],label=name,lw=1.8)
            axs[1,c].plot(freq,np.cumsum(mass[name][c]),style,color=colors[name],label=name,lw=1.8)
        axs[0,c].set_title(mod);axs[0,c].set_ylabel('Power density / total 2D power')
        axs[1,c].set_ylabel('Cumulative power fraction');axs[1,c].set_xlabel('Radial frequency (cycles / pixel)')
        for ax in axs[:,c]:ax.grid(alpha=.2);ax.set_xlim(0,np.sqrt(.5))
        axs[1,c].set_ylim(0,1.02)
    axs[0,0].legend(fontsize=9)
    fig.suptitle('MPI / OASIS3 / Mixed / FoMo45k vs existing BraTS21 reference\nFLAIR / T1 / T2; common 64 radial bins; unit total power',fontsize=14)
    fig.text(.5,.018,'Legacy BraTS21 has different intensity/background preprocessing. This is a descriptive spectral-shape comparison, not a controlled cohort-only comparison.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.92]);fig.savefig(OUT/'spectral_comparison.png',dpi=180);fig.savefig(OUT/'spectral_comparison.pdf');plt.close(fig)
    report=dict(status='PASS',sources=provenance,metrics=rows,mixture_max_relative_error=mixture_error,
        frequency_definition='Radius in 128x128 fftshift grid divided by 128; radial bin centers, max sqrt(0.5) cycles/pixel. Not cycles/mm because foreground crop is resized.',
        normalization='Each modality mean 2D power normalized to sum 1; radial energy includes radial pixel counts. No DC bin removed.',
        limitation='Existing BraTS21 spectrum uses legacy union_nonzero and original intensity preprocessing, unlike robust-IQR foreground centering/zero background. Absolute power is not compared; normalization does not remove all preprocessing effects.',
        interpretation='Smaller Jensen-Shannon divergence means more similar radial power distribution. It does not establish better anomaly detection.',
        brats_rebin='Exact radial aggregation of stored mean 2D power, not interpolation of its 128-bin radial curve.')
    (OUT/'comparison_report.json').write_text(json.dumps(report,indent=2))
    lines=['# Healthy training spectra compared with BraTS21','',report['limitation'],'',
           '| Dataset | Modality | JS vs BraTS21 (bits) | JS vs FoMo45k (bits) | High-frequency power >=0.25 |','|---|---|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['modality']} | {r['js_divergence_bits']:.5f} | {r['js_vs_fomo45k_bits']:.5f} | {r['high_power_fraction_ge_0p25']:.2%} |" for r in rows]
    lines += ['',report['frequency_definition'],'',report['normalization'],'',report['interpretation']]
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    main()
