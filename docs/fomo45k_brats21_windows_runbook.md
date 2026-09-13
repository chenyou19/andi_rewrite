# FOMO45K BraTS21 Windows runbook

This machine now has a checksum-verified preprocessing toolchain:

- ANTs `2.6.5-gfdce4d2`: `C:/ML/tools/ants-2.6.5/bin`
- FreeSurfer SynthStrip official script commit
  `cf4bccf24875a47245b4df9fc9372d0f1c3d784f` and official v1 weights
- BraTS modified SRI24 source: `C:/ML/data/atlases/brats_sri24/source_4d`
- Deterministic 3D atlas derivatives: `C:/ML/data/atlases/brats_sri24`

The Zenodo NIfTI source is `240x240x155x1`. It is preserved byte-for-byte in
`source_4d`; the pipeline uses a squeezed `240x240x155` derivative with the
same spatial qform, sform, affine, origin, direction, and 1-mm spacing. Source
and derivative hashes are in `atlas_provenance.json`.

The official SynthStrip script and weights are unchanged. Conda-forge surfa
0.6.3 has a Win64 Cython boundary that converts target shape to 32-bit C
`int`, although the binary signature requires `np.intp`. The installed wrapper
reproduces the official interpolation wrapper and changes only that internal
shape dtype cast to `np.intp`. This compatibility layer and all file hashes are
recorded in each run's `tool_versions_and_parameters.json`.

## Commands

Run preflight:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21_windows.py --preflight-only
```

Inventory all 243 sessions without registration:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21_windows.py --dry-run
```

Resume or validate the pilot case:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21_windows.py --case-id sub_10063/ses_1
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\run_validate_fomo45k_brats21_case.py C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH\sub_10063\ses_1 --dataset-root C:\ML\data\FOMO45K_SRI24_BraTS21\PT007_NIMH
```

After manual review of the pilot montage, process the complete cohort with
the exact same settings:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21_windows.py --workers 1
```

The run is resumable by default. Increase `--workers` only for between-case
parallelism; each ANTs invocation is restricted to one ITK thread inside the
pipeline. Use `--force` only when intentionally replacing an existing case
whose source/atlas/settings signature changed.

For model training, use the checked entry point. It rejects configs or
checkpoints whose channel count is not exactly `FLAIR,T1,T2 = 3`, without
editing or copying weights:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\train_fomo45k_brats21_checked.py --config configs\train_fomo45k_brats21.json --run-one-step
```

## Provenance

- BraTS preprocessing guide: https://brats.readthedocs.io/en/stable/guides/preprocessing/
- ANTs registration command anatomy: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
- ANTs Windows binaries: https://github.com/ANTsX/ANTs/wiki/Installing-ANTs-release-binaries
- SynthStrip official tool and weights: https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
- Modified SRI24 atlas record: https://zenodo.org/records/15927391
