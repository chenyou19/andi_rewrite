# FOMO45K BraTS21/SRI24 preprocessing

This pipeline publishes the PT007_NIMH T1, T2, and FLAIR volumes on the
BraTS21 SRI24 1-mm grid. It estimates a 3-D rigid+affine transform from T1,
applies the same T1-to-SRI24 transform to aligned modalities, uses one extra
modality-to-T1 rigid transform only when native grids differ, and never creates
a synthetic T1ce channel.

## Required external resources

The pipeline fails closed unless all of these are available:

- `antsRegistration`, `antsApplyTransforms`, and `N4BiasFieldCorrection`
- FreeSurfer `mri_synthstrip`
- `C:/ML/data/atlases/brats_sri24/brats_sri24.nii`
- `C:/ML/data/atlases/brats_sri24/brats_sri24_skullstripped.nii`
- `C:/ML/data/atlases/brats_sri24/brats_sri24_mask.nii.gz`

The three atlas images must share one affine, have shape `240x240x155`, and
have 1-mm isotropic spacing. Pin and record the atlas source/version; do not
substitute a generic MNI152 image under these filenames.

## Commands

Inventory all 243 sessions without registration:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21.py --dry-run
```

Check dependencies and atlas geometry:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21.py --preflight-only
```

Validate one real session before the full cohort:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21.py --case-id sub_10063/ses_1
```

After reviewing its `qc/montage.png` and `qc/metrics.json`, run the complete
cohort with the same settings:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\prepare_fomo45k_brats21.py --workers 1
```

Use `--workers` for between-case parallelism only. Each ANTs process is pinned
to one ITK thread for reproducibility. `--force` is required to replace an
existing case with a different input/configuration signature.

## ANDi adapter

`FOMO45KBraTS21VolumeDataset` returns named `flair`, `t1`, `t2`, and
`brain_mask` tensors plus a `FLAIR,T1,T2` stack matching BraTS21 inference.
The slice adapter performs the
same per-modality z-score only inside the shared brain mask, extracts eligible
axial slices, and resizes them to the current 128x128 2-D ANDi input boundary.
It does not reorient or resample physical MRI geometry.

Run training through the optional-builder wrapper so the shared factory is not
changed for existing experiments:

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\train_fomo45k_brats21.py --config configs\train_fomo45k_brats21.json --run-one-step
```

The supplied config sets `training.normalize_input=false` because the adapter
already returns z-scored data; applying the historical `[0,1] -> [-1,1]`
mapping would be incorrect.
