# BraTS inference with MPI-trained models

BraTS must be converted to the spatial/intensity contract used by the MPI/LEMON
training product before evaluation. The conversion is offline, resumable, and
fail-closed. Raw BraTS files are never modified.

## Preprocessing modes

- `mni_affine` is the default evaluation mode. It canonicalizes every subject
  to RAS, estimates a T1-to-MNI152 nonlinear 2009c asymmetric affine, applies
  that same transform to FLAIR/T1/T2 and GT, uses the exact stored MPI
  `fixed_roi.json`, and resizes XY to 128.
- `ras_fixed_fov` is the no-registration comparison. It canonicalizes to RAS
  and uses the cohort-derived 198 mm window `x=24:222, y=16:214` before the
  same XY resize.

Registration estimation uses a center-of-mass MNI-grid preview followed by a
residual affine. Final MRI volumes and GT are still sampled directly from their
canonical native grids exactly once.

## GT policy

The cache stores the categorical BraTS segmentation as `uint8` with labels
`0/1/2/4`. Spatial operations on GT always use nearest-neighbour interpolation.
GT is not normalized, filtered, histogram-matched, clipped by the brain mask,
or clipped by the MNI guard. The dataset adapter derives the evaluation target
as `segmentation > 0` only when a sample is loaded.

Cases are automatically excluded if a nonempty lesion becomes empty, if lesion
support leaves the registration target, or if any lesion is outside the final
ROI/Z support. Primary metrics compare model-grid prediction with the directly
transformed model-grid GT. Native-grid metrics, if added, must compare the
restored prediction with the original GT; GT must never be forward-then-inverse
resampled for scoring.

## Pipeline

Run from the repository root with the ANDi environment:

```powershell
python scripts/prepare_brats_mpi.py validate
python scripts/prepare_brats_mpi.py pilot
python scripts/prepare_brats_mpi.py full
python scripts/prepare_brats_mpi.py status
python scripts/prepare_brats_mpi.py manifests
```

`validate` checks all four NIfTI files per subject. `pilot` runs the configured
representative and boundary cases. `full` resumes cache entries only when their
source/config/resource fingerprint matches. A stale entry fails closed; replace
only explicitly named subjects by repeating `--overwrite-subject SUBJECT_ID`.

After `full`, run `manifests`. Evaluation configs consume generated PASS
manifests. Cross-mode comparisons must use `manifests/common_pass.csv` (or the
corresponding selection-specific common-PASS manifest), so both modes score the
same subjects.

## Cache and prediction layout

Each PASS cache subject contains:

```text
<cache_root>/<mode>/subjects/<subject_id>/
  volume.npz                 # image, categorical segmentation, brain_mask
  spatial_metadata.json      # grids, crop, transforms, fingerprint, QC
```

When both output grids are enabled, predictions are written as:

```text
<prediction_root>/<subject_id>/
  model_grid/*.nii.gz
  native_grid/*.nii.gz
  prediction_metadata.json
```

Continuous scores use linear interpolation during restoration; binary masks use
nearest-neighbour interpolation. Model, checkpoint, diffusion, noise spectrum,
anomaly aggregation, and postprocessing settings are unchanged by this data
pipeline. The native MPI checkpoint keeps the LEMON spectrum; the
MPI+BraTS-noise checkpoint keeps BraTS spectrum channels `[0,1,3]`.
