# Mixed robust-IQR dataset with ANDi histogram matching

`scripts/prepare_mixed_andi_histmatch.py` reads the existing MPI + OASIS3 +
FOMO45K mixed LMDB and writes a separate dataset at
`outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr_andi_histmatch_mean888`.
It preserves 84,341 train and 9,569 validation entries, their keys, splits,
channel order `[FLAIR,T1,T2]`, and source-entry manifests.

The target is an equal-case mean of 4,097 healthy-tissue quantiles from the
888 BraTS training subjects in `brats_remaining_train.csv`. Each BraTS volume
uses the existing per-volume robust-IQR normalization and 128-pixel resize;
pixels touched by a resized lesion mask are excluded. A synthetic 3-D image
combines the mean tissue quantiles with the median BraTS background-to-tissue
ratio. Every source case and modality is passed to the original ANDi operation,
`SimpleITK.HistogramMatching`, with its default parameters. This multi-subject
reference and the restored `-1` background are deliberate extensions to the
single-volume original method.

MPI and OASIS3 support masks come from their registered NIfTI volumes, resized
with the same antialiasing policy as the source LMDB. FOMO support masks come
from the original p99 LMDB: its train/validation manifests have identical
SHA-256 hashes to the robust-IQR FOMO manifests, and its positive pixels
recover the same resized support without needing the unavailable registered
NIfTI files. Output voxels outside support are `-1`; model input should not
apply `2*x-1` again. The output normalization metadata uses type
`andi_histmatch_mean888_robust_iqr`, so training configs should omit
`data.intensity_normalization: robust_iqr` while keeping
`training.normalize_input: false`.

Run in the ANDi conda environment from the repository root:

```powershell
python scripts/prepare_mixed_andi_histmatch.py reference
python scripts/prepare_mixed_andi_histmatch.py pilot
python scripts/prepare_mixed_andi_histmatch.py all
```

`all` also performs the earlier steps if needed. The reference is cached per
subject in a staging directory, so interrupted reference work can resume.
The final dataset is published only after six source/split pilots, complete
LMDB readback, provenance checks, and train-spectrum generation pass. The
staging directory is retained for inspection if any stage fails. The final
`build_report.json`, `pilot_report.json`, `reference_report.json`, and
`case_metrics.jsonl` record the checks and before/after quantile distances.

The train empirical spectrum is stored under `spectrum/` and uses the
background-aware mask mode. It is computed from the matched training LMDB,
not copied from the original mixed dataset.

The completed build passed the full 84,341/9,569-entry readback. The new
spectrum used 84,207 train slices and skipped 134 empty slices. Mean per-case
absolute quantile distance to the BraTS reference changed as follows (before
to after): train FLAIR `0.166 → 0.162`, T1 `0.127 → 0.264`, T2 `0.124 →
0.140`; validation FLAIR `0.167 → 0.169`, T1 `0.128 → 0.271`, T2 `0.130 →
0.139`. These are diagnostics, not a claim that the original SimpleITK
operation improves every channel; T1 in particular moved farther from the
averaged foreground quantile curve. The dataset is structurally verified, and
model performance remains unmeasured.
