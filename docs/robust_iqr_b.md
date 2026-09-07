# B: FOMO robust normalization pilot

This experiment trains on FOMO only for 20 epochs and then evaluates the final EMA checkpoint on 50 BraTS validation subjects. No BraTS fine-tuning or cohort histogram matching is performed.

## Normalization

For each modality of each complete skull-stripped 3D input volume, select finite input voxels greater than zero, calculate their median and interquartile range (Q75 - Q25), and transform foreground values as `(x - median) / IQR`. Background is -1 in model units. Do not clip the high-intensity tail. Compute these statistics before resizing or selecting slices, without segmentation labels. Empty modalities remain background; nonempty modalities with negligible IQR and nonfinite input fail validation.

Both source LMDB preparation and target volume loading call `data.robust_normalization.robust_normalize_volume`. Disable the usual model-input `2*x-1` transform because robust outputs already use model units. LMDB normalization sidecars prevent an existing p99 database from being used accidentally.

## Controlled choices and data

- Preserve the original FOMO participant split and selected slices: 30,784 training and 3,369 validation slices across 243 sessions.
- Preserve the original FOMO empirical noise spectrum, architecture, and batch size 52. Do not recompute the noise spectrum for this pilot.
- Use seed 73 to choose 50 subjects from the original BraTS training split. Check that these subjects are disjoint from the 251-subject test split. The generated CSV records the exact selection.
- Evaluate source validation loss with EMA, matching target inference. This differs from the older run's source validation setting, so its validation MSE is not directly comparable.
- Train 592 updates per epoch, 11,840 updates total. The 20-epoch cosine schedule has 592 warmup updates. It is a pilot schedule, not the first 20 epochs of the older 233-epoch schedule.

## Reproduce

Run from the repository root with the ANDi Python environment:

```powershell
python scripts/prepare_robust_b.py
python scripts/check_robust_b.py
.\scripts\launch_training_hidden.ps1 -Config configs/train_fomo45k_robust_iqr20_b.yaml -RunName fomo45k_robust_iqr20_b -EvalConfig configs/eval_brats_val50_robust_iqr20_b.yaml
```

The builder refuses to overwrite its output. It verifies counts and writes normalization statistics and split provenance in `outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json`. The preflight checks exact source preprocessing agreement, target loading, split separation, and a disposable GPU optimization step. Its report is `outputs/diagnostics/robust_b/preflight.json`; smoke weights are not used for training.

The hidden launcher saves config snapshots, launch metadata, stdout and stderr under `outputs/runs/fomo45k_robust_iqr20_b`. Checkpoints are written after epochs 5, 10, 15 and 20 (zero-based filenames 0004, 0009, 0014, 0019). After fitting, the same process evaluates epoch_0019 automatically and writes results under `evaluation/brats_val50`.

## Interpretation

A successful pilot establishes that the new preprocessing and training pipeline work. Robust normalization aligns each image's center and scale; it does not force FOMO and BraTS to have identical distributions. Spatial, acquisition and tissue-composition differences can remain.

To attribute an AUPRC change to normalization, evaluate a matched p99 control with the same 20-epoch update budget, schedule and BraTS validation subjects. Comparing this pilot's validation AUPRC directly with the older 233-epoch model's test AUPRC cannot establish an improvement. Keep the test split for a later fixed experiment.
