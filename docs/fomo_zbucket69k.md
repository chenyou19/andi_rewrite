# FOMO histogram-matched, 29 z buckets, 69,000 training entries

Build with `python scripts/prepare_zbucket69k.py` in the ANDi environment. Existing output or staging directories are not overwritten.

Source: `outputs/datasets/fomo45k_sri24_brats_histmatch`. Output: `outputs/datasets/fomo45k_sri24_brats_histmatch_zbucket29_69k`.

The 29 buckets are 0-4, 5-9, ..., 135-139, and 140-145. BraTS healthy counts are summed over these buckets and rescaled to 69,000 using largest remainders, with bucket order breaking ties. No source exists at z 146-154, so these reference positions are excluded.

Sampling is uniform over source slices within each bucket. Full repetitions are followed by a seeded sample without replacement for the remainder. Seed 73 also controls final shuffling. This matches bucket totals, not individual z totals. It uses 28,309 distinct source slices; no source is used more than seven times. Validation retains its original 3,369 entries.

Every serialized image is copied unchanged and verified against its source key. The output manifest records source keys, original case/session/z, bucket IDs and one-based duplicate sequence. `repetition_counts.csv` includes unused source slices with zero uses. Original intensity mapping provenance remains in the source dataset; the normalization sidecars and reference curves accompany the new data.

`build_report.json` contains source and reference manifest hashes, quotas, verification status, repetition statistics and sampled foreground brightness comparisons. `bucket_counts.csv`, `z_counts.csv`, `z_distribution.png` and `brightness_distribution.png` provide diagnostics. Brightness statistics use fixed uniform spatial samples weighted by exact slice multiplicity; images are not histogram-matched a second time.

Training is not launched. Future training should read the new train and val directories, with channels FLAIR/T1/T2 and `training.normalize_input: true` for the existing `2*x-1` input transform. Do not reuse the robust B input setting.
