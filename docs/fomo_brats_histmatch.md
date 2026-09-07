# FOMO matched to BraTS non-lesion foreground

This dataset preserves the SRI24 p99 LMDB's keys, sessions, split, z selection and spatial arrays. Each session and modality receives one positive-foreground monotone intensity mapping after resizing. The stored background remains zero. Target curves use 4,097 quantiles averaged equally over 888 BraTS training subjects; the 50 validation subjects and test subjects do not fit the target. BraTS uses full-volume positive-foreground p99 normalization before resizing and lesion exclusion. Exclusion uses any positive resized lesion-mask coverage. Source negative values in BraTS do not fit foreground quantiles; the reference image itself is not clipped.

Run in the ANDi environment from the repository root:

```powershell
python scripts/prepare_histmatch.py
python scripts/verify_histmatch.py
```

Per-case reference caches can resume and check source sizes/mtime. An existing output is never overwritten. A failed alignment stays in a staging directory. Inspect `outputs/diagnostics/fomo_brats_histmatch/status.json` for status and `verification.json` for independent read-back results. The final data is `outputs/datasets/fomo45k_sri24_brats_histmatch` only after passing the alignment gate.

Both train and validation must meet CDF maximum grid error <= 0.01 and trapezoidal Wasserstein estimate / reference IQR <= 0.02. These are fixed-grid numerical comparisons, not exact discrete histogram equality. Output verification checks every stored slice's shape, type, finite values and unchanged zero mask. Distribution-summary quantiles are sampled; the CDF gate uses all positive output pixels. The independent BraTS validation curve is diagnostic only and does not tune the target.

Data settings for a future training configuration (no training is launched):

```yaml
data:
  type: lmdb
  path: outputs/datasets/fomo45k_sri24_brats_histmatch/train
  channels: 3
  image_size: 128
validation:
  data:
    type: lmdb
    path: outputs/datasets/fomo45k_sri24_brats_histmatch/val
    channels: 3
    image_size: 128
training:
  normalize_input: true
```

This maps stored values with `2*x-1` at model input. Do not apply p99 or robust normalization again and do not reuse B's robust input settings/checkpoint. The maps can reshape tissue contrast; inspect the twelve fixed-window montage images before interpreting a distribution match as useful for anomaly detection. Background ratios and z sampling are unchanged and the full-image histogram need not match.
