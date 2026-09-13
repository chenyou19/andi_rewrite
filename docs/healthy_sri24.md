# MPI and OASIS3 SRI24 processing

Run `scripts/prepare_healthy_sri24.py COMMAND --dataset mpi` (or `oasis3`)
with `C:/Users/E-118-3/miniconda3/envs/ANDi/python.exe`.
Commands: inventory, preflight, pilot, full, build, audit, status.

Source data remains read-only. MPI requires a unique T1, T2 and high-resolution
FLAIR per session. OASIS requires a unique T1, T2 and FLAIR per visit; all
ambiguous visits are excluded. Header qform/sform disagreements larger than
0.1 mm at a volume corner are excluded pending investigation.

The existing FoMo pipeline supplies SynthStrip masks, N4 registration inputs,
T1 rigid+affine registration, conditional modality-to-T1 rigid registration,
single final image resampling, and spatial QC. Native intensities are retained
for the final warp. The reference atlas is LPS; no manual RAS/LPS flip is used.

Outputs are under `outputs/datasets/{mpi,oasis3}_sri24_robust_iqr`.
Inventory contains candidate paths and exclusions. Volumes contain individual
status files, transforms, masks, processing logs and QC montages. After pilot
montages have been inspected, a `pilot_review.json` with status PASS records
the review before the full command is allowed. Do not run two writers for the
same dataset simultaneously. Source/settings mismatch refuses replacement.

`scripts/run_healthy_sri24_batch.py` executes the two full cohorts sequentially
after both pilot reviews, retaining dataset logs and a controller report at
`outputs/reports/healthy_sri24_batch.json`. It uses an OS lock against duplicate
controllers. A successful cohort run automatically builds and audits its LMDB.
The workspace `scripts/synthstrip_healthy.cmd` invokes the pinned Python and
existing SynthStrip entrypoint directly: the external launcher reads a BOM from
`python_path.txt` under some Windows console environments and cannot find Python.
No external tool, model or input image is modified by this workaround.

The full command builds LMDB only after all cases have been attempted without
runtime exceptions; spatial QC failures are retained and excluded. LMDB uses
PASS cases, per-volume robust-IQR v1, mask-supported axial slices and the exact
FoMo Resize(128, antialias=True) operation. Values are float32 [FLAIR,T1,T2]
with shape [3,128,128]. Seed 73 splits unique participants, taking ceil(10%)
for validation. All visits of one person stay together.

An interrupted LMDB staging directory is kept for diagnosis, not overwritten.
Audit checks every entry, metadata, counts and participant separation.
Use intensity_normalization=robust_iqr and training.normalize_input=false
when consuming the finished train/val directories. No training is launched.
