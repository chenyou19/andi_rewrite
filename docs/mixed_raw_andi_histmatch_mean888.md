# Raw-source mean888 training experiment

The dataset `mpi_oasis3_fomo45k_sri24_raw_andi_histmatch_mean888` preserves all
84,341 train and 9,569 validation entries of the original mixed dataset.
Source median centering and IQR division are omitted; positive registered MRI
intensities remain raw and background is set to -1 before the same antialias
resize. The frozen robust-IQR BraTS train888 reference is unchanged. Matching
uses SimpleITK 2.3.0 defaults: 256 levels, one match point, mean threshold on.

`scripts/prepare_mixed_raw_andi_histmatch.py` reuses the accepted comparison
and FOMO raw cache. It checks raw-file provenance, reconstructs the original
robust-IQR slices, matches each case, reads back every entry and checks its
foreground support. Per-case quantiles must reproduce the accepted no-IQR
comparison. Source manifests are copied exactly. Staging is resumable and an
existing published dataset with different signatures is refused.

The new train spectrum is computed from all output train entries with the
background-aware mask, Hann window, margin 4 and 64 radial bins. The sidecar
content hash must match the audited LMDB. Data is published only after these
checks pass. The metadata type is `andi_histmatch_mean888_raw_source`.

Run the builder in the ANDi environment, then start
`scripts/run_mixed_raw_histmatch200_pipeline.py`; the controller can also
start while the builder is running. Both programs take exclusive locks and
write durable state under
`outputs/reports/mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh`.
On this Windows host, the builder runs under the workspace account with read
access to the private registered MPI/OASIS volumes; the hidden controller runs
under the MRI-data account with read access to the external BraTS files.
Existing source ACLs are not modified.

The controller verifies the fixed test251 roster and all expected input files,
free space, dataset report and spectrum hash. It snapshots train/eval configs,
runs a disposable GPU optimization step, trains from scratch for 200 epochs,
and checks first-epoch health. It saves epochs 19,39,...,199 (zero based).
Interrupted training resumes only this run's full optimizer/scheduler/EMA
checkpoint; a surviving child process can be reattached after controller restart.

Train and validation inputs already use model intensity units, so
`normalize_input=false`. Test251 retains robust-IQR input preprocessing.
Both train and inference use the newly computed spectrum. After verifying
epoch 199 and finite EMA weights, the controller evaluates the fixed 251
subjects and audits their raw/median-filtered score and label caches, CSVs,
and inference reports. Prediction NIfTI export is disabled and caches are
retained. `status.json` is COMPLETE only after the final cache audit passes.

Runtime logs and checkpoints are under
`outputs/runs/mixed_sri24_raw_andi_histmatch_mean888_200ep_fresh`.
On failure, inspect the active stage and stderr, correct the cause, and restart
the same builder/controller. No previous experiment is deleted or overwritten.
