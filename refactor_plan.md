# ANDi Rewrite Architecture Refactor Plan

## 1. Scope and baseline

This refactor separates runtime responsibilities without changing model, diffusion,
noise, anomaly-score, postprocessing, metric, dataset, preprocessing, checkpoint,
configuration-default, or report semantics.

The pre-refactor baseline was recorded before production changes with the required
interpreter:

```powershell
& 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe' -m pytest
```

Result: **98 passed, 0 failed, 0 skipped, 25 warnings in 7.80 seconds**. The warnings
are the existing Matplotlib/pyparsing deprecations, the legacy postprocess migration
warning, constant-volume threshold warnings, and synthetic Otsu shape warnings.

Initial target sizes:

| Module | Lines | Current responsibilities |
|---|---:|---|
| `engine/evaluator.py` | 1,652 | inference, collection, postprocessing, metrics, export, cache, streaming, CSV/result output |
| `anomaly/postprocess.py` | 1,160 | numerics, transforms, registry, pipelines, thresholds, result, policies, config migration/factory |
| `data/datasets.py` | 1,104 | common MRI helpers, five dataset implementations, discovery, geometry, factory, dataloader |
| `utils/reporting.py` | 871 | serialization, metric CSV parsing, metadata, training/inference reports, Markdown and artifact I/O |
| `data/preprocess.py` | 761 | MRI loading, healthy-slice selection, balancing, split policy, LMDB sizing/writing/orchestration |

The worktree already contained an untracked `logs/` directory. It is outside this
refactor and will not be edited, staged, or removed.

## 2. Current architecture and dependency graph

The primary runtime dependency direction is sound at package level:

```text
scripts
  -> engine.VolumeEvaluator / engine.Trainer
       -> anomaly.ANDiDetector and anomaly.postprocess
       -> metrics.classification
       -> engine.evaluation_cache
       -> utils.progress
  -> data.build_dataloader
       -> data.datasets implementations
  -> utils.reporting
```

There is no current import cycle. Coupling is concentrated inside the five target
modules rather than between top-level packages.

### 2.1 Evaluator coupling

`VolumeEvaluator` owns configuration parsing, Accelerate preparation, batch and
metadata interpretation, slice/volume reshaping, detector inference, in-memory
collection, policy orchestration, metric reduction, prediction NIfTI/JSON export,
cache fingerprint construction, disk collection/resume, dataset-wide normalization,
MF staging, sampled/exact AUPRC, CSV writing, and result assembly.

Direct callers are `scripts/eval.py`, `scripts/train.py`,
`scripts/eval_checkpoints50.py`, and `scripts/inspect_ljubljana_ms.py`. Public imports
are available through both `andi_rewrite.engine.evaluator` and `andi_rewrite.engine`.

Practical compatibility API includes:

- constructor `VolumeEvaluator(detector, config, accelerator=None)`;
- `prepare`, `evaluate`, `collect`, `process_raw_maps`, `threshold_values`,
  `summarize`, `summarize_processed`, and `write_original_style_csv`;
- `is_main_process`, output paths, selected policy/method/scope, and last processed
  results;
- test/subclass seams `_volume_scores`, `_prediction_postprocess`, and
  `_export_predictions`.

The in-memory and streaming paths duplicate result construction, batch iteration,
and parts of metric reduction. Streaming additionally reaches into the protected
`PostprocessPolicy._complete()` method and concrete policy attributes.

### 2.2 Postprocess coupling

`anomaly/postprocess.py` is consumed by `ANDiDetector`, `VolumeEvaluator`,
`scripts/eval.py`, diagnostics, reports, and tests. `anomaly/__init__.py` exports the
policy/result/registry/threshold surface. Direct-module callers additionally use
transform functions, concrete postprocessors, pipeline helpers, and
`sanitize_scores`.

The exact `PostprocessPolicy.describe()` keys feed cache fingerprints, inference
reports, and prediction metadata. Mutable registry objects and alias insertion order
are observable. Existing tests also patch implementation functions by their old
module path; those tests may be redirected to their new owner, while all documented
runtime imports remain available from `andi_rewrite.anomaly.postprocess`.

### 2.3 Dataset coupling

`data/datasets.py` implements LMDB slices, raw BraTS healthy slices, generic
BraTS-style volumes, UCSF-PDGM, and Shifts/Ljubljana-MS in one file. The factory maps
the following stable type names:

- `lmdb`;
- `brats_healthy_slices`, `healthy_slices`;
- `volume`, `mri_volume`, `brats_volume`;
- `ljubljana_ms_volume`, `shifts_ms_volume`, `shifts_volume`;
- `ucsf_pdgm`, `ucsf_pdgm_volume`.

Public classes/builders are re-exported from `data/__init__.py`. Tests also use
`_subject_file_path`, UCSF geometry helpers, adapter private resize/load methods, and
discovered subject records. Adapter-specific discovery, channel naming, resampling,
orientation, label, resizing, ordering, and metadata contracts must remain local to
each adapter and byte-for-byte compatible at the returned-data level.

### 2.4 Reporting and preprocessing coupling

`utils/reporting.py` is currently a useful dependency leaf: it uses duck typing and
does not import `engine`. It is called by standalone evaluation, training,
checkpoint evaluation, and empirical-stability scripts. Exact report keys are read
by other scripts. Public entry points are `save_training_report`,
`save_inference_report`, and `summarize_eval_metrics`.

`data/preprocess.py` is called by the healthy-slice scripts and `run_5fold.py`.
Checked-in YAML consumes the LMDB/CSV products rather than invoking preprocessing.
The formal package exports are `split_healthy_to_lmdb` and
`split_healthy_kfold_to_lmdb`; a maintenance script also imports `_load_nifti`.

`normalize_volume` is duplicated between runtime datasets and preprocessing. NIfTI
loading and resizing are also repeated, but several adapters intentionally differ
in interpolation, antialiasing, geometry validation, and error behavior, so only
semantically identical helpers will be shared.

## 3. Proposed architecture and ownership

The design follows responsibility rather than line count. Compatibility modules or
package `__init__.py` files remain the stable public surface.

```text
engine/
  evaluator.py                    VolumeEvaluator facade/orchestrator
  evaluation/
    __init__.py                   internal cohesive exports
    inputs.py                     batch parsing, metadata uncollation, label policy
    inference.py                  slice chunking and volume-score reconstruction
    metrics.py                    in-memory/stream metric reducers and threshold grid
    prediction_export.py          NIfTI/JSON product export
    output.py                     legacy CSV and result dictionary assembly
    fingerprints.py               stable cache fingerprint payloads
    streaming.py                  cache passes, global bounds, MF staging, AUPRC
  evaluation_cache.py             unchanged storage/manifest/atomic-write owner
```

`VolumeEvaluator` retains configuration/state and compatibility wrappers, but
delegates algorithms and I/O. Extracted modules receive explicit values or callbacks
and never import the facade.

```text
anomaly/postprocess/
  __init__.py                     complete old import surface
  base.py                         BasePostprocessor and common types
  numerics.py                     sanitization and min-max normalization
  transforms.py                   morphology functions and concrete steps
  registry.py                     stable registry objects and pipeline build/apply
  threshold.py                    method registry/validation and Yen/Otsu execution
  result.py                       PostprocessResult and compatibility aliases
  policies.py                     base, original-ANDi, and rewrite policies
  factory.py                      legacy configuration translation/policy construction
```

Threshold selection remains separate from mask postprocessing. Built-in threshold
functions use a small mapping registry so a future Li/Triangle/Isodata/Percentile
implementation can be added without expanding policy branches. Built-in transform
registration is deterministic and complete before factory use.

```text
data/datasets/
  __init__.py                     complete old import surface
  common.py                       identical shared path/list/shape helpers
  imaging.py                      shared normalization and carefully scoped MRI helpers
  lmdb.py                         LMDBSliceDataset
  brats.py                        BraTSHealthySliceDataset and MRIDataVolume
  ucsf_pdgm.py                    UCSF records, discovery, geometry, loading, metadata
  shifts_ms.py                    Shifts/Ljubljana records, discovery, resampling, metadata
  factory.py                      alias-to-builder mapping and DataLoader construction
```

The factory only chooses an adapter. All dataset-specific config interpretation
stays with the corresponding builder/adapter module. Adding a dataset should require
one adapter module plus one registry entry, not edits across existing adapters.

```text
reporting/
  __init__.py                     cohesive report API
  serialization.py               JSON-safe conversion and config snapshots
  metadata.py                    environment/git/component metadata helpers
  metrics_csv.py                 legacy/current metric CSV parsing and summary CSV
  training.py                    training snapshot and Markdown
  inference.py                   inference snapshot and Markdown
  io.py                          report artifact writes
utils/reporting.py                thin compatibility facade
```

Reporting stays engine-agnostic and preserves current duck-typed inputs. Explicit
context objects are deferred because introducing them now would increase migration
risk without changing behavior.

```text
data/
  preprocess.py                   high-level compatibility facade/orchestration
  healthy_slices.py               eligibility, candidates, z balancing
  subject_splits.py               CSV validation and fold/window split policy
  lmdb_io.py                      map sizing and LMDB/metadata writing
  imaging.py                      shared identical MRI normalization/loading primitives
```

Selection policy, subject split policy, and storage backend will have distinct
owners. High-level entry points and existing import paths remain stable.

## 4. Dependency rules

New leaf modules obey these rules:

```text
models / diffusion / noise
             -> anomaly
             -> engine
             -> scripts

data adapters -> data factory -> scripts/engine composition
```

- `metrics`, `anomaly`, `data`, `reporting`, and `utils` do not import `engine`.
- Engine leaf modules do not import `andi_rewrite.engine` or `VolumeEvaluator`.
- Internal imports target leaf modules, not package facades.
- `utils/reporting.py` and the new reporting package remain duck-typed leaves.
- Shared imaging code is a leaf imported by datasets and preprocessing; neither
  imports the other.
- Package initializers only re-export; they do not own runtime algorithms.

An import-walk smoke test will be used after each package conversion to catch partial
initialization and circular imports.

## 5. Compatibility strategy

### Public imports

These paths remain valid:

```python
from andi_rewrite.engine.evaluator import VolumeEvaluator
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.anomaly.postprocess import (
    BasePostprocessor,
    MedianFilterPostprocessor,
    NormalizePostprocessor,
    OriginalANDiPostprocessPolicy,
    PostprocessPolicy,
    PostprocessResult,
    RewritePostprocessPolicy,
    apply_postprocess_pipeline,
    build_postprocess_policy,
    otsu_threshold,
    sanitize_scores,
    yen_threshold,
)
from andi_rewrite.data.datasets import MRIDataVolume, build_dataset
from andi_rewrite.data import build_dataloader
from andi_rewrite.utils.reporting import save_inference_report, save_training_report
```

Package facades re-export the same class/function objects. Evaluator wrappers preserve
existing overridable/tested seams. No config key or dataset type alias is renamed.

### Numerical behavior

The following contracts are frozen by characterization tests:

- non-finite scores map to zero;
- Min-Max scope and epsilon behavior remain unchanged;
- Original ANDi filters the **unnormalized** raw map, then independently performs
  dataset Min-Max normalization for raw and MF branches;
- Rewrite `score_mf` continues consuming processed `score_raw` in configured order;
- Yen/Otsu run per 3-D subject and use strict `>`;
- constant-volume thresholds, warnings, and empty masks remain unchanged;
- metric sweep endpoint/rounding, per-subject mean Dice, micro rates, AUPRC modes,
  sampling seed/order, and exact external sort remain unchanged;
- dataset channel order, mask comparator, orientation, interpolation, resampling,
  metadata, and preprocessing output remain unchanged.

### Cache and artifact compatibility

- Cache manifest schema/version, float32/bool storage, validation, atomic writes,
  cleanup order, and resume behavior remain unchanged.
- Fingerprint implementation strings and serialized payloads remain unchanged.
- Threshold/mask selection remains excluded from the score fingerprint so Yen/Otsu
  changes can reuse score caches.
- CSV row order/schema, result keys, Yen compatibility aliases, NIfTI filenames,
  dtypes, affine/header reuse, prediction metadata, report filenames, and report keys
  remain unchanged.

## 6. Migration order

Each step follows `extract -> retain facade -> focused test -> full regression`.

1. Record baseline and this architecture plan.
2. Evaluator:
   - add compatibility/golden tests;
   - extract output/result assembly and metric reducers;
   - extract prediction export and inference/input helpers;
   - extract fingerprint and streaming orchestration last;
   - keep `VolumeEvaluator` as stateful facade and preserve callback/override seams.
3. Postprocess:
   - extract numerics/transforms/thresholds/result;
   - move registry and pipeline construction with deterministic built-in bootstrap;
   - move policies and factory last;
   - convert old module path into a package facade with complete re-exports.
4. Datasets:
   - extract one adapter at a time;
   - introduce alias-to-builder mapping after adapter parity;
   - convert old module path into a package facade.
5. Reporting:
   - characterize metric parsing and both report types;
   - split parser/serialization/metadata/training/inference/I/O;
   - retain `utils/reporting.py` as a thin facade.
6. Preprocessing:
   - characterize normalization, selection, balancing, splits, and LMDB products;
   - extract selection/split/storage leaves;
   - retain high-level facade functions.
7. Run complete compatibility, config, report, prediction, CLI, and real inference
   verification; then perform a final architecture/cycle review.

## 7. Risk analysis

| Risk | Protection |
|---|---|
| Original ANDi or rewrite ordering changes | fixed tensor golden/parity tests for every intermediate product |
| Streaming normalization/MF differs from in-memory | existing exact/sampled parity plus original/rewrite scope tests |
| Reduction order or dtype changes metrics | preserve loops/order/dtypes first; compare result dictionaries and CSVs |
| Existing cache becomes invalid or wrongly reused | golden fingerprints and unchanged schema/version payloads |
| Prediction artifacts change | compare filenames, arrays, dtype, shape, affine/header, and JSON fields |
| Shared policy identity breaks | retain description equality and detector/evaluator identity test |
| `_volume_scores` subclass seam breaks | facade collection and streaming callbacks continue through the wrapper |
| Registry loses built-ins/aliases | registry identity/order/third-party registration tests |
| Package conversion causes a cycle | direct leaf imports plus import-walk smoke after each phase |
| Dataset adapters are over-generalized | keep UCSF/Shifts interpolation, geometry, limit/order, and label contracts separate |
| Reporting schema drifts | exact JSON/Markdown/summary parser characterization tests |
| Preprocessing output changes | synthetic NIfTI/LMDB round-trip and exact CSV/key/value tests |
| Scope grows into unrelated cleanup | dead fields, legacy names, and known bugs remain unless required for extraction |

## 8. Test strategy

### Existing phase suites

- Evaluator: `tests/test_streaming_evaluator.py` and evaluator/report/export cases in
  `tests/test_postprocess_pipeline.py`.
- Postprocess: `tests/test_postprocess_pipeline.py` including Original ANDi,
  rewrite, Yen, Otsu, prediction, and report cases.
- Datasets: `tests/test_dataset_paths.py` and `tests/test_ucsf_pdgm_dataset.py`.
- Reporting: existing inference tests plus new training/parser tests.
- Preprocessing: new pure and temporary-directory NIfTI/LMDB tests.

### Required regression coverage

1. Old import paths and re-export identity.
2. Original ANDi intermediate numerical parity.
3. Explicit rewrite and both legacy profiles.
4. Yen and Otsu thresholds/masks, strict comparator, constants, empty/non-finite data.
5. In-memory/disk-streaming exact and sampled parity, resume, corruption,
   fingerprint reuse/invalidation, no-label behavior, prediction products.
6. Dataset alias matrix, discovery/order/limit, modality/geometry, metadata, resize,
   and spawned-worker imports.
7. Load every checked-in YAML and build policy/factory where this does not require
   external data.
8. Report JSON/Markdown/summary CSV contracts and training/inference imports.

### Final smoke tests

Use only the required interpreter:

```powershell
& 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe' -c "from andi_rewrite.engine.evaluator import VolumeEvaluator; from andi_rewrite.anomaly.postprocess import OriginalANDiPostprocessPolicy, RewritePostprocessPolicy, yen_threshold, otsu_threshold; from andi_rewrite.data.datasets import build_dataset; print('imports ok')"
& 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe' -m pytest
```

The repository provides a verified one-case real-data path and both required local
artifacts currently exist:

```powershell
& 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe' -B scripts\eval.py --config configs\eval_ucsf_pdgm_pyramid233_gaussian.yaml --run-eval
```

For final verification, a copied temporary config will redirect metrics,
predictions, and reports to an isolated workspace directory so existing experiment
artifacts are not overwritten.

## 9. Acceptance review

The final review will measure module sizes, enumerate dependency edges, import every
new module, inspect facade indirection, and demonstrate how to add one dataset,
threshold, postprocessor, and metric. Tests passing is necessary but not sufficient:
the evaluator must delegate real work, policies must not mix threshold algorithms
with mask transforms, the dataset factory must not implement adapters, reporting must
remain a leaf, and preprocessing selection must be independent of LMDB storage.
