# ANDi Rewrite Architecture Refactor Report

Date: 2026-08-13

Baseline commit: `3ded10261c64bc3a6af6904cd3848b280dab245f`

Working branch: `codex/ucsf-pdgm-brats-evaluation`

## 1. Summary

This refactor separated the five largest mixed-responsibility runtime modules while
preserving the established imports, configuration shapes, numerical behavior,
subclass/test seams, artifacts, and experiment defaults.

The result is a leaf-oriented architecture:

```text
scripts
  -> engine orchestration
       -> engine.evaluation leaves
       -> anomaly.postprocess policies/transforms/thresholds
  -> data.datasets adapters
  -> reporting builders and artifact I/O

data / anomaly / reporting / utils -X-> engine
```

Evidence at handoff:

- baseline suite: **98 passed, 0 failed, 0 skipped, 25 warnings in 7.80 s**;
- final suite: **141 passed, 0 failed, 0 skipped, 28 warnings in 34.34 s**;
- final compatibility review: **141 passed and 359 unittest subtests**;
- dependency scan: **79 runtime modules, 156 internal edges, 0 cycles**;
- no `anomaly`, `data`, `reporting`, or `utils` import of `engine`;
- a real one-subject UCSF-PDGM CUDA inference completed and produced all expected
  metrics, reports, metadata, score volumes, and masks.

The architecture design and final review used the requested Sol xhigh routing.
Mechanical extraction, re-export work, and focused compatibility coverage used the
requested Terra max routing.

## 2. Before / After architecture

Physical line counts include blank lines. Baseline counts were measured from
`3ded102`; the dataset monolith was 1,104 lines in that repository snapshot even
though the initial task estimate was approximately 705.

| Area | Before | After |
|---|---:|---|
| Evaluation | `engine/evaluator.py`: 1,652 | compatibility orchestrator: 648; eight focused leaves plus cohesive streaming backend |
| Postprocessing | `anomaly/postprocess.py`: 1,168 | package facade: 129; nine focused implementation modules |
| Datasets | `data/datasets.py`: 1,104 | package facade: 59; common helpers, adapter modules, and thin factory |
| Reporting | `utils/reporting.py`: 871 | compatibility facade: 61; six reporting leaves plus package exports |
| Preprocessing | `data/preprocess.py`: 766 | compatibility facade: 62; shared imaging, selection, split, and LMDB modules |

### 2.1 Evaluator

| Module | Lines | Ownership |
|---|---:|---|
| `engine/evaluator.py` | 648 | public `VolumeEvaluator`, orchestration, compatibility wrappers/hooks |
| `engine/evaluation/inputs.py` | 106 | batch and metadata interpretation |
| `engine/evaluation/inference.py` | 80 | model inference helpers |
| `engine/evaluation/collection.py` | 68 | in-memory score/label collection |
| `engine/evaluation/metrics.py` | 291 | shared metric formulas, stream accumulators, row projections |
| `engine/evaluation/prediction_export.py` | 275 | NIfTI/JSON prediction products |
| `engine/evaluation/output.py` | 107 | legacy CSV/result output helpers |
| `engine/evaluation/fingerprints.py` | 121 | raw/score cache fingerprints |
| `engine/evaluation/streaming.py` | 707 | disk cache traversal, staging, external sort, and streaming reduction |

`streaming.py` is above the ideal 500-line target but below 800 and remains one
cohesive disk-backed evaluation backend. Metric formulas were moved out so this
module no longer duplicates the in-memory metric ownership.

### 2.2 Postprocessing

| Module | Lines | Ownership |
|---|---:|---|
| `anomaly/postprocess/__init__.py` | 129 | stable public facade and compatibility dispatch |
| `anomaly/postprocess/_runtime.py` | 57 | narrow runtime indirection for historical monkeypatch seams |
| `anomaly/postprocess/base.py` | 54 | base type and score/mask registries |
| `anomaly/postprocess/numerics.py` | 54 | sanitization and normalization numerics |
| `anomaly/postprocess/transforms.py` | 260 | morphology/filter functions and registered transforms |
| `anomaly/postprocess/pipeline.py` | 127 | legacy translation, pipeline construction, execution |
| `anomaly/postprocess/threshold.py` | 169 | Yen/Otsu and dynamic threshold registry |
| `anomaly/postprocess/result.py` | 102 | method-neutral result plus legacy aliases |
| `anomaly/postprocess/policies.py` | 553 | base, Original ANDi, and rewrite policy semantics |
| `anomaly/postprocess/factory.py` | 67 | mode selection and legacy config migration |

### 2.3 Dataset adapters

| Module | Lines | Ownership |
|---|---:|---|
| `data/datasets/__init__.py` | 59 | legacy package facade |
| `data/datasets/common.py` | 26 | shared adapter types/helpers |
| `data/datasets/imaging.py` | 28 | re-export of the single shared normalization owner |
| `data/datasets/lmdb.py` | 70 | LMDB dataset adapter |
| `data/datasets/brats.py` | 218 | BraTS discovery and configuration translation |
| `data/datasets/ucsf_pdgm.py` | 420 | UCSF-PDGM discovery, orientation, labels, metadata |
| `data/datasets/shifts_ms.py` | 392 | Ljubljana/Shifts MS discovery and metadata |
| `data/datasets/factory.py` | 60 | alias-to-builder mapping and DataLoader construction |

Dataset-specific paths, modality names, subject parsing, geometry, and config
translation now stay in their owning adapter. The factory contains no adapter
implementation detail.

### 2.4 Reporting

| Module | Lines | Ownership |
|---|---:|---|
| `utils/reporting.py` | 61 | old import-path facade, including repo-root import context |
| `reporting/serialization.py` | 72 | JSON-safe conversion and config snapshots |
| `reporting/metadata.py` | 153 | runtime, Git, component, and config metadata |
| `reporting/metrics_csv.py` | 275 | metrics CSV parsing and stable summary schema |
| `reporting/training.py` | 140 | training report payload and Markdown |
| `reporting/inference.py` | 237 | inference report payload and Markdown |
| `reporting/io.py` | 130 | artifact paths and report writes |

### 2.5 Preprocessing

| Module | Lines | Ownership |
|---|---:|---|
| `data/preprocess.py` | 62 | old public facade |
| `data/imaging.py` | 79 | NIfTI loading, resize, and shared normalization |
| `data/healthy_slices.py` | 171 | healthy-slice discovery and selection |
| `data/subject_splits.py` | 138 | deterministic subject split and z balancing |
| `data/lmdb_io.py` | 460 | LMDB sizing/writing and current preprocessing workflow |

## 3. Responsibility map

| Concern | Authoritative owner | Consumers |
|---|---|---|
| Evaluator facade and subclass contract | `engine/evaluator.py` | scripts and existing tests/subclasses |
| Metric formulas and streaming accumulators | `engine/evaluation/metrics.py` | in-memory and disk-streaming evaluators |
| Disk streaming/cache/external AUPRC | `engine/evaluation/streaming.py` | `VolumeEvaluator` |
| Prediction products | `engine/evaluation/prediction_export.py` | both evaluation modes |
| Postprocess registries | `anomaly/postprocess/base.py` | pipeline builder and policies |
| Threshold selection registry | `anomaly/postprocess/threshold.py` | detector, factory, policies, CLI |
| Policy-owned streaming plan | `anomaly/postprocess/policies.py` | engine via neutral `ScorePipelineSpec` |
| Dataset selection | `data/datasets/factory.py` | `data` facade and scripts |
| Dataset behavior | individual adapter module | factory only through builder functions |
| MRI normalization | `data/imaging.py` | preprocess and dataset facades |
| Report parsing/building/writing | `reporting/*` leaves | scripts through the old facade; direct package API/tests |

The final AST dependency review found no cycles and no lower-layer reverse import
into `engine`. Postprocess leaves also do not reverse-import their facade; the small
`_runtime.py` holder preserves dynamic legacy patch behavior without a facade cycle
or `sys.modules` lookup.

## 4. Public API compatibility

The following established imports remain valid:

```python
from andi_rewrite.engine.evaluator import VolumeEvaluator
from andi_rewrite.anomaly.postprocess import (
    OriginalANDiPostprocessPolicy,
    RewritePostprocessPolicy,
    PostprocessResult,
    yen_threshold,
    otsu_threshold,
)
from andi_rewrite.data.datasets import build_dataset, MRIDataVolume
from andi_rewrite.utils.reporting import save_inference_report, save_training_report
```

Compatibility coverage verifies:

- all 47 historical `VolumeEvaluator` method names remain on the facade;
- public method signatures and descriptors match the baseline;
- practical private subclass hooks continue to dispatch through `self`, including
  nested metric hooks and streaming wrapper hooks;
- facade re-exports are the same owner objects where direct identity is expected;
- historical postprocess monkeypatch paths still control Original ANDi and rewrite
  pipelines;
- `PostprocessResult` exposes method-neutral fields while retaining Yen aliases;
- `SUPPORTED_THRESHOLD_METHODS` remains exactly `("yen", "otsu")` as the built-in
  compatibility snapshot, while the mutable loader registry is the runtime source;
- both `andi_rewrite.utils.reporting` and repo-root `utils.reporting` import and
  expose the complete 40-function legacy surface.

The literal import smoke command passed with the required interpreter from the
package parent `C:\ML\andi_test\Test`. Running it inside the package directory
without adding the parent to `PYTHONPATH` cannot resolve `andi_rewrite`; that is the
normal pre-existing Python package layout, not a refactor regression.

## 5. Config compatibility

- All **90** checked-in YAML files load as mappings through the public config loader.
- Every configured dataset type resolves to a registered adapter builder.
- Every `eval*.yaml` configuration builds the expected legacy or explicit
  postprocess policy without opening datasets or checkpoints.
- Both historical nested postprocess shapes and current explicit pipeline shapes
  remain accepted.
- Existing data type names, threshold defaults, postprocess defaults, training
  parameters, checkpoint shapes, and paths were not migrated or renamed.
- Dynamic threshold validation is shared by the CLI, detector, factory, and policy;
  built-in CLI choices and error behavior remain Yen/Otsu-compatible.

Focused config verification completed as **4 passed with 244 subtests**.

## 6. Numerical compatibility

### Original ANDi and rewrite

Fixed-tensor differential tests compare raw score, median-filtered score, adaptive
thresholds, raw/MF masks, and postprocessed masks. Original ANDi retains its required
order: median filtering is applied to the raw anomaly map before dataset-level
normalization. Rewrite retains its prior score and mask pipelines.

### Yen and Otsu

- thresholds and strict `>` masks match the baseline;
- constant-volume warnings and empty-mask behavior match;
- Yen's missing-scikit-image warning/mean fallback remains intact;
- Otsu's missing dependency raises the same chained `ImportError` text;
- legacy Yen result aliases and method-neutral adaptive fields agree.

### Metrics and streaming

- old-vs-live metric dictionaries have identical values and insertion order;
- adaptive-threshold reductions have identical floating-point bit patterns;
- in-memory and disk-streaming raw/MF CSV files are byte-identical in the
  compatibility differential;
- exact and sampled AUPRC, thresholds, Dice, predictions, cache resume, and
  fingerprint invalidation are covered;
- raw-cache fingerprints ignore threshold-only changes while score-pipeline changes
  invalidate the relevant score cache;
- policy descriptions and raw/score cache fingerprints match the baseline;
- a third custom policy produces exact in-memory/disk parity without engine imports
  of concrete policy classes or protected policy calls.

## 7. Tests and smoke verification

All commands used the required interpreter:

```powershell
& 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe' -m pytest
```

| Checkpoint | Result |
|---|---|
| Pre-refactor full baseline | 98 passed, 25 warnings, 7.80 s |
| Evaluator extraction focused suite | 45 passed, 12 warnings |
| Postprocess package contract | 5 passed, 35 subtests |
| Dataset package/path/UCSF suite | 24 passed, 24 subtests |
| First post-dataset full suite | 114 passed, 25 warnings |
| Reporting + preprocessing focused contracts | 11 passed, 28 subtests |
| Preprocessing/package integration | 32 passed, 55 subtests |
| Core-phase full suite | 132 passed, 25 warnings |
| Final full suite after Sol review fixes | **141 passed, 28 warnings, 34.34 s** |
| Final compatibility accounting | 141 passed, 359 subtests |

Additional verification:

- `py_compile` passed for the extracted evaluator, reporting, preprocess, dataset,
  and postprocess modules;
- a fresh-process import walk of 31 new modules passed from the package parent;
- reporting JSON, Markdown, and summary CSV were byte-compared against the baseline
  at the same output paths;
- `git diff --check` reports no whitespace errors (only Windows LF/CRLF notices).

### Real inference smoke

A real CUDA inference used:

- dataset: `C:/ML/data/UCSF-PDGM`;
- one subject: `UCSF-PDGM-0004`;
- checkpoint: `outputs/checkpoints/pyramid233_lmdb_full_gaussian/epoch_0232.pt`;
- postprocess: rewrite, dataset metric normalization, subject prediction
  normalization, Yen threshold;
- output: `outputs/refactor_smoke_ucsf_final_20260813/`.

The command completed in approximately 97.7 seconds wall time (87.027 seconds
recorded inference time) and produced 11 artifacts under `metrics/` and
`predictions/UCSF-PDGM-0004/`.

| Metric | Raw | Median filter |
|---|---:|---:|
| AUPRC | 0.6317271472310566 | 0.8780535476289731 |
| Yen Dice | 0.5832530083019807 | 0.9083930830873049 |
| Yen threshold | 0.134765625 | 0.099609375 |

The same values were produced by the pre-review and final-review smoke runs. The
temporary smoke config was removed after verification; ignored output artifacts
were retained for traceability and no existing output was overwritten.

## 8. Future extension examples

### New dataset

1. Add `data/datasets/oasis.py` containing OASIS discovery, geometry, metadata, and
   a `build_oasis_dataset(config)` adapter builder.
2. Import that builder in `data/datasets/factory.py` and add its aliases to the
   mutable `DATASET_BUILDERS` mapping.
3. Re-export only a class/function that must be public from `data/datasets/__init__.py`.

No existing adapter or monolithic dataset switch must be edited.

### New threshold method

```python
@register_threshold_method("triangle")
def load_triangle():
    from skimage.filters import threshold_triangle
    return threshold_triangle
```

The registry then drives validation, policy construction, detector behavior, CLI
override handling, evaluation result aliases, and CSV method naming. Reporting
remains independent of that registry: it infers a generic method from a base CSV row
and its exact `thr`/`threshold` sibling, with optional `sen`/`pre` siblings. Methods
such as `learned_threshold` are supported even when their name ends in a
reserved-looking suffix.

### New postprocessor

Subclass `BasePostprocessor`, decorate it with `@register_score_postprocessor(...)`
or `@register_mask_postprocessor(...)`, and use its name in a pipeline config. The
pipeline builder requires no new `if/elif` branch.

### New metric

Add the formula/accumulator and legacy row projection in
`engine/evaluation/metrics.py`, then expose it through the evaluator result/output
schema that owns the public artifact. Streaming no longer requires a second copy of
the same accumulator formula. A fully generic metric plugin registry is deliberately
deferred until a concrete per-subject or per-dataset metric requirement exists.

### New streaming-capable policy/backend

A policy can return an engine-neutral `ScorePipelineSpec` and override public
`complete_scores()`. The engine translates the plain spec into its backend plan and
does not branch on `OriginalANDiPostprocessPolicy` or
`RewritePostprocessPolicy`. An alternative backend can therefore consume the same
policy contract without importing concrete policies.

## 9. Remaining technical debt

1. `engine/evaluation/streaming.py` is 707 lines. It is cohesive today, but external
   sorting or cache traversal should become a leaf if either grows independently.
2. `data/lmdb_io.py` is 460 lines and still combines high-level fold orchestration,
   sampling-mode dispatch, and LMDB storage writes. A future non-LMDB backend should
   first extract a backend-neutral preprocessing workflow rather than duplicate it.
3. `engine/evaluator.py` remains 648 lines because it retains the historical method
   surface, subclass seams, and orchestration state. New algorithms should continue
   to be added to `engine/evaluation/*`, not back to the facade.
4. `tests/test_postprocess_pipeline.py` remains a 1,130-line behavioral regression
   suite. New package-specific contract files reduced production coupling without
   moving mature assertions, but a later tests-only cleanup could split it by
   transforms, thresholds, Original, rewrite, and compatibility.
5. A generic metric plugin/per-subject metric schema was intentionally not invented.
   The shared metric owner removes current duplication while leaving that design to
   a concrete future requirement.

These items do not create a cycle, numerical regression, public API break, or current
extension blocker.

## 10. Git commits and handoff state

Completed phase commits:

```text
9d02145 docs: plan architecture refactor
38ce24d refactor: split evaluator responsibilities
472ce42 refactor: modularize postprocessing
831d6c8 refactor: introduce dataset adapters
3a46de0 refactor: separate preprocessing storage and sampling
27b9a8a refactor: split reporting responsibilities
29f63d4 test: add architecture compatibility coverage
```

The final Sol review exposed and drove fixes for evaluator subclass dispatch,
streaming wrapper dispatch, concrete-policy coupling, protected policy calls,
authoritative dynamic threshold registration, shared stream metric ownership,
historical postprocess patch seams, and generic reporting of registered threshold
names. Those fixes, their regression tests, documentation path updates, and this
report are validated but remain **unstaged working-tree changes** because the Codex
environment rejected the final Git write after its usage quota was reached. No
fallback Git mutation was attempted.

The user-owned untracked `logs/` directory remains untouched and must not be included
in a future commit. The ignored real-inference output is intentionally retained as
verification evidence.
