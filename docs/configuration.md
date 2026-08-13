# YAML 設定參考

本文件依目前 factory、CLI、Trainer、Detector 與 Evaluator source 整理。範例中的 paths 是 placeholders；請勿直接把 repository 內實驗快照的絕對路徑套到其他機器。

## 1. 載入與組裝規則

`utils/config.py::load_config()` 使用 `yaml.safe_load()` 載入單一 YAML，並加入內部 `_config_path`。它沒有：

- schema validation；
- include / inheritance / interpolation；
- environment-variable expansion；
- 多檔 deep merge；
- 自動將相對 path 改成相對於 YAML 所在目錄。

相對 path 依 process 的 current working directory 解析。建議從 repository root 執行 scripts，並讓 config 明確指向現有 artifacts。

Canonical training sections：

```text
experiment + runtime + data + model + diffusion + noise + training
```

Canonical standalone evaluation sections：

```text
experiment + runtime + data + model + diffusion + noise
+ anomaly + metrics + prediction_output + evaluation
```

`experiment.name` 主要供人類辨識/report；training output 主要由 `training.run_name` 決定。

### Minimal training skeleton

```yaml
experiment:
  name: my_training
runtime:
  seed: 73
  device: auto
data:
  type: lmdb
  path: <healthy-lmdb>
  batch_size: 8
  image_size: 128
model:
  type: andi_unet
  in_channels: 4
  out_channels: 4
  image_size: 128
diffusion:
  steps: 1000
  beta_start: 0.0001
  beta_end: 0.02
noise:
  schedule:
    type: static
    sampler:
      type: gaussian
training:
  run_name: my_training
  epochs: 20
```

### Minimal evaluation skeleton

```yaml
experiment:
  name: my_evaluation
runtime:
  seed: 73
  device: auto
data:
  type: volume
  dataset_path: <volume-root>
  path_to_csv: <subjects.csv>
  modalities: [flair, t1, t1ce, t2]
  batch_size: 1
model:
  type: andi_unet
  in_channels: 4
  out_channels: 4
  image_size: 128
  checkpoint: <epoch_XXXX.pt>
  use_ema: true
diffusion:
  steps: 1000
noise:
  schedule:
    type: static
    sampler:
      type: gaussian
anomaly:
  t_lower: 75
  t_upper: 200
  aggregation: geometric_mean
  modality_pool: max
metrics:
  postprocess_mode: rewrite
  threshold_method: yen
  output_csv: outputs/metrics/my_evaluation/ANDi.csv
  output_mf_csv: outputs/metrics/my_evaluation/ANDi_mf.csv
evaluation:
  memory_mode: in_memory
```

## 2. Entry-point precedence 與差異

Standalone `scripts/eval.py` 給 evaluator 的 shallow merge：

```text
data < metrics < evaluation
```

也就是同名 key 以 `evaluation` 優先，接著另外注入完整的 top-level `prediction_output`、`model`、`anomaly` 與 `_run_config`。CLI `--threshold-method` 再覆蓋 `metrics.threshold_method`。

兩個路徑不是完全等價：

| Entry path | 傳給 evaluator | 影響 |
|---|---|---|
| `scripts/eval.py` | `data + metrics + evaluation`，另含 prediction/model/anomaly/full config | Canonical；支援 prediction export 與完整 cache/report metadata |
| `train.py` eval-after-fit | 只有 `data + metrics + evaluation` | 不啟用 top-level prediction export；fingerprint/report metadata 較少 |
| `eval_checkpoints50.py` | 只有 `data + metrics + evaluation` | 同上，並由 script 覆蓋 checkpoint/CSV/split 等 paths |

需要 NIfTI prediction 或完整 cache fingerprint 時，使用 standalone `scripts/eval.py`。

## 3. `runtime`

| Key | Code default | 支援值 / 行為 |
|---|---:|---|
| `seed` | `73` | 設定 Python、NumPy、Torch 與可用 CUDA RNG |
| `device` | `auto` | `auto` 優先 CUDA，否則 CPU；也可用任何 `torch.device` 字串 |
| `cudnn_benchmark` | train `true`; standalone eval `false` | 設定 `torch.backends.cudnn.benchmark` |
| `deterministic` | `false` | 設定 cuDNN deterministic flag；不等於全面 deterministic algorithms |
| `accelerate` | `false` | 使用 optional Accelerate |
| `distributed` | `false` | `accelerate` 的相容別名 |
| `find_unused_parameters` | `true` | 只在 Accelerate/DDP path 使用 |

限制：

- Accelerate/distributed evaluation 不能與 `evaluation.memory_mode: disk_streaming` 併用。
- `runtime.mixed_precision` 即使存在，目前只可能被 report 讀取；builder 沒有傳給 `Accelerator`，不應視為有效 execution setting。
- 種子與 cuDNN flags不能保證 bitwise reproducibility。

## 4. `data`

### DataLoader 共通 keys

| Key | Default | 說明 |
|---|---:|---|
| `type` | `lmdb` | Dataset factory type；應明確填寫 |
| `image_size` | `128` | XY resize target |
| `batch_size` | `1` | DataLoader batch size |
| `shuffle` | `false` | Training LMDB template 通常設為 true |
| `workers` | `0` | DataLoader processes；builder 不讀 `num_workers` |
| `pin_memory` | `false` | DataLoader pinned memory |

`data.channels` 主要作 metadata/cache fingerprint；它不改變 dataset 實際 output channels。通道數由 `modalities` 決定，必須與 model in/out channels 一致。

### `type: lmdb`

```yaml
data:
  type: lmdb
  path: <images.lmdb>       # required
  image_size: 128
```

LMDB keys 是連續八位數 ASCII；values 必須是可信、可由 pickle 還原的 NumPy arrays。輸出是 `[C,H,W]`，可選 resize。

### `type: brats_healthy_slices`

Aliases：`healthy_slices`。

```yaml
data:
  type: brats_healthy_slices
  dataset_path: <brats-root>       # required
  path_to_csv: <healthy-slices.csv> # required
  modalities: [flair, t1, t1ce, t2]
  slice_column: Slice
  filename_separator: "_"
  image_size: 128
```

CSV 第一欄為 subject id，且必須包含 `slice_column`。輸出 `[C,H,W]`，供 training 使用。

### `type: volume`

Aliases：`mri_volume`、`brats_volume`。

```yaml
data:
  type: volume
  dataset_path: <brats-style-root> # required
  path_to_csv: <subjects.csv>       # optional; omitted means discovery
  modalities: [flair, t1, t1ce, t2]
  segmentation_suffix: seg
  histogram_normalization: false
  shift_naming: false
  filename_separator: "_"
  return_metadata: false
```

| Key | Default / behavior |
|---|---|
| `modalities` | `[flair,t1,t1ce,t2]` |
| `segmentation_suffix` | `seg` |
| `shift_naming` | 未指定時依 `dataset_path` 是否含 `shifts` 推斷 |
| `return_metadata` | `false` |

輸出 image `[C,H,W,Z]`、mask `[H,W,Z]`，可選 metadata。若要 `prediction_output.restore_native_grid: true`，必須讓 dataset metadata 含有效 `reference_path`；evaluator 會從該 NIfTI 讀取 native shape/affine/header。

此通用 adapter 不驗證 modality 的 affine/orientation/spacing；只以讀到的 arrays 組合。

### `type: ljubljana_ms_volume`

Aliases：`shifts_ms_volume`、`shifts_volume`。

```yaml
data:
  type: ljubljana_ms_volume
  dataset_path: <shifts-root>       # required
  modalities: [flair, t1, t1ce, t2]
  modality_mapping: {}
  dataset_subdir: null
  locations: [ljubljana]
  preferred_locations: [ljubljana, best, msseg]
  splits: null
  reference_modality: flair
  require_segmentation: true
  require_modalities: true
  resample_to_reference: true
  histogram_normalization: false
  return_metadata: true
  subject_limit: null
```

`location` / `locations` 可為字串或列表；`subject_limit: null` 或 `0` 表示不限制。缺 modality 在 `require_modalities: false` 時補零。Grid 不同且 `resample_to_reference: true` 時，image 使用線性、mask 使用 nearest resampling；關閉 resample 則 mismatch 會失敗。

`require_segmentation: false` 時，缺 segmentation 的 subject 會回傳全零 bool mask 並在 metadata 標 `has_label: false`。只要一次 evaluation 中有任何這類 subject，整次 label-based metrics 都會 unavailable，不會只跳過該 subject。

### `type: ucsf_pdgm`

Alias：`ucsf_pdgm_volume`。完整 adapter/data contract 見 [datasets/ucsf-pdgm.md](datasets/ucsf-pdgm.md)。

```yaml
data:
  type: ucsf_pdgm_volume
  dataset_path: <ucsf-root>         # required
  path_to_csv: <subjects.csv>       # optional
  modalities: [flair, t1, t1ce, t2]
  modality_mapping:
    flair: FLAIR
    t1: T1
    t1ce: T1c
    t2: T2
  segmentation_suffix: tumor_segmentation
  reference_modality: flair
  model_orientation: LPS
  histogram_normalization: false
  return_metadata: true
  subject_limit: null
  duplicate_policy: error
```

- 選取 CSV 必須有 `subject_id`，不得空白或重複，並保留 CSV order。
- `duplicate_policy` 是 `error`（default）或 `first`。
- `reference_modality` 必須在 logical modalities 中。
- `model_orientation` / `expected_orientation` default `LPS`；設 null 可略過 orientation check。
- Adapter 驗證 complete modalities/segmentation、3D shape、spacing、orientation、affine。

## 5. `model`

### ANDi U-Net

```yaml
model:
  type: andi_unet
  in_channels: 4
  out_channels: 4
  image_size: 128
  time_dim: 256
```

| Key | Default / aliases |
|---|---|
| `type` | `andi_unet`; aliases `original_unet`, `unet` |
| `in_channels` | `channels` fallback，再 fallback `4` |
| `out_channels` | `channels` fallback，再 fallback `4`；不會自動 fallback 到 `in_channels` |
| `image_size` | `128` |
| `time_dim` | `256` |

ANDi U-Net attention grid 依 `image_size` 預先固定；實際輸入應使用對應的方形、可經四次二倍下採樣的 H/W。

### ConvNeXt U-Net

```yaml
model:
  type: convnext_unet
  in_channels: 4
  out_channels: 4
  base_channels: 64
  channel_mults: [1, 2, 4, 8]
  num_blocks: 2
  time_emb_dim: 256
  dropout: 0.0
```

Alias 是 `convnext-unet`。`channel_mults` 也接受 comma-separated string，但不可為空；`time_emb_dim` 可由 `time_dim` fallback。輸出會插值回 input H/W，`image_size` 不固定其動態 shape。

### Checkpoint keys

```yaml
model:
  checkpoint: <epoch_XXXX.pt>
  use_ema: true
```

`use_ema` code default 是 `false`；templates 常設為 `true`。有 `ema_model` 時才使用 EMA，否則載入 regular model state。

`model.checkpoint` 是 evaluation load path。Training resume 不讀它，而使用 `training.checkpoint.resume`。

## 6. `diffusion`

```yaml
diffusion:
  steps: 1000
  beta_start: 0.0001
  beta_end: 0.02
```

| Key | Default / constraint |
|---|---|
| `steps` / `noise_steps` | `1000`; must be > 1 |
| `beta_start` | `0.0001` |
| `beta_end` | `0.02` |

目前只支援 linear beta schedule。Training timesteps 是 `[1,steps)`；`anomaly.t_upper` 也必須落在有效範圍。

## 7. `noise`

簡寫與完整 static form 等價：

```yaml
noise:
  type: gaussian
```

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: gaussian
```

### Plans

| Schedule | Keys / behavior |
|---|---|
| `static` | `sampler`; 若沒有 nested sampler，schedule map 本身視為 sampler config |
| `epoch_switch` | `before`、`after` required；`switch_epoch` optional，否則 `switch_epoch_fraction: 0.5` |

Key 是 `switch_epoch_fraction`，不是 `switch_fraction`。Inference 沒有 epoch，會選 `before` branch。

### Samplers

```yaml
# Gaussian
sampler: {type: gaussian}

# Pyramid
sampler:
  type: pyramid
  discount: 0.8
  levels: 10
  normalize: true

# Analytic spectrum
sampler:
  type: spectrum
  exponent: 1.0
  low_frequency_bias: true
  normalize: true

# Hybrid
sampler:
  type: hybrid
  normalize: true
  components:
    - {type: gaussian, weight: 0.4}
    - {type: pyramid, weight: 0.6, discount: 0.8, levels: 10}
```

Hybrid components 不可空，可巢狀任何支援 sampler；每個 `weight` default `1.0`。

### Empirical spectrum

```yaml
sampler:
  type: empirical_spectrum
  stats_path: <spectrum-stats.npz>  # required
  mode: radial
  generation_method: fixed_magnitude
  spectrum_key: mean_amplitude
  radial_key: radial_amplitude
  spectrum_power_key: mean_power
  radial_power_key: radial_power
  per_channel: true
  strength: 1.0
  normalize: true
  eps: 1.0e-8
```

| Key | Values / constraint |
|---|---|
| `mode` | `radial` default；`2d` / `full2d` |
| `generation_method` | `fixed_magnitude` default（aliases `fixed`, `phase_randomized`）；`filtered_gaussian`（aliases `gaussian_filter`, `legacy_filter`） |
| `per_channel` | `true`; requested C 必須與 stats 相容 |
| `strength` | `1.0`; 必須在 `[0,1]` |
| `eps` | `1e-8`; must be > 0 |

Statistics H/W 必須符合 sample H/W。Radial mode 需要 `[C,R]` radial statistics；full-2D 需要 `[C,H,W]`。Filtered Gaussian 優先讀 power；缺少時以 amplitude squared fallback 並 warning。

## 8. `training`

```yaml
training:
  run_name: andi_rewrite
  epochs: 1
  normalize_input: true
  progress:
    enabled: true
  scheduler:
    type: warmup_cosine
    warmup_steps: 0.05
    start_lr: 2.0e-5
    target_lr: 1.0e-4
  ema:
    enabled: true
    decay: 0.995
    step_start: 2000
```

| Key | Default / behavior |
|---|---|
| `epochs` | `1` |
| `run_name` | `andi_rewrite` |
| `normalize_input` | `true`; 直接套用 `x * 2 - 1`，假設 input 約在 `[0,1]`；p99-normalized 資料未 clipping |
| `progress` | bool 或 `{enabled: bool}`; default true |
| `learning_rate` | `1e-4` when scheduler is disabled |
| `scheduler.type` | `warmup_cosine` default；`none`, `off`, `disabled` |
| `scheduler.warmup_steps` | `0.05`; < 1 表示總 training batches 的比例，否則是 step count |
| `scheduler.start_lr` | `2e-5` |
| `scheduler.target_lr` | `1e-4`，或 fallback `learning_rate` |
| `ema` | bool 或 mapping；mapping 未寫 enabled 時視為啟用 |
| `ema.decay` / `ema_decay` | `0.995` |
| `ema.step_start` | `2000` |

Warmup-cosine path 先以 optimizer LR `1.0` 建 AdamW，再由 LambdaLR 產生實際 LR 值；此時 `learning_rate` 不是 initial optimizer LR。

### Checkpoint

```yaml
training:
  checkpoint:
    dir: outputs/checkpoints
    start_epoch: 0
    save_every_epochs: 0
    save_last: false
    resume: null
```

Aliases：`checkpoint_dir`、`save_ckpt`、`start_ckpt`、`training.resume`。輸出 `<dir>/<run_name>/epoch_####.pt`。Resume 還原 model/optimizer/optional scheduler/EMA 並從 stored epoch + 1 繼續。

`save_last` 目前仍受 `save_every_epochs > 0` 的外層判斷影響。Checkpoint 只保存 training subsection config，且不保存 RNG/DataLoader state。

### Reverse samples

```yaml
training:
  samples:
    enabled: false
    output_dir: outputs/samples
    start_epoch: 0
    every_epochs: 1
    num_images: 3
    channels: 4
    image_size: 128
    mode: L
    nrow: null
    use_ema: true
    dtype: float32
    clip_denoised: true
```

Alias `sample`；`num_images` 也可 fallback `training.num_images`。`dtype` 只特別識別 `float16`、`bfloat16`，其他值使用 float32。

Sample grid 的 channel 顯示語意：`mode: L` 會把 batch 的每個 channel 拆成獨立 grayscale tile；非 `L` 且 channel 數不是 1 或 3 時，visualization helper 只保留前三個 channels。這只影響 PNG 呈現，不改 model sample tensor。

### Eval after fit

```yaml
training:
  eval_after_fit:
    enabled: true
    config: <eval.yaml>
```

也可直接設成 config path string。CLI `--eval-config` 優先，且只有 `--fit` 會觸發。這條路徑有前述 evaluator config forwarding 差異。

## 9. `anomaly`

```yaml
anomaly:
  t_lower: 75
  t_upper: 200
  aggregation:
    type: geometric_mean
  modality_pool:
    type: max
  threshold: yen
  eps: 1.0e-8
```

| Key | Default / values |
|---|---|
| `t_lower` / `start` | `75`; >= 1 |
| `t_upper` / `stop` | `200`; > lower；exclusive |
| `aggregation` | default `geometric`; mean/arithmetic, geometric/gmean, max, sum, weighted mean aliases |
| `modality_pool` | same registry; default `max` |
| `threshold` | `yen`; may be `otsu` or legacy numeric string |
| `eps` | `1e-8` |

`weighted_mean` 需要 non-empty `weights`，數量須與被聚合 axis 長度一致。Evaluator 的 adaptive method 應設定在 `metrics.threshold_method`；若沒有，才可能從 adaptive `anomaly.threshold` fallback。

目前 geometric aggregator 不使用傳入的 `eps` clamp，不能依 config 假設它對零值安全。

## 10. `metrics`

```yaml
metrics:
  output_csv: outputs/metrics/ANDi.csv
  output_mf_csv: outputs/metrics/ANDi_mf.csv
  thr_start: 0.01
  thr_end: 0.3
  thr_step: 0.001
  threshold: 0.5
  threshold_method: yen
  compute_auprc: true
  auprc_mode: exact
  auprc_seed: 73
```

| Key | Default / constraint |
|---|---|
| `output_csv` / `output` | `outputs/metrics/ANDi.csv` |
| `output_mf_csv` / `output_mf` | `outputs/metrics/ANDi_mf.csv` |
| `thr_start` / `threshold_start` | `0.01` |
| `thr_end` / `threshold_end` | `0.3` |
| `thr_step` / `threshold_step` | `0.001` |
| `threshold` | `0.5`; fixed summary/export cutoff，不是 adaptive method。Fixed summary rates 直接用 `score > threshold`；sweep/export fixed masks 才套 `threshold_mask.pipeline` |
| `threshold_method` | `yen` / `otsu` |
| `compute_auprc` | `true` |
| `auprc_mode` | `exact` / `sampled` |
| `auprc_max_samples` | sampled mode 必須 > 0；explicit exact mode 不可同時設定 |
| `auprc_seed` | `73` |

Threshold sweep 不含 `thr_end`，每個值 round 到三位。Legacy config 沒寫 `auprc_mode` 時，有 sample cap 會推定 sampled，否則 exact。

### Postprocess policy

```yaml
metrics:
  postprocess_mode: rewrite
  rewrite:
    normalization_scope: dataset
  postprocess:
    score:
      pipeline:
        - type: normalize
    score_mf:
      pipeline:
        - type: median_filter
          kernel_size: 5
          mode: 3d
        - type: normalize
    threshold_mask:
      pipeline: []
    binary_mask:
      pipeline:
        - type: binary_dilation
          rank: 3
          connectivity: 1
          iterations: 1
```

`postprocess_mode` 是 `rewrite` 或 `original_andi`。未設定時 warning 並使用 legacy-compatible rewrite。Rewrite normalization scope 是 `dataset` default 或 `subject`，也接受 `metrics.normalization_scope`。

Postprocess config precedence：

```text
metrics.postprocess > metrics.rewrite.postprocess > anomaly.postprocess
```

Score aliases：normalize/minmax/normalize_minmax、median_filter/median/mf、gray_dilation/grey_dilation。Mask aliases：binary_dilation/dilation、connected_components/remove_small_components/cc。

`original_andi` 只接受 dataset scope。3D median filter default enabled，也可明確設為 false；無論是否啟用，`mode` 仍必須是 `3d`，且 policy 初始化需要 SciPy + scikit-image：

```yaml
metrics:
  postprocess_mode: original_andi
  original_andi:
    normalization_scope: dataset
    median_filter:
      enabled: true
      kernel_size: 5
      mode: 3d
    binary_mask:
      binary_dilation:
        enabled: true
        rank: 3
        connectivity: 1
        iterations: 1
```

完整 stage semantics、fallback 與 exports 見 [postprocessing.md](postprocessing.md)。

## 11. `prediction_output`

```yaml
prediction_output:
  enabled: false
  directory: outputs/predictions
  normalization_scope:
    rewrite: subject
    original_andi: dataset
  restore_native_grid: true
  save_raw_score: true
  save_median_filtered_score: true
  save_binary_mask: true
  binary_mask_source: median_filtered
  threshold: 0.5
  threshold_source: median_filtered
  save_threshold_mask: false
```

| Key | Default / behavior |
|---|---|
| `enabled` | `false` |
| `directory` | `outputs/predictions` |
| `normalization_scope` | scalar `dataset`/`subject`，或依 mode mapping；rewrite default subject、original default dataset |
| `restore_native_grid` | `true`; requires dataset reference metadata |
| `save_raw_score` | `true` |
| `save_median_filtered_score` | `true` |
| `save_binary_mask` | legacy `save_yen_mask` fallback，最終 default true |
| `binary_mask_source` / `yen_source` | raw aliases 選 raw；其他值選 MF |
| `threshold` | fallback `metrics.threshold` |
| `threshold_source` | 同 source selection |
| `save_threshold_mask` | `false` |

這個 section 由 canonical standalone evaluation 使用。In-memory path 對收集完成的 tensors 匯出；disk-streaming path 則在 streaming metrics pass 逐 cached subject 匯出。Eval-after-fit 與 `eval_checkpoints50.py` 沒有把 section 傳入 evaluator。

## 12. `evaluation`

```yaml
evaluation:
  progress:
    enabled: true
  memory_mode: in_memory
  normalize_input: true
  size_splits: 155
```

| Key | Default / behavior |
|---|---|
| `progress` | bool 或 `{enabled: bool}`; true |
| `memory_mode` | `in_memory` default / `disk_streaming` |
| `size_splits` / `slice_batch_size` | `155`; often inherited from data via merge |
| `normalize_input` | `true`; often inherited from data |
| `rank` | `3` legacy morphology fallback |
| `connectivity` | `1` |
| `median_filter.enabled` | `true` legacy fallback |
| `median_filter.kernel_size` / `kernel_size` | `5` |

### Disk streaming cache

```yaml
evaluation:
  memory_mode: disk_streaming
  cache:
    directory: outputs/eval_cache/my_run
    resume: true
    keep_on_success: true
  external_sort:
    chunk_bytes: 268435456
```

| Key | Default / constraint |
|---|---|
| `cache.directory` | required、non-empty |
| `cache.resume` | `true` |
| `cache.keep_on_success` | `true` |
| `external_sort.chunk_bytes` | 256 MiB；must be > 0 |

Disk streaming 是 single-process；啟用 `prediction_output` 時可逐 cached subject 匯出。Resume 驗證 fingerprints、subject order、label availability、shape/dtype 與 cache files；mismatch/corruption 會 fail closed。

## 13. Checked-in config inventory

2026-08-12 inventory 中，`configs/` 有 90 份獨立 YAML；它們沒有 inheritance relationship，數量會隨 repository 演進：

- 32 份含 training section，58 份含 metrics，57 份含 evaluation。
- 目前所有 checked-in noise schedules 都是 static；source 另外支援 epoch-switch。
- 實際 config 使用 Gaussian、Pyramid、Empirical Spectrum、Hybrid；source 另外支援 analytic Spectrum。
- 大量檔案是 checkpoint、fold、dataset 或比較實驗快照，含 Windows absolute paths 以及被 `.gitignore` 排除的 checkpoints/NPZ。
- `configs/5fold*` 與 `fold4_full251` 是 checked-in experiment variants，不是生成系統的 include targets。

新工作應從四個 canonical templates 中選最接近者複製，然後逐一驗證 paths：

- `configs/train.yaml`
- `configs/train_lmdb.yaml`
- `configs/eval.yaml`
- `configs/eval_original_andi.yaml`

## 14. Run 前檢查清單

- Working directory 是否為 repository root？
- Dataset/CSV/LMDB/checkpoint/spectrum/output paths 是否存在或可寫？
- `model.checkpoint` 是否非空且來源可信？
- `modalities` 順序、dataset C、model input/output C、spectrum stats C 是否一致？
- `image_size` 是否符合 model 與 empirical stats H/W？
- `1 <= anomaly.t_lower < anomaly.t_upper <= diffusion.steps`？
- Postprocess mode/scope/pipeline 是否明確，不依 legacy migration？
- Large evaluation 是否應用 disk streaming，且已關閉 Accelerate？
- Disk streaming 是否需要 NIfTI prediction export，且 dataset metadata/reference paths 是否齊全？
- Output/cache/overwrite paths 是否已人工確認，不會覆寫需要保留的資料？
- 是否保存本次實際 YAML、git commit、checkpoint/stat identities 與 warnings？
