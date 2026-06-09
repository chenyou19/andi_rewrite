# ANDi Rewrite

`andi_rewrite` 是一個以 PyTorch 重寫的 ANDi MRI anomaly detection 研究框架。專案把 DDPM 訓練、可替換 noise sampler、ANDi anomaly map 推論、full-volume evaluation、metric 輸出與資料前處理拆成獨立模組，主要透過 YAML config 組裝實驗。

目前程式碼支援：

- 2D DDPM training，預設處理 4-channel MRI slice：`flair`、`t1`、`t1ce`、`t2`
- `ANDiUNet` noise-prediction model
- DDPM forward/reverse transition，預設 `1000` steps、`beta_start=1e-4`、`beta_end=0.02`
- Gaussian、Pyramid、Spectrum、Hybrid noise sampler
- static noise schedule 與 epoch-switch noise schedule
- EMA、AdamW、warmup cosine LR scheduler
- checkpoint save/resume 與 sample image export
- BraTS healthy-slice raw loader 與 LMDB loader
- full-volume evaluation，輸出 raw 與 median-filtered metrics CSV
- ANDi anomaly map time aggregation、modality pooling、Yen threshold 與 postprocess pipeline
- optional Accelerate distributed training/evaluation

## 專案結構

```text
andi_rewrite/
  anomaly/      ANDi detector、aggregation、threshold/postprocess pipeline
  configs/      training/evaluation YAML configs
  data/         BraTS/LMDB/volume datasets、healthy-slice preprocessing、registration helpers
  diffusion/    DDPM scheduler 與 transition math
  engine/       Trainer、VolumeEvaluator、EMA、checkpoint、LR scheduler
  metrics/      AUPRC、Dice、DiceYen、sensitivity、specificity、CSV writer
  models/       ANDiUNet 與 model factory
  noise/        Gaussian/Pyramid/Spectrum/Hybrid noise sampler 與 noise schedule
  scripts/      CLI entrypoints
  utils/        config、seed、progress、visualization helpers
```

`outputs/` 與 `results/` 目前包含已產生的 checkpoint、sample、metric 或測試資料，屬於執行產物，不是核心原始碼。

## 環境需求

這個 repository 目前沒有提供 `requirements.txt` 或 `environment.yml`。依照程式 import，至少需要下列套件：

```powershell
pip install torch torchvision numpy pandas pyyaml lmdb nibabel scipy scikit-image scikit-learn tqdm pillow
```

若要使用額外功能，還會需要：

- `accelerate`：distributed training/evaluation
- `dipy`：`prepare_data.py --register` 使用的 registration backend
- `SimpleITK`：histogram matching

建議先切到專案的上一層目錄或目前 package 目錄執行。以下範例假設目前在：

```powershell
cd C:\ML\andi_test\Test\andi_rewrite
```

## 資料格式

### BraTS volume layout

`MRIDataVolume` 與 healthy-slice preprocessing 預期每個 subject 一個資料夾，檔名格式如下：

```text
BraTS_2021/
  BraTS2021_00000/
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_seg.nii.gz
```

CSV 的第一欄會被視為 subject id：

```csv
BraTS_2021_subject
BraTS2021_00000
BraTS2021_00002
```

healthy-slice CSV 需要額外有 `Slice` 欄位：

```csv
BraTS_2021_subject,Slice
BraTS2021_00000,12
BraTS2021_00000,13
```

### LMDB training data

`scripts/split_healthy.py` 會從 raw BraTS volume 中挑出有 foreground 且 segmentation mask 為空的 healthy slices，並寫成：

- LMDB：key 為 `00000000` 這類連續編號，value 是 pickled numpy tensor，shape 為 `[C, H, W]`
- CSV：subject id 與 `Slice` metadata

建立 LMDB：

```powershell
python scripts\split_healthy.py `
  -d C:\ML\data\BraTS_2021 `
  -i C:\ML\ANDi\splits\BraTS21\scans_train.csv `
  -o C:\ML\ANDi\data\BraTS21\healthy_slices_train.csv `
  --lmdb-dir C:\ML\data\BraTS_2021_healthy_lmdb `
  --overwrite
```

常用參數：

- `-d, --data_set`：BraTS dataset root
- `-i, --input_file`：subject id CSV
- `-o, --output_file`：輸出的 healthy-slice CSV
- `-r, --resolution`：輸出 slice resolution，預設 `128`
- `--lmdb-dir`：LMDB 輸出位置，未指定時為 `<data_set>/healthy_slices`
- `--modalities`：預設 `flair t1 t1ce t2`
- `--map-size`：手動指定 LMDB map size
- `--overwrite`：覆寫既有 LMDB 目錄
- `--no-progress`：關閉 progress bar

建立 z 軸平衡的 healthy-slice LMDB（每個有 healthy candidate 的 z index 輸出固定張數）：
```powershell
python scripts\split_healthy.py `
  -d C:\ML\data\BraTS_2021 `
  -i C:\ML\ANDi\splits\BraTS21\scans_train.csv `
  -o C:\ML\ANDi\data\BraTS21\healthy_slices_train_zbalanced.csv `
  --lmdb-dir C:\ML\data\BraTS_2021_healthy_lmdb_zbalanced `
  --z-balanced `
  --per-z-count 447 `
  --balance-seed 42 `
  --overwrite
```

z-balanced options:
- `--sampling-mode {healthy,z_balanced}` / `--z-balanced`: default `healthy` keeps the original behavior; `z_balanced` balances output by z index
- `--per-z-count`: target output count for each z index in z-balanced mode, default `447`
- `--balance-seed`: seed for z-balanced random sampling, default `42`

Create one training LMDB per fold for 5-fold experiments:

```powershell
python scripts\split_healthy_kfold.py `
  -d C:\ML\data\BraTS_2021 `
  -i C:\ML\ANDi\splits\BraTS21\scans_train.csv `
  --test-file C:\ML\ANDi\splits\BraTS21\scans_test.csv `
  --combine-train-test `
  --combined-test-size 251 `
  --output-root C:\ML\ANDi\splits\BraTS21\5fold `
  --lmdb-root C:\ML\data\BraTS_2021_healthy_lmdb_5fold `
  --folds 5 `
  --split-seed 42 `
  --overwrite
```

Each fold writes:

- `fold_N/scans_train.csv`: subjects used to build that fold's training LMDB
- `fold_N/scans_test.csv`: test subjects for that fold when `--test-file` is set
- `fold_N/scans_val.csv`: held-out validation subjects when `--test-file` is not set
- `fold_N/healthy_slices_train.csv`: healthy-slice metadata for the training LMDB
- `<lmdb-root>/fold_N`: LMDB using the same key/value format as `scripts/split_healthy.py`

With `--combine-train-test --combined-test-size 251`, the 938-subject train CSV and 251-subject test CSV are combined into one 1189-subject pool, then each fold writes 938 train subjects and 251 test subjects. Because `251 * 5` is larger than `1189`, test sets are different deterministic windows over the shuffled pool but cannot be perfectly disjoint.

Use `--z-balanced --per-z-count 447` with the k-fold script when the fold LMDBs should use the same z-balanced sampling as the single-LMDB workflow.

Run the full 5-fold experiment in one command:

```powershell
python scripts\run_5fold.py `
  --dataset C:\ML\data\BraTS_2021 `
  --train-csv C:\ML\ANDi\splits\BraTS21\scans_train.csv `
  --test-csv C:\ML\ANDi\splits\BraTS21\scans_test.csv `
  --split-root C:\ML\ANDi\splits\BraTS21\5fold `
  --lmdb-root C:\ML\data\BraTS_2021_healthy_lmdb_5fold `
  --base-train-config configs\train_pyramid20_lmdb.yaml `
  --base-eval-config configs\eval_gaussian50_pyramid20.yaml `
  --config-dir configs\5fold `
  --combined-test-size 251 `
  --folds 5 `
  --overwrite-data
```

This prepares the combined train/test folds, writes `configs/5fold/train_fold_N.yaml` and `configs/5fold/eval_fold_N.yaml`, then runs each fold sequentially with `scripts/train.py --fit`. Use `--dry-run` to generate configs and print the fold commands without launching training.

## Training

先確認 config 內的資料路徑存在，例如 `configs/train_pyramid20_lmdb.yaml` 使用：

```yaml
data:
  type: lmdb
  path: C:/ML/data/BraTS_2021_healthy_lmdb
```

只載入 config 並建立 components，不跑訓練：

```powershell
python scripts\train.py --config configs\train_pyramid20_lmdb.yaml
```

跑一個 batch 的 smoke test：

```powershell
python scripts\train.py --config configs\train_pyramid20_lmdb.yaml --run-one-step
```

正式訓練：

```powershell
python scripts\train.py --config configs\train_pyramid20_lmdb.yaml --fit
```

只產生一次 sample grid：

```powershell
python scripts\train.py --config configs\train_pyramid20_lmdb.yaml --sample-once
```

訓練輸出：

```text
outputs/checkpoints/<run_name>/epoch_XXXX.pt
outputs/samples/<run_name>/epoch_XXXX.png
```

### Resume checkpoint

在 training config 設定：

```yaml
training:
  checkpoint:
    resume: C:/path/to/epoch_0019.pt
```

再執行：

```powershell
python scripts\train.py --config configs\train_pyramid20_lmdb.yaml --fit
```

checkpoint payload 包含：

- `model`
- `optimizer`
- `scheduler`
- `ema_model`
- `ema`
- `config`
- `epoch`

### Training config 列表

- `configs/train.yaml`：raw BraTS healthy-slice training template
- `configs/train_lmdb.yaml`：LMDB training template
- `configs/train_pyramid20_lmdb.yaml`：20 epoch pyramid-noise LMDB training，完成後自動跑 `configs/eval_gaussian50_pyramid20.yaml`
- `configs/train_pyramid233_lmdb_full_gaussian.yaml`：233 epoch pyramid-noise LMDB training，完成後自動跑 `configs/eval_full_gaussian_from_pyramid233.yaml`
- `configs/experiment/hybrid_spectrum_gaussian.yaml`：Spectrum/Gaussian hybrid noise 範例

## Evaluation

evaluation config 需要指定 volume dataset、checkpoint 與 metric output：

```yaml
data:
  type: volume
  dataset_path: C:/ML/data/BraTS_2021
  path_to_csv: C:/ML/ANDi/splits/BraTS21/scans_test.csv

model:
  checkpoint: outputs/checkpoints/pyramid233_lmdb_full_gaussian/epoch_0232.pt
  use_ema: true
```

只載入 config 並建立 detector：

```powershell
python scripts\eval.py --config configs\eval_full_gaussian_from_pyramid233.yaml
```

跑 full-volume evaluation：

```powershell
python scripts\eval.py --config configs\eval_full_gaussian_from_pyramid233.yaml --run-eval
```

常用 evaluation configs：

- `configs/eval.yaml`：evaluation template
- `configs/eval_gaussian50_pyramid20.yaml`：用 20-epoch pyramid checkpoint，搭配 Gaussian noise 評估 50 個 BraTS volumes
- `configs/eval_full_gaussian_from_pyramid233.yaml`：用 233-epoch pyramid checkpoint，搭配 Gaussian noise 做 full evaluation

metric 輸出格式為 original-style CSV，index 是 threshold 或 metric 名稱，欄位為 `value`：

```text
outputs/metrics/<run_name>/ANDi.csv
outputs/metrics/<run_name>/ANDi_mf.csv
```

`ANDi.csv` 使用 raw/postprocessed anomaly map；`ANDi_mf.csv` 使用 median-filtered anomaly map。兩者都會包含 threshold sweep、`yen`、`AUPRC`、`sensitivity`、`specificity`。

## ANDi anomaly 設定

Detector 會在 `t_lower <= t < t_upper` 的 DDPM timestep 範圍內計算 transition deviation，預設：

```yaml
anomaly:
  t_lower: 75
  t_upper: 200
  aggregation:
    type: geometric_mean
    eps: 1.0e-8
  modality_pool:
    type: max
  median_filter:
    enabled: true
    kernel_size: 5
  threshold: yen
```

支援的 aggregation / pooling：

- `mean`, `arithmetic`, `arithmetic_mean`
- `geometric`, `gmean`, `geometric_mean`
- `max`, `maximum`
- `sum`
- `weighted_mean`, `weighted`

weighted modality pooling 範例，權重數量需和 modality 數量一致：

```yaml
anomaly:
  modality_pool:
    type: weighted_mean
    weights: [0.25, 0.25, 0.25, 0.25]
```

## Noise 設定

### Gaussian

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: gaussian
```

### Pyramid

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: pyramid
      discount: 0.8
      levels: 10
      normalize: true
```

### Spectrum

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: spectrum
      exponent: 1.0
      low_frequency_bias: true
      normalize: true
```

### Empirical MRI spectrum

Compute empirical spectra from healthy LMDB slices:

```powershell
python scripts\compute_lmdb_spectrum.py `
  --lmdb-path C:\ML\data\BraTS_2021_healthy_lmdb `
  --out outputs\spectrum\brats21_healthy_empirical_spectrum.npz `
  --mask-mode union_nonzero `
  --window hann `
  --radial-bins 128 `
  --overwrite
```

Use the empirical spectrum noise sampler:

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: empirical_spectrum
      stats_path: outputs/spectrum/brats21_healthy_empirical_spectrum.npz
      mode: radial
      strength: 1.0
      normalize: true
      per_channel: true
```

- `mode: radial` uses a radial averaged spectrum, so it is less likely to learn fixed direction or position bias.
- `mode: 2d` or `mode: full2d` uses the full 2D spectrum and preserves directional frequency structure; use it as an ablation.
- `strength: 0` is Gaussian noise, `strength: 1` is fully empirical-spectrum-shaped noise.
- Recommended sweep: `strength = 0.25, 0.5, 0.75, 1.0`.
- Spectrum statistics crop around the nonzero foreground before FFT, which avoids contaminating the spectrum with outer black-background boundaries.

Compare one or more spectrum `.npz` files against the same LMDB MRI slice:

```powershell
python compare_npz_spectra.py `
  --spectrum-stats-paths `
    data/BraTS21/healthy_slices/healthy_spectrum_stats.npz `
    results/exp_a/spectrum_stats.npz `
    results/exp_b/spectrum_stats.npz `
  --labels healthy exp_a exp_b `
  --dataset-path data/BraTS21/healthy_slices `
  --index 100 `
  --channel 0 `
  --timestep 150 `
  --output-dir results/npz_spectrum_compare
```

This writes:

- `radial_spectrum_compare.png`
- `noise_compare_grid.png`
- `noised_mri_compare_grid.png`
- `npz_spectrum_compare_dashboard.png`
- `metadata.json`

### Hybrid

```yaml
noise:
  schedule:
    type: static
    sampler:
      type: hybrid
      normalize: true
      components:
        - type: spectrum
          weight: 0.6
          exponent: 1.0
          low_frequency_bias: true
          normalize: true
        - type: gaussian
          weight: 0.4
```

### Epoch switch

```yaml
noise:
  schedule:
    type: epoch_switch
    switch_epoch_fraction: 0.5
    before:
      type: spectrum
      exponent: 1.0
      low_frequency_bias: true
      normalize: true
    after:
      type: gaussian
```

也可使用 `switch_epoch` 指定明確 epoch。

## Postprocess pipeline

score-map postprocess 支援：

- `normalize`
- `median_filter`
- `gray_dilation`

mask postprocess 支援：

- `binary_dilation`
- `connected_components`

score map 範例：

```yaml
metrics:
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
```

mask 範例：

```yaml
metrics:
  postprocess:
    threshold_mask:
      pipeline:
        - type: binary_dilation
          rank: 3
          connectivity: 1
          iterations: 1
    yen_mask:
      pipeline:
        - type: binary_dilation
          rank: 3
          connectivity: 1
          iterations: 1
```

移除小 connected components：

```yaml
metrics:
  postprocess:
    yen_mask:
      pipeline:
        - type: connected_components
          min_size: 20
          connectivity: 3
```

## Shifts/MSSEG data preparation

整理 patient folders：

```powershell
python scripts\prepare_data.py `
  -d C:\ML\data\raw_shifts `
  --output-dir C:\ML\data\patients
```

registration：

```powershell
python scripts\prepare_data.py `
  -d C:\ML\data\patients `
  --register `
  -t C:\ML\atlas\SRI_template.nii
```

histogram matching：

```powershell
python scripts\prepare_data.py `
  -d C:\ML\data\patients `
  --norm `
  -i C:\ML\data\BraTS_2021\BraTS2021_00000
```

`--register` 目前使用 `dipy` backend，`--norm` 需要 `nibabel` 與 `SimpleITK`。

## Accelerate

在 config 啟用：

```yaml
runtime:
  accelerate: true
  find_unused_parameters: true
```

training：

```powershell
accelerate launch scripts\train.py --config configs\train_pyramid20_lmdb.yaml --fit
```

evaluation：

```powershell
accelerate launch scripts\eval.py --config configs\eval_full_gaussian_from_pyramid233.yaml --run-eval
```

## 擴充方式

### 新增 noise sampler

1. 在 `noise/` 新增繼承 `BaseNoise` 的 class
2. 實作 `sample(self, shape, device, dtype)`
3. 在 `noise/factory.py` 的 `build_noise_sampler()` 加入新的 `type`
4. 在 YAML 的 `noise.schedule.sampler.type` 使用新名稱

### 新增 dataset

1. 在 `data/datasets.py` 新增 `Dataset` class
2. 在 `build_dataset()` 加入新的 `data.type`
3. 在 YAML 設定：

```yaml
data:
  type: your_dataset
```

training dataset 的 `__getitem__()` 應回傳 `[C, H, W]` tensor；evaluation volume dataset 應回傳 `(image, mask)`，其中 image shape 為 `[C, H, W, Z]`。

### 新增 model

1. 在 `models/` 新增 model class
2. forward contract 需符合：

```python
model(x_t, timesteps) -> predicted_noise
```

3. 在 `models/factory.py` 的 `build_model()` 加入新的 `model.type`

### 新增 aggregation

1. 在 `anomaly/aggregation.py` 新增繼承 `BaseAggregator` 的 class
2. 使用 `@register_aggregator("your_name")`
3. 在 YAML 使用：

```yaml
anomaly:
  aggregation:
    type: your_name
```

或：

```yaml
anomaly:
  modality_pool:
    type: your_name
```

### 新增 postprocess step

score-map step：

1. 在 `anomaly/postprocess.py` 新增繼承 `BasePostprocessor` 的 class
2. 使用 `@register_score_postprocessor("your_step")`

mask step：

1. 在 `anomaly/postprocess.py` 新增繼承 `BasePostprocessor` 的 class
2. 使用 `@register_mask_postprocessor("your_step")`

YAML：

```yaml
metrics:
  postprocess:
    score:
      pipeline:
        - type: your_step
```

## 建議 workflow

1. 準備 raw BraTS volume 與 subject CSV
2. 用 `scripts/split_healthy.py` 建立 healthy-slice LMDB
3. 修改 training config 的 `data.path`、`training.run_name`、checkpoint/sample policy
4. 用 `--run-one-step` 做 smoke test
5. 用 `--fit` 訓練 DDPM
6. 修改 evaluation config 的 `data.dataset_path`、`data.path_to_csv`、`model.checkpoint`
7. 用 `scripts/eval.py --run-eval` 產生 `ANDi.csv` 與 `ANDi_mf.csv`
8. 比較 `AUPRC`、`yen`、threshold sweep Dice、sensitivity、specificity

## 注意事項

- `t_lower` 必須大於等於 `1`，因為 `x_0` 沒有上一個 transition。
- `t_upper` 必須大於 `t_lower`。
- training 預設會把 input 從 `[0, 1]` normalize 到 `[-1, 1]`。
- evaluation 預設也會做相同 input normalization。
- `model.use_ema: true` 時，若 checkpoint 含 `ema_model`，evaluation 會載入 EMA weights。
- `configs/eval.yaml` 是 template，預設 checkpoint 為空，正式 evaluation 前要填入 `model.checkpoint`。
- `configs/train_lmdb.yaml` 的 checkpoint/sample `start_epoch` 目前大於 `epochs`，因此用它直接訓練 20 epochs 時不會輸出 checkpoint/sample；若需要輸出，請改用 `configs/train_pyramid20_lmdb.yaml` 或調整 `start_epoch`。
