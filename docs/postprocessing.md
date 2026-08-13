# Anomaly score 後處理與 threshold

本文件描述 `anomaly/postprocess/` package 與 `engine/evaluation/` components 的目前行為；`engine/evaluator.py` 保留為相容 facade/orchestrator。YAML section 的完整位置見 [configuration.md](configuration.md)。

## 1. 核心概念

Detector 先產生 continuous 3D anomaly score。以下四階段與圖示描述 `rewrite` mode；`original_andi` 會從 sanitized raw 分岔，流程見第 3 節：

1. `score`：raw score branch 的 continuous transforms。
2. `score_mf`：rewrite 中從處理過的 raw branch 出發的第二個 continuous branch，通常加入 median filter。
3. Adaptive threshold：對每位 subject 的完整 3D raw/MF score 計算 Yen 或 Otsu threshold。
4. Mask postprocess：對 binary mask 做 dilation 或 connected-component filtering。

Fixed threshold sweep 與 adaptive threshold 使用不同的 mask pipeline：

```text
raw detector score
  └─ score.pipeline ───────────────► score_raw
                                      └─ score_mf.pipeline ─► score_mf
                                            │
                  ┌─────────────────────────┴──────────────────────────┐
                  │                                                    │
          fixed numeric sweep                                  Yen/Otsu per subject
                  │                                                    │
       threshold_mask.pipeline                               binary_mask.pipeline
```

Threshold operation不是 registry step，且 comparator 固定為 strict `score > threshold`。

## 2. Shared policy

`VolumeEvaluator` 建立時會將同一個 `PostprocessPolicy` instance 設回 `ANDiDetector`。因此 detector direct call、metrics 與 prediction export 共享同一組 mode、pipelines 與 threshold method，避免不同層各自實作 Yen/median/dilation。

Policy output `PostprocessResult` 保存 raw/MF scores、adaptive thresholds、threshold 前後的 masks 與 policy description。`ANDiDetector.postprocess()` 另外保留舊欄位 aliases，供既有呼叫端相容。

## 3. Modes

### `rewrite`

`RewritePostprocessPolicy` 是可組合模式：

```text
sanitize(raw)
  → score.pipeline
  → score_raw
  → score_mf.pipeline
  → score_mf
```

所以 MF branch 是從已處理的 `score_raw` 開始，不是永遠從原始 detector output 開始。Normalization scope 可為：

- `dataset`：以整個 evaluation set 的 min/max 正規化。
- `subject`：每位 subject 各自 min/max 正規化。

Standalone detector 只能看到呼叫者給的 batch；在 dataset scope 時，該 batch 就是它可用的 dataset。完整 evaluator 才能對整個測試集實作真正的 dataset scope。

Canonical template：[configs/eval.yaml](../configs/eval.yaml)。

### `original_andi`

`OriginalANDiPostprocessPolicy` 使用 reference-compatible ordering；median filter default enabled，但 config 可明確關閉：

```text
sanitize(raw) ───────────────────────→ dataset min-max → score_raw
       └─ optional 3-D median filter on unnormalized raw
                                      → independent dataset min-max → score_mf
```

啟用時先對未正規化 raw score 做 MF，再讓 raw/MF 兩個分支各自以全 dataset 的 min/max 正規化；關閉時 MF branch 是 raw 的 clone，仍獨立正規化。`mode` 即使在關閉狀態也必須是 `3d`。Metrics scope 強制為 `dataset`；不能用 rewrite 的 subject-scope 語意替代。

Reference flow 的預設 threshold 是 Yen，但目前 policy 同樣接受 Otsu。`original_andi` 不是「只能用 Yen」的 hard-coded branch。

Canonical template：[configs/eval_original_andi.yaml](../configs/eval_original_andi.yaml)。

### Legacy config

沒有明確設定 `metrics.postprocess_mode` 時，builder 會建立 legacy-compatible rewrite policy 並發出 `FutureWarning`。Legacy aliases（例如 `median_filter`、`kernel_size`、`yen_mask`）仍會編譯到新 pipeline，但新 config 應明確使用：

```yaml
metrics:
  postprocess_mode: rewrite
  threshold_method: yen
  postprocess:
    score: ...
    score_mf: ...
    threshold_mask: ...
    binary_mask: ...
```

`binary_mask` 是 adaptive threshold 後的通用名稱；不要在新文件或 config 把它寫成只屬於 Yen 的 `yen_mask`。

## 4. Score pipeline

Score registry 目前支援：

| Type / alias | 用途 | 常用參數 |
|---|---|---|
| `normalize` / `minmax` | Min-max normalize continuous score | `eps`，scope 由 policy 決定 |
| `median_filter` | Median filter | `kernel_size`, `mode` |
| `gray_dilation` | Grayscale dilation | `kernel_size`（default `3`，也可為各軸 tuple） |

Rewrite 範例：

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
```

如不需要某個分支的額外 transforms，使用 `pipeline: []`。所有 step 都保持 score shape。

只有 `mode: 2d` 會逐 slice filtering；Rewrite 中任何其他字串目前都落入 N-D SciPy median filter（通常是完整 `[H,W,Z]`），而不是被 validator 拒絕。`original_andi` 則只接受精確的 `3d`。新 config 應只使用明確的 `2d` 或 `3d`。

## 5. Adaptive Yen / Otsu

選擇方式：

```yaml
metrics:
  threshold_method: yen  # 或 otsu
```

或 standalone CLI override：

```powershell
python -B scripts\eval.py --config <eval.yaml> --run-eval --threshold-method otsu
```

行為：

- 每位 subject、每個 raw/MF 3D volume 分別計算 threshold。
- Dataset normalization scope 只控制 min/max 範圍，不會把 adaptive threshold 改成 dataset-global threshold。
- 判定固定為 `score > threshold`，不是 `>=`。
- Constant volume 記錄該常數為 threshold、輸出 empty mask並 warning。
- 非 finite score 先轉為零。

Dependency 行為：

- 有 scikit-image 時使用其 Yen/Otsu 實作。
- Rewrite Yen 在缺 scikit-image 時 warning 並 fallback 到 mean threshold。
- Otsu 在缺 scikit-image 時直接失敗。
- `original_andi` 建立時就要求 SciPy 與 scikit-image。

## 6. Binary mask pipeline

Mask registry 支援：

| Type / alias | 用途 | 重要參數 |
|---|---|---|
| `binary_dilation` | Binary morphology dilation | `rank`, `connectivity`, `iterations` |
| `connected_components` / `remove_small_components` | 移除過小 connected components | `min_size`, `connectivity` |

Adaptive mask：

```yaml
metrics:
  postprocess:
    binary_mask:
      pipeline:
        - type: binary_dilation
          rank: 3
          connectivity: 1
          iterations: 1
        - type: connected_components
          min_size: 20
          connectivity: 3
```

Fixed sweep mask：

```yaml
metrics:
  postprocess:
    threshold_mask:
      pipeline:
        - type: binary_dilation
          rank: 3
          connectivity: 1
          iterations: 1
```

兩條 pipeline 可相同，但不會自動互相繼承。若不需要 morphology，明確使用 `pipeline: []`。

Rewrite mode 在缺 SciPy/skimage 時，median/gray/binary dilation 或 component step 的部分實作可能 silent no-op；這是 optional dependency fallback，不代表結果仍與完整環境等價。

## 7. Metrics semantics

Evaluator 會對 raw 與 MF branches 各自產生 threshold sweep 與 adaptive metrics：

- Sweep thresholds：`np.arange(thr_start, thr_end, thr_step)`，end-exclusive，值 round 到三位。
- `threshold`：另用於固定 summary/export。Summary 的 sensitivity/specificity/precision 直接用 `score > threshold`，不套 `threshold_mask.pipeline`；sweep masks 與 exported fixed mask 才套該 pipeline。
- AUPRC：exact 或 sampled。
- Dice：逐 subject 計算後平均；predicted/ground-truth 都空時為 `0.0`。
- Sensitivity、specificity、precision：由所有 voxels 的 micro confusion counts 計算。
- Adaptive metrics：依 `threshold_method` 的 per-subject masks。

Original-style CSV 的實際欄位：

```csv
thr,value,dice,sensitivity,precision
```

`ANDi.csv` 對應 raw branch，`ANDi_mf.csv` 對應 MF branch。AUPRC、adaptive metrics 等 summary rows 仍使用同一 schema，不是「CSV index + value」格式。

若任何 subject 無 label，該次 evaluation 不會只略過該 subject，而是將整體 label-based metrics 視為 unavailable。

## 8. In-memory 與 disk streaming

### In-memory

Evaluator 先收集整個測試集的 raw maps 與 labels 到 CPU RAM，再執行 dataset/subject scope processing。這條路徑支援 Accelerate distributed gather。

### Disk streaming

```yaml
evaluation:
  memory_mode: disk_streaming
  cache:
    directory: outputs/eval_cache/my_run
    resume: true
    keep_on_success: true
  compute_auprc: true
  auprc_mode: sampled       # sampled 或 exact
  auprc_max_samples: 5000000
  auprc_seed: 73
  external_sort:
    chunk_bytes: 268435456  # exact 模式
```

Disk mode：

- 僅支援單程序，不能與 Accelerate/distributed evaluation 併用。
- Per-subject raw/label/MF arrays 與 manifest 寫入 cache。
- Dataset scope 透過多個 scans 求 global bounds；不是先把全部 maps 放 RAM。
- `resume: true` 會檢查 schema、fingerprints、subject order、label availability、shape/dtype 與檔案完整性。
- Mismatch/corruption 會停止，不會默默重用或自動覆寫。
- Threshold method 或 mask pipeline 改變可重用 score cache；score pipeline/config identity 改變則不能。
- `prediction_output.enabled: true` 時，會在 streaming metrics pass 逐 cached subject 匯出，不需要把全部 score 放入 RAM；檔案契約見 [Prediction export](#9-prediction-export)。
- `keep_on_success: false` 會在成功後刪除 cache directory。

`sampled` AUPRC 在 voxel 數超過上限時以 seeded、with-replacement sampling；未超過上限則使用全部 voxels。`exact` 使用磁碟 external merge sort，且不能同時要求 sample cap。

在相同設定與完整 dependencies 下，streaming 的目標是與 in-memory score/metric semantics 對齊；它的 storage、resume 與 global-normalization 執行方式不同。

## 9. Prediction export

```yaml
prediction_output:
  enabled: true
  directory: outputs/predictions
  normalization_scope:
    rewrite: subject
    original_andi: dataset
  restore_native_grid: true
  save_raw_score: true
  save_median_filtered_score: true
  binary_mask_source: score_mf
  save_binary_mask: true
```

每位 subject 的典型 artifacts：

```text
anomaly_score_raw.nii.gz
anomaly_score_mf.nii.gz
lesion_mask_<method>_raw.nii.gz
lesion_mask_<method>_mf.nii.gz
lesion_mask_<method>.nii.gz
lesion_mask_threshold.nii.gz           # save_threshold_mask: true
prediction_metadata.json
```

`<method>` 是 `yen` 或 `otsu`。Selected mask 由 `binary_mask_source` 決定。

若 prediction scope 等於 metric scope，可重用已處理 tensors；否則依 prediction scope 重跑 policy。`restore_native_grid: true` 需要 dataset metadata 的有效 `reference_path`；evaluator 從 reference NIfTI 讀取 native shape/affine/header，score 用 trilinear、mask 用 nearest resize。這不是 affine transform 或 registration，只是 shape restoration。

`train.py` 的 eval-after-fit 與 `eval_checkpoints50.py` 不完整轉送 top-level `prediction_output`；需要這些 exports 時應使用 standalone `scripts/eval.py`。

## 10. Numerical safety 與 edge cases

- Raw score 中 `NaN`、`+Inf`、`-Inf` 會 sanitize 成 `0`。
- Rewrite min-max denominator 使用 `eps` 防止除零；constant tensor 變成全零。
- Constant adaptive-threshold input 產生 empty mask。
- Threshold 一律 strict `>`。
- Dice 的雙空 mask結果是 `0.0`。
- Geometric anomaly aggregator 本身目前沒有使用 config 中的 `eps` clamp；它不屬於 postprocess numerical safety。
- Report writer 是 best-effort；report 缺失不一定代表 evaluation 失敗，應同時檢查 console、CSV/JSON 與 process exit。

## 11. Reports 與可追溯性

目前 generator 會嘗試寫入：

```text
inference_report.md
inference_report.json
inference_metrics_summary.csv
```

內容通常包含 postprocess mode、normalization scope、score/mask pipelines、threshold method、evaluation/cache/AUPRC 設定、model/config metadata 與 metrics summary。欄位會隨 generator 版本演進；`outputs/**/inference_report.md` 是 frozen run artifact，不應視為 current schema 或人工更新。

判讀實驗時應一起保存：原始 YAML、git commit、checkpoint identity、spectrum stats identity、CSV/JSON、prediction metadata 與 warnings。
