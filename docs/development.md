# 開發、執行與驗證

本文件說明如何在目前 repository 形態下工作。架構請見 [architecture.md](architecture.md)，YAML 請見 [configuration.md](configuration.md)。

## 1. Repository 執行模型

專案沒有 package/build manifest、console entrypoint 或已鎖定環境。主要使用方式是從 repository root 直接執行 `scripts/*.py`；大多數 scripts 會呼叫 `_bootstrap.py`，把 package parent 加入 `sys.path`。

```powershell
cd <path-to>\andi_rewrite
python -B scripts\train.py --help
python -B scripts\eval.py --help
```

`-B` 不是必要參數，只是避免產生 `.pyc`。根目錄的 `compare_npz_spectra.py` 與少數 figure scripts 使用不同的 import bootstrap，應保持在 repository root 執行。

## 2. Python 與 dependencies

Source 使用 `X | None` 等型別語法，因此至少需要 Python 3.10。Repository 沒有宣告支援版本或 lockfile；下表是功能分層，不是安裝 lock：

| 層級 | Dependencies | 何時需要 |
|---|---|---|
| 核心訓練 | PyTorch、torchvision、NumPy、pandas、PyYAML、Pillow | model、DDPM、DataLoader、config、sample image |
| MRI / LMDB | nibabel、lmdb | NIfTI datasets、healthy-slice store、spectrum statistics |
| Evaluation | SciPy、scikit-image | median/morphology、Yen/Otsu；`original_andi` 必要 |
| AUPRC | scikit-learn | 首選 AP 實作；缺少時部分路徑有本地 fallback |
| Figures / diagnostics | Matplotlib | visualization、comparison、noise-diagnostic scripts；Markdown/JSON reporting 本身不需要它 |
| Progress | tqdm | optional；缺少時使用內建 stderr reporter |
| Distributed | Hugging Face Accelerate | `runtime.accelerate` / `runtime.distributed` |
| Registration | DIPY、SimpleITK/elastix | `prepare_data.py` / registration helpers |
| Tests | pytest | 建議 runner；tests 本身以 `unittest` 撰寫 |

PyTorch/CUDA 應依平台與 GPU driver 安裝。不要把任意一台機器的 `pip freeze` 當作 repository 的支援矩陣。

### 本次文件驗證環境快照

2026-08-12 在既有 `ANDi` Conda environment 解析到：Python 3.11、PyTorch 2.1.0+cu118、torchvision 0.16.0+cu118、NumPy 1.24.1、pandas 2.1.1、PyYAML 6.0.1、nibabel 5.1.0、lmdb 1.4.1、SciPy 1.11.3、scikit-image 0.22.0、scikit-learn 1.3.1、Matplotlib 3.8.0、Accelerate 0.23.0 與 SimpleITK 2.3.0。這只記錄測試當下可工作的組合，不是版本保證。

## 3. 建議驗證順序

### 靜態/CLI smoke

```powershell
python -B scripts\train.py --help
python -B scripts\eval.py --help
```

CLI help 不會讀取大型 dataset。相反地，只執行 `train.py --config ...` 即使沒有 action flag，仍會建立 DataLoader 與 components，所以 data paths 必須有效。

### Unit/integration-style tests

```powershell
python -B -m pytest -q -p no:cacheprovider
```

或只使用標準函式庫 discovery：

```powershell
python -B -m unittest discover -s tests -v
```

本次完整 pytest 結果：`98 passed`，另有 `4 subtests passed`。Warnings 包含第三方 Matplotlib/pyparsing deprecation、legacy postprocess migration warning，以及 synthetic constant-score threshold warnings；沒有 test failure。

### 需要外部 artifacts 的 smoke

```powershell
python -B scripts\train.py --config <train-config.yaml> --run-one-step
python -B scripts\eval.py --config <eval-config.yaml> --run-eval
```

這些命令不是 repository-only test：config 必須指向真實 LMDB/NIfTI、checkpoint 與 optional spectrum NPZ。正式 evaluation 前確認 `model.checkpoint` 非空，否則程式會使用 random-initialized model。

## 4. Script inventory

### 核心 composition roots

| Script | 用途 | 主要 flags |
|---|---|---|
| `scripts/train.py` | 建立並執行 Trainer | `--config`, `--run-one-step`, `--fit`, `--sample-once`, `--eval-config` |
| `scripts/eval.py` | Standalone volume evaluation | `--config`, `--run-eval`, `--threshold-method {yen,otsu}` |
| `scripts/run_5fold.py` | 準備 folds/config 並依序訓練 | `--dry-run`, `--configs-only`, `--only-folds` 等 |

### 資料前處理

| Script | 用途 | 寫入/破壞性注意事項 |
|---|---|---|
| `scripts/split_healthy.py` | BraTS subjects → healthy slice CSV + LMDB | `--overwrite` 會遞迴刪除指定 LMDB directory |
| `scripts/split_healthy_kfold.py` | Subject folds + per-fold LMDB | 同樣有 overwrite 行為；combined windows 可重疊 |
| `scripts/prepare_data.py` | Shifts/MSSEG folder organization、registration、histogram matching | histogram normalization 會原地覆寫符合的 NIfTI |
| `scripts/rebuild_healthy_slices_metadata_tmp.py` | 從 CSV/LMDB 修補 metadata 的暫用 helper | 名稱與用途均為 maintenance，不是一般 pipeline |

### Spectrum 與 noise diagnostics

| Script | 用途 |
|---|---|
| `scripts/compute_lmdb_spectrum.py` | Healthy LMDB → empirical amplitude/power NPZ + metadata |
| `scripts/compare_noise_statistics.py` | 比較 noise samplers 的 moments、correlation、PSD 與 figures |
| `scripts/make_noise_comparison_figure.py` | 產生單張 noise/spectrum comparison figure |
| `compare_npz_spectra.py` | 比較多個 spectrum NPZ 與指定 LMDB slice |

### Evaluation、analysis 與 figures

| Script | 用途 |
|---|---|
| `scripts/eval_checkpoints50.py` | 依 pattern 批次評估 checkpoints；config forwarding 與 standalone eval 不完全相同 |
| `scripts/compare_otsu_yen_5_cases.py` | 比較固定五 cases 的 Otsu/Yen artifacts |
| `scripts/make_comparison_figures.py` | 將 NIfTI predictions 與 CSV/metadata 組成單張、比較或 batch figures |
| `scripts/inspect_ljubljana_ms.py` | 產生 Shifts/Ljubljana dataset inventory，並可做 loader/inference smoke |
| `scripts/empirical_stability.py` | 特定 empirical-spectrum 40-epoch experiment state machine；含硬編碼 workflow assumptions |

`scripts/_bootstrap.py` 是 import helper，不是 standalone entrypoint。

## 5. 常見工作流

### Healthy LMDB

先用相對或已人工確認的 paths：

```powershell
python -B scripts\split_healthy.py `
  -d <brats-root> `
  -i .\splits\BraTS21\scans_train.csv `
  -o <healthy-slices.csv> `
  --lmdb-dir <healthy-lmdb-dir>
```

需要覆寫時才加入 `--overwrite`。在 z-balanced mode 可加 `--z-balanced --per-z-count <n> --balance-seed <seed>`。

### Empirical spectrum

```powershell
python -B scripts\compute_lmdb_spectrum.py `
  --lmdb-path <healthy-lmdb-dir> `
  --out <spectrum.npz>
```

輸出包含 mean amplitude/power、radial counterparts、shape/channel metadata 與 `.metadata.json` sidecar。`EmpiricalSpectrumNoise` 會驗證 statistics 與 requested H/W/C。

### Training / resume / sampling

```powershell
python -B scripts\train.py --config <train.yaml> --run-one-step
python -B scripts\train.py --config <train.yaml> --fit
python -B scripts\train.py --config <train.yaml> --sample-once
```

Resume path 設在 `training.checkpoint.resume`。Checkpoint 不保存完整 run YAML、RNG 或 DataLoader state；保留實際使用的 config 檔與 git commit，不能只依 checkpoint payload 還原實驗。

### Standalone evaluation

```powershell
python -B scripts\eval.py --config <eval.yaml> --run-eval
python -B scripts\eval.py --config <eval.yaml> --run-eval --threshold-method otsu
```

需要 prediction export、完整 evaluation-cache fingerprint 或最完整 report metadata 時，使用這條 canonical path。`train.py` eval-after-fit 與 `eval_checkpoints50.py` 沒有轉送所有 top-level evaluation sections。

## 6. Extension points

### 新增 model

1. 在 `models/` 實作 `nn.Module`，遵循 `model(x_t, timesteps) -> predicted_noise`。
2. 在 `models/factory.py::build_model` 加入 canonical type/aliases。
3. 從 `models/__init__.py` export（若它是 public interface）。
4. 增加 shape、config、checkpoint load 與至少一個 forward test。

注意 ANDiUNet 的 attention spatial sizes 與 `image_size` 有關；不同 model 仍必須維持 input/output channels 相同，才能預測 noise。

### 新增 noise sampler 或 plan

1. 繼承 `noise/base.py::BaseNoise` 並實作 `sample(shape, device, dtype)`。
2. 在 `noise/factory.py::build_noise_sampler` 註冊 type。
3. 若是新的 epoch policy，實作 `NoisePlan` 並更新 `build_noise_plan`。
4. 測試 dtype/device/seed/shape、normalization 與 invalid config。

Detector inference 目前不提供 epoch；需要在 inference 選擇 switch branch 時，必須明確改變 contract，不能假設 training 的 epoch 會存在。

### 新增 dataset adapter

1. 在 `data/datasets.py` 實作 PyTorch Dataset。
2. Training slice 回傳 `[C,H,W]`；volume evaluation 可回傳 `(image, mask)`、`(image, mask, metadata)` 或 evaluator 支援的 dict batch。
3. Volume image/mask contract 分別是 `[C,H,W,Z]` / `[H,W,Z]`。
4. 在 `build_dataset` 加 type/aliases，視需要從 `data/__init__.py` export。
5. 測試 discovery、channel semantics、missing files、geometry、metadata、native restore。

若 adapter 支援 unlabeled data，metadata 的 label availability 必須一致；混合 labeled/unlabeled subjects 會使整次 label metrics unavailable。

### 新增 aggregation

繼承 `BaseAggregator`，使用 `register_aggregator()` 註冊 canonical 名稱/aliases，再測試 axis、shape、weights 與 numerical edge cases。Registry 是 process-local Python state，不是從外部動態載入。

### 新增 postprocess step

- Continuous score step：`register_score_postprocessor()`。
- Binary mask step：`register_mask_postprocessor()`。
- Step 必須保持 shape，接受/回傳 tensor，並明確處理 dtype、device、NaN/Inf 與 optional dependencies。
- 同時驗證 in-memory、disk streaming 與 prediction-export scope；修改 score pipeline 會改 score-cache fingerprint，修改 threshold method/mask pipeline 則可重用 score cache。

## 7. Tests 與 coverage map

| Test module | 主要 coverage |
|---|---|
| `test_dataset_paths.py` | BraTS/Shifts naming、discovery、factory |
| `test_ucsf_pdgm_dataset.py` | UCSF completeness、selection、channel order、geometry、resize |
| `test_empirical_spectrum.py` | spectrum modes/methods、normalization、seed、dtype/device |
| `test_noise_statistics.py` | moments、correlation、radial PSD、determinism |
| `test_postprocess_pipeline.py` | rewrite/original、Yen/Otsu、numerical safety、predictions/reports |
| `test_streaming_evaluator.py` | parity、resume、fingerprint、corruption、external AUPRC |
| `test_make_comparison_figures.py` | artifact validation、figure layout contract |

新功能若碰到 Trainer、checkpoint、scheduler、DDPM reverse loop、registration 或 distributed execution，應補測試；這些是目前 coverage 最薄的區域。

## 8. Generated artifacts 與文件政策

維護中的 Markdown 只放在 repository root 與 `docs/`。以下檔案由程式產生，不應手改：

- `outputs/metrics/**/inference_report.md`
- `outputs/checkpoints/**/training_report.md`
- noise/diagnostic/stability inventory reports under `outputs/`
- `.pytest_cache/README.md`

`outputs/**` 可能包含 generator 產生的 reports，也可能包含 frozen/manual historical captures；整個目錄都被忽略並視為 runtime artifacts。可重建的內容應從 `utils/reporting.py` 或對應 diagnostic script、同層 JSON/CSV、frozen config 與 checkpoint regenerate；人工快照只應 archive。舊 report schema 可能與目前 generator 不同，兩者都不能當成 current API contract。

`outputs/outputs_report.md` 是一份沒有可追溯 generator 的舊 static inventory，其 counts/paths 已過時；`outputs/reports/ucsf_pdgm_integration_report.md` 也只是歷史 integration artifact。兩者都被忽略且不應取代本文件或現行 source。

## 9. Debugging checklist

- Import error：確認從 repository root 執行、Python environment 包含對應 optional dependency。
- Dataset empty/path error：檢查 YAML 的絕對路徑、CSV 第一欄、subject naming、complete modality set。
- Channel/shape error：對齊 `data.channels`、modality order、model in/out channels、image size、spectrum H/W/C。
- Evaluation 結果不合理：確認 checkpoint 非空且 `use_ema` 是否符合預期；確認 `[0,1] -> [-1,1]` 設定。
- Streaming resume 被拒：保留原 cache 查 manifest，核對 checkpoint/split/stats path-size-mtime、subject order、score pipeline；不要手動混用 cache files。
- 沒有 label metrics：確認每一位 subject 都回傳有效 label availability。
- Otsu / original mode failure：確認 SciPy 與 scikit-image 已安裝。
- GPU nondeterminism：關閉 cuDNN benchmark、啟用 deterministic flag仍不代表 bitwise 保證；記錄實際 hardware/software/RNG state。
- Report 缺失但 run 完成：report writer 是 best-effort；先檢查 console warning、checkpoint/CSV 與 JSON。
