# 2026 年 9 月實驗成果整理

> 整理範圍：2026-09-01 至 2026-09-13（Asia/Taipei）。內容依各次 run 的 `training_report.json`、`inference_report.json`、CSV、資料集 build report 與診斷報告整理。
>
> 除非另有註明，模型為三模態 FLAIR/T1/T2、輸入 128×128、seed=73。報告中的 `AUPRC` 實際是 sampled sklearn Average Precision（最多 5,000,000 個 voxel、seed=73）；`MF` 是 3D median filter，kernel=5。`best Dice` 是 threshold sweep 的最佳 Dice，不是固定 Yen threshold 的 Dice。

## 一、月度結論

1. 原始 FOMO/SRI24 empirical-spectrum 模型在 BraTS21 test251 的 raw/MF AP 為 **0.4451/0.4702**；9 月的稽核指出，主要候選問題是資料域差異、全 volume 強度統計與訓練更新量不同，尚未完成可定量歸因的公平消融。
2. Gaussian 對照的 raw AP 降到 **0.2711**，但 median filter 後為 **0.4881**，顯示 noise model 與空間後處理存在交互作用，不能只用 raw AP 判定 empirical 一定較好。
3. Histogram matching、z-bucket sampling 與 robust-IQR normalization 的探索性結果大多落在 **0.67–0.79** 的 MF AP 範圍；其中 FOMO robust-IQR continuation 在 test251 得到 **0.7851**，但與舊 baseline 的資料處理、訓練量與 split 不完全相同，不能直接宣稱因果改善。
4. MPI、OASIS3 與混合 healthy-data 實驗均完成；在固定 50 個 BraTS subject 的診斷性評估中，MPI 的 MF AP 最高為 **0.7537**，OASIS3 為 **0.7093**，混合資料 20 epoch 為 **0.6689**。
5. Mixed 20→200 epoch checkpoint sweep 在 test50 的 MF AP 於 epoch 160–180 約 **0.755** 達峰，epoch 200 降至 **0.7457**；延長訓練沒有呈現單調提升。
6. MPI/OASIS3 dataset build、mixed LMDB 合併、healthy-data smoke test 與 20 組 tumor-free input comparison 均通過；healthy-only audit 本身**沒有新增模型訓練或 inference**。

## 二、主要模型實驗結果

| 日期 | 實驗與設定 | 訓練資料（train/val） | 評估 split | Raw AP / MF AP | Raw best Dice / MF best Dice | 判讀 |
|---|---|---:|---|---:|---:|---|
| 09-01 | FOMO/NIMH empirical，233 epochs | 33,798 / 1,777 | BraTS test50 | 0.4802 / 0.5090 | 0.4564 / 0.4693 | NIMH subset baseline；供後續探索比較 |
| 09-02 | FOMO/SRI24 Gaussian，233 epochs | 30,784 / 3,369 | BraTS test251 | 0.2711 / 0.4881 | 0.3080 / 0.5062 | Gaussian noise control |
| 09-05 | FOMO/SRI24 empirical，233 epochs | 30,784 / 3,369 | BraTS test251 | 0.4451 / 0.4702 | 0.4765 / 0.4934 | 原始 FOMO cross-domain baseline |
| 09-07 | FOMO robust-IQR pilot，20 epochs | 30,784 / 3,369 | BraTS val50 | 0.6863 / 0.7475 | 0.6233 / 0.6726 | normalization pilot；val50 來自 BraTS training split |
| 09-07 | Histogram match + 29 z-buckets，20 epochs | 69,000 / 3,369 | BraTS test50 | 0.6749 / 0.6955 | 0.6549 / 0.6732 | 分布匹配與 z-bucket sampling 探索 |
| 09-08 | Histogram match，epoch 20 checkpoint | 30,784 / 3,369 | BraTS test50 | 0.6671 / 0.6875 | 0.6517 / 0.6719 | 40-epoch run 的中途 checkpoint |
| 09-08 | Histogram match，epoch 40 checkpoint | 30,784 / 3,369 | BraTS test50 | 0.6789 / 0.6985 | 0.6563 / 0.6704 | 比 epoch 20 raw AP 上升；MF best Dice 略低 |
| 09-09 | FOMO robust-IQR continuation，至 epoch 199 | 30,784 / 3,369 | BraTS test251 | 0.7530 / 0.7851 | 0.6575 / 0.6889 | 本月 FOMO robust-IQR 的 test251 最佳完整評估 |
| 09-11 | MPI robust-IQR，60 epochs | 14,475 / 1,686 | BraTS test50 | 0.6893 / 0.7537 | 0.5834 / 0.6343 | healthy cohort cross-domain 探索 |
| 09-11 | OASIS3 robust-IQR，20 epochs | 39,082 / 4,514 | BraTS test50 | 0.6547 / 0.7093 | 0.5689 / 0.6100 | healthy cohort cross-domain 探索 |
| 09-11 | MPI+OASIS3+FOMO mixed robust-IQR，20 epochs | 84,341 / 9,569 | BraTS test50 | 0.5903 / 0.66890.744 | 0.5724 / 0.6219 | 混合資料的短訓練 baseline |
| 09-13 | MPI+OASIS3+FOMO mixed robust-IQR continuation，至 epoch 199 | 84,341 / 9,569 | BraTS test251 | 0.7056 / 0.7443 | 0.6466 / 0.6777 | full test251 評估；低於 FOMO-only robust-IQR |

### 2.1 非 9 月新訓練、但在 9 月作為參考的 BraTS 模型

BraTS healthy-slice empirical model 是 8 月完成的 in-domain reference，9 月 AUPRC 稽核重新使用其報告：BraTS test251 raw/MF AP **0.8341/0.8478**，best Dice **0.7867/0.8014**。它不是 9 月新跑的訓練，因此不列入上表的 9 月新實驗；它的用途是對照 FOMO cross-domain 結果。

## 三、各實驗與可核對成果

### 3.1 FOMO/NIMH baseline 與 Gaussian noise control

- 09-01 的 NIMH empirical run 完成 233 epochs，final train loss=0.003118、validation loss=0.006563；在同一個 BraTS test50 上得到 raw/MF AP=0.4802/0.5090。
- 09-02 的 Gaussian run 使用相同三模態 FOMO/SRI24 train/val 規模與 233 epochs；final train/validation loss=0.006087/0.006972。
- Gaussian 在 raw score 上比 09-05 empirical baseline 低約 0.174 AP，但 MF AP 反而高約 0.018。這表示 noise choice、anomaly score 與 median filtering 的效果不能拆開解讀。

證據：

- [NIMH inference report](../outputs/runs/fomo45k_nimh_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test50/inference_report.md)
- [Gaussian training report](../outputs/runs/fomo45k_sri24_flair_t1_t2_gaussian233/training_report.json)
- [Gaussian inference report](../outputs/runs/fomo45k_sri24_flair_t1_t2_gaussian233/evaluation/brats21_test251/inference_report.md)
- [FOMO empirical inference report](../outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/inference_report.md)

### 3.2 AUPRC 差異根因稽核（09-05）

這是診斷與重新計分，不是新的 model training。已確認的結果如下：

- FOMO empirical：raw/MF AP=0.4451/0.4702。
- BraTS healthy-slice reference：raw/MF AP=0.8341/0.8478。
- FOMO 訓練更新量為 137,936 steps；BraTS 為 310,123 steps，約為 FOMO 的 2.25 倍。
- BraTS audit 中 training/test 不重疊，但 validation/test 為 **251/251 subject 重疊**；若 validation 曾參與 checkpoint 或超參數選擇，不能把該 test 結果當成完全未使用的 final holdout。
- 12 位 BraTS training subject 的診斷中，FLAIR 的 `p99(full foreground) / p99(seg==0)` 中位數為 1.221，範圍 1.000–1.588；這支持 full-volume normalization 可能把病灶相關強度尺度帶入健康切片，但尚未量化它對 AP 差距的貢獻。
- 在匹配 z 的抽樣中，FOMO 與 BraTS 的正值前景均值仍有差異：FLAIR 0.651/0.482、T1 0.551/0.665、T2 0.374/0.360。這是描述性證據，不是因果證明。

目前最合理的解釋排序是資料域/強度分布差異、有效訓練量不同、切片選擇與 empirical spectrum 差異；AUPRC 計算錯誤或背景區域不是目前最有證據的主因。

證據：

- [AUPRC issue analysis](../AUPRC_issue_analysis.md)
- [Root-cause research report](../artifacts/auprc_root_cause_20260905/research_report.md)
- [Root-cause audit JSON](../artifacts/auprc_root_cause_20260905/audit.json)
- [Evidence summary figure](../artifacts/auprc_root_cause_20260905/evidence_summary.png)

### 3.3 Histogram matching 與 z-bucket sampling

資料處理結果：

- Histogram-matched dataset build status=PASS；以 888 個 BraTS training subjects 建立 target curves，保留 FOMO 原本 30,784 train / 3,369 val entries。
- train/val 各 modality 的 CDF fixed-grid error 在 mapping 後約為 1.3–2.2×10^-6，Wasserstein/reference-IQR ratio 約為 3.9–7.6×10^-6。
- z-bucket dataset status=PASS；29 個 z buckets、69,000 train entries、3,369 val entries、28,309 個 distinct source slices，單一 source slice 最多重複 7 次；每筆內容通過 byte verification。

模型結果顯示，histogram match 20/40 epochs 與 z-bucket 20 epochs 在 test50 的 MF AP 約為 0.6875–0.6985。與 NIMH baseline 的差異同時混合了 normalization、資料版本、epoch budget 與取樣方式，所以只能記錄為 promising exploratory result，不能單獨歸因為 histogram matching 或 z-bucket 的效果。

證據：

- [Histogram-match build report](../outputs/datasets/fomo45k_sri24_brats_histmatch/build_report.json)
- [z-bucket build report](../outputs/datasets/fomo45k_sri24_brats_histmatch_zbucket29_69k/build_report.json)
- [Histogram-match documentation](fomo_brats_histmatch.md)
- [z-bucket documentation](fomo_zbucket69k.md)
- [z-bucket inference report](../outputs/runs/fomo45k_sri24_brats_histmatch_zbucket29_69k_empirical_spectrum20/evaluation/brats21_test50/inference_report.md)
- [Histogram-match epoch 40 inference report](../outputs/runs/fomo45k_sri24_brats_histmatch_empirical_spectrum40/evaluation/brats21_test50_epoch0040/inference_report.md)

### 3.4 Robust-IQR normalization 與 FOMO continuation

使用的 normalization 是每個完整 3D volume、每個 modality 分別計算 `(x - median) / IQR`，背景固定為 -1，不 clipping，且 model `normalize_input=false`。這次處理不使用 segmentation label 來計算 normalization。

- FOMO robust-IQR pilot：243 sessions，30,784/3,369 entries；preflight 與資料 build 均 PASS。20 epochs 後在 BraTS training split 中抽出的 val50 得到 raw/MF AP=0.6863/0.7475。
- FOMO robust-IQR continuation：由 20-epoch checkpoint 接續至 epoch 199；final train/validation loss=0.010184/0.014187。在 BraTS test251 得到 raw/MF AP=0.7530/0.7851。
- 這兩個結果的評估 split 不同：pilot 是 val50，continuation 是 test251；因此不能直接把 0.7475 與 0.7851 當成單純的 epoch gain。

證據：

- [Robust-IQR documentation](robust_iqr_b.md)
- [Robust-IQR dataset build report](../outputs/datasets/fomo45k_sri24_robust_iqr/build_report.json)
- [Robust-IQR preflight](../outputs/diagnostics/robust_b/preflight.json)
- [Pilot inference report](../outputs/runs/fomo45k_robust_iqr20_b/evaluation/brats_val50/inference_report.md)
- [Continuation training report](../outputs/runs/fomo45k_robust_iqr200_continue20_resume2/training_report.json)
- [Continuation test251 inference report](../outputs/runs/fomo45k_robust_iqr200_continue20_resume2/evaluation/brats21_test251/inference_report.md)

### 3.5 MPI、OASIS3 與 mixed healthy-data 實驗

#### Dataset build 與資料量

| Dataset | Sessions | Train participants / entries | Val participants / entries | Build |
|---|---:|---:|---:|---|
| MPI robust-IQR | 115 | 103 / 14,475 | 12 / 1,686 | PASS |
| OASIS3 robust-IQR | 311 | 221 / 39,082 | 25 / 4,514 | PASS |
| FOMO robust-IQR | 243 | 216 / 30,784 | 24 / 3,369 | PASS |
| Mixed MPI+OASIS3+FOMO | — | 84,341 | 9,569 | PASS；每筆內容與 source byte-compare |

OASIS3 的前處理前 audit 找到 136 個 duplicate visits、195 個 duplicate groups；其中 112 個 T2 是不同 protocol，沒有發現 identical-file extra copies；另有 1 個 same-time/different-voxel case。後續 pipeline 以唯一且可判定的 modality/visit 為原則處理。

#### 模型結果

- MPI 60 epochs：test50 raw/MF AP=0.6893/0.7537。
- OASIS3 20 epochs：test50 raw/MF AP=0.6547/0.7093。
- Mixed 20 epochs：test50 raw/MF AP=0.5903/0.6689。
- Mixed continuation 至 epoch 199：test251 raw/MF AP=0.7056/0.7443。

所有 09-10 至 09-13 的 healthy training/evaluation sequence status 均為 COMPLETE，個別 stage status=PASS。

證據：

- [Healthy SRI24 processing documentation](healthy_sri24.md)
- [Healthy batch status](../outputs/reports/healthy_sri24_batch.json)
- [MPI build report](../outputs/datasets/mpi_sri24_robust_iqr/build_report.json)
- [OASIS3 build report](../outputs/datasets/oasis3_sri24_robust_iqr/build_report.json)
- [Mixed dataset build report](../outputs/datasets/mpi_oasis3_fomo45k_sri24_robust_iqr/build_report.json)
- [Healthy training sequence status](../outputs/reports/healthy_training_sequence/status.json)
- [Mixed epoch 200 test251 report](../outputs/runs/mixed_sri24_robust_iqr200_continue20/evaluation/brats21_test251/inference_report.md)

## 四、Mixed checkpoint sweep（09-13）

以下所有列均來自同一個 `scans_test_50.csv` 的 50-subject 診斷性評估；epoch 20 的結果沿用既有輸出，其餘 checkpoint 重新完成評估。這組結果適合看訓練趨勢，不適合當作獨立 final test claim。

| Checkpoint | Raw AP | MF AP | Raw best Dice | MF best Dice |
|---:|---:|---:|---:|---:|
| 20 | 0.5903 | 0.6689 | 0.5724 | 0.6219 |
| 40 | 0.6439 | 0.7080 | 0.5921 | 0.6349 |
| 60 | 0.6531 | 0.7121 | 0.5942 | 0.6349 |
| 80 | 0.6584 | 0.7128 | 0.5903 | 0.6272 |
| 100 | 0.6833 | 0.7321 | 0.6056 | 0.6421 |
| 120 | 0.6881 | 0.7367 | 0.6069 | 0.6432 |
| 140 | 0.6932 | 0.7388 | 0.6046 | 0.6395 |
| 160 | 0.7118 | **0.7552** | 0.6149 | 0.6496 |
| 180 | **0.7119** | 0.7551 | **0.6155** | **0.6499** |
| 200 | 0.7007 | 0.7457 | 0.6093 | 0.6425 |

結論：在這個 50-subject sweep 中，性能約於 epoch 160–180 飽和，epoch 200 下降；因此後續應以獨立 validation/holdout 做 checkpoint selection，不應直接以這個 test50 sweep 選最佳 epoch。

證據：[完整 checkpoint metrics CSV](../outputs/reports/mixed_checkpoints50/metrics_comparison.csv)

## 五、頻譜與輸入資料診斷

### 5.1 Healthy spectra comparison（09-10）

以 radial power distribution 比較各 healthy cohort 與既有 FOMO45K/BraTS legacy spectrum。下表為 Jensen–Shannon divergence（bits），越小表示 radial distribution 越接近；這不是 anomaly detection performance。

| Dataset | FLAIR：vs BraTS / vs FOMO | T1：vs BraTS / vs FOMO | T2：vs BraTS / vs FOMO |
|---|---:|---:|---:|
| MPI | 0.0520 / 0.0049 | 0.0873 / 0.0056 | 0.0117 / 0.0018 |
| OASIS3 | 0.0333 / 0.0006 | 0.0275 / 0.0096 | 0.0016 / 0.0149 |
| Mixed | 0.0356 / 0.0001 | 0.0415 / 0.0017 | 0.0068 / 0.0047 |

Mixed spectrum 由 FOMO、MPI、OASIS3 組成，因此接近 FOMO 是合理但不能被解讀為性能因果證據；BraTS legacy spectrum 使用不同的 preprocessing/mask convention，絕對 power 也沒有比較。

證據：[spectrum comparison README](../outputs/reports/healthy_noise_spectra/README.md)、[metrics CSV](../outputs/reports/healthy_noise_spectra/comparison_metrics.csv)、[comparison report JSON](../outputs/reports/healthy_noise_spectra/comparison_report.json)

### 5.2 Robust-IQR model input comparison（09-13）

以實際模型輸入流程比較 FOMO、MPI、OASIS3 與 BraTS 的 tumor-free slices：完整 volume median/IQR、背景 -1、既有 resize，沒有再次 `2*x-1`、逐張 normalization 或 histogram matching。

- FOMO、MPI、OASIS3 各完成 20 組配對；每組 3 個 z positions。
- 每個 BraTS 入選 slice 的 native segmentation 與 model-grid lesion mask 都是全零；這些是腫瘤病人的無腫瘤標註切片，不等同 healthy subject。
- 所有模型輸入 shape、finite values、affine 與模態順序通過檢查；展示陣列與既有 preprocessing 精確相等，最大 absolute error=0。

證據：

- [FOMO 20-pair README](../outputs/diagnostics/robust_b/input_comparison_tumor_free_20/README.md)
- [MPI 20-pair README](../outputs/diagnostics/robust_b/input_comparison_mpi_tumor_free_20/README.md)
- [OASIS3 20-pair README](../outputs/diagnostics/robust_b/input_comparison_oasis3_tumor_free_20/README.md)

## 六、小實驗、比較圖與 precision–recall

本節收錄 9 月產生的診斷性小實驗。它們用來檢查資料域、前處理與模型輸出差異；除非明確寫成 test251/full evaluation，不能取代固定 holdout 的正式泛化評估。

### 6.1 20 cases 三模型逐 slice comparison（09-13）

目的：在同一批病例、同一組顯示規則下，比較 Mixed epoch 0199、BraTS21 4-modal epoch 0232 與 BraTS21 3-modal epoch 0232 的 mask。病例不是隨機抽樣，而是依 Mixed 模型的 Dice 從 0.0–1.0 每個 0.1 bin 選 2 個最接近中點的 case；因此這是分層視覺診斷，不是獨立 test set。

- 20 cases × 155 axial slices = **3,100 張**逐 slice JPEG。
- 每張 panel 包含 Original FLAIR、GT、三個模型 mask，以及各模型的 FP/FN/TP 圖；FP=紅、FN=藍、TP=綠。
- 在這組分層病例中，兩個 BraTS model 都有 15/20 cases 的 Dice 高於 Mixed；平均 Dice 差為 4-modal **+0.1299**、3-modal **+0.1161**。

| Model | Mean Dice | Median Dice | 勝過 Mixed 的 cases |
|---|---:|---:|---:|
| Mixed epoch_0199 | 0.4959 | 0.4955 | — |
| BraTS21 4-modal epoch_0232 | 0.6258 | 0.7481 | 15 / 20 |
| BraTS21 3-modal epoch_0232 | 0.6120 | 0.6767 | 15 / 20 |

![20 cases 三模型 per-case Dice 分布](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/dice_distribution/dice_distribution_three_models_selected20.png)

可核對檔案：

- [比較實驗 README](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/README.md)
- [Dice distribution PNG](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/dice_distribution/dice_distribution_three_models_selected20.png)
- [case-level Dice CSV](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/case_dice_comparison.csv)
- [20 cases selection CSV](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/selected_cases.csv)
- [代表性逐 slice FP/FN/TP 圖：BraTS2021_00160 slice 078](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/figures/bin_09_09-10/BraTS2021_00160/slice_078.jpg)

### 6.2 Precision / recall 小實驗（09-13）

這組數字使用上述相同的 20 cases，對每個模型的 `lesion_mask_yen_mf` 計算 precision 與 recall。`mean` 是先逐 case 計算再平均；`pooled` 是把 20 cases 的 TP/FP/FN 合併後計算。

| Model | Mean precision | Pooled precision | Mean recall | Pooled recall | Total TP | Total FP | Total FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixed epoch_0199 | 0.8681 | 0.9121 | 0.3980 | 0.4022 | 768,738 | 74,044 | 1,142,400 |
| BraTS21 4-modal epoch_0232 | 0.9608 | 0.9551 | 0.5201 | 0.5262 | 1,005,674 | 47,296 | 905,464 |
| BraTS21 3-modal epoch_0232 | 0.9610 | 0.9511 | 0.5082 | 0.5073 | 969,450 | 49,826 | 941,688 |

在這個 selected-20 diagnostic 中，兩個 BraTS model 的 precision 約為 0.96，mean recall 比 Mixed 高約 0.110–0.122；這與 Dice 分布中 BraTS model 較高的現象一致。但這不是完整的 precision–recall curve，也不是前文 sampled AP/AUPRC：它只代表固定 Yen threshold 加 median filter 後的一個 operating point。若要產生正式 PR curve，需在同一固定 split 輸出連續 anomaly score，再對一系列 threshold 計算 precision/recall。

可核對檔案：

- [precision/recall summary CSV](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/precision_recall_summary.csv)
- [case-level precision/recall CSV](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/case_precision_recall.csv)

### 6.3 Histogram matching 的 CDF、histogram 與 montage（09-07）

這是 preprocessing 的小型 QA，不是模型性能實驗：以 888 個 BraTS training subjects 建立 reference curves，檢查 FOMO train/val 的 histogram mapping 是否正確。

- 共 243 sessions 完成 mapped-slice reconstruction，`exact_reconstruction=true`；另人工檢視 12 個 visual cases。
- mapping 後的 train CDF fixed-grid error：FLAIR **2.24×10^-6**、T1 **1.27×10^-6**、T2 **2.24×10^-6**；val 分別為 **2.20×10^-6**、**1.43×10^-6**、**2.20×10^-6**。
- 視覺上 FLAIR 通常變暗、T1 變亮，符合 mapping 設計；沒有看到明顯 spatial displacement。這不能被解讀成 clinical 或 model-performance validation。

![Histogram matching CDF comparison](../outputs/diagnostics/fomo_brats_histmatch/cdf_comparison.png)

可核對檔案：

- [CDF comparison PNG](../outputs/diagnostics/fomo_brats_histmatch/cdf_comparison.png)
- [histogram comparison PNG](../outputs/diagnostics/fomo_brats_histmatch/histogram_comparison.png)
- [12-case montage contact sheet](../outputs/diagnostics/fomo_brats_histmatch/montage_contact_sheet.png)
- [distribution summary CSV](../outputs/diagnostics/fomo_brats_histmatch/distribution_summary.csv)
- [reconstruction verification JSON](../outputs/diagnostics/fomo_brats_histmatch/verification.json)

### 6.4 Robust-IQR input / intensity comparison（09-09 至 09-13）

以實際 model input pipeline 做 FOMO、MPI、OASIS3 與 BraTS 的小樣本視覺與數值比較。FOMO、MPI、OASIS3 各有 20 組 paired comparison，每組使用 3 個 z positions；另以 20,000 pixels/case/modality 的分布診斷檢查 foreground 與 all-pixel statistics。

- 20-pair input comparison 的 shape、finite values、affine、modality order 均通過；展示切片與既有完整-volume normalization + resize 結果的最大 absolute error 為 **0**。
- 這些圖仍可看到個體腦形、紋理與 modality intensity-tail 差異，說明 robust-IQR 能固定數值定義，但不會把不同 cohort 變成相同影像分布。
- BraTS 的 tumor-free slice 是腫瘤病人的無腫瘤標註切面，不等於 healthy subject；相同 z 也不是精確解剖配對。

![FOMO vs BraTS robust-IQR distribution comparison](../outputs/diagnostics/robust_b/distribution/fomo_brats_robust_comparison.png)

可核對檔案：

- [robust-IQR distribution comparison PNG](../outputs/diagnostics/robust_b/distribution/fomo_brats_robust_comparison.png)
- [z-balanced comparison PNG](../outputs/diagnostics/robust_b/distribution/fomo_vs_brats_zbalanced.png)
- [FOMO 20-pair README](../outputs/diagnostics/robust_b/input_comparison_tumor_free_20/README.md)
- [MPI 20-pair README](../outputs/diagnostics/robust_b/input_comparison_mpi_tumor_free_20/README.md)
- [OASIS3 20-pair README](../outputs/diagnostics/robust_b/input_comparison_oasis3_tumor_free_20/README.md)
- [一組 FOMO/BraTS input overview](../outputs/diagnostics/robust_b/input_comparison/overview.png)

## 七、流程驗證與未完成事項

### 已完成

- Healthy SRI24 smoke：3 個 real cases，train=284、val=146，normalization/reference parity 與 model dataset reader 檢查 PASS。
- MPI/OASIS3 full preprocessing、LMDB build、audit 與 mixed merge 均完成；mixed merge 為 84,341 train / 9,569 val，無 oversampling，所有 output value 都與 source byte-compare。
- 09-13 mixed checkpoint 40–200 的 9 個 inference jobs 全部 PASS。
- 9 月新增的 healthy-only evaluation workflow 完成 inventory/audit；`healthy_only_plan_20260908/audit.json` 明確記錄本階段是 report/source audit，沒有新增 training/inference。

### 尚未完成、不能由本月結果直接宣稱的事項

- 尚未完成相同 optimizer steps、相同 validation/holdout、相同 normalization 的 FOMO-vs-BraTS 公平對照。
- 尚未完成 data-domain × noise-spectrum 的 2×2 controlled ablation。
- 尚未建立完全未參與模型選擇的 BraTS final holdout；尤其要處理 validation/test subject overlap 的問題。
- 尚未證明 FLAIR p99、scanner/acquisition、slice-z sampling 或 spectrum 差異各自占 AP 差距多少。
- 因此目前應把結果標為 baseline、pilot、diagnostic 或 exploratory，不應把不同 split/epoch 的 AP 直接排名成正式泛化性能。

## 八、主要檔案索引

| 類型 | 檔案 |
|---|---|
| AUPRC 根因分析 | [AUPRC_issue_analysis.md](../AUPRC_issue_analysis.md) |
| FOMO empirical baseline | [training report](../outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/training_report.md)、[inference report](../outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/inference_report.md) |
| Gaussian control | [training report](../outputs/runs/fomo45k_sri24_flair_t1_t2_gaussian233/training_report.md)、[inference report](../outputs/runs/fomo45k_sri24_flair_t1_t2_gaussian233/evaluation/brats21_test251/inference_report.md) |
| Robust-IQR continuation | [training report](../outputs/runs/fomo45k_robust_iqr200_continue20_resume2/training_report.md)、[test251 inference](../outputs/runs/fomo45k_robust_iqr200_continue20_resume2/evaluation/brats21_test251/inference_report.md) |
| Healthy cohort training | [sequence status](../outputs/reports/healthy_training_sequence/status.json)、[MPI report](../outputs/runs/mpi_sri24_robust_iqr60_own_spectrum/evaluation/brats21_test50/inference_report.md)、[OASIS3 report](../outputs/runs/oasis3_sri24_robust_iqr20_own_spectrum/evaluation/brats21_test50/inference_report.md) |
| Mixed training/checkpoints | [test251 report](../outputs/runs/mixed_sri24_robust_iqr200_continue20/evaluation/brats21_test251/inference_report.md)、[checkpoint CSV](../outputs/reports/mixed_checkpoints50/metrics_comparison.csv) |
| Healthy spectra | [README](../outputs/reports/healthy_noise_spectra/README.md)、[PNG](../outputs/reports/healthy_noise_spectra/spectral_comparison.png) |
| 20-case 三模型比較 | [README](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/README.md)、[Dice PNG](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/dice_distribution/dice_distribution_three_models_selected20.png)、[PR summary CSV](../outputs/figures/brats21_selected20_three_model_comparison_epoch0232_vs_mixed_epoch0199/precision_recall_summary.csv) |
| Histogram matching QA | [CDF PNG](../outputs/diagnostics/fomo_brats_histmatch/cdf_comparison.png)、[montage PNG](../outputs/diagnostics/fomo_brats_histmatch/montage_contact_sheet.png)、[verification JSON](../outputs/diagnostics/fomo_brats_histmatch/verification.json) |
| Robust-IQR input comparison | [distribution PNG](../outputs/diagnostics/robust_b/distribution/fomo_brats_robust_comparison.png)、[FOMO 20-pair README](../outputs/diagnostics/robust_b/input_comparison_tumor_free_20/README.md) |
