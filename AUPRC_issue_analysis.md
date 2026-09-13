# 三模態 AUPRC 差異問題分析

## 1. 問題摘要

目前兩個三模態模型在 BraTS21 test set 上的 AUPRC 差異如下：

| 實驗 | 訓練資料 | Raw AUPRC | Median filter AUPRC |
|---|---|---:|---:|
| FOMO/SRI24 健康影像模型 | FOMO45K/SRI24 healthy data | 0.445 | 0.470 |
| BraTS healthy-slice 模型 | BraTS21 training cases 的健康切片 | 0.834 | 0.848 |

目前檢查結果顯示，主要問題不是三模態模型架構或 AUPRC 計算錯誤，而是訓練資料與測試資料的 domain 不一致，並且兩個實驗的實際訓練更新量也不相同。

---

## 2. 重要實驗與結果檔案位置

### FOMO/SRI24 三模態健康影像模型

訓練設定：

- [Training config](configs/train_fomo45k_sri24_flair_t1_t2_empirical_spectrum233.yaml)
- [Training report](outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/training_report.json)
- [Training metrics](outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/training_metrics.csv)

測試設定與結果：

- [Evaluation config](configs/eval_brats21_251_fomo45k_sri24_flair_t1_t2_empirical_spectrum233_epoch0232.yaml)
- [Inference report](outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/inference_report.md)
- [Inference report JSON](outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/inference_report.json)

主要數值：

- Training slices：30,784
- Validation slices：3,369
- 最後 train loss：約 0.00411
- 最後 validation loss：約 0.00652
- BraTS21 test raw AUPRC：約 0.445
- Median filter 後 AUPRC：約 0.470

### BraTS21 三模態 healthy-slice 模型

訓練設定：

- [Training config](configs/train_brats_flair_t1_t2_empirical_spectrum233.yaml)
- [Training report](outputs/runs/brats_flair_t1_t2_empirical_spectrum233/training_report.json)
- [Training metrics](outputs/runs/brats_flair_t1_t2_empirical_spectrum233/training_metrics.csv)

測試設定與結果：

- [Evaluation config](configs/eval_brats21_251_brats_flair_t1_t2_empirical_spectrum233_epoch0232.yaml)
- [Inference report](outputs/metrics/brats_flair_t1_t2_empirical_spectrum233_epoch0232_brats21_251/inference_report.md)
- [Inference report JSON](outputs/metrics/brats_flair_t1_t2_empirical_spectrum233_epoch0232_brats21_251/inference_report.json)

主要數值：

- Training slices：69,190
- Validation slices：18,946
- 最後 train loss：約 0.00260
- 最後 validation loss：約 0.00507
- BraTS21 test raw AUPRC：約 0.834
- Median filter 後 AUPRC：約 0.848

---

## 3. 發生問題的可能原因

### 原因一：訓練資料 domain 與測試資料 domain 不一致（最主要原因）

FOMO 模型的健康影像來自 FOMO45K/SRI24，而測試資料來自 BraTS21。也就是：

```text
FOMO/SRI24 healthy images  ->  BraTS21 test images
```

雖然兩者都是健康影像，且都使用 FLAIR、T1、T2 三個模態，但 MRI 的以下特徵可能不同：

- Scanner 與 acquisition protocol
- 強度分布與對比
- 解剖範圍與背景結構
- 影像紋理與 noise pattern
- 不同資料集的正常解剖變異

模型的 anomaly score 會把「未在 FOMO 訓練資料中出現的正常 BraTS 影像特徵」誤判成異常，造成大量 false positives。

相關檔案：

- [FOMO dataset audit](C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH/audit_report.json)
- [FOMO dataset builder](data/fomo45k/brats21_lmdb.py)
- [BraTS healthy-slice selection](data/healthy_slices.py)
- [BraTS healthy split script](scripts/split_healthy.py)
- [BraTS21 test split](splits/BraTS21/scans_test.csv)

關鍵差異：

- BraTS 模型使用 BraTS21 training cases 中的健康切片，訓練與測試屬於相同資料域。
- FOMO 模型使用另一個資料來源的健康影像，測試時跨 domain 到 BraTS21。

因此，BraTS 模型屬於 in-domain anomaly detection，而 FOMO 模型屬於 cross-domain anomaly detection。

### 原因二：相同 epoch 不代表相同訓練量

兩個實驗都訓練 233 epochs，但每個 epoch 的資料量不同：

| 實驗 | Training slices | Optimizer/EMA steps |
|---|---:|---:|
| FOMO/SRI24 | 30,784 | 約 137,936 |
| BraTS healthy slices | 69,190 | 約 310,123 |

BraTS 模型的實際更新步數約為 FOMO 模型的 2.25 倍。因此目前的比較同時混合了：

1. Dataset domain 差異
2. Training data quantity 差異
3. Optimizer update 次數差異

FOMO 模型的 validation loss 也較高，表示其在目前設定下可能尚未達到與 BraTS 模型相同的收斂程度。

### 原因三：三模態影像強度統計不同

兩個 LMDB 都使用 percentile-based normalization，但 normalization 不會完全消除不同資料集的 scanner、contrast 與紋理差異。

目前抽樣統計顯示，訓練資料的模態分布不同。例如近似平均值：

| Modality | FOMO/SRI24 | BraTS healthy |
|---|---:|---:|
| FLAIR | 0.123 | 0.064 |
| T1 | 0.105 | 0.088 |
| T2 | 0.072 | 0.050 |

相關 normalization 檔案：

- [Shared imaging normalization](data/imaging.py)
- [FOMO LMDB loader](data/fomo45k/brats21_lmdb.py)
- [BraTS dataset loader](data/datasets/brats.py)
- [Training pipeline](engine/trainer.py)
- [Inference pipeline](engine/evaluation/inference.py)

目前沒有發現明顯的「完全忘記 normalization」問題；比較像是兩個資料域在 normalization 後仍然保留不同的影像分布。

### 原因四：Noise spectrum 不同（次要因素）

兩個模型使用不同來源建立的 empirical spectrum：

- [FOMO empirical spectrum](C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH/spectrum/fomo45k_sri24_flair_t1_t2_empirical_spectrum.npz)
- [BraTS empirical spectrum](C:/ML/data/spectrum/brats21_healthy_empirical_spectrum.npz)

FOMO checkpoint 改用 Gaussian noise 時，AUPRC 約降至 0.271，代表 noise model 確實會影響結果。但 FOMO 使用自身 empirical spectrum 時仍只有約 0.445，因此 noise spectrum 不是造成 0.445 與 0.834 差距的主要原因。

### 原因五：測試資料與 metric 設定不是主要問題

目前兩份 inference report 使用：

- 相同的 BraTS21 251-subject test split
- 相同的三模態：FLAIR、T1、T2
- 相同的 timestep range 與 aggregation 設定
- 相同的 AUPRC sampled maximum 與 seed

因此，BraTS21 test set 的異常比例不同並不能解釋這個差距；兩個模型是在同一測試集合上比較。

---

## 4. 初步結論

最可能的原因排序如下：

1. **FOMO/SRI24 訓練資料與 BraTS21 測試資料存在明顯 domain shift。**
2. **FOMO 模型的實際訓練更新量只有 BraTS 模型約 45%。**
3. **兩個資料來源的模態強度、背景與紋理分布不同。**
4. **Empirical noise spectrum 會影響結果，但不是主要差距來源。**
5. **目前沒有證據顯示是 AUPRC 計算或三模態 channel 數量造成的錯誤。**

換句話說，BraTS 模型是在學習 BraTS 的正常分布；FOMO 模型則是在用 FOMO/SRI24 的正常分布判斷 BraTS，正常的跨資料集差異被模型當成異常。

---

## 5. 建議的驗證實驗

### 5.1 先做 domain cross-evaluation

測試以下組合：

```text
FOMO-trained model  -> FOMO held-out healthy data
FOMO-trained model  -> BraTS21 test data
BraTS-trained model -> BraTS21 test data
```

若 FOMO 模型在 FOMO domain 表現良好、但到 BraTS21 明顯下降，即可直接確認 domain shift。

### 5.2 使用相同 optimizer steps 比較

不要只比較 233 epochs。建議讓兩個模型使用相同更新步數，例如：

- 固定相同的 optimizer steps
- 或將 FOMO 訓練至約 520 epochs，使總更新量接近 BraTS 模型

### 5.3 固定相同 noise spectrum

讓兩個 checkpoint 都使用同一個 BraTS empirical spectrum，並固定完全相同的 inference 設定，以隔離 noise spectrum 的影響。

### 5.4 比較同一病例的 anomaly map

檢查 FOMO 模型是否在以下位置產生較高分數：

- 正常腦組織邊界
- 背景區域
- MRI contrast 差異較大的區域
- BraTS 中 FOMO 訓練資料未見過的正常結構

如果這些位置普遍被打高分，便能進一步驗證 false positive 主要來自 domain shift。

