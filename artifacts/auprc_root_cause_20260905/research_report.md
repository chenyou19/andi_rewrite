**三模態 AUPRC 差異：原因研究與解決方案**

研究日期：2026-09-05。以附件、目前本機程式、實際 checkpoint、LMDB、來源 metadata 與 inference cache 為依據。本次完成 CPU 稽核與計分重算，尚未執行新的訓練消融；下文將已確認事實與因果假說分開說明。

**結論：優先處理 FLAIR／T1 的資料分布與正規化差異，先修正 validation/test 分割，再用控制實驗確認原因。**

跨資料域差異仍是最有證據支持的主因候選，但目前不能把 0.389 的 raw AP 差距定量歸因於單一因素。新發現包括：整個含腫瘤 volume 的 p99 會改變健康切片的亮度；FOMO 實際僅使用 NIMH 子集，且 FLAIR 含大量 5 mm 影像；BraTS validation 與 test 病人完全重疊；附件對 Gaussian 實驗的描述不正確。單純增加 epoch、調 Yen 門檻或移除背景都不是目前最有把握的優先方案。

![稽核證據摘要](C:/ML/andi_test/Test/andi_rewrite/artifacts/auprc_root_cause_20260905/evidence_summary.png)

**一、實驗結果已確認，Gaussian 對照需更正**

| 實際訓練模型 | Raw AP | 3D median filter AP | 訓練更新步數 |
|---|---:|---:|---:|
| FOMO/NIMH，empirical spectrum | 0.445085 | 0.470212 | 137,936 |
| BraTS healthy slices，empirical spectrum | 0.834116 | 0.847758 | 310,123 |
| FOMO/NIMH，Gaussian | 0.271051 | 0.488052 | 本表未另核對 |

數值來源為 [FOMO empirical 評估報告](C:/ML/andi_test/Test/andi_rewrite/outputs/runs/fomo45k_sri24_flair_t1_t2_empirical_spectrum233/evaluation/brats21_test251/inference_report.json)、[BraTS 評估報告](C:/ML/andi_test/Test/andi_rewrite/outputs/metrics/brats_flair_t1_t2_empirical_spectrum233_epoch0232_brats21_251/inference_report.json)、[FOMO Gaussian 評估報告](C:/ML/andi_test/Test/andi_rewrite/outputs/runs/fomo45k_sri24_flair_t1_t2_gaussian233/evaluation/brats21_test251/inference_report.json)。

0.271 來自 `fomo45k_sri24_flair_t1_t2_gaussian233/epoch_0232.pt`，其訓練與測試均使用 Gaussian。它不是 empirical checkpoint 僅在推論時更換噪聲。Gaussian 的 filtered AP 0.488 還略高於 empirical 的 0.470，顯示噪聲與空間後處理有交互影響；未做成對信賴區間前，不宣稱哪個較佳。這組結果不能證明「來源頻譜差異只是次要因素」。

此專案將 sklearn 的 average precision 稱作 AUPRC，並非梯形積分 PR-AUC。兩者在文獻比較時應明確區別。[scikit-learn 官方 AP 定義](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

**二、最值得優先驗證的機制：p99 正規化把含病灶 volume 的統計帶入健康切片**

[normalize_volume](C:/ML/andi_test/Test/andi_rewrite/data/imaging.py:21) 對每個 modality，用全 volume 的正值 voxel 求 p99，沒有 clipping。[load_subject_volume](C:/ML/andi_test/Test/andi_rewrite/data/imaging.py:62) 先完成此步驟，[healthy_slices_for_subject](C:/ML/andi_test/Test/andi_rewrite/data/healthy_slices.py:47) 才移除有 segmentation 的切片。FOMO LMDB 也採相同 p99 定義。

因此，「相同 normalization 程式」不保證「相同的健康組織數值尺度」。BraTS 訓練的健康切片仍使用同一病人腫瘤所在 volume 的分母；FOMO 健康受試者的分母沒有相同病灶影響。這不是直接把腫瘤標籤交給模型，而是樣本選擇與全 volume 強度統計的耦合。

本次對 12 位等距抽取的 **BraTS 訓練病人**，比較全前景 p99 與排除 segmentation>0 後的 p99：

| Modality | p99(全前景) / p99(seg==0)，中位數 | 範圍 |
|---|---:|---:|
| FLAIR | 1.221 | 1.000–1.588 |
| T1 | 0.999 | 0.985–1.013 |
| T2 | 0.998 | 0.986–1.148 |

FLAIR 的分母典型增加約 22%，對應正常組織的縮放值下降約 18%；個案影響程度不同。這支持「健康組織強度尺度不同」這個具體機制，但尚未證明它占 AP 差距多少。`seg==0` 只是未標註區域，不能當作全部經確認健康的組織。

為避免背景與切片位置混淆，另於 z=30、45、60、75、90、105、120，各取 12 張、每域共 84 張訓練切片：

| Modality | FOMO 正值前景均值 | BraTS 正值前景均值 |
|---|---:|---:|
| FLAIR | 0.651 | 0.482 |
| T1 | 0.551 | 0.665 |
| T2 | 0.374 | 0.360 |

匹配 z 後，FLAIR 與 T1 差異仍在，且方向不同。這也說明不能由含大量零背景的全圖均值，直接推定所有 modality 都偏亮或直接套同一個縮放比例。此抽樣尚未匹配組織、病人、年齡或 scanner。

可行修正是建立對病灶較不敏感、推論時不需真實 segmentation 的正規化對照：例如以 T1 估計正常白質參考，再對各 modality 定標，或比較 brain-mask 內 robust z-score。訓練、validation、test 必須走一致流程。WhiteStripe 論文提出利用正常白質作強度參考，也討論多模態延伸；它提供候選方法，並不保證本案會改善。[Shinohara 等，2014](https://adni.loni.usc.edu/adni-publications/Statistical%20normalization%20techniques%20for%20magnetic%20resonance%20imaging.pdf)

本次用真實 segmentation 排除病灶，僅為診斷上述機制。**不能把這個依賴 test ground truth 的操作當作正式推論前處理。**

**三、資料域差異還包含來源多樣性、原始解析度與切片選擇**

本機 source audit 顯示：FOMO 路徑雖含 45K，但這次實驗實際來源為 PT007_NIMH 的 240 位受試者／243 次 session；訓練為 216 人、219 sessions，validation 為 24 人。BraTS 訓練 metadata 則有 938 位病人。不能把本案解讀為「整個 FOMO45K 對 BraTS」的大型多中心比較。[本機來源稽核](C:/ML/data/FOMO45K_SRI24_BraTS21_ANDi/PT007_NIMH/source_validation.json)

[本機 acquisition metadata](C:/ML/data/FOMO45K_healthy_T1_T2_FLAIR_243.tsv) 顯示全部 243 sessions 標為 Control；FLAIR 中 151 次為 2D／5 mm、92 次為 3D／1 mm。3D T2 為 1 mm；T1 為 1 或 1.2 mm。這是來源群體的確認資訊，但本次未取得 BraTS 原始 acquisition 的同等統計，因此尚不能量化兩域 scanner／解析度差距。

兩域即使都輸出至 SRI24、1 mm 網格，也不會讓原始 5 mm 影像恢復成真正 1 mm 的量測解析度。FOMO 已有配準、去顱骨與模型輸入 p99 處理；不要直接歸因為「忘記 registration／normalization」。另須注意，本機 [FOMO 前處理](C:/ML/andi_test/Test/andi_rewrite/data/fomo45k/brats21.py:1) 明確將 N4 校正影像只用於估計配準，最終 transform 套回原始 intensities；不能因使用了 N4 工具就宣稱發布影像都已作強度 bias correction。是否需修改，應由偏場診斷與對照結果決定。BraTS 官方只確認其資料經共同模板配準、1 mm 重採樣與去顱骨，這些步驟本身不能消除所有 acquisition 差異。[BraTS 2021 官方資料說明](https://www.med.upenn.edu/cbica/brats2021/)

切片選擇也明顯不同：FOMO 訓練在 z=50–99 的比例為 35.57%，BraTS healthy-slice 為 13.69%。後者的健康條件會排除大量帶腫瘤的中央切片。這影響全圖均值、前景比例、loss 與噪聲統計，不能單從切片數或平均強度分離原因。先做相同 z／前景面積的抽樣控制，再判斷需不需要 z-balanced sampling；直接改成均勻 z 不保證較好。

文獻也支持把 scanner shift 列為重要假說：一篇包含 ANDi 的 2025 年原始研究預印本觀察到 scanner 與 lesion burden 的影響，並指出增加健康資料量的收益可能有限。這是外部支持，並不是本案的因果證明。[Frotscher 等，2025，預印本](https://arxiv.org/html/2512.01534v1)

**四、訓練更新量確實不同，但延長訓練並非已驗證的解法**

兩個 empirical checkpoint 及 training_metrics.csv 一致：FOMO 每 epoch 592 steps、總計 137,936；BraTS 每 epoch 1,331 steps、總計 310,123，比例 2.2483。兩者 EMA decay=0.995、step_start=2000；warmup 分別為 6,896／15,506 steps。架構、三個輸入 channel 與基本 diffusion 設定一致。

FOMO 最後 20 epoch validation 平均 0.006519，前一個 20 epoch 為 0.006466，已持平或略變差；最佳單次 validation 在完成第 188 epoch。BraTS 最後 20 epoch 也高於前 20 epoch。兩域的 validation 資料及噪聲不同，loss 絕對值不是可直接比較的收斂尺標。此外兩個訓練設定都是 `validation.use_ema=false`，最後 AP 卻評估 EMA；驗證對象也應對齊。

純計算量控制可讓 FOMO 從頭訓練約 524 epochs：524×592=310,208，與 BraTS 相差 85 steps、約 0.027%。若要求完全一致，需總步數停止在 310,123，即 523 個完整 epoch 加 507 steps。建議實作明確的 step budget 與 step-based scheduler；目前 trainer 沒有直接可用的 `max_steps` 設定，不應只把未知鍵加進 YAML。

**特別注意 resume 陷阱：** [checkpoint 載入](C:/ML/andi_test/Test/andi_rewrite/engine/checkpoint.py:70) 會還原 scheduler 狀態，連原有 `steps_total=137936` 一起恢復。[cosine 排程](C:/ML/andi_test/Test/andi_rewrite/engine/schedulers.py:28) 沒把 progress 限制在 1 以下；若直接改 epochs=524 後 resume，超過原排程終點的 LR 會再次回升。這不是目前 233 epochs 的故障，但會破壞「公平延長訓練」實驗。要明確選擇重建整段排程，或另開具有獨立 LR 設定的 continuation run，並逐步驗證載入後的 LR。公平的從頭訓練與追加訓練要分開標示。

**五、validation/test 完全重疊，需先修復實驗程序**

| 比對 | 重疊病人數 |
|---|---:|
| BraTS training／test | 0 |
| BraTS training／validation | 0 |
| BraTS validation／test | 251／251 |

依據是 [訓練切片 metadata](C:/ML/andi_test/Test/andi_rewrite/data/BraTS21/healthy_slices_train.csv)、[validation LMDB metadata](C:/ML/data/BraTS_2021_healthy_lmdb_val/healthy_slices.csv)、[test split](C:/ML/andi_test/Test/andi_rewrite/splits/BraTS21/scans_test.csv)。另從 train／val 各抽兩筆，以來源 NIfTI 重建 LMDB，四筆最大絕對誤差均為 0，支持 metadata 與內容相符；這不是全資料逐筆稽核。

目前 [validate](C:/ML/andi_test/Test/andi_rewrite/engine/trainer.py:282) 使用 eval、no_grad 與 RNG 保存／還原；看到的是 fixed warmup-cosine，未見用 validation loss 控制更新或 early stopping。因此不能直接說 0.834 是「拿 test 訓練」造成。**如果曾依 validation 選 checkpoint、噪聲或其他超參數，test 就已參與模型選擇。** 本次沒有足夠歷史紀錄判定是否發生。

建議從原 938 位訓練病人內，重新切出真正的 validation，例如約 750／188 人，保持現有 251 人只作最終評估；重新建立兩份 healthy-slice LMDB，所有 sessions 按病人分組。若原 251 人已被反覆調參使用，需額外未使用的外部 holdout 才能支持新的泛化主張。BraTS healthy slices 是以腫瘤 segmentation 篩選的「正常外觀切片」，應在研究設定中揭露，不能等同完全不使用病灶標註的外部健康受試者訓練。

**六、頻譜仍是混雜因素；計分錯誤與背景則可降低優先度**

兩個 sampler 實際都載入 `radial_power`，沒有 amplitude fallback，BraTS channel selection 為 [0,1,3]。但 FOMO 的 radial bins 是 64，BraTS 是 128，並非完全相同的頻譜估計設定。套用目前 sampler 後，三 modality 濾波器 cosine similarity 約 0.9968／0.9814／0.9944，形狀接近但不同；低頻功率也有差異。這些描述性數值不能取代因果消融。

分離資料與噪聲的有效實驗是 2×2：FOMO／BraTS 訓練資料，各搭 FOMO／BraTS 訓練來源頻譜，同一格內訓練與推論使用相同頻譜；統一 radial bins、更新步數、預處理、取樣與種子。若研究要求純外部健康資料，使用 BraTS 頻譜的格子應標記為使用目標域資料的控制組，而非純外部設定。只對既有 checkpoint 換推論頻譜，最多是噪聲錯配敏感度測試，不能代替這個實驗。

已完成的計分驗證：

- FOMO 是 `disk_streaming`、BraTS 是 `in_memory`，所以原本不能僅憑相同 YAML 指標設定就宣稱路徑等價。
- 依保存的 251 個 raw／mf score cache，以 seed=73 重建 5,000,000 個有放回抽樣的全域 voxel 索引；其中病灶 voxel 51,769，占 1.03538%。獨立呼叫 sklearn 得到 0.44508451815979344／0.47021173099001035，與原報告完全一致。
- 現有串流／記憶體 sampled 等價及 exact 等價測試均通過：2 passed。測試有來自 4×4×4 玩具影像的 Otsu RGB-shape 警告，未造成失敗。
- 兩個實驗皆採 dataset-level score normalization；BraTS export scope 設定不同但 predictions 關閉，不能用它解釋 AP 差距。

另在 test 清單等距抽 12 位病人，以 FLAIR>0 重採樣建立 brain-mask 近似、侵蝕兩個 model voxel 區分內部與邊緣，在既有 raw normalized threshold=0.055 下分析未做 dilation 的 mask：

| 非病灶區域 | 超過門檻的 voxel |
|---|---:|
| 腦內部 | 100,588 |
| 腦邊緣 | 746 |
| 腦外 | 98 |

99.17% 的假陽性在腦內部。這 12 例的 brain-only AP 與全圖 AP 差距也很小。因此單純 erode brain mask 或去背景，預期無法填補本案差距。這是小樣本、單一門檻的描述性結果，不能推廣成所有病人與所有操作點；更不能將此門檻視為未偏誤的部署門檻。

調 Yen 門檻會改變 Dice／precision／recall，不會改變以連續 score 排序計算的 AP。對全資料套同一個嚴格遞增縮放也不會改善 AP，除浮點 ties 等數值效應；輸入影像的強度處理則會改變模型輸出，屬於不同問題。

**七、具體解決方案與實驗順序**

先完成病人分割，保留測試集，統一評估 EMA、固定評估 noise seed 與 voxel sample，並保存 split／checkpoint／spectrum／程式版本 hash。原報告 `git_commit_hash=null` 且工作目錄已有修改，往後要保留可追溯的程式快照。

| 優先 | 實驗 | 固定條件與改變項 | 用來回答的問題 |
|---|---|---|---|
| P0 | 真正獨立的 validation；儲存各 modality score | 同一 checkpoint、同一輸入；另存聚合前 score 與 max 的來源 modality | 哪個 modality 在正常組織產生高分？EMA 與非 EMA 是否不同？ |
| P1 | 正規化對照 | 固定外部健康訓練資料、步數與 noise；比較現有 p99 與不需標籤的正常組織參考定標；每組 train/eval 一致 | 強度尺度是否是主要可修復原因？ |
| P1 | FOMO → BraTS healthy-slice 微調 | 只用新 BraTS train；先固定原 FOMO spectrum；同時計算來源 validation 與目標 validation | 少量目標域適應是否能降低腦內假陽性？ |
| P2 | 對齊更新量 | 比較約 137,936 與 310,123 steps；使用明確的 LR 排程，其他條件固定 | 計算量的邊際收益多大？ |
| P2 | 資料域×頻譜 2×2 | 相同 radial bins、步數、EMA、正規化；訓練和推論噪聲匹配 | 資料來源、頻譜及其交互作用各有多少影響？ |
| P3 | 解析度／偏場／對比 augmentation，z 或前景面積匹配 | 各項分開與 baseline 比較；按病人分組抽樣 | acquisition 與切片分布差異是否可以緩解？ |

若主要目標是提高 **BraTS 表現**，推薦先做目標域微調 pilot：從 FOMO checkpoint 開新 run，LR 可先以 1e-5–2e-5、10k–30k 更新作候選範圍；在新的 validation 選擇，保留 source validation 觀察遺忘。以相同目標 train、noise 與適應步數的從頭訓練模型作對照，再與完整訓練的 in-domain baseline 比較。上述數字是試驗起點，尚未驗證最佳值，也不保證達到 0.83。

若研究限制是 **只能用外部健康資料**，則不能把 BraTS 微調或 BraTS-derived spectrum 算進主要方法。優先做一致的強度標準化，增加符合三模態條件的其他健康來源，按原始 FLAIR 2D/3D、slice thickness 分層評估，並測試實際採樣解析度的 augmentation。FOMO45K 本身含健康與臨床來源，擴資料時必須查 cohort／group，排除病灶來源及與 BraTS 重複的 subject；資料集名稱不保證所有影像健康。[FOMO45K 官方 dataset card](https://huggingface.co/datasets/FOMO-MRI/FOMO45K)

若 P0 發現 max-pooling 幾乎總被單一 modality 的正常高分支配，可在獨立 healthy validation 估計各 modality 的 score 分布，測試分位數或 robust 尺度校正後再 pooling；比較時同時檢查病灶召回，避免校正掉真正訊號。這需要額外匯出 `aggregate_time` 後、`pool_modalities` 前的分數；目前快取已經 max pooled，無法由現有 raw cache 還原各模態貢獻。也不要把三 channel UNet 直接裁為一或二 channel 來當作有效對照。

各組同時回報 raw／mf pooled AP、每病人 AP 分布、正常腦內高分率、以 validation 選定門檻的 Dice／sensitivity，以及按 lesion size、modality acquisition 分層的結果。以**病人**為單位做成對 bootstrap 信賴區間，重要候選至少重複數個訓練 seed；不要把數百萬個相關 voxel 當作獨立受試者。若某項修正只降低 MSE、卻沒有改善獨立 validation 的 AP 與假陽性，便不支持它是有效解方。既有 test 的 `bestdice`／`bestthr` 屬於 test 上的 oracle sweep，不應當成可直接部署的門檻。

**八、研究產物與重現方式**

本次新增 [完整稽核數據](C:/ML/andi_test/Test/andi_rewrite/artifacts/auprc_root_cause_20260905/audit.json)、[正規化診斷](C:/ML/andi_test/Test/andi_rewrite/artifacts/auprc_root_cause_20260905/normalization_audit.json)、[來源 metadata 摘要](C:/ML/andi_test/Test/andi_rewrite/artifacts/auprc_root_cause_20260905/source_metadata.json)、[LMDB 來源抽查](C:/ML/andi_test/Test/andi_rewrite/artifacts/auprc_root_cause_20260905/lmdb_source_spotcheck.json) 及 PNG／SVG 圖表。原有模型、資料、config 與分析文件未修改。

在專案根目錄，以既有 ANDi Python 執行：

```powershell
$auditPython = 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe'
& $auditPython artifacts/auprc_root_cause_20260905/audit.py
& $auditPython artifacts/auprc_root_cause_20260905/normalization_audit.py
& $auditPython artifacts/auprc_root_cause_20260905/plot_summary.py
& $auditPython -m pytest tests/test_streaming_evaluator.py -k 'sampled_disk_streaming_matches_in_memory_and_resumes or exact_disk_streaming_matches_in_memory' -q -p no:cacheprovider
```

本次環境為 PyTorch 2.1.0+cu118、NumPy 1.24.1、sklearn 1.3.1。主要分析僅讀 LMDB／NIfTI／cache，輸出至本研究目錄；執行稽核會重寫此目錄的同名分析結果。沒有重跑 BraTS 全模型推論、沒有新的改善後 AP，也沒有完成 patient bootstrap。對「哪個修正可補回多少差距」的判定，仍以完成上述獨立 validation 消融為準。
