# 完全健康 T1／T2／FLAIR 訓練：實驗盤點與 AUPRC >0.8 路線

調查日期：2026-09-08。程式版本 HEAD：db4cb1f；本次查閱本機程式、設定快照、training/inference reports、資料建置與診斷紀錄，並重新核對 split 交集。未啟動訓練、未重跑 GPU 推論。下列 AP 是既有報告數值，非本次新實驗成果。9/5 稽核的 NIfTI／LMDB 抽查沿用其證據，未在本次重新逐筆執行。

**主要建議：以 FOMO robust IQR 20-epoch 模型為優先候選，先建立同一 validation、相同步數的 p99 對照，再做有上限的訓練延長與模態分數校正。**目前符合外部健康資料訓練路線的 robust pilot 在 validation 50 例為 raw AP 0.6863、MF AP 0.7475；尚無證據證明純健康三模態已在獨立最終測試達到 0.8。若要求 raw AP >0.8，差距是 0.1137；若接受固定 MF pipeline，差距是 0.0525。兩個目標必須分開報告。

## 1. 本案「完全健康」的操作定義

依使用者後續明確定義：**只要模型訓練影像全部來自 FOMO，就算本案的完全健康訓練。**這是本專案的操作定義。

允許使用 BraTS training/reference 資料建立 histogram reference、noise spectrum、z sampling statistics，以及使用獨立 validation 選參數；這些不會使 FOMO 訓練實驗失去資格。保留來源紀錄及 validation/test 分割，以便公平解讀結果。模型不加入 BraTS slices 或 BraTS 微調，輸入仍為 FLAIR/T1/T2。

因此 FOMO p99、robust IQR、histmatch、histmatch+z-bucket 都符合條件。LEMON 與 BraTS 訓練模型保留為歷史比較，不列入本次 FOMO-only 主線；不再要求所有前處理統計也必須源自健康受試者。

## 2. 目前專案如何運作

- 資料經去顱骨、配準、強度處理後，沿 axial z 軸切為 `[3,128,128]`，實際 channel order 為 `[FLAIR,T1,T2]`。LMDB 保存 float32 arrays。
- `ANDiUNet` 為 2D noise predictor，三模態模型約 972 萬參數。DDPM 1000 steps、線性 beta 0.0001→0.02。訓練隨機抽 t，用 MSE 預測 noise。
- 近期設定 batch=52、AdamW、warmup cosine、LR 2e-5→1e-4→2e-5，EMA decay=0.995、step_start=2000。原 FOMO/BraTS run 的 validation 用非 EMA，robust/histmatch 新 run 改用 EMA。
- 推論不是完整生成健康影像：每個 t=75…199 都從原始輸入獨立加噪，比較真實 noise 與模型預測 noise 所導出的 posterior mean，取平方差。先對 125 個時間點作 geometric mean，再對三模態取 max。
- 得到 `[128,128,Z]` 分數後，報 raw 與 3D median filter kernel=5 分數。dataset min-max 後用連續分數計 AP；Yen 和 binary dilation 屬於 mask/Dice 路徑。
- `metrics/classification.py` 使用 sklearn average_precision_score；專案 AUPRC 實際是非插值 average precision，不是梯形 PR-AUC。新 run 以 seed=73 有放回抽 5,000,000 個全域 voxel。不是每病人 AP 的平均，也不是原生 240×240 網格上的 AP。

程式依據：[detector](C:/ML/andi_test/Test/andi_rewrite/anomaly/detector.py)、[聚合](C:/ML/andi_test/Test/andi_rewrite/anomaly/aggregation.py)、[trainer](C:/ML/andi_test/Test/andi_rewrite/engine/trainer.py)、[評估](C:/ML/andi_test/Test/andi_rewrite/engine/evaluation/inference.py)、[AP](C:/ML/andi_test/Test/andi_rewrite/metrics/classification.py)。ANDi 的方法背景見[原論文](https://arxiv.org/abs/2312.01904)。

## 3. 前處理：已做與容易誤解的地方

### FOMO SRI24 主線

實際來源為 PT007_NIMH，不是所有 FOMO45K：240 人、243 sessions；216 training subjects／219 sessions、24 validation subjects，30,784／3,369 slices。9/5 source audit 記錄 FLAIR 151 sessions 為 2D/5 mm、92 為 3D/1 mm。重採樣為 1 mm 不會恢復原始缺少的 through-plane 資訊。

SynthStrip 估計 native mask；N4 影像用於配準估計。T1 rigid+affine 至 SRI24 240×240×155、1 mm grid；其他模態在 native grid 不一致時補 modality→T1 rigid，套用組合轉換及 shared T1 mask。**發布影像 transform 套回原始 intensities，不能把它描述成最終三模態都做過 N4 bias correction。**同 grid 分支不額外做 rigid，仍需視覺檢查 motion／跨模態對位，不能只靠 affine 相同認定解剖一致。

主 LMDB 使用每模態整個 3D 正前景 p99，無 clipping，再做 128×128 resize，進模型用 `2*x-1`。舊 `docs/fomo45k_brats21.md` 所述 z-score 是另一個 slice adapter；不能套用到這批 p99 LMDB。

依據：[實際配準](C:/ML/andi_test/Test/andi_rewrite/data/fomo45k/brats21.py)、[主 LMDB builder](C:/ML/andi_test/Test/andi_rewrite/data/fomo45k/brats21_lmdb.py)。

### BraTS healthy-slice 參考組

先對完整含腫瘤 volume 作 p99，再依 segmentation 去除有病灶的 slices。這是病人內正常外觀切片，不是完全健康受試者。9/5 對 12 位訓練病人抽查，FLAIR 全前景 p99／排除病灶 p99 中位數=1.221；因此健康切片也帶入病灶影響過的強度分母。該結果支持機制假說，不能定量歸因整個 AP 差距。

### Robust IQR：已實作且已跑完

在完整 3D 每模態正前景求 median 和 IQR，輸出 `(x-median)/(Q75-Q25)`，背景=-1、不裁高亮尾端；resize 在其後，train/eval `normalize_input=false`，避免再次 `2*x-1`。不使用病灶標註估計正規化。保留原始 FOMO split、slice selection、noise spectrum。

既有診斷抽樣中，robust 後 FOMO／BraTS 非病灶 FLAIR p99 約 1.418／1.430，中心尺度接近，仍不代表組織條件分布或紋理完全匹配。BraTS 遮罩在此只用於診斷，不是 robust 推論輸入。[程式](C:/ML/andi_test/Test/andi_rewrite/data/robust_normalization.py)、[診斷](C:/ML/andi_test/Test/andi_rewrite/outputs/diagnostics/robust_b/distribution/fomo_brats_robust_comparison.json)。

### Histogram matching／z-bucket：已跑過，非未執行建議

Histmatch 以 888 個 BraTS training subjects 的非病灶前景建立 4097 quantiles reference，排除 50 validation 與 test；每 session／模態作單調映射。影像來自 FOMO，mapping 使用 BraTS training 病人資料與 segmentation；依使用者定義，此實驗符合完全健康訓練條件。CDF 接近也不保證 GM／WM／CSF 對應正確。

z-bucket 對照按 BraTS healthy-slice z 分布分成 29 桶，重複抽成 69,000 entries；只有 28,309 distinct slices，每張最多出現 7 次。這增加呈現次數，不增加健康人數；並引入腫瘤切片排除造成的 z 偏好。文件中的「未啟動訓練」已落後於 9/7–9/8 實際 reports。

### LEMON 舊線

已有 T1/T2/high-resolution FLAIR、HD-BET、模態 rigid、T1→MNI affine、固定 physical ROI 與 128 輸出的另一套流程。115 sessions 中排除原 manual-review 18 sessions，保留 97（train93/val4），13,305／580 slices。不可直接把其 LMDB 與 SRI24 混在一起；增加此來源前要统一 grid、ROI、normalization 並重新保留健康 validation。

## 4. 已完成實驗的核心結果

下表 raw／MF 均為 AP；不同評估集合不作直接排名。epoch 為完成數，檔名為零起算，例如完成20對應0019。

| 模型／設定 | Epochs | 評估集合 | Raw AP | MF AP | 主方法資格 |
|---|---:|---|---:|---:|---|
| LEMON，自身 spectrum |233|BraTS test251|0.13335|0.13341|歷史比較，非 FOMO 訓練|
| LEMON，BraTS spectrum |233|BraTS test251|0.14596|0.14721|歷史比較，非 FOMO 訓練|
| LEMON/BraTS-noise，MNI affine 對照 |233|MNI QC-pass first50|0.29094|0.31145|歷史比較，非 FOMO；集合亦不同|
| FOMO NIMH 舊線，自身 spectrum |233|BraTS test50|0.48017|0.50899|健康來源候選|
| FOMO SRI24 p99，empirical |233|BraTS test251|0.44508|0.47021|健康來源候選|
| FOMO SRI24 p99，Gaussian train+eval |233|BraTS test251|0.27105|0.48805|健康來源候選|
| **FOMO robust IQR** |**20**|**BraTS validation50**|**0.68632**|**0.74749**|**本次優先候選**|
| FOMO histmatch，40-epoch run 中期 |20|BraTS test50|0.66706|0.68750|符合：FOMO 訓練＋BraTS reference|
| FOMO histmatch，40-epoch run 終點 |40|BraTS test50|0.67888|0.69853|符合：FOMO 訓練＋BraTS reference|
| FOMO histmatch + z-bucket69k |20|BraTS test50|0.67490|0.69545|符合：FOMO 訓練＋reference/z 統計|
| BraTS 三模態 healthy slices |233|BraTS test251|0.83412|0.84776|不符合：訓練影像不是 FOMO|

表中來源是各 `outputs/runs/*/evaluation/*/inference_metrics_summary.csv`、`outputs/comparisons/*/inference_metrics_summary.csv` 及 `outputs/metrics/brats_flair_t1_t2_empirical_spectrum233_epoch0232_brats21_251/inference_metrics_summary.csv`；完整索引見同目錄 audit.json。

可得出的結論：histmatch 20→40 MF 增加0.01103；z-bucket20 比未重抽40低0.00308，目前沒有證據支持 z 模仿是突破點。兩者更新量約26,540／23,680，仍不是完全等步數消融。robust20為11,840 updates，且最後數個 epochs 的來源 EMA validation MSE 持續下降，支持進行有限延長，但不保證 AP 上升。233-epoch p99 已接近 validation loss 平臺，不能把這一觀察套用到 robust20。

## 5. 評估與歸因必須先修正

本次重新讀 CSV：validation50=50、test50=50、test251=251、舊 train=938；validation50 與 test251 交集0、與舊 train 交集50；test50 完全包含在 test251。因此：

- robust validation50 可比較 FOMO-only methods；**舊 BraTS baseline 在這50人訓練過，不能在此作公平 in-domain baseline**，須重訓排除這50人。
- histmatch 在 test50 比較多 checkpoint，該子集已用於探索，不能再把整個251宣稱全新未使用 holdout；9/5以前251也已有反覆結果。新泛化結論最好用預留且未看過的外部 cohort，否則明確稱歷史 benchmark。
- 9/5 audit 發現舊 BraTS training validation 的251人就是 test251。這不等於有 test 梯度更新，但若由它選模型，會污染 model selection。
- 同一個嚴格單調的全域 score scaling 或 Yen 門檻不會解決 AP 排序問題。median filtering 與每模態／位置校正才會改變排序；需保存原分數以便分析。

## 6. 建議執行順序與決策門檻

### P0：先補最小公平對照

固定 validation50、同樣 voxel sample、seed、EMA、t75…199、max pool、3D MF5、健康 split 與 noise。既有 p99-233 在此重評可作參照，但正式 normalization attribution 要新建 p99-20（11,840 steps、20-epoch LR schedule）對照 robust-20。Histmatch-20 是40-epoch schedule中期，也不能替代它。

同時匯出 `aggregate_time` 之後、max 之前的三模態 maps，記錄正常腦內最高分來自何模態、per-subject AP、病灶大小與正常高分率。目前 evaluator 的 cache 已 max pooled，無法從它回推出原來三個分數。只分析這一步不需新模型訓練，但需額外推論。

可先以 CPU 對既有 robust score cache 做 per-patient AP 與病人層級 bootstrap；這能估0.7475穩定性，不能取代跨訓練seed重現或創造未用過的test。

### 納入 BraTS 統計的合格對照（依更新定義）

先將現有 histmatch-40、histmatch+z-bucket20 與 robust20 評估在同一 validation50，作候選篩選；不同步數與schedule仍須揭露。再對入選的前處理建立等步數對照，避免把不同test/validation的AP直接排名。

新增 FOMO 訓練 × spectrum 對照：固定 FOMO split、前處理、模型、radial bins、步數及seed，分別用 FOMO spectrum 與 BraTS training-derived spectrum，且每組 train/eval noise匹配。BraTS spectrum來源須核對並排除本次validation50/test，不直接沿用可能包含validation50的舊NPZ。兩組都符合使用者條件。此對照用來判斷目標域噪聲統計是否有幫助；只替既有checkpoint換推論noise不能取代它。

Histmatch與z-bucket保留為合格候選。現有結果尚未顯示z-bucket有明顯優勢，因此不因放寬定義就優先擴大重抽樣預算；優先比較robust與histmatch，再測最有希望分支的spectrum。

### P1：robust IQR 延長與加噪尺度分開測

從頭建立 robust-40/80 的預先固定學習率曲線，候選總預算23,680／47,360 updates。先跑40，若同一 validation AP 與病人層級分布有一致提升再投入80；報告配對CI，不能只看單次MSE。前處理、spectrum、pooling 都先固定。這些是不同 training-budget policies，不能將長排程中途checkpoint誤稱為完整短排程。

若為節省成本沿用robust20權重，開獨立 continuation run、重建 optimizer/scheduler，明確區分從頭訓練，記錄 EMA初始化策略。**不要只改epochs後resume**：現有checkpoint還原舊scheduler總步數，cosine進度超過1未截斷，LR可能回升。

新假說：robust 改了輸入尺度，所以同一 t 並非同一有效 SNR。`x_t=sqrt(alpha_bar)*x0+sqrt(1-alpha_bar)*noise`；需用健康前景尺度量測每模態相對噪声強度。先固定既有 t 範圍保留baseline，再以少量事先列定範圍（如50–125、75–200、125–250）作validation-only inference消融。若增加noise強度或重估spectrum須有匹配train/eval的新訓練，不能混淆為同一消融。上述範圍是候選，不是已證最佳參數。

### P2：用健康資料校正異常分數

以held-out健康影像估計每模態log-score中心／尺度，先做全模態robust calibration，再比較calibrated max與mean。若顯示解剖部位差異，再加粗z/組織分區並做收縮估計，避免24人validation建立過細atlas而過擬合。觀察病灶召回是否被壓掉；外部健康校正也可能無法消除BraTS-specific差異。

若max總被單一模態正常組織支配，這比盲目換大模型更直接。保留全部三模態輸入；僅pooling的FLAIR-only score可作診斷，不等於單模態重新訓練。正分位數映射可能壓平極端值，實作要考慮尾端外推與ties。

### P3：前處理品質與健康來源多樣性

WhiteStripe／T1估计正常白質參考定標作robust的候選對照，所有train/eval走同算法，不用test真實seg。這有[原始正規化研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4215426/)支持，但對本資料的改善仍須實測。

先按原生FLAIR 2D/3D、厚度分層查看healthy validation的分數和QC；比較3D-only及全部資料時匹配steps、subject sampling，避免把資料量減少誤認成解析度效果。加入保守的每volume對比、偏場、blur/downsample augmentation，幾何轉換跨三模態共享。強度變換應作用於適當原始正前景，不能直接對帶負值的robust輸出做非整數gamma。

N4最終強度校正只作有偏場證據時的獨立分支。避免同時更換配準、N4、norm、noise。本次維持FOMO-only，不加入LEMON；擴充FOMO來源時按來源／病人抽樣，不能只增加slice重複數。FOMO45K官方包含多種來源，新增資料需驗證Control、三模態完整性、個體重複與QC；不能把名稱45K等同45K個合格健康三模態個體。[官方資料卡](https://huggingface.co/datasets/FOMO-MRI/FOMO45K)。

### P4：以上失敗後才換模型路線

可作2.5D相鄰切片條件、healthy-only masked cross-modal prediction／patch diffusion對照。先證明主要剩餘錯誤是上下文或正常組織建模不足，再增加架構複雜度。不要因內部BraTS baseline已0.848就認為跨域0.8必然可達。大型健康資料benchmark也觀察scanner偏差及增加資料量收益有限，且並非本案相同三模態評估，不能拿其數字直接比。[原始研究預印本](https://arxiv.org/abs/2512.01534)。

## 7. 成功標準與投入控制

預先固定主要終點為同一grid、mask、sample policy的MF AP；raw AP一起報，若用戶要求raw>0.8則以raw為主要終點。Validation階段以0.78左右作是否投入大訓練的工程參考，不是統計保證；不因改評估集合或增大filter而宣告成功。重要候選至少3個training seeds、病人層級bootstrap、macro AP/病灶大小分層、固定validation threshold的Dice與healthy假陽性負担。

最終候選在真正未使用的holdout達AP>0.8才能宣告達標；信賴區間是否跨0.8需一同說明。現有50例GPU推論時間robust約19.6分鐘、histmatch約64分鐘，環境負載不同，不能保證新run耗時。先小規模檢查與固定預算，再做昂貴full test。

本次階段交付是調查與提案。長期AUPRC目標尚未達成；尚未跑新p9920公平對照、robust延長、模態校正或新holdout。
