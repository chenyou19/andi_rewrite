# Domain Classifier / C2ST protocol

這份文件描述目前 repository 內的可重現、可稽核 workflow。實驗資料與
classifier output 仍在分開的 artifacts 中；`domain_classifier.report` 只讀
這些 artifacts，不會替缺少的結果補值，也不會改寫 dataset、LMDB、checkpoint
或原始 audit。

## 目前實際的 model-input code path

目前 robust-IQR implementation 是
[`data/robust_normalization.py`](../data/robust_normalization.py) 的
`robust_normalize_volume`：輸入必須是完整 `[C,H,W,Z]` volume。每個 channel
只用 finite、`value > 0` 的 foreground 計算 median 與 `Q75-Q25`，輸出為
`(x-median)/IQR`，背景固定為 `-1`，不 clip；空 channel 保持背景，非空且
IQR 退化或含 NaN/Inf 會 fail。這個 function 明確要求統計發生在 slicing
之前，且沒有再做 `2*x-1`。

目前 domain-classifier adapter 是
[`data/domain_classifier/readers.py`](../data/domain_classifier/readers.py)。
`load_slice(..., stage="final")` 讀既有 final/LMDB model space，或以完整三
modality volume 呼叫同一個 `robust_normalize_volume` 後才取 axial slice；
`stage="registered"` 只讀配準後、IQR 前影像，不做額外 intensity scaling。
Resize 的輸出 boundary 是 `[3,128,128]`，channel order 由
[`data/domain_classifier/records.py`](../data/domain_classifier/records.py) 的
`MODALITIES = ("flair", "t1", "t2")` 固定為 `[FLAIR,T1,T2]`。讀取器不把
filename、participant、dataset、scanner、segmentation、LMDB key 或 split
送進 model。

FOMO/SRI24 的現有 preparation 與 dependency/QC gate 在
[`scripts/prepare_fomo45k_brats21.py`](../scripts/prepare_fomo45k_brats21.py)、
[`scripts/prepare_robust_b.py`](../scripts/prepare_robust_b.py)；MPI/OASIS3
healthy SRI24 pipeline 在
[`scripts/prepare_healthy_sri24.py`](../scripts/prepare_healthy_sri24.py)
及其 `data` adapters。現有文件中較舊的 p99/z-score 路徑不能取代上述
function-level contract，report 應以 manifest 的 `stage`、source fingerprint
與 `precomputed_final` provenance 為準。BraTS tumor-free eligibility 使用
manifest 的 native 與 model-grid mask；兩者任一 voxel 非零都必須排除。

## Data split、matching 與 model

[`data/domain_classifier/matching.py`](../data/domain_classifier/matching.py)
以 participant（不是 session 或 slice）固定 seed `73` 建立約
70/15/15 train/val/test；所有 visits/sessions 留在同一 split，並由
 `assert_participant_split_disjoint` 檢查交集為零。`build_pairs` 先選每位
 participant 的 primary case，再用 normalized `z = z/(Z-1)` 的 20 bins；每
 pair 每 bin 上限為 2，兩邊採相同 histogram。participant 配對是每個 split
 各自用獨立 seeded permutation（healthy 與 BraTS 使用不同 seed offset）後
 zip；不是 greedy matching，也不依 image intensity 配對。z-bin 只用來選取
 每個已配對 participant 的 slices，並保留 exclusions 供 audit。

正式 domain-classifier 的 main split 是 participant-level `70/15/15`、seed
`73`。negative control 有獨立預先登錄的 participant split `40/10/50`，不能
把 negative-control 的比例或結果混回 main split；兩者都必須在 manifest 中保存
自己的 split seed、subject lists 與 overlap audit。

Formal task 是 class 0 = Healthy、class 1 = BraTS21。四個 healthy cohort
（FOMO、MPI、OASIS3、Mixed）各自與 BraTS 比較，四種 channel set
（FLAIR、T1、T2、FLAIR+T1+T2）都必須在正式 rollout 中分開標記。模型
capacity 為 train-only standardized statistical Logistic、GroupNorm
Small CNN、from-scratch ResNet18（BatchNorm 轉 GroupNorm）；input 僅是
影像 tensor。

## Metrics、uncertainty 與 controls

每個 held-out prediction 必須保存 participant id、label、probability、split
與 provenance。slice ROC-AUC 是 secondary metric；主要 metric 先依
[`domain_classifier/metrics.py`](../domain_classifier/metrics.py) 將同一
participant 的 slice probabilities 取 mean，再計 subject ROC-AUC、PR-AUC、
accuracy、balanced accuracy、sensitivity、specificity 與 confusion counts。
Bootstrap 單位是 subject；若是 matched analysis 才可 resample complete pair，
不能把 slices 當 iid observations。

固定 classifier 的 pair swap 是條件於 within-pair exchangeability 的
association test，不能單獨當成 population-level C2ST。要補充 domain
separability evidence，需另保存 full-retrain label-permutation/null control；
該 null 仍是 conditional matched-sample procedure，不能被解讀成 unconditional
population test。
Permutation vector、bootstrap vector 只在 runner 實際保存時畫 distribution；
只有 CI 或 p-value 而沒有 vector 時，報告會列出 `MISSING`，不畫假分布。

四項 control gate 需分別保存：

- Tiny overfit：training accuracy `>= 0.99` 且 BCE `<= 0.02`。
- Registered positive：subject AUC `>= 0.80` 且 CI lower `> 0.50`。
- Negative control 與 label shuffle：CI 整段位於 `[0.35, 0.65]`，且沒有
  two-sided significant deviation。沒有顯著 p 值不等於 CI 接近 0.5。

低 separability 的正式 gate 是所有 capacity/control evidence 都 PASS，且
direction-invariant `max(AUC, 1-AUC)` 的 conservative bootstrap CI upper
`< 0.60`。任一 control FAIL 或缺失都只可回報 `INCONCLUSIVE`，不能寫成
「AUC 接近 0.5 所以兩個 dataset 完全相同」。AUC 很高只能說影像有強
dataset/acquisition/domain fingerprint；它不能直接指定 scanner hardware、
protocol、population、resolution 或 texture 哪一項是主因。

BraTS participants 是 tumor patients；native/model mask zero 只表示選定的
slice 通過 tumor-free criterion，不代表整個 participant 健康。其他 slice 的
lesion 仍可能影響 full-volume robust IQR normalization。20-bin z matching
控制的是粗略 slice-location 分布，不是 exact anatomical alignment。AUC=1
也不能單獨歸因為 scanner/acquisition、preprocessing quality 或 ANDi performance
gap 的因果來源，這些需要另外的 controlled analysis。

## Report artifacts

[`domain_classifier/report.py`](../domain_classifier/report.py) 的
`build_report(source, output_dir)` 會讀 run directory 或 explicit JSON/CSV
records，並寫入：

目前 runner 的 `train_one_seed` result 可直接傳入：identity 來自
`seed`、`split_seed` 與 `config.model`/`config.modalities`/`config.stage`；正式
test evidence 來自 `test_predictions`（每 slice 的
`label`、`probability`、`participant_id`、`pair_id`、`case_id`），而
`test_subject_predictions`、`test`、`test_statistics.subject_bootstrap` 會被
保留作 schema/provenance 參考。報告重新由 held-out per-slice predictions
聚合 subject mean，避免把 validation、train-final 或 control output 當成
正式 test row。

- `summary.csv`：先寫設計指定的 24 個欄位，再附 controls/gate、bootstrap、
  permutation、confusion、`result_kind`、`control_mode`、train-final tiny
  accuracy/BCE 與 provenance 欄位。`bootstrap_source` 會標出是 runner 保存的
  recorded subject bootstrap，或 report 在缺少保存區間時才做的 recomputed
  subject bootstrap。`formal_final` rows 是唯一可進入九個 scientific
  questions 的 rows；control、registered、negative、shuffle/permutation 與
  其他 stage 仍會列出但不會污染正式結論。
- `control_mode` 會把已完成的 same-cohort negative 與 explicit label shuffle
  分開標記；summary 的 `combined control gate` 是四項 control 的合併結果，
  不是每一項 individual gate。shuffle FAIL 只會停止 legacy v1/v2 protocol 的
  formal rollout，不會改寫 v3 Stage-A gate；它也不單獨證明 information
  leakage 或指定成因。
- `report_manifest.json`：report config、summary rows、實際讀到的 artifact paths、
  AP points、correlation、plot status，以及 optional `model_grid_v3`
  build-only context。這個檔名刻意與 Stage 1 的
  authoritative `experiment_manifest.json` 分開。
- `report_audit.json`：輸入 files 的 size/mtime/SHA-256、row/control count、
  formal/excluded row count 與缺失說明。Stage 1 的 `audit.json` 與
  `experiment_manifest.json` 只會被 reference/fingerprint，永不覆寫。
- `report.md`：逐一回答九個問題，分開呈現已讀到結果與尚未執行的部分；若
  source 有 Stage1 audit/parity，會列出實際 participant/pair counts、FOMO
  slice parity/max error，以及 MPI/OASIS3 的 ACL/incomplete 原因。
- 若已讀到 `shuffle_diagnostic_audit.json/.md` 與
  `shuffle_saved_model_orientation.json`，report 只加入 compact diagnostic
  status、seed273 orientation/contingency 摘要與三個檔案的 SHA-256；這些檔案
  不會改寫原 gate。其 interpretation 仍是 finite-sample train-label/domain
  imbalance 的一致性診斷，不是唯一成因證明；原先記錄的 one-shot
  balanced-pair shuffle/held-out-label sanity planning 欄位保留為 historical
  provenance，實際 protocol-v2 gate 會獨立列出，不能由 planning 欄位推導完成。
  powered-shuffle v1 的 test slice counts 會隨 seed 改變，因此不會被描述成
  fixed split/slice-order repeated-seed evidence；protocol v2 才要求固定 split
  seed 73 並先驗證三次 pair/slice-order hash 相同。
- 若 source 有 `fomo45k_supplementary_balanced_v2/`，report 會讀
  `protocol.json`、`preflight.json`、`gate_true_test.json`、
  `gate_random_test_labels.json` 與 `seed_*/result.json`，列出每個 seed 的
  subject AUC、原始 bootstrap CI 與 Holm p。這個 one-shot balanced diagnostic
  的 true-label gate 與 random-held-out-label sanity gate 是分開呈現；seed 73
  fixed split 的 rows 標成 `diagnostic_formalfit`、`formal_final=false`，永遠
  不會自動完成 formal matrix 或 low-separability claim。
- 若 source 有 `fomo45k_calibration_smallcnn/`，observed fit 會在 report 的
  `Observed calibration (diagnostic only)` 表獨立呈現；其 recorded subject
  bootstrap CI 從 `observed/result.json` 原樣讀取。`retrained_null/status.json`
  的 completed/requested/status 會原樣列出；observed row 不會進
  `summary.csv` 的 formal rows，也不會進 ANDi/AP correlation。
- 可用 `generate_calibration_diagnostic_plots(observed_result, output_dir)` 產生
  專用 `calibration_figures/`：observed ROC、PR、confusion、training curves
  與 matched-pair subject bootstrap。每張圖標成 diagnostic，且
  `plot_manifest.json` 保存 input SHA-256、圖檔狀態與
  `formal_matrix_rows=0`。full-retrained null histogram 只有在
  `retrained_null/status.json` terminal `complete` 且
  `completed == requested` 時才會寫入；未完成時只寫 `MISSING`，不讀 partial
  p-value。完成後 histogram 的統計量是 conditional matched-pair sample 的
  `T = abs(AUC - 0.5)`，199 次的 attainable minimum p 是 `0.005`。
- 若 source 是 `outputs/diagnostics/domain_classifier/model_grid_v3/` 或其
  `build_summary.json`，report 會建立獨立的 `model_grid_v3` context。這是
  **BUILD_ONLY** provenance：表格列出四個 comparison 的 build status、record
  counts、domain/split counts、audit pair counts，以及 audit 提供的 selected /
  eligible healthy participant source composition；它不會建立 summary.csv row、
  formal-final row、prediction 或 classifier result。Mixed 的 source counts 是
  participant-balanced selected subset，不能解讀成所有 eligible source participants
  或原始 ANDi slice-frequency mixture。
- v3 的 `canonical_parity=PASS` 只表示 audit 保存的 canonical shape/dtype/
  normalization/loader contract 檢查通過。report 明確標成
  `DESIGN_AUDIT_PENDING`，除非 audit 有明確 `full_candidate_coverage`
  candidate inventory/pool verification 或 scope status。缺少 coverage 欄位時是 `AUDIT_PENDING`；只有
  實際 old candidate source 或 `SUPERSEDED_PRETRAIN_SELECTION_BUG` 才會標成
  blocked。`selection_equivalence_proof=false` 只表示歷史 controls/results
  不能 reuse，不代表 candidate coverage 失敗。report 仍標示 exhaustive tensor
  parity 尚未建立；metadata 或 contract PASS 不會被寫成 scientific readiness。
  v3 `training_started=false` 也不代表 112 fits 已開始。
- 目前明確指定的 final revision 是
  `outputs/diagnostics/domain_classifier/model_grid_v3_fullcandidate_20260917_final/`。
  build-only counts 是 FOMO `5896/240`、MPI `2832/115`、OASIS3 `6026/246`、
  Mixed `15110/601`（records/pairs）；full CSV candidate inventory 是
  `938` subjects、`38967` candidate rows，`full_candidate_coverage=PASS`，
  且 `old_selected_pool_used_as_source=false`。這些 counts 是 metadata/build
  evidence，不是 112 個 classifier fits，也不是 exhaustive tensor parity。
  報告同時保留 Stage-A tiny/positive/negative 的實際 gate 狀態、formal fit count
  `0`、registered FOMO materialization `PASS` 與 independent full Mixed review
  的 compact tensor/map evidence。修正版 `final_external_audit_v2.json` 是
  selected audit；旁邊的 `final_external_audit.json` 以 `SUPERSEDED` historical
  provenance 保留，其舊 `candidate_rows_do_not_match_support_counts` 不得覆蓋
  v2 PASS。舊 FOMO calibration 仍是獨立 diagnostic，不能混成 v3 formal
  result；build sidecars、audits 與 source hashes 會寫進
  `report_manifest.json`/`report_audit.json`，不會覆寫 authoritative source audit。
- `scripts/replay_model_grid_v3.py` 是獨立的 producer-recovery audit。它只在
  明確指定的新 replay root 讀 full CSV/map、canonical healthy LMDB 與
  `MRIDataVolume`，重建 `38967` candidate rows、四組 deterministic pairing，
  再用完整 candidate-row projection 逐 `(participant,z)` 比對既有 rows，並
  比對每個 subject 的 completion ledger（resume nonce 不列為語義欄位）。
  BraTS `(participant,z)` 聯集在 candidate/pair replay 完成後，另以第二個
  canonical-volume verification pass（每個 selected subject 只讀一次）和既有
  `model_input_sha256` 比對；這不是全 938 subjects 的單一 pass。 同一 key 的多個歷史 hash 若互相衝突會直接
  fail closed。healthy source tensor 與 Mixed underlying LMDB 則逐張檢查
  shape/dtype/finite 與 bytewise equality，並要求 source sidecar historical
  fingerprint continuity；runtime 凍結的
  `selected_healthy_source_tensor_ledger.jsonl` 會再逐個 source/local identity
  比對 tensor SHA。缺少 historical fingerprint 或 selected-tensor ledger 時，
  報告會明確寫 `INCONCLUSIVE`，不把 replay 當成自我證明。這個 replay 不
  呼叫 selected-cache writer、不複製 NPZ、不啟動 training，也不覆寫既有
  source。任何差異都寫成 `FAIL_CLOSED`，`control_binding_status` 保持未綁定；
  只有完整 candidate/ledger/semantic/tensor evidence 通過才會標成可綁定。
  producer 的 e20 bytes 缺失仍會在 protocol/audit 中原樣記錄，不能把 current
  code hash 當成歷史 producer snapshot。
  目前 runtime freeze 將 ledger 放在
  `model_grid_v3_fullcandidate_20260917_final/observed_input_binding_v1/`；其
  canonical row 以 `(canonical_source_dataset, canonical_source_split,
  canonical_source_key, z, tensor_sha256)` 為主鍵，`memberships` 保存每個
  comparison 的 local/underlying identity。replay 先讀 summary 的
  `ledger_path`，再逐 membership 比對，並檢查 duplicate underlying rows 的
  SHA 必須一致。
- 若 runtime 之後提供明確的 `secondary_48_manifest.json`（或直接傳入
  `summarise_secondary_48(...)` 的 mapping），report 可讀取 48 個 secondary
  cells。每個 SmallCNN/ResNet cell 必須列出固定 seeds `73/173/273`，而
  statistical Logistic cell 只列 seed `73`；每個 entry
  必須是明確的 `seed_results`，每個
  entry 指向一個 `result.json` 或 inline runner result；讀取器會要求三個
  result 的 subject identifiers 與 labels 完全一致，先逐 subject 平均
  probability，再計 subject metrics。缺 seed、duplicate seed、subject/label
  mismatch、非 binary label 或非 finite/out-of-range probability 都會
  `FAIL_CLOSED`，不會改用 slice predictions 補值。
- secondary family 的 p-value 只能使用 cell 明確保存的
  `secondary_p_value`（或 `p_value_source=secondary_subject_ensemble`）。
  `holm` 只在預期 48 cells 都有有效 p-value 時產生 family-level adjusted
  values；Holm step-down 不要求獨立性，但 family membership 必須事先固定。
  optional `primary_full_retrained` section 使用獨立的
  `primary_full_retrained_holm4` family 與 `full_retrained_p_value` 欄位；
  held-out conditional pair-swap p-values 會被排除，不能混入任一 family。
  目前尚未收到 runtime 的正式 secondary manifest 時，這段只提供 parser
  與 fail-closed schema support，不宣告任何 48-cell empirical result。
- `figures/`：ROC、PR、confusion、training curves、permutation、bootstrap、
  modality/cohort/seed comparison，以及可用時的 ANDi AP scatter；
  `figures/plot_manifest.json` 明列每一張圖是 `COMPLETE` 還是 `MISSING`。

ANDi correlation 的 AP source 預設只讀下列現存 run artifacts，並將 path 與
SHA-256 寫入 manifest；檔案不存在時不使用 request text 中的數字：

| cohort | AP artifact |
|---|---|
| FOMO | `outputs/runs/fomo45k_robust_iqr200_continue20_resume2/evaluation/brats21_test50_checkpoints40_20/epoch_0199/inference_metrics_summary.csv` |
| MPI | `outputs/runs/mpi_sri24_robust_iqr60_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv` |
| OASIS3 | `outputs/runs/oasis3_sri24_robust_iqr20_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv` |
| Mixed | `outputs/runs/mixed_sri24_robust_iqr20_own_spectrum/evaluation/brats21_test50/inference_metrics_summary.csv` |

Scatter uses the fixed v3 primary selector: `stage=final`, all three
`FLAIR+T1+T2` modalities, `SmallCNN`, and seed/init seed `73`, with one exact
subject-AUC row for each of FOMO, MPI, OASIS3, and Mixed. A missing, duplicate,
or conflicting selected row is retained as `PENDING`/fail-closed; another
model, modality, or seed is never used as fallback. The y values are the
audited `median_filter`/MF AP endpoints. Four cohort points are a descriptive
diagnostic only; if either axis is constant, Pearson/Spearman are undefined
and remain `NA` while the points can still be plotted. These values cannot
establish that acquisition differences caused the ANDi performance gap.

## v3 frozen-input runtime interface

[`domain_classifier/v3_runtime.py`](../domain_classifier/v3_runtime.py) is the
shared binding layer for the new v3 entrypoint. Its public
`validate_and_materialize_v3_inputs(...)` resolves one frozen cohort, verifies
the manifest SHA anchors from the healthy-ledger summary, checks participant /
pair / label joins, reads the canonical `[3,128,128]` `float32` model input once
per source identity, and compares every healthy tensor with the frozen ledger
and every BraTS tensor with `provenance.model_input_sha256`. Mixed healthy rows
use their underlying dataset/split/key for the tensor cache while retaining
their local membership key for the ledger join. `projected(...)` selects
FLAIR/T1/T2 channels from that same cache; it does not reopen an LMDB or NPZ.

`permute_cached_v3_labels(...)` creates deterministic independent train/val/test
streams and flips complete participant pairs only. `fit_cached_v3_cell(...)`
is the integration boundary for the existing SmallCNN/ResNet runner and the
train-only standardized Logistic control. Logistic uses one train fit for
both validation and test, records train-only scaler/model coefficients,
convergence details, and emits the configured pair-bootstrap CI.

[`scripts/run_domain_classifier_primary_null_v3.py`](../scripts/run_domain_classifier_primary_null_v3.py)
is the fixed primary observed/null entrypoint. It accepts one cohort's
`--manifest-root`, `--build-root`, `--output-root`, ledger paths and optional
reference `--config-path`; the reference config's core model/split/optimization
fields are checked, while the runner retains the frozen primary execution
contract (SmallCNN, all three channels, init seed `73`, 40 epochs/patience 8,
and exactly `199` whole-pair full-retrain draws). `--observed-only` performs
the observed fit and can resume it; `--dry-run` performs the full input
materialization/hash binding and writes `preflight.json` without fitting.
Before a fit it records `code_snapshot/manifest.json`, `source_freeze.json`,
`input_binding_audit.json`, and `cache_manifest.json`. Small protocol,
manifest, config, cache and selected-input files are byte-hashed. Large LMDB
files use the prelaunch fingerprint/size/mtime anchor plus selected tensor
byte checks, so the audit does not claim to rehash an entire database for each
cell. Resume accepts an artifact only when its result index, deterministic
labels, prediction joins, checkpoint and recorded file hashes all agree; an
incomplete or changed artifact fails closed. Formal rows remain pending until
the observed fit and all fixed null draws are committed.

## Bounded rollout and commands

先做 manifest/split/z/mask audit 與 unit tests；接著只做 FOMO、3-modal、Small
CNN 加 tiny/positive/negative/shuffle controls。任何 control FAIL 都停止擴張。
通過後才加單模態、Logistic、ResNet18、MPI/OASIS3/Mixed、多 training seeds、
bootstrap 與 permutation。

在 repository root 使用指定 interpreter：

```powershell
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe -m pytest -q tests\test_domain_classifier_report.py
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe -m pytest -q tests -k domain_classifier
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe -m andi_rewrite.domain_classifier.report <run_dir> --output-dir outputs\diagnostics\domain_classifier
# Full frozen-input binding for one real cohort, without starting a fit:
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\run_domain_classifier_primary_null_v3.py --comparison mpi --manifest-root outputs\diagnostics\domain_classifier\model_grid_v3_fullcandidate_20260917_final\manifests --build-root outputs\diagnostics\domain_classifier\model_grid_v3_fullcandidate_20260917_final --config-path configs\domain_classifier.yaml --output-root outputs\diagnostics\domain_classifier\v3_runtime_dryrun_mpi_<timestamp> --device cpu --dry-run
# Runtime/freeze owner launches this only after choosing a fresh replay root:
C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe scripts\replay_model_grid_v3.py --source-root outputs\diagnostics\domain_classifier\model_grid_v3_fullcandidate_20260917_final --output-root outputs\diagnostics\domain_classifier\model_grid_v3_replay_<timestamp>
```

The replay command above is a recommended invocation; it was not run during
this report update because it performs the full 938-subject MRI scan.

`report.py` 本身不啟動 training，也不會把 proposal 寫成完成成果。若 source
中沒有 formal final predictions/metrics，report 的 scientific status 會是
`INCONCLUSIVE`；control-only artifacts 不會被當作正式 cohort result。

## Current delivery status (2026-09-17)

這次 bounded report work 的 owned files 與用途如下：

- `domain_classifier/report.py`：summary/manifest/audit aggregation、formal-row
  filtering、九個問題的判讀文字、calibration-only plot helper，以及 v3
  model-grid 的 build-only context（四 cohort counts、audit pair/source
  composition、candidate-coverage/parity scope），以及 explicit secondary-48
  fixed-seed subject ensemble/Holm parser。
- `tests/test_domain_classifier_report.py`：control 不污染結論、authoritative
  audit 不被覆寫、calibration observed plots、incomplete/complete null gate，
  v3 build-only scope/candidate-coverage reporting，以及 secondary-48
  subject alignment、missing-seed fail-closed 與 disjoint Holm families。
- `docs/domain_classifier.md`：實際 code path、pairing/eligibility limitation、
  source provenance 與本節的 delivery record。
- `scripts/replay_model_grid_v3.py` 與 `tests/test_model_grid_v3_replay.py`：
  producer bytes 遺失時的獨立 full-candidate replay audit、semantic manifest
  diff、BraTS canonical/cache hash pass 與 healthy/Mixed source tensor parity；
  replay 只接受新 output root，實際 full scan 由 runtime freeze 後另行啟動。
- `domain_classifier.report.export_model_grid_v3_rosters(...)` 與
  `scripts/export_model_grid_v3_rosters.py`：唯讀 frozen v3 manifests，按
  FOMO/MPI/OASIS3/Mixed 與 train/val/test 匯出 participant-level CSV（pair、
  source、session、selected-slice count），並寫 participant/pair overlap audit。
  本次實際輸出位於
  `outputs/diagnostics/domain_classifier/model_grid_v3_roster_audit_20260917/`；
  四 cohort 的 split pair counts 是 FOMO `168/36/36`、MPI `81/17/17`、
  OASIS3 `172/37/37`、Mixed `421/90/90`，overlap/error audit 為空。
- `outputs/diagnostics/domain_classifier/fomo45k_calibration_smallcnn/calibration_figures/`：
  五張 observed diagnostic PNG 與 `plot_manifest.json`；manifest 的
  `formal_matrix_rows=0`，並保存 observed/protocol/null source SHA-256。
- `domain_classifier/v3_runtime.py` 與
  `scripts/run_domain_classifier_primary_null_v3.py`：v3 canonical cache、
  ledger/manifest/source/code freeze binding、single-thread verification、
  observed resume 與 fixed-199 conditional full-retrained pair-null entrypoint。
  `tests/test_domain_classifier_v3_runtime.py` 的 11 項 fixture tests 覆蓋
  FOMO/Mixed membership、wrong digest/identity、projection、pair permutation、
  logistic single-fit/bootstrap、dry-run 與 artifact-resume fail-closed。
  真實 MPI `--dry-run` 亦已完成：2,832 rows、1,416 healthy memberships、
  1,416 BraTS tensor provenance hashes，實測 torch/inter-op threads 均為 1；
  output 是 `outputs/diagnostics/domain_classifier/v3_runtime_dryrun_mpi_20260917_r2/`。
  這是 input binding evidence，沒有 classifier fit 或 empirical result。

以下是本次 bounded validation 實際執行的命令結果（較早的命令紀錄另見
`outputs/diagnostics/domain_classifier/implementation_validation.json`），
不是建議命令：較早的 full domain-classifier validation entry 為 **50 passed**；
本次 selector、binding、v3 gate 與 roster 修正後，最新
`tests/test_domain_classifier_report.py` 為 **34 passed**，replay fixture suite
為 **13 passed**，同一個 bounded command 合併執行為 **47 passed**（13 個
matplotlib/pyparsing deprecation warnings）。這個 command 只跑 report 與
replay fixture tests，不代表 112 fits 或 full MRI scan 已執行。Roster CLI
亦已用指定 interpreter 從 repo root 實測 PASS；其輸出與 source-manifest
SHA 保存在 `model_grid_v3_roster_audit_20260917_cli/`。Observed calibration 產生 ROC、PR、confusion、training
curves 與 subject bootstrap，並在 full-retrained null 完成後產生第六張
conditional matched-pair histogram。`retrained_null` 現為 terminal **199/199**，
statistic 是 `T=abs(subject_mean_score_roc_auc-0.5)`、p+1=`0.005`、minimum
attainable p=`0.005`。只有 status terminal complete 且 completed=requested=199
才會讀 statistic/results；calibration 仍是 diagnostic sub-study，不會自動變成
formal matrix row。

Observed calibration 的 AUC=1、recorded CI [1, 1] 與 completed null 只屬
舊 FOMO diagnostic calibration setting；在此 setting 可回答 observed fingerprint、
程度與 capacity adequacy，但不外推其他 cohort/modalities。v2 true-label gate
FAIL、random-held-out-label sanity PASS 是舊 v1/v2 protocol 的 historical status；
它不會改寫 v3 Stage-A tiny/positive/negative gate。v3 FOMO observed fit 已完成，
屬預先指定 primary 112-cell family 的 `1/112` observed fit；其 subject AUC=1
與 recorded empirical subject-bootstrap CI `[1,1]` 只描述目前 72 個 held-out
subjects。v3 input-binding audit 是 post-fit `PASS`（5896 rows；2948 BraTS
canonical checks；source-cache digest equal），immutable test-label sanity
artifact 已完成但其 random-label gate 是 `INCONCLUSIVE`，不能當作
low-separability 證據。v3 的 conditional full-retrained null 由 report 讀取
實際 `completed/requested`，在未達 `199/199` 前不得宣稱 calibration/null 完成；
完整 112-cell matrix 仍未宣告完成。這些狀態和原始 audit/manifest 都保留在各自
source path，沒有由 report 輸出覆寫。
