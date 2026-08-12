# UCSF-PDGM adapter 與資料契約

本文件描述 `data/datasets/ucsf_pdgm.py::UCSFPDGMVolumeDataset` 的現行程式契約；舊的 `andi_rewrite.data.datasets` import 由 package facade 保持相容。本文把本機 storage inventory 與可重用的 adapter semantics 分開。較早的完整 header/intensity 掃描保留在 [UCSF-PDGM MRI dataset storage inspection report](../../UCSF_PDGM_storage_inspection_report.md)，但其中 case counts 是歷史快照。

## 1. 用途與 logical channel order

UCSF-PDGM adapter 將資料集實際檔名映射到專案共通的四通道語意：

| Model channel | Logical modality | UCSF suffix |
|---:|---|---|
| 0 | `flair` | `FLAIR` |
| 1 | `t1` | `T1` |
| 2 | `t1ce` | `T1c` |
| 3 | `t2` | `T2` |

因此 input order 是：

```text
[FLAIR, T1, T1c, T2]
```

不是 `[T1,T1c,T2,FLAIR]`。這個順序必須與 BraTS-trained model 的 logical `[flair,t1,t1ce,t2]` contract 一致。

## 2. Folder discovery 與 completeness

Adapter 從 `dataset_path` 遞迴尋找所有 `*_nifti` directories。假設 folder id 是 `<subject-or-timepoint>`，完整 case 必須同時有：

```text
<id>_FLAIR.nii.gz
<id>_T1.nii.gz
<id>_T1c.nii.gz
<id>_T2.nii.gz
<id>_tumor_segmentation.nii.gz
```

空目錄、下載暫存目錄或只包含部分 modalities 的 folder 不會成為可用 case。Complete id 若在不同路徑重複：

- `duplicate_policy: error`：default，初始化失敗；
- `duplicate_policy: first`：依 discovery 排序保留第一筆。

Follow-up id（如 `UCSF-PDGM-0391_FU016d`）保留完整 suffix，不會折疊成 base patient id。

## 3. CSV selection

`path_to_csv` optional；沒有時使用所有 discovered complete cases。有 CSV 時：

- 必須有名稱精確為 `subject_id` 的欄位；
- blank id、duplicate id、storage 中不存在/不完整的 id 都會失敗；
- output subject order 保留 CSV order；
- `subject_limit` 在 selection 後套用；null/0 表示不限制。

範例：

```csv
subject_id
UCSF-PDGM-0004
UCSF-PDGM-0391_FU016d
```

若 baseline 與 follow-up 屬於同一 patient，建立 train/validation/test splits 時應先正規化到 base patient id，避免不同 timepoint 跨 split 造成 leakage。Dataset adapter 本身不做 patient-group split。

## 4. YAML

```yaml
data:
  type: ucsf_pdgm_volume
  dataset_path: C:/ML/data/UCSF-PDGM
  path_to_csv: splits/UCSF-PDGM/scans_test_251.csv
  modalities: [flair, t1, t1ce, t2]
  modality_mapping:
    flair: FLAIR
    t1: T1
    t1ce: T1c
    t2: T2
  segmentation_suffix: tumor_segmentation
  reference_modality: flair
  model_orientation: LPS
  image_size: 128
  histogram_normalization: false
  return_metadata: true
  subject_limit: null
  duplicate_policy: error
  batch_size: 1
  shuffle: false
```

`reference_modality` 必須在 logical `modalities` 中。`model_orientation` 也可寫成 `expected_orientation`；設為 null 可略過 orientation check，但 shape/spacing/affine validation 仍是 adapter 的核心保護。

## 5. Geometry validation

每個 selected case 都會載入四個 modalities 與 tumor segmentation，並要求：

- 都是 3-D；
- shape 相同；
- voxel spacing 相同；
- orientation 相同，且預設為 `LPS`；
- affine matrix 在 tolerance 內相同。

這比通用 `MRIDataVolume` 嚴格。Mismatch 會拋出包含 subject/modality context 的 error；adapter 不會偷偷把 UCSF modalities resample 到彼此 grid。

`image_size` 只把 XY resize 到 model grid，Z 保持不變。Model input 因此是 `[4,H,W,Z]`，DataLoader 加 batch 後為 `[B,4,H,W,Z]`。

## 6. Intensity 與 segmentation

每個 modality 預設做 foreground p99 normalization：只使用正值估計第 99 percentile，再以該值相除；沒有 clipping。`histogram_normalization: true` 則選用 foreground histogram equalization path。

UCSF tumor segmentation 原始 labels 可包含 `{0,1,2,4}` 的子集合。Adapter 對 evaluation 回傳：

```text
whole_tumor_mask = segmentation > 0
```

因此目前 metrics 是 binary whole-tumor evaluation，不保留 ET/TC/WT 的多類別語意。

## 7. Metadata 與 prediction export

`return_metadata: true` 是 default。Metadata 提供 subject id、reference NIfTI path、native shape/geometry 與 label availability，供 evaluator：

- 在 report/cache 中識別 subject；
- 將 `[H,W,Z]` model-grid score/mask restore 到 native shape；
- 使用 reference NIfTI affine/header 寫 prediction。

Restore 只依 shape 對 score 做 trilinear、mask 做 nearest interpolation；它不是 affine registration。因 adapter 事先驗證共同 geometry，這個假設比通用 volume adapter 更受約束。

目前 metadata 的 `native_spacing` 文字欄位會把各 spacing 分量轉成 integer，因此非整數 spacing 可能失真；prediction export 不依賴這個文字欄位，而是重新載入 `reference_path` 的 NIfTI 作為 shape/affine/header authority。

Prediction export 需要 canonical standalone evaluation；in-memory 會在 collect 後匯出，disk streaming 則在 metrics pass 逐 cached subject 匯出。兩者都需要完整 reference metadata。

## 8. 本機 storage 輕量 inventory

2026-08-12 以與 adapter 相同的 filename completeness 規則，對 `C:\ML\data\UCSF-PDGM` 做唯讀重掃：

| 項目 | 數量 |
|---|---:|
| Candidate `*_nifti` folders | 563 |
| Complete case/timepoint folders | 257 |
| Standard ids | 251 |
| Follow-up ids | 6 |
| Duplicate complete ids | 0 |

Checked-in `splits/UCSF-PDGM/scans_test_251.csv` 有 251 個 unique ids；本次重掃確認它與 251 個 complete standard ids 完全相符，沒有 missing 或 extra standard id。

這是本機 filesystem 在該日期的 inventory，不是 dataset release 的永久規格。Storage 下載進度改變後應重新檢查；runtime 的最終真實來源仍是 adapter discovery/validation 與實際 CSV。

舊的 detailed inspection 記錄 245 個 complete cases，並對當時資料做完整 NIfTI header、affine、label 與 intensity 統計。那些詳細數值不能自動外推到後來新增的 12 個 standard cases；因此歷史報告保留原始統計並加上 snapshot 警告，而不是把所有 245-based 表格機械改成 257。

## 9. Failure modes

- Dataset root 不存在或找不到任何 complete case。
- CSV 缺 `subject_id`、有 blanks/duplicates、或要求的 id 不完整。
- 同 id 在 storage 重複且 policy 是 `error`。
- Required modality/segmentation 不存在。
- Non-3D image。
- Shape、spacing、orientation 或 affine mismatch。
- `reference_modality` 不在 logical modalities。
- Dataset/model/spectrum channel or H/W mismatch。
- Native prediction restore 缺 reference metadata/nibabel。

這些 validation 是 fail-fast；不會在錯誤 geometry 上自動繼續 inference。

## 10. Tests

`tests/test_ucsf_pdgm_dataset.py` 覆蓋 discovery、CSV selection、logical/physical channel order、geometry/orientation validation、partial archive rejection、resize 與 metadata。這些是 synthetic fixture tests；它們不取代對實際 storage、checkpoint compatibility 與完整 inference 的驗證。
