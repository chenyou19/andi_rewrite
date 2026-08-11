# UCSF-PDGM MRI dataset storage inspection report

## 檢查範圍與方法

- 檢查根目錄：`C:\ML\data\UCSF-PDGM`
- 全程唯讀；沒有修改、搬移、重新命名、轉檔、resampling、normalization、registration、cropping、padding 或 training。
- Python：`nibabel 5.1.0`、`numpy 1.24.1`
- 以 `nib.load()` 讀取 NIfTI header，檢查 `img.shape`、`img.header.get_zooms()`、`img.affine`、`nib.aff2axcodes(img.affine)`、dtype；再讀取完整 voxel data 做 segmentation label 與核心四模態 intensity 統計。

## 先看結論

1. 目前本機根目錄包含兩個資料樹：根目錄下 50 個完整 case，以及 `PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5` 下的下載樹。
2. package 樹有 501 個 case 目錄，但 299 個是空目錄、6 個只有 `.partial/.aspera-ckpt`、1 個部分完成；package 中真正完整可讀的是 195 個 case folder。
3. 合併目前兩個資料樹後，有 245 個完整可讀的 case/timepoint folder：239 個一般 case、6 個 follow-up case。
4. 所有完整 case 都有獨立的 `T1`、`T1c`、`T2`、`FLAIR` 三維 NIfTI volume；不是一個 4-channel 檔案。
5. 核心四模態以及其它已重採樣到主 grid 的 scalar map/mask 全部是 `(240, 240, 155)`、`1 mm` spacing、`LPS` orientation，且 affine 完全相同：

   ```text
   [[-1,  0, 0,   0],
    [ 0, -1, 0, 239],
    [ 0,  0, 1,   0],
    [ 0,  0, 0,   1]]
   ```

6. `DTI_eddy_noreg.nii.gz` 是例外：它是原始 4-D DTI，shape、spacing 不固定，orientation 是 `LAS`，不是核心 240×240×155 grid。
7. tumor mask 存在且不是單純二值；實際 label union 是 `{0, 1, 2, 4}`。

## 1. 最上層目錄結構

```text
C:\ML\data\UCSF-PDGM\
├── UCSF-PDGM-0004_nifti\
├── UCSF-PDGM-0005_nifti\
├── UCSF-PDGM-0007_nifti\
├── ...
├── UCSF-PDGM-0058_nifti\
└── PKG - UCSF-PDGM Version 5\
    └── UCSF-PDGM-v5\
        ├── UCSF-PDGM-0004_nifti\       [空目錄 placeholder]
        ├── ...
        ├── UCSF-PDGM-0288_nifti\       [空目錄 placeholder]
        ├── UCSF-PDGM-0345_nifti\       [.partial/.aspera-ckpt]
        ├── ...
        ├── UCSF-PDGM-0351_nifti\       [部分完成]
        ├── UCSF-PDGM-0352_nifti\       [完整]
        ├── UCSF-PDGM-0391_FU016d_nifti\ [完整 follow-up]
        ├── ...
        └── UCSF-PDGM-0541_nifti\       [完整]
```

### Case folder 統計

| 資料位置 | case folder 數 | 空目錄 | 部分下載 | 完整可讀 | 完整 case 內通常的檔案 |
|---|---:|---:|---:|---:|---|
| 根目錄直接子目錄 | 50 | 0 | 0 | 50 | 23 個 `.nii.gz` + 1 個 rotated bvec |
| `PKG - ...\UCSF-PDGM-v5` | 501 | 299 | 7 | 195 | 一般 23 個 `.nii.gz` + bvec；6 個 FU case 多一個 `ASL_M0` |
| 目前完整可讀資料合計 | 245 | — | — | 245 | 239 個一般 case + 6 個 FU case |

package 中的 7 個未完成目錄是：

```text
UCSF-PDGM-0345_nifti
UCSF-PDGM-0346_nifti
UCSF-PDGM-0347_nifti
UCSF-PDGM-0348_nifti
UCSF-PDGM-0349_nifti
UCSF-PDGM-0350_nifti   # 只有 .partial/.aspera-ckpt
UCSF-PDGM-0351_nifti   # 有 7 個已完成 NIfTI，其餘仍是暫存檔
```

根目錄直接 case 的實際命名不是連續編號，例如 `0004, 0005, 0007, ... 0058`。package 另外可見 follow-up 命名：

```text
UCSF-PDGM-0391_FU016d_nifti
UCSF-PDGM-0396_FU175d_nifti
UCSF-PDGM-0409_FU001d_nifti
UCSF-PDGM-0429_FU003d_nifti
UCSF-PDGM-0431_FU001d_nifti
UCSF-PDGM-0433_FU007d_nifti
```

代表 case 選擇：

| 資料樹 | 排序後第一個目錄 | 排序後中間目錄 | 排序後最後目錄 | 第一/中間/最後完整 case |
|---|---|---|---|---|
| 根目錄直接 case | `0004` | `0031` | `0058` | `0004 / 0031 / 0058` |
| package 全部目錄 | `0004`（空） | `0288`（空） | `0541`（完整） | — |
| package 完整 case | `0352` | `0444` | `0541` | `0352 / 0444 / 0541` |

## 2. 檔案格式

在整個 `C:\ML\data\UCSF-PDGM` 遞迴掃描到的副檔名：

| 類型 | 數量 | 說明 |
|---|---:|---|
| `.nii.gz` | 5,648 | NIfTI compressed；其中 5,641 個屬於完整 case，另外 7 個在部分完成的 `0351` |
| `.eddy_rotated_bvecs` | 246 | DTI rotated b-vector text file；完整 case 有 245 個 |
| `.partial` | 160 | Aspera 下載暫存檔 |
| `.aspera-ckpt` | 160 | Aspera checkpoint 暫存檔 |
| `.nii` | 0 | 未發現未壓縮 NIfTI |
| DICOM / `.dcm` | 0 | 未發現 |
| NRRD / `.nrrd` | 0 | 未發現 |
| MHA / `.mha` | 0 | 未發現 |
| `.bvec` / `.bval` | 0 | 只有 `.eddy_rotated_bvecs`，未發現標準副檔名 |

可讀 NIfTI 以 NIfTI-1 header (`magic = n+1`) 載入；實際儲存檔名是 `.nii.gz`。

## 3. 一個真實 case 實際怎麼存？

以根目錄的真實 case `UCSF-PDGM-0004_nifti` 為例，完整檔案名稱如下：

```text
C:\ML\data\UCSF-PDGM\UCSF-PDGM-0004_nifti\
├── UCSF-PDGM-0004_ADC.nii.gz
├── UCSF-PDGM-0004_ASL.nii.gz
├── UCSF-PDGM-0004_DTI_eddy.eddy_rotated_bvecs
├── UCSF-PDGM-0004_DTI_eddy_FA.nii.gz
├── UCSF-PDGM-0004_DTI_eddy_L1.nii.gz
├── UCSF-PDGM-0004_DTI_eddy_L2.nii.gz
├── UCSF-PDGM-0004_DTI_eddy_L3.nii.gz
├── UCSF-PDGM-0004_DTI_eddy_MD.nii.gz
├── UCSF-PDGM-0004_DTI_eddy_noreg.nii.gz
├── UCSF-PDGM-0004_DWI.nii.gz
├── UCSF-PDGM-0004_DWI_bias.nii.gz
├── UCSF-PDGM-0004_FLAIR.nii.gz
├── UCSF-PDGM-0004_FLAIR_bias.nii.gz
├── UCSF-PDGM-0004_SWI.nii.gz
├── UCSF-PDGM-0004_SWI_bias.nii.gz
├── UCSF-PDGM-0004_T1.nii.gz
├── UCSF-PDGM-0004_T1_bias.nii.gz
├── UCSF-PDGM-0004_T1c.nii.gz
├── UCSF-PDGM-0004_T1c_bias.nii.gz
├── UCSF-PDGM-0004_T2.nii.gz
├── UCSF-PDGM-0004_T2_bias.nii.gz
├── UCSF-PDGM-0004_brain_parenchyma_segmentation.nii.gz
├── UCSF-PDGM-0004_brain_segmentation.nii.gz
└── UCSF-PDGM-0004_tumor_segmentation.nii.gz
```

一般完整 case 的命名規則是：

```text
UCSF-PDGM-<case-or-followup-label>_<modality>.nii.gz
```

檔名使用 `T1c`，沒有看到名為 `T1CE`、`T1Gd` 或 `post-contrast` 的檔案。`T1c` 是本機資料實際使用的 contrast-enhanced T1 命名。

### 實際出現的 modality / 檔名群組

| 群組 | 實際 suffix |
|---|---|
| 核心 anatomical MRI | `T1`, `T1c`, `T2`, `FLAIR` |
| 其它 3-D MRI / quantitative map | `ADC`, `DWI`, `ASL`, `SWI` |
| bias-corrected / bias variant | `T1_bias`, `T1c_bias`, `T2_bias`, `FLAIR_bias`, `DWI_bias`, `SWI_bias` |
| DTI derived scalar maps | `DTI_eddy_FA`, `DTI_eddy_L1`, `DTI_eddy_L2`, `DTI_eddy_L3`, `DTI_eddy_MD` |
| 原始/未配準 DTI | `DTI_eddy_noreg`，4-D |
| DTI gradient text | `DTI_eddy.eddy_rotated_bvecs`；每個完整 case 168 個 whitespace-separated numbers，即 3×56 |
| segmentation | `brain_segmentation`, `brain_parenchyma_segmentation`, `tumor_segmentation` |
| follow-up 特有 | `ASL_M0`，只在 6 個 FU case 出現 |

每個完整 case 都有上述標準群組中的 23 個 `.nii.gz`；6 個 FU case 另有 `ASL_M0`，因此有 24 個 `.nii.gz`。

## 4. 代表 case 的 NIfTI metadata

以下是實際讀取的第一、中間、最後完整 case。核心 grid 的結果在 245 個完整 case 都相同。

| Case | T1/T1c/T2/FLAIR shape | header dtype | spacing | orientation | qform/sform |
|---|---|---|---|---|---|
| `0004` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |
| `0031` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |
| `0058` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |
| `0352` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |
| `0444` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |
| `0541` | `(240, 240, 155)` | `int16` | `(1, 1, 1)` mm | `LPS` | `1 / 1` |

以 `UCSF-PDGM-0004` 的檔案為例，逐個主要檔案讀到：

```text
T1:
  path: ...\UCSF-PDGM-0004_T1.nii.gz
  shape: (240, 240, 155)
  dtype: int16
  spacing: (1.0, 1.0, 1.0)
  orientation: LPS
  affine: [[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]]

T1c:
  path: ...\UCSF-PDGM-0004_T1c.nii.gz
  shape: (240, 240, 155)
  dtype: int16
  spacing: (1.0, 1.0, 1.0)
  orientation: LPS
  affine: same as T1

T2:
  path: ...\UCSF-PDGM-0004_T2.nii.gz
  shape: (240, 240, 155)
  dtype: int16
  spacing: (1.0, 1.0, 1.0)
  orientation: LPS
  affine: same as T1

FLAIR:
  path: ...\UCSF-PDGM-0004_FLAIR.nii.gz
  shape: (240, 240, 155)
  dtype: int16
  spacing: (1.0, 1.0, 1.0)
  orientation: LPS
  affine: same as T1

Segmentation:
  path: ...\UCSF-PDGM-0004_tumor_segmentation.nii.gz
  shape: (240, 240, 155)
  dtype: uint8
  spacing: (1.0, 1.0, 1.0)
  orientation: LPS
  affine: same as T1
```

其它已對齊到主 grid 的 scalar map（`ADC`, `DWI`, `ASL`, `SWI`、所有 bias variant、DTI FA/L1/L2/L3/MD）以及三個 segmentation，在全部 245 個完整 case 都得到同一個 `(240,240,155)`、`(1,1,1)`、`LPS`、T1 affine。

### `DTI_eddy_noreg` 的 metadata 例外

| Case | shape | voxel spacing | dtype | orientation | qform/sform |
|---|---|---|---|---|---|
| `0004` | `(256,256,69,56)` | `(1,1,2)` mm | `int16` | `LAS` | `0/2` |
| `0031` | `(256,256,75,56)` | `(1.0938,1.0938,2)` mm | `int16` | `LAS` | `0/2` |
| `0058` | `(256,256,75,56)` | `(1.0938,1.0938,2)` mm | `int16` | `LAS` | `0/2` |
| `0352` | `(256,256,78,56)` | `(1.0938,1.0938,2)` mm | `int16` | `LAS` | `0/2` |
| `0444` | `(256,256,73,56)` | `(1.0938,1.0938,2)` mm | `int16` | `LAS` | `0/2` |
| `0541` | `(256,256,71,56)` | approximately `(1.0938,1.0938,1.99993)` mm | `int16` | `LAS` | `0/2` |

## 5. Dataset-level shape / spacing / affine 統計

### 核心與已對齊 scalar volume

在 245 個完整 case 中，`T1`, `T1c`, `T2`, `FLAIR`, `ADC`, `DWI`, `ASL`, `SWI`、bias variants、DTI FA/L1/L2/L3/MD、三個 segmentation 的統計都是：

```text
shape distribution:
  (240, 240, 155): 245 cases

voxel spacing distribution:
  (1.0, 1.0, 1.0) mm: 245 cases

orientation distribution:
  LPS: 245 cases

affine distribution:
  上述同一個 4×4 affine: 245 cases
```

同一 case 內以 T1 對照，以下每一項都是全部通過：

| 檔案群組 | shape | spacing | affine | orientation |
|---|---:|---:|---:|---:|
| `T1c`, `T2`, `FLAIR` | 245/245 | 245/245 | 245/245 | 245/245 |
| `ADC`, `DWI`, `ASL`, `SWI` | 245/245 | 245/245 | 245/245 | 245/245 |
| bias variants | 245/245 | 245/245 | 245/245 | 245/245 |
| DTI FA/L1/L2/L3/MD | 245/245 | 245/245 | 245/245 | 245/245 |
| brain/tumor masks | 245/245 | 245/245 | 245/245 | 245/245 |
| `ASL_M0`（只有 6 個 FU） | 6/6 | 6/6 | 6/6 | 6/6 |

這表示核心四模態和上述 derived/mask volume 已經在同一個 voxel grid；從 shape、spacing、affine、orientation metadata 看，彼此是 co-registered/aligned 的。這個結論不適用於 `DTI_eddy_noreg`。

### 原始 DTI 4-D volume

所有 245 個完整 case 的 `DTI_eddy_noreg` 都是 4-D、最後一軸固定 56，但第三軸不固定；orientation 全部是 `LAS`，qform/sform 全部是 `0/2`。

以 `(256,256,Z,56)` 的 Z 分布表示：

```text
根目錄 50 個 case:
Z=45:1, 52:1, 63:2, 64:3, 66:3, 69:2, 70:1, 71:2,
Z=72:4, 73:2, 74:6, 75:14, 76:3, 77:1, 78:2, 80:1, 82:1, 83:1

package 完整 195 個 case:
Z=38:1, 45:1, 47:1, 48:1, 51:2, 52:1, 56:1, 57:1, 60:1,
Z=64:1, 65:1, 67:1, 70:4, 71:6, 72:13, 73:10, 74:17, 75:30,
Z=76:18, 77:5, 78:30, 79:1, 80:10, 81:5, 82:4, 83:8, 84:4,
Z=85:5, 86:3, 87:3, 88:4, 90:1, 91:1
```

原始 DTI 的 spacing（四捨五入到 0.001 mm）分布為：

```text
(1.000, 1.000, 2.000): 34
(1.094, 1.094, 2.000): 194
(1.094, 1.094, 2.700): 2
(1.125, 1.125, 3.000): 5
(0.938, 0.937, 2.000): 1
(1.000, 1.000, 2.500): 1
(1.016, 1.016, 2.000): 3
(1.016, 1.016, 3.000): 1
(1.094, 1.094, 3.000): 1
(1.094, 1.094, 3.600): 1
(1.133, 1.133, 2.000): 1
(1.172, 1.172, 3.000): 1
```

## 6. Segmentation / tumor mask

每個完整 case 都有以下三個 segmentation：

| 檔名 pattern | case 數 | dtype | shape / spacing / orientation | 實際 labels |
|---|---:|---|---|---|
| `<case>_brain_segmentation.nii.gz` | 245 | `int16` | `(240,240,155)`, 1 mm, LPS | `{0,1}`（nibabel physical scaling 可能顯示 `0.9999999998`） |
| `<case>_brain_parenchyma_segmentation.nii.gz` | 245 | `int16` | `(240,240,155)`, 1 mm, LPS | `{0,1}`（同上） |
| `<case>_tumor_segmentation.nii.gz` | 245 | `uint8` | `(240,240,155)`, 1 mm, LPS | union `{0,1,2,4}` |

tumor mask 的 label set 分布：

```text
{0,1,2,4}: 200 cases
{0,2}:     25 cases
{0,2,4}:   17 cases
{0,1,2}:    3 cases
```

因此不要把 `tumor_segmentation` 直接當成只有 `0/1` 的 binary mask；如果 pipeline 要 binary tumor mask，需要另外決定 label mapping，但本次沒有做任何 mapping。

## 7. MRI voxel intensity

以下是全部 245 個完整 case 的核心四模態，數值是用 nibabel data proxy 讀出的 physical intensity；global mean 是所有 finite voxels 的加權平均。

| Modality | global min | global max | global mean | 有負值的 volume | zero voxel fraction | 每 volume zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 0 | 13,308.66 | 322.53 | 0/245 | 82.885% | 77.949%–88.313% |
| T1c | 0 | 25,471.70 | 483.65 | 0/245 | 82.885% | 77.949%–88.313% |
| T2 | 0 | 5,603.49 | 109.58 | 0/245 | 82.883% | 77.949%–88.313% |
| FLAIR | 0 | 8,148.07 | 179.79 | 0/245 | 82.884% | 77.949%–88.313% |

核心四模態沒有 NaN/Inf，也沒有負值；background 通常是 0，而且零值約佔 83%。不過這個「無負值」結論只適用核心四模態，不應套用到所有 derived modality：

- 代表 case `0004` 的 `ADC` 約為 `-0.000256` 到 `0.005825`，有少量負值。
- `ASL`、`DTI_eddy_L1/L2/L3/MD` 在代表 case 中也可出現負值。
- 不同 volume 的 NIfTI `scl_slope/scl_inter` 不相同；例如 `0004_T1` 的 slope/intercept 約為 `0.03912/1281.87`，`0541_T1` 約為 `0.08414/2757.01`。四個核心模態的 scaling 參數在 245 個 case 中都不是固定單一值。
- 因此 header dtype 是 `int16` 不代表直接把 raw integer 當作統一 intensity；讀取 physical intensity 時應讓 nibabel 套用 NIfTI scaling，例如 `get_fdata(dtype=np.float32)`。

## 8. 接 3-D deep learning pipeline 的自然讀法

四個核心 volume 是四個獨立的 3-D 檔案；最自然的 input 組法是依照實際檔名明確指定：

```python
from pathlib import Path
import nibabel as nib
import numpy as np

case_dir = Path(r"C:\ML\data\UCSF-PDGM\UCSF-PDGM-0004_nifti")
case_id = "UCSF-PDGM-0004"

modality_order = ["T1", "T1c", "T2", "FLAIR"]
volumes = []

for modality in modality_order:
    path = case_dir / f"{case_id}_{modality}.nii.gz"
    img = nib.load(str(path))
    volume = img.get_fdata(dtype=np.float32)
    assert volume.shape == (240, 240, 155)
    volumes.append(volume)

image = np.stack(volumes, axis=0)
# image.shape == (4, 240, 240, 155)

seg = nib.load(
    str(case_dir / f"{case_id}_tumor_segmentation.nii.gz")
).get_fdata()
```

根據本機資料確認的 channel mapping 是：

```text
channel 0 = T1
channel 1 = T1c       # contrast-enhanced T1；檔名實際是 T1c
channel 2 = T2
channel 3 = FLAIR
```

`np.stack` 的結果確實可形成 `(4, 240, 240, 155)`。如果目前 model 把陣列 index 軸命名成 `(C,D,H,W)`，可以直接視為 `(4,D,H,W)`；但 NIfTI 的原始陣列軸是 `(i,j,k)`，且 `aff2axcodes` 是 `LPS`，所以 `D/H/W` 的語意仍應依目前 pipeline 的 axis convention 明確處理，不能只靠變數名稱猜測。

建議 default 4-channel input 只用 `T1/T1c/T2/FLAIR`；`*_bias`、ADC、DWI、ASL、SWI、DTI map 是另外的可選輸入，不應無意間混進四通道。`ASL_M0` 只在 6 個 FU case 存在，也不適合默認加入固定通道。

另需在 split 時注意：`UCSF-PDGM-0391_nifti` 與 `UCSF-PDGM-0391_FU016d_nifti` 共享 base patient ID，但代表不同 timepoint；若它們要被視為同一患者，train/validation/test split 不應讓同一患者的不同 timepoint 跨 split。

## 9. 與 BraTS 結構比較

概念上可以整理成類似 BraTS 的四個 input + 一個 segmentation：

```text
Case/
├── t1.nii.gz
├── t1ce.nii.gz
├── t2.nii.gz
├── flair.nii.gz
└── seg.nii.gz
```

但目前資料不能直接用「只尋找 BraTS 固定檔名」的 loader，差異是：

1. 實際檔名帶完整 prefix，例如 `UCSF-PDGM-0004_T1.nii.gz`，不是 `t1.nii.gz`。
2. contrast-enhanced T1 實際叫 `T1c`，不是 `t1ce`。
3. segmentation 實際叫 `tumor_segmentation`，不是 `seg`。
4. 每個 case 還有 ADC、DWI、ASL、SWI、bias variants、DTI derived maps、raw DTI 及 bvec text。
5. 有 `_FUxxx` follow-up folder，不能只用四位數 patient ID 當唯一資料夾 key。
6. tumor labels 實際是多類別 `{0,1,2,4}` 的子集合，不是保證 `{0,1}`。
7. 本機核心四模態的 geometry 已經非常接近可直接使用的 common grid：245 個完整 case 全部相同 shape/spacing/affine/orientation；但 raw `DTI_eddy_noreg` 仍是另一套 4-D geometry。

所以最小的 BraTS-like adapter 只需做檔名映射與 case completeness 檢查，不需要在本次 inspection 階段做 resampling 或資料轉換。

## 10. 最終 summary

| 項目 | 實際檢查結果 |
|---|---|
| Case 數量 | 245 個完整可讀 case/timepoint folder；239 個一般 + 6 個 FU。另有 package 空/未完成目錄。 |
| MRI 格式 | `.nii.gz`，可讀為 NIfTI-1；未發現 `.nii`、DICOM、NRRD、MHA。 |
| T1 | `<case>_T1.nii.gz`，245/245，`(240,240,155)`，int16 header，LPS，1 mm。 |
| T1CE / post-contrast T1 | 實際命名 `<case>_T1c.nii.gz`，245/245。 |
| T2 | `<case>_T2.nii.gz`，245/245。 |
| FLAIR | `<case>_FLAIR.nii.gz`，245/245。 |
| Segmentation | brain、brain parenchyma、tumor 三種各 245 個；tumor labels union `{0,1,2,4}`。 |
| 常見 shape | 核心/已對齊 volume：`(240,240,155)`；raw DTI：`(256,256,Z,56)` 且 Z 變動。 |
| shape 是否全部一致 | 核心四模態及 aligned maps/masks：是；raw DTI：否。 |
| voxel spacing | 核心 grid：`(1,1,1)` mm；raw DTI：case-dependent。 |
| orientation | 核心 grid：`LPS`；raw DTI：`LAS`。 |
| affine | 核心 245 個 case 完全相同；aligned masks/maps 也與 T1 相同。 |
| 是否已配準 | 核心四模態及 aligned maps/masks 的 metadata 顯示已在同一 common grid；`DTI_eddy_noreg` 是未配準/原始 grid 例外。 |
| 最自然 input | `np.stack([T1,T1c,T2,FLAIR], axis=0)`，得到 `(4,240,240,155)`，再依 pipeline axis convention 處理。 |

本報告只做 dataset inspection + storage format analysis；沒有修改任何 dataset 檔案或現有程式碼。
