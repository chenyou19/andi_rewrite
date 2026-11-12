# 可選 AER bottleneck 精煉

AER（Anchored Energy Refinement）在 DDPM denoiser 的既有 encoder 與 decoder 之間精煉一次 bottleneck。ANDiUNet 的插入點在 `sa4` 後、`up1` 前。ConvNeXtUNet 的插入點在 bottleneck stage 後、decoder loop 前。原有 timestep embedding 已進入 bottleneck，能量係數只在當次 denoiser forward 內使用。對同一張影像換一個 diffusion timestep 呼叫模型，會重新建立係數。原有 skip 與 `model(x_t, t)` 的 noise tensor 介面沿用既有行為。

## 使用方式

可從 [configs/train_aer.yaml](../configs/train_aer.yaml) 開始。資料與 split 路徑沿用原始 `configs/train.yaml`，請依本機資料位置填寫。現有 YAML 省略 `model.aer` 時維持停用。

```powershell
python -B scripts\train.py --config configs\train_aer.yaml --run-one-step
python -B scripts\train.py --config configs\train_aer.yaml --fit
```

以下命令在 repository root 執行。合成 smoke 使用 PyTorch 與現有 DDPM／Trainer 元件，無需 MRI 資料或 MRI 套件。測試與 smoke 入口使用兩條 CPU intra-op threads。Smoke 各訓練兩種 backbone 三次，驗證正常推論、能量 trace、auxiliary gradient、checkpoint 與 EMA 重載。輸出位於 `outputs/aer_smoke/`，可透過 `--output-dir` 指定位置。

```powershell
python -B scripts\smoke_aer.py
python -B -m unittest discover -s tests -p test_aer.py -v
```

## `model.aer` 設定

| 欄位 | 預設 | 行為 |
| --- | --- | --- |
| `enabled` | `false` | 為 true 才建立 AER 參數。 |
| `steps` | `3` | 每次 denoiser forward 的更新上限。設 0 直接回傳原 bottleneck。 |
| `latent_channels` | `16` | fine、coarse 與 token 的維度。須可被 `attention_heads` 整除。 |
| `num_tokens` | `4` | 每張影像的全域 token 數量。 |
| `attention_heads` | `2` | coarse 與 token 雙向 cross-attention 的 head 數。 |
| `mode` | `guarded` | 逐樣本驗收 learned proposal。其他選項為 `reference`、`unguarded`、`pcg`。 |
| `aux_loss_weight` | `0.1` | 訓練時 proposal 蒸餾的權重。設 0 可關閉蒸餾。 |
| `residual_scale` | `0.05` | 將精煉位移接回 bottleneck 的固定倍率，預設為小而非零。 |
| `residual_tolerance` | `null` | 設非負數時，以每樣本 `‖∇E‖₂ / w_min` 早停。 |
| `freeze_backbone` | `false` | 只有明確開啟時才固定原始 denoiser 權重及其 eval 行為。 |
| `w_min` / `w_max` | `0.8` / `1.2` | 各 latent 分量的 anchor precision 範圍。 |
| `lambda_m` / `lambda_s` / `lambda_e` | `0.7` / `0.5` / `0.4` | 跨尺度、全域 routing、局部邊的非負耦合係數。 |

`reference` 使用 `v - ∇E / L`，`unguarded` 直接採 learned proposal，`pcg` 使用相同能量的矩陣自由 Jacobi-preconditioned CG。對 K=0、reference 和 PCG，訓練無 proposal 蒸餾。Accelerate 多程序模式應沿用 `runtime.find_unused_parameters: true`，因為部分模式或早停可以略過 proposal 參數。訓練經由包裝後的模型 forward 回傳預測與蒸餾 loss，避免在模型實例上保留計算圖。

## 訓練、凍結與度量

Trainer 對 AER 啟用的 run 回報 `noise_loss`、未加權 `auxiliary_loss` 和 `total_loss`。總 loss 為 noise MSE 加上 `aux_loss_weight * auxiliary_loss`。蒸餾以從當前 latent 出發的三步 PCG 位移為目標，目標停止梯度。即使 guarded 拒絕候選，蒸餾仍可訓練 proposer。Validation 仍評估 noise MSE。新 run 的 `training_metrics.csv` 會記錄每 epoch 的 `train_noise_loss`、`train_auxiliary_loss` 與原有 `train_loss`（總 loss）。

`freeze_backbone: true` 時，Trainer 只將可訓練的 AER 參數放入 optimizer，並讓原有 encoder／decoder 子模組保持 eval 模式。Decoder 的運算仍可微，使 noise MSE 的梯度能抵達 AER。原始權重需要先透過 warm start 載入，才能進行固定預訓練 backbone 的比較。

## Checkpoint 與 EMA

停用 AER 時，舊 checkpoint 可照原流程 strict 載入，模型建構也維持原本的 RNG 消耗。開啟 AER 後，`training.checkpoint.resume` 應使用對應 AER 架構與訓練設定的 checkpoint。Eval 使用相同模型結構，依原先的 `model.use_ema` 選擇一般或 EMA 權重。EMA 包含 AER 參數。

要從舊 denoiser 啟動 AER 訓練，使用下列權重匯入流程並新建 Trainer。Helper 會預先檢查所有 backbone keys 與 tensor shapes，載入後保留新建的 AER 權重。

```python
import sys
from pathlib import Path
import torch

# 此範例從 repository root 執行，先加入 package parent。
sys.path.insert(0, str(Path.cwd().parent))
from andi_rewrite.models import build_model
from andi_rewrite.models.aer import load_pretrained_backbone
from andi_rewrite.utils import load_config

config = load_config("configs/train_aer.yaml")
model = build_model(config["model"], device="cpu")
payload = torch.load("path/to/old_checkpoint.pt", map_location="cpu", weights_only=True)
load_pretrained_backbone(model, payload["model"])
```

直接用舊 checkpoint resume AER 訓練會因缺少 AER 權重而報錯。上述 warm start 只匯入選定的模型權重，EMA、optimizer 與 epoch 由新的 Trainer 初始化。

## 內部運算與驗證範圍

- Input adapter 將 BCHW bottleneck 投影為 fine anchor。奇數高寬在右側與底部用 replicate padding 補成偶數，coarse 證據由 fine anchor 做 2×2 nonoverlap 平均取得。全域 tokens 由 pooled coarse 證據與可學 token embeddings 產生。
- 能量加總 anchor precision、跨尺度差、coarse 與 routed tokens 差、水平／垂直邊差。各項按影像加總所有空間與通道分量，每條無向邊算一次，包含 padding 格點。平均池化的 adjoint 是重複 2×2 再除以 4。保守步長上界為 `w_max + 1.25*lambda_m + lambda_s*(1 + ‖R‖₁‖R‖∞) + 2*lambda_e*max_weighted_degree`。
- 共享 proposer 用 gated convolution 與 coarse↔token cross-attention 產生位移。Guarded 逐張比較 FP32 或 FP64 能量，只有有限且 `E(candidate) <= E(reference)` 才採用候選。非有限的候選在 guarded 推論會回退，初始能量或 reference 失效時報錯。數值下降推導以固定有效係數與實數運算為前提，浮點近等值受捨入影響。
- Output adapter 以無 bias 的 1×1 projection 將三組狀態的差量映回外部通道，裁回原 bottleneck 尺寸，再乘上 `residual_scale` 加回原 bottleneck。AER 支援自己的奇數格點，ANDiUNet 原有固定 `image_size` 與 self-attention 解析度要求仍適用。
- 可獨立呼叫 `AERFeatureRefiner2D(C, config)(feature, return_trace=True)` 取得 `.feature`、`.trace` 與訓練期間可選的 `.proposal_loss`。Trace 提供各影像的能量、gradient L2、reference／candidate energy、採用情況、有效更新數與收斂旗標。獨立呼叫 `bypass=True` 直接回傳原 Tensor，完整模型可用 `steps: 0` 比較 identity。早停用 mask 凍結樣本，剩餘 active sample 仍讓算子處理整批。
- Smoke 與測試驗證可執行性及數值行為。這些結果未量測 MRI anomaly detection 效能、臨床品質或完整模型延遲。
