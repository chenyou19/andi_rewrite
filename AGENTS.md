# 專案代理指引

本文件適用於整個 repository；子目錄若有 `AGENTS.md`，該目錄的工作同時遵循其指引。

## 執行環境

- 所有 Python 指令、測試、訓練與評估均使用既有的 Conda 環境 `ANDI`。
- 從 repository 根目錄執行指令。互動式 PowerShell 可先執行 `conda activate ANDI`。
- 自動化或未啟用環境的 shell，優先使用 `conda run -n ANDI --no-capture-output python ...`，避免誤用系統 Python、Conda base 或 repository 內其他虛擬環境。
- 執行前可用下列指令確認 interpreter：

  ```powershell
  conda run -n ANDI python -c "import sys; print(sys.executable)"
  ```

- 若 shell 找不到 `conda`，先查找本機既有 Conda 安裝並使用其完整路徑；若找不到 `ANDI`，回報環境缺失，不要默默切換環境或自行重建。
- 安裝依賴時也必須指定 `ANDI`；使用 pip 時採用 `conda run -n ANDI python -m pip ...`。避免無關的套件升級。

## 專案與修改原則

- 本專案是以 PyTorch 與 YAML 設定驅動的 MRI anomaly detection CLI 框架。
- 修改前先閱讀 `README.md`，並依工作範圍參考 `docs/architecture.md`、`docs/configuration.md`、`docs/development.md`。
- 保持既有模組分工、factory/registry 慣例與設定格式；修改行為或設定契約時同步更新相關文件。
- 保留使用者現有變更；避免與任務無關的重構。
- `outputs/`、checkpoints、LMDB、NPZ 與 NIfTI 為資料或執行產物，不應當作原始碼任意修改或提交。

## 驗證

依變更範圍執行相關測試；需要完整測試時使用：

```powershell
conda run -n ANDI --no-capture-output python -B -m pytest -q -p no:cacheprovider
```

亦可使用標準 unittest discovery：

```powershell
conda run -n ANDI --no-capture-output python -B -m unittest discover -s tests -v
```

CLI 入口檢查：

```powershell
conda run -n ANDI --no-capture-output python -B scripts/train.py --help
conda run -n ANDI --no-capture-output python -B scripts/eval.py --help
```

回報實際執行的驗證與結果；若因依賴、資料或硬體限制無法執行，明確說明，不宣稱測試通過。

## 訓練與資料操作

- 完整訓練與大量評估依使用者任務範圍執行，不作為一般修改的預設驗證。
- 評估前確認資料路徑、split CSV 與 `model.checkpoint` 有效且 checkpoint 非空，避免誤評估隨機初始化模型。
- 保留既有 modality 順序 `[flair, t1, t1ce, t2]` 及 dataset shape/geometry 契約。
- 執行資料原地覆寫、LMDB `--overwrite` 或 cache 清理前，確認目標路徑與任務授權範圍。
