# GPU 伺服器訓練與回傳結果（給報告用）

## 1. 設定檔（已預設 GPU）

- `configs/transformer_caption.example.json`：`"device": "cuda"`、`num_workers` 預設 4（Linux + GPU 常見；若 worker 錯誤可改 0 或 2）。
- 資料路徑仍用 `configs/data_archive.example.json`（`archive/captions.txt`、`archive/Images`）。

若伺服器沒有 GPU，訓練腳本會自動退回 CPU 並在 stderr 提示。

## 2. 打包「程式碼」上伺服器（不含圖片，檔案較小）

在專案根目錄：

```bash
bash scripts/package_code_for_server.sh
```

會產生 `deeplearning-project_code_for_server.zip`（含 `src/`、`scripts/`、`configs/`、`metadata/splits`、`metadata/vocab`、`requirements.txt` 等）。

**請另外上傳整個 `archive/`**（或你在本機的 `data/`），內含：

- `archive/captions.txt`
- `archive/Images/*.jpg`

可用 `rsync`、`scp` 或另一個 zip，因為圖很多、zip 會很大。

## 3. 在伺服器上

```bash
unzip deeplearning-project_code_for_server.zip -d project && cd project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 若需要 CUDA 版 PyTorch，請依官方指示安裝對應的 torch / torchvision wheel

python scripts/train_transformer_caption.py --config configs/transformer_caption.example.json
```

訓練時會：

- 印出 **device / GPU 名稱**、train/val **batch 數**。
- 每個 epoch 用 **tqdm** 顯示 **train / val** 進度條與 **即時 batch loss**。
- 寫入 `artifacts/transformer_caption/training_log.jsonl`。
- 儲存 **checkpoints** 與 **sample_captions_val.json**。

## 4. 訓練完打包「結果」下載回來分析／寫報告

```bash
bash scripts/package_training_results.sh
```

會產生 `transformer_training_results.zip`，內含：

- `artifacts/transformer_caption/`（`checkpoint_*.pt`、`training_log.jsonl`、`sample_captions_val.json` 等）
- `docs/transformer_model_design.md`（設計說明）
- 相關 `configs/*.json`

建議另建一個純文字 `REPORT_NOTES.txt`（訓練中遇到的錯誤、改過的 batch size、實際 GPU 型號、epoch 時間），再手動加進 zip 或：

```bash
zip -r transformer_training_results.zip REPORT_NOTES.txt
```

把 zip 給協作者或再開啟分析即可對照 **Save checkpoints and sample captions**、**design decisions and issues**。

## 5. 報告可引用的檔案

| 項目 | 檔案 |
|------|------|
| Checkpoints | `artifacts/transformer_caption/checkpoint_epoch_*.pt`、`checkpoint_latest.pt` |
| 訓練曲線 / 數字 | `training_log.jsonl` |
| 範例句 | `sample_captions_val.json` |
| 設計與限制 | `docs/transformer_model_design.md` |

評估 BLEU/METEOR 需再跑 `scripts/generate_captions_transformer.py` 產生 predictions JSON，並用 `scripts/evaluate_captions.py`（與 references JSON）。

## 6. 只能「手動上傳」時（portal 有容量限制）

1. **先傳小檔**：只上傳 `bash scripts/package_code_for_server.sh` 產生的 **`deeplearning-project_code_for_server.zip`**。
2. **圖片分批**：把 `archive/Images` 拆成多個 zip（例如每 5000 張一包），或先傳 **`archive/captions.txt`**（很小），再分批傳圖；到伺服器後解壓到同一個 `archive/Images/` 目錄。
3. **若伺服器可下外網**：把大檔放在雲端硬碟，在伺服器用 **`wget` / `curl`** 下載，可避免瀏覽器一次上傳數十 GB。
4. 路徑維持與 **`configs/data_archive.example.json`** 一致：`archive/captions.txt`、`archive/Images/`。

### 上傳一整包 `archive.zip` 很慢時

- **正常現象**：三萬多張圖好幾 GB，家用上傳頻寬有限，可能要**數小時到隔夜**。
- **拆成小 zip 再上傳**（可中斷、分几天传）：在本機執行  
  `bash scripts/zip_images_in_chunks.sh archive/Images 4000`  
  會在 **`upload_chunks/`** 產生 `archive_Images_part01.zip`、`part02.zip`… 每包約 4000 張（可改數字）。先傳程式包，再每晚傳 1～2 包。
- **能 SSH 到伺服器時**：用 **`scp -r archive`** 或 **`rsync -avP archive/`** 通常比網頁上傳穩、可**斷點續傳**。
- **伺服器能連外網**：在伺服器直接 **`wget`/`curl`** 下載資料集壓縮檔（或 Kaggle API），**不必**從本機上傳圖檔。

英文報告文字（第 3 節）已寫在 **`docs/report_section3_transformer_EN.md`**，可直接貼到報告再補上你的實驗數字與圖表。
