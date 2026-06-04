# 使用自訓 DINOv3 Backbone + 自有資料集

這份教學說明如何用「你自己用 Meta DINOv3 SSL 訓練出來的 ViT-Base backbone」當作 FoundAD 的 encoder，並在「你自己的資料集」上訓練與推論。

官方流程（MVTec / VisA + 官方 DINOv3 權重）請看 [README.md](./README.md)；這份文件只講自訂流程。

---

## 1. 這次改了什麼

| 檔案 | 變更 |
|------|------|
| `foundad/src/foundad.py` | 新增兩種 encoder 類型：`dinov3_local`（用固定 torch.hub 架構 + 載入你的權重）與 `dinov3_cfg`（用你的 DINOv3 訓練 yaml 以 `build_model_from_cfg` 方式重建「完全相同」的架構再載入權重）。兩者都會自動拆解 teacher/student/backbone 前綴。 |
| `foundad/configs/app/train_custom.yaml` | 新增訓練設定範本，可指定 `weights_path` / `repo_dir` / `arch` / `config_file`。 |
| `foundad/src/train.py`、`AD.py` | 把 `weights_path` / `repo_dir` / `arch` / `config_file` 傳進模型；放寬只限 mvtec/visa 的檢查。 |
| `foundad/src/utils/synthesis.py` | 異常合成的前景遮罩：遇到非 MVTec/VisA 的類別名稱不再報錯，改用整張圖當前景。 |

> Encoder 全程**凍結**，只訓練輕量的 Manifold Projector，這點和論文一致。

---

## 2. Backbone 需求

- 必須是 **DINOv3 系列的 ViT**（這個 repo 透過 `torch.hub` 載入 DINOv3 的「架構程式碼」，再把你的權重灌進去）。
- `meta.arch` 要對應你訓練的尺寸：**`embed_dim=768` → `dinov3_vitb16`（ViT-B/16）**、`embed_dim=384` → `dinov3_vits16`（ViT-S/16）。
- **patch size 與輸入解析度**：patch 數量 = `(crop_size / patch_size)^2`。`dinov3_vitb16` + `crop_size=512` → 32×32 = 1024 patches。`crop_size` 要對齊你 SSL 訓練的 `IMG_SIZE`（預設 512）且能被 patch size 整除。
- 你的 checkpoint 可以是：
  - SSL 訓練檔 → 自動拆解。支援的外層 key（依優先序）：`teacher_backbone` → `student_backbone` → `teacher` → `student` → `model` → `state_dict` → `backbone`，並自動移除 `module.` / `backbone.` 等前綴、丟掉 DINO/iBOT head。
  - 或已整理過的純 backbone `state_dict`。
- 載入時會印出 `matched / missing / unexpected` 的 key 數量。**`matched` 為 0 會直接報錯**，代表 `arch` 跟 checkpoint 對不上（最常見是 ViT-S vs ViT-B 選錯）。

### 對應你自己的 `dinov3` repo（`autoresearch_ssl`）
你用 `autoresearch_ssl/train.py` 訓練後，checkpoint 存成 `autoresearch_ssl/runs/last_teacher.pth`，其結構為：

```python
{"teacher_backbone": <backbone state_dict>,   # ← 載入時優先使用這個（EMA teacher）
 "teacher_head": ..., "student_backbone": ..., "student_head": ...,
 "step": ..., "best_anomaly_score": ...}
```

`teacher_backbone` 已是乾淨的 backbone state_dict（key 形如 `patch_embed.proj.weight`、`blocks.N...`、`norm.weight`），loader 會自動取出，不需手動處理。

> 架構預設（`autoresearch_ssl/train.py`）是 `embed_dim=384, depth=12, heads=6, patch=16`，即 **ViT-S/16**。若你訓練「ViT-Base」是把 `EMBED_DIM=768`、`NUM_HEADS=12` 用 config 蓋過去，那 `arch` 就要設 `dinov3_vitb16`；否則維持 `dinov3_vits16`。不確定的話先看訓練 log 印出的 `Architecture: embed_dim=...`。

### 離線載入（建議）
直接指向本機這份 dinov3 repo，用 `source="local"` 載入架構，完全不需連網：

```
app.meta.repo_dir=/path/to/dinov3   # 例如你本機的 C:/Users/.../claude/dinov3
```

不填 `repo_dir` 則會嘗試從 GitHub 抓 DINOv3 架構碼（該 repo 為受限存取）。兩種方式都只用你的權重，不會下載官方權重。

### 兩種載入模式：`dinov3_local` vs `dinov3_cfg`

`meta.model` 可以選兩種方式把你的 backbone 接進來：

| | `dinov3_local` | `dinov3_cfg` |
|---|---|---|
| 架構來源 | 固定的 torch.hub entrypoint（`meta.arch`，例如 `dinov3_vitb16`） | 你的 DINOv3 訓練/eval yaml（`meta.config_file`），以 `build_model_from_cfg` 方式重建 |
| 必填欄位 | `arch` + `weights_path` | `config_file` + `weights_path` |
| 適用情境 | 你的 backbone 就是標準 ViT-S/B/L 尺寸 | 你在 SSL 時用 config 改過架構細節（patch、RoPE、layerscale、storage tokens…），要「完全一致」地重建 |
| `repo_dir` 用途 | torch.hub 的 `source="local"`（離線載架構碼） | 加進 `sys.path` 讓 `import dinov3` 解析到你本機的 clone（免安裝） |

`dinov3_cfg` 的流程（對應 `dinov3.models.build_model_for_eval` 的精神）：
1. 讀你的 yaml：`get_cfg_from_args`（不是 `setup_config`，避免分散式 assert 與寫檔副作用）。
2. 以 `build_model(cfg.student, only_teacher=True, img_size=...)` 建出 teacher backbone（架構與你訓練時完全相同）。
3. 載入 `weights_path`（一樣會自動拆 `teacher_backbone` 等容器與前綴），印出 `matched / missing / unexpected`。

> 想精準重現訓練架構、或 `dinov3_local` 出現 `unexpected/missing` 對不上的 key 時，建議用 `dinov3_cfg`。
> `config_file` 請指向你 SSL 訓練實際用的那份 yaml（裡頭的 `student.arch` / `student.patch_size` 等就是架構定義）。

---

## 3. 資料集結構

### 3.1 訓練資料（正常樣本）
訓練只需要**正常（無瑕疵）影像**。結構如下，`train/` 下至少要有一層子資料夾：

```
<data_path>/<data_name>/
└── train/
    └── normal/                # 子資料夾名稱隨意，可放多個類別
        ├── 000.png
        ├── 001.png
        └── ...
```

- `train_root = ${data.data_path}/${data.data_name}`。
- 子資料夾名稱（例如 `normal`）會被當成「類別名」傳給異常合成模組；自訂名稱沒問題，會自動 fallback 成整張圖前景。
- 少樣本（few-shot）就放 1 / 2 / 4 張即可。

> 也可以用內建的 few-shot 取樣腳本（若你的原始資料是 `類別/train/good` 之類的結構）：
> ```bash
> python foundad/src/sample.py source=/path/to/your_raw_dataset target=/path/to/your_fewshot seed=42 num_samples=1
> ```

### 3.2 推論資料
- **產生熱力圖（不需要標註）** → 用 `mode=demo`，`test_root` 指向任何一個放測試圖的資料夾即可（會遞迴找圖）。
- **量化評估（需要 pixel ground-truth）** → 需要 MVTec/VisA 風格的結構（`<class>/test/<defect>/` + `<class>/ground_truth/`）。自訂資料若沒有 mask，建議只用 demo。

---

## 4. 訓練

最小指令（在 `train_custom.yaml` 內填好路徑，或用 CLI 覆寫）：

```bash
python foundad/main.py \
  mode=train \
  app=train_custom \
  app.meta.weights_path=/path/to/dinov3/autoresearch_ssl/runs/last_teacher.pth \
  app.meta.arch=dinov3_vitb16 \
  app.meta.repo_dir=/path/to/dinov3 \
  data.dataset=mydata \
  data.data_name=mydata_1shot \
  data.data_path=/path/to/fewshot_root \
  optimization.epochs=2000 \
  diy_name=_pretrained
```

若想用 `dinov3_cfg`（以訓練 yaml 重建架構），把 `arch` 換成 `config_file`、`model` 換成 `dinov3_cfg`：

```bash
python foundad/main.py \
  mode=train \
  app=train_custom \
  app.meta.model=dinov3_cfg \
  app.meta.weights_path=/path/to/dinov3/autoresearch_ssl/runs/last_teacher.pth \
  app.meta.config_file=/path/to/dinov3/your_run_config.yaml \
  app.meta.repo_dir=/path/to/dinov3 \
  data.dataset=mydata \
  data.data_name=mydata_1shot \
  data.data_path=/path/to/fewshot_root \
  optimization.epochs=2000 \
  diy_name=_pretrained
```

重點參數：
- `app.meta.weights_path`：**你的 backbone 權重路徑（必填）**。
- `app.meta.model`：`dinov3_local`（用 `arch`）或 `dinov3_cfg`（用 `config_file`）。
- `app.meta.arch`：`dinov3_local` 用，backbone 架構，對應你訓練的尺寸。
- `app.meta.config_file`：`dinov3_cfg` 用，你的 DINOv3 訓練 yaml（精準重建架構）。
- `app.meta.repo_dir`：本機 DINOv3 repo 路徑（建議填，可離線）；不填則從 GitHub 抓架構碼。
- `data.dataset`：自訂名稱（非 mvtec/visa 即視為自訂，會略過 benchmark 檢查）。
- `data.data_name` / `data.data_path`：少樣本資料夾名稱與其所在路徑，`train_root` 由兩者組合。
- `diy_name`：模型輸出資料夾的後綴；**訓練與後續推論要用同一個值**。

checkpoint 會存在：`logs/<data_name>/<model>＜diy_name＞/train-step<N>.pth.tar`，
例如 `logs/mydata_1shot/dinov3_local_pretrained/train-step2000.pth.tar`。
同資料夾還會存一份 `params.yaml`（推論時會自動讀回 `meta`，包含你的 `weights_path`）。

---

## 5. 推論 / 產生熱力圖

訓練用的 `meta` 會從 `params.yaml` 自動載入，所以推論時只要對上資料夾名稱即可。
注意 `app.model_name` 要等於訓練時的 `app.meta.model`（`dinov3_local` 或 `dinov3_cfg`），這樣才找得到 log 資料夾。

```bash
python foundad/main.py \
  mode=demo \
  app=test \
  app.model_name=dinov3_local \
  data.dataset=mydata \
  data.data_name=mydata_1shot \
  data.test_root=/path/to/your/test_images \
  diy_name=_pretrained \
  testing.segmentation_vis=True
```

輸出的疊圖熱力圖會存在 `logs/.../demo/heatmaps/` 底下，保留原始資料夾相對結構。

### （選用）量化評估
若你的資料已整理成 MVTec/VisA 風格（含 `ground_truth` mask），可跑：

```bash
python foundad/main.py \
  mode=AD \
  app=test \
  app.model_name=dinov3_local \
  data.dataset=mvtec \
  data.data_name=mydata_1shot \
  data.test_root=/path/to/dataset \
  app.ckpt_step=2000 \
  diy_name=_pretrained
```

> `mode=AD` 的完整指標（I-AUROC / PRO 等）目前只支援 `data.dataset` 為 `mvtec` 或 `visa`。自訂資料若沒有 mask，請使用 `mode=demo`。

---

## 6. 疑難排解

- **`matched=0` 報錯**：`dinov3_local` 時是 `app.meta.arch` 跟 checkpoint 對不上；`dinov3_cfg` 時是 `app.meta.config_file` 的架構跟 checkpoint 對不上，或檔案不是 DINOv3 backbone。確認尺寸（vitb/vits/...）與 patch size。
- **`missing` keys 很多**：常見於 SSL 檔有額外 head；只要 backbone（blocks / patch_embed / norm）有對上即可，head 本來就會被丟掉。
- **`dinov3_cfg` 報 `Could not import the DINOv3 package`**：`app.meta.repo_dir` 要指向「包含 `dinov3` 套件資料夾」的那層 clone，會被加進 `sys.path`。
- **抓不到 DINOv3 架構碼 / 無網路**：clone DINOv3 到本機並設定 `app.meta.repo_dir`（`dinov3_local` 用 `source="local"`，`dinov3_cfg` 用來 `import dinov3`）。
- **patch 數量不符 / reshape 錯誤**：調整 `app.meta.crop_size` 讓它能被 patch size 整除。
- **推論找不到 ckpt**：確認 `app.model_name`、`data.data_name`、`diy_name` 三者與訓練時一致。
