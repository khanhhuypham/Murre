# Hướng Dẫn Sử Dụng MURRE (Reimplementation)

Tái tạo thực nghiệm của bài báo **MURRE: Multi-Hop Table Retrieval with Removal for
Open-Domain Text-to-SQL** (COLING 2025) — bao gồm phương pháp chính (MURRE), 2
baseline so sánh (Single-hop, CRUSH), và 3 biến thể ablation (w/o removal, w/o
tabulation, w/o early stop) dùng để tái tạo Table 2, 3 và 4 của paper.

## Mục Lục

1. [Cấu trúc project](#1-cấu-trúc-project)
2. [Cài đặt](#2-cài-đặt)
3. [Đặt data files](#3-đặt-data-files)
4. [Cấu hình config.yaml](#4-cấu-hình-configyaml)
5. [Option 1 — Offline/Debug (1 câu hỏi)](#5-option-1--offlinedebug-1-câu-hỏi)
6. [Option 2 — Batch (toàn dataset)](#6-option-2--batch-toàn-dataset)
7. [Tái tạo Table 2/3 — So sánh MURRE vs Single-hop vs CRUSH](#7-tái-tạo-table-23--so-sánh-murre-vs-single-hop-vs-crush)
8. [Tái tạo Table 4 — Ablation Study](#8-tái-tạo-table-4--ablation-study)
9. [Chạy API Service](#9-chạy-api-service)
10. [Cấu trúc outputs](#10-cấu-trúc-outputs)
11. [Chạy trong PyCharm](#11-chạy-trong-pycharm)
12. [Troubleshooting](#12-troubleshooting)
13. [Giới hạn đã biết](#13-giới-hạn-đã-biết)

---

## 1. Cấu trúc project

```
MURRE/                               ← Thư mục gốc (Working Directory khi chạy)
├── src/                             ← Toàn bộ code (Sources Root)
│   ├── config.py
│   ├── main.py
│   ├── api.py
│   ├── core/
│   │   ├── encoder.py               ← SGPTEncoder (Bi-Encoder, §3.2.1)
│   │   ├── llm.py                   ← LLMGenerator (OpenAI-compatible)
│   │   ├── rewriter.py              ← QueryRewriter (Removal + Tabulation, §3.4)
│   │   ├── pipeline.py              ← MURREPipeline (beam search, §3.2/§3.5)
│   │   └── factory.py               ← build_retriever(): chọn đúng method theo config
│   ├── methods/
│   │   ├── single_hop.py            ← Baseline Single-hop (§4.2)
│   │   └── crush.py                 ← Baseline CRUSH (§4.2, prompt tự suy luận — xem mục 13)
│   ├── steps/                       ← Batch pipeline (Option 2)
│   │   ├── embed.py                 ← Bước 1: mã hóa corpus
│   │   ├── retrieve.py              ← Bước 2: dense retrieval mỗi hop
│   │   ├── rewrite.py               ← Bước 3: Removal/Splice mỗi hop
│   │   ├── score.py                 ← Bước 4: Score_Path / Score_Table (§3.5)
│   │   ├── infer.py                 ← Bước 5: sinh SQL
│   │   └── run_baseline.py          ← Chạy Single-hop/CRUSH trên toàn dev set
│   ├── utils/
│   │   ├── schema.py                ← build_schema_corpus, pack_table, filter
│   │   ├── metric.py                ← recall@K, complete_recall@K
│   │   └── logger.py                ← Logger dùng chung (MURRE_LOG_LEVEL)
│   └── dataset/
│       └── loader.py                ← load_tables, load_dev, load_gold
│
├── dataset/                         ← DATA THẬT (khác với src/dataset/ — chỉ code)
│   ├── spider/{tables.json, dev.json, gold.txt}
│   └── bird/{tables.json, dev.json, gold.txt}
│
├── prompts/                         ← Few-shot prompt cho Rewrite/CRUSH
│   ├── spider_rewrite.txt / bird_rewrite.txt                       (Tabulation)
│   ├── spider_rewrite_no_tabulation.txt / bird_rewrite_no_tabulation.txt
│   └── spider_crush.txt / bird_crush.txt
│
├── outputs/                         ← Kết quả tự động tạo khi chạy
├── .env / .env.example
├── config.yaml
└── requirements.txt
```

**Lưu ý quan trọng:** `dataset/loader.py` (code) nằm trong `src/dataset/`, còn dữ
liệu thật (`tables.json`, `dev.json`, `gold.txt`) nằm ở `dataset/` **ngoài** `src/`
— hai thư mục trùng tên nhưng khác vị trí, khác vai trò. `config.yaml`/`.env`/
`prompts/`/`outputs/`/`dataset/` (data) đều nằm ở gốc project vì code đọc chúng
bằng đường dẫn tương đối, tính từ **Working Directory = thư mục gốc `MURRE/`**.

## 2. Cài đặt

```bash
cd MURRE
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env             # rồi điền OPENAI_API_KEY (hoặc dùng Ollama, xem mục 12)
```

Cho Python biết `src/` là nơi chứa module (chọn 1 trong 2 cách):

**Cách A — file `.pth` trong venv (khuyến nghị, không cần nhớ gõ lệnh mỗi lần):**
```bash
python -c "
import site, os
path = os.path.join(site.getsitepackages()[-1] if hasattr(site, 'getsitepackages') else site.ENABLE_USER_SITE, 'murre_src.pth')
with open(path, 'w', encoding='ascii') as f:
    f.write(os.path.abspath('src'))
print('Đã tạo:', path)
"
```
Nếu chạy trên Windows PowerShell và script trên không hoạt động, tạo file `.pth`
thủ công: xem mục 12 (Troubleshooting) — quan trọng là file phải lưu **không có
BOM**, dùng `[System.IO.File]::WriteAllText(...)` chứ không dùng `Out-File`.

**Cách B — set `PYTHONPATH` mỗi lần mở terminal mới:**
```bash
export PYTHONPATH=src        # Linux/Mac
$env:PYTHONPATH = "src"      # Windows PowerShell
```

## 3. Đặt data files

```bash
cp path/to/dev.json     dataset/spider/dev.json
cp path/to/tables.json  dataset/spider/tables.json
cp path/to/gold.txt     dataset/spider/gold.txt
# Tương tự cho dataset/bird/ nếu cần
```

**Định dạng bắt buộc** (không phải Spider/BIRD gốc, mà là bản "Union" đã tiền xử lý):
- `tables.json`: list các database, mỗi database có field `db_id` và `schema` (list
  chuỗi `"db.table(col1, col2, ...)"`).
- `dev.json`: list câu hỏi, mỗi câu có `utterance` (câu hỏi) và `rel_schema` (list
  schema đúng — dùng làm gold để tính recall).

## 4. Cấu hình config.yaml

Các tham số quan trọng nhất:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `general.dataset` | `spider` | `spider` hoặc `bird` |
| `pipeline.method` | `murre` | `murre` \| `single_hop` \| `crush` — chọn phương pháp retrieval |
| `pipeline.beam_size` | `5` | B trong paper |
| `pipeline.max_hop` | `3` | H trong paper |
| `pipeline.ablation.removal` | `true` | `false` = "w/o removal" (Table 4) |
| `pipeline.ablation.tabulation` | `true` | `false` = "w/o tabulation" (Table 4) |
| `pipeline.ablation.early_stop` | `true` | `false` = "w/o early stop" (Table 4) |
| `run_option.mode` | `offline` | `offline` (Option 1) hoặc `batch` (Option 2) |

3 cờ `ablation.*` **chỉ có tác dụng khi `pipeline.method: murre`** — khi
`method` là `single_hop`/`crush`, các cờ này bị bỏ qua.

## 4b. Dùng nhiều model LLM local (Ollama) — chuyển đổi nhanh

Nếu bạn thử nghiệm với nhiều model local khác nhau, `config.yaml` hỗ trợ khai báo
sẵn nhiều "profile" trong `llm.profiles`, mỗi profile là 1 model/endpoint riêng:

```yaml
llm:
  active_profile: qwen2.5-7b   # ← chỉ cần đổi dòng này để chuyển model
  profiles:
    qwen2.5-7b:
      model_name: qwen2.5:7b
      base_url: "http://localhost:11434/v1"
      api_key: ""
      temperature: 0.0
    qwen2.5-14b:
      model_name: qwen2.5:14b
      base_url: "http://localhost:11434/v1"
      api_key: ""
      temperature: 0.0
    # ... thêm profile mới bằng cách copy 1 khối và đổi tên/model_name
```

Xem nhanh các profile đã khai báo:
```bash
python -m main --list-llm-profiles
```

Đổi model cho **cả project** (batch, offline, API): sửa `llm.active_profile` trong
`config.yaml`.

Đổi model **tạm thời chỉ cho 1 lần chạy Option 1** (không cần sửa config.yaml):
```bash
python -m main --question "..." --llm-profile qwen2.5-14b
```

Đổi model trong code (ví dụ khi viết script riêng để so sánh nhiều model):
```python
from core.llm import LLMGenerator
llm_a = LLMGenerator(profile="qwen2.5-7b")
llm_b = LLMGenerator(profile="qwen2.5-14b")
```

Mỗi model mới pull về qua Ollama (`ollama pull <model>`), chỉ cần thêm 1 khối
profile mới trong `config.yaml` — không cần sửa code.

## 5. Option 1 — Offline/Debug (1 câu hỏi)

```yaml
run_option:
  mode: offline
```

```bash
python -m main --question "Which airlines have a flight to AHD?" --verbose
python -m main --show-config
```

`--verbose` in chi tiết từng hop (chỉ có ý nghĩa với `method: murre`).

## 6. Option 2 — Batch (toàn dataset)

```yaml
run_option:
  mode: batch
```

```bash
python -m main
```

Nếu `pipeline.method: murre`, pipeline tự chạy tuần tự: `embed → retrieve(hop0) →
[rewrite(hop) → retrieve(hop+1, mỗi beam)] × max_hop → score → infer`.

Nếu `pipeline.method` là `single_hop`/`crush`, pipeline chạy gọn hơn: `embed →
run_baseline` (không cần multi-hop).

Cũng có thể chạy từng bước riêng lẻ (khi method=murre):
```bash
python -m steps.embed
python -m steps.retrieve --hop 0
python -m steps.rewrite  --hop 0
python -m steps.retrieve --hop 1 --beam 0   # ... lặp cho beam 1..beam_size-1
python -m steps.rewrite  --hop 1
python -m steps.retrieve --hop 2 --beam 0   # ...
python -m steps.score
python -m steps.infer --top_k 5
```

## 7. Tái tạo Table 2/3 — So sánh MURRE vs Single-hop vs CRUSH

Đổi `pipeline.method` trong `config.yaml` rồi chạy lại `python -m main` (mode: batch)
cho từng giá trị:

```yaml
pipeline:
  method: murre        # rồi: python -m main
  method: single_hop   # rồi: python -m main
  method: crush         # rồi: python -m main
```

Mỗi lần chạy, kết quả `recall@k` / `complete_recall@k` được in ra và lưu tại
`outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/score.json` — vì đường dẫn
output có `{method}`, kết quả của 3 phương pháp không đè lên nhau.

## 8. Tái tạo Table 4 — Ablation Study

Giữ `pipeline.method: murre`, chỉ đổi từng cờ ablation (mỗi lần đổi 1 cờ, giữ 2 cờ
còn lại = true, đúng thiết kế ablation "chỉ tắt một cơ chế mỗi lần" của paper):

```yaml
pipeline:
  ablation:
    removal: false       # → "w/o removal" — python -m main (mode: batch)
    tabulation: true
    early_stop: true
```
```yaml
pipeline:
  ablation:
    removal: true
    tabulation: false     # → "w/o tabulation"
    early_stop: true
```
```yaml
pipeline:
  ablation:
    removal: true
    tabulation: true
    early_stop: false      # → "w/o early stop"
```

**Lưu ý:** đường dẫn output không phân biệt theo cờ ablation (chỉ phân biệt theo
`method`) — nếu muốn giữ kết quả của nhiều lần chạy ablation khác nhau, đổi tạm
`general.scale` (ví dụ `125m-no-removal`) trước khi chạy, hoặc backup thư mục
`outputs/` sau mỗi lần.

## 9. Chạy API Service

```bash
cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Xem docs tại `http://localhost:8000/docs`. Endpoint `/retrieve` dùng đúng
`pipeline.method` đang cấu hình trong `config.yaml`.

## 10. Cấu trúc outputs

```
outputs/
├── {dataset}_embeddings.pt                         ← Cache cho Option 1 / run_baseline
└── {dataset}/{scale}/{method}/
    ├── embeddings.json
    ├── turn0/dev.json                              ← Kết quả Hop-0 (mọi method)
    ├── rewrite/outputs/turn{hop}/dev.{beam}.json    ← Chỉ method=murre
    ├── turn{hop}/dev.{beam}.json                    ← Chỉ method=murre
    └── result/turn{max_hop}/
        ├── dev.json         ← Kết quả cuối cùng (ranked schemas)
        ├── score.json       ← Metrics recall@K / complete_recall@K
        ├── sql.{k}.txt      ← SQL dự đoán với top-k bảng
        └── inp.{k}.txt      ← Prompt đã dùng (debug)
```

## 11. Chạy trong PyCharm

1. Chuột phải vào thư mục `src/` → **Mark Directory as → Sources Root**.
2. Vào **Run → Edit Configurations** cho mỗi script, đặt **Working directory** =
   thư mục gốc `MURRE/` (nơi có `config.yaml`) — **không phải** `src/`.
3. Nếu chạy qua Terminal tích hợp của PyCharm mà vẫn gặp `ModuleNotFoundError:
   No module named 'config'`, dùng Cách A (file `.pth`) ở mục 2, hoặc chạy bằng
   `python -m main` / `python -m steps.embed` (có `-m`, không có đuôi `.py`) sau
   khi `set PYTHONPATH=src`.

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
→ `src/` chưa được thêm vào `sys.path`. Dùng file `.pth` (mục 2, Cách A) hoặc
`PYTHONPATH=src` + chạy bằng `python -m ...` (không chạy trực tiếp bằng đường dẫn
file như `python src/main.py`).

**Tạo file `.pth` bằng PowerShell `Out-File` nhưng vẫn không nhận `src`**
→ `Out-File -Encoding utf8` tự thêm ký tự BOM vô hình ở đầu file, khiến Python đọc
sai đường dẫn dòng đầu. Dùng lệnh sau để tạo file không BOM:
```powershell
[System.IO.File]::WriteAllText(
  "<đường-dẫn-venv>\Lib\site-packages\murre_src.pth",
  "<đường-dẫn-tuyệt-đối-tới-src>`n",
  [System.Text.Encoding]::ASCII
)
```

**`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`**
→ Xung đột OpenMP, thường do máy có cả Anaconda và venv riêng. Cách nhanh:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE      # Linux/Mac
$env:KMP_DUPLICATE_LIB_OK = "TRUE"    # Windows PowerShell
```
Cách lâu dài: mở Environment Variables của Windows, đưa các đường dẫn
`Anaconda3`, `Anaconda3\Scripts`, `Anaconda3\condabin` xuống **cuối** biến `PATH`
(không xóa — Anaconda vẫn dùng được bình thường, chỉ không được ưu tiên tìm trước).

**`FileNotFoundError: config.yaml`**
→ Working Directory đang không phải thư mục gốc `MURRE/`. Với PyCharm, sửa trong
Run Configuration; với terminal, `cd` về đúng thư mục gốc trước khi chạy.

**HuggingFace tải model chậm/bị chặn**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit')"
```

**Hết RAM khi encode** → giảm `encoder.batch_size` trong `config.yaml`.

## 13. Giới hạn đã biết

- **Prompt CRUSH** (`prompts/{dataset}_crush.txt`): paper không công khai prompt
  gốc của Kothyari et al. (2023) trong phần chính — prompt ở đây là suy luận hợp
  lý theo mô tả "converting the user question into a table format through
  hallucination", **không phải nguyên văn**. Cần thay bằng prompt gốc nếu muốn
  khớp chính xác số liệu Table 2/3.
- **Prompt Removal 9-shot/8-shot**: paper dùng prompt 9-shot cho SpiderUnion và
  8-shot cho BirdUnion (§4.1), nhưng chỉ in 2 ví dụ đầu do giới hạn trang (Table
  6/7). Prompt trong `prompts/` ở đây chỉ có 2-shot — đủ để chạy đúng logic,
  nhưng **chưa đủ số lượng ví dụ để tái tạo chính xác con số trong Table 2/3/4**.
  Bổ sung thêm ví dụ nếu cần khớp số liệu chính xác.
- **Union corpus**: code giả định `tables.json`/`dev.json` đã ở định dạng Union
  sẵn (xem mục 3) — không tự động gộp Spider/BIRD gốc thành Union corpus.
