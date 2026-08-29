# Hướng Dẫn Sử Dụng MURRE (Reimplementation)

Tái tạo thực nghiệm của bài báo **MURRE: Multi-Hop Table Retrieval with Removal for
Open-Domain Text-to-SQL** (COLING 2025) — bao gồm phương pháp chính (MURRE), 2
baseline so sánh (Single-hop, CRUSH), và 3 biến thể ablation (w/o removal, w/o
tabulation, w/o early stop) dùng để tái tạo Table 2, 3 và 4 của paper.

## Mục Lục

1. [Cấu trúc project](#1-cấu-trúc-project)
2. [Đặt data files](#2-đặt-data-files)
3. [Cấu hình project](#3-cấu-hình-project)
    - [3b. Dùng nhiều model LLM local (Ollama)](#3b-dùng-nhiều-model-llm-local-ollama--chuyển-đổi-nhanh)
4. [Option 1 — Offline/Debug (1 câu hỏi)](#4-option-1--offlinedebug-1-câu-hỏi)
    - [4b. Test riêng một method](#4b-test-riêng-một-method-python--m-methods)
5. [Option 2 — Batch (toàn dataset)](#5-option-2--batch-toàn-dataset)
    - [5b. Chạy từng bước riêng lẻ](#5b-chạy-từng-bước-riêng-lẻ-chỉ-khi-method-murre)
6. [Tái tạo Table 2/3 — So sánh MURRE vs Single-hop vs CRUSH](#6-tái-tạo-table-23--so-sánh-murre-vs-single-hop-vs-crush)
    - [6b. Chạy nhanh 2 baseline không cần đổi method](#6b-chạy-nhanh-2-baseline-mà-không-cần-đổi-pipelinemethod)
7. [Tái tạo Table 4 — Ablation Study](#7-tái-tạo-table-4--ablation-study)
8. [Chạy API Service](#8-chạy-api-service)
    - [8b. GET /evaluate — metric thật của lần chạy](#8b-get-evaluate--tính-metric-thật-của-lần-chạy-trên-máy-này)
9. [Cấu trúc outputs](#9-cấu-trúc-outputs)
10. [Chạy trong PyCharm](#10-chạy-trong-pycharm)
11. [Troubleshooting](#11-troubleshooting)
12. [Giới hạn đã biết](#12-giới-hạn-đã-biết)

---

## 1. Cấu trúc project

```
MURRE/                               ← Thư mục gốc (Working Directory khi chạy)
├── src/                             ← Toàn bộ code (Sources Root)
│   ├── config.py                    ← Settings + MẶC ĐỊNH của general/pipeline/paths/
│   │                                  logging/api/run_option (xem mục 3)
│   ├── main.py
│   ├── api.py
│   ├── core/                        ← Hạ tầng dùng chung cho MỌI method
│   │   ├── encoder.py               ← SGPTEncoder (Bi-Encoder, §3.2.1)
│   │   ├── llm.py                   ← LLMGenerator (OpenAI-compatible)
│   │   ├── rewriter.py              ← QueryRewriter (Removal + Tabulation, §3.4)
│   │   ├── corpus.py                ← prepare(): corpus + embeddings (có cache)
│   │   └── factory.py               ← build_retriever(): chọn đúng method theo config
│   ├── methods/                     ← 3 method của Table 2, cùng interface run()
│   │   ├── murre.py                 ← MURREPipeline — method chính (beam search, §3.2/§3.5)
│   │   ├── single_hop.py            ← Baseline Single-hop (§4.2)
│   │   └── crush.py                 ← Baseline CRUSH (§4.2, prompt tự suy luận — xem mục 12)
│   ├── steps/                       ← Batch pipeline (Option 2)
│   │   ├── embed.py                 ← Bước 1: mã hóa corpus
│   │   ├── retrieve.py              ← Bước 2: dense retrieval mỗi hop
│   │   ├── rewrite.py               ← Bước 3: Removal/Splice mỗi hop
│   │   ├── score.py                 ← Bước 4: Score_Path / Score_Table (§3.5)
│   │   ├── infer.py                 ← Bước 5: sinh SQL
│   │   └── _common.py               ← find_json_files() dùng chung cho rewrite/score
│   ├── models/                      ← Đối tượng NỘI BỘ dùng chung giữa các module
│   │   ├── retrieval.py             ← RetrievedTable (method trả về), RetrievedRow (dòng JSON)
│   │   └── records.py               ← TurnRecord / RewriteRecord / ResultRecord
│   ├── schemas/                     ← Pydantic model cho request/response của API
│   │   ├── retrieve.py              ← RetrieveRequest, TableResult
│   │   ├── evaluate.py              ← EvalResult, AvailableRun
│   │   └── health.py                ← HealthStatus
│   ├── utils/
│   │   ├── schema.py                ← build_schema_corpus, pack_table, filter
│   │   ├── metrics.py               ← recall@K, complete_recall@K
│   │   ├── scoring.py               ← Norm(s) và Score_Path (§3.5, Appendix C)
│   │   ├── display.py               ← print_results(): in bảng kết quả ra terminal
│   │   └── logger.py                ← Logger dùng chung
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
├── config.yaml                      ← CHỈ còn 2 section: encoder + llm (xem mục 3)
└── requirements.txt
```

**Lưu ý quan trọng:** `dataset/loader.py` (code) nằm trong `src/dataset/`, còn dữ
liệu thật (`tables.json`, `dev.json`, `gold.txt`) nằm ở `dataset/` **ngoài** `src/`
— hai thư mục trùng tên nhưng khác vị trí, khác vai trò. `config.yaml`/`.env`/
`prompts/`/`outputs/`/`dataset/` (data) đều nằm ở gốc project vì code đọc chúng
bằng đường dẫn tương đối, tính từ **Working Directory = thư mục gốc `MURRE/`**.

## 2. Đặt data files

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

## 3. Cấu hình project

Cấu hình nằm ở **hai chỗ**, sửa đúng chỗ thì mới có tác dụng:

| Section | Sửa ở đâu |
|---|---|
| `encoder`, `llm` | `config.yaml` (gốc project) |
| `general`, `pipeline`, `paths`, `logging`, `api`, `run_option` | `src/config.py` — giá trị mặc định của các class `*Config` |

`config.yaml` chỉ còn 2 section `encoder` và `llm`. Các section còn lại đã chuyển
thành **giá trị mặc định ngay trong `src/config.py`** (`GeneralConfig`,
`PipelineConfig`, `PathsConfig`, `LoggingConfig`, `ApiConfig`, `RunOptionConfig`) —
đổi tham số nghĩa là sửa giá trị mặc định của field tương ứng trong class đó.

> **Vẫn dùng lại được YAML bất cứ lúc nào.** `Settings` đọc `config.yaml` rồi ghi đè
> lên mặc định trong code, nên thêm lại section vào `config.yaml` (ví dụ dán lại cả
> khối `pipeline:`) là section đó thắng, không cần sửa `src/config.py`. Cách này tiện
> khi muốn giữ nhiều cấu hình song song qua biến môi trường `MURRE_CONFIG_PATH`.

Các tham số quan trọng nhất:

| Tham số | Mặc định | Sửa ở | Ý nghĩa |
|---|---|---|---|
| `general.dataset` | `spider` | `GeneralConfig` (config.py) | `spider` hoặc `bird` |
| `general.scale` | `125m` | `GeneralConfig` (config.py) | `125m` \| `1.3b` \| `2.7b` \| `5.8b` |
| `pipeline.method` | `murre` | `PipelineConfig` (config.py) | `murre` \| `single_hop` \| `crush` — chọn phương pháp retrieval |
| `pipeline.beam_size` | `5` | `PipelineConfig` (config.py) | B trong paper |
| `pipeline.max_hop` | `3` | `PipelineConfig` (config.py) | H trong paper |
| `pipeline.ablation.removal` | `True` | `AblationConfig` (config.py) | `False` = "w/o removal" (Table 4) |
| `pipeline.ablation.tabulation` | `True` | `AblationConfig` (config.py) | `False` = "w/o tabulation" (Table 4) |
| `pipeline.ablation.early_stop` | `True` | `AblationConfig` (config.py) | `False` = "w/o early stop" (Table 4) |
| `run_option.mode` | `offline` | `RunOptionConfig` (config.py) | `offline` (Option 1) hoặc `batch` (Option 2) |
| `encoder.model_name` | SGPT-125M | `config.yaml` | PHẢI khớp `general.scale` |
| `encoder.batch_size` | `256` | `config.yaml` | Giảm nếu hết RAM |
| `llm.active_profile` | `qwen2.5-0.5b` | `config.yaml` | Xem mục 3b |

3 cờ `ablation.*` **chỉ có tác dụng khi `pipeline.method = "murre"`** — khi
`method` là `single_hop`/`crush`, các cờ này bị bỏ qua.

Xem cấu hình đang thực sự hiệu lực (sau khi trộn YAML + mặc định trong code + `.env`):

```bash
python -m config           # in toàn bộ config dạng JSON
python -m config --paths   # in mọi đường dẫn đã resolve theo cfg hiện tại
python -m config --llm     # liệt kê profile LLM trong config.yaml
```

> Mọi lệnh **xem** cấu hình đều nằm ở `config.py`. `main.py` chỉ để **chạy** pipeline.

## 3b. Dùng nhiều model LLM local (Ollama) — chuyển đổi nhanh

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
python -m config --llm
```

Đổi model cho **cả project** (batch, offline, API): sửa `llm.active_profile` trong
`config.yaml`.

Đổi model **tạm thời chỉ cho 1 lần chạy Option 1** (không cần sửa config.yaml): đặt
`LLM_PROFILE` trong khối `__main__` của `src/main.py`:

```python
LLM_PROFILE: str | None = "qwen2.5-14b"
```

Đổi model trong code (ví dụ khi viết script riêng để so sánh nhiều model):
```python
from core.llm import LLMGenerator
llm_a = LLMGenerator(profile="qwen2.5-7b")
llm_b = LLMGenerator(profile="qwen2.5-14b")
```

Mỗi model mới pull về qua Ollama (`ollama pull <model>`), chỉ cần thêm 1 khối
profile mới trong `config.yaml` — không cần sửa code.

## 4. Option 1 — Offline/Debug (1 câu hỏi)

Trong `src/config.py`, class `RunOptionConfig`:

```python
class RunOptionConfig(BaseModel):
    mode: str = "offline"      # ← Option 1
```

`src/main.py` **không nhận tham số dòng lệnh** — sửa thẳng mấy biến HOA trong khối
`__main__` ở cuối file (cùng kiểu với `methods/murre.py`), rồi:

```bash
python -m main
```

```python
# src/main.py — khối __main__
QUESTION: str | None = "Which airlines have a flight to AHD?"
VERBOSE: bool = True        # in chi tiết từng hop (chỉ có ý nghĩa với method murre)
TOP_N: int = 5              # số bảng in ra
```

`QUESTION` khác `None` thì tự chạy Option 1, kể cả khi `run_option.mode` là `batch`.
Để `QUESTION = None` thì lấy câu đầu tiên trong `dev.json`.

### 4b. Test riêng một method (`python -m methods.*`)

Mỗi file trong `src/methods/` có entry point độc lập, chạy đúng **một** method trên
**một** câu hỏi rồi in top-N bảng kèm đối chiếu gold. Dùng khi muốn kiểm tra nhanh
một method mà không phải chạy cả dataset, và khi muốn debug đúng file đó.

```bash
python -m methods.single_hop        # không cần LLM
python -m methods.crush             # cần Ollama đang chạy
python -m methods.murre             # cần Ollama đang chạy
```

Bỏ trống `--question` thì lấy câu đầu tiên trong `dev.json`. Nếu câu hỏi có trong
`dev.json`, output tự đối chiếu với `rel_schema` và đánh dấu `✓`:

```
==============================================================================
  METHOD  : Single-hop
  DATASET : spider | ENCODER SCALE: 125m
  QUESTION: Show name, country, age for all singers ordered by age ...
==============================================================================
Gold (1 bảng):
  - concert_singer.singer(singer id, name, country, song name, ...)
Top-5 bảng retrieve được:
     1. 0.4472  singer.song(song id, title, singer id, sales, highest position)
  ✓  2. 0.4374  concert_singer.singer(singer id, name, country, song name, ...)
     ...
  → recall@5 = 1/1 = 100.00%
==============================================================================
```

**Tham số chung cả 3 method:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--question` | câu đầu `dev.json` | Câu hỏi cần test |
| `--dataset` | `general.dataset` | `spider` \| `bird` |
| `--top_n` | `5` | Số bảng in ra |

**Tham số riêng:**

| Method | Tham số riêng | Ý nghĩa |
|---|---|---|
| `methods.crush` | `--llm-profile` | Đổi model LLM cho lần chạy này |
| | `--no-collective` | Ablation: gộp mọi bảng đoán thành 1 query |
| `methods.murre` | `--llm-profile` | Đổi model LLM cho lần chạy này |
| | `--verbose` | In chi tiết từng hop (Removal, retrieve, early stop) |
| | `--beam_size` | Ghi đè `pipeline.beam_size` |
| | `--max_hop` | Ghi đè `pipeline.max_hop` |

`--beam_size` / `--max_hop` rất hữu ích khi debug MURRE: mặc định beam 5 × 3 hop là
15 lần gọi LLM cho **một** câu hỏi, giảm xuống `--beam_size 2 --max_hop 1` chỉ còn 2
lần.

```bash
# CRUSH: xem LLM hallucinate ra schema gì rồi mới retrieve
python -m methods.crush --llm-profile qwen2.5-7b --top_n 3

# So collective vs gộp-1-query trên cùng câu hỏi
python -m methods.crush --question "..." 
python -m methods.crush --question "..." --no-collective

# MURRE debug nhanh, in từng hop
python -m methods.murre --verbose --beam_size 2 --max_hop 1
```

`methods.crush` in thêm danh sách bảng mà LLM đoán ra trước khi retrieve — chỗ hay
sai nhất của CRUSH. `methods.murre` in cấu hình đang dùng ở dòng đầu để không nhầm
lẫn giữa các lần chạy ablation:

```
  Cấu hình: beam_size=2, max_hop=1, removal=True, tabulation=True, early_stop=True
```

## 5. Option 2 — Batch (toàn dataset)

Trong `src/config.py`, class `RunOptionConfig`:

```python
class RunOptionConfig(BaseModel):
    mode: str = "batch"        # ← Option 2
```

```bash
python -m main
```

`main.py` chọn đường chạy theo `pipeline.method`:

| `pipeline.method` | Đường chạy | Các bước |
|---|---|---|
| `murre` | chuỗi `steps/*.py` | `embed → retrieve(hop0) → [rewrite(hop) → retrieve(hop+1, mỗi beam)] × max_hop → score → infer` |
| `single_hop` / `crush` | `core/runner.py` | gọi thẳng `methods/*.run()` cho từng câu rồi ghi `result` + `score` |

2 baseline không có multi-hop nên không dùng được chuỗi `steps/`; `core/runner.py`
chạy chúng trên cả `dev.json` bằng một đường code riêng. Đường này **không sinh SQL**
(không có bước `infer`) — chỉ MURRE chạy đủ 5 bước.

Bước `embed` tự bỏ qua nếu cache `.pt` đã có; đặt `FORCE_EMBED = True` để ép
mã hóa lại (cần khi đổi dataset hoặc đổi encoder).

Chạy thử nhanh trên vài câu đầu (chỉ `single_hop`/`crush`):

```python
# src/main.py — khối __main__
LIMIT: int | None = 8
```

### 5b. Chạy từng bước riêng lẻ (chỉ khi `method: murre`)

Chạy tay từng bước hữu ích khi debug: xem được output trung gian của mỗi hop và
chạy lại đúng một bước bị lỗi mà không phải làm lại từ đầu.

**Thứ tự phụ thuộc.** `retrieve` và `rewrite` xen kẽ nhau, mỗi bước đọc output của
bước trước. Sơ đồ dưới đây viết đầy đủ **không rút gọn**, đúng theo cấu hình mặc định
của paper (`beam_size: 5` → beam `0..4`; `max_hop: 3` → hop `0..3`). Mọi đường dẫn
đều tính từ `outputs/{dataset}/{scale}/{method}/`, trừ cache embeddings nằm ở
`outputs/{dataset}_{scale}_embeddings.pt` (dùng chung cho cả 3 method):

```
embed                                           → {dataset}_{scale}_embeddings.pt
└─ retrieve --hop 0                             → turn0/dev.json
   └─ rewrite  --hop 0                          → rewrite/outputs/turn0/dev.0.json
      │                                           rewrite/outputs/turn0/dev.1.json
      │                                           rewrite/outputs/turn0/dev.2.json
      │                                           rewrite/outputs/turn0/dev.3.json
      │                                           rewrite/outputs/turn0/dev.4.json
      ├─ retrieve --hop 1 --beam 0              → turn1/dev.0.json
      ├─ retrieve --hop 1 --beam 1              → turn1/dev.1.json
      ├─ retrieve --hop 1 --beam 2              → turn1/dev.2.json
      ├─ retrieve --hop 1 --beam 3              → turn1/dev.3.json
      └─ retrieve --hop 1 --beam 4              → turn1/dev.4.json
         └─ rewrite  --hop 1                    → rewrite/outputs/turn1/dev.0.json
            │                                     rewrite/outputs/turn1/dev.1.json
            │                                     rewrite/outputs/turn1/dev.2.json
            │                                     rewrite/outputs/turn1/dev.3.json
            │                                     rewrite/outputs/turn1/dev.4.json
            ├─ retrieve --hop 2 --beam 0        → turn2/dev.0.json
            ├─ retrieve --hop 2 --beam 1        → turn2/dev.1.json
            ├─ retrieve --hop 2 --beam 2        → turn2/dev.2.json
            ├─ retrieve --hop 2 --beam 3        → turn2/dev.3.json
            └─ retrieve --hop 2 --beam 4        → turn2/dev.4.json
               └─ rewrite  --hop 2              → rewrite/outputs/turn2/dev.0.json
                  │                               rewrite/outputs/turn2/dev.1.json
                  │                               rewrite/outputs/turn2/dev.2.json
                  │                               rewrite/outputs/turn2/dev.3.json
                  │                               rewrite/outputs/turn2/dev.4.json
                  ├─ retrieve --hop 3 --beam 0  → turn3/dev.0.json
                  ├─ retrieve --hop 3 --beam 1  → turn3/dev.1.json
                  ├─ retrieve --hop 3 --beam 2  → turn3/dev.2.json
                  ├─ retrieve --hop 3 --beam 3  → turn3/dev.3.json
                  └─ retrieve --hop 3 --beam 4  → turn3/dev.4.json
                     └─ score                   → result/turn3/dev.json
                        │                         result/turn3/score.json
                        └─ infer --top_k 5      → result/turn3/sql.5.txt
                                                  result/turn3/inp.5.txt
```

Ba quy tắc đọc ra từ sơ đồ trên:

- **`retrieve` chạy 5 lần mỗi hop, `rewrite` chỉ chạy 1 lần.** Một lệnh `rewrite
  --hop N` đọc *toàn bộ* 5 file trong `turn{N}/` rồi tự sinh ra cả 5 file beam mới —
  không có tham số `--beam` cho `rewrite`.
- **Thiếu một beam sẽ không báo lỗi.** Nếu quên `retrieve --hop 2 --beam 3`, thì
  `rewrite --hop 2` vẫn chạy bình thường nhưng chỉ thấy 4 đường đi, và `score` tổng
  hợp trên dữ liệu thiếu → số liệu sai mà không có cảnh báo nào. Đếm số file trong
  `turn{N}/` phải đúng bằng `beam_size` trước khi qua bước sau.
- **Hop cuối không cần rewrite.** `rewrite` chạy ở hop `0..max_hop-1` (0, 1, 2), còn
  `retrieve` chạy ở hop `0..max_hop` (0, 1, 2, 3). `rewrite --hop 3` là vô nghĩa vì
  không còn hop 4 nào dùng output của nó.

Nếu bạn đổi `beam_size` hoặc `max_hop` (class `PipelineConfig` trong
`src/config.py`), số nhánh trong sơ đồ đổi
theo: mỗi hop có đúng `beam_size` lệnh `retrieve`, và có `max_hop` lệnh `rewrite`
(hop `0` tới `max_hop-1`).

**Chuỗi lệnh đầy đủ** với cấu hình mặc định của paper (`beam_size: 5` → beam
`0..4`, `max_hop: 3` → hop `0..3`) — tổng cộng 22 lệnh, chạy đúng theo thứ tự này:

```bash
# ── Bước 1: mã hóa corpus (chỉ cần chạy lại khi đổi dataset hoặc encoder) ──
python -m steps.embed

# ── Hop 0: retrieve trên TOÀN BỘ corpus bằng câu hỏi gốc ──
python -m steps.retrieve --hop 0
python -m steps.rewrite  --hop 0

# ── Hop 1: retrieve trong pool đã hẹp, cho từng beam ──
python -m steps.retrieve --hop 1 --beam 0
python -m steps.retrieve --hop 1 --beam 1
python -m steps.retrieve --hop 1 --beam 2
python -m steps.retrieve --hop 1 --beam 3
python -m steps.retrieve --hop 1 --beam 4
python -m steps.rewrite  --hop 1

# ── Hop 2 ──
python -m steps.retrieve --hop 2 --beam 0
python -m steps.retrieve --hop 2 --beam 1
python -m steps.retrieve --hop 2 --beam 2
python -m steps.retrieve --hop 2 --beam 3
python -m steps.retrieve --hop 2 --beam 4
python -m steps.rewrite  --hop 2

# ── Hop 3 = max_hop: chỉ retrieve, KHÔNG rewrite nữa ──
python -m steps.retrieve --hop 3 --beam 0
python -m steps.retrieve --hop 3 --beam 1
python -m steps.retrieve --hop 3 --beam 2
python -m steps.retrieve --hop 3 --beam 3
python -m steps.retrieve --hop 3 --beam 4

# ── Tổng hợp Score_Path / Score_Table (§3.5) rồi sinh SQL ──
python -m steps.score
python -m steps.infer --top_k 5
```

**Rút gọn bằng vòng lặp.** Nếu đổi `beam_size`/`max_hop`, dùng vòng lặp thay vì sửa
tay danh sách trên.

PowerShell (Windows):
```powershell
$BEAM = 5; $MAXHOP = 3
python -m steps.embed
python -m steps.retrieve --hop 0
foreach ($hop in 1..$MAXHOP) {
    python -m steps.rewrite --hop ($hop - 1)
    foreach ($b in 0..($BEAM - 1)) { python -m steps.retrieve --hop $hop --beam $b }
}
python -m steps.score
python -m steps.infer --top_k 5
```

Bash (Linux/Mac/Git Bash):
```bash
BEAM=5; MAXHOP=3
python -m steps.embed
python -m steps.retrieve --hop 0
for hop in $(seq 1 $MAXHOP); do
    python -m steps.rewrite --hop $((hop - 1))
    for b in $(seq 0 $((BEAM - 1))); do
        python -m steps.retrieve --hop "$hop" --beam "$b"
    done
done
python -m steps.score
python -m steps.infer --top_k 5
```

**Tham số của từng bước:**

| Lệnh | Tham số | Đọc từ | Ghi ra |
|---|---|---|---|
| `steps.embed` | *(không có)* | `dataset/{ds}/tables.json` | `{dataset}_{scale}_embeddings.pt` |
| `steps.retrieve` | `--hop N` (bắt buộc), `--beam B` (bắt buộc khi N≥1) | hop 0: `dev.json`<br>hop N: `rewrite/outputs/turn{N-1}/dev.{B}.json` | hop 0: `turn0/dev.json`<br>hop N: `turn{N}/dev.{B}.json` |
| `steps.rewrite` | `--hop N` (bắt buộc) | `turn{N}/` (mọi beam) | `rewrite/outputs/turn{N}/dev.{0..B-1}.json` |
| `steps.score` | *(không có)* | `turn0/` + `turn{1..max_hop}/` | `result/turn{max_hop}/{dev,score}.json` |
| `steps.infer` | `--top_k K` (mặc định 5) | `result/turn{max_hop}/dev.json` | `result/turn{max_hop}/sql.{K}.txt` |

Lưu ý về `--beam`: argparse **không** bắt buộc tham số này (`--hop` thì có). Chạy
`python -m steps.retrieve --hop 1` mà quên `--beam` sẽ không báo lỗi tham số, mà đi
tìm file `dev.None.json` rồi mới chết với `FileNotFoundError` — thấy `None` trong
tên file nghĩa là thiếu `--beam`. Ở `--hop 0` thì `--beam` bị bỏ qua (hop 0 chưa có
beam nào).

**Chi phí LLM.** Mỗi `rewrite --hop N` gọi LLM `beam_size × số câu hỏi` lần — với
Spider dev (658 câu), beam 5 là 3.290 lần gọi mỗi hop, 3 hop là ~9.870 lần. Chạy
model local qua Ollama thì miễn phí nhưng mất nhiều giờ; nếu chỉ muốn kiểm tra
pipeline chạy đúng, hãy giảm tạm `beam_size = 2` và `max_hop = 1` trong
`PipelineConfig` (`src/config.py`) trước.

## 6. Tái tạo Table 2/3 — So sánh MURRE vs Single-hop vs CRUSH

Đổi `method` trong class `PipelineConfig` (`src/config.py`) rồi chạy lại
`python -m main` (mode: batch) cho từng giá trị:

```python
class PipelineConfig(BaseModel):
    method: str = "murre"        # rồi: python -m main
    # method: str = "single_hop" # rồi: python -m main
    # method: str = "crush"      # rồi: python -m main
```

Mỗi lần chạy, kết quả `recall@k` / `complete_recall@k` được in ra và lưu tại
`outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/score.json` — vì đường dẫn
output có `{method}`, kết quả của 3 phương pháp không đè lên nhau.

### 6b. Chạy nhanh 2 baseline mà không cần đổi `pipeline.method`

Mỗi baseline có khối `__main__` riêng, test được **một** câu hỏi mà không phải đổi
`pipeline.method` trong `src/config.py`:

```bash
python src/methods/single_hop.py
python src/methods/crush.py
```

`single_hop.py` nhận tham số dòng lệnh (`--question`, `--dataset`, `--top_n`).
`crush.py` thì sửa thẳng mấy biến ở đầu khối `__main__`:

| Biến trong `crush.py` | Ý nghĩa |
|---|---|
| `QUESTION` | Câu hỏi cần test; `None` → lấy câu đầu tiên trong `dev.json` |
| `DATASET` | Ghi đè `general.dataset`; `None` → theo `src/config.py` |
| `TOP_N` | Số bảng in ra |
| `LLM_PROFILE` | Ghi đè `llm.active_profile`; `None` → theo `config.yaml` |
| `COLLECTIVE` | `False` = tắt collective retrieval (xem giải thích bên dưới) |

> **Muốn số liệu Table 2/3 cho 2 baseline** thì không dùng khối `__main__` (chỉ 1 câu)
> mà chạy batch: đặt `pipeline.method` thành `single_hop`/`crush` rồi `python -m main`
> với `run_option.mode = "batch"`. `core/runner.py` chạy cả `dev.json` và ghi ra
> `result/turn{max_hop}/{dev,score}.json` — đúng file mà `/evaluate` đọc (mục 5).

**Collective retrieval trong CRUSH.** LLM thường hallucinate ra *nhiều* bảng, mỗi bảng
một dòng. Mặc định (`collective=True`) mỗi dòng được encode thành **một query riêng**,
rồi mỗi bảng trong corpus lấy điểm similarity **cao nhất** trên tất cả các query — đúng
cơ chế của CRUSH gốc, và cùng quy ước max-aggregation mà
`steps/retrieve.py::retrieve_for_queries()` dùng cho multi-subquery của MURRE, nên hai
phương pháp so sánh được công bằng. Với `COLLECTIVE = False`, toàn bộ output của LLM bị
gộp thành **một** chuỗi rồi encode một lần — embedding bị trung bình hóa giữa các bảng
khác nhau. Cờ này để chạy ablation, không phải cấu hình mặc định nên dùng.

`CrushRetriever.last_hallucinated` giữ nguyên văn output của LLM cho câu hỏi vừa chạy;
khối `__main__` của `crush.py` in ra luôn — dùng để xem LLM đoán sai schema ở đâu.

**Chất lượng LLM ảnh hưởng rất mạnh tới CRUSH.** Model quá nhỏ (ví dụ `qwen2.5:0.5b`)
hay **lặp lại nguyên văn ví dụ few-shot** trong prompt thay vì suy luận theo câu hỏi
— ví dụ trả về `company.employee(employee_id, name, bonus, department)` cho câu hỏi về
ca sĩ Pháp. Dùng model đủ lớn (`qwen2.5:7b` trở lên) khi lấy số liệu báo cáo; paper
gốc dùng `gpt-3.5-turbo` (§4.1).

## 7. Tái tạo Table 4 — Ablation Study

Giữ `pipeline.method = "murre"`, chỉ đổi từng cờ trong class `AblationConfig`
(`src/config.py`) — mỗi lần đổi 1 cờ, giữ 2 cờ còn lại = `True`, đúng thiết kế
ablation "chỉ tắt một cơ chế mỗi lần" của paper:

```python
class AblationConfig(BaseModel):
    removal: bool = False      # → "w/o removal" — python -m main (mode: batch)
    tabulation: bool = True
    early_stop: bool = True
```
```python
class AblationConfig(BaseModel):
    removal: bool = True
    tabulation: bool = False   # → "w/o tabulation"
    early_stop: bool = True
```
```python
class AblationConfig(BaseModel):
    removal: bool = True
    tabulation: bool = True
    early_stop: bool = False   # → "w/o early stop"
```

**Lưu ý:** đường dẫn output không phân biệt theo cờ ablation (chỉ phân biệt theo
`method`) — nếu muốn giữ kết quả của nhiều lần chạy ablation khác nhau, đổi tạm
`general.scale` trong `GeneralConfig` (ví dụ `"125m-no-removal"`) trước khi chạy,
hoặc backup thư mục `outputs/` sau mỗi lần.

## 8. Chạy API Service

```bash
cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Xem docs tại `http://localhost:8000/docs`.

| Endpoint | Việc |
|---|---|
| `POST /retrieve` | Retrieve bảng cho 1 câu hỏi, dùng đúng `pipeline.method` đang cấu hình |
| `GET /health` | Trạng thái service + dataset đã nạp |
| `GET /config` | Toàn bộ cấu hình đang hiệu lực (`config.yaml` + mặc định trong `src/config.py`) |
| `GET /evaluate` | **Tính recall@k / complete_recall@k thật** của lần chạy (xem dưới) |
| `GET /evaluate/available` | Liệt kê tổ hợp (dataset, model, method) đã có kết quả |

**Service nạp lười (lazy).** Khởi động KHÔNG encode corpus nào — vì các endpoint tra
cứu số liệu không cần model. Encoder/LLM/embeddings chỉ được nạp ở lần gọi
`/retrieve` đầu tiên của từng dataset, nên **request `/retrieve` đầu tiên sẽ chậm**
(nạp SGPT; và nếu chưa có cache `outputs/{dataset}_{scale}_embeddings.pt` thì phải
encode cả corpus — BirdUnion mất vài phút trên CPU). Các request sau nhanh bình
thường. `GET /health` cho biết trạng thái:

```json
{"datasets_available": ["spider", "bird"],   // có tables.json trên đĩa
 "datasets_loaded":    []}                    // đã nạp embeddings vào RAM
```

### 8b. `GET /evaluate` — tính metric THẬT của lần chạy trên máy này

Trả về `recall@k` / `complete_recall@k` **tính từ kết quả pipeline đã chạy trên máy
bạn** (không phải số trong paper). Metric được tính **tại thời điểm gọi** từ file
`result/turn{max_hop}/dev.json` (chứa danh sách bảng đã xếp hạng của từng câu hỏi),
chứ không đọc lại `score.json` — nên **k nào cũng được**, kể cả k chưa từng xuất
hiện trong `general.top_k`.

| Tham số | Mặc định | Giá trị |
|---|---|---|
| `dataset` | `spider` | `spider`, `bird` |
| `model` | `125m` | `125m`, `1.3b`, `2.7b`, `5.8b` (nhận cả `SGPT-125M`) |
| `method` | `murre` | `murre`, `single_hop`, `crush` |
| `k` | `5` | Một hoặc nhiều: `?k=3&k=5&k=10` |

```bash
# 1 giá trị k
curl "http://localhost:8000/evaluate?dataset=spider&model=125m&method=single_hop&k=5"

# nhiều k trong 1 request — đủ 4 cột như Table 2 của paper
curl "http://localhost:8000/evaluate?method=single_hop&k=3&k=5&k=10&k=20"

# xem tổ hợp nào đã có kết quả
curl "http://localhost:8000/evaluate/available"
```

```json
[{"dataset":"spider","model":"125m","method":"single_hop","k":5,
  "recall":73.1,"complete_recall":65.81,
  "num_questions":658,"retrieved_depth":100,
  "result_file":"outputs/spider/125m/single_hop/result/turn3/dev.json"}]
```

- `recall` / `complete_recall`: đơn vị **%**, làm tròn 2 số — so trực tiếp được với
  cột `r@K` / `k = K` trong Table 2 của paper.
- `num_questions`: số câu hỏi thực có trong file. **Kiểm tra số này** — nếu là 20
  thay vì 658 thì lần chạy đó dùng `--limit`, không so được với paper.
- `retrieved_depth`: số bảng đã lưu mỗi câu (chính là `pipeline.top_k_pool`), là
  giới hạn trên của `k`.

Các trường hợp lỗi:

| Tình huống | Mã | Ghi chú |
|---|---|---|
| Chưa chạy tổ hợp đó | `404` | Kèm luôn lệnh cần chạy trước |
| `k` > `retrieved_depth` | `400` | Chặn để không trả về `0.0` sai lệch âm thầm |
| `dataset`/`method` sai | `400` | Kèm danh sách giá trị hợp lệ |

Hàm `evaluate_run(dataset, model, method, k)` trong `src/api.py` dùng được trực tiếp
trong script Python, không cần chạy API.

Model request/response của API nằm riêng trong `src/schemas/` (`retrieve.py`,
`evaluate.py`, `health.py`) — `api.py` chỉ giữ routing và logic, không định nghĩa
model. Sửa/thêm field thì sửa trong `src/schemas/`, phần `/docs` và `openapi.json`
tự cập nhật theo.

## 9. Cấu trúc outputs

```
outputs/
├── {dataset}_{scale}_embeddings.pt                 ← Cache embeddings dùng chung
├── logs/murre.log                                  ← LoggingConfig.log_to_file = True
└── {dataset}/{scale}/{method}/
    ├── turn0/dev.json                              ← Kết quả Hop-0 (mọi method)
    ├── rewrite/outputs/turn{hop}/dev.{beam}.json    ← Chỉ method=murre
    ├── turn{hop}/dev.{beam}.json                    ← Chỉ method=murre
    └── result/turn{max_hop}/
        ├── dev.json         ← Kết quả cuối cùng (ranked schemas)
        ├── score.json       ← Metrics recall@K / complete_recall@K
        ├── sql.{k}.txt      ← SQL dự đoán với top-k bảng
        └── inp.{k}.txt      ← Prompt đã dùng (debug)
```

## 10. Chạy trong PyCharm

1. Chuột phải vào thư mục `src/` → **Mark Directory as → Sources Root**.
2. Vào **Run → Edit Configurations** cho mỗi script, đặt **Working directory** =
   thư mục gốc `MURRE/` (nơi có `config.yaml`) — **không phải** `src/`.
3. Nếu chạy qua Terminal tích hợp của PyCharm mà vẫn gặp `ModuleNotFoundError:
   No module named 'config'`, tạo file `.pth` trong venv trỏ tới `src/` (xem mục
   11), hoặc chạy bằng `python -m main` / `python -m steps.embed` (có `-m`, không
   có đuôi `.py`) sau khi `set PYTHONPATH=src`.

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
→ `src/` chưa được thêm vào `sys.path`. Chọn 1 trong 2 cách:

```bash
export PYTHONPATH=src        # Linux/Mac — phải set lại mỗi terminal mới
$env:PYTHONPATH = "src"      # Windows PowerShell
```

hoặc tạo file `.pth` cố định trong venv (không cần gõ lại mỗi lần):

```bash
python -c "
import site, os
path = os.path.join(site.getsitepackages()[-1] if hasattr(site, 'getsitepackages') else site.ENABLE_USER_SITE, 'murre_src.pth')
with open(path, 'w', encoding='ascii') as f:
    f.write(os.path.abspath('src'))
print('Đã tạo:', path)
"
```

Sau đó chạy bằng `python -m ...` (không chạy trực tiếp bằng đường dẫn file như
`python src/main.py`).

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

**`FileNotFoundError: Không tìm thấy file JSON trong: .../turn{N}`** (khi chạy
`steps.rewrite --hop N`)
→ Chưa chạy `steps.retrieve --hop N --beam b` cho hop đó. `rewrite --hop N` đọc thư
mục `turn{N}/`, nên phải retrieve xong hop N trước. Xem sơ đồ phụ thuộc ở mục 5b.

**`ValueError: File ... không có bảng nào trong trường 'retrieved'`** (khi chạy
`steps.rewrite`)
→ Thư mục đầu vào chứa file *đã viết lại* (`rewrite/outputs/turn{N}/`) chứ không phải
kết quả retrieve. File rewrite không có trường `retrieved`. Đúng thư mục đầu vào phải
là `turn{N}/` — xem bảng "Tham số của từng bước" ở mục 5b.

> Trước đây lỗi này hiện ra dưới dạng `KeyError: 'retrieved'`. Từ khi các file JSON
> có model riêng (`src/models/records.py`), `steps/rewrite.py` kiểm tra ngay lúc đọc
> file và báo rõ nguyên nhân kèm cách sửa.

**`FileNotFoundError: config.yaml`**
→ Working Directory đang không phải thư mục gốc `MURRE/`. Với PyCharm, sửa trong
Run Configuration; với terminal, `cd` về đúng thư mục gốc trước khi chạy.

**HuggingFace tải model chậm/bị chặn**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit')"
```

**Hết RAM khi encode** → giảm `encoder.batch_size` trong `config.yaml`.

## 12. Giới hạn đã biết

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
  sẵn (xem mục 2) — không tự động gộp Spider/BIRD gốc thành Union corpus.
