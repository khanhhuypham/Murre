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
4. [Option 1 — One question (1 câu hỏi)](#4-option-1--one-question-1-câu-hỏi)
    - [4b. Test riêng một method](#4b-test-riêng-một-method-python--m-methods)
5. [Option 2 — Batch (toàn dataset)](#5-option-2--batch-toàn-dataset)
    - [5b. Xem chi tiết từng hop khi debug](#5b-xem-chi-tiết-từng-hop-khi-debug)
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
│   ├── main.py                      ← Điểm vào pipeline (python -m main)
│   ├── server.py                    ← Điểm vào API: tạo FastAPI, lifespan, gắn
│   │                                  router (uvicorn server:app) — mục 8
│   ├── api/                         ← REST service, không có endpoint ở ngoài
│   │   ├── dependencies.py          ← Nạp lười encoder/LLM/embeddings theo dataset
│   │   ├── evaluator.py             ← evaluate_run(): tính metric từ file kết quả
│   │   ├── jobs.py                  ← Thân một job chạy pipeline trong thread riêng
│   │   └── routers/                 ← Mỗi file một nhóm endpoint
│   │       ├── health.py            ← /health, /config
│   │       ├── retrieve.py          ← /retrieve
│   │       ├── pipeline.py          ← /pipeline/run, /pipeline/jobs
│   │       └── evaluate.py          ← /evaluate, /evaluate/available
│   ├── core/                        ← Hạ tầng dùng chung cho MỌI method
│   │   ├── encoder.py               ← SGPTEncoder (Bi-Encoder, §3.2.1)
│   │   ├── llm.py                   ← LLMGenerator (OpenAI-compatible)
│   │   ├── rewriter.py              ← QueryRewriter (Removal + Tabulation, §3.4)
│   │   └── corpus.py                ← prepare(): corpus + embeddings (có cache)
│   ├── methods/                     ← 3 method của Table 2 + cách chọn và chạy chúng
│   │   ├── murre.py                 ← MurreRetriever — method chính (beam search, §3.2/§3.5)
│   │   ├── single_hop.py            ← Baseline Single-hop (§4.2)
│   │   ├── crush.py                 ← Baseline CRUSH (§4.2, prompt tự suy luận — xem mục 12)
│   │   ├── build.py                 ← build_retriever() / build_dataset(): ráp bộ retrieval
│   │   └── runner.py                ← NƠI DUY NHẤT viết cách chạy: run_one_question /
│   │                                  run_pipeline / run_batch
│   ├── steps/                       ← Hai bước phụ trợ chạy riêng được
│   │   ├── embed.py                 ← Mã hóa corpus trước (tuỳ chọn, có cache)
│   │   └── infer.py                 ← Sinh SQL từ file result
│   ├── models/                      ← Đối tượng NỘI BỘ dùng chung giữa các module
│   │   ├── retrieval.py             ← RetrievedTable (method trả về), RetrievedRow (dòng JSON)
│   │   └── records.py               ← ResultRecord (định dạng file result)
│   ├── schemas/                     ← Pydantic model cho request/response của API
│   │   ├── retrieve.py              ← RetrieveRequest, TableResult
│   │   ├── evaluate.py              ← EvalResult, AvailableRun
│   │   ├── pipeline.py              ← PipelineRunRequest, PipelineJob
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


Các tham số quan trọng nhất:

| Tham số | Mặc định | Sửa ở | Ý nghĩa |
|---|---|---|---|
| `general.dataset` | `spider` | `GeneralConfig` (config.py) | `spider` hoặc `bird` |
| `pipeline.method` | `murre` | `PipelineConfig` (config.py) | `murre` \| `single_hop` \| `crush` — chọn phương pháp retrieval |
| `pipeline.beam_size` | `5` | `PipelineConfig` (config.py) | B trong paper |
| `pipeline.max_hop` | `3` | `PipelineConfig` (config.py) | H trong paper |
| `pipeline.ablation.removal` | `True` | `AblationConfig` (config.py) | `False` = "w/o removal" (Table 4) |
| `pipeline.ablation.tabulation` | `True` | `AblationConfig` (config.py) | `False` = "w/o tabulation" (Table 4) |
| `pipeline.ablation.early_stop` | `True` | `AblationConfig` (config.py) | `False` = "w/o early stop" (Table 4) |
| `run_option.mode` | `batch` | `RunOptionConfig` (config.py) | `one_question` (Option 1) hoặc `batch` (Option 2) |
| `encoder.model_name` | SGPT-125M | `config.yaml` | Model THẬT được nạp, và cũng là **nguồn duy nhất** của nhãn thư mục `outputs/` (xem `cfg.encoder.slug`) — đổi model là outputs/ tự tách, không có biến nào phải sửa kèm |
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

Đổi model cho **cả project** (batch, one_question, API): sửa `llm.active_profile` trong
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

## 4. Option 1 — One question (1 câu hỏi)

Trong `src/config.py`, class `RunOptionConfig`:

```python
class RunOptionConfig(BaseModel):
    mode: str = "one_question"      # ← Option 1 (mặc định là "batch")
```

Option 1 chạy **một** câu hỏi trong process này rồi in top-N bảng ra terminal —
không ghi file nào. Muốn số liệu ghi ra `outputs/` thì dùng Option 2 (mục 5).

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
  DATASET : spider | ENCODER: Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit
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

Cả 3 method đi CHUNG một đường: `run_batch()` → `run_pipeline()` → retriever tương
ứng trong `methods/` (ráp qua `methods/build.py`). Mỗi method có đúng MỘT bản cài
đặt, nên số của `python -m main`, `/retrieve` và `/pipeline/run` luôn khớp nhau.

| `pipeline.method` | Lớp chạy | Các bước |
|---|---|---|
| `murre` | `MurreRetriever` (`methods/murre.py`) | hop 0 toàn corpus → `[Completing Tables → retrieve trong pool → tỉa beam] × max_hop` → Score_Path/Score_Table → sinh SQL |
| `single_hop` | `SingleHopRetriever` | retrieve một lần trên toàn corpus |
| `crush` | `CrushRetriever` | LLM hallucinate schema → retrieve |

Chỉ `murre` chạy thêm bước sinh SQL (`steps/infer.py`) sau khi retrieve xong — hai
baseline dừng ở `result` + `score`.

> Trước đây `murre` có đường batch RIÊNG: chuỗi `steps/{retrieve,rewrite,score}.py`,
> tức một bản cài đặt MURRE thứ hai đọc/ghi file trung gian. Bản đó khởi tạo điểm mọi
> bảng hop-0 bằng `0.0` trong khi `Score_Path` luôn ≤ 0, nên xếp hạng cuối bằng đúng
> thứ tự hop 0 — multi-hop bị vô hiệu hoàn toàn. Đã xoá; lấy lại từ git history nếu
> cần đối chiếu.

Embeddings tự nạp từ cache `.pt`, chưa có thì tự encode rồi lưu. Đặt
`FORCE_EMBED = True` để xoá cache và mã hóa lại (cần khi đổi encoder).

Chạy thử nhanh trên vài câu đầu (dùng được với cả 3 method):

```python
# src/main.py — khối __main__
LIMIT: int | None = 8
```

### 5b. Xem chi tiết từng hop khi debug

Không còn file trung gian để mở ra xem: `MurreRetriever` giữ toàn bộ beam search trong
RAM. Muốn theo dõi từng hop thì chạy Option 1 với `verbose=True` — nó in pool hop 0,
số ứng viên và số beam giữ lại ở mỗi hop, beam nào early-stop, và top-3 cuối cùng:

```bash
python -m methods.murre
```

Khối `__main__` của `src/methods/murre.py` có sẵn `VERBOSE = True` và cho phép ghi đè
`BEAM_SIZE` / `MAX_HOP` ngay tại chỗ, không phải sửa `config.yaml`.

Hai bước vẫn chạy riêng được:

| Lệnh | Tham số | Đọc từ | Ghi ra |
|---|---|---|---|
| `python -m steps.embed` | *(không có)* | `dataset/{ds}/tables.json` | `{dataset}_{model}_embeddings.pt` |
| `python -m steps.infer` | `--top_k K` (mặc định 5) | `result/turn{max_hop}/dev.json` | `result/turn{max_hop}/sql.{K}.txt` |

`steps.embed` không bắt buộc — mọi đường chạy đều tự encode và lưu cache khi thiếu.
Chạy trước cho xong hữu ích với BirdUnion (mất vài phút trên CPU).

**Chi phí LLM.** Mỗi hop gọi LLM `beam_size` lần cho mỗi câu hỏi — với Spider dev
(658 câu), beam 5 × 3 hop là ~9.870 lần gọi. Model local qua Ollama thì miễn phí
nhưng mất nhiều giờ; muốn kiểm tra pipeline chạy đúng thì đặt `LIMIT` nhỏ trong
`main.py`, hoặc giảm `beam_size = 2` và `max_hop = 1` trong `PipelineConfig`.

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
`outputs/{dataset}/{model}/{method}/result/turn{max_hop}/score.json` — vì đường dẫn
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
> với `run_option.mode = "batch"`. `methods/runner.py` chạy cả `dev.json` và ghi ra
> `result/turn{max_hop}/{dev,score}.json` — đúng file mà `/evaluate` đọc (mục 5).

**Collective retrieval trong CRUSH.** LLM thường hallucinate ra *nhiều* bảng, mỗi bảng
một dòng. Mặc định (`collective=True`) mỗi dòng được encode thành **một query riêng**,
rồi mỗi bảng trong corpus lấy điểm similarity **cao nhất** trên tất cả các query — đúng
cơ chế của CRUSH gốc. Với `COLLECTIVE = False`, toàn bộ output của LLM bị
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

**Lưu ý:** đường dẫn output phân biệt theo `{dataset}/{model}/{method}` — KHÔNG
phân biệt theo cờ ablation. Chạy nhiều lần ablation trên cùng model + method thì
lần sau ghi đè lần trước, nên hãy backup thư mục `outputs/` sau mỗi lần chạy.

Nhãn `{model}` suy ra từ `encoder.model_name` (bỏ phần org, hạ chữ thường), nên
đổi model là kết quả tự vào thư mục khác. Tra lại bằng
`/evaluate?model=sgpt-1.3b-weightedmean-msmarco-specb-bitfit`, hoặc gõ luôn tên
HuggingFace đầy đủ — endpoint tự chuẩn hoá về cùng nhãn đó.

## 8. Chạy API Service

```bash
cd src && uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Xem docs tại `http://localhost:8000/docs`. Không cần `--reload` thì chạy thẳng
`cd src && python -m server` (sửa host/port trong khối `__main__` của
`src/server.py`, cùng kiểu với `main.py`).

**Phải đứng trong `src/`.** `config.py` gọi `os.chdir(PROJECT_ROOT)` lúc import, nên
thư mục lúc KHỞI ĐỘNG mới là thứ quyết định `sys.path`; chạy từ gốc project sẽ nhận
`ModuleNotFoundError: No module named 'server'`.

`server.py` tạo FastAPI app và gắn router — mọi endpoint nằm trong
`src/api/routers/`. Thư mục `api/` là namespace package (không có `__init__.py`, giống
`core/`, `methods/`, `schemas/`, `utils/`) nên uvicorn cần một module phẳng như
`server.py` để nạp, thay vì trỏ thẳng vào package.

| Endpoint | Việc |
|---|---|
| `POST /retrieve` | Retrieve bảng cho 1 câu hỏi, dùng đúng `pipeline.method` đang cấu hình |
| `GET /health` | Trạng thái service + dataset đã nạp |
| `GET /config` | Toàn bộ cấu hình đang hiệu lực (`config.yaml` + mặc định trong `src/config.py`) |
| `GET /evaluate` | **Tính recall@k / complete_recall@k thật** của lần chạy (xem 8b) |
| `GET /evaluate/available` | Liệt kê tổ hợp (dataset, model, method) đã có kết quả |
| `POST /pipeline/run` | **Chạy** pipeline rồi tính metric tại k (xem 8c) |
| `GET /pipeline/jobs` | Danh sách job đã tạo từ khi service khởi động |
| `GET /pipeline/jobs/{job_id}` | Trạng thái / tiến độ một lần chạy |

**Service nạp lười (lazy).** Khởi động KHÔNG encode corpus nào — vì các endpoint tra
cứu số liệu không cần model. Encoder/LLM/embeddings chỉ được nạp ở lần gọi
`/retrieve` đầu tiên của từng dataset, nên **request `/retrieve` đầu tiên sẽ chậm**
(nạp SGPT; và nếu chưa có cache `outputs/{dataset}_{model}_embeddings.pt` thì phải
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
| `model` | nhãn suy ra từ `encoder.model_name` của server | Tên thư mục model trong `outputs/` (nhận cả tên HuggingFace đầy đủ) |
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
| `model` sai | `404` | Kèm đường dẫn đã tìm + các nhãn model đang có trên đĩa |

### 8c. `POST /pipeline/run` — CHẠY pipeline qua API

`/evaluate` chỉ ĐỌC metric của lần chạy đã có. Endpoint này mới là thứ TẠO ra lần
chạy đó: nó ghi `paths.result` + `paths.score`, chạy xong là `/evaluate` tra được
ngay với mọi k. Tương đương `run_batch` trong `main.py`, nhưng gọi được qua HTTP.

Body (`PipelineRunRequest`) — **không có `model`**: encoder do server quyết định,
lấy từ `encoder.model_name`. Gửi thừa field sẽ nhận `422` (`extra="forbid"`).

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `dataset` | `spider` | `spider`, `bird` |
| `method` | `murre` | `murre`, `single_hop`, `crush` |
| `k` | `5` | k dùng để báo recall sau khi chạy xong |
| `limit` | `null` | Chỉ chạy N câu đầu của `dev.json` (bỏ trống = chạy hết) |

Query `?wait=` quyết định cách trả kết quả:

| `wait` | Mã | Hành vi |
|---|---|---|
| `true` (mặc định) | `200` | Giữ request tới khi chạy xong, response đã có sẵn `result` |
| `false` | `202` | Trả job ngay, tự poll `GET /pipeline/jobs/{job_id}` |

```bash
# chạy thử nhanh 20 câu, chờ tới khi xong
curl -X POST "http://localhost:8000/pipeline/run"      -H "Content-Type: application/json"      -d '{"dataset":"spider","method":"single_hop","k":5,"limit":20}'

# lần chạy dài — trả job ngay rồi poll
curl -X POST "http://localhost:8000/pipeline/run?wait=false"      -H "Content-Type: application/json"      -d '{"dataset":"spider","method":"murre","k":5}'
curl "http://localhost:8000/pipeline/jobs/<job_id>"
```

`PipelineJob` có `status` (`queued` / `running` / `succeeded` / `failed`),
tiến độ `processed`/`total`, `started_at`/`finished_at`, và `result` (chính là
`EvalResult` ở 8b) khi `status=succeeded` — hoặc `error` khi `failed`.

Hai lưu ý:

- **Mỗi lúc chỉ chạy được MỘT job.** `cfg` là biến toàn cục của process, nên gọi
  khi đang có job chưa kết thúc sẽ nhận `409` kèm danh sách job đang chạy.
- **Job chỉ nằm trong RAM** — restart service là mất danh sách job. Kết quả thì đã
  ghi ra đĩa, tra lại bằng `/evaluate` hoặc `/evaluate/available`.

Thời gian chạy đủ 658 câu, đo trên máy này (Ollama + `qwen2.5:0.5b`):
`single_hop` ~2.7 phút | `crush` ~32 phút | `murre` (beam 5 × hop 3) ~1.6 giờ.
Vì vậy `wait=true` chỉ hợp lý khi có `limit` nhỏ.

Hàm `evaluate_run(dataset, model, method, k)` trong `src/api/evaluator.py` dùng được
trực tiếp trong script Python, không cần chạy API (`from api.evaluator import evaluate_run`).

`src/api/` chia theo vai trò: `routers/*.py` chỉ khai báo endpoint và chuyển tiếp,
phần làm việc thật nằm ở `dependencies.py` (nạp model), `evaluator.py` (tính metric),
`jobs.py` (chạy pipeline nền). Thêm nhóm endpoint mới thì tạo file trong `routers/`
rồi `include_router` trong `src/server.py`.

Trạng thái dùng chung (encoder, LLM, dataset đã nạp, danh sách job) nằm trong
`app.state`, khởi tạo ở `server.py::lifespan`; router lấy qua `request.app.state`
nên không file nào phải import ngược lại `app`.

Model request/response của API nằm riêng trong `src/schemas/` (`retrieve.py`,
`evaluate.py`, `pipeline.py`, `health.py`) — `src/api/` chỉ giữ routing và logic,
không định nghĩa
model. Sửa/thêm field thì sửa trong `src/schemas/`, phần `/docs` và `openapi.json`
tự cập nhật theo.

## 9. Cấu trúc outputs

```
outputs/
├── {dataset}_{model}_embeddings.pt                 ← Cache embeddings dùng chung
├── logs/murre.log                                  ← LoggingConfig.log_to_file = True
└── {dataset}/{model}/{method}/
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
   11), hoặc chạy bằng `python -m main` / `python -m steps.infer` (có `-m`, không
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

**`FileNotFoundError: .../result/turn{max_hop}/dev.json`** (khi chạy
`python -m steps.infer`)
→ Chưa có file kết quả để sinh SQL. Chạy `python -m main` (mode `batch`) hoặc
`POST /pipeline/run` trước; `run_batch()` với `method: murre` đã tự gọi `infer` ở cuối
nên bình thường không cần chạy tay bước này.

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
