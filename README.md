# MURRE — Multi-Hop Table Retrieval

Cho một câu hỏi tự nhiên, tìm những **bảng SQL liên quan** trong kho schema nhiều
database, rồi sinh câu SQL từ các bảng đó. Hỗ trợ **tiếng Anh** (Spider, BIRD) và
**tiếng Việt** (ViText2SQL).

Cài đặt theo paper *MURRE: Multi-Hop Table Retrieval with Removal for Open-Domain
Text-to-SQL* (COLING 2025). Năm chỗ bản này bám paper thay vì bám code phát hành
của tác giả: xem docstring đầu [src/pipeline/retriever.py](src/pipeline/retriever.py).

**Nhánh này KHÔNG nhằm tái hiện số của paper.** Thuật toán retrieval vẫn theo
paper, nhưng encoder thì không: SGPT-125M đã bị bỏ, thay bằng một bi-encoder đa
ngữ dùng cho mọi dataset (xem mục 6). Nghĩa là số đo trên Spider/BIRD của bản này
không so trực tiếp được với Bảng 2 của paper — cần bản đối chiếu thì xem lịch sử
git.

Dùng được hai đường: **CLI** (chạy batch, đo metric) và **HTTP API** (service).

---

## 1. Cách hoạt động

Một câu hỏi thường cần nhiều bảng, nhưng encode cả câu thành một vector thì bảng
nào "nổi" nhất sẽ át các bảng còn lại. MURRE đi **nhiều hop**, mỗi hop tìm thêm
một bảng:

```
hop 1        câu hỏi gốc ──encode──> quét toàn corpus ──> top-B bảng  (B đường đi)

hop 2..H     mỗi đường đi:
               Removal   LLM xoá thông tin các bảng ĐÃ CÓ khỏi câu hỏi gốc,
                         trả về phần còn thiếu dưới dạng bảng.
                         Trả "None" → nhánh đó dừng sớm.
               Retrieval câu vừa nhận quét lại toàn corpus → top-B bảng.
             B×B đường đi mới, tỉa còn B đường tốt nhất cho hop kế.

cuối         Score_Path  = tích Norm(similarity) trên đường đi
             Score_Table = max Score_Path trong các đường đi chứa bảng đó
             → xếp hạng giảm dần, trả về danh sách bảng.
```

Hai model tham gia:

| Vai trò | Model | Khai ở |
| --- | --- | --- |
| Encode câu hỏi & schema | bi-encoder đa ngữ (`multilingual-e5-base`) | `encoders` + `datasets.<ds>.encoder` |
| Pha Removal + sinh SQL | LLM qua API tương thích OpenAI | `llm.active_profile` |

---

## 2. Cài đặt

Cần Python 3.11+ và một endpoint LLM (Ollama local, OpenAI, hoặc Groq).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Chạy LLM local bằng Ollama (mặc định của `config.yaml`):

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

Dùng OpenAI hoặc Groq thay vì Ollama thì tạo file `.env` ở gốc project:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.groq.com/openai/v1     # bỏ dòng này nếu dùng OpenAI
```

rồi đổi `llm.active_profile` trong `config.yaml`. `.env` chỉ giữ **bí mật** —
mọi thứ khác nằm trong `config.yaml` (file đó nằm trong git nên đừng để key vào).

Lần chạy đầu sẽ tải encoder từ HuggingFace (`intfloat/multilingual-e5-base`,
~1.1 GB) và mã hoá cả corpus schema — mất vài phút. Kết quả cache lại
(`outputs/{dataset}_{model}_embeddings.pt`), lần sau nạp gần như tức thì.

Kiểm tra cấu hình đang thực sự hiệu lực:

```bash
cd src && python -m cli config
```

---

## 3. CLI

Mọi lệnh chạy **từ thư mục `src/`**.

```bash
cd src

python -m cli ask                            # câu đầu dev.json, in top-K bảng
python -m cli ask -q "Which airlines fly to AHD?" -v -k 10
python -m cli ask --dataset vitext2sql -v    # tiếng Việt
python -m cli run --limit 20                 # 20 câu đầu, ghi kết quả ra outputs/
python -m cli run --dataset bird             # cả dev.json của BIRD
python -m cli run --sql --top-k 5            # chạy xong thì sinh luôn SQL
python -m cli embed                          # chỉ mã hoá corpus rồi lưu cache
```

`run` ghi **checkpoint sau từng câu**, nên Ctrl-C giữa chừng không mất gì — chạy
lại là đi tiếp từ chỗ dừng. Đổi `beam_size`, `max_hop`, encoder hay LLM thì
checkpoint cũ tự bị loại (đổi tên thành `.stale`), không trộn hai cấu hình vào
một bảng điểm.

Đầu ra nằm trong `outputs/{dataset}/{encoder}/turn{max_hop}/`:

| File | Nội dung |
| --- | --- |
| `dev.json` | mỗi câu hỏi: bảng gold + danh sách bảng đã retrieve, kèm điểm |
| `score.json` | recall@k và complete_recall@k tại các k của `general.top_k` |
| `sql.{k}.txt` | SQL sinh từ top-k bảng (chỉ có khi chạy `--sql`) |
| `checkpoint.jsonl` | tiến độ để chạy tiếp; xoá được sau khi xong |

---

## 4. HTTP API

```bash
cd src && python -m server
# hoặc: cd src && uvicorn server:app --host 0.0.0.0 --port 8000
```

Docs tương tác: <http://localhost:8000/docs>

| Endpoint | Việc |
| --- | --- |
| `GET /health` | trạng thái; **503** khi còn warm-up → dùng làm readiness probe |
| `GET /config` | cấu hình đang hiệu lực (`api_key` đã che) |
| `POST /retrieve` | câu hỏi → danh sách bảng đã xếp hạng |
| `POST /sql` | câu hỏi → bảng liên quan → câu lệnh SQL |
| `POST /pipeline/run` | chạy cả dev.json rồi trả metric; `?wait=false` → job để poll |
| `GET /pipeline/jobs` | danh sách job đã tạo (chỉ trong RAM, restart là mất) |
| `GET /pipeline/jobs/{id}` | tiến độ / kết quả của một lần chạy |
| `GET /evaluate` | tính lại recall@k từ kết quả đã có trên đĩa |
| `GET /evaluate/available` | những lần chạy nào đã có kết quả |

**`/retrieve` và `/sql` nhận đúng cùng một body** — cùng ba field, cùng tên
(`schemas/common.py`):

```json
{
  "question": "Có tất cả bao nhiêu kiến trúc sư nữ ?",
  "dataset": "vitext2sql",
  "top_k": 5
}
```

| Field | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `question` | có | câu hỏi tự nhiên, không được rỗng |
| `dataset` | không | `spider` (mặc định) \| `bird` \| `vitext2sql` |
| `top_k` | không | số bảng lấy ra, 1–20; mặc định `pipeline.top_k_output` |

Body từ chối field lạ: gõ `top_n` sẽ nhận **422 chỉ thẳng chỗ sai**, thay vì bị bỏ
qua âm thầm rồi tưởng server đã nhận.

```bash
curl -X POST localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which airlines fly to AHD?", "dataset": "spider", "top_k": 5}'
```

**Một server phục vụ được cả ba dataset cùng lúc** — chỉ cần đổi field `dataset`
trong body. Mỗi dataset khai encoder của nó, nhưng hiện cả ba trỏ cùng một profile
nên model chỉ nạp **một** bản trong RAM (~1.1 GB), phần thêm mỗi dataset chỉ là
corpus + embeddings của nó. `GET /health` in ra dataset nào đang phục vụ và encoder
nào đi với nó.

Với `api.preload: true` (mặc định), service nạp encoder + embeddings + ping LLM
**trước khi** nhận request: khởi động chậm hàng chục giây nhưng request đầu tiên
không lỗi vì thiếu model. `/health` trả 503 trong suốt lúc đó.

`cfg` là biến toàn cục của process nên **mỗi lúc chỉ chạy được một** job
`/pipeline/run`; gọi chồng lên sẽ nhận `409`.

**Thời gian**: mỗi câu tốn `beam_size × (max_hop − 1)` lần gọi LLM cho pha Removal,
cộng 1 lần sinh SQL. Với mặc định 5 × 2 = 10 lượt, đo trên Ollama + qwen2.5-coder:7b
thì `/sql` mất **80–200 giây** một câu (câu cần nhiều bảng thì lâu hơn). Gọi bằng
Postman thì đặt **Settings → Request timeout in ms** = `0`, kẻo bị ngắt giữa chừng.

---

## 5. Docker

```bash
docker build -t murre .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v murre-models:/models \
  -v "$PWD/outputs:/app/outputs" \
  murre
```

Hai volume đó là bắt buộc trên thực tế: `/models` giữ encoder đã tải từ
HuggingFace, `/app/outputs` giữ cache embeddings và kết quả chạy. Không mount thì
mỗi lần khởi động lại tải và mã hoá lại từ đầu.

LLM chạy trên host (Ollama) thì trỏ `OPENAI_BASE_URL` vào
`http://host.docker.internal:11434/v1`.

---

## 6. Cấu hình

**Chỉ một file [config.yaml](config.yaml)** cho mọi dataset. Mặc định của mọi khoá
nằm trong [src/config.py](src/config.py); `config.yaml` chỉ khai những gì cần đổi.

Điểm mấu chốt: **mỗi dataset khai encoder của nó**, nên đổi ngôn ngữ chỉ là đổi
`general.dataset` — encoder tự đi theo, không phải nhớ sửa thêm gì. Hiện cả ba
dataset trỏ chung một profile đa ngữ, nhưng tách ra chỉ là thêm một profile.

```yaml
encoders:
    multilingual: { type: sentence, model_name: intfloat/multilingual-e5-base,
                    query_prefix: "query: ", doc_prefix: "passage: " }

datasets:
    spider:     { encoder: multilingual }
    bird:       { encoder: multilingual }
    # format: dữ liệu trên đĩa đang ở dạng nào — xem mục 7b
    vitext2sql: { encoder: multilingual, format: vitext2sql }
```

Những khoá hay chạm nhất:

| Khoá | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `general.dataset` | `spider` | `spider` \| `bird` \| `vitext2sql` |
| `encoders.<tên>.type` | `sentence` | họ encoder; hiện chỉ có `sentence` (mean pooling + tiền tố) |
| `datasets.<ds>.encoder` | `multilingual` | dataset này dùng profile encoder nào |
| `datasets.<ds>.format` | `murre` | `murre` (đã tiền xử lý) hoặc `vitext2sql` (thô) |
| `pipeline.beam_size` | 5 | B — số nhánh giữ lại mỗi hop |
| `pipeline.max_hop` | 3 | H — **đếm cả hop 1**, nên H=3 là 2 lượt Removal |
| `pipeline.top_k_output` | 5 | số bảng mặc định lấy ra (API và CLI) |
| `llm.active_profile` | `qwen2.5-coder-7b` | chọn một khối trong `llm.profiles` |
| `api.datasets` | cả ba | dataset service phục vụ; càng nhiều khởi động càng lâu |
| `api.preload` | `true` | nạp sẵn model lúc khởi động |
| `encoders.<tên>.max_batch_tokens` | 16384 | trần bộ nhớ khi mã hoá — giảm nếu máy chết (mục 9) |

Bí mật đi qua `.env` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) vì `config.yaml` nằm
trong git. Muốn mỗi môi trường triển khai một file riêng thì dùng
`--config <đường dẫn>` (hoặc env `MURRE_CONFIG_PATH`):

```bash
cd src && python -m server --config ../config.prod.yaml
```

---

## 7. Dữ liệu

Có HAI bố cục, phân biệt bằng `datasets.<ds>.format` trong `config.yaml`.

**`format: murre`** — đã tiền xử lý sẵn, đi kèm repo:

```
dataset/{spider,bird}/
├── tables.json    schema của mọi database, kèm khoá `schema` đã tiền xử lý
├── dev.json       câu hỏi + bảng gold (rel_schema)  ← đầu vào của retrieval
└── gold.txt       câu SQL đúng, để đối chiếu sql.{k}.txt bằng công cụ ngoài
```

**`format: vitext2sql`** — dữ liệu THÔ, giữ nguyên xi bản gốc, tải riêng. Bố cục
khác hẳn và **không có `gold.txt`** (câu SQL đúng nằm sẵn trong `dev.json`).
Xem mục 7b.

Thêm dataset mới dạng `murre`: đặt `tables.json` + `dev.json` theo quy ước trên,
thêm `prompts/{tên}_rewrite.txt`, rồi thêm một field vào `DatasetsConfig`
(`src/config.py`) và một member vào `Dataset` (`src/enums.py`).

### 7b. Tiếng Việt — ViText2SQL

[ViText2SQL](https://github.com/VinAIResearch/ViText2SQL) (VinAI) là bản dịch
Spider sang tiếng Việt: cùng 166 database / 876 bảng, nhưng câu hỏi và tên
bảng/cột đều là tiếng Việt. **Dữ liệu không đi kèm repo và không có script tải** —
lấy bằng tay từ [thư mục `data/` của upstream](https://github.com/VinAIResearch/ViText2SQL/tree/master/data)
rồi đặt vào `dataset/vitext2sql/data/` theo đúng bố cục bên dưới (~76 MB).

Rồi chạy — không cần bước chuyển đổi nào:

```bash
cd src
python -m cli ask -v --dataset vitext2sql
python -m cli run --dataset vitext2sql --limit 20
```

Với API thì chỉ cần `"dataset": "vitext2sql"` trong body.

**Dữ liệu giữ nguyên xi bản gốc.** Cây thư mục sao y upstream — đặt file vào đúng
chỗ này thì `loader.py` tìm thấy, không cần khai đường dẫn trong `config.yaml`:

```
dataset/vitext2sql/data/           ← bản sao nguyên vẹn của ViText2SQL/data/
├── syllable-level/{dev,test,train,tables}.json, test_gold.sql
└── word-level/                    (như trên)
```

**Tải bằng tay thì không có gì kiểm tra hộ.** Lấy file từ GitHub qua trình duyệt
rất dễ ra HTML của trang thay vì JSON, hoặc lẫn `word-level` với `syllable-level`.
Cả hai lỗi đó KHÔNG nổ ra lúc đọc — chỉ thấy recall tụt bất thường. Dùng nút
**Download raw file**, rồi kiểm nhanh:

```bash
python -c "import json;d=json.load(open('dataset/vitext2sql/data/syllable-level/dev.json',encoding='utf-8'));print(len(d),d[0]['question'])"
```

Đúng thì in ra `954` kèm một câu hỏi tiếng Việt (split dev, mức âm tiết).

**MURRE cần định dạng khác, và phần đó nằm trong CODE.** Dữ liệu thô thiếu ba khoá
mà pipeline cần:

| MURRE cần | ViText2SQL có | Ai bù |
| --- | --- | --- |
| `tables.json` có khoá `schema` | không có | `dataset/vitext2sql.py::adapt_tables` |
| `dev.json` có `utterance` | `question` | `adapt_split` |
| `dev.json` có `rel_schema` (bảng gold) | không có | `adapt_split` → `gold_tables` |

Việc bù chạy **lúc đọc file**, không ghi đè gì. `datasets.vitext2sql.format:
vitext2sql` trong `config.yaml` là chỗ bật nó; `dataset/loader.py` rẽ theo khoá đó
rồi trả về cùng một hình dạng cho cả ba dataset, nên phần còn lại của pipeline
không cần biết có hai định dạng.

Đổi lại: mỗi lần đọc phải dựng lại `schema` và `rel_schema` — đo trên split dev
(954 câu, 166 database) là dưới một giây, không đáng kể so với mã hoá corpus, mà
corpus thì đã có cache riêng.

Bảng gold suy ra bằng cách lấy **mọi bảng trong FROM/JOIN, kể cả trong subquery** —
quy tắc này đã đối chiếu với `dataset/spider/dev.json` và khớp 658/658 câu. Suy từ
hai nguồn rồi hợp lại, vì mỗi nguồn thiếu một kiểu: cây cú pháp `sql` của
ViText2SQL **bỏ sót bảng thứ ba ở các câu JOIN 3 bảng** (44/954 câu của split dev),
còn đọc token thì không thấy bảng nằm trong subquery ở mệnh đề FROM.

**Encoder phải là model đa ngữ.** Đây là lý do encoder SGPT của paper đã bị bỏ
khỏi nhánh này: nó chỉ học tiếng Anh, dùng nguyên cho tiếng Việt thì retrieval gần
như ngẫu nhiên. Đo trên chính corpus này (954 câu, 876 schema, chỉ hop 1 nên không
có LLM xen vào):

| encoder | r@3 | r@5 | r@10 | r@20 |
| --- | --- | --- | --- | --- |
| SGPT-125M (đã bỏ) | 4.0 | 5.6 | 10.2 | 16.6 |
| multilingual-e5-base (`type: sentence`) | **73.4** | **82.2** | **90.7** | **94.6** |

Nguyên nhân nằm ở tokenizer: BPE tiếng Anh của SGPT băm mỗi schema tiếng Việt ra
**82 token** (schema tiếng Anh tương đương chỉ 24), câu hỏi 67 token thay vì 15 —
văn bản vỡ vụn thành byte nên vector không còn nghĩa, mà lại chậm hơn. E5 dùng
tokenizer đa ngữ, cùng nội dung đó hết 32 token.

Việc chọn encoder đã nằm sẵn trong `config.yaml`: `datasets.vitext2sql.encoder`
trỏ tới profile `multilingual`, kèm hai tiền tố `query: ` / `passage: ` mà họ E5
bắt buộc phải có.

**Đổi mức tách từ hoặc đổi split.** ViText2SQL có `syllable` (âm tiết rời,
"kiến trúc sư") và `word` (nối gạch dưới, "kiến_trúc_sư"). Mức `word` dành cho
model tiếng Việt có word segmentation (PhoBERT); với bi-encoder thông thường thì
`syllable` tự nhiên hơn nên đó là mặc định. Cả hai mức đã nằm sẵn trên đĩa — đổi
bằng cách trỏ thẳng đường dẫn trong `config.yaml`:

```yaml
datasets:
    vitext2sql:
        encoder: multilingual
        format: vitext2sql
        tables: dataset/vitext2sql/data/word-level/tables.json
        dev:    dataset/vitext2sql/data/word-level/dev.json
```

Trỏ `dev` sang `test.json` là đánh giá trên split test. Hai mức có **cùng 876
schema** nên ghép nhầm `tables` mức này với `dev` mức kia sẽ không lộ ra ở số
lượng — `adapt_split` bắt bằng cách đối chiếu `db_id` và báo lỗi rõ. Cache
embeddings cũng mang vân tay nội dung corpus nên đổi mức là tự mã hoá lại chứ
không dùng nhầm vector cũ.

**Prompt Removal.** `prompts/vitext2sql_rewrite.txt` đã có sẵn trong repo, dựng từ
split **train** (không phải dev — ví dụ trùng câu đang đo thì điểm đo được là điểm
của trí nhớ). Ví dụ là tiếng Việt nhưng phần khung giữ nguyên tiếng Anh:
`Completing Tables:` và `None` chính là thứ `core/rewriter.py` dò để biết một nhánh
đã đủ bảng, dịch nhãn đi là Early Stop không bao giờ khớp.

Chuyển sang `word-level` thì dựng lại prompt, kẻo ví dụ trong prompt viết khác hẳn
schema thật mà model đang nhìn:

```bash
python scripts/build_vitext2sql_prompt.py --level word
```

---

## 8. Cấu trúc mã nguồn

```
config.yaml                    toàn bộ cấu hình, mọi dataset
requirements.txt               phụ thuộc lúc chạy (-dev.txt thêm pytest)
pytest.ini                     đặt sys.path = src + scripts + gốc
Dockerfile                     image chạy API, torch bản CPU

prompts/                       prompt few-shot của pha Removal, một file / dataset
dataset/                       DỮ LIỆU + cách đọc dữ liệu
├── loader.py                  đọc mọi dataset, rẽ theo `format`, trả cùng hình dạng
├── vitext2sql.py              thích nghi ViText2SQL THÔ sang định dạng MURRE
├── spider/  bird/             dữ liệu đã tiền xử lý, đi kèm repo
└── vitext2sql/data/           bản sao nguyên xi upstream (tải bằng tay, xem 7b)

src/
├── cli.py                     điểm vào CLI (ask / run / embed / config)
├── server.py                  điểm vào API — tạo app, lifespan, exception handler
├── config.py                  nạp & validate config.yaml, dựng mọi đường dẫn
├── enums.py                   Dataset, JobStatus
├── core/                      encoder (bi-encoder đa ngữ), llm, rewriter (Removal), corpus
├── pipeline/                  retriever (MURRE), runner (chạy batch), factory, sql
├── api/                       dependencies (vòng đời model), evaluator, jobs, routers/
├── models/                    kiểu dữ liệu miền: errors, records, retrieval, metrics
├── schemas/                   DTO pydantic của API (common.py giữ phần chung)
└── utils/                     logger, metrics, scoring, schema, display

scripts/
└── build_vitext2sql_prompt.py dựng prompt Removal từ split train

tests/                         121 test, không tải model và không gọi mạng
```

Hai ranh giới đáng nhớ:

- **`dataset/` nằm NGOÀI `src/`** vì nó vừa là dữ liệu vừa là cách đọc dữ liệu.
  Mọi khác biệt định dạng bị chặn lại ở `loader.py`; phần còn lại của pipeline
  chỉ thấy một hình dạng duy nhất.
- **Tầng lõi (`core/`, `pipeline/`, `models/`) không import FastAPI.** Lỗi có ngữ
  nghĩa được ném dưới dạng `AppError`, và `server.py` dịch sang HTTP đúng một lần
  bằng exception handler.

---

## 9. Sự cố hay gặp

**Tiến trình chết không báo lỗi Python** — PyCharm hiện
`Process finished with exit code -1073741819 (0xC0000005)`, terminal thì im lặng.

Đó là ACCESS_VIOLATION: torch hết bộ nhớ TRONG code C nên không kịp ném
`MemoryError` của Python. Gần như luôn xảy ra ở bước mã hoá corpus lúc khởi động —
xem dòng log cuối, nếu là `[Corpus] Đang encode ...` thì đúng nó.

Cách xử lý: giảm `encoders.<tên>.max_batch_tokens` trong `config.yaml`
(16384 → 8192 → 4096). Trần này giới hạn `số câu × độ dài câu dài nhất` của một
batch; giảm nó là đỉnh bộ nhớ giảm theo, chỉ chậm hơn chút.

Bộ nhớ tăng theo BÌNH PHƯƠNG độ dài chuỗi, nên thủ phạm luôn là vài schema dài bất
thường chứ không phải corpus lớn: schema dài nhất của BIRD là 598 token, gộp chung
một batch 256 câu thì cả 256 phải pad tới 598 — riêng một tensor attention đã
~4.4 GB. Đo trên máy 16 GB: RSS đỉnh **8.4 GB** trước khi có trần này, **2.7 GB**
sau khi có (và mã hoá còn nhanh hơn vì bớt padding thừa).

Đóng bớt ứng dụng nặng cũng giúp: bản thân PyCharm giữ 1–2 GB, đủ để đẩy lần chạy
sát trần thành lần chạy chết.

**`[Errno 10048] only one usage of each socket address ...`** — cổng 8000 đã có
tiến trình khác giữ (thường là server chạy từ lần trước chưa tắt). Server tự kiểm
tra cổng TRƯỚC khi nạp model nên báo ngay, kèm lệnh giải phóng:

```bash
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
```

**422 `Input should be 'spider', 'bird' or 'vitext2sql'`** — sai chính tả tên
dataset. Tên đúng là `vitext2sql`, **có số 2**.

**422 `Extra inputs are not permitted`** — sai tên field. Cả `/retrieve` và `/sql`
đều dùng `top_k` (tên cũ `top_n` đã bỏ). Body từ chối field lạ thay vì bỏ qua âm
thầm, nên lỗi chỉ thẳng chỗ gõ sai.

**Lần khởi động đầu sau khi cập nhật code chạy lâu.** Cache embeddings mang vân tay
nội dung corpus + tên model; đổi một trong hai là nó tự mã hoá lại (log báo
`thuộc corpus/model khác → encode lại`). Chỉ xảy ra một lần.

**`ollama serve` chưa bật** — lỗi báo ngay lúc khởi động kèm gợi ý, vì
`api.preload: true` ping thử LLM trước khi nhận request.

**Gọi API mà thấy sai dataset / sai encoder** — kiểm tra `GET /health` trước: nó in
ra encoder của từng dataset đang phục vụ.

---

## 10. Test

```bash
pip install -r requirements-dev.txt
pytest
```

121 test, chạy khoảng 20 giây — gần hết là thời gian import torch. Encoder và
LLM đều được thay bằng bản giả: không tải model, không gọi mạng, không đụng tới
`dataset/` hay `outputs/`.
