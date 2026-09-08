"""scripts/prepare_vitext2sql.py — ViText2SQL (VinAI) → định dạng dataset của MURRE.

    python scripts/prepare_vitext2sql.py                    # tải về, mức syllable
    python scripts/prepare_vitext2sql.py --level word       # mức word (có gạch dưới)
    python scripts/prepare_vitext2sql.py --split test       # dùng split test làm eval
    python scripts/prepare_vitext2sql.py --emit-prompt      # dựng lại prompt Removal
    python scripts/prepare_vitext2sql.py --src <thư mục>    # đã tải sẵn thì đọc local

Nguồn: https://github.com/VinAIResearch/ViText2SQL (bản dịch Spider sang tiếng Việt).

HAI ĐỊNH DẠNG KHÁC NHAU CHỖ NÀO
-------------------------------
ViText2SQL giữ nguyên định dạng THÔ của Spider, còn MURRE cần bản đã tiền xử lý:

    tables.json   thiếu khoá `schema` — danh sách chuỗi "db.bảng(cột, cột, ...)"
                  mà retrieval encode. Script dựng từ table_names + column_names.

    dev.json      thô có: db_id, query, query_toks, question, question_toks, sql
                  MURRE cần: id, db_id, utterance, query, rel_schema
                  `rel_schema` là các bảng gold — không có sẵn, phải suy ra từ câu
                  truy vấn (xem gold_tables).

    gold.txt      ViText2SQL chỉ có test_gold.sql cho split test. Script tự ghi lại
                  theo đúng định dạng "<query>\\t<db_id>" của spider/bird.

    prompt        prompts/vitext2sql_rewrite.txt — prompt few-shot của pha Removal,
                  dựng từ split TRAIN (--emit-prompt). Xem build_prompt().

MỨC TÁCH TỪ: `word` nối các âm tiết của một từ bằng gạch dưới ("kiến_trúc_sư"),
`syllable` để rời ("kiến trúc sư"). Mức word dành cho model tiếng Việt có word
segmentation (PhoBERT); với bi-encoder đa ngữ thông thường thì `syllable` là văn
bản tự nhiên hơn — nên đó là mặc định.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

RAW_URL: str = (
    "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/{level}-level/{name}"
)
LEVELS: Tuple[str, ...] = ("syllable", "word")
SPLITS: Tuple[str, ...] = ("dev", "test", "train")

DEFAULT_OUT: str = "dataset/vitext2sql"
DEFAULT_PROMPT: str = "prompts/vitext2sql_rewrite.txt"


# ---------------------------------------------------------------------------
# Đọc dữ liệu nguồn
# ---------------------------------------------------------------------------
def load_source(name: str, level: str, src: Optional[str]) -> Any:
    """Đọc một file của ViText2SQL — từ thư mục `src` nếu có, không thì tải về."""
    if src:
        path: str = os.path.join(src, name)
        if not os.path.exists(path):
            raise SystemExit(
                f"Không thấy {path}. --src phải trỏ vào thư mục chứa tables.json "
                f"và <split>.json của ViText2SQL."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    url: str = RAW_URL.format(level=level, name=name)
    print(f"  tải {url}")
    with urllib.request.urlopen(url, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# tables.json → thêm khoá `schema`
# ---------------------------------------------------------------------------
def schema_strings(db: Dict[str, Any]) -> List[str]:
    """Một chuỗi "db_id.bảng(cột, cột, ...)" cho mỗi bảng — đầu vào của retrieval.

    Đúng quy ước của dataset/spider/tables.json: dùng tên đã chuẩn hoá
    (table_names / column_names, không phải *_original), bỏ cột ảo "*".
    """
    db_id: str = db["db_id"]
    columns: List[List[str]] = [[] for _ in db["table_names"]]
    for table_idx, column_name in db["column_names"]:
        if table_idx >= 0:  # -1 là cột ảo "*", không thuộc bảng nào
            columns[table_idx].append(column_name)

    return [
        f"{db_id}.{table_name}({', '.join(columns[i])})"
        for i, table_name in enumerate(db["table_names"])
    ]


def convert_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**db, "schema": schema_strings(db=db)} for db in tables]


# ---------------------------------------------------------------------------
# Suy ra bảng gold của một câu hỏi
# ---------------------------------------------------------------------------
def tables_from_tokens(
    query_toks: Sequence[str], table_names: Sequence[str],
) -> List[str]:
    """Bảng đứng ngay sau FROM / JOIN trong câu truy vấn, theo thứ tự xuất hiện.

    Đọc `query_toks` chứ không regex trên chuỗi: tên bảng tiếng Việt có dấu cách
    ("kiến trúc sư") nhưng ViText2SQL vẫn giữ nó thành MỘT token, nên so khớp
    token là chính xác, không phải đoán biên từ.

    FROM mở subquery thì token kế là "(" — không khớp tên bảng nào, và bảng bên
    trong subquery do tables_from_ast() nhặt.
    """
    known: Dict[str, str] = {name.lower(): name for name in table_names}
    toks: List[str] = [t.lower() for t in query_toks]

    out: List[str] = []
    for i, tok in enumerate(toks[:-1]):
        if tok in ("from", "join"):
            name: Optional[str] = known.get(toks[i + 1])
            if name is not None and name not in out:
                out.append(name)
    return out


def tables_from_ast(node: Any, found: Optional[List[int]] = None) -> List[int]:
    """Mọi ["table_unit", <chỉ số>] trong cây `sql`, kể cả trong subquery.

    Duyệt đệ quy vì bảng gold không chỉ nằm ở FROM ngoài cùng mà còn ở subquery
    trong WHERE/HAVING và ở các nhánh INTERSECT/UNION/EXCEPT.
    """
    if found is None:
        found = []

    if isinstance(node, dict):
        for value in node.values():
            tables_from_ast(node=value, found=found)
    elif isinstance(node, list):
        if len(node) == 2 and node[0] == "table_unit" and isinstance(node[1], int):
            if node[1] not in found:
                found.append(node[1])
            return found
        for item in node:
            tables_from_ast(node=item, found=found)

    return found


def gold_tables(row: Dict[str, Any], db: Dict[str, Any]) -> List[str]:
    """Tên các bảng gold của một câu hỏi = HỢP của hai cách suy ra.

    Quy tắc "mọi bảng trong FROM/JOIN, kể cả subquery" đã đối chiếu với
    dataset/spider/dev.json: khớp tập bảng 658/658 câu.

    Phải dùng CẢ HAI nguồn vì mỗi nguồn thiếu một kiểu:
      - Cây `sql` của ViText2SQL BỎ SÓT bảng thứ ba ở câu JOIN 3 bảng (44/954 câu
        của split dev) — lấy mình nó thì gold thiếu bảng và recall đo ra sai.
      - Token chỉ thấy bảng viết thẳng sau FROM/JOIN, không thấy bảng nằm trong
        subquery ở mệnh đề FROM.
    """
    names: List[str] = list(db["table_names"])
    out: List[str] = tables_from_tokens(query_toks=row["query_toks"], table_names=names)

    for idx in tables_from_ast(node=row["sql"]):
        if idx < len(names) and names[idx] not in out:
            out.append(names[idx])
    return out


def convert_split(
    rows: List[Dict[str, Any]], dbs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        db: Dict[str, Any] = dbs[row["db_id"]]
        # schema[j] ứng với table_names[j] — schema_strings() dựng theo đúng thứ tự đó.
        by_name: Dict[str, str] = dict(zip(db["table_names"], db["schema"]))

        out.append({
            "id": i,
            "db_id": row["db_id"],
            "utterance": row["question"],
            "query": row["query"],
            "rel_schema": [by_name[n] for n in gold_tables(row=row, db=db)],
        })
    return out


# ---------------------------------------------------------------------------
# Prompt few-shot của pha Removal
# ---------------------------------------------------------------------------
# Khung prompt GIỮ NGUYÊN TIẾNG ANH, chỉ nội dung ví dụ là tiếng Việt. Không phải
# vì lười dịch: nhãn "Completing Tables:" và câu trả lời "None" chính là thứ
# core/rewriter.py dò để biết một nhánh đã đủ bảng (_EARLY_STOP_INDICATORS,
# _ECHO_PREFIXES). Dịch nhãn sang tiếng Việt thì Early Stop không bao giờ khớp:
# mọi nhánh chạy đủ max_hop, số lần gọi LLM tăng gấp đôi mà kết quả kém đi.
_PROMPT_HEADER: str = (
    "Given the following SQL tables, your job is to complete the possible left SQL "
    "tables given a user’s request.\n"
    "Return None if no left SQL tables according to the user’s request.\n"
)
_PROMPT_FOOTER: str = "Question: {question}\nDatabase: {database}\nCompleting Tables: "

# 6 ví dụ có bảng còn thiếu (C) / 3 ví dụ None (N), xen kẽ — cùng tỉ lệ và cùng bố
# cục với prompts/spider_rewrite.txt để hai prompt so sánh được với nhau.
_PROMPT_LAYOUT: str = "CNCCNCNCC"

# Chặn ví dụ quá ngắn (không đủ ngữ cảnh) và quá dài (prompt phình, model yếu lạc).
_QUESTION_LEN: Tuple[int, int] = (25, 90)
_MAX_SCHEMA_CHARS: int = 220


def _candidates(
    rows: List[Dict[str, Any]], dbs: Dict[str, Dict[str, Any]], n_tables: int,
) -> Dict[str, Tuple[str, List[str]]]:
    """Ứng viên làm ví dụ, MỖI DATABASE nhiều nhất một câu (ngắn nhất mà đạt yêu cầu).

    Sắp xếp tất định nên chạy lại cho ra prompt y hệt.
    """
    best: Dict[str, Tuple[str, List[str]]] = {}
    for row in sorted(rows, key=lambda r: (r["db_id"], len(r["question"]), r["question"])):
        db_id: str = row["db_id"]
        if db_id in best:
            continue

        question: str = row["question"]
        if not _QUESTION_LEN[0] <= len(question) <= _QUESTION_LEN[1]:
            continue

        db: Dict[str, Any] = dbs[db_id]
        names: List[str] = gold_tables(row=row, db=db)
        if len(names) != n_tables:
            continue

        by_name: Dict[str, str] = dict(zip(db["table_names"], db["schema"]))
        schemas: List[str] = [by_name[n] for n in names]
        if sum(len(s) for s in schemas) > _MAX_SCHEMA_CHARS:
            continue

        best[db_id] = (question, schemas)
    return best


def _spread(items: List[str], count: int) -> List[str]:
    """`count` phần tử RẢI ĐỀU trong danh sách, không phải `count` phần tử đầu.

    Chọn theo thứ tự bảng chữ cái thì cả 6 ví dụ đều rơi vào database vần A —
    prompt nhìn như chỉ có mỗi một miền dữ liệu.
    """
    if len(items) <= count:
        return items
    step: float = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def build_prompt(train: List[Dict[str, Any]], dbs: Dict[str, Dict[str, Any]]) -> str:
    """Dựng prompt few-shot cho pha Removal từ split TRAIN.

    Lấy từ train chứ không phải dev: ví dụ trong prompt mà trùng câu đang đo thì
    điểm đo được là điểm của trí nhớ, không phải của retrieval. (Prompt tiếng Anh
    gốc của paper lấy ví dụ từ chính các database của dev — ở đây không làm vậy.)

    Mỗi ví dụ mô phỏng đúng việc Removal phải làm: cho câu hỏi + MỘT bảng đã tìm
    được, nói ra bảng còn thiếu; hoặc trả None nếu bảng đó đã đủ.
    """
    pairs: Dict[str, Tuple[str, List[str]]] = _candidates(rows=train, dbs=dbs, n_tables=2)
    singles: Dict[str, Tuple[str, List[str]]] = _candidates(rows=train, dbs=dbs, n_tables=1)

    n_pairs: int = _PROMPT_LAYOUT.count("C")
    n_singles: int = _PROMPT_LAYOUT.count("N")

    chosen_pairs: List[str] = _spread(sorted(pairs), n_pairs)
    # Một database không vừa làm ví dụ "có bảng thiếu" vừa làm ví dụ "None": model
    # dễ học nhầm rằng câu trả lời phụ thuộc database chứ không phụ thuộc câu hỏi.
    chosen_singles: List[str] = _spread(
        [d for d in sorted(singles) if d not in chosen_pairs], n_singles,
    )
    if len(chosen_pairs) < n_pairs or len(chosen_singles) < n_singles:
        raise SystemExit(
            f"Không đủ ví dụ trong train để dựng prompt: cần {n_pairs} câu 2 bảng "
            f"và {n_singles} câu 1 bảng, mới có {len(chosen_pairs)} và {len(chosen_singles)}."
        )

    blocks: List[str] = []
    for kind in _PROMPT_LAYOUT:
        if kind == "C":
            question, schemas = pairs[chosen_pairs.pop(0)]
            given: List[str] = [schemas[0]]
            answer: str = schemas[1]
        else:
            question, given = singles[chosen_singles.pop(0)]
            answer = "None"

        # Dấu ngăn là 2 KÝ TỰ "\" và "n" viết ra, không phải xuống dòng — cả khối
        # Database nằm trên một dòng. core/rewriter.py ghép đúng như vậy lúc chạy.
        database: str = " \\n ".join(given)
        blocks.append(
            f"Question: {question}\nDatabase: {database}\nCompleting Tables: {answer}"
        )

    return _PROMPT_HEADER + "\n" + "\n\n".join(blocks) + "\n\n" + _PROMPT_FOOTER


# ---------------------------------------------------------------------------
# Kiểm tra trước khi ghi
# ---------------------------------------------------------------------------
def check(rows: List[Dict[str, Any]], corpus: Sequence[str]) -> None:
    """Chặn những hỏng hóc âm thầm làm recall sai mà không báo lỗi gì."""
    problems: List[str] = []
    known: Set[str] = set(corpus)

    empty: int = sum(1 for r in rows if not r["rel_schema"])
    if empty:
        problems.append(f"{empty} câu không suy ra được bảng gold nào")

    unknown: List[str] = [s for r in rows for s in r["rel_schema"] if s not in known]
    if unknown:
        problems.append(
            f"{len(unknown)} bảng gold không có trong corpus, ví dụ: {unknown[0]!r}"
        )

    blank: int = sum(1 for r in rows if not r["utterance"].strip())
    if blank:
        problems.append(f"{blank} câu hỏi rỗng")

    if len(known) != len(corpus):
        problems.append(f"corpus có {len(corpus) - len(known)} schema trùng nhau")

    if problems:
        raise SystemExit("Dữ liệu chuyển đổi không hợp lệ:\n  - " + "\n  - ".join(problems))


def describe(rows: List[Dict[str, Any]], corpus: Sequence[str]) -> None:
    sizes: List[int] = [len(r["rel_schema"]) for r in rows]
    multi: int = sum(1 for n in sizes if n > 1)
    print(
        f"  {len(rows)} câu hỏi | {len(corpus)} schema | "
        f"{multi} câu cần nhiều bảng ({100 * multi / len(rows):.0f}%) | "
        f"nhiều nhất {max(sizes)} bảng"
    )


# ---------------------------------------------------------------------------
def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  → {path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--level", choices=LEVELS, default="syllable")
    parser.add_argument(
        "--split", choices=SPLITS, default="dev",
        help="split của ViText2SQL dùng làm tập đánh giá (ghi ra dev.json)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="thư mục đích trong project")
    parser.add_argument(
        "--src", default=None,
        help="thư mục đã có sẵn tables.json/<split>.json; bỏ trống thì tải từ GitHub",
    )
    parser.add_argument(
        "--emit-prompt", action="store_true",
        help=f"dựng lại {DEFAULT_PROMPT} từ split train (phải đọc thêm ~27MB)",
    )
    parser.add_argument("--prompt-out", default=DEFAULT_PROMPT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"ViText2SQL {args.level}-level, split={args.split}")

    tables: List[Dict[str, Any]] = convert_tables(
        tables=load_source(name="tables.json", level=args.level, src=args.src)
    )
    dbs: Dict[str, Dict[str, Any]] = {db["db_id"]: db for db in tables}

    rows_raw: List[Dict[str, Any]] = load_source(
        name=f"{args.split}.json", level=args.level, src=args.src
    )
    missing: Set[str] = {r["db_id"] for r in rows_raw} - set(dbs)
    if missing:
        raise SystemExit(
            f"{len(missing)} db_id có trong split nhưng không có trong tables.json: "
            f"{sorted(missing)[:5]}"
        )

    rows: List[Dict[str, Any]] = convert_split(rows=rows_raw, dbs=dbs)
    corpus: List[str] = [s for db in tables for s in db["schema"]]

    check(rows=rows, corpus=corpus)
    describe(rows=rows, corpus=corpus)

    write_json(path=os.path.join(args.out, "tables.json"), data=tables)
    write_json(path=os.path.join(args.out, "dev.json"), data=rows)

    # gold.txt: cùng định dạng "<query>\t<db_id>" với spider/bird, để chấm execution
    # accuracy bằng công cụ ngoài.
    write_text(
        path=os.path.join(args.out, "gold.txt"),
        text="\n\n".join(f"{r['query']}\t{r['db_id']}" for r in rows),
    )

    # Ghi lại mức tách từ đang nằm trên đĩa: hai mức có CÙNG số schema nên nhìn file
    # không phân biệt được, mà chạy lẫn thì kết quả sai âm thầm.
    write_json(
        path=os.path.join(args.out, "meta.json"),
        data={
            "source": "https://github.com/VinAIResearch/ViText2SQL",
            "level": args.level,
            "split": args.split,
            "num_questions": len(rows),
            "num_schemas": len(corpus),
        },
    )

    if args.emit_prompt:
        train: List[Dict[str, Any]] = (
            rows_raw if args.split == "train"
            else load_source(name="train.json", level=args.level, src=args.src)
        )
        write_text(path=args.prompt_out, text=build_prompt(train=train, dbs=dbs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
