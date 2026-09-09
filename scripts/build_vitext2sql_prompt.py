"""scripts/build_vitext2sql_prompt.py — Dựng prompt few-shot cho pha Removal.

    python scripts/build_vitext2sql_prompt.py                    # từ syllable-level
    python scripts/build_vitext2sql_prompt.py --level word       # từ word-level

Đọc split TRAIN của ViText2SQL (bản THÔ đã tải bằng scripts/download_vitext2sql.py)
rồi ghi ra prompts/vitext2sql_rewrite.txt. KHÔNG sửa gì trong dataset/.

File prompt đã có sẵn trong repo, nên chỉ cần chạy lại khi đổi mức tách từ: prompt
đang commit dùng ví dụ mức syllable, chạy word-level mà để nguyên thì ví dụ trong
prompt viết khác hẳn schema thật mà model đang nhìn.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset.vitext2sql import adapt_tables, gold_tables   # noqa: E402

RAW_DIR: str = "dataset/vitext2sql/data/{level}-level"
DEFAULT_PROMPT: str = "prompts/vitext2sql_rewrite.txt"


def _read(path: str) -> Any:
    if not os.path.exists(path):
        raise SystemExit(
            f"Không thấy {path}.\n"
            f"  Tải dữ liệu trước: python scripts/download_vitext2sql.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt few-shot của pha Removal
# ---------------------------------------------------------------------------
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--level", choices=("syllable", "word"), default="syllable")
    parser.add_argument("--out", default=DEFAULT_PROMPT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    base: str = RAW_DIR.format(level=args.level)
    tables: List[Dict[str, Any]] = adapt_tables(raw=_read(f"{base}/tables.json"))
    train: List[Dict[str, Any]] = _read(f"{base}/train.json")
    dbs: Dict[str, Dict[str, Any]] = {db["db_id"]: db for db in tables}

    print(f"ViText2SQL {args.level}-level — {len(train)} câu train, {len(dbs)} database")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_prompt(train=train, dbs=dbs))
    print(f"  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
