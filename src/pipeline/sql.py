"""pipeline/sql.py — Bước cuối của paper: top-K bảng đã retrieve → câu lệnh SQL.

    build_sql()   MỘT câu hỏi + danh sách schema → SQL. POST /sql dùng hàm này.
    run_infer()   cả file result (do pipeline/runner.py ghi) → sql.{k}.txt.

Hai đường dùng CHUNG hàm dựng prompt (_build_table_prompt) nên không thể lệch nhau.

Chạy riêng: python -m pipeline.sql --top-k 5
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from config import cfg
from core.llm import LLMGenerator
from dataset.loader import load_tables
from models.records import ResultRecord
from utils import logger
from utils.schema import build_db_index, filter_ret_tables_from_db, pack_table

_ZERO_SHOT_PROMPT: str = """{table}

-- Using valid SQLite, answer the following question for the tables provided above.

-- {question}
SELECT"""

# Số token tối đa cho một câu SQL. Mặc định 100 của LLMGenerator cắt cụt câu dài
# (nhiều JOIN + subquery), sinh ra SQL không parse được.
_SQL_MAX_TOKENS: int = 512


def _locate_schema_idx(schema_list: List[str], target_prefix: str) -> int:
    for idx, schema in enumerate(schema_list):
        if target_prefix == schema.split("(")[0]:
            return idx
    return len(schema_list)


def _build_table_prompt(
    schema_strings: List[str], dbs_dict: Dict[str, Dict[str, Any]],
) -> str:
    """Các schema đã xếp hạng → khối CREATE TABLE, giữ nguyên thứ tự xếp hạng."""
    db_tables: Dict[str, List[str]] = {}
    for schema in schema_strings:
        db_id: str = schema.split(".")[0]
        db_tables.setdefault(db_id, []).append(schema)

    tables_input: List[str] = [""] * len(schema_strings)

    for db_id, schemas in db_tables.items():
        db_dict = dbs_dict.get(db_id)
        if db_dict is None:
            continue

        table_names_to_keep: List[str] = [s.split("(")[0].split(".")[1] for s in schemas]
        filtered: Dict[str, Any] = filter_ret_tables_from_db(
            db_dict=db_dict, db_id=db_id, ret_tables_list=table_names_to_keep,
        )
        filtered_table_names: List[str] = filtered["table_names"]
        packed_sql: str = pack_table(db=filtered, use_original=True)

        for part_idx, sql_block in enumerate(packed_sql.split("\n\n")):
            if part_idx >= len(filtered_table_names):
                break
            target_prefix: str = f"{db_id}.{filtered_table_names[part_idx]}"
            slot_idx: int = _locate_schema_idx(
                schema_list=schema_strings, target_prefix=target_prefix,
            )
            if slot_idx < len(tables_input):
                tables_input[slot_idx] = sql_block

    return "\n\n".join(t for t in tables_input if t)


# Model chat gói code trong ```sql ... ``` và viết thêm lời giải thích quanh đó;
# model completion thì trả về đúng phần đuôi của câu. _normalize_sql() lo cả hai.
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"\bselect\b", re.IGNORECASE)


def _normalize_sql(raw: str) -> str:
    """Output thô của LLM → ĐÚNG MỘT dòng SQL, không dấu `;`, bắt đầu bằng SELECT.

    Đây là định dạng của gold.txt, cũng là thứ mọi công cụ chấm execution accuracy
    mong đợi. Không chuẩn hoá thì file sql.{k}.txt dính nguyên khối markdown và lời
    giải thích, không parse được câu nào.
    """
    text: str = raw.strip()

    # 1. Có khối ```...``` thì SQL nằm trong đó, phần ngoài là lời giải thích.
    for block in _FENCE_RE.findall(text):
        if _SELECT_RE.search(block):
            text = block
            break

    # 2. Cắt bỏ phần dẫn trước SELECT. Không tìm thấy SELECT nào nghĩa là model chỉ
    #    trả về phần ĐUÔI (prompt kết thúc bằng "SELECT") → nối từ khoá vào.
    match = _SELECT_RE.search(text)
    text = text[match.start():] if match else f"SELECT {text}"

    # 3. Một dòng, dừng ở dấu `;` đầu tiên — sau đó thường là câu kế hoặc lời bàn.
    sql: str = " ".join(text.split())
    return sql.split(";", 1)[0].strip()


def build_sql(
    llm: LLMGenerator,
    question: str,
    schemas: Sequence[str],
    dataset: Optional[str] = None,
) -> str:
    """Prompt zero-shot → SQL, cho MỘT câu hỏi.

    dataset=None → tables.json của general.dataset. Truyền tường minh khi phục vụ
    nhiều dataset trong một process (API): schema phải tra trong đúng dataset đã
    retrieve, không phải dataset mặc định của config.
    """
    if not schemas:
        raise ValueError("Không có bảng nào để sinh SQL.")

    dbs: Dict[str, Dict[str, Any]] = build_db_index(tables=load_tables(dataset=dataset))
    prompt: str = _ZERO_SHOT_PROMPT.format(
        table=_build_table_prompt(schema_strings=list(schemas), dbs_dict=dbs),
        question=question,
    )
    return _normalize_sql(raw=llm.generate(prompt=prompt, max_tokens=_SQL_MAX_TOKENS))


def run_infer(top_k: int, llm: LLMGenerator | None = None) -> str:
    """Sinh SQL cho MỌI câu trong file result của lần chạy hiện tại.

    Trả về đường dẫn file sql.{k}.txt vừa ghi.
    """
    logger.info(f"[Infer] Sinh SQL — top_k={top_k}")

    result_file: str = cfg.outputs.result()
    with open(result_file, "r", encoding="utf-8") as f:
        data: List[ResultRecord] = ResultRecord.from_list(items=json.load(f))

    dbs_dict: Dict[str, Dict[str, Any]] = build_db_index(tables=load_tables())
    generator: LLMGenerator = llm if llm is not None else LLMGenerator()

    predicted_sqls: List[str] = []
    for d in data:
        table_block: str = _build_table_prompt(
            schema_strings=d.schemas[:top_k], dbs_dict=dbs_dict,
        )
        prompt: str = _ZERO_SHOT_PROMPT.format(table=table_block, question=d.utterance)
        predicted_sqls.append(
            _normalize_sql(raw=generator.generate(prompt=prompt, max_tokens=_SQL_MAX_TOKENS))
        )

    sql_file: str = cfg.outputs.sql(k=top_k)
    os.makedirs(os.path.dirname(sql_file), exist_ok=True)
    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(predicted_sqls))

    logger.info(f"[Infer] Đã sinh {len(predicted_sqls)} câu SQL → {sql_file}")
    return sql_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sinh SQL từ file result đã có")
    parser.add_argument("--top-k", type=int, default=cfg.pipeline.top_k_output)
    run_infer(top_k=parser.parse_args().top_k)
