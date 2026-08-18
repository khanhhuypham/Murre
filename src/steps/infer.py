"""steps/infer.py — BƯỚC 5 (Option 2/Batch): sinh SQL từ Top-K bảng đã retrieve.

Chạy: python -m steps.infer --top_k 5
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from config import get_path
from core.llm import LLMGenerator
from dataset.loader import load_tables
from utils import logger
from utils.schema import build_db_index, filter_ret_tables_from_db, pack_table

_ZERO_SHOT_PROMPT: str = """{table}

-- Using valid SQLite, answer the following question for the tables provided above.

-- {question}
SELECT"""


def _locate_schema_idx(schema_list: List[str], target_prefix: str) -> int:
    for idx, schema in enumerate(schema_list):
        if target_prefix == schema.split("(")[0]:
            return idx
    return 100


def _build_table_prompt(schema_strings: List[str], dbs_dict: Dict[str, Dict[str, Any]]) -> str:
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
            slot_idx: int = _locate_schema_idx(schema_list=schema_strings, target_prefix=target_prefix)
            if slot_idx < len(tables_input):
                tables_input[slot_idx] = sql_block

    return "\n\n".join(t for t in tables_input if t)


def run_infer(top_k: int) -> None:
    logger.info("=" * 60)
    logger.info(f"  BƯỚC 5: SINH SQL — top_k={top_k}")
    logger.info("=" * 60)

    result_file: str = get_path(key="result")
    with open(result_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    tables: List[Dict[str, Any]] = load_tables()
    dbs_dict: Dict[str, Dict[str, Any]] = build_db_index(tables=tables)

    llm: LLMGenerator = LLMGenerator()

    predicted_sqls: List[str] = []
    prompts_used: List[str] = []

    for d in data:
        top_schemas: List[str] = [r["schema"] for r in d["retrieved"][:top_k]]
        table_block: str = _build_table_prompt(schema_strings=top_schemas, dbs_dict=dbs_dict)

        question: str = (
            d.get("utterance") if isinstance(d.get("utterance"), str)
            else (d.get("utterance_org") or [""])[0] if isinstance(d.get("utterance_org"), list)
            else d.get("question", "")
        )

        prompt: str = _ZERO_SHOT_PROMPT.format(table=table_block, question=question)
        prompts_used.append(prompt)

        sql_output: str = llm.generate(prompt=prompt)
        if not sql_output.lower().startswith("select"):
            sql_output = "select " + sql_output
        sql_output = sql_output.replace("\n", " ").strip()

        predicted_sqls.append(sql_output)

    sql_file: str = get_path(key="sql", k=top_k)
    inp_file: str = os.path.join(os.path.dirname(sql_file), f"inp.{top_k}.txt")
    os.makedirs(os.path.dirname(sql_file), exist_ok=True)

    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(predicted_sqls))
    with open(inp_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(prompts_used))

    logger.info(f"[Infer] top_k={top_k} → đã sinh {len(predicted_sqls)} câu SQL, lưu tại: {sql_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bước Sinh SQL của MURRE")
    parser.add_argument("--top_k", type=int, default=5, help="Số bảng tối đa đưa vào prompt")
    args = parser.parse_args()

    run_infer(top_k=args.top_k)
