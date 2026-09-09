"""Đọc ViText2SQL THÔ theo cách MURRE hiểu được — không đụng file trên đĩa.

Hai chỗ dễ sai âm thầm và đều làm recall đo ra sai, nên test bám chặt vào chúng:
  - dựng chuỗi schema "db.bảng(cột, ...)" từ table_names + column_names
  - suy ra bảng gold, biết rằng cây `sql` của ViText2SQL BỎ SÓT bảng ở JOIN 3 bảng
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dataset.vitext2sql import (
    adapt_split,
    adapt_tables,
    gold_tables,
    schema_strings,
    tables_from_ast,
    tables_from_tokens,
)

# Một database tiếng Việt 2 bảng, đúng định dạng THÔ của ViText2SQL: không có
# khoá `schema`, tên bảng có dấu cách.
DB_RAW: Dict[str, Any] = {
    "db_id": "architecture",
    "table_names": ["kiến trúc sư", "cầu"],
    "table_names_original": ["kiến trúc sư", "cầu"],
    "column_names": [[-1, "*"], [0, "id"], [0, "tên"], [1, "id"], [1, "id kiến trúc sư"]],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "tên"], [1, "id"], [1, "id kiến trúc sư"]],
    "column_types": ["text", "number", "text", "number", "number"],
    "primary_keys": [1, 3],
    "foreign_keys": [[4, 1]],
}


@pytest.fixture
def tables() -> List[Dict[str, Any]]:
    return adapt_tables(raw=[DB_RAW])


# --- tables.json -----------------------------------------------------------
def test_schema_string_has_db_prefix_table_and_columns() -> None:
    assert schema_strings(db=DB_RAW) == [
        "architecture.kiến trúc sư(id, tên)",
        "architecture.cầu(id, id kiến trúc sư)",
    ]


def test_star_column_is_not_part_of_any_table() -> None:
    """Cột ảo "*" có table_idx = -1; lọt vào schema là làm nhiễu vector."""
    assert "*" not in " ".join(schema_strings(db=DB_RAW))


def test_adapt_tables_keeps_the_original_keys(tables) -> None:
    assert set(DB_RAW).issubset(tables[0])
    assert tables[0]["schema"] == schema_strings(db=DB_RAW)


def test_adapt_tables_does_not_mutate_the_raw_dict() -> None:
    """File trên đĩa giữ nguyên xi thì object đọc lên cũng không được sửa tại chỗ."""
    adapt_tables(raw=[DB_RAW])
    assert "schema" not in DB_RAW


def test_schema_order_matches_table_names_order(tables) -> None:
    """adapt_split ghép schema với tên bảng theo vị trí — lệch là gán nhầm bảng."""
    for name, schema in zip(tables[0]["table_names"], tables[0]["schema"]):
        assert schema.startswith(f"architecture.{name}(")


# --- suy ra bảng gold ------------------------------------------------------
def test_tokens_match_multi_syllable_table_names() -> None:
    """Tên bảng tiếng Việt có dấu cách nhưng vẫn là MỘT token trong query_toks."""
    toks = ["select", "*", "from", "kiến trúc sư", "where", "id", "=", "1"]
    assert tables_from_tokens(query_toks=toks, table_names=DB_RAW["table_names"]) == [
        "kiến trúc sư"
    ]


def test_tokens_collect_every_join_in_order_without_duplicates() -> None:
    toks = ["select", "*", "from", "cầu", "join", "kiến trúc sư", "join", "cầu"]
    assert tables_from_tokens(query_toks=toks, table_names=DB_RAW["table_names"]) == [
        "cầu", "kiến trúc sư",
    ]


def test_tokens_ignore_a_subquery_after_from() -> None:
    """FROM mở subquery thì token kế là "(" — bảng bên trong do AST nhặt."""
    toks = ["select", "*", "from", "(", "select", "*", "from", "cầu", ")"]
    assert tables_from_tokens(query_toks=toks, table_names=DB_RAW["table_names"]) == ["cầu"]


def test_ast_walks_into_nested_queries() -> None:
    """Bảng gold nằm cả trong subquery của WHERE và trong nhánh EXCEPT."""
    ast = {
        "from": {"table_units": [["table_unit", 0]]},
        "where": [[False, 8, [0, [0, 1, False], None], {
            "from": {"table_units": [["table_unit", 1]]},
        }, None]],
        "except": None,
    }
    assert tables_from_ast(node=ast) == [0, 1]


def test_ast_ignores_a_sql_typed_table_unit() -> None:
    """["sql", {...}] là subquery chứ không phải bảng — chỉ ["table_unit", int] mới là."""
    ast = {"from": {"table_units": [["sql", {"from": {"table_units": [["table_unit", 1]]}}]]}}
    assert tables_from_ast(node=ast) == [1]


def test_gold_tables_recovers_the_table_the_ast_dropped(tables) -> None:
    """Ca thật của ViText2SQL: JOIN 3 bảng nhưng cây `sql` chỉ liệt kê 2.

    Chỉ tin cây `sql` là gold thiếu bảng — 44/954 câu dev dính lỗi này, và mỗi câu
    thiếu bảng làm complete_recall bị nới lỏng.
    """
    row = {
        "query_toks": ["select", "*", "from", "cầu", "join", "kiến trúc sư"],
        "sql": {"from": {"table_units": [["table_unit", 1]]}},   # chỉ có "cầu"
    }
    assert gold_tables(row=row, db=tables[0]) == ["cầu", "kiến trúc sư"]


def test_gold_tables_adds_ast_only_tables_after_the_token_ones(tables) -> None:
    row = {
        "query_toks": ["select", "*", "from", "cầu"],
        "sql": {"from": {"table_units": [["table_unit", 0]]}},   # "kiến trúc sư"
    }
    assert gold_tables(row=row, db=tables[0]) == ["cầu", "kiến trúc sư"]


# --- split thô → record MURRE ----------------------------------------------
def _row(question: str = "Có bao nhiêu cây cầu ?") -> Dict[str, Any]:
    return {
        "db_id": "architecture",
        "question": question,
        "query": "select count ( * ) from cầu",
        "query_toks": ["select", "count", "(", "*", ")", "from", "cầu"],
        "sql": {"from": {"table_units": [["table_unit", 1]]}},
    }


def test_adapt_split_produces_the_keys_murre_reads(tables) -> None:
    out = adapt_split(raw=[_row()], tables=tables)
    assert list(out[0]) == ["id", "db_id", "utterance", "query", "rel_schema"]


def test_question_becomes_utterance_and_gold_becomes_rel_schema(tables) -> None:
    out = adapt_split(raw=[_row()], tables=tables)[0]
    assert out["utterance"] == "Có bao nhiêu cây cầu ?"
    assert out["rel_schema"] == ["architecture.cầu(id, id kiến trúc sư)"]


def test_ids_are_sequential(tables) -> None:
    out = adapt_split(raw=[_row("a"), _row("b"), _row("c")], tables=tables)
    assert [r["id"] for r in out] == [0, 1, 2]


def test_mixing_word_and_syllable_level_fails_loudly(tables) -> None:
    """Hai mức có cùng số schema — ghép nhầm thì phải báo, không được chạy tiếp."""
    row = {**_row(), "db_id": "khong_co_trong_tables"}
    with pytest.raises(ValueError, match="mức tách từ"):
        adapt_split(raw=[row], tables=tables)
