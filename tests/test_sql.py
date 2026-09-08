"""Chuẩn hoá output của LLM thành SQL dùng được.

Prompt kết thúc bằng "SELECT" nên model completion trả về phần đuôi, còn model
chat (qwen2.5-coder, gpt-4o...) trả về nguyên khối markdown kèm lời giải thích.
File sql.{k}.txt phải ra một dòng SQL cho cả hai trường hợp.
"""
from __future__ import annotations

import pytest

from pipeline.sql import _normalize_sql


def test_completion_style_tail_gets_the_keyword_back() -> None:
    assert _normalize_sql(raw="name, country FROM singer") == (
        "SELECT name, country FROM singer"
    )


def test_plain_statement_passes_through() -> None:
    assert _normalize_sql(raw="SELECT name FROM singer") == "SELECT name FROM singer"


def test_markdown_fence_is_unwrapped() -> None:
    raw = "```sql\nSELECT name FROM singer;\n```"
    assert _normalize_sql(raw=raw) == "SELECT name FROM singer"


def test_prose_around_a_fence_is_dropped() -> None:
    raw = (
        "To get the singers you can use:\n\n"
        "```sql\nSELECT name, age FROM singer ORDER BY age DESC;\n```\n\n"
        "This orders them from oldest to youngest."
    )
    assert _normalize_sql(raw=raw) == "SELECT name, age FROM singer ORDER BY age DESC"


def test_prose_without_a_fence_is_cut_at_the_statement() -> None:
    raw = "Here is the query: SELECT name FROM singer"
    assert _normalize_sql(raw=raw) == "SELECT name FROM singer"


def test_fence_without_sql_is_ignored() -> None:
    """Khối ``` chứa kết quả mẫu chứ không phải câu lệnh — bỏ qua, tìm tiếp."""
    raw = "```\nname | age\n```\nSELECT name, age FROM singer"
    assert _normalize_sql(raw=raw) == "SELECT name, age FROM singer"


def test_newlines_collapse_to_one_line() -> None:
    raw = "SELECT name\nFROM singer\nWHERE age > 30"
    assert _normalize_sql(raw=raw) == "SELECT name FROM singer WHERE age > 30"


def test_only_the_first_statement_is_kept() -> None:
    raw = "SELECT a FROM t; SELECT b FROM u;"
    assert _normalize_sql(raw=raw) == "SELECT a FROM t"


@pytest.mark.parametrize("raw", ["SELECT 1", "select 1", "SeLeCt 1"])
def test_keyword_match_is_case_insensitive(raw: str) -> None:
    """Giữ nguyên chữ hoa/thường của model — chỉ nhận diện, không viết lại."""
    assert _normalize_sql(raw=raw) == raw
