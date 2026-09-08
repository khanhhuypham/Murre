"""Chuyển ViText2SQL (định dạng Spider thô) sang định dạng dataset của MURRE.

Hai chỗ dễ sai âm thầm và đều làm recall đo ra sai, nên test bám chặt vào chúng:
  - dựng chuỗi schema "db.bảng(cột, ...)" từ table_names + column_names
  - suy ra bảng gold, biết rằng cây `sql` của ViText2SQL BỎ SÓT bảng ở JOIN 3 bảng
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from prepare_vitext2sql import (
    build_prompt,
    check,
    convert_split,
    convert_tables,
    gold_tables,
    main,
    schema_strings,
    tables_from_ast,
    tables_from_tokens,
)

# Một database tiếng Việt 2 bảng, đúng định dạng thô của ViText2SQL (không có
# khoá `schema`, tên bảng có dấu cách).
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
def dbs() -> Dict[str, Dict[str, Any]]:
    return {db["db_id"]: db for db in convert_tables(tables=[DB_RAW])}


# --- tables.json -----------------------------------------------------------
def test_schema_string_has_db_prefix_table_and_columns() -> None:
    assert schema_strings(db=DB_RAW) == [
        "architecture.kiến trúc sư(id, tên)",
        "architecture.cầu(id, id kiến trúc sư)",
    ]


def test_star_column_is_not_part_of_any_table() -> None:
    """Cột ảo "*" có table_idx = -1; lọt vào schema là làm nhiễu vector."""
    assert "*" not in " ".join(schema_strings(db=DB_RAW))


def test_convert_tables_keeps_the_original_keys() -> None:
    out = convert_tables(tables=[DB_RAW])[0]
    assert set(DB_RAW).issubset(out)
    assert out["schema"] == schema_strings(db=DB_RAW)


def test_schema_order_matches_table_names_order() -> None:
    """convert_split ghép schema với tên bảng theo vị trí — lệch là gán nhầm bảng."""
    out = convert_tables(tables=[DB_RAW])[0]
    for name, schema in zip(out["table_names"], out["schema"]):
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


def test_gold_tables_recovers_the_table_the_ast_dropped() -> None:
    """Ca thật của ViText2SQL: JOIN 3 bảng nhưng cây `sql` chỉ liệt kê 2.

    Chỉ tin cây `sql` là gold thiếu bảng — 44/954 câu dev dính lỗi này, và mỗi câu
    thiếu bảng làm complete_recall bị nới lỏng.
    """
    row = {
        "query_toks": ["select", "*", "from", "cầu", "join", "kiến trúc sư"],
        "sql": {"from": {"table_units": [["table_unit", 1]]}},   # chỉ có "cầu"
    }
    assert gold_tables(row=row, db=convert_tables(tables=[DB_RAW])[0]) == [
        "cầu", "kiến trúc sư",
    ]


def test_gold_tables_adds_ast_only_tables_after_the_token_ones() -> None:
    row = {
        "query_toks": ["select", "*", "from", "cầu"],
        "sql": {"from": {"table_units": [["table_unit", 0]]}},   # "kiến trúc sư"
    }
    assert gold_tables(row=row, db=convert_tables(tables=[DB_RAW])[0]) == [
        "cầu", "kiến trúc sư",
    ]


# --- dev.json --------------------------------------------------------------
def _row(question: str = "Có bao nhiêu cây cầu ?") -> Dict[str, Any]:
    return {
        "db_id": "architecture",
        "question": question,
        "query": "select count ( * ) from cầu",
        "query_toks": ["select", "count", "(", "*", ")", "from", "cầu"],
        "sql": {"from": {"table_units": [["table_unit", 1]]}},
    }


def test_convert_split_produces_the_keys_murre_reads(dbs) -> None:
    out = convert_split(rows=[_row()], dbs=dbs)
    assert list(out[0]) == ["id", "db_id", "utterance", "query", "rel_schema"]


def test_question_becomes_utterance_and_gold_becomes_rel_schema(dbs) -> None:
    out = convert_split(rows=[_row()], dbs=dbs)[0]
    assert out["utterance"] == "Có bao nhiêu cây cầu ?"
    assert out["rel_schema"] == ["architecture.cầu(id, id kiến trúc sư)"]


def test_ids_are_sequential(dbs) -> None:
    out = convert_split(rows=[_row("a"), _row("b"), _row("c")], dbs=dbs)
    assert [r["id"] for r in out] == [0, 1, 2]


# --- kiểm tra trước khi ghi ------------------------------------------------
def test_check_rejects_a_gold_table_missing_from_the_corpus() -> None:
    rows = [{"utterance": "q", "rel_schema": ["db.không có(x)"]}]
    with pytest.raises(SystemExit, match="không có trong corpus"):
        check(rows=rows, corpus=["db.t(x)"])


def test_check_rejects_a_question_with_no_gold_table() -> None:
    rows = [{"utterance": "q", "rel_schema": []}]
    with pytest.raises(SystemExit, match="bảng gold"):
        check(rows=rows, corpus=["db.t(x)"])


def test_check_rejects_a_blank_question() -> None:
    rows = [{"utterance": "   ", "rel_schema": ["db.t(x)"]}]
    with pytest.raises(SystemExit, match="rỗng"):
        check(rows=rows, corpus=["db.t(x)"])


def test_check_accepts_valid_data() -> None:
    check(rows=[{"utterance": "q", "rel_schema": ["db.t(x)"]}], corpus=["db.t(x)"])


# --- prompt Removal --------------------------------------------------------
def _train_rows() -> List[Dict[str, Any]]:
    """Đủ database để dựng prompt: 6 ví dụ 2 bảng + 3 ví dụ None, mỗi db một câu."""
    rows: List[Dict[str, Any]] = []
    for i in range(12):
        rows.append({
            "db_id": f"db{i:02d}",
            "question": f"Câu hỏi thử nghiệm số {i} dài vừa đủ để lọt bộ lọc ?",
            "query": "...",
            "query_toks": ["select", "*", "from", "bảng a", "join", "bảng b"],
            "sql": {"from": {"table_units": []}},
        })
    for i in range(12, 20):
        rows.append({
            "db_id": f"db{i:02d}",
            "question": f"Câu hỏi một bảng số {i} dài vừa đủ để lọt bộ lọc ?",
            "query": "...",
            "query_toks": ["select", "*", "from", "bảng a"],
            "sql": {"from": {"table_units": []}},
        })
    return rows


def _train_dbs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(20):
        db_id = f"db{i:02d}"
        out[db_id] = {
            "db_id": db_id,
            "table_names": ["bảng a", "bảng b"],
            "schema": [f"{db_id}.bảng a(x, y)", f"{db_id}.bảng b(z)"],
        }
    return out


def test_prompt_keeps_the_english_labels_early_stop_matches() -> None:
    """core/rewriter.py dò "Completing Tables:" và "None" — dịch nhãn là hỏng Early Stop."""
    from core.rewriter import QueryRewriter

    prompt = build_prompt(train=_train_rows(), dbs=_train_dbs())
    assert prompt.count("Completing Tables:") == 10   # 9 ví dụ + chỗ trống cuối
    assert QueryRewriter.is_early_stop(rewrite_output="None") is True
    assert "Completing Tables: None" in prompt


def test_prompt_has_the_placeholders_the_rewriter_fills() -> None:
    prompt = build_prompt(train=_train_rows(), dbs=_train_dbs())
    assert prompt.format(question="Q", database="D").endswith(
        "Question: Q\nDatabase: D\nCompleting Tables: "
    )


def test_prompt_examples_come_from_distinct_databases() -> None:
    """Một db vừa dạy "có bảng thiếu" vừa dạy "None" thì model học nhầm theo db."""
    prompt = build_prompt(train=_train_rows(), dbs=_train_dbs())
    used = [ln.split(".")[0].removeprefix("Database: ") for ln in prompt.splitlines()
            if ln.startswith("Database: db")]
    assert len(used) == len(set(used)) == 9


def test_prompt_is_deterministic() -> None:
    rows, dbs_ = _train_rows(), _train_dbs()
    assert build_prompt(train=rows, dbs=dbs_) == build_prompt(train=rows[::-1], dbs=dbs_)


def test_prompt_fails_loudly_when_train_is_too_small() -> None:
    with pytest.raises(SystemExit, match="Không đủ ví dụ"):
        build_prompt(train=_train_rows()[:2], dbs=_train_dbs())


# --- chạy trọn vẹn ---------------------------------------------------------
def test_main_writes_every_file_murre_needs(tmp_path, monkeypatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "tables.json").write_text(json.dumps([DB_RAW]), encoding="utf-8")
    (src / "dev.json").write_text(json.dumps([_row()]), encoding="utf-8")

    out = tmp_path / "out"
    assert main(["--src", str(src), "--out", str(out)]) == 0

    assert {p.name for p in out.iterdir()} == {
        "tables.json", "dev.json", "gold.txt", "meta.json",
    }
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["level"] == "syllable" and meta["num_questions"] == 1
    # gold.txt: "<query>\t<db_id>", cùng định dạng với spider/bird.
    assert (out / "gold.txt").read_text(encoding="utf-8") == (
        "select count ( * ) from cầu\tarchitecture"
    )


def test_main_records_the_level_on_disk(tmp_path) -> None:
    """Hai mức có CÙNG số schema — không ghi lại thì không biết trên đĩa đang là mức nào."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tables.json").write_text(json.dumps([DB_RAW]), encoding="utf-8")
    (src / "dev.json").write_text(json.dumps([_row()]), encoding="utf-8")

    out = tmp_path / "out"
    main(["--src", str(src), "--out", str(out), "--level", "word"])
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["level"] == "word"
