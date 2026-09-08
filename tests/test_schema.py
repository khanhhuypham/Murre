"""Xử lý schema: corpus cho retrieval và CREATE TABLE cho bước sinh SQL."""
from __future__ import annotations

from typing import Any, Dict

from utils.schema import build_db_index, build_schema_corpus, filter_ret_tables_from_db, pack_table

# Một database 2 bảng, đúng định dạng tables.json của Spider.
DB: Dict[str, Any] = {
    "db_id": "shop",
    "schema": ["shop.item(id, name)", "shop.sale(id, item_id)"],
    "table_names": ["item", "sale"],
    "table_names_original": ["item", "sale"],
    "column_names": [[-1, "*"], [0, "id"], [0, "name"], [1, "id"], [1, "item id"]],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [1, "id"], [1, "item_id"]],
    "column_types": ["text", "number", "text", "number", "number"],
    "primary_keys": [1, 3],
    "foreign_keys": [[4, 1]],
}


def test_build_schema_corpus_flattens_every_table_of_every_db() -> None:
    corpus = build_schema_corpus(tables=[DB, DB])
    assert corpus == DB["schema"] * 2


def test_build_db_index_keys_by_db_id() -> None:
    assert build_db_index(tables=[DB]) == {"shop": DB}


def test_filter_keeps_only_requested_tables() -> None:
    out = filter_ret_tables_from_db(db_dict=DB, db_id="shop", ret_tables_list=["item"])
    assert out["table_names"] == ["item"]
    # Cột của bảng "sale" biến mất, cột ảo "*" ở lại.
    assert out["column_names_original"] == [[-1, "*"], [0, "id"], [0, "name"]]
    # Khoá ngoại trỏ sang bảng đã bị lọc thì bỏ, không để lại chỉ số treo.
    assert out["foreign_keys"] == []


def test_filter_reindexes_primary_keys() -> None:
    out = filter_ret_tables_from_db(db_dict=DB, db_id="shop", ret_tables_list=["sale"])
    # pk của "sale" là cột 3 trong DB gốc; sau khi lọc chỉ còn [*, id, item_id] → 1.
    assert out["primary_keys"] == [1]


def test_filter_does_not_mutate_the_source_db() -> None:
    before = DB["table_names"][:]
    filter_ret_tables_from_db(db_dict=DB, db_id="shop", ret_tables_list=["item"])
    assert DB["table_names"] == before


def test_pack_table_renders_create_table_with_keys() -> None:
    sql = pack_table(db=DB, use_original=True)
    assert "create table item" in sql
    assert "primary key (id)" in sql
    assert "foreign key (item_id) references item(id)" in sql
    # Hai bảng cách nhau đúng 2 dòng trống — pipeline/sql.py tách theo dấu này.
    assert len(sql.split("\n\n")) == len(DB["table_names"])
