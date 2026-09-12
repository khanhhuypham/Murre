# =============================================================================
# dataset/loader.py — Tải dữ liệu của mọi dataset, trả về CÙNG MỘT hình dạng
#
# Dataset khai `format` trong config để nói file trên đĩa đang ở dạng nào:
#
#   murre       đã tiền xử lý sẵn (spider, bird) — đọc thẳng, không đụng gì.
#   vitext2sql  THÔ, y hệt upstream — thích nghi lúc đọc, xem dataset/vitext2sql.py.
#
# Nhờ vậy dữ liệu ViText2SQL trên đĩa giữ nguyên xi bản gốc mà phần còn lại của
# pipeline không cần biết có hai định dạng.
# =============================================================================
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config import cfg
from dataset import vitext2sql
from utils import logger


def _load_json(path: str) -> Any:
    """Đọc file JSON và trả về đối tượng Python."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tables(dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tải tables.json — luôn có khoá `schema`, dù trên đĩa là dạng nào.

    dataset=None → dùng dataset đang chọn (general.dataset).
    """
    spec = cfg.dataset_config(dataset)
    data: List[Dict[str, Any]] = _load_json(path=spec.tables)

    if spec.format == "vitext2sql":
        data = vitext2sql.adapt_tables(raw=data)

    logger.info(f"[Loader] Đã tải {len(data)} databases từ: {spec.tables}")
    return data


def load_dev(dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tải split đánh giá — luôn có `utterance` và `rel_schema`.

    dataset=None → dùng dataset đang chọn (general.dataset).
    """
    spec = cfg.dataset_config(dataset)
    data: List[Dict[str, Any]] = _load_json(path=spec.dev)

    if spec.format == "vitext2sql":
        # Cần tables để suy ra rel_schema; adapt_tables() đã chạy trong load_tables().
        data = vitext2sql.adapt_split(raw=data, tables=load_tables(dataset=dataset))

    logger.info(f"[Loader] Đã tải {len(data)} câu hỏi từ: {spec.dev}")
    return data


_DEV_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _dev_cached() -> List[Dict[str, Any]]:
    """load_dev() nhưng nhớ kết quả — resolve_question() và gold_for() hay được gọi
    liền nhau trong 1 lần chạy CLI, không cần đọc & parse dev.json hai lần.

    Cache theo đường dẫn nên đổi general.dataset giữa chừng vẫn ra dữ liệu đúng.
    """
    path: str = cfg.dataset_config().dev
    if path not in _DEV_CACHE:
        _DEV_CACHE[path] = load_dev()
    return _DEV_CACHE[path]


def resolve_question(question: Optional[str] = None) -> str:
    """Trả về 1 câu hỏi để chạy thử — tự lấy câu đầu dev.json nếu không truyền."""
    if question:
        return question

    data: List[Dict[str, Any]] = _dev_cached()
    if not data:
        raise ValueError('dev.json rỗng — hãy truyền question="..."')
    return data[0].get("utterance", "")


def gold_for(question: str) -> List[str]:
    """Schema đúng (rel_schema) của 1 câu hỏi, nếu nó có trong dev.json — để đối chiếu
    nhanh xem retrieval đúng hay sai. Không tra được thì trả list rỗng."""
    for d in _dev_cached():
        if d.get("utterance", "").strip() == question.strip():
            return d.get("rel_schema", [])
    return []
