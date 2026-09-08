# =============================================================================
# dataset/loader.py — Tải dữ liệu từ file JSON của Spider và BIRD
# =============================================================================
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config import cfg
from utils import logger


def _load_json(path: str) -> Any:
    """Đọc file JSON và trả về đối tượng Python."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tables(dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tải tables.json. dataset=None → dùng dataset đang chọn (general.dataset)."""
    path: str = cfg.dataset_config(dataset).tables
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} databases từ: {path}")
    return data


def load_dev(dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tải dev.json. dataset=None → dùng dataset đang chọn (general.dataset)."""
    path: str = cfg.dataset_config(dataset).dev
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} câu hỏi từ: {path}")
    return data


_DEV_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _dev_cached() -> List[Dict[str, Any]]:
    """load_dev() nhưng nhớ kết quả — resolve_question() và gold_for() hay được gọi
    liền nhau trong 1 lần chạy CLI, không cần đọc & parse dev.json hai lần.

    Cache theo đường dẫn nên đổi general.dataset giữa chừng vẫn ra dữ liệu đúng.
    """
    path: str = cfg.dataset_paths.dev
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
