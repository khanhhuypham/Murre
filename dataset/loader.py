# =============================================================================
# dataset/loader.py — Tải dữ liệu từ file JSON/TXT của Spider và BIRD
# =============================================================================
import os
import json
from typing import Any, Dict, List, Optional
from config import cfg
from utils import logger

def _load_json(path: str) -> Any:
    """Đọc file JSON và trả về đối tượng Python."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_lines(path: str) -> List[str]:
    """Đọc file text, trả về danh sách dòng (không có ký tự xuống dòng)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# -----------------------------------------------------------------------------
# Hàm tải theo dataset hiện tại trong config
# -----------------------------------------------------------------------------

def load_tables() -> List[Dict[str, Any]]:
    """Tải tables.json của dataset đang chọn trong config.general.dataset."""
    path: str = cfg.dataset_paths.tables
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} databases từ: {path}")
    return data


def load_dev() -> List[Dict[str, Any]]:
    """Tải dev.json của dataset đang chọn."""
    path: str = cfg.dataset_paths.dev
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} câu hỏi từ: {path}")
    return data


_DEV_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _dev_cached() -> List[Dict[str, Any]]:
    """load_dev() nhưng nhớ kết quả — resolve_question() và gold_for() hay được gọi
    liền nhau trong 1 lần test, không cần đọc & parse dev.json hai lần.

    Cache theo đường dẫn nên đổi general.dataset giữa chừng vẫn ra dữ liệu đúng.
    """
    path: str = cfg.dataset_paths.dev
    if path not in _DEV_CACHE:
        _DEV_CACHE[path] = load_dev()
    return _DEV_CACHE[path]


def resolve_question(question: Optional[str] = None) -> str:
    """Trả về 1 câu hỏi để test — tự lấy câu đầu dev.json nếu không truyền question.

    Dùng cho khối `__main__` của các file trong methods/: chỉ cần set QUESTION=None
    là có ngay câu hỏi mẫu, khỏi phải copy tay từ dev.json.
    """
    if question:
        return question

    data: List[Dict[str, Any]] = _dev_cached()
    if not data:
        raise SystemExit("dev.json rỗng — hãy truyền question=\"...\"")
    return data[0].get("utterance", "")


def gold_for(question: str) -> List[str]:
    """Schema đúng (rel_schema) của 1 câu hỏi, nếu nó có trong dev.json — để đối chiếu
    nhanh xem method retrieve đúng hay sai. Không tra được thì trả list rỗng."""
    for d in _dev_cached():
        if d.get("utterance", "").strip() == question.strip():
            return d.get("rel_schema", [])
    return []


def load_gold() -> List[str]:
    """Tải gold.txt của dataset đang chọn."""
    path: str = cfg.dataset_paths.gold
    data: List[str] = _load_lines(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} câu SQL vàng từ: {path}")
    return data


# -----------------------------------------------------------------------------
# Hàm tải cụ thể (khi cần chỉ định rõ dataset)
# -----------------------------------------------------------------------------

def load_spider_tables() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/spider/tables.json")


def load_bird_tables() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/bird/tables.json")



# =============================================================================
# ĐOẠN CODE CHẠY THỬ TRÊN SPYDER (TEST CELL)
# =============================================================================
if __name__ == '__main__':
    try:
        tables_data: List[Dict[str, Any]] = load_tables()
        dev_data: List[Dict[str, Any]] = load_dev()
        gold_data: List[str] = load_gold()

        if tables_data:
            logger.info("--- [TEST SUCCESS] ---")
            logger.info(f"DB đầu tiên: '{tables_data[0].get('db_id', 'Không rõ')}'")
        else:
            logger.warning("[TEST WARNING] Không tìm thấy dữ liệu trong file json.")

    except Exception as e:
        logger.error(f"--- [TEST ERROR] --- Lỗi: {e} | CWD: {os.getcwd()}")
