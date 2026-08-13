# =============================================================================
# dataset/loader.py — Tải dữ liệu từ file JSON/TXT của Spider và BIRD
# =============================================================================
import os
import json
from typing import Any, Dict, List
from config import get_dataset_path
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
    path: str = get_dataset_path(key="tables")
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} databases từ: {path}")
    return data


def load_dev() -> List[Dict[str, Any]]:
    """Tải dev.json của dataset đang chọn."""
    path: str = get_dataset_path(key="dev")
    data: List[Dict[str, Any]] = _load_json(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} câu hỏi từ: {path}")
    return data


def load_gold() -> List[str]:
    """Tải gold.txt của dataset đang chọn."""
    path: str = get_dataset_path(key="gold")
    data: List[str] = _load_lines(path=path)
    logger.info(f"[Loader] Đã tải {len(data)} câu SQL vàng từ: {path}")
    return data


# -----------------------------------------------------------------------------
# Hàm tải cụ thể (khi cần chỉ định rõ dataset)
# -----------------------------------------------------------------------------

def load_spider_tables() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/spider/tables.json")


def load_spider_dev() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/spider/dev.json")


def load_bird_tables() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/bird/tables.json")


def load_bird_dev() -> List[Dict[str, Any]]:
    return _load_json(path="dataset/bird/dev.json")



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
