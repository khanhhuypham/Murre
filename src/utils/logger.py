# =============================================================================
# utils/logger.py — Logger đơn giản cho project MURRE
#
# mọi module import logger từ đây để:
#  - Có timestamp + tên module + mức độ (INFO/WARNING/ERROR) đi kèm mỗi dòng log.
#  - Dễ tắt/bật mức độ log (ví dụ chỉ hiện WARNING trở lên khi chạy batch dài).
#  - Dễ đổi sang ghi log ra file khi cần (chỉ sửa 1 chỗ duy nhất ở đây).
#   Cách dùng ở module khác:
#       from utils import logger
#       logger.info("Đã tải xong model")
#       logger.warning("Không tìm thấy cache, sẽ tính lại từ đầu")
#       logger.error("Gọi LLM thất bại", exc_info=True)
# =============================================================================
from __future__ import annotations
import logging
import os
import sys
from typing import Optional, Any
from config import cfg

_DEFAULT_LEVEL = logging.INFO

def _resolve_level() -> int:
    """Đọc mức log từ cfg.logging.level (LoggingConfig trong src/config.py).

    Vẫn dùng getattr() có mặc định để logger không chết nếu section bị đổi tên.
    """
    section: Optional[Any] = getattr(cfg, "logging", None)
    cfg_level: str = str(getattr(section, "level", "INFO")).upper().strip()
    return getattr(logging, cfg_level, _DEFAULT_LEVEL)

def _resolve_log_file_path() -> Optional[str]:
    """Trả về đường dẫn file log nếu LoggingConfig.log_to_file = True (src/config.py),
    None nếu chỉ in ra console."""
    section: Optional[Any] = getattr(cfg, "logging", None)

    if not bool(getattr(section, "log_to_file", False)):
        return None

    log_dir: str = getattr(section, "log_dir", "outputs/logs")
    log_file: str = getattr(section, "log_file", "murre.log")
    return os.path.join(log_dir, log_file)

def _build_logger(name: str = "murre") -> logging.Logger:
    log: logging.Logger = logging.getLogger(name=name)

    if not log.handlers:  # tránh tạo handler trùng lặp nếu module bị import nhiều lần
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Console Windows mặc định là cp1252 → log tiếng Việt gây UnicodeEncodeError
        # ("charmap codec can't encode character"). Ép stdout sang UTF-8 và thay ký tự
        # không hiển thị được bằng "?" thay vì làm sập dòng log.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(fmt=formatter)
        log.addHandler(hdlr=stream_handler)

        file_path: Optional[str] = _resolve_log_file_path()
        if file_path:
            os.makedirs(name=os.path.dirname(file_path) or ".", exist_ok=True)
            file_handler: logging.FileHandler = logging.FileHandler(
                filename=file_path, mode="a", encoding="utf-8",
            )
            file_handler.setFormatter(fmt=formatter)
            log.addHandler(hdlr=file_handler)

        log.propagate = False

    log.setLevel(level=_resolve_level())

    return log

# Logger mặc định dùng chung toàn project — import trực tiếp các hàm này.
_logger: logging.Logger = _build_logger(name="murre")

def info(msg: str, *args: object, **kwargs: object) -> None:
    _logger.info(msg, *args, **kwargs)

def warning(msg: str, *args: object, **kwargs: object) -> None:
    _logger.warning(msg, *args, **kwargs)

def error(msg: str, *args: object, **kwargs: object) -> None:
    _logger.error(msg, *args, **kwargs)

def debug(msg: str, *args: object, **kwargs: object) -> None:
    _logger.debug(msg, *args, **kwargs)

def get_logger(name: str) -> logging.Logger:
    """Dùng khi muốn 1 logger riêng cho từng module (tên hiện rõ trong mỗi dòng log),
    ví dụ: log = get_logger(name=__name__)
    """
    return _build_logger(name=name)



