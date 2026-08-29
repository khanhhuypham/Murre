# =============================================================================
# main.py — Điểm vào duy nhất của project MURRE
#
# File này CHỈ khai báo tham số rồi gọi. Toàn bộ logic chạy pipeline nằm ở
# methods/runner.py (run_offline / run_batch / run_batch_murre) — main.py,
# methods/*.py và api.py đều import từ đó, không nơi nào viết lại trình tự chạy.
#
# KHÔNG dùng tham số dòng lệnh: muốn đổi gì thì sửa thẳng mấy biến HOA trong khối
# __main__ ở cuối file rồi chạy lại — cùng kiểu với methods/murre.py,
# methods/crush.py, methods/single_hop.py.
#
# Chạy cái gì là do run_option.mode trong src/config.py quyết định:
#
#   "offline" (Option 1) → chạy MỘT câu hỏi trong process này, in ra top-N bảng.
#                          Dùng để học thuật toán và debug.
#   "batch"   (Option 2) → chạy TOÀN BỘ dev.json, ghi kết quả ra outputs/.
#
# CÁCH CHẠY:
#   python -m main
# =============================================================================
from __future__ import annotations

from config import cfg
from methods.runner import run_batch, run_offline
from enums import Method
from utils import logger


# =============================================================================
# ĐIỂM CHẠY — sửa trực tiếp mấy biến dưới đây rồi: python -m main
# =============================================================================
if __name__ == "__main__":

    mode: str = cfg.run_option.mode
    method: Method = Method(cfg.pipeline.method)

    logger.info(
        f"[Main] mode={mode} | dataset={cfg.general.dataset} | scale={cfg.general.scale} "
        f"| method={method}"
    )

    # mode là MỘT chuỗi nên 3 nhánh dưới đây loại trừ nhau — đúng chỗ dùng elif.
    if mode == "offline":
        QUESTION: str | None = None  # None → lấy câu đầu tiên trong dev.json
        VERBOSE: bool = False  # in chi tiết từng hop (chỉ có ý nghĩa với murre)
        LLM_PROFILE: str | None = None  # None → dùng llm.active_profile

        run_offline(
            method=method,
            question=QUESTION,
            top_n=5,
            verbose=VERBOSE,
            llm_profile=LLM_PROFILE,
        )
    elif mode == "batch":
        TOP_K: int = 5  # số bảng đưa vào bước sinh SQL
        LIMIT: int | None = None  # chỉ chạy N câu đầu (chỉ single_hop/crush)
        FORCE_EMBED: bool = False  # True → mã hóa lại corpus dù đã có cache .pt

        run_batch(
            method=method, 
            top_k=TOP_K,
            force_embed=FORCE_EMBED,
            limit=LIMIT
        )

    else:
        raise SystemExit(
            f"run_option.mode={mode!r} không hợp lệ — chỉ nhận 'offline' hoặc "
            f"'batch'. Sửa trong RunOptionConfig (src/config.py)."
        )
