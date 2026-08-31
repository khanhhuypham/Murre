# =============================================================================
# main.py — Điểm vào của project MURRE.  Chạy: python -m main
# =============================================================================
from __future__ import annotations

from config import cfg
from methods.runner import run_batch, run_one_question
from enums import Method
from utils import logger



if __name__ == "__main__":
    mode: str = cfg.run_option.mode
    method: Method = Method(cfg.pipeline.method)

    logger.info(
        f"[Main] mode={mode} | dataset={cfg.general.dataset} | model={cfg.encoder.slug} "
        f"| method={method}"
    )

    # Option 1 — MỘT câu hỏi, in top-N ra terminal, không ghi file.
    if mode == "one_question":
        QUESTION: str | None = None  # None → lấy câu đầu tiên trong dev.json
        VERBOSE: bool = False  # in chi tiết từng hop (chỉ có ý nghĩa với murre)
        LLM_PROFILE: str | None = None  # None → dùng llm.active_profile

        run_one_question(
            method=method,
            question=QUESTION,
            top_n=5,
            verbose=VERBOSE,
            llm_profile=LLM_PROFILE,
        )

    # Option 2 — CẢ dev.json, ghi kết quả ra outputs/.
    elif mode == "batch":
        TOP_K: int = 5  # số bảng đưa vào bước sinh SQL (chỉ murre)
        LIMIT: int | None = 2  # chỉ chạy N câu đầu (dùng được với cả 3 method)
        FORCE_EMBED: bool = False  # True → mã hóa lại corpus dù đã có cache .pt

        run_batch(
            method=method,
            top_k=TOP_K,
            force_embed=FORCE_EMBED,
            limit=LIMIT
        )

    else:
        raise SystemExit(
            f"run_option.mode={mode!r} không hợp lệ — chỉ nhận 'one_question' "
            f"hoặc 'batch'. Sửa trong RunOptionConfig (src/config.py)."
        )
