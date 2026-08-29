# =============================================================================
# main.py — Điểm vào duy nhất của project MURRE
#
# KHÔNG dùng tham số dòng lệnh: muốn đổi gì thì sửa thẳng mấy biến HOA trong khối
# __main__ ở cuối file rồi chạy lại — cùng kiểu với methods/murre.py,
# methods/crush.py, methods/single_hop.py.
#
# Chạy cái gì là do MODE quyết định (None → lấy theo run_option.mode trong
# src/config.py):
#
#   "offline" (Option 1) → chạy MỘT câu hỏi trong process này, in ra top-N bảng.
#                          Dùng để học thuật toán và debug.
#   "batch"   (Option 2) → chạy TOÀN BỘ dev.json, ghi kết quả ra outputs/.
#
# CÁCH CHẠY:
#   python -m main
#
# File này CHỈ để chạy pipeline. Muốn XEM cấu hình thì dùng config.py:
#   python -m config           # toàn bộ cấu hình đang hiệu lực
#   python -m config --paths   # đường dẫn đã resolve theo cfg
#   python -m config --llm     # danh sách profile LLM
#
# Xem HUONG_DAN.md mục 4 (Option 1) và mục 5 (Option 2).
# =============================================================================
from __future__ import annotations

import os
from typing import List, Optional

from config import cfg
from core.corpus import prepare
from core.factory import build_retriever
from core.llm import LLMGenerator
from dataset.loader import resolve_question
from enums import Dataset, Method, ModelScale
from utils.display import print_results
from models.retrieval import RetrievedTable
from utils import logger


# ---------------------------------------------------------------------------
# Option 1 — Offline: một câu hỏi, chạy thẳng trong process này
# ---------------------------------------------------------------------------
def run_offline(
    method: Method,
    question: Optional[str] = None,
    top_n: int = 5,
    verbose: bool = False,
    llm_profile: Optional[str] = None,
) -> None:
    """
    Chạy đúng một câu hỏi qua `method` rồi in bảng kết quả.
    question=None → lấy câu đầu tiên trong dev.json (xem resolve_question).
    """
    from core.runner import override_cfg

    q: str = resolve_question(question=question)

    # build_retriever() đọc cfg.pipeline.method ngầm chứ không nhận tham số, nên
    # đồng bộ cfg với `method` trong lúc chạy rồi trả về nguyên trạng.
    with override_cfg(method=method):
        encoder, corpus, embs = prepare()
        llm: Optional[LLMGenerator] = (
            LLMGenerator(profile=llm_profile) if method.needs_llm else None
        )
        retriever = build_retriever(encoder=encoder, llm=llm)

        results: List[RetrievedTable] = retriever.run(
            question=q, corpus=corpus, schema_embeddings=embs, verbose=verbose,
        )

    print_results(method=str(method), question=q, results=results, top_n=top_n)


# ---------------------------------------------------------------------------
# Option 2 — Batch: cả dev.json
# ---------------------------------------------------------------------------
def run_batch_murre(top_k: int = 5, force_embed: bool = False) -> None:
    """Chuỗi 5 bước của MURRE, đúng thứ tự trong HUONG_DAN.md mục 5b.

    Tương đương gõ tay lần lượt: steps.embed → steps.retrieve --hop 0 →
    (steps.rewrite --hop N-1 → steps.retrieve --hop N --beam 0..B-1) × max_hop →
    steps.score → steps.infer.

    Chạy tay từng lệnh vẫn hữu ích khi debug (xem được output trung gian, chạy lại
    đúng một bước bị lỗi); hàm này chỉ để khỏi phải gõ 22 lệnh.
    """
    from steps.embed import run_embed
    from steps.infer import run_infer
    from steps.retrieve import run_retrieve
    from steps.rewrite import run_rewrite
    from steps.score import run_score

    beam_size: int = cfg.pipeline.beam_size
    max_hop: int = cfg.pipeline.max_hop

    # --- Bước 1: mã hóa corpus ------------------------------------------------
    # Chỉ cần chạy lại khi đổi dataset hoặc đổi encoder, nên mặc định bỏ qua nếu
    # file đã có — encode lại cả corpus tốn hàng phút mà không được gì.
    emb_file: str = cfg.outputs.embeddings_cache()
    if force_embed or not os.path.exists(emb_file):
        run_embed()
    else:
        logger.info(f"[Main] Đã có {emb_file} → bỏ qua bước embed (FORCE_EMBED=True để ép chạy lại).")

    # --- Bước 2: retrieve hop 0 trên toàn bộ corpus ---------------------------
    run_retrieve(hop=0)

    # --- Bước 3+4: xen kẽ rewrite / retrieve cho từng hop ---------------------
    # rewrite chạy ở hop 0..max_hop-1, retrieve chạy ở hop 0..max_hop.
    for hop in range(1, max_hop + 1):
        run_rewrite(hop=hop - 1)
        for beam in range(beam_size):
            run_retrieve(hop=hop, beam=beam)

    # --- Bước 5: tổng hợp điểm rồi sinh SQL -----------------------------------
    run_score()
    run_infer(top_k=top_k)


def run_batch(
    method: Method,
    top_k: int = 5,
    force_embed: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    Chạy cả dev.json bằng đường phù hợp với `method`.
    murre               → chuỗi steps/*.py (có cả bước sinh SQL).
    single_hop / crush  → core/runner.py, vì steps/*.py chỉ ghép multi-hop cho murre.

    `method` truyền vào từ chỗ gọi, KHÔNG đọc cfg.pipeline.method ở đây — nhờ vậy
    chạy nhiều method liên tiếp trong một script mà không phải sửa config.
    """
    dataset: Dataset = Dataset(cfg.general.dataset)
    scale: ModelScale = ModelScale(cfg.general.scale)

    match method:
        case Method.MURRE:
            if limit is not None:
                raise SystemExit(
                    "LIMIT chỉ dùng được với method single_hop/crush (đường core/runner.py).\n"
                    "  Muốn chạy thử nhanh MURRE thì giảm beam_size/max_hop trong "
                    "PipelineConfig (src/config.py) — xem HUONG_DAN.md mục 5b."
                )

            # steps/*.py điền {method} vào đường dẫn output theo cfg.pipeline.method
            # (xem PathsConfig), nên phải đồng bộ cfg với tham số `method` trong lúc
            # chạy — nếu không, kết quả MURRE sẽ ghi nhầm vào thư mục method khác.
            from core.runner import override_cfg

            logger.info("[Main] Batch MURRE — chạy chuỗi steps/*.py")
            with override_cfg(method=method):
                run_batch_murre(top_k=top_k, force_embed=force_embed)

        case Method.SINGLE_HOP | Method.CRUSH:
            # 2 baseline không có multi-hop nên không dùng chuỗi steps/ được.
            # run_pipeline tự gọi override_cfg bên trong nên không cần bọc lại.
            from core.runner import run_pipeline

            logger.info(f"[Main] Batch {method} — chạy qua core/runner.py")
            summary = run_pipeline(
                dataset=dataset, scale=scale, method=method, limit=limit,
            )
            logger.info(
                f"[Main] Xong {summary['num_questions']} câu. "
                f"result → {summary['result_file']} | score → {summary['score_file']}"
            )

        case _:
            raise SystemExit(f"Method mới chưa xử lý trong run_batch: {method}")


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
