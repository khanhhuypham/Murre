"""methods/runner.py — NƠI DUY NHẤT viết cách chạy pipeline.

    run_one_question()  Option 1 — MỘT câu hỏi, in ra terminal, không ghi file.
    run_pipeline()      Option 2 — cả dev.json trong process này, ghi result + score.
                        Đường batch duy nhất cho single_hop/crush; POST /pipeline dùng nó.
    run_batch_murre()   Option 2 — chuỗi 5 bước steps/*.py cho murre (có sinh SQL).
    run_batch()         Option 2 — chọn 1 trong 2 đường trên theo `method`.

`cfg` là biến toàn cục của process, một lần chạy phải tạm ghi đè nó (override_cfg)
nên mỗi lúc chỉ cho phép MỘT lần chạy (_RUN_LOCK). Trong lúc chạy, /retrieve cũng
thấy cfg đã bị ghi đè — muốn cách ly hẳn thì chạy steps/*.py ở process riêng.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from config import cfg
from core.corpus import prepare
from core.llm import LLMGenerator
from dataset.loader import load_dev, resolve_question
from enums import Dataset, Method
from methods.factory import build_retriever
from models.errors import AppError
from models.records import ResultRecord
from models.retrieval import RetrievedTable
from utils import logger
from utils.display import print_results
from utils.metrics import compute_res

_RUN_LOCK: threading.Lock = threading.Lock()

# Callback báo tiến độ: (số câu đã xong, tổng số câu).
ProgressFn = Callable[[int, int], None]


@contextmanager
def override_cfg(
    dataset: Optional[Dataset] = None,
    method: Optional[Method] = None,
) -> Iterator[None]:
    """Tạm ghi đè cfg cho một lần chạy rồi trả lại nguyên trạng. None = giữ nguyên.

    Cần context manager vì vẫn còn chỗ đọc cfg NGẦM: SGPTEncoder() đọc
    encoder.model_name, template trong PathsConfig đọc {dataset} {method}.

    KHÔNG đụng general.scale / encoder.model_name — một lần chạy không tự đổi model
    giữa chừng; muốn scale khác thì sửa config rồi chạy lại.
    """
    saved: Dict[str, Any] = {
        "dataset": cfg.general.dataset,
        "method": cfg.pipeline.method,
    }

    changed: List[str] = []
    if dataset is not None:
        cfg.general.dataset = dataset.value
        changed.append(f"dataset={dataset}")
    if method is not None:
        cfg.pipeline.method = method.value
        changed.append(f"method={method}")

    if changed:
        logger.info(f"[Runner] cfg tạm: {' '.join(changed)}")
    try:
        yield
    finally:
        cfg.general.dataset = saved["dataset"]
        cfg.pipeline.method = saved["method"]
        if changed:
            logger.info("[Runner] Đã trả cfg về nguyên trạng.")


# ---------------------------------------------------------------------------
# Option 1 — one_question
# ---------------------------------------------------------------------------
def run_one_question(
    method: Method,
    question: Optional[str] = None,
    top_n: int = 5,
    verbose: bool = False,
    llm_profile: Optional[str] = None,
    crush_collective: bool = True,
) -> List[RetrievedTable]:
    """Chạy MỘT câu hỏi rồi in bảng kết quả.

        question         : None → câu đầu tiên trong dev.json
        verbose          : in chi tiết từng hop (murre) / bảng LLM đoán (crush)
        llm_profile      : None → dùng llm.active_profile
        crush_collective : chỉ có tác dụng với Method.CRUSH
    """
    q: str = resolve_question(question=question)

    encoder, corpus, embs = prepare()
    llm: Optional[LLMGenerator] = (
        LLMGenerator(profile=llm_profile) if method.needs_llm else None
    )

    retriever = build_retriever(
        encoder=encoder, llm=llm, method=method, crush_collective=crush_collective,
    )
    results: List[RetrievedTable] = retriever.run(
        question=q, corpus=corpus, schema_embeddings=embs, verbose=verbose,
    )

    print_results(method=str(method), question=q, results=results, top_n=top_n)
    return results


# ---------------------------------------------------------------------------
# Option 2 — batch
# ---------------------------------------------------------------------------
def run_pipeline(
    dataset: Dataset,
    method: Method,
    limit: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Chạy pipeline trên dev.json rồi ghi result + score ra đĩa.

        limit       : chỉ chạy N câu đầu (None = cả dev.json) — murre gọi LLM mỗi
                      hop mỗi beam nên chạy đủ 658 câu rất lâu.
        on_progress : callback(đã_xong, tổng).

    Trả về {"result_file", "score_file", "num_questions", "retrieved_depth", "metrics"}.
    Ném AppError 409 nếu đang có lần chạy khác.
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise AppError.pipeline_busy()
    try:
        with override_cfg(dataset=dataset, method=method):
            return _run_locked(method=method, limit=limit, on_progress=on_progress)
    finally:
        _RUN_LOCK.release()


def _run_locked(
    method: Method,
    limit: Optional[int],
    on_progress: Optional[ProgressFn],
) -> Dict[str, Any]:
    """Thân của run_pipeline — đã giữ lock và đã ghi đè cfg."""
    dev: List[Dict[str, Any]] = load_dev()
    if limit is not None:
        dev = dev[:limit]
    total: int = len(dev)
    if total == 0:
        raise ValueError("dev.json rỗng — không có câu hỏi nào để chạy.")

    # prepare(): tables.json → corpus → embeddings (có cache).
    encoder, corpus, embs = prepare()
    embs: torch.Tensor

    llm: Optional[LLMGenerator] = LLMGenerator() if method.needs_llm else None
    retriever = build_retriever(encoder=encoder, llm=llm, method=method)

    logger.info(f"[Runner] Bắt đầu {method} trên {total} câu, corpus {len(corpus)} schemas.")

    output: List[ResultRecord] = []
    for i, d in enumerate(dev, start=1):
        hits: List[RetrievedTable] = retriever.run(
            question=d["utterance"], corpus=corpus, schema_embeddings=embs,
        )
        output.append(ResultRecord(
            utterance=d["utterance"],
            gold=d.get("rel_schema", []),
            # to_rows(): đánh số rank + đổi khóa `score` → `similarity` của file format.
            retrieved=RetrievedTable.to_rows(tables=hits),
        ))
        if on_progress is not None:
            on_progress(i, total)
        if i % 20 == 0 or i == total:
            logger.info(f"[Runner] {i}/{total} câu xong.")

    top_k_list: List[int] = list(cfg.general.top_k)
    metrics: Dict[str, Dict[int, float]] = compute_res(top_k=top_k_list, data=output)

    result_file: str = cfg.outputs.result()
    score_file: str = cfg.outputs.score()
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in output], f, ensure_ascii=False, indent=2)
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    depth: int = min(len(d.retrieved) for d in output)
    logger.info(f"[Runner] Xong. result → {result_file} | score → {score_file}")
    return {
        "result_file": result_file,
        "score_file": score_file,
        "num_questions": total,
        "retrieved_depth": depth,
        "metrics": metrics,
    }


def run_batch_murre(
    top_k: int = 5,
    force_embed: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Chuỗi 5 bước của MURRE (HUONG_DAN.md mục 5b), thay cho việc gõ tay 22 lệnh:
    embed → retrieve hop 0 → (rewrite hop N-1 → retrieve hop N × beam) × max_hop
    → score → infer.
    """
    from steps.embed import run_embed
    from steps.infer import run_infer
    from steps.retrieve import run_retrieve
    from steps.rewrite import run_rewrite
    from steps.score import run_score

    beam_size: int = cfg.pipeline.beam_size
    max_hop: int = cfg.pipeline.max_hop

    # Bước 1 — mã hóa corpus. Chỉ cần chạy lại khi đổi dataset/encoder, nên có cache
    # thì bỏ qua: encode lại cả corpus tốn hàng phút.
    emb_file: str = cfg.outputs.embeddings_cache()
    if force_embed or not os.path.exists(emb_file):
        run_embed()
    else:
        logger.info(f"[Runner] Đã có {emb_file} → bỏ qua bước embed (FORCE_EMBED=True để ép chạy lại).")

    # Bước 2 — retrieve hop 0 trên toàn bộ corpus.
    run_retrieve(hop=0, limit=limit)

    # Bước 3+4 — xen kẽ rewrite (hop 0..max_hop-1) / retrieve (hop 1..max_hop).
    for hop in range(1, max_hop + 1):
        run_rewrite(hop=hop - 1)
        for beam in range(beam_size):
            run_retrieve(hop=hop, beam=beam)

    # Bước 5 — tổng hợp điểm rồi sinh SQL.
    run_score()
    run_infer(top_k=top_k)


def run_batch(
    method: Method,
    top_k: int = 5,
    force_embed: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Chạy cả dev.json bằng đường phù hợp với `method`:
    murre → chuỗi steps/*.py; single_hop/crush → run_pipeline() (steps/*.py chỉ
    ghép được multi-hop cho murre).
    """
    dataset: Dataset = Dataset(cfg.general.dataset)

    match method:
        case Method.MURRE:
            logger.info("[Runner] Batch MURRE — chạy chuỗi steps/*.py")
            with override_cfg(method=method):
                run_batch_murre(top_k=top_k, force_embed=force_embed, limit=limit)

        case Method.SINGLE_HOP | Method.CRUSH:

            logger.info(f"[Runner] Batch {method} — chạy qua run_pipeline()")
            summary = run_pipeline(dataset=dataset, method=method, limit=limit)
            logger.info(
                f"[Runner] Xong {summary['num_questions']} câu. "
                f"result → {summary['result_file']} | score → {summary['score_file']}"
            )

        case _:
            raise SystemExit(f"Method mới chưa xử lý trong run_batch: {method}")
