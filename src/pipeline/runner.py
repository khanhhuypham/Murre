"""pipeline/runner.py — NƠI DUY NHẤT viết cách chạy pipeline.

    run_one_question()  MỘT câu hỏi, in ra terminal, không ghi file.
    run_pipeline()      cả dev.json, ghi result + score. POST /pipeline/run và
                        `python -m cli run` đều gọi hàm này.

Chạy dataset khác mặc định thì ghi đè `cfg.general.dataset` (override_dataset).
`cfg` là biến toàn cục nên mỗi lúc chỉ cho phép MỘT lần chạy (_RUN_LOCK).
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from config import cfg
from core.llm import LLMGenerator
from dataset.loader import load_dev, resolve_question
from enums import Dataset
from models.errors import AppError
from models.metrics import MetricScores
from models.records import ResultRecord
from models.retrieval import RetrievedTable
from pipeline.retriever import MurreRetriever
from utils import logger
from utils.display import print_results
from utils.metrics import compute_res

_RUN_LOCK: threading.Lock = threading.Lock()

# Callback báo tiến độ: (số câu đã xong, tổng số câu).
ProgressFn = Callable[[int, int], None]


@contextmanager
def override_dataset(dataset: Optional[Dataset] = None) -> Iterator[None]:
    """Tạm ghi đè general.dataset rồi trả lại nguyên trạng.

    general.dataset được đọc ngầm ở template đường dẫn và dataset/loader.py.
    """
    if dataset is None:
        yield
        return

    saved: str = cfg.general.dataset
    cfg.general.dataset = dataset.value
    logger.info(f"[Runner] cfg tạm: dataset={dataset}")
    try:
        yield
    finally:
        cfg.general.dataset = saved
        logger.info("[Runner] Đã trả cfg về nguyên trạng.")


# ---------------------------------------------------------------------------
# Một câu hỏi
# ---------------------------------------------------------------------------
def run_one_question(
    question: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = False,
    llm_profile: Optional[str] = None,
) -> List[RetrievedTable]:
    """Chạy MỘT câu hỏi rồi in top-K bảng ra terminal, không ghi file.

        question    : None → câu đầu tiên trong dev.json của dataset đang chọn
        top_k       : số bảng in ra
        verbose     : in chi tiết từng hop
        llm_profile : None → dùng llm.active_profile

    Chọn dataset bằng cách bọc lời gọi trong override_dataset().
    """
    q: str = resolve_question(question=question)

    # Dựng LLM trước corpus: endpoint chưa bật thì hỏng ngay.
    retriever: MurreRetriever = MurreRetriever.for_dataset(
        llm=LLMGenerator(profile=llm_profile),
    )
    results: List[RetrievedTable] = retriever.run(question=q, verbose=verbose)

    print_results(question=q, results=results, top_k=top_k)
    return results


# ---------------------------------------------------------------------------
# Cả dev.json
# ---------------------------------------------------------------------------
def run_pipeline(
    dataset: Optional[Dataset] = None,
    limit: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Chạy retrieval trên cả dev.json rồi ghi result + score ra đĩa.

        dataset     : None → giữ nguyên general.dataset đang có.
        limit       : chỉ chạy N câu đầu (None = cả dev.json).
        on_progress : callback(đã_xong, tổng).
        verbose     : log chi tiết từng hop của MỌI câu, như `cli ask -v`. Mỗi câu
                      in thêm một dòng tiêu đề để biết khối [MURRE] bên dưới thuộc
                      câu nào. Lượt chạy dài thì log phình rất to — bật khi cần soi.

    Trả về {"result_file", "score_file", "num_questions", "retrieved_depth", "metrics"}.
    Ném AppError 409 nếu đang có lần chạy khác.
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise AppError.pipeline_busy()
    try:
        with override_dataset(dataset=dataset):
            return _run_locked(limit=limit, on_progress=on_progress, verbose=verbose)
    finally:
        _RUN_LOCK.release()


def _run_fingerprint() -> Dict[str, Any]:
    """Tham số mà đổi đi thì checkpoint cũ không dùng lại được."""
    return {
        "dataset": cfg.general.dataset,
        "encoder": cfg.encoder_for().model_name,
        "llm_profile": cfg.llm.active_profile,
        "beam_size": cfg.pipeline.beam_size,
        "max_hop": cfg.pipeline.max_hop,
    }


def _load_checkpoint(path: str, fingerprint: Dict[str, Any]) -> Dict[int, ResultRecord]:
    """Đọc các câu đã chạy xong. Cấu hình khác với lần trước → bỏ, không nối tiếp."""
    if not os.path.exists(path):
        return {}

    done: Dict[int, ResultRecord] = {}
    is_stale: bool = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                # Dòng cuối viết dở do bị ngắt giữa chừng — bỏ, chạy lại câu đó.
                logger.warning("[Runner] Checkpoint có dòng hỏng, bỏ qua dòng đó.")
                continue
            if "_meta" in obj:
                if obj["_meta"] != fingerprint:
                    is_stale = True
                    break
                continue
            done[int(obj["index"])] = ResultRecord.from_dict(d=obj["record"])

    # Đổi tên SAU KHI đã đóng file: Windows không cho rename file đang mở.
    if is_stale:
        stale: str = path + ".stale"
        os.replace(path, stale)
        logger.warning(
            f"[Runner] Checkpoint thuộc cấu hình KHÁC → đổi tên thành {stale} "
            f"và chạy lại từ đầu."
        )
        return {}

    if done:
        logger.info(f"[Runner] Checkpoint: đã có {len(done)} câu, sẽ bỏ qua.")
    return done


def _run_locked(
    limit: Optional[int],
    on_progress: Optional[ProgressFn],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Thân của run_pipeline — đã giữ lock và đã ghi đè cfg."""
    dev: List[Dict[str, Any]] = load_dev()
    if limit is not None:
        dev = dev[:limit]
    total: int = len(dev)
    if total == 0:
        raise AppError.bad_request(message="dev.json rỗng — không có câu hỏi nào để chạy.")

    # Đọc checkpoint trước khi dựng retriever: lượt đã xong thì khỏi nạp model.
    ckpt_file: str = cfg.outputs.checkpoint()
    fingerprint: Dict[str, Any] = _run_fingerprint()
    done: Dict[int, ResultRecord] = _load_checkpoint(path=ckpt_file, fingerprint=fingerprint)
    todo: List[int] = [i for i in range(total) if i not in done]

    # Báo tiến độ trước khi nạp model, kẻo lượt đã xong bị báo 0/0.
    if on_progress is not None:
        on_progress(len(done), total)

    retriever: MurreRetriever = MurreRetriever.for_dataset()

    logger.info(
        f"[Runner] Bắt đầu trên {total} câu ({len(todo)} câu còn phải chạy), "
        f"corpus {len(retriever.corpus)} schemas."
    )

    os.makedirs(os.path.dirname(ckpt_file) or ".", exist_ok=True)
    is_new: bool = not os.path.exists(ckpt_file)
    retries: int = cfg.pipeline.question_retries

    with open(ckpt_file, "a", encoding="utf-8") as ckpt:
        if is_new:
            ckpt.write(json.dumps({"_meta": fingerprint}, ensure_ascii=False) + "\n")
            ckpt.flush()

        for n, idx in enumerate(todo, start=1):
            d: Dict[str, Any] = dev[idx]

            # Tiêu đề phải in TRƯỚC khi chạy: mấy chục dòng [MURRE] ngay bên dưới
            # là của câu này, không có dòng này thì log batch không quy về câu nào.
            if verbose:
                logger.info(
                    f"[Runner] Câu {n}/{len(todo)} (#{idx}): {d['utterance']}"
                )
            started: float = time.perf_counter()

            # Thử lại từng câu thay vì để hỏng cả lượt chạy.
            hits: Optional[List[RetrievedTable]] = None
            for attempt in range(1, retries + 1):
                try:
                    hits = retriever.run(question=d["utterance"], verbose=verbose)
                    break
                except Exception as exc:
                    if attempt == retries:
                        logger.error(
                            f"[Runner] Câu #{idx} hỏng sau {retries} lần thử. "
                            f"{len(done)} câu đã xong vẫn nằm trong {ckpt_file} — "
                            f"chạy lại để tiếp tục."
                        )
                        raise
                    wait: float = 2.0 * attempt
                    logger.warning(
                        f"[Runner] Câu #{idx} lỗi lần {attempt}/{retries}: "
                        f"{type(exc).__name__}: {exc}. Chờ {wait:.0f}s rồi thử lại."
                    )
                    time.sleep(wait)

            assert hits is not None
            record = ResultRecord(
                utterance=d["utterance"],
                gold=d.get("rel_schema", []),
                # to_rows(): đánh số rank + đổi khóa `score` → `similarity`.
                retrieved=RetrievedTable.to_rows(tables=hits),
                # SQL đúng, để đối chiếu với câu pipeline/sql.py sinh ra sau này.
                gold_sql=d.get("query", ""),
            )
            done[idx] = record
            ckpt.write(json.dumps(
                {"index": idx, "record": record.to_dict()}, ensure_ascii=False,
            ) + "\n")
            ckpt.flush()  # flush từng câu để tắt máy giữa chừng vẫn giữ được

            if on_progress is not None:
                on_progress(len(done), total)
            if verbose:
                gold: List[str] = list(d.get("rel_schema", []))
                top: List[str] = [t.schema for t in hits[: max(len(gold), 1)]]
                found: int = sum(1 for g in gold if g in top)
                hit_note: str = (
                    f" | gold {found}/{len(gold)} trong top-{len(top)}" if gold else ""
                )
                logger.info(
                    f"[Runner] Câu #{idx} xong sau {time.perf_counter() - started:.1f}s"
                    f" | {len(hits)} bảng{hit_note}"
                )
            if n % 20 == 0 or n == len(todo):
                logger.info(f"[Runner] {len(done)}/{total} câu xong.")

    output: List[ResultRecord] = [done[i] for i in range(total)]

    top_k_list: List[int] = list(cfg.general.top_k)
    metrics: MetricScores = compute_res(top_k=top_k_list, data=output)

    result_file: str = cfg.outputs.result()
    score_file: str = cfg.outputs.score()
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in output], f, ensure_ascii=False, indent=2)
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)

    depth: int = min(len(d.retrieved) for d in output)
    logger.info(
        f"[Runner] Xong {total} câu. result → {result_file} | score → {score_file}"
    )
    return {
        "result_file": result_file,
        "score_file": score_file,
        "num_questions": total,
        "retrieved_depth": depth,
        "metrics": metrics,
    }
