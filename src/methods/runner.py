"""methods/runner.py — NƠI DUY NHẤT viết cách chạy pipeline.

    run_one_question()  Option 1 — MỘT câu hỏi, in ra terminal, không ghi file.
    run_pipeline()      Option 2 — cả dev.json, ghi result + score. POST /pipeline
                        và run_batch() đều gọi hàm này.
    run_batch()         Option 2 — run_pipeline() + sinh SQL (murre).

Cả 3 method dùng chung một đường, bộ retrieval ráp qua methods/build.py.

`cfg` là biến toàn cục của process nên mỗi lúc chỉ cho phép MỘT lần chạy
(_RUN_LOCK); trong lúc chạy, /retrieve cũng thấy cfg đã bị ghi đè.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from config import cfg
from dataset.loader import load_dev, resolve_question
from enums import Dataset, Method
from methods.build import LoadedDataset, build_dataset
from models.errors import AppError
from models.metrics import MetricScores
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

    Chỉ đụng general.dataset và pipeline.method — đó là hai giá trị mà nhiều chỗ
    đọc NGẦM (template đường dẫn trong PathsConfig, build_retriever()).
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
    """Chạy MỘT câu hỏi rồi in top-N bảng ra terminal, không ghi file.

        question         : None → câu đầu tiên trong dev.json
        verbose          : in chi tiết từng hop (murre) / bảng LLM đoán (crush)
        llm_profile      : None → dùng llm.active_profile
        crush_collective : chỉ có tác dụng với Method.CRUSH
    """
    q: str = resolve_question(question=question)

    loaded: LoadedDataset = build_dataset(
        method=method, llm_profile=llm_profile, crush_collective=crush_collective,
    )
    results: List[RetrievedTable] = loaded.retriever.run(
        question=q, corpus=loaded.corpus, schema_embeddings=loaded.embs, verbose=verbose,
    )

    print_results(method=str(method), question=q, results=results, top_n=top_n)
    return results


# ---------------------------------------------------------------------------
# Option 2 — batch
# ---------------------------------------------------------------------------
def run_pipeline(
    method: Method,
    dataset: Optional[Dataset] = None,
    limit: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Chạy retrieval trên cả dev.json rồi ghi result + score ra đĩa.

        dataset     : None → giữ nguyên general.dataset đang có.
        limit       : chỉ chạy N câu đầu (None = cả dev.json).
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


def _run_fingerprint(method: Method) -> Dict[str, Any]:
    """Các tham số mà đổi đi thì kết quả đã lưu trong checkpoint không dùng lại được.

    Đường dẫn checkpoint đã có dataset/model/method/max_hop, nhưng beam_size, pool
    hay ablation thì không nằm trong tên file — đổi chúng mà vẫn nối tiếp checkpoint
    cũ là trộn hai cấu hình vào một bảng điểm.
    """
    ab = cfg.pipeline.ablation
    return {
        "dataset": cfg.general.dataset,
        "method": method.value,
        "encoder": cfg.encoder.model_name,
        "llm_profile": cfg.llm.active_profile,
        "beam_size": cfg.pipeline.beam_size,
        "max_hop": cfg.pipeline.max_hop,
        "top_k_pool": cfg.pipeline.top_k_pool,
        "ablation": [ab.removal, ab.tabulation],
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
    method: Method,
    limit: Optional[int],
    on_progress: Optional[ProgressFn]
) -> Dict[str, Any]:
    """Thân của run_pipeline — đã giữ lock và đã ghi đè cfg."""
    dev: List[Dict[str, Any]] = load_dev()
    if limit is not None:
        dev = dev[:limit]
    total: int = len(dev)
    if total == 0:
        raise ValueError("dev.json rỗng — không có câu hỏi nào để chạy.")

    # Checkpoint đọc TRƯỚC khi dựng dataset: chạy lại một lượt đã xong thì không phải
    # nạp encoder/LLM làm gì.
    ckpt_file: str = cfg.outputs.checkpoint()
    fingerprint: Dict[str, Any] = _run_fingerprint(method=method)
    done: Dict[int, ResultRecord] = _load_checkpoint(path=ckpt_file, fingerprint=fingerprint)
    todo: List[int] = [i for i in range(total) if i not in done]

    loaded: LoadedDataset = build_dataset(method=method)

    logger.info(
        f"[Runner] Bắt đầu {method} trên {total} câu ({len(todo)} câu còn phải chạy), "
        f"corpus {len(loaded.corpus)} schemas."
    )

    os.makedirs(os.path.dirname(ckpt_file) or ".", exist_ok=True)
    is_new: bool = not os.path.exists(ckpt_file)
    retries: int = max(1, cfg.pipeline.question_retries)

    with open(ckpt_file, "a", encoding="utf-8") as ckpt:
        if is_new:
            ckpt.write(json.dumps({"_meta": fingerprint}, ensure_ascii=False) + "\n")
            ckpt.flush()

        for n, idx in enumerate(todo, start=1):
            d: Dict[str, Any] = dev[idx]

            # Một lượt đầy đủ là hàng nghìn lần gọi LLM; timeout/429/Ollama bận là
            # chuyện thường. Thử lại từng câu thay vì để hỏng cả lượt chạy.
            hits: Optional[List[RetrievedTable]] = None
            for attempt in range(1, retries + 1):
                try:
                    hits = loaded.retriever.run(
                        question=d["utterance"],
                        corpus=loaded.corpus,
                        schema_embeddings=loaded.embs,
                    )
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
                # to_rows(): đánh số rank + đổi khóa `score` → `similarity` của file format.
                retrieved=RetrievedTable.to_rows(tables=hits),
            )
            done[idx] = record
            ckpt.write(json.dumps(
                {"index": idx, "record": record.to_dict()}, ensure_ascii=False,
            ) + "\n")
            ckpt.flush()  # flush từng câu: tắt máy giữa chừng vẫn giữ được

            if on_progress is not None:
                on_progress(len(done), total)
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


def run_batch(
    method: Method,
    top_k: int = 5,
    force_embed: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Chạy cả dev.json cho MỘT method, rồi sinh SQL nếu là murre.
        top_k       : số bảng đưa vào bước sinh SQL (chỉ dùng với murre).
        force_embed : xoá cache embeddings để encode lại corpus.
        limit       : chỉ chạy N câu đầu (None = cả dev.json).
    """
    if force_embed:
        cache_file: str = cfg.outputs.embeddings_cache()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"[Runner] force_embed → đã xoá {cache_file}, sẽ encode lại.")

    logger.info(f"[Runner] Batch {method} — chạy qua run_pipeline()")

    run_pipeline(method=method, limit=limit)

    # Bước cuối của paper: top-K bảng → SQL, đọc lại chính file result vừa ghi.
    # Import trong thân hàm cho đồng bộ với methods/murre.py::build_sql.
    if method is Method.MURRE:
        from steps.infer import run_infer
        run_infer(top_k=top_k)
