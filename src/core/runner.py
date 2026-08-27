"""core/runner.py — Chạy TRỌN pipeline cho một tổ hợp (dataset, model, method).

Đây là bản "Option 1 — offline" (`run_option.mode` trong config.yaml): gọi thẳng
`methods/*.run()` trong process, lặp qua từng câu của dev.json, rồi ghi ra ĐÚNG file
mà `/evaluate` đọc (`paths.result` + `paths.score`). Nhờ vậy cả 3 method đều chạy
được bằng một đường code, khác với `steps/*.py` (Option 2 — batch) chỉ ghép được
multi-hop cho method=murre.

Lưu ý về đồng thời: `cfg` là biến toàn cục của process, nên một lần chạy phải tạm
ghi đè `cfg.general.dataset/scale`, `cfg.encoder.model_name`, `cfg.pipeline.method`.
Vì vậy mỗi thời điểm chỉ cho phép MỘT lần chạy (`_RUN_LOCK`), và trong lúc chạy thì
`/retrieve` cũng thấy cfg đã bị ghi đè — muốn cách ly hoàn toàn thì phải chạy
`steps/*.py` ở process riêng.
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
from core.factory import build_retriever
from core.llm import LLMGenerator
from dataset.loader import load_dev
from enums import Dataset, Method, ModelScale
from utils import logger
from utils.metrics import compute_res

# Chỉ một lần chạy tại một thời điểm — xem docstring module.
_RUN_LOCK: threading.Lock = threading.Lock()

# Callback báo tiến độ: (số câu đã xong, tổng số câu).
ProgressFn = Callable[[int, int], None]


class PipelineBusyError(RuntimeError):
    """Đã có một lần chạy pipeline đang diễn ra."""


@contextmanager
def override_run_config(
    dataset: Dataset,
    scale: ModelScale,
    method: Method,
) -> Iterator[None]:
    """Tạm ghi đè cfg cho một lần chạy rồi trả lại nguyên trạng.

    `encoder.model_name` phải đổi kèm `general.scale`: scale chỉ quyết định ĐƯỜNG DẪN
    file, còn model thực sự nạp lên là `encoder.model_name`. Đổi một cái mà quên cái
    kia sẽ ghi vector của model này vào cache mang tên model khác.
    """
    saved: Dict[str, Any] = {
        "dataset": cfg.general.dataset,
        "scale": cfg.general.scale,
        "model_name": cfg.encoder.model_name,
        "method": cfg.pipeline.method,
    }
    cfg.general.dataset = dataset.value
    cfg.general.scale = scale.value
    cfg.encoder.model_name = scale.model_name
    cfg.pipeline.method = method.value
    logger.info(
        f"[Runner] cfg tạm: dataset={dataset} scale={scale} method={method} "
        f"encoder={scale.model_name}"
    )
    try:
        yield
    finally:
        cfg.general.dataset = saved["dataset"]
        cfg.general.scale = saved["scale"]
        cfg.encoder.model_name = saved["model_name"]
        cfg.pipeline.method = saved["method"]
        logger.info("[Runner] Đã trả cfg về nguyên trạng.")


def run_pipeline(
    dataset: Dataset,
    scale: ModelScale,
    method: Method,
    limit: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Chạy pipeline trên dev.json rồi ghi result + score ra đĩa.

    Tham số:
        limit       : chỉ chạy `limit` câu đầu (None = cả dev.json). Dùng để thử
                      nhanh — murre gọi LLM mỗi hop mỗi beam nên chạy đủ 658 câu
                      rất lâu.
        on_progress : callback(đã_xong, tổng) để báo tiến độ ra ngoài.

    Trả về:
        {"result_file", "score_file", "num_questions", "retrieved_depth", "metrics"}

    Ném PipelineBusyError nếu đang có lần chạy khác.
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise PipelineBusyError(
            "Đang có một lần chạy pipeline khác. cfg là biến toàn cục nên phải chạy "
            "lần lượt — đợi job hiện tại xong rồi thử lại."
        )
    try:
        with override_run_config(dataset=dataset, scale=scale, method=method):
            return _run_locked(method=method, limit=limit, on_progress=on_progress)
    finally:
        _RUN_LOCK.release()


def _run_locked(
    method: Method,
    limit: Optional[int],
    on_progress: Optional[ProgressFn],
) -> Dict[str, Any]:
    """Phần thân của run_pipeline — đã giữ lock và đã ghi đè cfg."""
    dev: List[Dict[str, Any]] = load_dev()
    if limit is not None:
        dev = dev[:limit]
    total: int = len(dev)
    if total == 0:
        raise ValueError("dev.json rỗng — không có câu hỏi nào để chạy.")

    # prepare() đọc tables.json → build corpus → nạp/encode embeddings (có cache).
    encoder, corpus, embs = prepare()
    embs: torch.Tensor

    llm: Optional[LLMGenerator] = LLMGenerator() if method.needs_llm else None
    retriever = build_retriever(encoder=encoder, llm=llm)

    logger.info(f"[Runner] Bắt đầu {method} trên {total} câu, corpus {len(corpus)} schemas.")

    output: List[Dict[str, Any]] = []
    for i, d in enumerate(dev, start=1):
        hits: List[Dict[str, Any]] = retriever.run(
            question=d["utterance"], corpus=corpus, schema_embeddings=embs,
        )
        output.append({
            "utterance": d["utterance"],
            "gold": d.get("rel_schema", []),
            # Đặt tên khóa "similarity" để khớp file của steps/score.py — compute_res()
            # và /evaluate chỉ đọc "schema", nhưng giữ cùng format cho dễ so sánh.
            "retrieved": [
                {"rank": rank, "schema": h["schema"], "similarity": h["score"]}
                for rank, h in enumerate(hits)
            ],
        })
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
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    depth: int = min(len(d["retrieved"]) for d in output)
    logger.info(f"[Runner] Xong. result → {result_file} | score → {score_file}")
    return {
        "result_file": result_file,
        "score_file": score_file,
        "num_questions": total,
        "retrieved_depth": depth,
        "metrics": metrics,
    }
