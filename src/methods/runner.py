"""methods/runner.py — NƠI DUY NHẤT viết cách chạy pipeline.

main.py, khối `__main__` của cả 3 file trong methods/, và api.py đều import từ đây.
Không nơi nào tự viết lại trình tự "prepare → dựng retriever → run → in kết quả".

  * run_offline()     Option 1 — MỘT câu hỏi, in bảng kết quả ra terminal.
  * run_pipeline()    Option 1 — CẢ dev.json trong process này, ghi result + score.
                      Là đường batch duy nhất cho single_hop/crush, vì steps/*.py
                      chỉ ghép được multi-hop cho murre. Cũng là thứ POST /pipeline dùng.
  * run_batch_murre() Option 2 — chuỗi 5 bước steps/*.py cho murre (có sinh SQL).
  * run_batch()       Option 2 — chọn 1 trong 2 đường trên theo `method`.
  * override_cfg()    tạm ghi đè cfg rồi trả lại nguyên trạng (xem docstring riêng).
  * _RUN_LOCK         chặn chạy chồng.

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
from core.llm import LLMGenerator
from dataset.loader import load_dev, resolve_question
from enums import Dataset, Method, ModelScale
from methods.factory import build_retriever
from models.records import ResultRecord
from models.retrieval import RetrievedTable
from utils import logger
from utils.display import print_results
from utils.metrics import compute_res

# Chỉ một lần chạy tại một thời điểm — xem docstring module.
_RUN_LOCK: threading.Lock = threading.Lock()

# Callback báo tiến độ: (số câu đã xong, tổng số câu).
ProgressFn = Callable[[int, int], None]


class PipelineBusyError(RuntimeError):
    """Đã có một lần chạy pipeline đang diễn ra."""


@contextmanager
def override_cfg(
    dataset: Optional[Dataset] = None,
    scale: Optional[ModelScale] = None,
    method: Optional[Method] = None,
) -> Iterator[None]:
    """Tạm ghi đè cfg cho một lần chạy rồi trả lại nguyên trạng.

    CHỈ đụng tới thứ được truyền vào; tham số để None thì giữ nguyên. Nhờ vậy một
    hàm phục vụ được cả hai nhu cầu: đổi trọn bộ (run_pipeline) và chỉ đổi mỗi
    method (main.py).

    Phải dùng context manager vì vẫn còn chỗ đọc cfg NGẦM, không nhận tham số:
      * core/encoder.SGPTEncoder()  → cfg.encoder.model_name
      * template trong PathsConfig  → {dataset} {scale} {method}, tức mọi file mà
        steps/*.py và run_pipeline() ghi ra outputs/

    (build_retriever() thì KHÔNG còn nằm trong danh sách này — nó nhận thẳng tham số
    `method`, nên chỗ nào chỉ cần chọn retriever thì không phải ghi đè cfg.)

    `scale` kéo theo `encoder.model_name`: scale chỉ quyết định ĐƯỜNG DẪN file, còn
    model thực sự nạp lên là encoder.model_name. Đổi một cái mà quên cái kia sẽ ghi
    vector của model này vào cache mang tên model khác. Ngược lại, truyền MỖI
    `method` thì không đụng encoder — giữ nguyên model tuỳ chỉnh đặt qua config.yaml
    hoặc .env (ENCODER_MODEL_NAME).
    """
    saved: Dict[str, Any] = {
        "dataset": cfg.general.dataset,
        "scale": cfg.general.scale,
        "model_name": cfg.encoder.model_name,
        "method": cfg.pipeline.method,
    }

    changed: List[str] = []
    if dataset is not None:
        cfg.general.dataset = dataset.value
        changed.append(f"dataset={dataset}")
    if scale is not None:
        cfg.general.scale = scale.value
        cfg.encoder.model_name = scale.model_name
        changed.append(f"scale={scale} encoder={scale.model_name}")
    if method is not None:
        cfg.pipeline.method = method.value
        changed.append(f"method={method}")

    if changed:
        logger.info(f"[Runner] cfg tạm: {' '.join(changed)}")
    try:
        yield
    finally:
        # Trả lại cả 4 cho gọn — gán lại đúng giá trị cũ thì không ảnh hưởng gì.
        cfg.general.dataset = saved["dataset"]
        cfg.general.scale = saved["scale"]
        cfg.encoder.model_name = saved["model_name"]
        cfg.pipeline.method = saved["method"]
        if changed:
            logger.info("[Runner] Đã trả cfg về nguyên trạng.")


# ---------------------------------------------------------------------------
# Option 1 — Offline: một câu hỏi, chạy thẳng trong process này
# ---------------------------------------------------------------------------
def run_offline(
    method: Method,
    question: Optional[str] = None,
    top_n: int = 5,
    verbose: bool = False,
    llm_profile: Optional[str] = None,
    crush_collective: bool = True,
) -> List[RetrievedTable]:
    """Chạy đúng MỘT câu hỏi qua `method` rồi in bảng kết quả.

    Đây là chỗ DUY NHẤT viết trình tự chạy 1 câu hỏi. main.py và khối `__main__`
    của cả 3 file trong methods/ đều gọi hàm này, không tự lặp lại trình tự nữa.

    Tham số:
        question         : None → lấy câu đầu tiên trong dev.json (resolve_question)
        verbose          : in chi tiết từng hop (murre) / danh sách bảng LLM đoán (crush)
        llm_profile      : None → dùng llm.active_profile
        crush_collective : chỉ có tác dụng với Method.CRUSH — xem CrushRetriever

    Trả về danh sách bảng đã xếp hạng (cũng đã in sẵn ra terminal).
    """
    q: str = resolve_question(question=question)

    encoder, corpus, embs = prepare()
    llm: Optional[LLMGenerator] = (
        LLMGenerator(profile=llm_profile) if method.needs_llm else None
    )
    # KHÔNG cần override_cfg ở đây: build_retriever nhận thẳng `method`, và hàm này
    # không ghi file nào nên cũng không đụng tới template đường dẫn có {method}.
    retriever = build_retriever(
        encoder=encoder, llm=llm, method=method, crush_collective=crush_collective,
    )
    results: List[RetrievedTable] = retriever.run(
        question=q, corpus=corpus, schema_embeddings=embs, verbose=verbose,
    )

    print_results(method=str(method), question=q, results=results, top_n=top_n)
    return results


# ---------------------------------------------------------------------------
# Option 1 — Offline: cả dev.json, chạy trong process này
# ---------------------------------------------------------------------------
def run_pipeline(
    dataset: Dataset,
    method: Method,
    scale: Optional[ModelScale] = None,
    limit: Optional[int] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Chạy pipeline trên dev.json rồi ghi result + score ra đĩa.

    Tham số:
        scale       : None → giữ nguyên general.scale hiện tại, và KHÔNG đụng tới
                      encoder.model_name (giữ model tuỳ chỉnh đặt qua .env). Chỉ
                      truyền khi thật sự muốn đổi scale cho lần chạy này.
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
        with override_cfg(dataset=dataset, scale=scale, method=method):
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
            # to_rows() vừa đánh số rank vừa đổi `score` sang tên khóa `similarity`
            # của file format — xem RetrievedTable.to_row().
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
        logger.info(f"[Runner] Đã có {emb_file} → bỏ qua bước embed (FORCE_EMBED=True để ép chạy lại).")

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
    single_hop / crush  → run_pipeline() ở trên, vì steps/*.py chỉ ghép multi-hop
                          cho murre.
    """
    dataset: Dataset = Dataset(cfg.general.dataset)

    match method:
        case Method.MURRE:
            if limit is not None:
                raise SystemExit(
                    "LIMIT chỉ dùng được với method single_hop/crush (đường run_pipeline).\n"
                    "  Muốn chạy thử nhanh MURRE thì giảm beam_size/max_hop trong "
                    "PipelineConfig (src/config.py) — xem HUONG_DAN.md mục 5b."
                )

            # steps/*.py điền {method} vào đường dẫn output theo cfg.pipeline.method
            # (xem PathsConfig), nên phải đồng bộ cfg với tham số `method` trong lúc
            # chạy — nếu không, kết quả MURRE sẽ ghi nhầm vào thư mục method khác.
            logger.info("[Runner] Batch MURRE — chạy chuỗi steps/*.py")
            with override_cfg(method=method):
                run_batch_murre(top_k=top_k, force_embed=force_embed)

        case Method.SINGLE_HOP | Method.CRUSH:
            # 2 baseline không có multi-hop nên không dùng chuỗi steps/ được.
            # run_pipeline tự gọi override_cfg bên trong nên không cần bọc lại.
            logger.info(f"[Runner] Batch {method} — chạy qua run_pipeline()")
            # Không truyền scale: đã lấy từ cfg thì truyền lại cũng vậy, mà còn kéo
            # theo việc ép encoder.model_name — xem docstring override_cfg.
            summary = run_pipeline(dataset=dataset, method=method, limit=limit)
            logger.info(
                f"[Runner] Xong {summary['num_questions']} câu. "
                f"result → {summary['result_file']} | score → {summary['score_file']}"
            )

        case _:
            raise SystemExit(f"Method mới chưa xử lý trong run_batch: {method}")
