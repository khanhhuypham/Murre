# =============================================================================
# src/steps/retrieve.py — BƯỚC 2: Dense Retrieval (Tìm kiếm bảng theo embedding)
#
# Tương đương với retrieve/retrieve.py của tác giả.
#
# Ý NGHĨA THAM SỐ:
#   --hop 0   : Retrieve trên TOÀN BỘ corpus, dùng câu hỏi gốc trong dev.json.
#               Cần cho MỌI phương pháp (murre/single_hop/crush), không cần --beam.
#   --hop N   : (chỉ áp dụng khi pipeline.method=murre trong config.yaml)
#               Retrieve trong pool đã hẹp lại từ hop 0, dùng câu hỏi đã qua
#               Removal/Splice (output của steps/rewrite.py --hop N-1).
#               Cần chỉ định --beam tương ứng.
#
# ĐIỀU KIỆN CHẠY ĐƯỢC (phải chạy đúng thứ tự các bước trước):
#   --hop 0            cần: steps/embed.py đã chạy xong (có embeddings.json)
#   --hop N (N>=1)     cần: steps/retrieve.py --hop 0 VÀ
#                            steps/rewrite.py --hop (N-1) đã chạy xong
#
# CÁCH CHẠY (Option 2 — Batch mode; beam_size mặc định = 5 → beam 0..4):
#   python -m steps.retrieve --hop 0
#   python -m steps.rewrite  --hop 0
#   python -m steps.retrieve --hop 1 --beam 0
#   python -m steps.retrieve --hop 1 --beam 1
#   python -m steps.retrieve --hop 1 --beam 2
#   python -m steps.retrieve --hop 1 --beam 3
#   python -m steps.retrieve --hop 1 --beam 4
#   # ... lặp lại rewrite + retrieve cho các hop tiếp theo tới khi hết max_hop
#
# Lưu ý: nếu chạy trực tiếp bằng đường dẫn file (python src/steps/retrieve.py ...)
# thay vì "python -m steps.retrieve", cần đảm bảo src/ đã có trong sys.path
# (xem HUONG_DAN.md mục 11 — file .pth hoặc PYTHONPATH=src), nếu không sẽ gặp
# ModuleNotFoundError: No module named 'config'.
# =============================================================================

import json
import os
from typing import Any, Dict, List, Tuple, Optional

import torch
from scipy.spatial.distance import cosine
from config import cfg
from core.encoder import SGPTEncoder
from utils import logger
from utils.metrics import compute_recall


def _cosine_sim(q: torch.Tensor, d: torch.Tensor) -> float:
    """Tính cosine similarity giữa 2 vector (dùng scipy để chính xác)."""
    return 1.0 - float(cosine(u=q.numpy(), v=d.numpy()))


def retrieve_for_queries(
    query_embeddings_grouped: List[List[torch.Tensor]],
    pool_embs: List[torch.Tensor],
    pool_docs: List[str],
    top_k_max: int,
) -> List[List[Tuple[int, float]]]:
    """
    Tính cosine similarity và trả về top_k_max kết quả cho mỗi query.
    Xử lý trường hợp query có nhiều câu (multi-line utterance).
    """
    all_results: List[List[Tuple[int, float]]] = []

    for q_embs in query_embeddings_grouped:
        # Tính similarity tổng hợp qua tất cả sub-query của câu hỏi
        sims: List[Tuple[int, float]] = []
        for q in q_embs:
            for di, d in enumerate(pool_embs):
                sims.append((di, _cosine_sim(q, d)))

        # Sắp xếp giảm dần và de-duplicate
        sorted_sims: List[Tuple[int, float]] = sorted(sims, key=lambda x: x[1], reverse=True)
        seen: set[int] = set()
        result: List[Tuple[int, float]] = []
        for idx, sim in sorted_sims:
            if len(result) >= top_k_max:
                break

            if idx not in seen:
                result.append((idx, sim))
                seen.add(idx)

        all_results.append(result)

    return all_results


def run_retrieve(
    hop: int,
    beam: Optional[int] = None,
    queries_file: Optional[str] = None,
    last_retrieved_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    Chạy bước retrieval cho một hop và một beam cụ thể.

    Tham số:
        hop                 : số hop hiện tại (0 = full corpus, 1+ = hẹp pool)
        beam                : chỉ số beam (None với hop=0, 0..beam_size-1 với hop>0)
        queries_file        : đường dẫn file câu hỏi (None → dùng config)
        last_retrieved_file : file kết quả hop trước (None với hop=0)
        output_file         : đường dẫn output (None → dùng config)
    """
    top_k_list: List[int] = cfg.general.top_k
    top_k_max: int = max(top_k_list)

    logger.info(f"\n{'=' * 60}\n  BƯỚC 2: RETRIEVAL — Hop {hop}" + (f" | Beam {beam}" if beam is not None else "") + f"\n{'=' * 60}")

    # ── Tải embeddings corpus ──────────────────────────────────────────────
    emb_path: str = cfg.outputs.embeddings()
    logger.info(f"[Retrieve] Tải embeddings từ: {emb_path}")
    with open(emb_path, "r", encoding="utf-8") as f:
        original_docs: List[Dict[str, Any]] = json.load(f)

    all_docs: List[str] = [x["pred_schema"] for x in original_docs]
    all_doc_embs: List[torch.Tensor] = [
        torch.tensor(x["embedding"]) for x in original_docs
    ]
    schema_map: Dict[str, int] = {d["pred_schema"]: i for i, d in enumerate(original_docs)}

    # ── Tải câu hỏi ───────────────────────────────────────────────────────
    if queries_file is None:
        if hop == 0:
            queries_file = cfg.dataset_paths.dev
        else:
            queries_file = cfg.outputs.rewrite_output(hop=hop - 1) + f"/dev.{beam}.json"

    logger.info(f"[Retrieve] Tải câu hỏi từ: {queries_file}")
    with open(queries_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # ── Xác định pool tìm kiếm ────────────────────────────────────────────
    last_retrieved_data: List[Dict[str, Any]] = []
    if last_retrieved_file is None and hop > 0:
        last_retrieved_file = cfg.outputs.turn0()

    if last_retrieved_file:
        logger.info(f"[Retrieve] Hop {hop}: hẹp pool từ {last_retrieved_file}")
        with open(last_retrieved_file, "r", encoding="utf-8") as f:
            last_retrieved_data = json.load(f)

    # ── Mã hóa câu hỏi ────────────────────────────────────────────────────
    encoder: SGPTEncoder = SGPTEncoder()

    # Tách câu hỏi thành nhiều sub-query nếu có nhiều dòng (từ bước rewrite)
    queries: List[List[str]] = [
        d["utterance"].split("\n") if isinstance(d.get("utterance"), str)
        else [d.get("utterance", "")]
        for d in data
    ]

    # Mã hóa phẳng rồi nhóm lại
    indexed_subqueries: List[Tuple[int, str]] = [
        (qi, q_line) for qi, q_lines in enumerate(queries) for q_line in q_lines
    ]
    logger.info(f"[Retrieve] Mã hóa {len(indexed_subqueries)} chuỗi câu hỏi ...")
    flat_embs: torch.Tensor = encoder.encode(texts=[x[1] for x in indexed_subqueries], is_query=True)

    query_embs_grouped: List[List[torch.Tensor]] = [[] for _ in range(len(queries))]
    for i, (qi, _) in enumerate(indexed_subqueries):
        query_embs_grouped[qi].append(flat_embs[i])

    # ── Retrieve ───────────────────────────────────────────────────────────
    total_recall: Dict[int, float] = {k: 0.0 for k in top_k_list}
    output_data: List[Dict[str, Any]] = []

    for qi, d in enumerate(data):
        # Xác định pool document cho query này
        pool_embs: List[torch.Tensor]
        pool_docs: List[str]
        if last_retrieved_data:
            prev_schemas: List[str] = [r["schema"] for r in last_retrieved_data[qi]["retrieved"]]
            pool_embs = [all_doc_embs[schema_map[s]] for s in prev_schemas if s in schema_map]
            pool_docs = [all_docs[schema_map[s]] for s in prev_schemas if s in schema_map]
        else:
            pool_embs = all_doc_embs
            pool_docs = all_docs

        # Tính similarity và lấy top-K
        ranked: List[Tuple[int, float]] = retrieve_for_queries(
            query_embeddings_grouped=[query_embs_grouped[qi]],
            pool_embs=pool_embs,
            pool_docs=pool_docs,
            top_k_max=top_k_max,
        )[0]

        retrieved_docs: List[Dict[str, Any]] = [
            {"rank": rank, "schema": pool_docs[idx], "similarity": sim}
            for rank, (idx, sim) in enumerate(ranked)
        ]
        pred_schemas: List[str] = [pool_docs[idx] for idx, _ in ranked]

        # Tính recall
        gold: List[str] = d.get("rel_schema", [])
        recall_per_k: Dict[int, float] = {}
        for k in top_k_list:
            r: float = compute_recall(pred_list=pred_schemas[:k], gold_list=gold) if gold else 0.0
            total_recall[k] += r
            recall_per_k[k] = r

        utterance_org = d.get("utterance_org", [])
        if isinstance(utterance_org, str):
            utterance_org = [utterance_org]

        output_data.append({
            "utterance":     d.get("utterance", ""),
            "input":         queries[qi],
            "utterance_org": utterance_org,
            "selected_database": d.get("selected_database", []),
            "retrieved":     retrieved_docs,
            "gold":          gold,
            "recall":        recall_per_k,
        })

    # ── In kết quả recall ─────────────────────────────────────────────────
    n: int = len(data)
    logger.info("[Retrieve] Kết quả recall:")
    for k in top_k_list:
        logger.info(f"  recall@{k}: {total_recall[k] / n:.4f}")

    # ── Lưu kết quả ───────────────────────────────────────────────────────
    if output_file is None:
        if hop == 0:
            output_file = cfg.outputs.turn0()
        else:
            output_file = cfg.outputs.turn_n(hop=hop, beam=beam)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[Retrieve] Đã lưu → {output_file}")
    logger.info("=" * 60)


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP (Option 2 — Batch mode)
# Thêm import argparse chỉ ở đây (entry point), không dùng trong logic chính
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bước Retrieval của MURRE")
    parser.add_argument("--hop",  type=int, required=True, help="Hop hiện tại (0=full, 1+=hẹp pool)")
    parser.add_argument("--beam", type=int, default=None,  help="Chỉ số beam (chỉ cần cho hop > 0)")
    args = parser.parse_args()

    run_retrieve(hop=args.hop, beam=args.beam)
