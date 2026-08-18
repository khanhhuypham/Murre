"""steps/run_baseline.py — Chạy batch Single-hop hoặc CRUSH trên toàn bộ dev set
(dùng để tái tạo hàng "Single-hop"/"CRUSH" trong Table 2 và 3 của paper).

Khác với MURRE, 2 baseline này không cần multi-hop (embed → retrieve → rewrite lặp),
chỉ cần 1 lượt retrieve cho mỗi câu hỏi, nên có script riêng gọn hơn.

Chạy: python -m steps.run_baseline
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch

from config import cfg, get_dataset_path, get_path
from core.encoder import SGPTEncoder
from core.factory import build_retriever
from core.llm import LLMGenerator
from dataset.loader import load_tables
from utils import logger
from utils.metric import compute_res
from utils.schema import build_schema_corpus


def run_baseline() -> None:
    method: str = cfg.pipeline.method.lower()
    assert method in ("single_hop", "crush"), (
        f"run_baseline() chỉ dùng cho single_hop/crush, method hiện tại='{method}'. "
        "Dùng steps/embed.py + steps/rewrite.py + steps/retrieve.py cho method=murre."
    )

    logger.info("=" * 60)
    logger.info(f"  CHẠY BASELINE: {method.upper()}")
    logger.info("=" * 60)

    tables: List[Dict[str, Any]] = load_tables()
    corpus: List[str] = build_schema_corpus(tables=tables)

    encoder: SGPTEncoder = SGPTEncoder()

    cache_path: str = get_path(key="embeddings_cache")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    schema_embeddings: torch.Tensor
    if os.path.exists(cache_path):
        schema_embeddings = torch.load(cache_path, weights_only=True)
    else:
        schema_embeddings = encoder.encode(texts=corpus, is_query=False)
        torch.save(obj=schema_embeddings, f=cache_path)

    llm: LLMGenerator | None = LLMGenerator() if method == "crush" else None
    retriever = build_retriever(encoder=encoder, llm=llm)

    dev_path: str = get_dataset_path(key="dev")
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data: List[Dict[str, Any]] = json.load(f)

    top_k_list: List[int] = cfg.general.top_k
    final_output: List[Dict[str, Any]] = []

    for i, sample in enumerate(dev_data):
        question: str = sample.get("utterance", "")
        gold: List[str] = sample.get("rel_schema", [])

        results: List[Dict[str, Any]] = retriever.run(
            question=question, corpus=corpus, schema_embeddings=schema_embeddings, verbose=False,
        )
        retrieved: List[Dict[str, Any]] = [
            {"rank": rank, "schema": r["schema"], "similarity": r["score"]}
            for rank, r in enumerate(results)
        ]

        final_output.append({"utterance": question, "gold": gold, "retrieved": retrieved})

        if i % 50 == 0:
            logger.info(f"[{method}] Đã xử lý {i}/{len(dev_data)} câu hỏi ...")

    result_file: str = get_path(key="result")
    score_file: str = get_path(key="score")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    metrics: Dict[str, Dict[int, float]] = compute_res(top_k=top_k_list, data=final_output)
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"[{method}] Kết quả:")
    for k in top_k_list:
        r: float = metrics["recall"].get(k, 0.0)
        c: float = metrics["complete_recall"].get(k, 0.0)
        logger.info(f"  k={k:>3} | recall={r:.4f} | complete_recall={c:.4f}")

    logger.info(f"[{method}] Đã lưu → {result_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_baseline()
