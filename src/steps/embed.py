# =============================================================================
# steps/embed.py — BƯỚC 1: Mã hóa toàn bộ corpus schema thành vector embedding
#
# Tương đương với retrieve/embed.py của tác giả.
#
# CÁCH CHẠY (Option 2 — Batch mode):
#   python src/steps/embed.py
#
# Kết quả lưu tại: outputs/{dataset}/{scale}/embeddings.json
#   Format: [{"pred_schema": "db.table(cols)", "embedding": [0.1, 0.2, ...]}, ...]
# =============================================================================

import json
import os
from typing import List, Dict, Any
import torch
from config import get_path
from core.encoder import SGPTEncoder
from dataset.loader import load_tables
from utils.schema import build_schema_corpus
from utils import logger

def run_embed() -> None:
    """
    Chạy bước mã hóa corpus.
    Đọc tables.json → build corpus → encode bằng SGPT → lưu ra embeddings.json
    """

    logger.info(f"\n{'=' * 60}\n  BƯỚC 1: MÃ HÓA CORPUS SCHEMA (EMBED)\n{'=' * 60}")

    # Tải dữ liệu bảng
    tables = load_tables()
    # Xây dựng danh sách schema phẳng
    corpus: List[str] = build_schema_corpus(tables=tables)
    logger.info(f"[Embed] Tổng số schema cần mã hóa: {len(corpus)}")

    # Khởi tạo encoder và mã hóa
    encoder: SGPTEncoder = SGPTEncoder()
    logger.info(f"[Embed] Bắt đầu mã hóa {len(corpus)} schemas ...")
    embeddings: torch.Tensor = encoder.encode(texts=corpus, is_query=False)  # shape: (N, D)

    # Chuẩn bị đường dẫn output
    out_path: str = get_path(key="embeddings")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Lưu kết quả theo format của tác giả
    records: List[Dict[str, Any]] = [
        {
            "pred_schema": schema,
            "embedding": emb.tolist()
        }
        for schema, emb in zip(corpus, embeddings)
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"[Embed] Đã lưu {len(records)} embeddings → {out_path}")
    logger.info("=" * 60)


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP (Option 2 — Batch mode)
# =============================================================================
if __name__ == "__main__":
    run_embed()
