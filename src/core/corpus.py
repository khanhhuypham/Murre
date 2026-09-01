"""core/corpus.py — Nạp corpus schema + embeddings (có cache), dùng chung cho mọi
method và mọi entry point.

Gom lại một chỗ vì logic "đọc tables.json → build corpus → nạp/encode embeddings →
lưu cache" trước đây bị lặp ở api/ và các script test.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

from config import cfg
from core.encoder import SGPTEncoder
from dataset.loader import load_tables
from utils import logger
from utils.schema import build_schema_corpus


def build_corpus(dataset: Optional[str] = None) -> List[str]:
    """Danh sách phẳng mọi schema của dataset. dataset=None → dùng general.dataset.

    KHÔNG ghi đè cfg.general.dataset: API phục vụ nhiều dataset cùng lúc, đổi biến
    toàn cục là mọi request đang chạy thấy theo.
    """
    return build_schema_corpus(tables=load_tables(dataset=dataset))


def load_embeddings(
    encoder: SGPTEncoder,
    corpus: List[str],
    dataset: Optional[str] = None,
) -> torch.Tensor:
    """Nạp embeddings corpus từ cache, chưa có thì encode rồi lưu lại.

    Cache nằm ở paths.embeddings_cache — có cả {dataset} và {model} trong tên, nên
    đổi encoder.model_name sẽ dùng file cache khác chứ không nạp nhầm vector cũ.
    """
    cache_path: str = cfg.outputs.for_run(dataset=dataset).embeddings_cache()

    if os.path.exists(cache_path):
        logger.info(f"[Corpus] Nạp embeddings từ cache: {cache_path}")
        cached: torch.Tensor = torch.load(cache_path, weights_only=True)
        # Cache cũ của model/corpus khác vẫn nạp được nhưng số vector sẽ lệch với số
        # schema → điểm số gán nhầm bảng. Bắt tại đây thay vì chấm điểm sai âm thầm.
        if cached.shape[0] != len(corpus):
            raise RuntimeError(
                f"Cache embeddings không khớp corpus: {cached.shape[0]} vector "
                f"cho {len(corpus)} schema.\n"
                f"  Cache: {cache_path}\n"
                f"  Cách xử lý: xoá file cache đó rồi chạy lại để encode lại."
            )
        return cached

    logger.info(f"[Corpus] Chưa có cache, đang encode {len(corpus)} schemas ...")
    embs: torch.Tensor = encoder.encode(texts=corpus, is_query=False)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(obj=embs, f=cache_path)
    logger.info(f"[Corpus] Đã lưu cache → {cache_path}")
    return embs


def prepare(dataset: Optional[str] = None) -> Tuple[SGPTEncoder, List[str], torch.Tensor]:
    """Chuẩn bị đủ 3 thứ mà mọi retriever.run() cần: encoder, corpus, embeddings."""
    corpus: List[str] = build_corpus(dataset=dataset)
    encoder: SGPTEncoder = SGPTEncoder()
    embs: torch.Tensor = load_embeddings(encoder=encoder, corpus=corpus, dataset=dataset)
    return encoder, corpus, embs
