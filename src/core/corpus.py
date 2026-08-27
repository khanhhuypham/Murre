"""core/corpus.py — Nạp corpus schema + embeddings (có cache), dùng chung cho mọi
method và mọi entry point.

Gom lại một chỗ vì logic "đọc tables.json → build corpus → nạp/encode embeddings →
lưu cache" trước đây bị lặp ở api.py và các script test.
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
    """Danh sách phẳng mọi schema của dataset. dataset=None → dùng general.dataset."""
    if dataset:
        cfg.general.dataset = dataset
    return build_schema_corpus(tables=load_tables())


def load_embeddings(
    encoder: SGPTEncoder,
    corpus: List[str],
    dataset: Optional[str] = None,
) -> torch.Tensor:
    """Nạp embeddings corpus từ cache, chưa có thì encode rồi lưu lại.

    Cache nằm ở paths.embeddings_cache — có cả {dataset} và {scale} trong tên, nên
    đổi encoder scale sẽ dùng file cache khác chứ không nạp nhầm vector cũ.
    """
    cache_path: str = cfg.outputs.for_run(dataset=dataset).embeddings_cache()

    if os.path.exists(cache_path):
        logger.info(f"[Corpus] Nạp embeddings từ cache: {cache_path}")
        return torch.load(cache_path, weights_only=True)

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
