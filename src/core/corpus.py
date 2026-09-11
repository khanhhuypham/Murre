"""core/corpus.py — Nạp corpus schema + embeddings (có cache).

Chuỗi "đọc tables.json → build corpus → nạp/encode embeddings → lưu cache".
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

import torch

from config import cfg
from core.encoder import SentenceEncoder
from dataset.loader import load_tables
from utils import logger
from utils.schema import build_schema_corpus


def build_corpus(dataset: Optional[str] = None) -> List[str]:
    """Danh sách phẳng mọi schema của dataset. None → general.dataset."""
    return build_schema_corpus(tables=load_tables(dataset=dataset))


def corpus_fingerprint(corpus: List[str], model_name: str) -> str:
    """Vân tay của (nội dung corpus, model) — quyết định cache dùng lại được.

    Đếm số vector không đủ: hai corpus khác nhau vẫn có thể cùng số schema.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for schema in corpus:
        h.update(b"\0")
        h.update(schema.encode("utf-8"))
    return h.hexdigest()[:16]


def load_embeddings(
    encoder: SentenceEncoder,
    corpus: List[str],
    dataset: Optional[str] = None,
) -> torch.Tensor:
    """Nạp embeddings từ cache, chưa có thì encode rồi lưu lại.

    Tên file cache có {dataset} và {model}; nội dung corpus đổi thì vân tay bắt.
    """
    cache_path: str = cfg.outputs.for_run(dataset=dataset).embeddings_cache()
    fingerprint: str = corpus_fingerprint(corpus=corpus, model_name=encoder.model_name)

    if os.path.exists(cache_path):
        cached: Any = torch.load(cache_path, weights_only=True)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            logger.info(f"[Corpus] Nạp embeddings từ cache: {cache_path}")
            return cached["embeddings"]

        # Cache của corpus/model khác → encode lại và ghi đè.
        logger.warning(
            f"[Corpus] Cache {cache_path} thuộc corpus/model khác → encode lại."
        )

    logger.info(f"[Corpus] Đang encode {len(corpus)} schemas ...")
    embs: torch.Tensor = encoder.encode(texts=corpus, is_query=False)

    payload: Dict[str, Any] = {"fingerprint": fingerprint, "embeddings": embs}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(obj=payload, f=cache_path)
    logger.info(f"[Corpus] Đã lưu cache → {cache_path}")
    return embs
