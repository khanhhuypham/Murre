"""core/corpus.py — Nạp corpus schema + embeddings (có cache).

Một chỗ duy nhất cho chuỗi "đọc tables.json → build corpus → nạp/encode embeddings
→ lưu cache". Hai chỗ gọi: MurreRetriever.for_dataset() (đường chạy thật của cả
CLI lẫn API) và `python -m cli embed` (chỉ mã hoá trước rồi lưu cache).
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
    """Danh sách phẳng mọi schema của dataset. dataset=None → dùng general.dataset.

    KHÔNG ghi đè cfg.general.dataset: API phục vụ nhiều dataset cùng lúc, đổi biến
    toàn cục là mọi request đang chạy thấy theo.
    """
    return build_schema_corpus(tables=load_tables(dataset=dataset))


def corpus_fingerprint(corpus: List[str], model_name: str) -> str:
    """Vân tay của (nội dung corpus, model) — quyết định một cache có dùng lại được.

    Đếm số vector thôi thì KHÔNG đủ: hai corpus khác hẳn nhau vẫn có thể cùng số
    schema. ViText2SQL mức syllable và mức word là đúng ca đó — cùng 876 bảng, chỉ
    khác cách viết tên ("kiến trúc sư" / "kiến_trúc_sư") — nên chuyển mức rồi chạy
    lại sẽ dùng lại vector của mức cũ và chấm điểm sai mà không báo gì.
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
    """Nạp embeddings corpus từ cache, chưa có thì encode rồi lưu lại.

    Cache nằm ở paths.embeddings_cache — có cả {dataset} và {model} trong tên, nên
    đổi encoder.model_name sẽ dùng file cache khác chứ không nạp nhầm vector cũ.
    Nội dung corpus đổi mà tên file không đổi thì vân tay bên trong bắt được.
    """
    cache_path: str = cfg.outputs.for_run(dataset=dataset).embeddings_cache()
    fingerprint: str = corpus_fingerprint(corpus=corpus, model_name=encoder.model_name)

    if os.path.exists(cache_path):
        cached: Any = torch.load(cache_path, weights_only=True)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            logger.info(f"[Corpus] Nạp embeddings từ cache: {cache_path}")
            return cached["embeddings"]

        # Cache của corpus/model KHÁC. Encode lại và ghi đè — nó chỉ là cache, dựng
        # lại được, nên không bắt người dùng phải đi xoá tay.
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
