"""core/factory.py — Khởi tạo đúng phương pháp retrieval theo cfg.pipeline.method,
để main.py / steps/*.py không phải biết chi tiết từng phương pháp.
"""
from __future__ import annotations

from typing import Union

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from core.pipeline import MURREPipeline
from core.rewriter import QueryRewriter
from methods.crush import CrushRetriever
from methods.single_hop import SingleHopRetriever

RetrieverType = Union[MURREPipeline, SingleHopRetriever, CrushRetriever]


def build_retriever(encoder: SGPTEncoder, llm: LLMGenerator | None = None) -> RetrieverType:
    """Trả về đúng retriever theo cfg.pipeline.method ("murre" | "single_hop" | "crush")."""
    method: str = cfg.pipeline.method.lower()

    if method == "murre":
        assert llm is not None, "MURRE cần LLMGenerator cho pha Removal."
        rewriter: QueryRewriter = QueryRewriter(llm=llm)
        return MURREPipeline(encoder=encoder, rewriter=rewriter)

    if method == "single_hop":
        return SingleHopRetriever(encoder=encoder)

    if method == "crush":
        assert llm is not None, "CRUSH cần LLMGenerator để hallucinate schema."
        return CrushRetriever(encoder=encoder, llm=llm)

    raise ValueError(f"pipeline.method='{method}' không hợp lệ. Chỉ nhận: murre | single_hop | crush")
