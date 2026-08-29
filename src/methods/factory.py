"""methods/factory.py — Khởi tạo đúng phương pháp retrieval theo cfg.pipeline.method,
để main.py / steps/*.py không phải biết chi tiết từng phương pháp.

Cả 3 retriever có chung một giao diện:

    run(question, corpus, schema_embeddings, verbose=False) -> List[RetrievedTable]

(xem models/retrieval.py). Nhờ vậy chỗ gọi không cần biết đang dùng method nào.
"""
from __future__ import annotations

from typing import Union

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from core.rewriter import QueryRewriter
from enums import Method
from methods.crush import CrushRetriever
from methods.murre import MURREPipeline
from methods.single_hop import SingleHopRetriever

RetrieverType = Union[MURREPipeline, SingleHopRetriever, CrushRetriever]


def build_retriever(
    encoder: SGPTEncoder,
    llm: LLMGenerator | None = None,
    method: Method | None = None,
    crush_collective: bool = True,
) -> RetrieverType:
    """Trả về đúng retriever theo `method` (xem enum Method).

    method           : None → đọc cfg.pipeline.method. Truyền thẳng vào để chọn
                       method mà KHÔNG phải sửa config, cũng không phải tạm ghi đè
                       cfg — đây là cách methods/{murre,crush,single_hop}.py và
                       runner.run_offline() dùng.
    crush_collective : chỉ dùng cho Method.CRUSH — xem CrushRetriever.collective.
    """
    if method is None:
        try:
            method = Method(cfg.pipeline.method)
        except ValueError:
            raise ValueError(
                f"pipeline.method={cfg.pipeline.method!r} không hợp lệ. "
                f"Chỉ nhận: {Method.values()}"
            ) from None

    # Lưu ý khi thêm nhánh: LUÔN viết dạng có chấm (Method.MURRE). Viết trống tên
    # (case MURRE) thì Python hiểu là "bắt mọi giá trị rồi gán vào biến MURRE" —
    # nhánh đầu tiên sẽ nuốt hết mọi method mà không báo lỗi gì.
    match method:
        case Method.MURRE:
            assert llm is not None, "MURRE cần LLMGenerator cho pha Removal."
            rewriter: QueryRewriter = QueryRewriter(llm=llm)
            return MURREPipeline(encoder=encoder, rewriter=rewriter)

        case Method.SINGLE_HOP:
            return SingleHopRetriever(encoder=encoder)

        case Method.CRUSH:
            assert llm is not None, "CRUSH cần LLMGenerator để hallucinate schema."
            return CrushRetriever(encoder=encoder, llm=llm, collective=crush_collective)

        case _:
            raise ValueError(f"Method mới chưa xử lý trong build_retriever: {method}")
