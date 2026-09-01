"""methods/build.py — NƠI DUY NHẤT ráp một bộ retrieval sẵn sàng chạy.

    build_retriever()  method → retriever. Chỉ CHỌN class, không nạp gì, tức thì.
    build_dataset()    + LLM + corpus + embeddings → LoadedDataset. Nặng: đọc đĩa,
                       gọi mạng, có thể phải encode cả corpus.

CLI (methods/runner.py) và API (api/dependencies.py) đều ráp qua đây; khác nhau
duy nhất ở chỗ AI SỞ HỮU encoder/LLM — truyền vào thì dùng lại, để None thì tự tạo.

Cả 3 retriever có chung một giao diện:
    run(question, corpus, schema_embeddings, verbose=False) -> List[RetrievedTable]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from config import cfg
from core.corpus import build_corpus, load_embeddings
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from core.rewriter import QueryRewriter
from enums import Method
from methods.crush import CrushRetriever
from methods.murre import MurreRetriever
from methods.single_hop import SingleHopRetriever

RetrieverType = Union[MurreRetriever, SingleHopRetriever, CrushRetriever]


@dataclass
class LoadedDataset:
    """Một dataset đã sẵn sàng chạy: retriever + corpus + embeddings.

    Ba thứ này phải khớp nhau (`embs[i]` là vector của `corpus[i]`), nên gói chung
    một object để không lỡ tay ghép embeddings dataset này với corpus dataset kia.
    """

    retriever: RetrieverType
    corpus: List[str]
    embs: torch.Tensor


def build_retriever(
    encoder: SGPTEncoder,
    llm: LLMGenerator | None = None,
    method: Method | None = None,
    crush_collective: bool = True,
) -> RetrieverType:
    """Trả về đúng retriever theo `method` (xem enum Method).
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
            return MurreRetriever(encoder=encoder, rewriter=rewriter, llm=llm)

        case Method.SINGLE_HOP:
            return SingleHopRetriever(encoder=encoder)

        case Method.CRUSH:
            assert llm is not None, "CRUSH cần LLMGenerator để hallucinate schema."
            return CrushRetriever(encoder=encoder, llm=llm, collective=crush_collective)

        case _:
            raise ValueError(f"Method mới chưa xử lý trong build_retriever: {method}")


def build_dataset(
    method: Method,
    dataset: Optional[str] = None,
    encoder: Optional[SGPTEncoder] = None,
    llm: Optional[LLMGenerator] = None,
    llm_profile: Optional[str] = None,
    crush_collective: bool = True,
) -> LoadedDataset:
    """Ráp đủ bộ để chạy một dataset: retriever + corpus + embeddings.

        dataset : None → dataset đang chọn (general.dataset).
        encoder : None → tạo mới. TRUYỀN VÀO để dùng lại — API giữ đúng MỘT
                  encoder trong app.state cho mọi dataset, không nạp model 2 lần.
        llm     : None → tự tạo nếu method cần. Truyền vào để dùng lại, cùng lý do.

    LLM dựng TRƯỚC corpus: endpoint chưa bật thì hỏng ngay, không mất công encode
    cả corpus rồi mới báo lỗi.
    """
    if llm is None and method.needs_llm:
        llm = LLMGenerator(profile=llm_profile)
    if encoder is None:
        encoder = SGPTEncoder()

    corpus: List[str] = build_corpus(dataset=dataset)
    embs: torch.Tensor = load_embeddings(encoder=encoder, corpus=corpus, dataset=dataset)
    retriever: RetrieverType = build_retriever(
        encoder=encoder,
        llm=llm,
        method=method,
        crush_collective=crush_collective,
    )
    return LoadedDataset(retriever=retriever, corpus=corpus, embs=embs)
