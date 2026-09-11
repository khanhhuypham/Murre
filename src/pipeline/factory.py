"""pipeline/factory.py — NƠI DUY NHẤT ráp một bộ retrieval sẵn sàng chạy.

CLI (cli.py → pipeline/runner.py) và API (api/dependencies.py) đều ráp qua đây;
khác nhau duy nhất ở chỗ AI SỞ HỮU encoder/LLM — truyền vào thì dùng lại, để None
thì tự tạo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from core.corpus import build_corpus, load_embeddings
from core.encoder import SentenceEncoder
from core.llm import LLMGenerator
from core.rewriter import QueryRewriter
from pipeline.retriever import MurreRetriever


@dataclass
class LoadedDataset:
    """Một dataset đã sẵn sàng chạy: retriever + corpus + embeddings.

    Ba thứ này phải khớp nhau (`embs[i]` là vector của `corpus[i]`), nên gói chung
    một object để không lỡ tay ghép embeddings dataset này với corpus dataset kia.
    """
    retriever: MurreRetriever
    corpus: List[str]
    embs: torch.Tensor


def build_dataset(
    dataset: Optional[str] = None,
    encoder: Optional[SentenceEncoder] = None,
    llm: Optional[LLMGenerator] = None,
    llm_profile: Optional[str] = None,
) -> LoadedDataset:
    """Ráp đủ bộ để chạy một dataset: retriever + corpus + embeddings.

        dataset : None → dataset đang chọn (general.dataset).
        encoder : None → SentenceEncoder.get(dataset) — instance này đã dùng
                  lại theo tên profile, nên spider và bird (cùng profile) chỉ nạp
                  model một lần. Chỉ TRUYỀN VÀO khi cần một encoder khác hẳn
                  (test, hay hai cấu hình song song).
        llm     : None → tự tạo. Truyền vào để dùng lại — API giữ một LLM trong
                  app.state cho mọi dataset.

    LLM dựng TRƯỚC corpus: endpoint chưa bật thì hỏng ngay, không mất công encode
    cả corpus rồi mới báo lỗi.
    """
    if llm is None:
        llm = LLMGenerator(profile=llm_profile)
    if encoder is None:
        encoder = SentenceEncoder.get(dataset=dataset)

    corpus: List[str] = build_corpus(dataset=dataset)
    embs: torch.Tensor = load_embeddings(encoder=encoder, corpus=corpus, dataset=dataset)
    retriever = MurreRetriever(
        encoder=encoder,
        rewriter=QueryRewriter(llm=llm, dataset=dataset),
        llm=llm,
    )
    return LoadedDataset(retriever=retriever, corpus=corpus, embs=embs)
