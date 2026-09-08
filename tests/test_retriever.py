"""MurreRetriever: beam search, early stop, xếp hạng — không cần model thật.

Encoder và LLM đều được thay bằng bản giả nên test chạy trong mili-giây và kết quả
chỉ phụ thuộc vào logic của retriever.
"""
from __future__ import annotations

from typing import List

import pytest
import torch

from pipeline.retriever import MurreRetriever

CORPUS: List[str] = ["db.a(x)", "db.b(y)", "db.c(z)"]


class FakeEncoder:
    """Mỗi văn bản → một vector one-hot, chọn theo từ khoá xuất hiện trong text.

    Nhờ vậy cosine similarity là tất định và test viết được kỳ vọng chính xác.
    """

    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        rows: List[List[float]] = []
        for t in texts:
            if "b(" in t or "second" in t:
                rows.append([0.0, 1.0, 0.0])
            elif "c(" in t or "third" in t:
                rows.append([0.0, 0.0, 1.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return torch.tensor(rows)


class ScriptedRewriter:
    """Trả lần lượt các câu đã dựng sẵn; hết kịch bản thì báo Early Stop."""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs: List[str] = list(outputs)
        self.calls: List[List[str]] = []

    def rewrite(self, question: str, retrieved_schemas: List[str]) -> str:
        self.calls.append(list(retrieved_schemas))
        return self.outputs.pop(0) if self.outputs else "None"

    @staticmethod
    def is_early_stop(rewrite_output: str) -> bool:
        return rewrite_output.strip().startswith("None")

    # MurreRetriever gọi is_early_stop QUA instance rewriter (không qua class), nên
    # bản giả này thay được cả hai nửa của pha Removal.


@pytest.fixture
def embeddings() -> torch.Tensor:
    return FakeEncoder().encode(texts=CORPUS)


def _retriever(rewriter: ScriptedRewriter, beam: int = 2, hops: int = 2) -> MurreRetriever:
    return MurreRetriever(
        encoder=FakeEncoder(), rewriter=rewriter, beam_size=beam, max_hop=hops,
    )


def test_requires_a_rewriter_or_an_llm() -> None:
    with pytest.raises(ValueError, match="Removal"):
        MurreRetriever(encoder=FakeEncoder())


def test_max_hop_one_returns_top_b_tables(embeddings: torch.Tensor) -> None:
    r = _retriever(rewriter=ScriptedRewriter(outputs=[]), beam=2, hops=1)
    out = r.run(question="first", corpus=CORPUS, schema_embeddings=embeddings)
    assert [t.schema for t in out][:1] == ["db.a(x)"]
    assert len(out) == 2  # beam_size bảng, không gọi LLM lần nào


def test_second_hop_finds_a_table_the_first_hop_missed(embeddings: torch.Tensor) -> None:
    rewriter = ScriptedRewriter(outputs=["second", "second"])
    r = _retriever(rewriter=rewriter, beam=1, hops=2)

    out = r.run(question="first", corpus=CORPUS, schema_embeddings=embeddings)

    assert {t.schema for t in out} == {"db.a(x)", "db.b(y)"}
    # §3.4: Removal luôn nhận CÂU HỎI GỐC + toàn bộ bảng trên đường đi.
    assert rewriter.calls == [["db.a(x)"]]


def test_early_stop_freezes_the_branch_but_keeps_it_scored(
    embeddings: torch.Tensor,
) -> None:
    r = _retriever(rewriter=ScriptedRewriter(outputs=["None"]), beam=1, hops=3)
    out = r.run(question="first", corpus=CORPUS, schema_embeddings=embeddings)
    # Nhánh dừng ở hop 1 nhưng vẫn nằm trong all_paths → vẫn được xếp hạng.
    assert [t.schema for t in out] == ["db.a(x)"]


def test_results_are_sorted_by_score_descending(embeddings: torch.Tensor) -> None:
    r = _retriever(rewriter=ScriptedRewriter(outputs=["second", "third"]), beam=2, hops=2)
    out = r.run(question="first", corpus=CORPUS, schema_embeddings=embeddings)
    assert [t.score for t in out] == sorted((t.score for t in out), reverse=True)


def test_empty_corpus_returns_nothing() -> None:
    r = _retriever(rewriter=ScriptedRewriter(outputs=[]))
    out = r.run(question="first", corpus=[], schema_embeddings=torch.empty(0, 3))
    assert out == []
