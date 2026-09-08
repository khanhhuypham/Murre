"""Chọn encoder và gộp token — không tải model thật.

build_encoder() là NƠI DUY NHẤT quyết định dùng lớp nào. Chọn nhầm lớp không nổ
lỗi, chỉ ra vector vô nghĩa (SPECB áp lên model BERT, hay bỏ tiền tố "query: "
của E5), nên phần đáng test nhất là cái quyết định đó.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

import core.encoder as encoder_module
from config import cfg
from core.encoder import (
    SentenceEncoder,
    SGPTEncoder,
    _encode_batched,
    build_encoder,
    plan_batches,
)


@pytest.fixture
def encoder_cfg():
    """Profile encoder của spider, trả về nguyên trạng sau mỗi test (cfg toàn cục)."""
    profile = cfg.encoder_for("spider")
    saved: Dict[str, Any] = profile.model_dump()
    yield profile
    for key, value in saved.items():
        setattr(profile, key, value)


class _StubLoad:
    """Thay _load() để build_encoder() chạy được mà không tải model từ HuggingFace."""

    def __init__(self) -> None:
        self.seen: List[str] = []

    def __call__(self, model_name: str):
        self.seen.append(model_name)
        return object(), object(), "cpu"


def test_type_sgpt_builds_the_paper_encoder(encoder_cfg, monkeypatch) -> None:
    monkeypatch.setattr(encoder_module, "_load", _StubLoad())
    monkeypatch.setattr(SGPTEncoder, "_encode_single_char_as_token_id", lambda self, char: 1)
    encoder_cfg.type = "sgpt"
    assert isinstance(build_encoder(), SGPTEncoder)


def test_type_sentence_builds_the_multilingual_encoder(encoder_cfg, monkeypatch) -> None:
    monkeypatch.setattr(encoder_module, "_load", _StubLoad())
    encoder_cfg.type = "sentence"
    assert isinstance(build_encoder(), SentenceEncoder)


def test_unknown_type_fails_loudly_with_the_valid_ones(encoder_cfg, monkeypatch) -> None:
    monkeypatch.setattr(encoder_module, "_load", _StubLoad())
    encoder_cfg.type = "phobert"
    with pytest.raises(ValueError, match="sentence"):
        build_encoder()


def test_model_name_comes_from_the_dataset_profile(encoder_cfg, monkeypatch) -> None:
    stub = _StubLoad()
    monkeypatch.setattr(encoder_module, "_load", stub)
    encoder_cfg.type = "sentence"
    encoder_cfg.model_name = "intfloat/multilingual-e5-base"

    build_encoder(dataset="spider")
    assert stub.seen == ["intfloat/multilingual-e5-base"]


def test_each_dataset_builds_its_own_encoder_class(monkeypatch) -> None:
    """Không còn encoder toàn cục: dataset nào ra lớp của dataset đó."""
    monkeypatch.setattr(encoder_module, "_load", _StubLoad())
    monkeypatch.setattr(SGPTEncoder, "_encode_single_char_as_token_id", lambda self, char: 1)

    assert isinstance(build_encoder(dataset="spider"), SGPTEncoder)
    assert isinstance(build_encoder(dataset="vitext2sql"), SentenceEncoder)


# --- mean pooling ----------------------------------------------------------
def test_mean_pooling_ignores_padding() -> None:
    """Token padding lọt vào trung bình là kéo lệch vector của mọi câu ngắn."""
    hidden = torch.tensor([[[2.0, 2.0], [4.0, 4.0], [99.0, 99.0]]])
    tokens = {"attention_mask": torch.tensor([[1, 1, 0]])}

    out = SentenceEncoder._mean_pooling(tokens=tokens, hidden=hidden)
    assert torch.allclose(out, torch.tensor([[3.0, 3.0]]))


def test_mean_pooling_of_an_all_padding_row_is_finite() -> None:
    """Chia cho 0 sẽ ra NaN và lan ra toàn bộ ma trận điểm."""
    hidden = torch.tensor([[[1.0, 1.0]]])
    tokens = {"attention_mask": torch.tensor([[0]])}

    out = SentenceEncoder._mean_pooling(tokens=tokens, hidden=hidden)
    assert torch.isfinite(out).all()


# --- tiền tố theo vai trò --------------------------------------------------
class _RecordingSentenceEncoder(SentenceEncoder):
    """Bắt lại đúng chuỗi đưa vào tokenizer, để kiểm tra tiền tố đã được nối."""

    def __init__(self, query_prefix: str, doc_prefix: str) -> None:
        self.batch_size = 8
        self.max_batch_tokens = 1024
        self.max_length = 512
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.device = "cpu"
        self.seen: List[str] = []

        outer = self

        def tokenizer(text, padding=None, **kwargs):
            # Gọi lần đầu là để ĐO ĐỘ DÀI (padding=False), lần sau mới là batch thật.
            if padding is False:
                return {"input_ids": [[0] * len(t) for t in text]}
            outer.seen.extend(text)
            return {"attention_mask": torch.ones(len(text), 1)}

        self.tokenizer = tokenizer
        self.model = lambda **kw: type(
            "O", (), {"last_hidden_state": torch.ones(len(kw["attention_mask"]), 1, 2)}
        )()


def test_query_and_document_get_their_own_prefix() -> None:
    """E5 học với "query: " / "passage: "; thiếu tiền tố là điểm tụt hẳn."""
    enc = _RecordingSentenceEncoder(query_prefix="query: ", doc_prefix="passage: ")

    enc.encode(texts=["câu hỏi"], is_query=True)
    enc.encode(texts=["db.bảng(x)"], is_query=False)

    assert enc.seen == ["query: câu hỏi", "passage: db.bảng(x)"]


def test_empty_prefix_leaves_the_text_alone() -> None:
    enc = _RecordingSentenceEncoder(query_prefix="", doc_prefix="")
    enc.encode(texts=["nguyên văn"], is_query=True)
    assert enc.seen == ["nguyên văn"]


# --- chia batch theo ngân sách token ---------------------------------------
def test_every_index_lands_in_exactly_one_batch() -> None:
    lengths = [5, 100, 7, 300, 2, 50]
    batches = plan_batches(lengths=lengths, max_items=4, max_tokens=1000)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))


def test_batches_respect_the_item_cap() -> None:
    batches = plan_batches(lengths=[1] * 10, max_items=3, max_tokens=10**9)
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_a_long_text_gets_a_small_batch() -> None:
    """Chính là ca làm sập tiến trình: 1 schema 598 token không được kéo theo 255 câu."""
    lengths = [20] * 500 + [598]
    batches = plan_batches(lengths=lengths, max_items=256, max_tokens=16384)

    longest_batch = next(b for b in batches if 500 in b)
    assert len(longest_batch) * 598 <= 16384


def test_peak_batch_cost_is_bounded_by_the_budget() -> None:
    lengths = [20] * 590 + [598, 500, 400, 300, 250, 200, 150]
    batches = plan_batches(lengths=lengths, max_items=256, max_tokens=16384)

    peak = max(len(b) * max(lengths[i] for i in b) for b in batches)
    assert peak <= 16384
    # Cách cũ (số câu cố định) tốn gấp gần 10 lần.
    assert 256 * 598 > 9 * peak


def test_an_oversized_single_text_still_gets_its_own_batch() -> None:
    """Một câu vượt trần thì vẫn phải chạy — bỏ nó đi là mất schema khỏi corpus."""
    batches = plan_batches(lengths=[50_000], max_items=256, max_tokens=16384)
    assert batches == [[0]]


def test_empty_input_plans_nothing() -> None:
    assert plan_batches(lengths=[], max_items=8, max_tokens=100) == []


def test_results_come_back_in_the_original_order() -> None:
    """Xếp sai thứ tự = embs[i] không còn là vector của corpus[i].

    Không có lỗi nào nổ ra, chỉ là mọi điểm số gán nhầm bảng — nên test bám đúng
    vào đây. Model giả trả về chính giá trị số của văn bản để đối chiếu được.
    """
    texts = [str(i) for i in range(20)]
    # Độ dài so le → plan_batches chắc chắn phải xáo thứ tự.
    lengths = [(i * 7) % 20 + 1 for i in range(20)]

    def forward(batch):
        return torch.tensor([[float(t)] for t in batch])

    out = _encode_batched(
        texts=texts, lengths=lengths, max_items=3, max_tokens=12, forward=forward,
    )
    assert out.flatten().tolist() == [float(i) for i in range(20)]


def test_encode_batched_of_nothing_is_empty() -> None:
    out = _encode_batched(
        texts=[], lengths=[], max_items=4, max_tokens=100,
        forward=lambda b: torch.empty(0, 2),
    )
    assert out.numel() == 0
