"""Cache embeddings: chỉ dùng lại khi đúng corpus và đúng model."""
from __future__ import annotations

from typing import List

import torch

from core.corpus import corpus_fingerprint, load_embeddings

MODEL: str = "some/encoder"
CORPUS: List[str] = ["db.a(x, y)", "db.b(z)"]


class CountingEncoder:
    """Encoder giả: đếm số lần bị gọi để biết cache có được dùng lại không."""

    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name: str = model_name
        self.calls: int = 0

    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        self.calls += 1
        return torch.ones(len(texts), 4)


def test_fingerprint_changes_with_corpus_content() -> None:
    """Hai corpus CÙNG SỐ LƯỢNG nhưng khác nội dung phải ra vân tay khác nhau.

    Đây đúng là ca ViText2SQL syllable / word: cùng 876 bảng, chỉ khác cách viết
    tên bảng. Đếm số vector thì không phân biệt được.
    """
    a: str = corpus_fingerprint(corpus=["db.kiến trúc sư(x)"], model_name=MODEL)
    b: str = corpus_fingerprint(corpus=["db.kiến_trúc_sư(x)"], model_name=MODEL)
    assert a != b


def test_fingerprint_changes_with_model() -> None:
    assert corpus_fingerprint(corpus=CORPUS, model_name="a") != corpus_fingerprint(
        corpus=CORPUS, model_name="b"
    )


def test_fingerprint_is_stable() -> None:
    assert corpus_fingerprint(corpus=CORPUS, model_name=MODEL) == corpus_fingerprint(
        corpus=list(CORPUS), model_name=MODEL
    )


def test_fingerprint_is_order_sensitive() -> None:
    """embs[i] phải là vector của corpus[i] — đảo thứ tự là gán điểm nhầm bảng."""
    assert corpus_fingerprint(corpus=CORPUS, model_name=MODEL) != corpus_fingerprint(
        corpus=CORPUS[::-1], model_name=MODEL
    )


def _cache_to(tmp_path, monkeypatch) -> None:
    """Trỏ paths.embeddings_cache vào tmp_path để test không đụng outputs/ thật."""
    from config import cfg

    monkeypatch.setattr(
        cfg.paths, "embeddings_cache", str(tmp_path / "{dataset}_{model}.pt"),
    )


def test_second_load_reuses_the_cache(tmp_path, monkeypatch) -> None:
    _cache_to(tmp_path, monkeypatch)
    encoder = CountingEncoder()

    first = load_embeddings(encoder=encoder, corpus=CORPUS)
    second = load_embeddings(encoder=encoder, corpus=CORPUS)

    assert encoder.calls == 1
    assert torch.equal(first, second)


def test_a_changed_corpus_re_encodes_instead_of_scoring_wrong(tmp_path, monkeypatch) -> None:
    """Corpus khác mà cùng tên file cache → mã hoá lại, không nạp nhầm vector cũ."""
    _cache_to(tmp_path, monkeypatch)
    encoder = CountingEncoder()

    load_embeddings(encoder=encoder, corpus=CORPUS)
    load_embeddings(encoder=encoder, corpus=["db.khác(x, y)", "db.b(z)"])

    assert encoder.calls == 2


def test_a_changed_model_re_encodes(tmp_path, monkeypatch) -> None:
    _cache_to(tmp_path, monkeypatch)

    first = CountingEncoder(model_name="a")
    load_embeddings(encoder=first, corpus=CORPUS)

    # Cùng nhãn thư mục nhưng model khác → không được dùng lại vector cũ.
    second = CountingEncoder(model_name="b")
    load_embeddings(encoder=second, corpus=CORPUS)

    assert second.calls == 1
