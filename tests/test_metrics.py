"""recall@k và complete_recall@k (§4.1)."""
from __future__ import annotations

import pytest

from models.records import ResultRecord
from models.retrieval import RetrievedTable
from utils.metrics import compute_recall, compute_res


def _record(retrieved: list[str], gold: list[str]) -> ResultRecord:
    return ResultRecord(
        utterance="q",
        gold=gold,
        retrieved=RetrievedTable.to_rows(
            tables=[RetrievedTable(schema=s, score=0.0) for s in retrieved]
        ),
    )


def test_recall_counts_by_set_so_duplicates_do_not_exceed_one() -> None:
    assert compute_recall(pred_list=["a", "a"], gold_list=["a"]) == 1.0


def test_recall_of_empty_gold_is_zero() -> None:
    assert compute_recall(pred_list=["a"], gold_list=[]) == 0.0


def test_complete_recall_needs_every_gold_table_in_top_k() -> None:
    data = [_record(retrieved=["a", "b", "c"], gold=["a", "c"])]

    scores = compute_res(top_k=[2, 3], data=data)
    # top-2 = [a, b] → thiếu "c"
    assert scores.recall_at(2) == pytest.approx(0.5)
    assert scores.complete_recall_at(2) == 0.0
    # top-3 = [a, b, c] → đủ
    assert scores.recall_at(3) == 1.0
    assert scores.complete_recall_at(3) == 1.0


def test_k_larger_than_the_list_uses_the_whole_list() -> None:
    """pred_list[:k] tự cắt; k lớn hơn độ dài KHÔNG được tính thành 0."""
    scores = compute_res(top_k=[99], data=[_record(retrieved=["a"], gold=["a"])])
    assert scores.recall_at(99) == 1.0


def test_metrics_average_over_questions() -> None:
    data = [
        _record(retrieved=["a"], gold=["a"]),   # recall 1.0
        _record(retrieved=["x"], gold=["a"]),   # recall 0.0
    ]
    assert compute_res(top_k=[1], data=data).recall_at(1) == pytest.approx(0.5)
