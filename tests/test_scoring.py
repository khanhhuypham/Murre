"""Công thức chấm điểm của paper — Eq. C.1, Eq. D.2, Algorithm 1."""
from __future__ import annotations

import pytest

from utils import scoring


def test_normalize_maps_cosine_range_to_unit_range() -> None:
    # Eq. C.1: Norm(s) = (s + 1) / 2, đưa [-1, 1] về [0, 1].
    assert scoring.normalize(s=-1.0) == 0.0
    assert scoring.normalize(s=0.0) == 0.5
    assert scoring.normalize(s=1.0) == 1.0


def test_path_score_is_product_of_normalized_similarities() -> None:
    # Eq. D.2: Score_Path = Π Norm(sim).
    assert scoring.path_score(similarities=[1.0, 1.0]) == 1.0
    assert scoring.path_score(similarities=[0.0, 0.0]) == pytest.approx(0.25)


def test_empty_path_scores_one() -> None:
    """Tích rỗng = 1.0 — chưa đi bước nào thì chưa mất mát gì."""
    assert scoring.path_score(similarities=[]) == 1.0


def test_longer_paths_score_lower() -> None:
    """Hệ quả của việc bám paper: Norm ≤ 1 nên đường đi càng dài điểm càng thấp."""
    short: float = scoring.path_score(similarities=[0.8])
    long: float = scoring.path_score(similarities=[0.8, 0.8])
    assert long < short


def test_score_tables_takes_max_over_paths_containing_the_table() -> None:
    # Algorithm 1: Score_Table(t) = max Score_Path trên mọi đường đi chứa t.
    paths = [
        (("a", "b"), (0.2, 0.2)),   # điểm thấp
        (("a",), (0.9,)),           # điểm cao, cũng chứa "a"
    ]
    scores = scoring.score_tables(paths=paths)
    assert scores["a"] == pytest.approx(scoring.path_score(similarities=[0.9]))
    assert scores["b"] == pytest.approx(scoring.path_score(similarities=[0.2, 0.2]))


def test_score_tables_ignores_tables_not_on_any_path() -> None:
    assert scoring.score_tables(paths=[]) == {}
