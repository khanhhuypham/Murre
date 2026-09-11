# =============================================================================
# utils/scoring.py — Toàn bộ công thức tính điểm của MURRE, khai báo ĐÚNG MỘT LẦN
#
# Bám ĐÚNG paper (COLING 2025), không bám code phát hành của tác giả:
#
#   normalize()    Eq. C.1 (Appendix C)      Norm(s) = (s + 1) / 2
#   path_score()   Eq. D.2 (Appendix D)      Score_Path = Π Norm(sim) trên đường đi
#   score_tables() Algorithm 1 (Appendix E)  Score_Table(t) = max Score_Path trên
#                                            mọi đường đi có chứa t
#
# Norm(s) ∈ [0, 1] nên Score_Path là tích các số ≤ 1 → đường đi càng dài điểm
# càng thấp; bảng tìm ở hop 1 thường có điểm cao hơn bảng tìm ở hop 3.
# =============================================================================
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

# Một đường đi rút gọn còn đúng hai thứ mà công thức cần: bảng nào và sim bao nhiêu.
PathScores = Tuple[Sequence[str], Sequence[float]]


def normalize(s: float) -> float:
    """Norm(s) = (s + 1) / 2 — Eq. C.1: cosine [-1, 1] → xác suất [0, 1]."""
    return (s + 1.0) / 2.0


def path_score(similarities: Iterable[float]) -> float:
    """Score_Path = Π_j Norm(sim_j) — Eq. D.2. Đường đi rỗng → 1.0."""
    score: float = 1.0
    for s in similarities:
        score *= normalize(s=s)
    return score


def score_tables(paths: Iterable[PathScores]) -> Dict[str, float]:
    """Algorithm 1 — Score_Table(t) = max Score_Path trên mọi đường đi chứa t.

    `paths` là MỌI đường đi đã sinh ra (kể cả bị tỉa và dừng sớm), mỗi phần tử
    là cặp (danh sách schema, danh sách similarity) cùng độ dài.
    """
    best: Dict[str, float] = {}
    for schemas, sims in paths:
        score: float = path_score(similarities=sims)
        for schema in schemas:
            if score > best.get(schema, -1.0):
                best[schema] = score
    return best
