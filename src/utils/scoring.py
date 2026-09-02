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
# Chỉ MỘT công thức điểm đường đi cho cả hai chỗ dùng:
#   - tỉa beam sau mỗi hop  (§3.3: "choose B results ... method detailed in §3.5")
#   - xếp hạng bảng ở cuối  (§3.5)
# Paper không phân biệt hai chỗ này, nên ở đây cũng không.
#
# HỆ QUẢ CỦA VIỆC BÁM PAPER: Norm(s) ∈ [0, 1] nên Score_Path là TÍCH các số ≤ 1 →
# đường đi càng dài điểm càng THẤP. Đó đúng là ý paper (Score_Path là xác suất,
# Eq. D.2). Bảng tìm được ở hop 1 vì thế thường có điểm cao hơn bảng tìm ở hop 3 —
# trừ khi similarity của nó thấp hơn hẳn.
# =============================================================================
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

# Một đường đi rút gọn còn đúng hai thứ mà công thức cần: bảng nào và sim bao nhiêu.
PathScores = Tuple[Sequence[str], Sequence[float]]


def normalize(s: float) -> float:
    """Norm(s) = (s + 1) / 2 — Equation C.1, Appendix C.

    Đưa cosine similarity từ [-1, 1] về [0, 1] để đọc như một xác suất
    P̂(t_i | q_h,b) trong Equation 3.1. Norm tăng theo s nên similarity càng cao
    thì xác suất bảng được retrieve càng lớn.
    """
    return (s + 1.0) / 2.0


def path_score(similarities: Iterable[float]) -> float:
    """Score_Path(Path_h,b) = Π_j Norm(sim_j) — Equation D.2, Appendix D.

    Xác suất của cả đường đi = tích xác suất từng bước. Đường đi rỗng → 1.0
    (tích rỗng), đúng nghĩa "chưa đi bước nào thì chưa mất mát gì".
    """
    score: float = 1.0
    for s in similarities:
        score *= normalize(s=s)
    return score


def score_tables(paths: Iterable[PathScores]) -> Dict[str, float]:
    """Algorithm 1 (Appendix E) — chấm điểm mọi bảng nằm trên các đường đi.

        Score_Table(t_i) = max_{path ∈ Path_ti} Score_Path(path)

    `paths` là all_paths của Algorithm 1: MỌI đường đi đã sinh ra trong quá trình
    tìm kiếm (mọi hop, kể cả đường đi bị tỉa và đường đi dừng sớm), mỗi phần tử là
    cặp (danh sách schema, danh sách similarity) cùng độ dài và cùng thứ tự hop.

    Bảng không nằm trên đường đi nào thì KHÔNG có điểm và không xuất hiện trong kết
    quả — paper chỉ xếp hạng bảng đã được retrieve.
    """
    best: Dict[str, float] = {}
    for schemas, sims in paths:
        score: float = path_score(similarities=sims)
        for schema in schemas:
            if score > best.get(schema, -1.0):
                best[schema] = score
    return best
