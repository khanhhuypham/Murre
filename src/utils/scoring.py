# =============================================================================
# utils/scoring.py — Hai công thức tính điểm của paper, khai báo ĐÚNG MỘT LẦN
#
# Trước đây mỗi công thức nằm rải ở nhiều file (methods/murre.py, steps/score.py,
# steps/rewrite.py) và đã bắt đầu lệch nhau. Gom về đây để sửa một chỗ là mọi
# đường chạy (Option 1 in-process và Option 2 batch) cùng đổi theo.
# =============================================================================
from __future__ import annotations

import math
from typing import Iterable

# Chặn dưới trước khi lấy log: Norm(s) = 0 khi similarity = -1, mà log(0) thì
# ValueError. Giá trị này chỉ để không vỡ, không ảnh hưởng xếp hạng thực tế.
_LOG_FLOOR: float = 1e-9


def normalize(s: float) -> float:
    """Norm(s) = (s + 1) / 2 — đưa cosine similarity từ [-1, 1] về [0, 1].

    Công thức Appendix C của paper.
    """
    return (s + 1) / 2


def path_score(similarities: Iterable[float]) -> float:
    """Score_Path = Π P̂(t|q) dọc một nhánh beam (§3.5), tính trong log-space.

    Dùng log để tránh underflow: tích nhiều xác suất nhỏ sẽ về 0 trong float.
    Score_Path = Σ log(Norm(sim_k)) cho k = 1..H

    `similarities` là similarity THÔ (chưa Norm) của từng hop trên nhánh.
    """
    return sum(math.log(max(normalize(s=s), _LOG_FLOOR)) for s in similarities)
