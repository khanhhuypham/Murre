# =============================================================================
# utils/scoring.py — Toàn bộ công thức tính điểm của MURRE, khai báo ĐÚNG MỘT LẦN
#
# Ba hàm, dùng ở hai chỗ khác nhau trong methods/murre.py:
#
#   pruning_score()  tỉa beam sau mỗi hop        (rewrite/sample.py của tác giả)
#   table_score()    xếp hạng bảng ở cuối        (rewrite/score.py của tác giả)
#   normalize()      hàm pos(), dùng bởi 2 cái trên
#
# LƯU Ý QUAN TRỌNG: pruning_score và table_score KHÔNG cùng công thức. Tỉa beam
# dùng similarity thô, xếp hạng bảng thì cho qua pos() trước. Bản cài đặt gốc đúng
# như vậy — không phải chép nhầm.
# =============================================================================
from __future__ import annotations

import math
from typing import Iterable, Sequence

# Chặn dưới trước khi lấy log: cosine âm làm math.log ném ValueError. Bản gốc không
# chặn nên sẽ vỡ; ở đây chỉ đỡ cho khỏi vỡ, không đổi thứ hạng thực tế.
_LOG_FLOOR: float = 1e-9


def normalize(s: float) -> float:
    """pos(s) = (s + 2) / 2 — đúng hàm `pos()` trong rewrite/score.py của tác giả.

    Đây là chỗ paper và code KHÔNG khớp, và ta chọn theo CODE:

        Appendix C viết   Norm(s) = (s + 1) / 2  → [-1,1] về [0,1], là xác suất.
        rewrite/score.py  pos(s)  = (s + 2) / 2  → [-1,1] về [0.5,1.5], KHÔNG phải
                                                   xác suất.

    Hệ quả lớn: log(pos(s)) DƯƠNG khi s > 0, nên đường đi càng dài điểm càng CAO —
    ngược hẳn công thức trong paper (tích xác suất thì càng dài càng thấp). Bản code
    mới là bản sinh ra Bảng 2, nên bám theo nó.
    """
    return (s + 2) / 2


def _log_pos(s: float) -> float:
    """log(pos(s)), đã chặn dưới."""
    return math.log(max(normalize(s=s), _LOG_FLOOR))


def pruning_score(similarities: Iterable[float]) -> float:
    """Điểm để TỈA BEAM sau mỗi hop = Σ log(sim thô) — rewrite/sample.py.

    Không gọi pos(). Chỉ dùng để so các nhánh với nhau trong cùng một hop, mà các
    nhánh trong cùng hop luôn dài bằng nhau, nên hằng số cộng thêm không đổi thứ tự.
    """
    return sum(math.log(max(s, _LOG_FLOOR)) for s in similarities)


def path_score(similarities: Iterable[float]) -> float:
    """Phần đóng góp của một đường đi vào điểm bảng = Σ log(pos(sim)).

    Tách riêng vì table_score() gọi nó cho mọi ứng viên trên cùng một đường đi —
    tính một lần rồi dùng lại.
    """
    return sum(_log_pos(s=s) for s in similarities)


def table_score(candidate_similarity: float, path_similarities: Sequence[float]) -> float:
    """Điểm của một bảng ứng viên khi nối nó vào cuối một đường đi — rewrite/score.py.

        có đường đi   → log(pos(sim_ứng_viên)) + Σ log(pos(sim trên đường đi))
        đường đi rỗng → pos(sim_ứng_viên), KHÔNG lấy log

    Nhánh thứ hai xảy ra khi câu hỏi dừng sớm ngay hop đầu, lúc đó chưa có đường đi
    nào. score.py xử lý bất nhất như vậy (một bên có log, một bên không) — giữ
    nguyên để khớp bản gốc. Không gây lệch thứ hạng vì hai nhánh không bao giờ
    xuất hiện cùng lúc trong một câu hỏi.
    """
    if not path_similarities:
        return normalize(s=candidate_similarity)
    return _log_pos(s=candidate_similarity) + path_score(similarities=path_similarities)
