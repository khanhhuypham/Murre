# =============================================================================
# utils/metric.py — Các hàm đánh giá hiệu suất retrieval
#
# Gồm 3 metric chính từ bài báo (Section 4.1):
#   - recall@K (r@K)    : tỉ lệ bảng đúng được tìm thấy trong top-K kết quả
#   - complete recall=K : tỉ lệ câu hỏi mà TẤT CẢ bảng đúng đều trong top-K
#   - compute_res()     : tính tổng hợp cả hai metric cho toàn bộ dataset
# =============================================================================

from typing import Dict, List

from models.metrics import MetricScores
from models.records import ResultRecord


def compute_recall(pred_list: List[str], gold_list: List[str]) -> float:
    """recall = số bảng đúng tìm được / tổng số bảng đúng, trong [0, 1]."""
    if not gold_list:
        return 0.0
    # Đếm theo tập hợp: pred_list trùng lặp thì đếm tuyến tính sẽ cho recall > 1.
    return len(set(pred_list) & set(gold_list)) / len(gold_list)


def compute_recall_at_k(
    top_k: List[int],
    pred_list: List[str],
    gold_list: List[str],
) -> Dict[int, float]:
    """recall@K cho nhiều K cùng lúc → {K: recall}.

    K lớn hơn len(pred_list) vẫn tính: `pred_list[:k]` tự cắt đúng.
    """
    return {k: compute_recall(pred_list[:k], gold_list) for k in top_k}


def compute_complete_recall_at_k(
    top_k: List[int],
    pred_list: List[str],
    gold_list: List[str],
) -> Dict[int, float]:
    """complete recall: 1.0 nếu TẤT CẢ bảng đúng đều trong top-K, else 0.0."""
    em: Dict[int, float] = {k: 0.0 for k in top_k}
    for k in top_k:
        if set(gold_list).issubset(set(pred_list[:k])):
            em[k] = 1.0
    return em


def compute_res(
    top_k: List[int],
    data: List[ResultRecord],
) -> MetricScores:
    """Trung bình recall@K và complete recall=K trên toàn bộ dataset.

        top_k : danh sách các K, ví dụ [3, 5, 10, 20]
        data  : record của turn{H}/dev.json (xem models/records.py)
    """
    recall_sum:   Dict[int, float] = {k: 0.0 for k in top_k}
    complete_sum: Dict[int, float] = {k: 0.0 for k in top_k}

    for d in data:
        pred: List[str] = d.schemas
        gold: List[str] = d.gold

        rec: Dict[int, float] = compute_recall_at_k(top_k, pred, gold)
        com: Dict[int, float] = compute_complete_recall_at_k(top_k, pred, gold)

        for k in top_k:
            recall_sum[k]   += rec.get(k, 0.0)
            complete_sum[k] += com.get(k, 0.0)

    n: int = len(data)
    return MetricScores(
        recall={k: v / n for k, v in recall_sum.items()},
        complete_recall={k: v / n for k, v in complete_sum.items()},
    )
