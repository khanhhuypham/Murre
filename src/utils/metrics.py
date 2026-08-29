# =============================================================================
# utils/metric.py — Các hàm đánh giá hiệu suất retrieval
#
# Gồm 3 metric chính từ bài báo (Section 4.1):
#   - recall@K (r@K)    : tỉ lệ bảng đúng được tìm thấy trong top-K kết quả
#   - complete recall=K : tỉ lệ câu hỏi mà TẤT CẢ bảng đúng đều trong top-K
#   - compute_res()     : tính tổng hợp cả hai metric cho toàn bộ dataset
# =============================================================================

from typing import Dict, List

from models.records import ResultRecord


def compute_recall(pred_list: List[str], gold_list: List[str]) -> float:
    """
    Tính recall = số bảng đúng tìm được / tổng số bảng đúng.

    Tham số:
        pred_list : danh sách schema dự đoán (có thứ tự)
        gold_list : danh sách schema đúng (ground truth)

    Trả về:
        Giá trị recall trong [0, 1]
    """
    if not gold_list:
        return 0.0
    correct = sum(1 for p in pred_list if p in gold_list)
    return correct / len(gold_list)


def compute_recall_at_k(
    top_k: List[int],
    pred_list: List[str],
    gold_list: List[str],
) -> Dict[int, float]:
    """
    Tính recall@K cho nhiều giá trị K cùng lúc.

    Trả về:
        {K: recall_value} cho mỗi K trong top_k
    """
    return {
        k: compute_recall(pred_list[:k], gold_list)
        for k in top_k
        if k <= len(pred_list)
    }


def compute_complete_recall_at_k(
    top_k: List[int],
    pred_list: List[str],
    gold_list: List[str],
) -> Dict[int, float]:
    """
    Tính complete recall (k=K): 1 nếu TẤT CẢ bảng đúng đều có trong top-K, 0 nếu không.
    Đây là metric nghiêm ngặt hơn recall@K, quan trọng trong text-to-SQL
    vì thiếu 1 bảng là không sinh được SQL đúng.

    Trả về:
        {K: 1.0 hoặc 0.0} cho mỗi K trong top_k
    """
    em: Dict[int, float] = {k: 0.0 for k in top_k}
    for k in top_k:
        if set(gold_list).issubset(set(pred_list[:k])):
            em[k] = 1.0
    return em


def compute_res(
    top_k: List[int],
    data: List[ResultRecord],
) -> Dict[str, Dict[int, float]]:
    """
    Tính trung bình recall@K và complete recall=K trên toàn bộ dataset.

    Tham số:
        top_k : danh sách các giá trị K (ví dụ: [3, 5, 10, 20])
        data  : các record của file result/turn{H}/dev.json — xem models/records.py.
                Đọc từ file thì dùng ResultRecord.from_list(json.load(f)).

    Trả về:
        {
          "recall":          {K: avg_recall@K},
          "complete_recall": {K: avg_complete_recall=K}
        }
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
    return {
        "recall":          {k: v / n for k, v in recall_sum.items()},
        "complete_recall": {k: v / n for k, v in complete_sum.items()},
    }
