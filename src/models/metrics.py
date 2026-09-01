"""models/metrics.py — Định dạng điểm số của một lần chạy (file score.json).

    utils/metrics.py::compute_res  ──tính──>  MetricScores
    methods/runner.py              ──ghi───>  score.json
    api/evaluator.py               ──đọc───>  recall_at(k) / complete_recall_at(k)

QUY TẮC KHI SỬA: tên field CHÍNH LÀ khóa ghi ra score.json (xem to_dict).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricScores:
    """recall@k và complete_recall@k, trung bình trên toàn bộ dataset."""

    recall: Dict[int, float]
    complete_recall: Dict[int, float]

    def recall_at(self, k: int) -> float:
        """Tỉ lệ bảng gold tìm được trong top-k."""
        return self.recall[k]

    def complete_recall_at(self, k: int) -> float:
        """Tỉ lệ câu hỏi có TẤT CẢ bảng gold trong top-k."""
        return self.complete_recall[k]

    def to_dict(self) -> Dict[str, Dict[int, float]]:
        return {
            "recall": self.recall,
            "complete_recall": self.complete_recall
        }
