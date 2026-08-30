"""methods/single_hop.py — Baseline Single-hop (§4.2, Table 2/3): retrieve trực tiếp
top-N bảng từ câu hỏi gốc, không có hop thứ hai, không Removal/Supplement.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from config import cfg
from core.encoder import SGPTEncoder
from models.retrieval import RetrievedTable


class SingleHopRetriever:
    """Baseline đối chiếu: chỉ retrieve một lần bằng câu hỏi gốc."""

    def __init__(self, encoder: SGPTEncoder) -> None:
        self.encoder: SGPTEncoder = encoder
        self.top_k_pool: int = cfg.pipeline.top_k_pool

    def run(
        self,
        question: str,
        corpus: List[str],
        schema_embeddings: torch.Tensor,
        verbose: bool = False,
    ) -> List[RetrievedTable]:
        q_emb: torch.Tensor = self.encoder.encode(texts=[question], is_query=True)
        q_norm_vec: torch.Tensor = F.normalize(input=q_emb, p=2, dim=1)
        d_norm_vec: torch.Tensor = F.normalize(input=schema_embeddings, p=2, dim=1)

        scores: torch.Tensor = (q_norm_vec @ d_norm_vec.T).squeeze(0)
        k: int = min(self.top_k_pool, scores.size(0))
        top_vals, top_idx = torch.topk(input=scores, k=k)

        if verbose:
            from utils import logger
            logger.info(f"[SingleHop] Top-3: {[corpus[i] for i in top_idx.tolist()[:3]]}")

        return [
            RetrievedTable(schema=corpus[idx], score=float(val))
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP — test riêng method này với 1 câu hỏi
# =============================================================================
if __name__ == "__main__":
    # Chạy riêng method này trên MỘT câu hỏi, không cần đổi pipeline.method.
    # Trình tự chạy nằm ở methods/runner.run_one_question() — xem file đó.
    from methods.runner import run_one_question
    from dataset.loader import load_dev
    from enums import Method

    # --- Chỉnh trực tiếp mấy biến này để test ------------------------------
    DATASET: str | None = None       # None → dùng general.dataset của src/config.py
    if DATASET:
        # Phải set TRƯỚC load_dev(), nếu không sẽ đọc dev.json của dataset cũ.
        cfg.general.dataset = DATASET

    # Lấy câu số 21 của dev.json (câu cần 3 bảng). Đổi số trong [] để test câu khác;
    # đặt None để lấy câu đầu tiên; hoặc gõ thẳng một chuỗi tự viết.
    QUESTION: str | None = load_dev()[21]["utterance"]
    TOP_N: int = 5                   # số bảng in ra
    # ----------------------------------------------------------------------

    run_one_question(method=Method.SINGLE_HOP, question=QUESTION, top_n=TOP_N)
