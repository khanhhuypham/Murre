"""methods/single_hop.py — Baseline Single-hop (§4.2, Table 2/3): retrieve trực tiếp
top-N bảng từ câu hỏi gốc, không có hop thứ hai, không Removal/Supplement.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from config import cfg
from core.encoder import SGPTEncoder


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
    ) -> List[Dict[str, Any]]:
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
            {"schema": corpus[idx], "score": float(val)}
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]
