"""methods/crush.py — Baseline CRUSH (Kothyari et al., 2023; §4.2, Table 2/3):
dùng LLM "hallucinate" (đoán) một schema giả định trực tiếp từ câu hỏi gốc (không có
ngữ cảnh database nào), rồi retrieve bằng chính văn bản hallucinate đó.

LƯU Ý: paper KHÔNG công khai chi tiết prompt CRUSH gốc trong phần chính (chỉ nhắc
"converting the user question into a table format through hallucination"). Prompt
dùng ở đây (prompts/{dataset}_crush.txt) là suy luận hợp lý theo mô tả đó, KHÔNG
phải nguyên văn prompt của Kothyari et al. (2023) — nếu cần khớp chính xác số liệu
paper, hãy thay bằng prompt gốc từ https://github.com/kothyari/CRUSH4SQL nếu có.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from config import cfg, get_dataset_path
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator


class CrushRetriever:
    """Baseline đối chiếu: LLM hallucinate schema từ câu hỏi, rồi retrieve 1 lần."""

    def __init__(self, encoder: SGPTEncoder, llm: LLMGenerator) -> None:
        self.encoder: SGPTEncoder = encoder
        self.llm: LLMGenerator = llm
        self.top_k_pool: int = cfg.pipeline.top_k_pool

        prompt_path: str = get_dataset_path(key="prompt_crush")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template: str = "\n".join(line.rstrip("\n") for line in f)

    def _hallucinate_schema(self, question: str) -> str:
        prompt: str = self.prompt_template.format(question=question)
        return self.llm.generate(prompt=prompt)

    def run(
        self,
        question: str,
        corpus: List[str],
        schema_embeddings: torch.Tensor,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        hallucinated: str = self._hallucinate_schema(question=question)

        if verbose:
            from utils import logger
            logger.info(f"[CRUSH] Hallucinated schema: {hallucinated!r}")

        q_emb: torch.Tensor = self.encoder.encode(texts=[hallucinated], is_query=True)
        q_norm_vec: torch.Tensor = F.normalize(input=q_emb, p=2, dim=1)
        d_norm_vec: torch.Tensor = F.normalize(input=schema_embeddings, p=2, dim=1)

        scores: torch.Tensor = (q_norm_vec @ d_norm_vec.T).squeeze(0)
        k: int = min(self.top_k_pool, scores.size(0))
        top_vals, top_idx = torch.topk(input=scores, k=k)

        return [
            {"schema": corpus[idx], "score": float(val)}
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]
