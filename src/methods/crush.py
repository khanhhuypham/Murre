"""methods/crush.py — Baseline CRUSH (Kothyari et al., 2023; §4.2, Table 2/3):
dùng LLM "hallucinate" (đoán) một schema giả định trực tiếp từ câu hỏi gốc (không có
ngữ cảnh database nào), rồi retrieve bằng chính văn bản hallucinate đó.

LLM thường đoán ra NHIỀU bảng (mỗi bảng 1 dòng). CRUSH gốc retrieve riêng cho từng
bảng đã đoán rồi hợp nhất kết quả ("collective retrieval") — nếu gộp cả nhiều dòng
thành 1 chuỗi rồi encode 1 lần, embedding bị trung bình hóa giữa các bảng khác nhau
và điểm similarity bị loãng. Vì vậy run() tách từng dòng, encode riêng, và lấy điểm
CAO NHẤT trên mỗi bảng corpus (max-aggregation) — đúng quy ước mà
steps/retrieve.py::retrieve_for_queries() dùng cho multi-subquery của MURRE, nên hai
phương pháp so sánh được với nhau một cách công bằng.

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

    def __init__(
        self,
        encoder: SGPTEncoder,
        llm: LLMGenerator,
        collective: bool = True,
    ) -> None:
        self.encoder: SGPTEncoder = encoder
        self.llm: LLMGenerator = llm
        self.top_k_pool: int = cfg.pipeline.top_k_pool

        # collective=True  : mỗi bảng LLM đoán ra là 1 query riêng, hợp nhất bằng max
        # collective=False : gộp toàn bộ output thành 1 query duy nhất (để ablation)
        self.collective: bool = collective

        # Lưu schema hallucinate gần nhất để in ra / phân tích lỗi sau này
        # (xem LLM đoán sai schema ở đâu).
        self.last_hallucinated: str = ""

        prompt_path: str = get_dataset_path(key="prompt_crush")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template: str = "\n".join(line.rstrip("\n") for line in f)

    def _hallucinate_schema(self, question: str) -> str:
        prompt: str = self.prompt_template.format(question=question)
        return self.llm.generate(prompt=prompt)

    @staticmethod
    def _split_schemas(hallucinated: str) -> List[str]:
        """Tách output của LLM thành danh sách bảng đã đoán (1 bảng / 1 dòng).

        Đồng thời bỏ nhãn "Guessed Schema:" mà model yếu hay lặp lại, và bỏ các
        gạch đầu dòng ("- ", "* ", "1. ") nếu model tự thêm vào.
        """
        schemas: List[str] = []
        for raw_line in hallucinated.splitlines():
            line: str = raw_line.strip()
            if not line:
                continue
            # Bỏ nhãn dẫn nếu model lặp lại prompt
            if ":" in line:
                head, _, tail = line.partition(":")
                if "schema" in head.lower() or "question" in head.lower():
                    line = tail.strip()
            line = line.lstrip("-*•0123456789. ").strip()
            if line:
                schemas.append(line)

        # Không parse được dòng nào → dùng nguyên văn output để không mất câu hỏi
        return schemas or [hallucinated.strip() or " "]

    def run(
        self,
        question: str,
        corpus: List[str],
        schema_embeddings: torch.Tensor,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        hallucinated: str = self._hallucinate_schema(question=question)
        self.last_hallucinated = hallucinated

        queries: List[str] = (
            self._split_schemas(hallucinated) if self.collective else [hallucinated]
        )

        if verbose:
            from utils import logger
            logger.info(f"[CRUSH] {len(queries)} bảng đã đoán: {queries}")

        q_emb: torch.Tensor = self.encoder.encode(texts=queries, is_query=True)
        q_norm_vec: torch.Tensor = F.normalize(input=q_emb, p=2, dim=1)
        d_norm_vec: torch.Tensor = F.normalize(input=schema_embeddings, p=2, dim=1)

        # (số bảng đoán, số bảng corpus) → lấy max theo chiều query = collective retrieval
        sim_matrix: torch.Tensor = q_norm_vec @ d_norm_vec.T
        scores: torch.Tensor = sim_matrix.max(dim=0).values

        k: int = min(self.top_k_pool, scores.size(0))
        top_vals, top_idx = torch.topk(input=scores, k=k)

        return [
            {"schema": corpus[idx], "score": float(val)}
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP — test riêng method này với 1 câu hỏi
# =============================================================================
if __name__ == "__main__":
    from core.corpus import prepare
    from dataset.loader import load_dev, resolve_question
    from methods._cli import print_results

    # --- Chỉnh trực tiếp mấy biến này để test ------------------------------
    DATASET: str | None = None       # None → dùng general.dataset của config.yaml
    if DATASET:
        # Phải set TRƯỚC load_dev(), nếu không sẽ đọc dev.json của dataset cũ.
        cfg.general.dataset = DATASET

    # Lấy câu số 21 của dev.json (câu cần 3 bảng). Đổi số trong [] để test câu khác;
    # đặt None để lấy câu đầu tiên; hoặc gõ thẳng một chuỗi tự viết (khi đó không tra
    # được gold nên sẽ không có ✓ và recall).
    QUESTION: str | None = load_dev()[21]["utterance"]
    TOP_N: int = 5                   # số bảng in ra
    LLM_PROFILE: str | None = None   # None → dùng llm.active_profile
    COLLECTIVE: bool = True          # True: retrieve riêng từng bảng LLM đoán
    #                                  False: gộp mọi bảng thành 1 query duy nhất
    # ----------------------------------------------------------------------

    encoder, corpus, embs = prepare(dataset=DATASET)
    question: str = resolve_question(question=QUESTION)

    retriever = CrushRetriever(
        encoder=encoder,
        llm=LLMGenerator(profile=LLM_PROFILE),
        collective=COLLECTIVE,
    )
    results = retriever.run(
        question=question, corpus=corpus, schema_embeddings=embs, verbose=False,
    )

    # Phần đặc trưng của CRUSH: xem LLM đã đoán ra schema gì
    print()
    print(f"  LLM hallucinate ({'collective' if COLLECTIVE else 'gộp 1 query'}):")
    for line in retriever._split_schemas(retriever.last_hallucinated):
        print(f"    · {line}")

    print_results(method="CRUSH", question=question, results=results, top_n=TOP_N)
