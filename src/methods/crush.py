"""methods/crush.py — Baseline CRUSH (Kothyari et al., 2023; §4.2, Table 2/3).

LLM "hallucinate" (đoán) một schema giả định từ câu hỏi gốc, không có ngữ cảnh
database nào, rồi retrieve bằng chính văn bản đoán được đó.

LLM thường đoán ra nhiều bảng, mỗi bảng 1 dòng: run() encode TỪNG dòng thành một
query riêng rồi lấy điểm cao nhất cho mỗi bảng corpus (max-aggregation). Lý do và
cờ COLLECTIVE: xem HUONG_DAN.md mục 6b.

LƯU Ý: prompts/{dataset}_crush.txt là prompt tự suy luận, KHÔNG phải nguyên văn của
Kothyari et al. — số liệu chạy ra không khớp chính xác Table 2/3 (HUONG_DAN.md mục 12).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from models.retrieval import RetrievedTable


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

        prompt_path: str = cfg.dataset_paths.prompt_crush
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
    ) -> List[RetrievedTable]:
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
            RetrievedTable(schema=corpus[idx], score=float(val))
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP — test riêng method này với 1 câu hỏi
# =============================================================================
if __name__ == "__main__":
    from core.corpus import prepare
    from dataset.loader import load_dev, resolve_question
    from utils.display import print_results

    # --- Chỉnh trực tiếp mấy biến này để test ------------------------------
    DATASET: str | None = None       # None → dùng general.dataset của src/config.py
    if DATASET:
        # Phải set TRƯỚC load_dev(), nếu không sẽ đọc dev.json của dataset cũ.
        cfg.general.dataset = DATASET

    # Lấy câu số 21 của dev.json (câu cần 3 bảng). Đổi số trong [] để test câu khác;
    QUESTION: str | None = load_dev()[21]["utterance"]
    TOP_N: int = 5
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

    print()
    print(f"  LLM hallucinate ({'collective' if COLLECTIVE else 'gộp 1 query'}):")
    for line in retriever._split_schemas(retriever.last_hallucinated):
        print(f"    · {line}")

    print_results(method="CRUSH", question=question, results=results, top_n=TOP_N)
