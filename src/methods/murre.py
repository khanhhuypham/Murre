"""MURRE in-process: retrieve đa hop bằng beam search → xếp hạng bảng → sinh SQL.

    hop 0      retrieve toàn corpus, giữ top_k_pool làm pool   → _retrieve()
    hop 1..H   Completing Tables (LLM) → retrieve trong pool
               → mở rộng beam → tỉa top-B                      → run()
    scoring    Score_Path → Score_Table                        → _rank()
    sinh SQL   top-K bảng → prompt zero-shot → LLM             → generate_sql()

`run()` cùng giao diện với các method khác (methods/factory.py).
Chạy thử một câu:  python -m methods.murre
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from core.rewriter import QueryRewriter
from models.retrieval import RetrievedTable
from utils import logger
from utils.scoring import path_score


@dataclass(frozen=True)
class Hit:
    """Một bảng retrieve được: schema, cosine similarity, vị trí trong corpus."""

    schema: str
    similarity: float
    index: int


@dataclass(frozen=True)
class Path:
    """Một beam: chuỗi bảng đã đi qua kèm similarity từng bước.

    `query`   : câu truy vấn cho hop kế tiếp.
    `stopped` : LLM trả "None" → ngừng mở rộng, vẫn được chấm điểm.
    """

    schemas: Tuple[str, ...]
    sims: Tuple[float, ...]
    query: str
    stopped: bool = False

    @property
    def score(self) -> float:
        """Score_Path = Π Norm(sim) dọc đường đi, tính trong log-space."""
        return path_score(similarities=self.sims)

    def extend(self, hit: Hit) -> "Path":
        """Beam mới = beam này nối thêm `hit` vào cuối đường đi."""
        return Path(
            schemas=self.schemas + (hit.schema,),
            sims=self.sims + (hit.similarity,),
            query=self.query,
            stopped=False,
        )


class MurreRetriever:
    """MURRE đầy đủ, chạy trong RAM, không đọc/ghi file trung gian.

        retriever = MurreRetriever(encoder=encoder, rewriter=rewriter)
        tables = retriever.run(question, corpus, schema_embeddings)
        sql = retriever.generate_sql(question, tables)

    beam_size / max_hop / top_k_pool để None thì lấy từ cfg.pipeline.
    """

    def __init__(
        self,
        encoder: SGPTEncoder,
        rewriter: Optional[QueryRewriter] = None,
        llm: Optional[LLMGenerator] = None,
        *,
        beam_size: Optional[int] = None,
        max_hop: Optional[int] = None,
        top_k_pool: Optional[int] = None,
    ) -> None:
        self.encoder: SGPTEncoder = encoder
        self.llm: Optional[LLMGenerator] = llm

        self.beam_size: int = beam_size if beam_size is not None else cfg.pipeline.beam_size
        self.max_hop: int = max_hop if max_hop is not None else cfg.pipeline.max_hop
        self.top_k_pool: int = top_k_pool if top_k_pool is not None else cfg.pipeline.top_k_pool

        self.use_removal: bool = cfg.pipeline.ablation.removal
        self.use_early_stop: bool = cfg.pipeline.ablation.early_stop

        # Removal cần LLM; Splice (ablation w/o removal) thì không.
        self.rewriter: Optional[QueryRewriter] = rewriter
        if self.use_removal and self.rewriter is None:
            if llm is None:
                raise ValueError(
                    "MurreRetriever cần `rewriter` hoặc `llm` khi ablation.removal=true. "
                    "Đặt removal=false để chạy chế độ Splice không cần LLM."
                )
            self.rewriter = QueryRewriter(llm=llm)

    # -- §3.2 Retrieval ------------------------------------------------------
    @staticmethod
    def _subqueries(query: str) -> List[str]:
        """Tách câu truy vấn thành từng sub-query, mỗi dòng một cái.

        Removal có thể trả nhiều bảng trên nhiều dòng; encode riêng để không
        trung bình hóa các bảng khác nhau vào một vector.
        """
        lines: List[str] = [line.strip() for line in query.splitlines() if line.strip()]
        return lines or [query.strip()]

    def _retrieve(
        self,
        query: str,
        corpus: List[str],
        embeddings: torch.Tensor,
        pool: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
    ) -> List[Hit]:
        """Top-k bảng theo cosine similarity, trong toàn corpus hoặc trong `pool`.

        Nhiều sub-query → mỗi bảng lấy điểm cao nhất (max, không cộng/trung bình).
        """
        queries: List[str] = self._subqueries(query=query)
        q: torch.Tensor = F.normalize(
            input=self.encoder.encode(texts=queries, is_query=True), p=2, dim=1
        )

        indices: List[int] = list(pool) if pool is not None else list(range(len(corpus)))
        docs: torch.Tensor = F.normalize(input=embeddings[indices], p=2, dim=1)

        # [n_subquery, n_doc] → max theo sub-query → [n_doc]
        sims: torch.Tensor = (q @ docs.T).amax(dim=0)
        k: int = min(top_k or self.top_k_pool, sims.size(0))
        values, positions = torch.topk(input=sims, k=k)

        return [
            Hit(schema=corpus[indices[p]], similarity=float(v), index=indices[p])
            for v, p in zip(values.tolist(), positions.tolist())
        ]

    # -- §3.3 Completing Tables ---------------------------------------------
    def _next_query(self, question: str, schemas: Sequence[str]) -> Tuple[str, bool]:
        """Sinh câu truy vấn cho hop kế từ câu hỏi gốc + các bảng đã có.

        Trả về (câu truy vấn, có early stop hay không). Luôn dùng câu hỏi gốc,
        không dùng câu viết lại ở hop trước, để nhiễu không cộng dồn.
        """
        if not self.use_removal:
            # Ablation w/o removal: nối câu hỏi với bảng đã có, không gọi LLM.
            joined: str = " | ".join(schemas)
            return (f"{question} | {joined}" if joined else question), False

        assert self.rewriter is not None
        out: str = self.rewriter.rewrite(question=question, retrieved_schemas=list(schemas)).strip()
        stopped: bool = self.use_early_stop and QueryRewriter.is_early_stop(rewrite_output=out)
        return out, stopped

    # -- §3.5 Scoring --------------------------------------------------------
    @staticmethod
    def _rank(beams: Sequence[Path], hop0: Sequence[Hit]) -> List[RetrievedTable]:
        """Xếp hạng bảng: Score_Table(t) = max Score_Path trong các beam giữ lại.

        Chỉ tính trên beam đã chọn — thêm ứng viên hop 0 như đường độ dài 1 thì
        đường ngắn luôn thắng và multi-hop mất tác dụng.

        Sau đó nối đuôi dự phòng (ngoài paper, để đủ recall@20): các bảng còn lại
        của pool hop 0, giữ nguyên thứ tự hop 0, luôn xếp dưới bảng có đường đi.
        """
        best: Dict[str, float] = {}
        for path in beams:
            score: float = path.score
            for schema in path.schemas:
                if score > best.get(schema, float("-inf")):
                    best[schema] = score

        ranked: List[RetrievedTable] = [
            RetrievedTable(schema=schema, score=score)
            for schema, score in sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ]

        # Đuôi: giữ nguyên thứ hạng hop 0, dịch điểm xuống dưới bảng thấp nhất đang có
        # để thứ tự trong file kết quả đọc được như một danh sách xếp hạng liền mạch.
        floor: float = min(best.values()) if best else 0.0
        for rank, hit in enumerate(hop0, start=1):
            if hit.schema not in best:
                ranked.append(RetrievedTable(schema=hit.schema, score=floor - rank))
        return ranked

    # -- Toàn bộ pipeline ----------------------------------------------------
    def run(
        self,
        question: str,
        corpus: List[str],
        schema_embeddings: torch.Tensor,
        verbose: bool = False,
    ) -> List[RetrievedTable]:
        """Chạy đủ pipeline retrieve cho MỘT câu hỏi → bảng xếp theo Score_Table."""
        # ── Hop 0: toàn corpus ───────────────────────────────────────────────
        hop0: List[Hit] = self._retrieve(
            query=question, corpus=corpus, embeddings=schema_embeddings,
        )
        if not hop0:
            return []

        pool: List[int] = [h.index for h in hop0]

        # §3.4: chỉ top-B ứng viên hop 0 trở thành beam khởi đầu.
        beams: List[Path] = [
            Path(schemas=(h.schema,), sims=(h.similarity,), query=question)
            for h in hop0[: self.beam_size]
        ]

        if verbose:
            logger.info(f"[MURRE] Hop 0: pool {len(pool)} bảng | beam {len(beams)}")
            for h in hop0[:3]:
                logger.info(f"         {h.similarity:.4f}  {h.schema}")

        # ── Hop 1..max_hop ───────────────────────────────────────────────────
        for hop in range(1, self.max_hop + 1):
            active: List[Path] = [b for b in beams if not b.stopped]
            if not active:
                if verbose:
                    logger.info(f"[MURRE] Hop {hop}: mọi beam đã dừng sớm → kết thúc")
                break

            candidates: List[Path] = []
            for i, beam in enumerate(active):
                query, stopped = self._next_query(question=question, schemas=beam.schemas)

                if stopped:
                    # Đường đi đủ bảng: đóng băng, vẫn giữ để chấm điểm.
                    beams = [replace(b, stopped=True) if b is beam else b for b in beams]
                    if verbose:
                        logger.info(f"[MURRE] Hop {hop} beam {i}: early stop")
                    continue

                hits: List[Hit] = self._retrieve(
                    query=query, corpus=corpus, embeddings=schema_embeddings, pool=pool,
                )
                grown: Path = replace(beam, query=query)
                candidates.extend(
                    grown.extend(hit=h) for h in hits if h.schema not in beam.schemas
                )

            if not candidates:
                break

            candidates.sort(key=lambda p: p.score, reverse=True)
            beams = candidates[: self.beam_size] + [b for b in beams if b.stopped]

            if verbose:
                logger.info(f"[MURRE] Hop {hop}: {len(candidates)} ứng viên → giữ {len(beams)} beam")

        results: List[RetrievedTable] = self._rank(beams=beams, hop0=hop0)
        if verbose:
            logger.info(f"[MURRE] Xong: {len(results)} bảng | top-3:")
            for r in results[:3]:
                logger.info(f"         {r.score:.4f}  {r.schema}")
        return results

    # -- Bước cuối: sinh SQL -------------------------------------------------
    def generate_sql(
        self,
        question: str,
        tables: Sequence[RetrievedTable],
        top_k: Optional[int] = None,
    ) -> str:
        """Sinh SQL từ top-K bảng đã retrieve. Cần truyền `llm=` khi dựng retriever."""
        if self.llm is None:
            raise ValueError("generate_sql() cần LLMGenerator — truyền `llm=` khi dựng retriever.")
        k: int = top_k if top_k is not None else cfg.pipeline.top_n_output
        return build_sql(llm=self.llm, question=question, schemas=[t.schema for t in tables[:k]])


def build_sql(llm: LLMGenerator, question: str, schemas: Sequence[str]) -> str:
    """Prompt zero-shot → SQL. Hàm rời để /sql dùng được với mọi method.

    Dùng chung hàm dựng prompt với steps/infer.py để hai đường sinh SQL không
    lệch nhau; import trong thân hàm vì steps/ kéo theo dataset/loader.
    """
    from dataset.loader import load_tables
    from steps.infer import _ZERO_SHOT_PROMPT, _build_table_prompt
    from utils.schema import build_db_index

    if not schemas:
        raise ValueError("Không có bảng nào để sinh SQL.")

    dbs: Dict[str, Dict] = build_db_index(tables=load_tables())
    prompt: str = _ZERO_SHOT_PROMPT.format(
        table=_build_table_prompt(schema_strings=list(schemas), dbs_dict=dbs),
        question=question,
    )
    sql: str = llm.generate(prompt=prompt).strip()
    if not sql.lower().startswith("select"):
        sql = "select " + sql
    return " ".join(sql.split())


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP — thử MURRE trên MỘT câu hỏi
# =============================================================================
if __name__ == "__main__":
    from dataset.loader import load_dev
    from enums import Method
    from methods.runner import run_one_question

    # --- Chỉnh trực tiếp mấy biến này để test ------------------------------
    DATASET: Optional[str] = None    # None → theo general.dataset
    if DATASET:
        cfg.general.dataset = DATASET

    # Câu số 21 của dev.json cần 3 bảng — đúng ca multi-hop mà MURRE nhắm tới.
    QUESTION: Optional[str] = load_dev()[21]["utterance"]
    TOP_N: int = 5
    LLM_PROFILE: Optional[str] = None
    VERBOSE: bool = True
    BEAM_SIZE: Optional[int] = None  # None → theo pipeline.beam_size
    MAX_HOP: Optional[int] = None    # None → theo pipeline.max_hop
    # ----------------------------------------------------------------------

    # Ghi đè TRƯỚC khi dựng pipeline: __init__ đọc cfg một lần duy nhất.
    if BEAM_SIZE is not None:
        cfg.pipeline.beam_size = BEAM_SIZE
    if MAX_HOP is not None:
        cfg.pipeline.max_hop = MAX_HOP

    ab = cfg.pipeline.ablation
    print(
        f"\n  Cấu hình: beam_size={cfg.pipeline.beam_size}, max_hop={cfg.pipeline.max_hop}, "
        f"top_k_pool={cfg.pipeline.top_k_pool}, removal={ab.removal}, "
        f"tabulation={ab.tabulation}, early_stop={ab.early_stop}"
    )

    run_one_question(
        method=Method.MURRE,
        question=QUESTION,
        top_n=TOP_N,
        verbose=VERBOSE,
        llm_profile=LLM_PROFILE,
    )
