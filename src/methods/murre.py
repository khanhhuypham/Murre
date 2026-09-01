"""MURRE: retrieve bảng đa hop bằng beam search → xếp hạng bảng → sinh SQL.

VÒNG ĐỜI MỘT CÂU HỎI
--------------------
    turn 0      _retrieve() trên toàn corpus bằng câu hỏi gốc, lấy 100 bảng.
                100 bảng này thành POOL — mọi turn sau chỉ tìm trong đây.

    turn 1..H-1 lặp 3 bước:
                  a. _grow()    mỗi nhánh nở ≤3B ứng viên, gộp lại, giữ top-B
                  b. _removal() LLM xoá thông tin đã có khỏi câu hỏi
                                → nếu LLM báo "đủ rồi" thì dừng cả câu hỏi
                  c. _retrieve() lại trong pool bằng câu vừa viết lại

    cuối        _rank() chấm điểm mọi ứng viên của turn cuối → danh sách xếp hạng

TỪ VỰNG
-------
    Hit           một bảng retrieve được (schema + similarity + vị trí corpus)
    RetrievalPath một đường đi: dãy bảng đã qua + similarity từng bước (Eq D.1)
    Beam          một nhánh đang sống: đường đi + toàn bộ ứng viên nó retrieve được
    turn          danh sách Beam ở một hop. Turn 0 có 1 phần tử (đường đi rỗng),
                  các turn sau có B phần tử.

BÁM CODE TÁC GIẢ, KHÔNG BÁM CHỮ TRONG PAPER
-------------------------------------------
Ở 5 chỗ dưới đây, github.com/zhxlia/Murre làm khác paper. Ta theo CODE vì đó là
bản đã sinh ra Bảng 2 — số chạy được mới so sánh trực tiếp được.

    | Chỗ            | Paper viết              | Code làm — ta theo cái này      |
    |----------------|-------------------------|---------------------------------|
    | Chuẩn hoá      | Norm(s) = (s+1)/2 (App.C)| pos(s) = (s+2)/2               |
    | Phạm vi hop ≥1 | mọi bảng (§3.3)         | khoá trong pool 100 của turn 0  |
    | Nhánh mỗi beam | B × B (§3.3)            | 3B × B rồi tỉa còn B            |
    | Câu cho Removal| luôn câu gốc (§3.4)     | câu viết lại của hop trước      |
    | Chấm điểm bảng | bảng trên path (Alg.1)  | mọi ứng viên của turn cuối      |

Nguồn: retrieve/retrieve.py, rewrite/sample.py, rewrite/score.py, slurm/run.sh.
Công thức điểm nằm ở utils/scoring.py.

`run()` cùng giao diện với các method khác (methods/build.py).
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
from utils import logger, scoring


# =============================================================================
# Kiểu dữ liệu
# =============================================================================
@dataclass(frozen=True)
class Hit:
    """Một bảng retrieve được."""
    schema: str
    similarity: float
    index: int  # vị trí trong corpus


@dataclass(frozen=True)
class RetrievalPath:
    """Path_h,b của paper (Equation D.1): dãy bảng đã đi qua + similarity từng bước.

    `schemas[i]` và `sims[i]` luôn cùng độ dài và cùng thứ tự hop.
    `query` là câu truy vấn đã dùng ở hop hiện tại — hop kế sẽ viết lại TỪ CÂU NÀY,
    không phải từ câu hỏi gốc (xem bảng đối chiếu ở đầu file).
    """

    schemas: Tuple[str, ...]
    sims: Tuple[float, ...]
    query: str

    def extend(self, hit: Hit) -> "RetrievalPath":
        """Đường đi mới = đường này nối thêm `hit` vào cuối."""
        return RetrievalPath(
            schemas=self.schemas + (hit.schema,),
            sims=self.sims + (hit.similarity,),
            query=self.query,
        )


@dataclass(frozen=True)
class Beam:
    """Một nhánh đang sống: đường đi, kèm TOÀN BỘ ứng viên mà đường đi đó retrieve ra.

    Giữ cả danh sách ứng viên vì hai bước dùng nó theo hai cách khác nhau:
        _grow()  chỉ lấy 3B ứng viên đầu để nở nhánh
        _rank()  dùng CẢ danh sách để chấm điểm bảng
    """

    path: RetrievalPath
    candidates: List[Hit]


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

        # sample.py lấy `retrieved[:3*top_k]` với top_k = beam size.
        self.expand_width: int = 3 * self.beam_size

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

    # =========================================================================
    # Bước a — Retrieval (§3.3, Equation 3.1)
    # =========================================================================
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
        doc_embeddings: torch.Tensor,
        top_k: int,
        index_map: Optional[Sequence[int]] = None,
    ) -> List[Hit]:
        """Top-k bảng theo cosine similarity.

            doc_embeddings : ma trận ĐÃ chuẩn hoá L2. Turn 0 truyền cả corpus, các
                             turn sau truyền ma trận con của pool.
            index_map      : ánh xạ hàng của ma trận con về chỉ số corpus.
                             None = ma trận đầy đủ, hàng i chính là corpus[i].

        Nhiều sub-query → mỗi bảng lấy điểm CAO NHẤT (max, không cộng/trung bình).
        """
        queries: List[str] = self._subqueries(query=query)
        q: torch.Tensor = F.normalize(
            input=self.encoder.encode(texts=queries, is_query=True), p=2, dim=1,
        )

        # [n_subquery, n_doc] → max theo sub-query → [n_doc]
        sims: torch.Tensor = (q @ doc_embeddings.T).amax(dim=0)
        values, positions = torch.topk(input=sims, k=min(top_k, sims.size(0)))

        def to_corpus_index(row: int) -> int:
            return index_map[row] if index_map is not None else row

        return [
            Hit(schema=corpus[to_corpus_index(p)], similarity=float(v), index=to_corpus_index(p))
            for v, p in zip(values.tolist(), positions.tolist())
        ]

    # =========================================================================
    # Bước b — Removal / Completing Tables (§3.4)
    # =========================================================================
    def _next_query(self, query: str, schemas: Sequence[str]) -> Tuple[str, bool]:
        """Nhờ LLM xoá thông tin của `schemas` khỏi `query`.

        Trả về (câu truy vấn cho hop kế, LLM có báo "đủ bảng rồi" hay không).

        `query` là câu của HOP HIỆN TẠI — tức câu đã viết lại ở hop trước, không
        phải câu hỏi gốc. Xem bảng đối chiếu ở đầu file.
        """
        if not self.use_removal:
            # Ablation w/o removal: nối câu hỏi với bảng đã có, không gọi LLM.
            joined: str = " | ".join(schemas)
            return (f"{query} | {joined}" if joined else query), False

        assert self.rewriter is not None
        out: str = self.rewriter.rewrite(question=query, retrieved_schemas=list(schemas)).strip()
        stopped: bool = self.use_early_stop and QueryRewriter.is_early_stop(rewrite_output=out)
        return out, stopped

    # =========================================================================
    # Ba bước của một hop
    # =========================================================================
    def _grow(self, turn: Sequence[Beam]) -> List[RetrievalPath]:
        """Nở nhánh rồi tỉa: mỗi beam sinh ≤3B đường đi mới, gộp lại, giữ top-B.

        Bỏ qua bảng đã nằm trên chính đường đi đó — một đường không đi lại bảng cũ.
        """
        grown: List[RetrievalPath] = [
            beam.path.extend(hit=hit)
            for beam in turn
            for hit in beam.candidates[: self.expand_width]
            if hit.schema not in beam.path.schemas
        ]
        grown.sort(key=lambda p: scoring.pruning_score(similarities=p.sims), reverse=True)
        return grown[: self.beam_size]

    def _removal(
        self,
        paths: Sequence[RetrievalPath],
        corpus: List[str],
        pool_docs: torch.Tensor,
        pool: Sequence[int],
    ) -> Optional[List[Beam]]:
        """Removal + retrieve cho từng đường đi → turn kế tiếp.

        Trả về None khi LLM báo dừng sớm. Early stop là quyết định của CẢ CÂU HỎI
        chứ không của riêng một nhánh: score.py đặt end_turn[câu] = turn-1 ngay khi
        BẤT KỲ nhánh nào báo dừng, rồi chấm điểm bằng turn TRƯỚC đó. Nên trả None là
        đủ — chỗ gọi giữ nguyên turn hiện tại và đem đi chấm điểm.
        """
        next_turn: List[Beam] = []
        for path in paths:
            query, stopped = self._next_query(query=path.query, schemas=path.schemas)
            if stopped:
                return None
            next_turn.append(Beam(
                path=replace(path, query=query),
                candidates=self._retrieve(
                    query=query,
                    corpus=corpus,
                    doc_embeddings=pool_docs,
                    top_k=self.top_k_pool,
                    index_map=pool,
                ),
            ))
        return next_turn

    # =========================================================================
    # Bước cuối — Score (§3.5)
    # =========================================================================
    @staticmethod
    def _rank(turn: Sequence[Beam], pool_hits: Sequence[Hit]) -> List[RetrievedTable]:
        """Chấm điểm MỌI ứng viên của turn cuối, không chỉ các bảng nằm trên đường đi.

        Với mỗi beam và mỗi ứng viên `x` mà beam đó retrieve được, điểm là
        `scoring.table_score(x, đường đi)` — tức chấm x như thể nối nó vào cuối
        đường đi. Điểm đó gán cho CẢ `x` LẪN mọi bảng trên đường đi, lấy max nếu
        một bảng được chấm nhiều lần.

        Bảng nào thuộc pool mà không lần nào được chấm thì giữ 0.0 và tự rơi xuống
        đáy, vì log(pos(sim)) > 0 với mọi sim > 0.

        `pool_hits` (= kết quả turn 0) quyết định TẬP BẢNG được xếp hạng: mọi ứng
        viên ở các turn sau đều lấy từ pool này nên không có bảng nào lọt ra ngoài.
        """
        best: Dict[str, float] = {hit.schema: 0.0 for hit in pool_hits}

        for beam in turn:
            for hit in beam.candidates:
                score: float = scoring.table_score(
                    candidate_similarity=hit.similarity,
                    path_similarities=beam.path.sims,
                )
                for schema in (hit.schema, *beam.path.schemas):
                    if score > best[schema]:
                        best[schema] = score

        return [
            RetrievedTable(schema=schema, score=score)
            for schema, score in sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ]

    # =========================================================================
    # Ráp lại
    # =========================================================================
    def run(
        self,
        question: str,
        corpus: List[str],
        schema_embeddings: torch.Tensor,
        verbose: bool = False,
    ) -> List[RetrievedTable]:
        """Chạy đủ pipeline retrieve cho MỘT câu hỏi → bảng xếp theo điểm giảm dần."""
        # Chuẩn hoá corpus MỘT lần cho cả câu hỏi thay vì mỗi lần _retrieve.
        docs: torch.Tensor = F.normalize(input=schema_embeddings, p=2, dim=1)

        # --- turn 0: toàn corpus, câu hỏi GỐC không tabulate (Appendix K) ----
        pool_hits: List[Hit] = self._retrieve(
            query=question,
            corpus=corpus,
            doc_embeddings=docs,
            top_k=self.top_k_pool
        )
        if not pool_hits:
            return []

        # Pool cố định cho mọi turn sau. run.sh truyền `--last_retrieved_file turn0`
        # cho CẢ hop 2 lẫn hop 3, nên luôn là kết quả turn 0, không phải turn trước.
        pool: List[int] = [hit.index for hit in pool_hits]
        pool_docs: torch.Tensor = docs[pool]

        turn: List[Beam] = [Beam(
            path=RetrievalPath(schemas=(), sims=(), query=question),
            candidates=pool_hits,
        )]
        if verbose:
            self._log_turn_0(pool_hits=pool_hits)

        # --- turn 1 .. max_hop-1 --------------------------------------------
        # max_hop ĐẾM CẢ turn 0: max_hop=3 → turn 0, 1, 2, tức chỉ 2 vòng ở đây.
        # Table 5 của paper cũng vậy — "max hop 1" nghĩa là single-hop.
        for hop in range(1, self.max_hop):
            paths: List[RetrievalPath] = self._grow(turn=turn)
            if not paths:
                break

            next_turn: Optional[List[Beam]] = self._removal(
                paths=paths, corpus=corpus, pool_docs=pool_docs, pool=pool,
            )
            if next_turn is None:
                if verbose:
                    logger.info(f"[MURRE] Turn {hop}: early stop → chấm bằng turn {hop - 1}")
                break

            turn = next_turn
            if verbose:
                logger.info(f"[MURRE] Turn {hop}: giữ {len(turn)} nhánh")

        results: List[RetrievedTable] = self._rank(turn=turn, pool_hits=pool_hits)
        if verbose:
            self._log_results(results=results)
        return results

    # -- Log ------------------------------------------------------------------
    @staticmethod
    def _log_turn_0(pool_hits: Sequence[Hit]) -> None:
        logger.info(f"[MURRE] Turn 0: pool {len(pool_hits)} bảng")
        for hit in pool_hits[:3]:
            logger.info(f"  {hit.similarity:.4f}  {hit.schema}")

    @staticmethod
    def _log_results(results: Sequence[RetrievedTable]) -> None:
        logger.info(f"[MURRE] Xong: {len(results)} bảng | top-3:")
        for table in results[:3]:
            logger.info(f"   {table.score:.4f}  {table.schema}")

    # =========================================================================
    # Bước cuối cùng của paper — sinh SQL
    # =========================================================================
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
