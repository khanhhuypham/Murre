"""MURRE: retrieve bảng đa hop bằng beam search → xếp hạng bảng → sinh SQL.

Bản cài đặt này BÁM THEO PAPER (COLING 2025, §3.2–3.5 + Appendix C/D/E), không
bám theo code phát hành ở github.com/zhxlia/Murre. Xem mục cuối docstring để biết
5 chỗ hai bản khác nhau.

VÒNG ĐỜI MỘT CÂU HỎI (§3.2 — Figure 2)
--------------------------------------
    hop 1       Retrieval: encode CÂU HỎI GỐC, quét TOÀN BỘ corpus, lấy top-B bảng
                → B đường đi, mỗi đường đi dài 1.

    hop 2..H    lặp hai pha của paper:
                  Removal   (§3.4) mỗi đường đi → LLM xoá thông tin các bảng đã có
                                   khỏi CÂU HỎI GỐC, trả về phần còn thiếu dưới
                                   dạng bảng. Trả "None" → đường đi đó dừng sớm,
                                   đóng băng lại nhưng VẪN được đem đi chấm điểm.
                  Retrieval (§3.3) mỗi câu hỏi Removal quét TOÀN BỘ corpus, lấy
                                   top-B bảng → tối đa B×B đường đi mới; tỉa còn
                                   B đường đi tốt nhất cho hop kế.

    cuối        Score (§3.5, Algorithm 1): gom MỌI đường đi đã sinh ra ở mọi hop,
                Score_Path = tích Norm(similarity), Score_Table = max Score_Path
                trên các đường đi chứa bảng đó → xếp hạng giảm dần.

TỪ VỰNG
-------
    Hit           một bảng retrieve được (schema + similarity + vị trí corpus)
    RetrievalPath Path_h,b của paper (Eq. D.1): dãy bảng đã qua + similarity từng
                  bước, kèm câu truy vấn đã dùng ở bước cuối
    frontier      B đường đi sống sót sau khi tỉa, sẽ được nở tiếp ở hop sau
    all_paths     all_paths của Algorithm 1 — MỌI đường đi từng sinh ra, kể cả
                  đường bị tỉa và đường dừng sớm. Đây là đầu vào duy nhất của Score.

KHÁC GÌ SO VỚI CODE PHÁT HÀNH CỦA TÁC GIẢ
-----------------------------------------
Code của tác giả (retrieve/retrieve.py, rewrite/sample.py, rewrite/score.py,
slurm/run.sh) lệch paper ở 5 chỗ. Bản này chọn PAPER ở cả 5:

    | Chỗ             | Code tác giả                  | Ở đây — theo paper           |
    |-----------------|-------------------------------|------------------------------|
    | Chuẩn hoá       | pos(s) = (s+2)/2, rồi lấy log | Norm(s) = (s+1)/2 (Eq. C.1)  |
    | Phạm vi hop ≥2  | khoá trong pool 100 của hop 1 | toàn corpus mỗi hop (§3.3)   |
    | Nhánh mỗi beam  | 3B ứng viên rồi tỉa còn B     | đúng B → B×B rồi tỉa (§3.3)  |
    | Câu cho Removal | câu viết lại của hop trước    | luôn CÂU HỎI GỐC q (§3.4)    |
    | Chấm điểm bảng  | mọi ứng viên của turn cuối    | bảng trên đường đi (Alg. 1)  |

Hệ quả cần biết: danh sách trả về chỉ gồm những bảng THỰC SỰ nằm trên một đường
đi, tối đa B + (H-1)·B² bảng (B=5, H=3 → ≤55, thường ít hơn vì trùng lặp). Muốn
đo r@K với K lớn thì phải tăng beam_size, không còn pool 100 để đệm nữa.

`run()` cùng giao diện với các method khác (methods/build.py).
Chạy thử một câu:  python -m methods.murre
"""
from __future__ import annotations

from dataclasses import dataclass
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
    `query` là câu truy vấn ĐÃ dùng để retrieve bảng cuối cùng của đường đi — hop 1
    là câu hỏi gốc, các hop sau là câu do Removal sinh ra. Chỉ dùng để log/soi.
    """

    schemas: Tuple[str, ...]
    sims: Tuple[float, ...]
    query: str

    def extend(self, hit: Hit, query: str) -> RetrievalPath:
        """Đường đi mới = đường này nối thêm `hit` tìm được bằng `query`."""
        return RetrievalPath(
            schemas=self.schemas + (hit.schema,),
            sims=self.sims + (hit.similarity,),
            query=query,
        )


class MurreRetriever:
    """MURRE đầy đủ, chạy trong RAM, không đọc/ghi file trung gian.

        retriever = MurreRetriever(encoder=encoder, rewriter=rewriter)
        tables = retriever.run(question, corpus, schema_embeddings)
        sql = retriever.generate_sql(question, tables)

    beam_size / max_hop để None thì lấy từ cfg.pipeline.
    """

    def __init__(
        self,
        encoder: SGPTEncoder,
        rewriter: Optional[QueryRewriter] = None,
        llm: Optional[LLMGenerator] = None,
        *,
        beam_size: Optional[int] = None,
        max_hop: Optional[int] = None,
    ) -> None:
        self.encoder: SGPTEncoder = encoder
        self.llm: Optional[LLMGenerator] = llm

        # B và H của paper (§4.1: B = 5, H = 3).
        self.beam_size: int = beam_size if beam_size is not None else cfg.pipeline.beam_size
        self.max_hop: int = max_hop if max_hop is not None else cfg.pipeline.max_hop

        # Removal cần LLM; Splice (ablation w/o removal) thì không. Chế độ Splice để
        # rewriter=None LUÔN, kể cả khi chỗ gọi có truyền vào — nhờ vậy `_removal()`
        # chỉ cần xem rewriter có hay không, không phải giữ thêm một cờ song song.
        self.rewriter: Optional[QueryRewriter] = None
        if cfg.pipeline.ablation.removal:
            if rewriter is None and llm is None:
                raise ValueError(
                    "MurreRetriever cần `rewriter` hoặc `llm` khi ablation.removal=true. "
                    "Đặt removal=false để chạy chế độ Splice không cần LLM."
                )
            self.rewriter = rewriter if rewriter is not None else QueryRewriter(llm=llm)

    # =========================================================================
    # Pha 1 — Retrieval (§3.3, Equation 3.1)
    # =========================================================================
    def _retrieve(
        self,
        query: str,
        corpus: List[str],
        doc_embeddings: torch.Tensor,
        top_k: int,
    ) -> List[Hit]:
        """Top-k bảng của TOÀN BỘ corpus theo cosine similarity — Equation 3.1.

            doc_embeddings : ma trận ĐÃ chuẩn hoá L2, hàng i là vector của corpus[i].

        Cả câu truy vấn được encode thành MỘT vector Emb(q_h,b), kể cả khi Removal
        trả về nhiều bảng trên nhiều dòng — đúng Equation 3.1, không tách sub-query.
        """
        text: str = " ".join(query.split()) or query
        q: torch.Tensor = F.normalize(
            input=self.encoder.encode(texts=[text], is_query=True),
            p=2,
            dim=1
        )

        sims: torch.Tensor = (q @ doc_embeddings.T).squeeze(0)
        values, positions = torch.topk(input=sims, k=min(top_k, sims.size(0)))

        return [
            Hit(schema=corpus[p], similarity=float(v), index=p)
            for v, p in zip(values.tolist(), positions.tolist())
        ]

    # =========================================================================
    # Pha 2 — Removal (§3.4)
    # =========================================================================
    def _removal(self, question: str, path: RetrievalPath) -> Tuple[str, bool]:
        """Xoá thông tin các bảng trên `path` khỏi CÂU HỎI GỐC `question`.

        Trả về (câu truy vấn cho hop kế, LLM có báo "đủ bảng rồi" hay không).

        §3.4 nói rõ: Removal xoá thông tin của Path_h,b khỏi *the user question q*.
        Nên tham số truyền vào LUÔN là câu hỏi gốc, còn `path.schemas` là TẤT CẢ
        bảng từ hop 1 tới hop h trên nhánh đó — không phải chỉ bảng của hop cuối.
        """
        if self.rewriter is None:
            # Ablation w/o removal (§4.3): nối câu hỏi với bảng đã có, không gọi LLM.
            joined: str = " | ".join(path.schemas)
            return (f"{question} | {joined}" if joined else question), False

        out: str = self.rewriter.rewrite(
            question=question,
            retrieved_schemas=list(path.schemas),
        ).strip()
        return out, QueryRewriter.is_early_stop(rewrite_output=out)

    # =========================================================================
    # Tỉa beam (§3.3 → §3.5)
    # =========================================================================
    def _prune(self, paths: Sequence[RetrievalPath]) -> List[RetrievalPath]:
        """B×B đường đi → giữ B đường có Score_Path cao nhất.

        §3.3: "we then choose B results from these for the subsequent Removal
        phase, with the selection method detailed in §3.5" — tức dùng đúng
        Score_Path của §3.5, không có công thức tỉa riêng.
        """
        ranked: List[RetrievalPath] = sorted(
            paths, key=lambda p: scoring.path_score(similarities=p.sims), reverse=True,
        )
        return ranked[: self.beam_size]

    # =========================================================================
    # Pha 3 — Score (§3.5, Algorithm 1)
    # =========================================================================
    @staticmethod
    def _rank(all_paths: Sequence[RetrievalPath]) -> List[RetrievedTable]:
        """Algorithm 1: Score_Table(t) = max Score_Path trên mọi đường đi chứa t.

        `all_paths` phải là MỌI đường đi từng sinh ra — kể cả đường bị tỉa ở bước
        _prune và đường dừng sớm — vì §3.5 định nghĩa Path_ti là "all retrieval
        paths containing table ti", không giới hạn ở B nhánh sống sót.
        """
        scores: Dict[str, float] = scoring.score_tables(
            paths=((p.schemas, p.sims) for p in all_paths),
        )
        return [
            RetrievedTable(schema=schema, score=score)
            for schema, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
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

        # --- hop 1: câu hỏi GỐC, không tabulate (Appendix K) -----------------
        first_hits: List[Hit] = self._retrieve(
            query=question,
            corpus=corpus,
            doc_embeddings=docs,
            top_k=self.beam_size
        )
        if not first_hits:
            return []

        empty: RetrievalPath = RetrievalPath(schemas=(), sims=(), query=question)
        frontier: List[RetrievalPath] = [
            empty.extend(hit=hit, query=question) for hit in first_hits
        ]
        all_paths: List[RetrievalPath] = list(frontier)
        if verbose:
            self._log_hop(hop=1, paths=frontier)

        # --- hop 2 .. H -------------------------------------------------------
        # max_hop ĐẾM CẢ hop 1: max_hop=3 → hop 1, 2, 3, tức 2 vòng Removal.
        # Table 5 của paper cũng vậy — "max hop 1" nghĩa là single-hop.
        for hop in range(2, self.max_hop + 1):
            grown: List[RetrievalPath] = []
            num_stopped: int = 0

            for path in frontier:
                next_query, stopped = self._removal(question=question, path=path)
                if stopped:
                    # §3.4 + Figure 2: early stop là quyết định của RIÊNG nhánh này.
                    # Nhánh đóng băng tại đây nhưng vẫn nằm trong all_paths để chấm
                    # điểm; các nhánh khác đi tiếp bình thường.
                    num_stopped += 1
                    continue

                # Retrieval của hop này: câu Removal quét toàn corpus, lấy top-B.
                hits: List[Hit] = self._retrieve(
                    query=next_query,
                    corpus=corpus,
                    doc_embeddings=docs,
                    top_k=self.beam_size,
                )

                # Bỏ bảng đã nằm trên chính đường đi này: đi lại thì không thêm được
                # thông tin gì — Removal lẽ ra đã xoá nó khỏi câu hỏi.
                new_hits: List[Hit] = [
                    hit for hit in hits if hit.schema not in path.schemas
                ]

                # Mỗi bảng còn lại nối vào cuối đường đi thành một đường đi mới.
                for hit in new_hits:
                    grown.append(path.extend(hit=hit, query=next_query))

            if verbose and num_stopped:
                logger.info(f"[MURRE] Hop {hop}: {num_stopped} nhánh early stop")

            if not grown:
                if verbose:
                    logger.info(f"[MURRE] Hop {hop}: không còn nhánh nào để nở → dừng")
                break

            # Mọi đường đi sinh ra đều được chấm điểm; chỉ B đường tốt nhất đi tiếp.
            all_paths.extend(grown)
            frontier = self._prune(paths=grown)
            if verbose:
                self._log_hop(hop=hop, paths=frontier, grown=len(grown))

        results: List[RetrievedTable] = self._rank(all_paths=all_paths)
        if verbose:
            self._log_results(results=results, num_paths=len(all_paths))
        return results

    # -- Log ------------------------------------------------------------------
    @staticmethod
    def _log_hop(
        hop: int, paths: Sequence[RetrievalPath], grown: Optional[int] = None,
    ) -> None:
        origin: str = f"{grown} ứng viên → " if grown is not None else ""
        logger.info(f"[MURRE] Hop {hop}: {origin}giữ {len(paths)} nhánh")
        for path in paths[:3]:
            score: float = scoring.path_score(similarities=path.sims)
            logger.info(f"   {score:.4f}  {' -> '.join(path.schemas)}")

    @staticmethod
    def _log_results(results: Sequence[RetrievedTable], num_paths: int) -> None:
        logger.info(f"[MURRE] Xong: {num_paths} đường đi → {len(results)} bảng | top-3:")
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
        f"removal={ab.removal}, tabulation={ab.tabulation}"
    )

    run_one_question(
        method=Method.MURRE,
        question=QUESTION,
        top_n=TOP_N,
        verbose=VERBOSE,
        llm_profile=LLM_PROFILE,
    )
