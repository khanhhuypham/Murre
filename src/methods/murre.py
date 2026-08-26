# =============================================================================
# methods/murre.py — Phương pháp chính MURRE, chạy trong bộ nhớ (in-process)
#
# Cùng nhóm với methods/single_hop.py và methods/crush.py: cả 3 đều là một
# retriever với cùng interface run(question, corpus, schema_embeddings, verbose),
# và đều được build_retriever() (core/factory.py) trả về theo cfg.pipeline.method.
#
# File này dành cho MỤC ĐÍCH HỌC VÀ DEBUG:
#   - Toàn bộ pipeline chạy trong 1 lần gọi pipeline.run()
#   - Dễ đặt breakpoint, theo dõi từng hop
#   - Phù hợp khi test với 1-10 câu hỏi
#
# Để chạy production (658 câu hỏi), dùng các file trong steps/ thay thế.
# Xem HUONG_DAN.md → Phần "Option 1 vs Option 2" để biết thêm.
#
# Thuật toán MURRE (từ bài báo Section 3):
#   Hop 0: Retrieve top-B bảng từ toàn bộ corpus với câu hỏi gốc
#   Hop 1..H:
#     1. [Removal/Splice] LLM dự đoán bảng còn thiếu → câu truy vấn mới
#        (hoặc nối/splice câu hỏi gốc với schema đã tìm, nếu ablation.removal=false)
#     2. [Retrieval] Retrieve trong pool hop-0 với câu truy vấn mới
#     3. [Early Stop] Nếu LLM trả "None" → dừng beam đó
#        (bị bỏ qua nếu ablation.early_stop=false, hoặc không tồn tại ở nhánh Splice)
#   Scoring: Score_Path = Π P̂(t|q) cho mỗi hop; Score_Table = max(Score_Path)
#
# Hỗ trợ 3 cờ ablation trong config.yaml (pipeline.ablation, Table 4 của paper §4.3):
#   removal    False → "w/o removal": dùng _next_query_with_splice thay vì LLM Removal
#   tabulation False → xử lý bên trong QueryRewriter (chọn prompt tương ứng)
#   early_stop False → không bao giờ dừng sớm, luôn chạy hết max_hop
#
# CÁCH CHẠY (Option 1 — Offline/Debug, chỉ cần config.yaml → run_option.mode: offline):
#   python -m main --question "Which airlines have a flight to AHD?" --verbose
#   python -m main                          # dùng câu hỏi mẫu đầu tiên trong dev.json
#
# CHẠY/TEST ĐỘC LẬP (xem khối __main__ ở cuối file):
#   python -m methods.murre
#   python -m methods.murre --question "..." --verbose
#   python -m methods.murre --beam_size 2 --max_hop 1      # chạy nhanh khi debug
# =============================================================================

import math
from typing import Any, Dict, List, TypedDict, Tuple

import torch
import torch.nn.functional as F

from config import cfg
from core.encoder import SGPTEncoder
from core.rewriter import QueryRewriter
from utils import logger

class RetrievalHit(TypedDict):
    schema: str
    similarity: float
    global_idx: int


class BeamState(TypedDict, total=False):
    path_schemas: List[str]
    path_sims: List[float]
    current_query: str
    early_stopped: bool


def _normalize(x: float) -> float:
    """
    Chuẩn hóa cosine similarity từ [-1,1] về [0,1].
    Công thức từ Appendix C bài báo: Norm(s) = (s + 1) / 2
    """
    return (x + 1) / 2

# def _pos(x: float) -> float:
#     """
#     Hàm pos() dùng trong log-sum path scoring (Appendix E bài báo).
#     pos(x) = (x + 2) / 2 → đảm bảo giá trị dương để tính log
#     """
#     return (x + 2) / 2


def _path_score(similarities: List[float]) -> float:
    """
    Tính Score_Path = Π P̂(t|q) trên tất cả các hop của một beam path.
    Trong code dùng log để tránh underflow (tích xác suất nhỏ → về 0).
    Score_Path = Σ log(Norm(sim_k)) cho k = 1..H
    """
    return sum(math.log(max(_normalize(x=s), 1e-9)) for s in similarities)

class MURREPipeline:
    """
    Pipeline MURRE chạy in-process — dùng để debug và học thuật toán.

    Cách dùng:
        pipeline = MURREPipeline(encoder, rewriter)
        results = pipeline.run(question, corpus, schema_embeddings)
        # results là list of {"schema": str, "score": float}
    """

    def __init__(self, encoder: SGPTEncoder, rewriter: QueryRewriter) -> None:
        self.encoder: SGPTEncoder = encoder
        self.rewriter: QueryRewriter = rewriter

        # Đọc tham số từ config
        self.beam_size: int = cfg.pipeline.beam_size
        self.max_hop: int = cfg.pipeline.max_hop
        self.top_k_pool: int = cfg.pipeline.top_k_pool

        self.use_removal: bool = cfg.pipeline.ablation.removal
        self.use_early_stop: bool = cfg.pipeline.ablation.early_stop

    # --------------------------------------------------------------------------
    # Các phương thức nội bộ
    # --------------------------------------------------------------------------

    def _retrieve_top_k(
        self,
        query_text:        str,
        corpus:            List[str],
        schema_embeddings: torch.Tensor,
        pool_indices:      List[int] | None = None,
        top_k:             int = 100,
    ) -> List[RetrievalHit]:
        """
        Mã hóa câu truy vấn rồi tính cosine similarity với corpus.
        Trả về top_k kết quả dạng {"schema", "similarity", "global_idx"}.

        pool_indices: nếu không None → chỉ search trong tập con này (Hop 1+)
        """
        # Mã hóa câu hỏi thành vector
        q_emb: torch.Tensor  = self.encoder.encode(texts=[query_text], is_query=True)  # shape (1, D)
        q_norm_vec: torch.Tensor = F.normalize(input=q_emb, p=2, dim=1)

        # Xác định pool document cần so sánh
        pool_embs: torch.Tensor
        actual_pool: List[int]
        if pool_indices is not None:
            pool_embs = schema_embeddings[pool_indices]
            actual_pool = pool_indices
        else:
            pool_embs = schema_embeddings
            actual_pool = list(range(len(corpus)))

        # Tính cosine similarity: (1, D) × (pool_size, D)^T = (1, pool_size)
        d_norm_vec: torch.Tensor = F.normalize(input=pool_embs, p=2, dim=1)
        scores: torch.Tensor = (q_norm_vec @ d_norm_vec.T).squeeze(0)  # shape (pool_size,)

        # Lấy top_k kết quả
        k: int = min(top_k, scores.size(0))
        top_vals, top_local_idx = torch.topk(input=scores, k=k)

        results: List[RetrievalHit] = []
        for val, local_idx in zip(top_vals.tolist(), top_local_idx.tolist()):
            global_idx: int = actual_pool[local_idx]
            results.append(
                RetrievalHit(
                    schema=corpus[global_idx],
                    similarity=float(val),
                    global_idx=global_idx,
                )
            )
        return results

    def _next_query_with_removal(self, question: str, path_schemas: List[str]) -> Tuple[str, bool]:
        """LLM Removal: trả về (câu hỏi mới, đã_early_stop?)."""
        rewrite_out: str = self.rewriter.rewrite(question=question, retrieved_schemas=path_schemas)
        stopped: bool = self.use_early_stop and QueryRewriter.is_early_stop(rewrite_output=rewrite_out)
        return rewrite_out, stopped

    @staticmethod
    def _next_query_with_splice(question: str, path_schemas: List[str]) -> Tuple[str, bool]:
        """Baseline Supplement (w/o removal, §2.1/§4.3): nối câu hỏi gốc với schema đã tìm.

        Không có tín hiệu Early Stop tự nhiên trong cách làm này (không gọi LLM), nên
        beam luôn tiếp tục tới max_hop.
        """
        spliced: str = question + " | " + " | ".join(path_schemas)
        return spliced, False

    # --------------------------------------------------------------------------
    # API công khai
    # --------------------------------------------------------------------------

    def run(
            self,
            question: str,
            corpus: List[str],
            schema_embeddings: torch.Tensor,
            verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Chạy toàn bộ thuật toán MURRE multi-hop beam search.

        Tham số:
            question          : câu hỏi gốc của người dùng
            corpus            : danh sách phẳng các chuỗi schema
            schema_embeddings : tensor embedding của corpus, shape (N, D)
            verbose           : True → in chi tiết từng hop (dùng khi debug)

        Trả về:
            Danh sách {"schema", "score"} sắp xếp theo score giảm dần
        """

        # ── HOP 0: Retrieve toàn bộ corpus với câu hỏi gốc ────────────────────
        if verbose:
            logger.info(
                f"\n{'=' * 60}\n"
                f"[HOP 0] Retrieve với câu hỏi gốc\n"
                f"  Câu hỏi: {question}"
            )

        hop0_results: List[RetrievalHit] = self._retrieve_top_k(
            query_text=question,
            corpus=corpus,
            schema_embeddings=schema_embeddings,
            pool_indices=None,
            top_k=self.top_k_pool,
        )

        # Lưu lại pool indices của hop 0 để các hop sau search trong đây
        hop0_pool_indices: List[int] = [r["global_idx"] for r in hop0_results]

        if verbose:
            logger.info("  Top-3 kết quả hop 0:")
            for r in hop0_results[:3]:
                logger.info(f"    {r['similarity']:.4f} | {r['schema']}")

        # ── KHỞI TẠO BEAMS từ top-B kết quả hop 0 ────────────────────────────
        # Mỗi beam là một "đường đi" (path) qua các bảng, gồm:
        #   path_schemas  : danh sách schema đã đi qua trên nhánh này
        #   path_sims     : danh sách similarity (chưa Norm) tương ứng
        #   current_query : câu truy vấn hiện tại (sau Removal/Splice)
        #   early_stopped : True nếu đã dừng sớm (chỉ có thể xảy ra ở nhánh Removal)
        beams: List[BeamState] = [
            BeamState(
                path_schemas=[r["schema"]],
                path_sims=[r["similarity"]],
                current_query=question,
                early_stopped=False,
            )
            for r in hop0_results[: self.beam_size]
        ]

        # ── HOP 1..max_hop ────────────────────────────────────────────────────
        for hop in range(1, self.max_hop + 1):
            active: List[BeamState] = [b for b in beams if not b["early_stopped"]]
            stopped: List[BeamState] = [b for b in beams if b["early_stopped"]]

            if verbose:
                logger.info(f"\n{'=' * 60}\n[HOP {hop}] Active beams: {len(active)} | Stopped: {len(stopped)}")

            # Nếu tất cả beam đã dừng sớm → kết thúc
            if not active:
                break

            candidates: List[BeamState] = []

            for beam_idx, beam in enumerate(active):
                # ── BƯỚC REMOVAL/SPLICE: dự đoán bảng còn thiếu → câu truy vấn mới ──
                # Chọn đúng cơ chế theo cờ ablation.removal trong config.yaml.
                if self.use_removal:
                    next_query, is_stopped = self._next_query_with_removal(
                        question=question,  # Luôn dùng câu hỏi gốc (theo Appendix K)
                        path_schemas=beam["path_schemas"],
                    )
                else:
                    next_query, is_stopped = self._next_query_with_splice(
                        question=question,
                        path_schemas=beam["path_schemas"],
                    )

                if verbose:
                    logger.info(f"\n  [Beam {beam_idx}] Path: {beam['path_schemas']}")
                    logger.info(f"  [Beam {beam_idx}] Next query: {next_query!r}")

                # Kiểm tra Early Stop (is_stopped đã tự tính theo self.use_early_stop
                # bên trong _next_query_with_removal, và luôn False ở nhánh Splice)
                if is_stopped:
                    if verbose:
                        logger.info(f"  [Beam {beam_idx}] → EARLY STOP")
                    stopped.append({**beam, "early_stopped": True})
                    continue

                # ── BƯỚC RETRIEVAL: Retrieve trong pool hop-0 ─────────────────
                new_results: List[RetrievalHit] = self._retrieve_top_k(
                    query_text=next_query,
                    corpus=corpus,
                    schema_embeddings=schema_embeddings,
                    pool_indices=hop0_pool_indices,  # Chỉ search trong pool hop-0
                    top_k=self.top_k_pool,
                )

                # Mở rộng beam: thêm từng bảng mới vào đường đi
                for r in new_results:
                    # Bỏ qua bảng đã có trong đường đi này (tránh lặp)
                    if r["schema"] in beam["path_schemas"]:
                        continue

                    # Tính score của đường đi mới
                    new_sims: List[float] = beam["path_sims"] + [r["similarity"]]
                    new_score: float = _path_score(similarities=new_sims)

                    candidates.append(
                        BeamState(
                            path_schemas=beam["path_schemas"] + [r["schema"]],
                            path_sims=new_sims,
                            current_query=next_query,
                            early_stopped=False,
                            **{"_score": new_score},  # type: ignore[typeddict-item]
                        )
                    )

            if not candidates:
                beams = stopped
                break

            # ── BEAM PRUNING: Giữ top-B đường đi tốt nhất ────────────────────
            candidates.sort(key=lambda c: c["_score"], reverse=True)  # type: ignore[typeddict-item]
            selected: List[BeamState] = candidates[: self.beam_size]

            # Xóa key tạm "_score" trước khi lưu
            for b in selected:
                b.pop("_score", None)  # type: ignore[typeddict-item]

            # Gộp beam active mới với beam đã dừng sớm
            beams = selected + stopped

        # ── SCORING: Tính Score_Table = max(Score_Path) cho mỗi bảng ─────────
        # Mỗi bảng có thể xuất hiện trong nhiều đường đi khác nhau.
        # Score_Table = max trong số tất cả Score_Path có chứa bảng đó.
        table_score: Dict[str, float] = {}

        for beam in beams:
            path_s: float = _path_score(similarities=beam["path_sims"])
            for schema in beam["path_schemas"]:
                # Lấy giá trị lớn nhất (max-pooling qua các đường đi)
                table_score[schema] = max(
                    path_s,
                    table_score.get(schema, float("-inf"))
                )

        # Sắp xếp và trả về kết quả
        ranked: List[Tuple[str, float]] = sorted(table_score.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            logger.info(f"\n{'=' * 60}")
            logger.info("[KẾT QUẢ] Top-5 bảng sau scoring:")
            for i, (schema, score) in enumerate(ranked[:5]):
                logger.info(f"  {i + 1}. {schema}  (score={score:.4f})")

        return [{"schema": s, "score": sc} for s, sc in ranked]


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP — test riêng method này với 1 câu hỏi
# =============================================================================
if __name__ == "__main__":
    from core.corpus import prepare
    from core.llm import LLMGenerator
    from dataset.loader import load_dev, resolve_question
    from methods._cli import print_results

    # --- Chỉnh trực tiếp mấy biến này để test ------------------------------
    DATASET: str | None = None       # None → dùng general.dataset của config.yaml
    if DATASET:
        # Phải set TRƯỚC load_dev(), nếu không sẽ đọc dev.json của dataset cũ.
        cfg.general.dataset = DATASET

    # Lấy câu số 21 của dev.json (câu cần 3 bảng — đúng ca multi-hop mà MURRE nhắm tới).
    # Đổi số trong [] để test câu khác; đặt None để lấy câu đầu tiên; hoặc gõ thẳng
    # một chuỗi tự viết (khi đó không tra được gold nên sẽ không có ✓ và recall).
    QUESTION: str | None = load_dev()[21]["utterance"]
    TOP_N: int = 5                   # số bảng in ra
    LLM_PROFILE: str | None = None   # None → dùng llm.active_profile
    VERBOSE: bool = False            # True: in chi tiết từng hop (Removal, retrieve, early stop)
    BEAM_SIZE: int | None = None     # None → theo pipeline.beam_size; để 2 cho nhanh khi debug
    MAX_HOP: int | None = None       # None → theo pipeline.max_hop; để 1 cho nhanh khi debug
    # ----------------------------------------------------------------------

    # Ghi đè config TRƯỚC khi khởi tạo pipeline, vì MURREPipeline.__init__ đọc
    # beam_size/max_hop/ablation từ cfg một lần duy nhất.
    if BEAM_SIZE is not None:
        cfg.pipeline.beam_size = BEAM_SIZE
    if MAX_HOP is not None:
        cfg.pipeline.max_hop = MAX_HOP

    encoder, corpus, embs = prepare(dataset=DATASET)
    question: str = resolve_question(question=QUESTION)

    ab = cfg.pipeline.ablation
    print()
    print(
        f"  Cấu hình: beam_size={cfg.pipeline.beam_size}, max_hop={cfg.pipeline.max_hop}, "
        f"removal={ab.removal}, tabulation={ab.tabulation}, early_stop={ab.early_stop}"
    )

    pipeline = MURREPipeline(
        encoder=encoder,
        rewriter=QueryRewriter(llm=LLMGenerator(profile=LLM_PROFILE)),
    )
    results = pipeline.run(
        question=question, corpus=corpus, schema_embeddings=embs, verbose=VERBOSE,
    )
    print_results(method="MURRE", question=question, results=results, top_n=TOP_N)
