# =============================================================================
# steps/rewrite.py — BƯỚC 3: Viết lại câu hỏi bằng LLM (Completing Tables)
#
# Tương đương với rewrite/sample.py + rewrite/rewrite.py của tác giả.
# Gộp 2 bước trên thành 1 bước duy nhất để dễ quản lý.
#
# CÁCH CHẠY (Option 2 — Batch mode):
#   python steps/rewrite.py --hop 0
#   python steps/rewrite.py --hop 1
# =============================================================================

import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from config import cfg
from core.llm import LLMGenerator
from core.rewriter import QueryRewriter
from utils import logger

def _find_json_files(directory: str) -> List[str]:
    """Tìm tất cả file JSON trong một thư mục, sắp xếp theo tên."""
    files: List[str] = []
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def _sample_beams(retrieved_dir: str, top_k: int) -> List[List[Dict[str, Any]]]:
    """
    Lấy mẫu top-K beam path từ các file retrieved.
    Tương đương với rewrite/sample.py của tác giả.

    Tham số:
        retrieved_dir : thư mục chứa các file dev.*.json đã retrieve
        top_k         : số beam cần lấy

    Trả về:
        List of [beam_0_data, beam_1_data, ...] → mỗi item là danh sách samples cho 1 beam
    """
    logger.info(f"[SampleBeams] Đang tải file JSON từ: {retrieved_dir}")

    # Tải tất cả file retrieved trong thư mục
    all_files_data: List[List[Dict[str, Any]]] = []
    for fpath in _find_json_files(directory=retrieved_dir):
        logger.debug(f"[SampleBeams] Đang đọc file: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            all_files_data.append(json.load(f))

    if not all_files_data:
        raise FileNotFoundError(f"Không tìm thấy file JSON trong: {retrieved_dir}")

    logger.info(f"[SampleBeams] Đã tải {len(all_files_data)} file, "
                f"{len(all_files_data[0])} sample/file. Đang lấy top-{top_k} beam ...")

    # Kết quả: top_k slots, mỗi slot chứa danh sách sample
    results: List[List[Dict[str, Any]]] = [[] for _ in range(top_k)]

    num_samples: int = len(all_files_data[0])
    for i in range(num_samples):
        logger.debug(f"[SampleBeams] --- Sample {i}/{num_samples} ---")

        # Tập hợp tất cả (utterance, selected_database, sims, utterance_org)
        # của sample thứ i qua tất cả file
        candidates: List[Tuple[str, List[str], List[float], Any]] = []

        for file_idx, file_data in enumerate(all_files_data):
            sample = file_data[i]

            if "selected_database" not in sample:
                sample["selected_database"] = []

            logger.debug(
                f"[SampleBeams] File {file_idx}: utterance={sample.get('utterance', '')!r}, "
                f"đã có {len(sample['selected_database'])} bảng trong đường đi, "
                f"xét {min(3 * top_k, len(sample['retrieved']))} bảng ứng viên mới"
            )

            # Xét top 3*top_k bảng đã retrieve để tạo beam path mới
            for x in sample["retrieved"][:3 * top_k]:
                # Bỏ qua bảng đã có trong đường đi hiện tại
                if x["schema"] in [t[0] for t in sample["selected_database"]]:
                    logger.debug(f"[SampleBeams] Bỏ qua (đã có trong path): {x['schema']}")
                    continue

                utterance: str = sample.get("utterance") or sample.get("question", "")
                assert utterance, f"Câu hỏi rỗng tại sample {i}"

                logger.debug(f"[SampleBeams] Ứng viên mới: {x['schema']} (similarity={x['similarity']:.4f})")

                candidates.append((
                    utterance,
                    [t[0] for t in sample["selected_database"]] + [x["schema"]],
                    [t[1] for t in sample["selected_database"]] + [x["similarity"]],
                    sample.get("utterance_org"),
                ))

        # Sắp xếp theo log-sum path score và lấy top_k
        candidates = sorted(
            candidates,
            key=lambda x: sum(math.log(max(s, 1e-9)) for s in x[2]),
            reverse=True,
        )[:top_k]

        logger.debug(f"[SampleBeams] → Chọn được {len(candidates)}/{top_k} candidate tốt nhất cho sample {i}")

        if len(candidates) < top_k:
            logger.debug(
                f"[SampleBeams] Sample {i}: chỉ có {len(candidates)}/{top_k} candidate hợp lệ "
                f"(một số beam slot phía sau có thể thiếu sample này)."
            )

        # Gán vào từng beam slot
        for j, (utterance, schemas, sims, utterance_org) in enumerate(candidates):
            logger.debug(f"[SampleBeams] Beam slot {j}: path={schemas}")

            results[j].append({
                "utterance":         utterance,
                "utterance_org":     utterance_org,
                "selected_database": list(zip(schemas, sims)),
                "rel_schema":        all_files_data[0][i].get("gold", []),
            })

        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            logger.info(f"[SampleBeams] Đã xử lý {i + 1}/{num_samples} sample ...")

    logger.info(f"[SampleBeams] Hoàn tất. Số sample mỗi beam: {[len(r) for r in results]}")

    return results

def _splice(question: str, schemas: List[str]) -> str:
    """Baseline Supplement (w/o removal): nối câu hỏi gốc với các schema đã tìm."""
    return question + " | " + " | ".join(schemas) if schemas else question

def run_rewrite(hop: int) -> None:
    """
    Chạy bước Rewrite cho một hop.

    Gồm 2 giai đoạn:
      1. Sample beams từ kết quả retrieve của hop trước
      2. Gọi LLM để viết lại từng beam (Completing Tables)
    """
    beam_size: int = cfg.pipeline.beam_size
    use_removal: bool = cfg.pipeline.ablation.removal

    logger.info(
        f"{'=' * 60}\n"
        f"  BƯỚC 3: REWRITE — Hop {hop} ({'Removal' if use_removal else 'Splice (w/o removal)'})\n"
        f"{'=' * 60}"
    )

    # ── Giai đoạn 1: Sample beams ─────────────────────────────────────────
    # Đường dẫn thư mục chứa kết quả retrieve của hop trước
    # Beam sampling luôn đọc KẾT QUẢ RETRIEVE của chính hop này (thư mục turn{hop}/),
    # vì _sample_beams() cần trường "retrieved" — thứ chỉ steps/retrieve.py sinh ra.
    # Thư mục rewrite/outputs/turn{hop}/ chỉ chứa câu hỏi đã viết lại (không có
    # "retrieved"), nên đọc từ đó sẽ lỗi KeyError: 'retrieved'.
    retrieved_dir: str
    if hop == 0:
        retrieved_dir = os.path.dirname(cfg.outputs.turn0())
    else:
        retrieved_dir = os.path.dirname(cfg.outputs.turn_n(hop=hop, beam=0))

    logger.info(f"[Rewrite] Lấy mẫu beams từ: {retrieved_dir}")
    beam_samples: List[List[Dict[str, Any]]] = _sample_beams(retrieved_dir=retrieved_dir, top_k=beam_size)

    # ── Giai đoạn 2: Gọi LLM cho từng beam ───────────────────────────────
    rewriter: QueryRewriter | None = None
    if use_removal:
        llm: LLMGenerator = LLMGenerator()
        rewriter = QueryRewriter(llm=llm)

    out_dir: str = cfg.outputs.rewrite_output(hop=hop)
    os.makedirs(out_dir, exist_ok=True)

    for beam_idx, beam_data in enumerate(beam_samples):
        logger.info(f"\n[Rewrite] Đang xử lý beam {beam_idx}/{beam_size - 1} ...")

        for sample_idx, d in enumerate(beam_data):
            # Lấy danh sách schema trong đường đi beam hiện tại
            sel_dbs: List[Tuple[str, float]] = d.get("selected_database", [])
            schemas: List[str] = [s[0] for s in sel_dbs] if sel_dbs else []

            logger.info(
                f"[Rewrite] Beam {beam_idx} | Sample {sample_idx}: "
                f"utterance gốc={d['utterance']!r}, path_schemas={schemas}"
            )

            # Gọi LLM
            rewritten: str
            if use_removal and rewriter is not None:
                result: str = rewriter.rewrite(question=d["utterance"], retrieved_schemas=schemas)
                rewritten = result.strip()
                logger.info(f"[Rewrite] Beam {beam_idx} | Sample {sample_idx}: LLM trả về={rewritten!r}")
            else:
                rewritten = _splice(question=d["utterance"], schemas=schemas)
                logger.info(f"[Rewrite] Beam {beam_idx} | Sample {sample_idx}: đã splice={rewritten!r}")

            # Cập nhật lịch sử utterance (theo tác giả)
            utterance_before: str = deepcopy(x=d["utterance"])
            if d.get("utterance_org") and isinstance(d["utterance_org"], list):
                d["utterance_org"].append(utterance_before)
            elif d.get("utterance_org") and isinstance(d["utterance_org"], str):
                d["utterance_org"] = [d["utterance_org"], utterance_before]
            else:
                d["utterance_org"] = [utterance_before]

            d["utterance"] = rewritten

            if QueryRewriter.is_early_stop(rewrite_output=rewritten):
                logger.info(f"[Rewrite] Beam {beam_idx} | Sample {sample_idx}: → phát hiện Early Stop")

            if sample_idx % 50 == 0:
                logger.info(f"  [{sample_idx}/{len(beam_data)}] xong")

        # Lưu kết quả beam này
        out_file: str = os.path.join(out_dir, f"dev.{beam_idx}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(beam_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Rewrite] Đã lưu beam {beam_idx} → {out_file}")

    logger.info(f"\n[Rewrite] Hoàn thành hop {hop}")
    logger.info("=" * 60)


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP (Option 2 — Batch mode)
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bước Rewrite của MURRE")
    parser.add_argument("--hop", type=int, required=True, help="Hop hiện tại (bắt đầu từ 0)")
    args = parser.parse_args()

    run_rewrite(hop=args.hop)
