"""steps/score.py — BƯỚC 4 (Option 2/Batch, method=murre): Score theo §3.5.

Score_Path = tích Norm(similarity) dọc một nhánh (tính trong log-space).
Score_Table = max(Score_Path) trong số các nhánh chứa cùng một bảng.

Nếu cfg.pipeline.ablation.early_stop = False, nhãn "None" bị bỏ qua — mọi câu hỏi
được coi như chạy hết max_hop (đúng tinh thần ablation "w/o early stop" của Table 4).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from config import cfg
from core.rewriter import QueryRewriter
from models.records import BeamStep, ResultRecord, TurnRecord
from models.retrieval import RetrievedRow
from steps._common import find_json_files
from utils import logger
from utils.metrics import compute_recall_at_k, compute_res
from utils.scoring import path_score

def run_score() -> None:
    top_k_list: List[int] = cfg.general.top_k
    top_k_max: int = max(top_k_list)
    max_hop: int = cfg.pipeline.max_hop
    use_early_stop: bool = cfg.pipeline.ablation.early_stop

    logger.info("=" * 60)
    logger.info("  BƯỚC 4: SCORE")
    logger.info("=" * 60)

    # [hop][file][sample] — mỗi hop có beam_size file, trừ hop 0 chỉ có 1 file.
    retrieved_data: List[List[List[TurnRecord]]] = []

    turn0_file: str = cfg.outputs.turn0()
    with open(turn0_file, "r", encoding="utf-8") as f:
        retrieved_data.append([TurnRecord.from_list(items=json.load(f))])

    for hop in range(1, max_hop + 1):
        turn_dir: str = os.path.dirname(cfg.outputs.turn_n(hop=hop, beam=0))
        hop_files: List[str] = find_json_files(directory=turn_dir)
        if not hop_files:
            break
        hop_data: List[List[TurnRecord]] = []
        for fpath in hop_files:
            with open(fpath, "r", encoding="utf-8") as f:
                hop_data.append(TurnRecord.from_list(items=json.load(f)))
        retrieved_data.append(hop_data)

    num_turns: int = len(retrieved_data)
    num_samples: int = len(retrieved_data[0][0])

    schema_score: List[Dict[str, float]] = []
    for sample in retrieved_data[0][0]:
        score_map: Dict[str, float] = {}
        for x in sample.retrieved:
            score_map.setdefault(x.schema, 0.0)
        schema_score.append(score_map)

    # Xác định hop mỗi câu hỏi dừng (Early Stop, hoặc hết max_hop nếu ablation tắt)
    end_turn: List[int] = [num_turns - 1] * num_samples
    if use_early_stop:
        for turn_idx in range(1, num_turns):
            for sample_idx in range(num_samples):
                for file_data in retrieved_data[turn_idx]:
                    sample = file_data[sample_idx]
                    utterance: str = sample.utterance
                    if QueryRewriter.is_early_stop(rewrite_output=utterance) and end_turn[sample_idx] > turn_idx - 1:
                        end_turn[sample_idx] = turn_idx - 1
    else:
        logger.info("[Score] Ablation w/o early_stop: bỏ qua nhãn 'None', dùng hết max_hop.")

    stop_dist: Dict[int, int] = {}
    for et in end_turn:
        stop_dist[et] = stop_dist.get(et, 0) + 1
    logger.info(f"[Score] Phân bố hop dừng: {stop_dist}")

    for sample_idx in range(num_samples):
        i_turn: int = end_turn[sample_idx]

        for file_data in retrieved_data[i_turn]:
            d = file_data[sample_idx]

            sel_db: List[BeamStep] = d.selected_database

            for x in d.retrieved:
                # Score_Path của nhánh "đường đi đã có + bảng x" — cùng một hàm mà
                # methods/murre.py dùng, xem utils/scoring.py.
                score: float = path_score(
                    similarities=[s[1] for s in sel_db] + [x.similarity]
                )

                schema_score[sample_idx][x.schema] = max(
                    score, schema_score[sample_idx].get(x.schema, float("-inf"))
                )
                for s_item in sel_db:
                    schema_score[sample_idx][s_item[0]] = max(
                        score, schema_score[sample_idx].get(s_item[0], float("-inf"))
                    )

    final_output: List[ResultRecord] = []
    for sample_idx, scores in enumerate(schema_score):
        ranked: List[Tuple[str, float]] = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k_max]

        retrieved_list: List[RetrievedRow] = [
            RetrievedRow(rank=k, schema=schema, similarity=score)
            for k, (schema, score) in enumerate(ranked)
        ]

        # Giữ nguyên record của turn0 rồi chỉ thay retrieved/recall — nhờ `extra` của
        # ResultRecord mà input/utterance_org/selected_database không bị mất, và thứ
        # tự khóa trong file ra vẫn y như trước.
        base: TurnRecord = retrieved_data[0][0][sample_idx]
        recall: Dict[int, float] = compute_recall_at_k(
            top_k=top_k_list, pred_list=[r.schema for r in retrieved_list], gold_list=base.gold,
        )

        merged: Dict[str, Any] = base.to_dict()
        merged["recall"] = recall
        record: ResultRecord = ResultRecord.from_dict(d=merged)
        record.retrieved = retrieved_list
        final_output.append(record)

    result_file: str = cfg.outputs.result()
    score_file: str = cfg.outputs.score()
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in final_output], f, ensure_ascii=False, indent=2)

    metrics: Dict[str, Dict[int, float]] = compute_res(top_k=top_k_list, data=final_output)
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("[Score] Kết quả:")
    for k in top_k_list:
        r: float = metrics["recall"].get(k, 0.0)
        c: float = metrics["complete_recall"].get(k, 0.0)
        logger.info(f"  k={k:>3} | recall={r:.4f} | complete_recall={c:.4f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    run_score()
