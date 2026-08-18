"""steps/score.py — BƯỚC 4 (Option 2/Batch, method=murre): Score theo §3.5.

Score_Path = tích Norm(similarity) dọc một nhánh (tính trong log-space).
Score_Table = max(Score_Path) trong số các nhánh chứa cùng một bảng.

Nếu cfg.pipeline.ablation.early_stop = False, nhãn "None" bị bỏ qua — mọi câu hỏi
được coi như chạy hết max_hop (đúng tinh thần ablation "w/o early stop" của Table 4).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

from config import cfg, get_path
from utils import logger
from utils.metric import compute_recall_at_k, compute_res

_EARLY_STOP_INDICATORS: List[str] = [
    "There is no",
    "None of the given tables",
    "No additional tables",
    "No completion needed",
    "None",
]


def _find_json_files(directory: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def norm(s: float) -> float:
    """Norm(s) = (s + 1) / 2 — Appendix C của paper."""
    return (s + 1) / 2


def _judge_early_stop(utterance: str) -> bool:
    return any(indicator in utterance for indicator in _EARLY_STOP_INDICATORS)


def run_score() -> None:
    top_k_list: List[int] = cfg.general.top_k
    top_k_max: int = max(top_k_list)
    max_hop: int = cfg.pipeline.max_hop
    use_early_stop: bool = cfg.pipeline.ablation.early_stop

    logger.info("=" * 60)
    logger.info("  BƯỚC 4: SCORE")
    logger.info("=" * 60)

    retrieved_data: List[List[List[Dict[str, Any]]]] = []

    turn0_file: str = get_path(key="turn0")
    with open(turn0_file, "r", encoding="utf-8") as f:
        retrieved_data.append([json.load(f)])

    for hop in range(1, max_hop + 1):
        turn_dir: str = os.path.dirname(get_path(key="turn_n", hop=hop, beam=0))
        hop_files: List[str] = _find_json_files(directory=turn_dir)
        if not hop_files:
            break
        hop_data: List[List[Dict[str, Any]]] = []
        for fpath in hop_files:
            with open(fpath, "r", encoding="utf-8") as f:
                hop_data.append(json.load(f))
        retrieved_data.append(hop_data)

    num_turns: int = len(retrieved_data)
    num_samples: int = len(retrieved_data[0][0])

    schema_score: List[Dict[str, float]] = []
    for sample in retrieved_data[0][0]:
        score_map: Dict[str, float] = {}
        for x in sample["retrieved"]:
            score_map.setdefault(x["schema"], 0.0)
        schema_score.append(score_map)

    # Xác định hop mỗi câu hỏi dừng (Early Stop, hoặc hết max_hop nếu ablation tắt)
    end_turn: List[int] = [num_turns - 1] * num_samples
    if use_early_stop:
        for turn_idx in range(1, num_turns):
            for sample_idx in range(num_samples):
                for file_data in retrieved_data[turn_idx]:
                    sample = file_data[sample_idx]
                    utterance: str = sample.get("utterance", "")
                    if _judge_early_stop(utterance=utterance) and end_turn[sample_idx] > turn_idx - 1:
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

            for x in d["retrieved"]:
                sel_db = d.get("selected_database", [])

                path_score: float
                if sel_db:
                    path_score = math.log(norm(s=x["similarity"])) + sum(
                        math.log(norm(s=s[1])) for s in sel_db
                    )
                else:
                    path_score = math.log(norm(s=x["similarity"]))

                schema_score[sample_idx][x["schema"]] = max(
                    path_score, schema_score[sample_idx].get(x["schema"], float("-inf"))
                )
                for s_item in sel_db:
                    schema_score[sample_idx][s_item[0]] = max(
                        path_score, schema_score[sample_idx].get(s_item[0], float("-inf"))
                    )

    final_output: List[Dict[str, Any]] = []
    for sample_idx, scores in enumerate(schema_score):
        ranked: List[Tuple[str, float]] = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k_max]

        retrieved_list: List[Dict[str, Any]] = [
            {"rank": k, "schema": schema, "similarity": score}
            for k, (schema, score) in enumerate(ranked)
        ]

        base = retrieved_data[0][0][sample_idx]
        gold: List[str] = base.get("gold", [])
        recall: Dict[int, float] = compute_recall_at_k(
            top_k=top_k_list, pred_list=[r["schema"] for r in retrieved_list], gold_list=gold,
        )

        base["retrieved"] = retrieved_list
        base["recall"] = recall
        final_output.append(base)

    result_file: str = get_path(key="result")
    score_file: str = get_path(key="score")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

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
