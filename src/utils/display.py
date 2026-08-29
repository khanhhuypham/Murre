"""utils/display.py — In kết quả retrieve ra terminal cho người đọc.

Dùng chung cho main.py và khối `__main__` của 3 file trong methods/, để không phải
viết lại phần in 4 lần.
"""
from __future__ import annotations

from typing import List

from config import cfg
from dataset.loader import gold_for
from models.retrieval import RetrievedTable


def print_results(
    method: str,
    question: str,
    results: List[RetrievedTable],
    top_n: int,
) -> None:
    """In kết quả retrieve, đánh dấu ✓ vào bảng trùng gold (nếu tra được gold)."""
    gold: List[str] = gold_for(question=question)

    print()
    print("=" * 78)
    print(f"  METHOD  : {method}")
    print(f"  DATASET : {cfg.general.dataset} | ENCODER SCALE: {cfg.general.scale}")
    print(f"  QUESTION: {question}")
    print("=" * 78)

    if gold:
        print(f"\nGold ({len(gold)} bảng):")
        for g in gold:
            print(f"  - {g}")

    print(f"\nTop-{top_n} bảng retrieve được:")
    hits: int = 0
    for i, r in enumerate(results[:top_n], start=1):
        mark: str = " "
        if gold and r.schema in gold:
            mark = "✓"
            hits += 1
        print(f"  {mark} {i:>2}. {r.score:.4f}  {r.schema}")

    if gold:
        print(f"\n  → recall@{top_n} = {hits}/{len(gold)} = {hits / len(gold):.2%}")
    print("=" * 78)
