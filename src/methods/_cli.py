"""methods/_cli.py — Phần dùng chung cho entry point của 3 file method.

Mỗi method (murre.py / single_hop.py / crush.py) tự có khối `__main__` riêng để
chạy và test độc lập; mỗi khối tự khai báo biến riêng ở đầu; file này chỉ giữ phần
in kết quả dùng chung để không phải viết lại 3 lần.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config import cfg
from dataset.loader import gold_for


def print_results(
    method: str,
    question: str,
    results: List[Dict[str, Any]],
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
        if gold and r["schema"] in gold:
            mark = "✓"
            hits += 1
        print(f"  {mark} {i:>2}. {r['score']:.4f}  {r['schema']}")

    if gold:
        print(f"\n  → recall@{top_n} = {hits}/{len(gold)} = {hits / len(gold):.2%}")
    print("=" * 78)
