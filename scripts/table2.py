"""scripts/table2.py — Chạy từng ô của Bảng 2 và in ra để so với paper.

    # 1 ô (một dataset × một method), chạy hết dev.json
    python scripts/table2.py run --dataset spider --method single_hop

    # thử 20 câu trước cho chắc, và ép max_hop=2 để nửa số lần gọi LLM
    python scripts/table2.py run --dataset spider --method murre --limit 20 --max-hop 2

    # đổi model local (tên profile trong config.yaml)
    python scripts/table2.py run --dataset spider --method murre --llm llama3.2

    # in bảng tổng hợp từ mọi score.json đã có, kèm cột paper và độ lệch
    python scripts/table2.py report

Chạy lại một lệnh `run` đã xong thì gần như tức thì: kết quả nằm trong
checkpoint.jsonl, runner chỉ chạy những câu còn thiếu. Bấm Ctrl-C giữa chừng
không mất gì — chạy lại là đi tiếp.

Không cần sửa main.py. Mọi tham số truyền qua cờ dòng lệnh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import cfg                                    # noqa: E402
from enums import Dataset, Method                         # noqa: E402
from methods.runner import run_pipeline                   # noqa: E402

# Bảng 2 của paper (COLING 2025, tr. 5793). Khoá: (dataset, encoder_slug_prefix, method).
# Thứ tự giá trị: k=3, k=5, k=10, k=20, r@3, r@5, r@10, r@20.
PAPER: Dict[str, List[float]] = {
    "spider|125m|single_hop": [54.3, 66.0, 75.4, 82.2, 63.0, 73.1, 80.7, 86.3],
    "spider|125m|crush":      [60.2, 71.3, 80.7, 86.8, 68.9, 76.3, 83.4, 88.9],
    "spider|125m|murre":      [65.0, 74.2, 81.0, 85.3, 70.2, 77.5, 82.3, 86.9],
    "spider|5.8b|single_hop": [76.3, 86.8, 94.1, 97.6, 84.0, 91.5, 96.2, 98.7],
    "spider|5.8b|crush":      [68.2, 80.1, 88.4, 92.2, 75.5, 85.1, 91.2, 94.5],
    "spider|5.8b|murre":      [86.0, 93.5, 96.7, 97.3, 89.3, 94.3, 96.8, 97.5],
    "bird|125m|single_hop":   [39.0, 50.3, 62.1, 70.9, 54.0, 63.2, 73.3, 80.9],
    "bird|125m|crush":        [42.1, 56.1, 70.2, 77.7, 60.2, 70.0, 79.5, 86.1],
    "bird|125m|murre":        [51.4, 62.7, 72.9, 78.3, 64.8, 72.7, 79.6, 84.2],
    "bird|5.8b|single_hop":   [55.3, 67.3, 79.4, 86.4, 72.9, 80.8, 88.6, 92.8],
    "bird|5.8b|crush":        [52.2, 63.5, 78.4, 88.1, 70.0, 77.9, 87.5, 93.0],
    "bird|5.8b|murre":        [69.1, 80.1, 88.7, 92.7, 81.0, 87.6, 92.6, 95.4],
}
KS: List[int] = [3, 5, 10, 20]


def _encoder_size() -> str:
    """Nhãn cỡ encoder suy từ tên model, để tra vào PAPER."""
    slug: str = cfg.encoder.slug.lower()
    for size in ("125m", "1.3b", "2.7b", "5.8b"):
        if size in slug:
            return size
    return slug


def _row(scores: Dict[str, Dict[str, float]]) -> List[Optional[float]]:
    """score.json → 8 giá trị theo đúng thứ tự cột của Bảng 2 (đơn vị %)."""
    out: List[Optional[float]] = []
    for field in ("complete_recall", "recall"):
        for k in KS:
            v = scores.get(field, {}).get(str(k), scores.get(field, {}).get(k))
            out.append(round(100 * v, 1) if v is not None else None)
    return out


def _fmt(row: List[Optional[float]]) -> str:
    return " ".join(f"{v:>5.1f}" if v is not None else "    -" for v in row)


HEADER: str = f"{'dataset/encoder/method':<34} " + " ".join(
    f"{c:>5}" for c in ("k=3", "k=5", "k=10", "k=20", "r@3", "r@5", "r@10", "r@20")
)


def cmd_run(args: argparse.Namespace) -> None:
    dataset: Dataset = Dataset(args.dataset)
    method: Method = Method(args.method)

    # Ghi đè cfg TRƯỚC khi dựng pipeline — retriever đọc cfg đúng một lần lúc khởi tạo.
    if args.llm:
        if args.llm not in cfg.llm.profiles:
            raise SystemExit(
                f"Không có profile LLM {args.llm!r}. Có: {sorted(cfg.llm.profiles)}"
            )
        cfg.llm.active_profile = args.llm
    if args.beam_size is not None:
        cfg.pipeline.beam_size = args.beam_size
    if args.max_hop is not None:
        cfg.pipeline.max_hop = args.max_hop

    calls: int = cfg.pipeline.beam_size * (cfg.pipeline.max_hop - 1)
    print(
        f"\n  {dataset} / {cfg.encoder.slug} / {method}\n"
        f"  beam={cfg.pipeline.beam_size} max_hop={cfg.pipeline.max_hop} "
        f"pool={cfg.pipeline.top_k_pool} llm={cfg.llm.active_profile}\n"
        f"  tối đa {calls} lần gọi LLM mỗi câu"
        f"{' (single_hop không gọi LLM)' if method is Method.SINGLE_HOP else ''}\n"
    )

    res: Dict[str, Any] = run_pipeline(
        method=method,
        dataset=dataset,
        limit=args.limit,
        on_progress=(lambda done, total: None),
    )

    row: List[Optional[float]] = _row(res["metrics"].to_dict())
    key: str = f"{dataset}|{_encoder_size()}|{method}"
    print("\n" + HEADER)
    print(f"{key:<34} {_fmt(row)}   ← ta chạy ({res['num_questions']} câu)")
    if key in PAPER and args.limit is None:
        print(f"{'':<34} {_fmt(PAPER[key])}   ← paper")
    print(f"\n  score.json → {res['score_file']}")


def cmd_report(args: argparse.Namespace) -> None:
    print("\n" + HEADER)
    found: int = 0
    for ds in ("spider", "bird"):
        for mt in ("single_hop", "crush", "murre"):
            for hop in sorted({cfg.pipeline.max_hop, 2, 3}):
                paths = cfg.outputs.for_run(dataset=ds, method=mt, max_hop=hop)
                f: str = paths.score()
                if not os.path.exists(f):
                    continue
                with open(f, "r", encoding="utf-8") as fh:
                    row = _row(json.load(fh))
                key = f"{ds}|{_encoder_size()}|{mt}"
                found += 1
                print(f"{key + f' (turn{hop})':<34} {_fmt(row)}   ← ta chạy")
                if key in PAPER:
                    delta = [
                        (a - b) if a is not None else None
                        for a, b in zip(row, PAPER[key])
                    ]
                    print(f"{'':<34} {_fmt(PAPER[key])}   ← paper")
                    print(f"{'  lệch':<34} {_fmt(delta)}\n")
    if not found:
        print("\n  Chưa có score.json nào. Chạy `table2.py run ...` trước.")
    print(
        "\n  Ô nào chưa có: đổi encoder.model_name trong config.yaml (cho dòng 5.8B)\n"
        "  hoặc chạy thêm --dataset/--method còn thiếu.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="chạy một ô của Bảng 2")
    r.add_argument("--dataset", default="spider", choices=Dataset.values())
    r.add_argument("--method", default="murre", choices=Method.values())
    r.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu")
    r.add_argument("--llm", default=None, help="tên profile trong config.yaml")
    r.add_argument("--beam-size", type=int, default=None)
    r.add_argument("--max-hop", type=int, default=None, help="2 = nửa số lần gọi LLM")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="in bảng tổng hợp từ score.json đã có")
    rep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
