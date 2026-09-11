"""cli.py — Giao diện dòng lệnh của MURRE. Chạy từ src/: `python -m cli <lệnh>`

    python -m cli ask                       một câu hỏi (câu đầu dev.json), in ra terminal
    python -m cli ask -q "..." -v           câu tự viết, in chi tiết từng hop
    python -m cli run --limit 20            chạy 20 câu đầu dev.json, ghi outputs/
    python -m cli run --dataset bird        chạy cả dev.json của BIRD
    python -m cli run --sql                 chạy xong thì sinh luôn SQL từ top-K bảng
    python -m cli embed                     chỉ encode corpus rồi lưu cache .pt
    python -m cli config                    in cấu hình đang hiệu lực (JSON)

Mọi giá trị không có cờ tương ứng đều lấy từ config.yaml. Cờ chỉ ghi đè cho một
lần chạy, không sửa file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

from config import cfg
from core.corpus import build_corpus, load_embeddings
from core.encoder import SentenceEncoder
from enums import Dataset
from models.errors import AppError
from models.metrics import MetricScores
from pipeline.runner import override_dataset, run_one_question, run_pipeline
from utils import logger


def cmd_ask(args: argparse.Namespace) -> None:
    """Một câu hỏi → in top-N bảng ra terminal, không ghi file."""
    with override_dataset(dataset=args.dataset):
        run_one_question(
            question=args.question,
            top_k=args.top_k,
            verbose=args.verbose,
            llm_profile=args.llm_profile,
        )


def cmd_run(args: argparse.Namespace) -> None:
    """Cả dev.json → ghi result + score ra outputs/, tuỳ chọn sinh SQL."""
    with override_dataset(dataset=args.dataset):
        if args.force_embed:
            cache_file: str = cfg.outputs.embeddings_cache()
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"[CLI] --force-embed → đã xoá {cache_file}, sẽ encode lại.")

        result: Dict[str, Any] = run_pipeline(limit=args.limit)
        _print_scores(metrics=result["metrics"], num_questions=result["num_questions"])
        print(f"  result → {result['result_file']}")
        print(f"  score  → {result['score_file']}")

        if args.sql:
            # Import trong thân hàm: bước sinh SQL không cần thiết cho retrieval.
            from pipeline.sql import run_infer
            print(f"  sql    → {run_infer(top_k=args.top_k)}")


def cmd_embed(args: argparse.Namespace) -> None:
    """Encode corpus rồi lưu cache .pt — bước tuỳ chọn, `run` cũng tự làm.

    Hữu ích khi muốn mã hóa trước cho xong (BIRD mất vài phút trên CPU) rồi mới
    bắt đầu lượt chạy dài, hoặc khi dựng image/volume sẵn cache cho service.
    """
    with override_dataset(dataset=args.dataset):
        corpus: List[str] = build_corpus()
        embs: torch.Tensor = load_embeddings(encoder=SentenceEncoder.get(), corpus=corpus)
        print(
            f"  {len(corpus)} schema | embeddings {tuple(embs.shape)} "
            f"→ {cfg.outputs.embeddings_cache()}"
        )


def cmd_config(_: argparse.Namespace) -> None:
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def _print_scores(metrics: MetricScores, num_questions: int) -> None:
    scores: Dict[str, Dict[int, float]] = metrics.to_dict()
    print(f"\n  Kết quả trên {num_questions} câu:")
    for name, by_k in scores.items():
        cells: str = "  ".join(f"@{k}={100 * v:.1f}" for k, v in by_k.items())
        print(f"    {name:<16} {cells}")


def _dataset(value: str) -> Dataset:
    try:
        return Dataset(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"dataset={value!r} không hợp lệ. Chỉ nhận: {Dataset.values()}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Cờ dùng chung cho mọi lệnh có chạm dataset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset", type=_dataset, default=None,
        help=f"{' | '.join(Dataset.values())} (mặc định: general.dataset trong config.yaml)",
    )

    ask = sub.add_parser("ask", parents=[common], help="một câu hỏi, in ra terminal")
    ask.add_argument("-q", "--question", default=None, help="mặc định: câu đầu dev.json")
    ask.add_argument(
        "-k", "--top-k", type=int, default=cfg.pipeline.top_k_output,
        help="số bảng in ra",
    )
    ask.add_argument("-v", "--verbose", action="store_true", help="in chi tiết từng hop")
    ask.add_argument("--llm-profile", default=None, help="tên profile trong config.yaml")
    ask.set_defaults(func=cmd_ask)

    run = sub.add_parser("run", parents=[common], help="cả dev.json, ghi kết quả ra outputs/")
    run.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu")
    run.add_argument("--sql", action="store_true", help="sinh SQL sau khi chạy xong")
    run.add_argument(
        "--top-k", type=int, default=cfg.pipeline.top_k_output,
        help="số bảng đưa vào prompt sinh SQL (chỉ dùng với --sql)",
    )
    run.add_argument(
        "--force-embed", action="store_true", help="xoá cache .pt để encode lại corpus",
    )
    run.set_defaults(func=cmd_run)

    embed = sub.add_parser("embed", parents=[common], help="chỉ encode corpus rồi lưu cache")
    embed.set_defaults(func=cmd_embed)

    conf = sub.add_parser("config", help="in cấu hình đang hiệu lực")
    conf.set_defaults(func=cmd_config)

    return parser


def main() -> int:
    args: argparse.Namespace = build_parser().parse_args()
    try:
        args.func(args)
    except AppError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # `run` ghi checkpoint sau từng câu, nên Ctrl-C không mất phần đã chạy.
        print("\nĐã dừng. Chạy lại để tiếp tục từ checkpoint.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
