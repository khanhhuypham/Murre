"""api/evaluator.py — TÍNH metric THẬT từ kết quả pipeline đã chạy trên máy này.

Tách khỏi router để dùng được trực tiếp trong script Python (không cần chạy API)
và để jobs.py đọc lại metric bằng đúng đường code của /evaluate.
"""
from __future__ import annotations

import json
import os
from typing import List

from config import cfg, model_slug
from enums import Dataset
from models.errors import AppError
from models.metrics import MetricScores
from models.records import ResultRecord
from schemas.evaluate import EvalResult
from utils.metrics import compute_res

# Chuỗi giả để dò vị trí {model} trong template đường dẫn.
_PROBE: str = "__model__"


def models_on_disk(dataset: Dataset) -> List[str]:
    """Các nhãn model đã thực sự có thư mục trong outputs/ — dùng cho thông báo lỗi.

    Không có danh sách hợp lệ khai ở đâu cả: nhãn suy ra từ model_name của encoder
    dùng cho lần chạy, nên thứ đáng nói khi tra hụt là "đĩa đang có nhãn nào".
    """
    # Dựng đường dẫn với nhãn giả rồi cắt lấy phần đứng trước → thư mục cha chứa
    # các nhãn. Bám theo template nên đổi paths.result cũng không hỏng.
    probe: str = cfg.outputs.for_run(dataset=dataset, model=_PROBE).result()
    parent: str = probe.split(_PROBE)[0]
    if not os.path.isdir(parent):
        return []
    return sorted(d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)))


def evaluate_run(dataset: Dataset | str, model: str, k: int) -> EvalResult:
    """Đọc kết quả retrieval đã lưu của một lần chạy rồi TÍNH LẠI metric tại k.

    Đây là số THẬT của máy này, không phải số trong paper. Metric được tính tại
    thời điểm gọi (không đọc lại score.json), nên k nào cũng được — miễn là
    k <= số bảng đã lưu cho mỗi câu.
    """
    try:
        ds: Dataset = Dataset(dataset)
    except ValueError:
        raise AppError.bad_request(
            message=f"dataset={dataset!r} không hợp lệ. Chỉ nhận: {Dataset.values()}"
        ) from None

    # Chuẩn hoá qua đúng hàm sinh nhãn, nên gõ cả tên HuggingFace đầy đủ
    # ("Muennighoff/SGPT-125M-...") hay gõ sẵn nhãn đều ra cùng một thư mục.
    # Sai tên thì rơi vào nhánh 404 bên dưới, ở đó in ra các nhãn có thật trên đĩa.
    slug: str = model_slug(name=str(model))

    result_file: str = cfg.outputs.for_run(dataset=ds, model=slug).result()
    if not os.path.exists(result_file):
        have: List[str] = models_on_disk(dataset=ds)
        raise AppError.not_found(
            message=(
                f"Chưa có kết quả cho dataset={ds}, model={slug} "
                f"(không tìm thấy {result_file}). "
                f"Model đã có trên đĩa: {have or 'chưa có model nào'}. "
                f"Chạy trước: POST /pipeline/run với dataset={ds}, "
                f"hoặc `python -m cli run --dataset {ds}`."
            )
        )

    with open(result_file, "r", encoding="utf-8") as f:
        data: List[ResultRecord] = ResultRecord.from_list(items=json.load(f))

    if not data:
        raise AppError.conflict(message=f"{result_file} rỗng.")

    depth: int = min(len(d.retrieved) for d in data)
    if k > depth:
        # compute_recall_at_k() cắt danh sách theo độ dài thật, nên k lớn hơn depth
        # chỉ lặp lại đúng số của k=depth mà không báo gì. Chặn ở đây để khỏi hiểu
        # nhầm là recall đã bão hoà.
        #
        # MURRE chỉ xếp hạng bảng nằm trên đường đi (§3.5), tối đa B + (H-1)·B²,
        # nên muốn depth lớn hơn thì phải tăng beam_size / max_hop rồi chạy lại.
        raise AppError.bad_request(
            message=(
                f"k={k} lớn hơn số bảng đã lưu mỗi câu ({depth}). Tăng "
                f"pipeline.beam_size (và/hoặc pipeline.max_hop) rồi chạy lại, "
                f"hoặc dùng k <= {depth}."
            )
        )

    metrics: MetricScores = compute_res(top_k=[k], data=data)
    return EvalResult(
        dataset=ds,
        k=k,
        recall=round(metrics.recall_at(k) * 100, 2),
        complete_recall=round(metrics.complete_recall_at(k) * 100, 2),
        num_questions=len(data),
        retrieved_depth=depth,
        result_file=result_file,
    )
