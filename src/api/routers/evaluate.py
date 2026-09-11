"""api/routers/evaluate.py — ĐỌC metric THẬT của những lần chạy đã có trên máy này.

Phần tính toán nằm ở api/evaluator.py::evaluate_run — router chỉ lo HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

from fastapi import APIRouter, Query

from api.dependencies import available_datasets
from api.evaluator import evaluate_run
from config import cfg
from enums import Dataset
from models.records import ResultRecord
from schemas.evaluate import AvailableRun, EvalResult

router = APIRouter(tags=["evaluate"])


@router.get(
    "/evaluate",
    response_model=List[EvalResult],
    summary="Tính recall@k / complete_recall@k THẬT từ kết quả đã chạy",
)
async def evaluate(
    dataset: Dataset = Query(default=Dataset.SPIDER, description="spider | bird | vitext2sql"),
    model: Optional[str] = Query(
        default=None,
        description="Nhãn encoder của lần chạy — chính là thư mục trong "
                    "outputs/{dataset}/{model}/. Bỏ trống = encoder mà dataset đó "
                    "đang khai trong config. Nhận cả tên HuggingFace đầy đủ; tra "
                    "hụt thì 404 có kèm danh sách nhãn đang có trên đĩa.",
    ),
    k: List[int] = Query(default=[5], description="Một hoặc nhiều k, ví dụ ?k=3&k=5&k=10"),
) -> List[EvalResult]:
    """Trả về metric THẬT tính từ file kết quả của lần chạy tương ứng trên máy này.

    Không có kết quả cho tổ hợp đó → `404` kèm lệnh cần chạy trước.
    """
    slug: str = model or cfg.encoder_for(dataset=dataset).slug
    return [
        await asyncio.to_thread(evaluate_run, dataset=dataset, model=slug, k=kk)
        for kk in k
    ]


@router.get(
    "/evaluate/available",
    response_model=List[AvailableRun],
    summary="Các lần chạy đã có kết quả trên máy này",
)
async def evaluate_available() -> List[AvailableRun]:
    """Quét outputs/ để biết dataset nào đã chạy xong.

    Chỉ quét những dataset service này phục vụ (api.datasets) — kết quả của dataset
    khác vẫn nằm trên đĩa nhưng không thuộc phạm vi của service này.

    Chỉ quét nhãn encoder hiện tại của server (suy ra từ encoder.model_name), không
    duyệt mọi model đang có trên đĩa — cùng quy ước với /pipeline/run: encoder do
    server quyết định. Nhãn thực tế vẫn đọc được từ `result_file`.
    """
    found: List[AvailableRun] = []
    for ds in available_datasets():
        # Không truyền model → for_run giữ nguyên encoder.slug của cfg.
        f: str = cfg.outputs.for_run(dataset=ds).result()
        if not os.path.exists(f):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = ResultRecord.from_list(items=json.load(fh))
        found.append(AvailableRun(
            dataset=ds,
            num_questions=len(data),
            retrieved_depth=min((len(d.retrieved) for d in data), default=0),
            result_file=f,
        ))
    return found
