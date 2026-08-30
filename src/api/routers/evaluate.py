"""api/routers/evaluate.py — ĐỌC metric THẬT của những lần chạy đã có trên máy này.

Phần tính toán nằm ở api/evaluator.py::evaluate_run — router chỉ lo HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List

from fastapi import APIRouter, Query

from api.evaluator import evaluate_run
from config import cfg
from enums import Dataset, Method
from models.records import ResultRecord
from schemas.evaluate import AvailableRun, EvalResult

router = APIRouter(tags=["evaluate"])


@router.get(
    "/evaluate",
    response_model=List[EvalResult],
    summary="Tính recall@k / complete_recall@k THẬT từ kết quả đã chạy",
)
async def evaluate(
    dataset: Dataset = Query(default=Dataset.SPIDER, description="spider | bird"),
    model: str = Query(
        default=cfg.general.scale,
        description="Nhãn scale của lần chạy — chính là thư mục trong "
                    "outputs/{dataset}/{scale}/{method}/ (mặc định = general.scale "
                    "của server). Nhận cả 'SGPT-125M'; tra hụt thì 404 có kèm "
                    "danh sách scale đang có trên đĩa.",
    ),
    method: Method = Query(default=Method.MURRE, description="murre | single_hop | crush"),
    k: List[int] = Query(default=[5], description="Một hoặc nhiều k, ví dụ ?k=3&k=5&k=10"),
) -> List[EvalResult]:
    """Trả về metric THẬT tính từ file kết quả của lần chạy tương ứng trên máy này.

    Không có kết quả cho tổ hợp đó → `404` kèm lệnh cần chạy trước.
    """
    return [
        await asyncio.to_thread(evaluate_run, dataset, model, method, kk)
        for kk in k
    ]


@router.get("/evaluate/available", response_model=List[AvailableRun], summary="Các lần chạy đã có kết quả trên máy này")
async def evaluate_available() -> List[AvailableRun]:
    """Quét outputs/ để biết tổ hợp (dataset, method) nào đã chạy xong.

    Chỉ quét scale hiện tại của server (general.scale), không duyệt cả 4 scale —
    cùng quy ước với /pipeline/run: scale do server quyết định. Scale thực tế vẫn
    đọc được từ `result_file`.
    """
    found: List[AvailableRun] = []
    for ds in Dataset:
        for mt in Method:
            # Không truyền scale → for_run giữ nguyên general.scale của cfg.
            f: str = cfg.outputs.for_run(dataset=ds, method=mt).result()
            if not os.path.exists(f):
                continue
            with open(f, "r", encoding="utf-8") as fh:
                data = ResultRecord.from_list(items=json.load(fh))
            found.append(AvailableRun(
                dataset=ds,
                method=mt,
                num_questions=len(data),
                retrieved_depth=min((len(d.retrieved) for d in data), default=0),
                result_file=f,
            ))
    return found
