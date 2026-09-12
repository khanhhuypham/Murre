"""api/routers/retrieve.py — /retrieve: lấy top-N bảng liên quan cho một câu hỏi."""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Request

from api.dependencies import load_dataset_once, require_dataset
from models.errors import AppError
from models.retrieval import RetrievedTable
from pipeline.retriever import MurreRetriever
from schemas.retrieve import RetrieveRequest, TableResult
from utils import logger

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=List[TableResult], summary="Retrieve bảng cho câu hỏi")
async def retrieve_tables(payload: RetrieveRequest, request: Request) -> List[TableResult]:
    if not payload.question.strip():
        raise AppError.bad_request(message="Câu hỏi không được để trống.")

    require_dataset(ds_name=payload.dataset)
    retriever: MurreRetriever = await load_dataset_once(
        state=request.app.state,
        ds_name=payload.dataset,
    )

    # Tiêu đề in TRƯỚC: mấy chục dòng [MURRE] ngay bên dưới là của request này.
    logger.info(f"[API] /retrieve {payload.dataset}: {payload.question}")

    # MURRE gọi LLM ở mỗi hop → blocking. Đẩy sang thread để không chẹn event loop.
    # to_thread() chuyển tiếp nguyên keyword argument xuống hàm được gọi.
    #
    # verbose=True vì lẽ như ở /sql: một câu mỗi request, log từng hop là đường duy
    # nhất soi được beam và nhánh early stop đứng sau thứ hạng trả về.
    results: List[RetrievedTable] = await asyncio.to_thread(
        retriever.run,
        question=payload.question,
        verbose=True,
    )
    return [
        TableResult(rank=i + 1, table_schema=r.schema, score=r.score)
        for i, r in enumerate(results[: payload.top_k])
    ]
