"""api/routers/retrieve.py — /retrieve: lấy top-N bảng liên quan cho một câu hỏi."""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Request

from api.dependencies import LoadedDataset, load_dataset_once, require_dataset
from models.errors import AppError
from models.retrieval import RetrievedTable
from schemas.retrieve import RetrieveRequest, TableResult

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=List[TableResult], summary="Retrieve bảng cho câu hỏi")
async def retrieve_tables(payload: RetrieveRequest, request: Request) -> List[TableResult]:
    if not payload.question.strip():
        raise AppError.bad_request(message="Câu hỏi không được để trống.")

    require_dataset(ds_name=payload.dataset)
    ds: LoadedDataset = await load_dataset_once(
        state=request.app.state,
        ds_name=payload.dataset,
    )

    # MURRE gọi LLM ở mỗi hop → blocking. Đẩy sang thread để không chẹn event loop.
    # to_thread() chuyển tiếp nguyên keyword argument xuống hàm được gọi, nên gọi
    # tên ra được: `ds.embs` đứng thứ ba thì không ai đoán được nó là
    # `schema_embeddings` chứ không phải một corpus thứ hai.
    results: List[RetrievedTable] = await asyncio.to_thread(
        ds.retriever.run,
        question=payload.question,
        corpus=ds.corpus,
        schema_embeddings=ds.embs,
    )
    return [
        TableResult(rank=i + 1, table_schema=r.schema, score=r.score)
        for i, r in enumerate(results[: payload.top_k])
    ]
