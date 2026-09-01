"""api/routers/retrieve.py — /retrieve: lấy top-N bảng liên quan cho một câu hỏi."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Request

from api.dependencies import LoadedDataset, available_datasets, load_dataset_once
from schemas.retrieve import RetrieveRequest, TableResult

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=List[TableResult], summary="Retrieve bảng cho câu hỏi")
async def retrieve_tables(payload: RetrieveRequest, request: Request) -> List[TableResult]:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    if payload.dataset not in available_datasets():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{payload.dataset}' không có dataset/{payload.dataset}/tables.json. "
                   f"Có sẵn: {available_datasets()}",
        )

    try:
        ds: LoadedDataset = await load_dataset_once(request.app.state, payload.dataset)
        results = ds.retriever.run(
            question=payload.question,
            corpus=ds.corpus,
            schema_embeddings=ds.embs,
            verbose=True
        )
        return [
            TableResult(rank=i + 1, table_schema=r.schema, score=r.score)
            for i, r in enumerate(results[: payload.top_n])
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình retrieve: {str(e)}") from e
