"""api/routers/sql.py — /sql: câu hỏi tự nhiên → bảng liên quan → câu lệnh SQL.

Đây là bước cuối của paper (retrieve rồi mới sinh SQL). Đường batch làm việc này
bằng steps/infer.py sau khi chạy xong cả dev.json; endpoint này chạy thẳng cho một
câu, dùng chung hàm dựng prompt nên hai đường không thể lệch nhau.

Bảng do retriever của dataset trả về (method theo pipeline.method), nên /sql dùng
được với cả murre / single_hop / crush.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Request

from api.dependencies import LoadedDataset, available_datasets, ensure_dataset
from core.llm import LLMGenerator
from methods.murre import build_sql
from models.retrieval import RetrievedTable
from schemas.sql import SqlRequest, SqlResponse

router = APIRouter(tags=["sql"])


@router.post("/sql", response_model=SqlResponse, summary="Sinh SQL từ câu hỏi tự nhiên")
async def generate_sql(payload: SqlRequest, request: Request) -> SqlResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    if payload.dataset not in available_datasets():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{payload.dataset}' chưa có dataset/{payload.dataset}/tables.json. "
                   f"Đang có: {available_datasets()}",
        )

    state = request.app.state

    # LLM có thể chưa được nạp nếu pipeline.method hiện tại không cần (single_hop).
    if state.llm is None:
        state.llm = await asyncio.to_thread(LLMGenerator)
    llm: LLMGenerator = state.llm

    ds: LoadedDataset = await ensure_dataset(state, payload.dataset)

    # Cả retrieve lẫn gọi LLM đều blocking → đẩy sang thread để không chẹn event loop.
    tables: List[RetrievedTable] = await asyncio.to_thread(
        ds.retriever.run, payload.question, ds.corpus, ds.embs,
    )
    if not tables:
        raise HTTPException(status_code=422, detail="Không tìm được bảng nào cho câu hỏi này.")

    schemas: List[str] = [t.schema for t in tables[: payload.top_k]]
    sql: str = await asyncio.to_thread(build_sql, llm, payload.question, schemas)

    return SqlResponse(question=payload.question, sql=sql, tables=schemas)
