"""api/routers/sql.py — /sql: câu hỏi tự nhiên → bảng liên quan → câu lệnh SQL.

Đây là bước cuối của paper (retrieve rồi mới sinh SQL). Đường batch làm việc này
bằng pipeline/sql.py::run_infer sau khi chạy xong cả dev.json; endpoint này chạy
thẳng cho một câu, dùng chung hàm dựng prompt nên hai đường không thể lệch nhau.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Request

from api.dependencies import LoadedDataset, load_dataset_once, require_dataset
from models.errors import AppError
from models.retrieval import RetrievedTable
from pipeline.sql import build_sql
from schemas.sql import SqlRequest, SqlResponse

router = APIRouter(tags=["sql"])


@router.post("/sql", response_model=SqlResponse, summary="Sinh SQL từ câu hỏi tự nhiên")
async def generate_sql(payload: SqlRequest, request: Request) -> SqlResponse:
    if not payload.question.strip():
        raise AppError.bad_request(message="Câu hỏi không được để trống.")

    require_dataset(payload.dataset)
    state = request.app.state
    ds: LoadedDataset = await load_dataset_once(state, payload.dataset)

    # Cả retrieve lẫn gọi LLM đều blocking → đẩy sang thread để không chẹn event loop.
    tables: List[RetrievedTable] = await asyncio.to_thread(
        ds.retriever.run, payload.question, ds.corpus, ds.embs,
    )
    if not tables:
        raise AppError.conflict(message="Không tìm được bảng nào cho câu hỏi này.")

    schemas: List[str] = [t.schema for t in tables[: payload.top_k]]
    sql: str = await asyncio.to_thread(
        build_sql, state.llm, payload.question, schemas, payload.dataset,
    )

    return SqlResponse(question=payload.question, sql=sql, tables=schemas)
