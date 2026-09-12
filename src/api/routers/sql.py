"""api/routers/sql.py — /sql: câu hỏi tự nhiên → bảng liên quan → câu lệnh SQL.

Đây là bước cuối của paper (retrieve rồi mới sinh SQL). Đường batch làm việc này
bằng pipeline/sql.py::run_infer sau khi chạy xong cả dev.json; endpoint này chạy
thẳng cho một câu, dùng chung hàm dựng prompt nên hai đường không thể lệch nhau.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Request

from api.dependencies import load_dataset_once, require_dataset
from models.errors import AppError
from models.retrieval import RetrievedTable
from pipeline.retriever import MurreRetriever
from pipeline.sql import build_sql
from schemas.sql import SqlRequest, SqlResponse
from utils import logger

router = APIRouter(tags=["sql"])


@router.post("/sql", response_model=SqlResponse, summary="Sinh SQL từ câu hỏi tự nhiên")
async def generate_sql(payload: SqlRequest, request: Request) -> SqlResponse:
    if not payload.question.strip():
        raise AppError.bad_request(message="Câu hỏi không được để trống.")

    require_dataset(ds_name=payload.dataset)
    state = request.app.state
    retriever: MurreRetriever = await load_dataset_once(
        state=state,
        ds_name=payload.dataset,
    )

    # Tiêu đề in TRƯỚC: mấy chục dòng [MURRE] ngay bên dưới là của request này.
    logger.info(f"[API] /sql {payload.dataset}: {payload.question}")

    # Cả retrieve lẫn gọi LLM đều blocking → đẩy sang thread để không chẹn event loop.
    # to_thread() chuyển tiếp nguyên keyword argument xuống hàm được gọi.
    #
    # verbose=True: một request chỉ có MỘT câu hỏi nên log từng hop chỉ thêm chục
    # dòng, mà đó là thứ duy nhất cho biết vì sao SQL lại dùng mấy bảng đó —
    # response chỉ có bảng cuối cùng, không có đường đi. Đường batch thì ngược lại,
    # verbose do người gọi bật (run_pipeline).
    tables: List[RetrievedTable] = await asyncio.to_thread(
        retriever.run,
        question=payload.question,
        verbose=True,
    )
    if not tables:
        raise AppError.conflict(message="Không tìm được bảng nào cho câu hỏi này.")

    schemas: List[str] = [t.schema for t in tables[: payload.top_k]]
    sql: str = await asyncio.to_thread(
        build_sql,
        llm=state.llm,
        question=payload.question,
        schemas=schemas,
        dataset=payload.dataset,
    )

    logger.info(f"[API] /sql → {len(schemas)} bảng | SQL: {sql}")
    return SqlResponse(question=payload.question, sql=sql, tables=schemas)
