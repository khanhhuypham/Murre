"""schemas/sql.py — Request/response cho POST /sql (retrieve + sinh SQL)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from schemas.common import QuestionRequest


class SqlRequest(QuestionRequest):
    """Body của POST /sql — câu hỏi → bảng liên quan → câu lệnh SQL.

    Cùng bộ tham số với POST /retrieve (xem schemas/common.py): question, dataset,
    top_k. Ở đây `top_k` là số bảng đưa vào prompt sinh SQL.
    """


class SqlResponse(BaseModel):
    """SQL sinh ra kèm những bảng đã dùng để sinh — để truy vết khi SQL sai."""

    question: str = Field(..., description="Câu hỏi đã hỏi")
    sql: str = Field(..., description="Câu lệnh SQL do LLM sinh")
    tables: List[str] = Field(..., description="Các schema đã đưa vào prompt, theo thứ hạng")
