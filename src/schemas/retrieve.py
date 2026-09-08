"""schemas/retrieve.py — Request/response model cho endpoint /retrieve."""
from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.common import QuestionRequest


class RetrieveRequest(QuestionRequest):
    """Body của POST /retrieve — câu hỏi → danh sách bảng đã xếp hạng.

    Cùng bộ tham số với POST /sql (xem schemas/common.py): question, dataset, top_k.
    """


class TableResult(BaseModel):
    """Một bảng trong kết quả retrieve, đã xếp hạng."""

    rank: int = Field(..., description="Thứ hạng (bắt đầu từ 1)")
    table_schema: str = Field(..., description="Chuỗi schema bảng: db.table(col1, col2, ...)")
    score: float = Field(..., description="Score_Table (§3.5) — càng lớn càng liên quan")
