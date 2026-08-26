"""schemas/retrieve.py — Request/response model cho endpoint /retrieve."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Body của POST /retrieve."""

    question: str = Field(..., description="Câu hỏi tự nhiên cần tìm bảng liên quan")
    dataset: str = Field(default="spider", description="Dataset: 'spider' hoặc 'bird'")
    top_n: int = Field(default=5, ge=1, le=20, description="Số lượng bảng trả về (1-20)")


class TableResult(BaseModel):
    """Một bảng trong kết quả retrieve, đã xếp hạng."""

    rank: int = Field(..., description="Thứ hạng (bắt đầu từ 1)")
    table_schema: str = Field(..., description="Chuỗi schema bảng: db.table(col1, col2, ...)")
    score: float = Field(..., description="Điểm số tổng hợp (log-space) từ pipeline")
