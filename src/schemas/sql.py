"""schemas/sql.py — Request/response cho POST /sql (retrieve + sinh SQL)."""
from __future__ import annotations

from typing import List

from enums import Dataset
from pydantic import BaseModel, Field


class SqlRequest(BaseModel):
    """Body của POST /sql."""

    question: str = Field(..., description="Câu hỏi tự nhiên cần chuyển thành SQL")
    dataset: Dataset = Field(default=Dataset.SPIDER, description="Dataset: 'spider' hoặc 'bird'")
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Số bảng top-đầu đưa vào prompt sinh SQL (paper dùng 5)",
    )


class SqlResponse(BaseModel):
    """SQL sinh ra kèm những bảng đã dùng để sinh — để truy vết khi SQL sai."""

    question: str = Field(..., description="Câu hỏi đã hỏi")
    sql: str = Field(..., description="Câu lệnh SQL do LLM sinh")
    tables: List[str] = Field(..., description="Các schema đã đưa vào prompt, theo thứ hạng")
