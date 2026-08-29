"""schemas/health.py — Response model cho endpoint /health."""
from __future__ import annotations

from typing import List

from enums import Dataset, Method
from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Trạng thái service.

    Phân biệt rõ 2 mức: `datasets_available` là dataset có tables.json trên đĩa,
    `datasets_loaded` là dataset đã thực sự nạp embeddings vào RAM (service nạp
    lười, chỉ nạp ở lần gọi /retrieve đầu tiên của từng dataset).
    """

    status: str = Field(..., description="'ok' nếu service đang chạy")
    method: Method = Field(..., description="pipeline.method đang cấu hình trong src/config.py")
    datasets_available: List[Dataset] = Field(..., description="Có dataset/{ds}/tables.json trên đĩa")
    datasets_loaded: List[Dataset] = Field(..., description="Đã nạp embeddings vào RAM")
    beam_size: int = Field(..., description="pipeline.beam_size")
    max_hop: int = Field(..., description="pipeline.max_hop")
