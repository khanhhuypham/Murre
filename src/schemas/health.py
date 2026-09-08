"""schemas/health.py — Response model cho endpoint /health."""
from __future__ import annotations

from typing import Dict, List

from enums import Dataset
from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Trạng thái service.

    Phân biệt rõ 2 mức: `datasets_available` là dataset có tables.json trên đĩa,
    `datasets_loaded` là dataset đã thực sự nạp embeddings vào RAM.
    """

    status: str = Field(..., description="'ok' khi sẵn sàng, 'starting' khi còn warm-up")
    ready: bool = Field(..., description="False → service chưa nhận được /retrieve (HTTP 503)")
    datasets_available: List[Dataset] = Field(..., description="Có dataset/{ds}/tables.json trên đĩa")
    datasets_loaded: List[Dataset] = Field(..., description="Đã nạp embeddings vào RAM")
    encoders: Dict[str, str] = Field(
        ..., description="dataset → model_name của encoder nó dùng"
    )
    llm: str = Field(..., description="Model của llm.active_profile")
    beam_size: int = Field(..., description="pipeline.beam_size")
    max_hop: int = Field(..., description="pipeline.max_hop")
