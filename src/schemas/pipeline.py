"""schemas/pipeline.py — Request/response model cho nhóm endpoint /pipeline.

Khác /evaluate (chỉ ĐỌC metric của lần chạy đã có), /pipeline/run THỰC SỰ CHẠY
pipeline nên tốn nhiều thời gian → trả về một job để poll trạng thái.
"""
from __future__ import annotations

from typing import Optional

from enums import Dataset, JobStatus, Method, ModelScale
from pydantic import BaseModel, Field

from schemas.evaluate import EvalResult


class PipelineRunRequest(BaseModel):
    """Body của POST /pipeline/run — 4 tham số chính + limit để chạy thử nhanh."""

    dataset: Dataset = Field(default=Dataset.SPIDER, description="spider | bird")
    model: ModelScale = Field(
        default=ModelScale.M_125M,
        description="Scale encoder SGPT: 125m | 1.3b | 2.7b | 5.8b (nhận cả 'SGPT-125M')",
    )
    method: Method = Field(
        default=Method.MURRE, description="murre | single_hop | crush"
    )
    k: int = Field(default=5, ge=1, description="k dùng để báo recall@k sau khi chạy xong")
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Chỉ chạy N câu đầu của dev.json (bỏ trống = chạy hết). "
                    "Murre gọi LLM mỗi hop mỗi beam nên chạy đủ 658 câu rất lâu — "
                    "để thử nhanh hãy đặt limit=20.",
    )


class PipelineJob(BaseModel):
    """Trạng thái một lần chạy pipeline."""

    job_id: str = Field(..., description="Dùng để poll GET /pipeline/jobs/{job_id}")
    status: JobStatus = Field(..., description="queued | running | succeeded | failed")
    dataset: Dataset = Field(..., description="Dataset của lần chạy")
    model: ModelScale = Field(..., description="Scale encoder của lần chạy")
    method: Method = Field(..., description="Method của lần chạy")
    k: int = Field(..., description="k dùng để báo metric")
    limit: Optional[int] = Field(default=None, description="Số câu giới hạn, None = chạy hết")
    processed: int = Field(default=0, description="Số câu đã xử lý")
    total: int = Field(default=0, description="Tổng số câu sẽ xử lý (0 khi chưa nạp dev.json)")
    started_at: Optional[str] = Field(default=None, description="Thời điểm bắt đầu (ISO-8601 UTC)")
    finished_at: Optional[str] = Field(default=None, description="Thời điểm kết thúc (ISO-8601 UTC)")
    result: Optional[EvalResult] = Field(
        default=None, description="Metric tại k — chỉ có khi status=succeeded"
    )
    error: Optional[str] = Field(
        default=None, description="Nguyên nhân lỗi — chỉ có khi status=failed"
    )
