"""schemas/pipeline.py — Request/response model cho nhóm endpoint /pipeline.

Khác /evaluate (chỉ ĐỌC metric của lần chạy đã có), /pipeline/run THỰC SỰ CHẠY
pipeline nên tốn nhiều thời gian → trả về một job để poll trạng thái.
"""
from __future__ import annotations

from typing import Optional

from enums import Dataset, JobStatus, Method
from pydantic import BaseModel, ConfigDict, Field

from schemas.evaluate import EvalResult


class PipelineRunRequest(BaseModel):
    """Body của POST /pipeline/run — 3 tham số chính + limit để chạy thử nhanh.

    KHÔNG có `model`: encoder do server quyết định, lấy từ encoder.model_name trong
    config.yaml. Lý do: model phải khớp với cache embeddings đã có sẵn trên máy chủ,
    để client tự chọn thì dễ sinh ra lần chạy phải tải model mới (SGPT-5.8B ~23GB).
    Nhãn model thực tế vẫn được báo lại trong PipelineJob.model.
    """

    # extra="forbid": client cũ còn gửi "model" sẽ nhận 422 kèm tên field sai, thay
    # vì bị bỏ qua âm thầm rồi tưởng server đã chạy đúng model mình chọn.
    model_config = ConfigDict(extra="forbid")

    dataset: Dataset = Field(default=Dataset.SPIDER, description="spider | bird")
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
    """Trạng thái một lần chạy pipeline.

    KHÔNG có `model`, giống PipelineRunRequest và EvalResult: encoder do server
    quyết định. Nhãn model thực tế đọc được từ `result.result_file`.
    """

    job_id: str = Field(..., description="Dùng để poll GET /pipeline/jobs/{job_id}")
    status: JobStatus = Field(..., description="queued | running | succeeded | failed")
    dataset: Dataset = Field(..., description="Dataset của lần chạy")
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
