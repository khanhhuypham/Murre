"""schemas/pipeline.py — Request/response model cho nhóm endpoint /pipeline.

Khác /evaluate (chỉ ĐỌC metric của lần chạy đã có), /pipeline/run THỰC SỰ CHẠY
pipeline nên tốn nhiều thời gian → trả về một job để poll trạng thái.
"""
from __future__ import annotations

from typing import Optional

from enums import Dataset, JobStatus
from pydantic import BaseModel, ConfigDict, Field

from schemas.evaluate import EvalResult


class PipelineRunRequest(BaseModel):
    """Body của POST /pipeline/run.

    KHÔNG có `model`: encoder do server quyết định, lấy từ encoder.model_name trong
    config.yaml. Lý do: model phải khớp với cache embeddings đã có sẵn trên máy chủ,
    để client tự chọn thì dễ sinh ra lần chạy phải tải model mới (multilingual-e5-
    large ~2.2GB) rồi mã hoá lại cả corpus.
    """

    # extra="forbid": client cũ còn gửi "method"/"model" sẽ nhận 422 kèm tên field
    # sai, thay vì bị bỏ qua âm thầm rồi tưởng server đã chạy đúng thứ mình chọn.
    model_config = ConfigDict(extra="forbid")

    dataset: Dataset = Field(default=Dataset.SPIDER, description="spider | bird | vitext2sql")
    k: int = Field(default=5, ge=1, description="k dùng để báo recall@k sau khi chạy xong")
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Chỉ chạy N câu đầu của dev.json (bỏ trống = chạy hết). "
                    "MURRE gọi LLM mỗi hop mỗi beam nên chạy đủ dev.json rất lâu — "
                    "để thử nhanh hãy đặt limit=20.",
    )
    sql: bool = Field(
        default=True,
        description="Chạy xong retrieval thì sinh luôn SQL cho mọi câu bằng LLM, "
                    "điền vào `sql` + `sql_top_k` của file result (cạnh `gold_sql`) "
                    "và ghi thêm sql.{k}.txt. Dùng `k` làm số bảng đưa vào prompt. "
                    "Tốn thêm MỘT lần gọi LLM mỗi câu — đặt false nếu chỉ cần đo "
                    "recall.",
    )
    verbose: bool = Field(
        default=False,
        description="Ghi log chi tiết từng hop của mọi câu (beam giữ lại, nhánh "
                    "early stop, bảng đã xếp hạng). Log nằm ở outputs/logs/murre.log chứ "
                    "không có trong response — bật khi cần soi, vì lượt chạy dài "
                    "sinh ra log rất lớn.",
    )


class PipelineJob(BaseModel):
    """Trạng thái một lần chạy pipeline.

    KHÔNG có `model`, giống PipelineRunRequest và EvalResult: encoder do server
    quyết định. Nhãn model thực tế đọc được từ `result.result_file`.
    """

    job_id: str = Field(..., description="Dùng để poll GET /pipeline/jobs/{job_id}")
    status: JobStatus = Field(..., description="queued | running | succeeded | failed")
    dataset: Dataset = Field(..., description="Dataset của lần chạy")
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
    sql_file: Optional[str] = Field(
        default=None,
        description="Đường dẫn sql.{k}.txt — chỉ có khi chạy với sql=true",
    )
