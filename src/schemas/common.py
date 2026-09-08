"""schemas/common.py — Phần chung của các request "hỏi một câu".

/retrieve và /sql nhận đúng cùng một bộ tham số, nên chúng kế thừa MỘT base ở đây
thay vì mỗi bên khai lại. Trước đây khai riêng nên hai bên trôi mỗi nơi một kiểu —
`/retrieve` gọi là `top_n` còn `/sql` gọi là `top_k` — và gõ nhầm giữa hai endpoint
là chuyện xảy ra thật. Khai một chỗ thì không lệch lại được.
"""
from __future__ import annotations

from config import cfg
from enums import Dataset
from pydantic import BaseModel, ConfigDict, Field


class QuestionRequest(BaseModel):
    """Một câu hỏi + dataset để tra + số bảng lấy ra."""

    # extra="forbid": gõ sai tên field sẽ nhận 422 chỉ thẳng chỗ sai, thay vì bị bỏ
    # qua âm thầm rồi tưởng server đã nhận giá trị mình gửi.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="Câu hỏi tự nhiên")
    dataset: Dataset = Field(
        default=Dataset.SPIDER, description="spider | bird | vitext2sql"
    )
    top_k: int = Field(
        # Mặc định lấy từ config, nên đổi pipeline.top_k_output là API đổi theo.
        default_factory=lambda: cfg.pipeline.top_k_output,
        ge=1,
        le=20,
        description="Số bảng top-đầu lấy ra (1-20). Paper dùng 5.",
    )
