"""schemas/evaluate.py — Response model cho endpoint /evaluate."""
from __future__ import annotations

from typing import Any, Dict, List

from enums import Dataset, Method
from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Metric THẬT tính từ một lần chạy pipeline trên máy này (không phải số paper).

    KHÔNG có `model`: scale encoder do server quyết định (general.scale trong
    src/config.py), client không chọn. Muốn biết scale nào thì đọc `result_file` —
    đường dẫn có sẵn: outputs/{dataset}/{scale}/{method}/...
    """

    dataset: Dataset = Field(..., description="spider | bird")
    method: Method = Field(..., description="murre | single_hop | crush")
    k: int = Field(..., description="Số bảng top-đầu dùng để tính metric")
    recall: float = Field(..., description="recall@k (%) — tỉ lệ bảng gold tìm được trong top-k")
    complete_recall: float = Field(
        ..., description="complete recall@k (%) — tỉ lệ câu hỏi có TẤT CẢ bảng gold trong top-k"
    )
    num_questions: int = Field(
        ..., description="Số câu hỏi thực sự có trong file kết quả — phải là 658 (Spider dev) "
                         "mới so được với paper; 20 nghĩa là lần chạy đó dùng --limit"
    )
    retrieved_depth: int = Field(..., description="Số bảng đã lưu cho mỗi câu (giới hạn trên của k)")
    result_file: str = Field(..., description="File kết quả đã dùng để tính")


class AvailableRun(BaseModel):
    """Một tổ hợp (dataset, method) đã có kết quả trên đĩa.

    KHÔNG có `model`: endpoint chỉ quét scale hiện tại của server (general.scale
    trong src/config.py), giống /pipeline/run — nên mỗi (dataset, method) chỉ ra
    đúng một dòng. Muốn biết scale nào thì nhìn `result_file`, đường dẫn có sẵn
    trong đó: outputs/{dataset}/{scale}/{method}/...
    """

    dataset: Dataset = Field(..., description="spider | bird")
    method: Method = Field(..., description="murre | single_hop | crush")
    num_questions: int = Field(..., description="Số câu hỏi trong file kết quả")
    retrieved_depth: int = Field(..., description="Số bảng đã lưu cho mỗi câu")
    result_file: str = Field(..., description="Đường dẫn file kết quả")
