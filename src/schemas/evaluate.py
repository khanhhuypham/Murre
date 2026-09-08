"""schemas/evaluate.py — Response model cho endpoint /evaluate."""
from __future__ import annotations

from enums import Dataset
from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Metric THẬT tính từ một lần chạy pipeline trên máy này (không phải số paper).

    KHÔNG có `model`: encoder do server quyết định (encoder.model_name trong
    config.yaml), client không chọn. Muốn biết model nào thì đọc `result_file` —
    đường dẫn có sẵn nhãn model: outputs/{dataset}/{model}/...
    """

    dataset: Dataset = Field(..., description="spider | bird | vitext2sql")
    k: int = Field(..., description="Số bảng top-đầu dùng để tính metric")
    recall: float = Field(..., description="recall@k (%) — tỉ lệ bảng gold tìm được trong top-k")
    complete_recall: float = Field(
        ..., description="complete recall@k (%) — tỉ lệ câu hỏi có TẤT CẢ bảng gold trong top-k"
    )
    num_questions: int = Field(
        ..., description="Số câu hỏi thực sự có trong file kết quả — phải là 658 (Spider dev) "
                         "mới so được với paper; ít hơn nghĩa là lần chạy đó dùng limit"
    )
    retrieved_depth: int = Field(..., description="Số bảng đã lưu cho mỗi câu (giới hạn trên của k)")
    result_file: str = Field(..., description="File kết quả đã dùng để tính")


class AvailableRun(BaseModel):
    """Một dataset đã có kết quả trên đĩa.

    KHÔNG có `model`: endpoint chỉ quét nhãn encoder hiện tại của server (suy ra từ
    encoder.model_name), giống /pipeline/run. Muốn biết model nào thì nhìn
    `result_file`, đường dẫn có sẵn trong đó: outputs/{dataset}/{model}/...
    """

    dataset: Dataset = Field(..., description="spider | bird | vitext2sql")
    num_questions: int = Field(..., description="Số câu hỏi trong file kết quả")
    retrieved_depth: int = Field(..., description="Số bảng đã lưu cho mỗi câu")
    result_file: str = Field(..., description="Đường dẫn file kết quả")
