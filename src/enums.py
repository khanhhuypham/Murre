"""enums.py — Các tập giá trị hợp lệ của project, dùng thay cho chuỗi trần.

Cách dùng:
    from enums import Dataset, Method

    Dataset("spider")        → Dataset.SPIDER
    Dataset("SPIDER")        → Dataset.SPIDER      (không phân biệt hoa/thường)
    Dataset("mysql")         → ValueError

Scale encoder KHÔNG có enum ở đây, và cũng không có danh sách hợp lệ ở đâu cả:
`general.scale` chỉ là NHÃN thư mục outputs/, chuỗi tự do. Model thật sự nạp lên
là `encoder.model_name` trong config.yaml.

Vì kế thừa `str`, mọi member vẫn dùng được như chuỗi (`f"dataset/{ds}/..."`,
so sánh với `"spider"`, FastAPI/Pydantic serialize ra đúng giá trị JSON), nên
không cần `.value` ở chỗ nào.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional


class BaseStrEnum(str, Enum):
    """Enum chuỗi khoan dung khi parse: bỏ khoảng trắng, không phân biệt hoa/thường.

    `_missing_` được Enum gọi khi tra cứu theo giá trị thất bại — nhờ vậy
    `Method(" MURRE ")` vẫn ra `Method.MURRE` thay vì nổ ValueError.
    """

    @classmethod
    def _missing_(cls, value: object) -> Optional["BaseStrEnum"]:
        if not isinstance(value, str):
            return None
        normalized: str = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return None

    @classmethod
    def values(cls) -> List[str]:
        """Danh sách giá trị hợp lệ — tiện để đưa vào thông báo lỗi."""
        return [member.value for member in cls]

    def __str__(self) -> str:
        # Để f-string in ra "spider" chứ không phải "Dataset.SPIDER".
        return self.value


class Dataset(BaseStrEnum):
    """Dataset text-to-SQL được hỗ trợ (khớp `general.dataset` trong src/config.py)."""
    SPIDER = "spider"
    BIRD = "bird"


class Method(BaseStrEnum):
    """Phương pháp retrieval (khớp `pipeline.method` trong src/config.py)."""

    MURRE = "murre"
    SINGLE_HOP = "single_hop"
    CRUSH = "crush"

    @property
    def needs_llm(self) -> bool:
        """MURRE cần LLM cho pha Removal, CRUSH cần để hallucinate schema."""
        return self in (Method.MURRE, Method.CRUSH)


class JobStatus(BaseStrEnum):
    """Trạng thái một lần chạy pipeline qua API (POST /pipeline/run)."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_final(self) -> bool:
        """True khi job đã kết thúc — không còn thay đổi nữa."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED)
