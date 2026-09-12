# =============================================================================
# models/records.py — Định dạng RECORD của file kết quả trong outputs/
#
#   pipeline/runner.py           ──ghi──>  ResultRecord  (turn{H}/dev.json)
#   pipeline/sql.py              ──đọc──>  ResultRecord  (để sinh SQL)
#   utils/metrics.py, /evaluate  ──đọc──>  ResultRecord
#
# QUY TẮC KHI SỬA: thứ tự field trong class CHÍNH LÀ thứ tự khóa ghi ra JSON.
# Đổi thứ tự thì file mới vẫn đọc được (JSON không quan tâm thứ tự), nhưng diff
# giữa hai lần chạy sẽ nhiễu. Thêm field mới thì thêm vào CUỐI.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from models.retrieval import RetrievedRow


@dataclass
class ResultRecord:
    """Một câu hỏi trong turn{H}/dev.json — kết quả CUỐI CÙNG của pipeline.

    Đây là định dạng mà utils/metrics.py, pipeline/sql.py và /evaluate cùng đọc, nên
    3 trường dưới đây là phần hợp đồng thật sự giữa các module.
    """

    utterance: str
    gold: List[str]
    retrieved: List[RetrievedRow]

    # SQL — thêm ở CUỐI theo quy tắc đầu file, và có mặc định nên record cũ
    # (checkpoint của lượt chạy trước) vẫn đọc được.
    #   gold_sql   câu SQL đúng, chép từ `query` của dev.json
    #   sql        câu SQL do LLM sinh — rỗng cho tới khi chạy pipeline/sql.py
    #   sql_top_k  sinh từ top mấy bảng; None = chưa sinh
    gold_sql: str = ""
    sql: str = ""
    sql_top_k: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultRecord":
        return cls(
            utterance=d.get("utterance", ""),
            gold=list(d.get("gold", [])),
            retrieved=RetrievedRow.from_list(items=d.get("retrieved", [])),
            gold_sql=d.get("gold_sql", ""),
            sql=d.get("sql", ""),
            sql_top_k=d.get("sql_top_k"),
        )

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> List["ResultRecord"]:
        return [cls.from_dict(d=d) for d in items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utterance": self.utterance,
            "gold": self.gold,
            "retrieved": [r.to_dict() for r in self.retrieved],
            "gold_sql": self.gold_sql,
            "sql": self.sql,
            "sql_top_k": self.sql_top_k,
        }

    @property
    def schemas(self) -> List[str]:
        """Danh sách schema đã xếp hạng — thứ mà metrics và sinh SQL thực sự cần."""
        return [r.schema for r in self.retrieved]
