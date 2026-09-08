# =============================================================================
# models/retrieval.py — Đối tượng miền cho KẾT QUẢ RETRIEVAL
#
# Hai kiểu ở đây trông giống nhau nhưng KHÁC vai trò, đừng nhầm:
#
#   RetrievedTable : kết quả TRONG BỘ NHỚ mà MurreRetriever.run() trả về → (schema, score)
#   RetrievedRow   : một dòng trong mảng "retrieved" của FILE JSON      → (rank, schema, similarity)
#
# Việc đổi tên khóa giữa hai thế giới ("score" → "similarity") nằm gọn trong
# RetrievedTable.to_row(), không chỗ gọi nào phải nhớ.
#
# Vì sao KHÔNG dùng pydantic như src/schemas/: schemas/ là DTO của API (dữ liệu từ
# ngoài vào, cần validate). Hai kiểu này chạy trong vòng lặp nóng — mỗi câu hỏi sinh
# ra hàng chục object — và không bao giờ nhận dữ liệu từ người dùng, nên NamedTuple
# vừa nhẹ vừa đủ an toàn.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple


class RetrievedRow(NamedTuple):
    """Một dòng trong mảng "retrieved" của file kết quả (turn{H}/dev.json).

    Thứ tự field ở đây CHÍNH LÀ thứ tự khóa khi ghi ra JSON — giữ nguyên
    rank/schema/similarity để file mới đọc được bằng code cũ và ngược lại.
    """

    # Thứ hạng trong danh sách, đếm từ 0
    rank: int
    # Chuỗi schema "db_id.table(col1, col2, ...)"
    schema: str
    # Score_Table của câu hỏi (§3.5). Tên khóa là `similarity` vì file format của
    # tác giả gốc như vậy — giữ nguyên để công cụ đọc file cũ vẫn dùng được.
    similarity: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RetrievedRow:
        return cls(rank=int(d["rank"]), schema=d["schema"], similarity=float(d["similarity"]))

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> List[RetrievedRow]:
        """Đọc cả mảng "retrieved" của một record."""
        return [cls.from_dict(d=d) for d in items]

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "schema": self.schema, "similarity": self.similarity}


class RetrievedTable(NamedTuple):
    """Một bảng do MurreRetriever.run() trả về — trong bộ nhớ, không ghi thẳng ra file."""

    # Chuỗi schema "db_id.table(col1, col2, ...)"
    schema: str
    # Score_Table (§3.5, Algorithm 1). Càng lớn càng tốt.
    score: float

    def to_row(self, rank: int) -> RetrievedRow:
        """Đổi sang dòng của file JSON. `score` được ghi dưới tên khóa `similarity`
        vì đó là tên trong file format gốc — xem RetrievedRow.similarity."""
        return RetrievedRow(rank=rank, schema=self.schema, similarity=self.score)

    @staticmethod
    def to_rows(tables: Iterable[RetrievedTable]) -> List[RetrievedRow]:
        """Đánh số rank 0..n-1 theo đúng thứ tự đã xếp hạng rồi đổi sang RetrievedRow."""
        return [t.to_row(rank=rank) for rank, t in enumerate(tables)]
