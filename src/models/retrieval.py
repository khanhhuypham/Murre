# =============================================================================
# models/retrieval.py — Đối tượng miền cho KẾT QUẢ RETRIEVAL
#
# Hai kiểu ở đây trông giống nhau nhưng KHÁC vai trò, đừng nhầm:
#
#   RetrievedTable : kết quả TRONG BỘ NHỚ mà methods/*.run() trả về  → (schema, score)
#   RetrievedRow   : một dòng trong mảng "retrieved" của FILE JSON   → (rank, schema, similarity)
#
# Trước đây cả hai đều là dict trần (`List[Dict[str, Any]]`), nên chỗ nối giữa hai
# thế giới phải đổi tên khóa bằng tay — methods/runner.py từng có nguyên một comment
# giải thích vì sao "score" phải viết ra thành "similarity". Nay việc đó nằm gọn
# trong RetrievedTable.to_row(), không ai phải nhớ nữa.
#
# Vì sao KHÔNG dùng pydantic như src/schemas/: schemas/ là DTO của API (dữ liệu từ
# ngoài vào, cần validate). Hai kiểu này chạy trong vòng lặp nóng — mỗi câu hỏi sinh
# ra top_k_pool object — và không bao giờ nhận dữ liệu từ người dùng, nên NamedTuple
# vừa nhẹ vừa đủ an toàn.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple


class RetrievedRow(NamedTuple):
    """Một dòng trong mảng "retrieved" của file JSON (turn*/dev*.json, result/dev.json).

    Thứ tự field ở đây CHÍNH LÀ thứ tự khóa khi ghi ra JSON — giữ nguyên
    rank/schema/similarity để file mới đọc được bằng code cũ và ngược lại.
    """

    # Thứ hạng trong danh sách, đếm từ 0
    rank: int
    # Chuỗi schema "db_id.table(col1, col2, ...)"
    schema: str
    # Điểm tương đồng với câu truy vấn. Ở turn*/dev*.json đây là cosine similarity;
    # ở result/dev.json thì là Score_Table (log-scale, có thể âm) — cùng tên khóa vì
    # file format của tác giả gốc như vậy.
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
    """Một bảng do methods/*.run() trả về — dùng trong bộ nhớ, không ghi thẳng ra file.

    Cả 3 method (murre / single_hop / crush) đều trả về List[RetrievedTable]; đây
    chính là giao diện chung mà methods/factory.build_retriever() hứa hẹn.
    """

    # Chuỗi schema "db_id.table(col1, col2, ...)"
    schema: str
    # Điểm xếp hạng của method: cosine similarity (single_hop/crush) hoặc
    # Score_Table (murre). Càng lớn càng tốt.
    score: float

    def to_row(self, rank: int) -> RetrievedRow:
        """Đổi sang dòng của file JSON. `score` được ghi dưới tên khóa `similarity`
        vì đó là tên trong file format gốc — xem RetrievedRow.similarity."""
        return RetrievedRow(rank=rank, schema=self.schema, similarity=self.score)

    @staticmethod
    def to_rows(tables: Iterable[RetrievedTable]) -> List[RetrievedRow]:
        """Đánh số rank 0..n-1 theo đúng thứ tự đã xếp hạng rồi đổi sang RetrievedRow."""
        return [t.to_row(rank=rank) for rank, t in enumerate(tables)]
