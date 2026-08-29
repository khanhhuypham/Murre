# =============================================================================
# models/records.py — Định dạng RECORD của các file JSON trong outputs/
#
# Pipeline có ĐÚNG 3 định dạng record, mỗi bước đọc của bước trước rồi ghi ra
# định dạng tiếp theo:
#
#   steps/retrieve.py  ──ghi──>  TurnRecord    (turn{N}/dev*.json)
#   steps/rewrite.py   ──ghi──>  RewriteRecord (rewrite/outputs/turn{N}/dev.{B}.json)
#   steps/score.py     ──ghi──>  ResultRecord  (result/turn{H}/dev.json)
#   core/runner.py     ──ghi──>  ResultRecord  (cùng file, đường Option 1 offline)
#
# Trước đây cả 3 đều là `List[Dict[str, Any]]`, nên đọc nhầm định dạng chỉ vỡ lúc
# chạy: `KeyError: 'retrieved'` khi đưa file rewrite vào chỗ đợi file retrieve —
# lỗi này nhiều tới mức đã thành một mục trong HUONG_DAN.md phần Troubleshooting.
#
# QUY TẮC KHI SỬA: thứ tự field trong mỗi class CHÍNH LÀ thứ tự khóa ghi ra JSON.
# Đổi thứ tự thì file mới vẫn đọc được (JSON không quan tâm thứ tự), nhưng diff
# giữa hai lần chạy sẽ nhiễu. Thêm field mới thì thêm vào CUỐI.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.retrieval import RetrievedRow

# Một bước trên đường đi của beam: (schema đã chọn, similarity lúc chọn nó).
# Trong JSON nó là mảng 2 phần tử ["db.t(...)", 0.447], không phải object.
BeamStep = Tuple[str, float]


def _to_beam_steps(raw: Any) -> List[BeamStep]:
    """Đọc `selected_database` từ JSON: list of [schema, similarity]."""
    if not raw:
        return []
    return [(item[0], float(item[1])) for item in raw]


def _to_str_list(raw: Any) -> List[str]:
    """`utterance_org` khi thì là chuỗi, khi thì là list — chuẩn hóa về list."""
    if raw is None:
        return []
    return [raw] if isinstance(raw, str) else list(raw)


def _to_recall(raw: Any) -> Dict[int, float]:
    """Chuẩn hóa khóa của `recall` về int.

    JSON bắt buộc khóa phải là chuỗi, nên `{3: 1.0}` ghi ra thành `{"3": 1.0}` rồi
    đọc lại thành khóa chuỗi. Ép về int ngay tại đây để trong bộ nhớ luôn là int,
    tránh cảnh cùng một dict mà `d[3]` chạy được ở bước này nhưng `d["3"]` mới chạy
    được ở bước sau. Ghi ra thì json.dump tự đổi lại thành chuỗi như cũ.
    """
    if not raw:
        return {}
    return {int(k): float(v) for k, v in raw.items()}


@dataclass
class TurnRecord:
    """Một câu hỏi trong file turn{N}/dev*.json — đầu ra của steps/retrieve.py."""

    # Câu truy vấn dùng cho hop này (hop 0 là câu hỏi gốc, hop N là câu đã Removal)
    utterance: str
    # Câu truy vấn sau khi tách dòng thành các sub-query
    input: List[str]
    # Câu hỏi GỐC của người dùng, giữ nguyên qua mọi hop
    utterance_org: List[str]
    # Đường đi của beam tới trước hop này: [(schema, similarity), ...]
    selected_database: List[BeamStep]
    # Danh sách bảng retrieve được ở hop này, đã xếp hạng
    retrieved: List[RetrievedRow]
    # Schema đúng (từ rel_schema của dev.json) — dùng để tính recall
    gold: List[str]
    # recall@k của riêng hop này, khóa là k
    recall: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TurnRecord:
        return cls(
            # Fallback "question": giữ tương thích với file của tác giả gốc, vốn đặt
            # tên trường câu hỏi là "question" thay vì "utterance".
            utterance=d.get("utterance") or d.get("question", ""),
            input=_to_str_list(d.get("input")),
            utterance_org=_to_str_list(d.get("utterance_org")),
            selected_database=_to_beam_steps(d.get("selected_database")),
            retrieved=RetrievedRow.from_list(items=d.get("retrieved", [])),
            gold=list(d.get("gold", [])),
            recall=_to_recall(d.get("recall")),
        )

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> List[TurnRecord]:
        return [cls.from_dict(d=d) for d in items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utterance": self.utterance,
            "input": self.input,
            "utterance_org": self.utterance_org,
            "selected_database": [list(step) for step in self.selected_database],
            "retrieved": [r.to_dict() for r in self.retrieved],
            "gold": self.gold,
            "recall": self.recall,
        }


@dataclass
class RewriteRecord:
    """Một câu hỏi trong rewrite/outputs/turn{N}/dev.{B}.json — đầu ra của steps/rewrite.py.

    KHÔNG có trường `retrieved`: đây mới chỉ là câu hỏi đã viết lại, chưa retrieve.
    Đưa file này vào chỗ đợi TurnRecord chính là nguyên nhân của `KeyError: 'retrieved'`.

    Trường gold ở đây tên là `rel_schema` (không phải `gold`) vì file này được
    steps/retrieve.py đọc lại như một file "dev.json" ở hop sau.
    """

    utterance: str
    utterance_org: List[str]
    selected_database: List[BeamStep]
    rel_schema: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RewriteRecord:
        return cls(
            utterance=d.get("utterance", ""),
            utterance_org=_to_str_list(d.get("utterance_org")),
            selected_database=_to_beam_steps(d.get("selected_database")),
            rel_schema=list(d.get("rel_schema", [])),
        )

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> List[RewriteRecord]:
        return [cls.from_dict(d=d) for d in items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utterance": self.utterance,
            "utterance_org": self.utterance_org,
            "selected_database": [list(step) for step in self.selected_database],
            "rel_schema": self.rel_schema,
        }


@dataclass
class ResultRecord:
    """Một câu hỏi trong result/turn{H}/dev.json — kết quả CUỐI CÙNG của pipeline.

    Đây là định dạng mà utils/metrics.py, steps/infer.py và /evaluate cùng đọc, nên
    3 trường dưới đây là phần hợp đồng thật sự giữa các module.

    `extra` giữ nguyên mọi khóa khác của record. Hai đường sinh ra file này ghi số
    khóa khác nhau — core/runner.py (Option 1) ghi đúng 3 khóa, còn steps/score.py
    (Option 2) tái sử dụng luôn record của turn0 nên còn kèm input/utterance_org/
    selected_database/recall. Nhờ `extra` mà đọc-rồi-ghi-lại không làm mất khóa nào,
    và thứ tự khóa cũng giữ nguyên như file gốc.
    """

    utterance: str
    gold: List[str]
    retrieved: List[RetrievedRow]
    # Các khóa ngoài 3 khóa trên, giữ nguyên để ghi lại không mất dữ liệu
    extra: Dict[str, Any] = field(default_factory=dict)
    # Thứ tự khóa của record gốc, chỉ dùng để ghi lại cho khớp file cũ
    _key_order: Optional[List[str]] = None

    _OWN_KEYS = ("utterance", "gold", "retrieved")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResultRecord:
        return cls(
            utterance=d.get("utterance", ""),
            gold=list(d.get("gold", [])),
            retrieved=RetrievedRow.from_list(items=d.get("retrieved", [])),
            extra={k: v for k, v in d.items() if k not in cls._OWN_KEYS},
            _key_order=list(d.keys()),
        )

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> List[ResultRecord]:
        return [cls.from_dict(d=d) for d in items]

    def to_dict(self) -> Dict[str, Any]:
        own: Dict[str, Any] = {
            "utterance": self.utterance,
            "gold": self.gold,
            "retrieved": [r.to_dict() for r in self.retrieved],
        }
        merged: Dict[str, Any] = {**own, **self.extra}
        if self._key_order is None:
            return merged
        # Ghi lại đúng thứ tự khóa của file gốc; khóa mới (nếu có) xếp cuối.
        ordered: Dict[str, Any] = {k: merged[k] for k in self._key_order if k in merged}
        ordered.update({k: v for k, v in merged.items() if k not in ordered})
        return ordered

    @property
    def schemas(self) -> List[str]:
        """Danh sách schema đã xếp hạng — thứ mà metrics và infer thực sự cần."""
        return [r.schema for r in self.retrieved]
