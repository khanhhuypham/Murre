# =============================================================================
# models/records.py — Định dạng RECORD của các file JSON trong outputs/
#
# Còn đúng MỘT định dạng, là kết quả cuối của pipeline:
#
#   methods/runner.py   ──ghi──>  ResultRecord  (result/turn{H}/dev.json)
#   steps/infer.py      ──đọc──>  ResultRecord  (để sinh SQL)
#   utils/metrics.py, /evaluate  ──đọc──>  ResultRecord
#
# TurnRecord/RewriteRecord (file trung gian turn{N}/, rewrite/outputs/turn{N}/) đã
# bị xoá cùng chuỗi steps/{retrieve,rewrite,score}.py — bản cài đặt MURRE thứ hai.
# MurreRetriever giữ mọi thứ trong RAM nên không còn file trung gian nào.
#
# QUY TẮC KHI SỬA: thứ tự field trong mỗi class CHÍNH LÀ thứ tự khóa ghi ra JSON.
# Đổi thứ tự thì file mới vẫn đọc được (JSON không quan tâm thứ tự), nhưng diff
# giữa hai lần chạy sẽ nhiễu. Thêm field mới thì thêm vào CUỐI.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from models.retrieval import RetrievedRow

@dataclass
class ResultRecord:
    """Một câu hỏi trong result/turn{H}/dev.json — kết quả CUỐI CÙNG của pipeline.

    Đây là định dạng mà utils/metrics.py, steps/infer.py và /evaluate cùng đọc, nên
    3 trường dưới đây là phần hợp đồng thật sự giữa các module.

    `extra` giữ nguyên mọi khóa khác của record. methods/runner.py ghi đúng 3 khóa,
    nhưng file cũ do steps/score.py (đã xoá) sinh ra còn kèm input/utterance_org/
    selected_database/recall. Nhờ `extra` mà /evaluate vẫn đọc-rồi-ghi-lại được
    những file đó không mất khóa nào, và giữ nguyên thứ tự khóa như file gốc.
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
