"""Pha Removal: nhận diện Early Stop và cắt phần model chép lại prompt (§3.4)."""
from __future__ import annotations

import pytest

from core.rewriter import QueryRewriter


@pytest.mark.parametrize(
    "output",
    [
        "None",
        "There is no additional table needed.",
        "No completion needed",
        "None of the given tables",
        "Completing Tables: None",   # model chép lại nhãn rồi mới trả lời
    ],
)
def test_early_stop_signals(output: str) -> None:
    assert QueryRewriter.is_early_stop(rewrite_output=output) is True


@pytest.mark.parametrize(
    "output",
    [
        "flight_2.airlines(airline id, airline name, country)",
        # "None" nằm GIỮA một danh sách hợp lệ không phải tín hiệu dừng — model nhỏ
        # sinh ra kiểu này liên tục, quét cả chuỗi bằng `in` là dừng sớm oan.
        "concert.singer(name, country)\nNone of these matter",
        "",
    ],
)
def test_not_early_stop(output: str) -> None:
    assert QueryRewriter.is_early_stop(rewrite_output=output) is False


def test_strip_echo_removes_only_the_leading_label() -> None:
    cleaned = QueryRewriter._strip_echo(text="Completing Tables: db.t(a, b)")
    assert cleaned == "db.t(a, b)"


def test_strip_echo_keeps_multiline_body() -> None:
    cleaned = QueryRewriter._strip_echo(text="Rewritten Question:\ndb.a(x)\ndb.b(y)")
    assert cleaned == "db.a(x)\ndb.b(y)"
