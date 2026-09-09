"""Tải ViText2SQL: đối chiếu file trên đĩa với upstream từng byte.

Điểm quan trọng nhất của script tải là KHÔNG âm thầm chấp nhận file lệch. Test ở
đây không gọi mạng — chỉ kiểm phần băm và phần so khớp.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

import pytest

from download_vitext2sql import check, git_blob_sha


def _entry(data: bytes) -> Dict[str, Any]:
    return {"size": len(data), "sha": git_blob_sha(data)}


def test_git_blob_sha_matches_gits_own_formula() -> None:
    """Băm phải trùng cách git băm, nếu không thì không đối chiếu được với API."""
    data = b"xin chao"
    expected = hashlib.sha1(b"blob 8\0" + data).hexdigest()
    assert git_blob_sha(data) == expected


def test_git_blob_sha_of_empty_file() -> None:
    # Blob rỗng của git là hằng số ai cũng biết — mốc tốt để chốt công thức.
    assert git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_identical_bytes_pass() -> None:
    data = b'{"db_id": "architecture"}'
    assert check(data=data, entry=_entry(data)) is None


def test_a_single_changed_byte_is_caught() -> None:
    """Cùng kích thước nhưng khác nội dung — chỉ sha mới bắt được."""
    data = b'{"db_id": "architecture"}'
    entry = _entry(data)
    tampered = data.replace(b"architecture", b"architectvre")

    assert len(tampered) == entry["size"]          # kích thước không đổi
    assert "sha" in check(data=tampered, entry=entry)


def test_a_truncated_file_is_caught_by_size() -> None:
    """Tải dở giữa chừng — bắt được ngay ở bước so kích thước."""
    data = b"noi dung day du"
    assert "kích thước" in check(data=data[:5], entry=_entry(data))


def test_reformatted_json_is_caught() -> None:
    """json.load rồi json.dump lại là đổi thụt lề/escape → không còn khớp upstream.

    Đây chính là lý do script ghi bytes thô thay vì parse rồi ghi lại.
    """
    original = rb'{"a": 1, "b": "ti\u1ebfng vi\u1ec7t"}'
    reformatted = '{"a": 1, "b": "tiếng việt"}'.encode("utf-8")
    assert check(data=reformatted, entry=_entry(original)) is not None
