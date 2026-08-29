# =============================================================================
# steps/_common.py — Phần dùng chung giữa các bước của Option 2 (batch)
#
# Giữ đoạn code mà nhiều bước trong steps/ cùng cần, để không phải chép đi chép lại.
# =============================================================================
from __future__ import annotations

import os
from typing import List


def find_json_files(directory: str) -> List[str]:
    """Mọi file .json trong thư mục (kể cả thư mục con), sắp xếp theo tên.

    Sắp xếp là quan trọng: thứ tự file quyết định thứ tự beam khi steps/rewrite.py
    và steps/score.py duyệt qua turn{N}/, nên phải ổn định giữa các lần chạy và
    giữa các hệ điều hành.
    """
    files: List[str] = []
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)
