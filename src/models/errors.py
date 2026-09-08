"""models/errors.py — Tập lỗi chuẩn của hệ thống.

Một lớp AppError + classmethod cho từng loại lỗi. Thêm loại mới = thêm 1 classmethod
3 dòng, không thêm lớp.

    raise AppError.not_found(message=f"Không có job '{job_id}'.")
    raise AppError.bad_request(message=f"k={k} lớn hơn số bảng đã lưu ({depth}).")

`message` là câu trả cho client; lỗi gốc nằm ở `cause` (KHÔNG serialize) để không
lộ thông tin nội bộ.

Tầng lõi ném AppError (không import FastAPI); tầng API dịch sang HTTP đúng MỘT lần
bằng exception handler trong server.py, đọc `err.status`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    """Lỗi có chủ đích của hệ thống.

    message : câu trả cho client
    status  : HTTP status
    cause   : lỗi gốc — CHỈ log phía server, không bao giờ serialize ra client
    """

    def __init__(
        self,
        message: str,
        status: int,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.status: int = status
        self.cause: Optional[BaseException] = cause

    # --- Factory: mỗi loại lỗi một classmethod, message mặc định sẵn ----------
    @classmethod
    def bad_request(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> "AppError":
        return cls(message or "Yêu cầu không hợp lệ", 400, cause)

    @classmethod
    def not_found(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> "AppError":
        return cls(message or "Không tìm thấy", 404, cause)

    @classmethod
    def conflict(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> "AppError":
        return cls(message or "Dữ liệu bị xung đột", 409, cause)

    @classmethod
    def pipeline_busy(cls) -> "AppError":
        """cfg là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT lần chạy."""
        return cls("Đang có một lần chạy pipeline khác, đợi xong rồi thử lại.", 409)

    @classmethod
    def internal(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> "AppError":
        return cls(message or "Lỗi hệ thống", 500, cause)

    # --- Truy vấn ------------------------------------------------------------
    def root_cause(self) -> Optional[BaseException]:
        """Đệ quy unwrap qua nhiều lớp AppError lồng nhau để lấy lỗi gốc thật sự."""
        if isinstance(self.cause, AppError):
            return self.cause.root_cause()
        return self.cause

    def to_dict(self) -> Dict[str, Any]:
        """Body trả cho client. `cause` KHÔNG có ở đây."""
        return {"message": self.message, "status": self.status}

    def __str__(self) -> str:
        root: Optional[BaseException] = self.root_cause()
        return f"{self.message} | {root}" if root is not None else self.message
