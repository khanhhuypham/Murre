"""models/errors.py — Tập lỗi chuẩn của hệ thống. MỚI CHỈ ĐỊNH NGHĨA, chưa gắn.

Một lớp AppError + classmethod cho từng loại lỗi. Thêm loại mới = thêm 1 classmethod
3 dòng, không thêm lớp.

    raise AppError.not_found()
    raise AppError.entity_not_found("job", cause=exc)
    raise AppError.bad_request(message=f"k={k} lớn hơn số bảng đã lưu ({depth}).")
    raise AppError.internal(cause=exc).with_debug(traceback.format_exc())

`message` là câu CHUNG trả cho client; chi tiết thật nằm ở `causes` (không serialize)
để không lộ thông tin nội bộ. Chỗ nào chi tiết vô hại thì truyền `message=` đè lên.

Tầng lõi ném AppError (không import FastAPI); tầng API dịch sang HTTP một lần bằng
exception handler, đọc `err.status`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    """Lỗi có chủ đích của hệ thống.

    message : câu tiếng Việt trả cho client
    status  : HTTP status
    causes  : lỗi gốc — CHỈ log phía server, không bao giờ serialize ra client
    debug   : chỉ bật ở môi trường development, tránh lộ thông tin nhạy cảm
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
        self.causes: Optional[BaseException] = cause
        self.debug: Optional[str] = None

    # --- Factory: mỗi loại lỗi một classmethod, message mặc định sẵn ----------
    @classmethod
    def bad_request(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> AppError:
        return cls(message or "Yêu cầu không hợp lệ", 400, cause)

    @classmethod
    def not_found(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> AppError:
        return cls(message or "Không tìm thấy", 404, cause)

    @classmethod
    def entity_not_found(cls, entity: str, cause: Optional[BaseException] = None) -> AppError:
        return cls(f"Không tìm thấy {entity}", 404, cause)

    @classmethod
    def conflict(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> AppError:
        return cls(message or "Dữ liệu bị xung đột", 409, cause)

    @classmethod
    def pipeline_busy(cls) -> AppError:
        """cfg là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT lần chạy."""
        return cls("Đang có một lần chạy pipeline khác, đợi xong rồi thử lại.", 409)

    @classmethod
    def internal(
        cls, cause: Optional[BaseException] = None, message: Optional[str] = None
    ) -> AppError:
        return cls(message or "Lỗi hệ thống", 500, cause)

    # --- Truy vấn ------------------------------------------------------------
    def root_cause(self) -> Optional[BaseException]:
        """Đệ quy unwrap qua nhiều lớp AppError lồng nhau để lấy lỗi gốc thật sự."""
        if isinstance(self.causes, AppError):
            return self.causes.root_cause()
        return self.causes

    def with_debug(self, debug: str) -> AppError:
        """Gắn thông tin debug rồi trả lại chính nó — để viết được một dòng."""
        self.debug = debug
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Body trả cho client. `causes` KHÔNG có ở đây; `debug` bỏ qua nếu None."""
        body: Dict[str, Any] = {"message": self.message, "status": self.status}
        if self.debug:
            body["debug"] = self.debug
        return body

    def __str__(self) -> str:
        """Cause nil thì trả message, không gọi .Error() trên nil."""
        root: Optional[BaseException] = self.root_cause()
        return f"{self.message} | {root}" if root is not None else self.message
