# =============================================================================
# server.py — Điểm vào của API service (endpoint nằm hết trong api/).
#
# api/ là namespace package (không có __init__.py, giống core/, methods/, ...)
# nên uvicorn cần một module phẳng như file này để nạp.
#
# CÁCH CHẠY — phải đứng trong src/ (config.py chdir lúc import):
#   cd src && uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#   cd src && python -m server        (không reload, sửa cổng ở dưới)
#
# Docs: http://localhost:8000/docs
# =============================================================================
from __future__ import annotations

from api.app import app

__all__ = ["app"]


# =============================================================================
# ĐIỂM CHẠY — sửa trực tiếp mấy biến dưới đây rồi: python -m server
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    uvicorn.run(app, host=HOST, port=PORT)
