"""server.py — Điểm vào API service: tạo FastAPI app, lifespan, gắn router.

api/ là namespace package (không có __init__.py, giống core/, pipeline/, ...) nên
uvicorn cần một module phẳng như file này để nạp. Endpoint nằm hết trong
api/routers/; file này không chứa endpoint nào.

Trạng thái dùng chung nằm trong app.state (khởi tạo ở lifespan); router đọc qua
request.app.state nên không file nào phải import ngược lại đây.

CÁCH CHẠY — phải đứng trong src/ (config.py chdir lúc import):
  cd src && uvicorn server:app --host 0.0.0.0 --port 8000
  cd src && python -m server        (host/port lấy từ config.yaml)

Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import socket
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import warmup_datasets
from api.routers import evaluate, health, pipeline, retrieve, sql
from config import cfg
from models.errors import AppError
from utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Ô trạng thái rỗng — warmup_datasets() bên dưới sẽ đổ đầy.
    app.state.llm = None
    app.state.datasets = {}      # Dataset -> LoadedDataset
    app.state.load_lock = asyncio.Lock()
    app.state.jobs = {}          # job_id -> PipelineJob
    app.state.job_tasks = {}     # job_id -> asyncio.Task (giữ ref để không bị GC)
    app.state.ready = False      # /health trả 503 tới khi warmup xong

    await warmup_datasets(app.state)
    yield


async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """AppError → HTTP. Tầng lõi ném lỗi có ngữ nghĩa, dịch sang HTTP đúng ở đây."""
    if exc.status >= 500:
        logger.error(f"[API] {request.method} {request.url.path} → {exc}")
    return JSONResponse(status_code=exc.status, content=exc.to_dict())


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Lỗi không lường trước → 500 với message chung; chi tiết chỉ vào log.

    api.debug_errors=true thì kèm traceback vào response — CHỈ dùng khi debug.
    """
    logger.exception(f"[API] Lỗi không lường trước ở {request.method} {request.url.path}")
    body = AppError.internal().to_dict()
    if cfg.api.debug_errors:
        body["debug"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=body)


def create_app() -> FastAPI:
    """Tạo FastAPI app và gắn toàn bộ router."""
    application = FastAPI(
        title="MURRE — Multi-Hop Table Retrieval API",
        description=(
            "API cho hệ thống MURRE: retrieve bảng SQL liên quan từ câu hỏi tự nhiên.\n\n"
            "Dựa trên: MURRE: Multi-Hop Table Retrieval with Removal for Open-Domain "
            "Text-to-SQL (COLING 2025)"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    if cfg.api.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    application.add_exception_handler(AppError, _app_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)

    application.include_router(health.router)
    application.include_router(retrieve.router)
    application.include_router(pipeline.router)
    application.include_router(evaluate.router)
    application.include_router(sql.router)
    return application


app = create_app()


# =============================================================================
# ĐIỂM CHẠY — host/port lấy từ api.host / api.port trong config.yaml
# =============================================================================
def ensure_port_free(host: str, port: int) -> None:
    """Kiểm tra cổng còn trống TRƯỚC khi nạp model.

    uvicorn chỉ bind sau khi lifespan chạy xong, mà lifespan là chỗ nạp encoder và
    mã hoá cả corpus — mất vài phút. Cổng đang bận thì phải đợi hết ngần ấy thời
    gian mới thấy `[Errno 10048]`, và công mã hoá đó vứt đi. Thử bind trước ở đây
    thì lỗi hiện ra trong tích tắc.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
            return
        except OSError as exc:
            raise SystemExit(
                f"Cổng {port} đang bị chiếm — không khởi động được ({exc.strerror}).\n"
                f"  Xem ai đang giữ:  netstat -ano | findstr :{port}\n"
                f"  Giải phóng (PowerShell):\n"
                f"    Get-NetTCPConnection -LocalPort {port} -State Listen | "
                f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}\n"
                f"  Hoặc đổi api.port trong config.yaml sang cổng khác."
            ) from None


if __name__ == "__main__":
    import uvicorn

    # Bind thử vào chính host trong config; 0.0.0.0 thì thử localhost cho tương đương.
    ensure_port_free(host=cfg.api.host, port=cfg.api.port)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
