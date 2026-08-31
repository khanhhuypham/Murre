"""api/app.py — Khởi tạo FastAPI app: lifespan + đăng ký router.

File này KHÔNG chứa endpoint nào. Cấu trúc cả gói:

    server.py           ← điểm vào: uvicorn server:app (nằm ngoài, ở src/)
    api/
    ├── app.py          ← tạo FastAPI, lifespan, gắn router  (file này)
    ├── dependencies.py ← nạp encoder/LLM/embeddings theo dataset
    ├── evaluator.py    ← evaluate_run(): tính metric từ file kết quả
    ├── jobs.py         ← thân một job chạy pipeline trong thread riêng
    └── routers/        ← health | retrieve | pipeline | evaluate | sql

Trạng thái dùng chung nằm trong app.state (khởi tạo ở lifespan); router đọc qua
request.app.state nên không file nào phải import ngược lại app.

Chạy: cd src && uvicorn server:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.dependencies import available_datasets, warmup_datasets
from api.routers import evaluate, health, pipeline, retrieve, sql
from utils import logger

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ô trạng thái rỗng — warmup_datasets() bên dưới sẽ đổ đầy.
    app.state.encoder = None
    app.state.llm = None
    app.state.datasets = {}      # Dataset -> LoadedDataset
    app.state.load_lock = asyncio.Lock()
    app.state.jobs = {}          # job_id -> PipelineJob
    app.state.job_tasks = {}     # job_id -> asyncio.Task (giữ ref để không bị GC)

    await warmup_datasets(app.state)

    logger.info(f"[API] Sẵn sàng. Dataset có sẵn: {available_datasets()} | "
                f"đã nạp: {[str(d) for d in app.state.datasets]}")
    yield


def create_app() -> FastAPI:
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
    application.include_router(health.router)
    application.include_router(retrieve.router)
    application.include_router(pipeline.router)
    application.include_router(evaluate.router)
    application.include_router(sql.router)
    return application


app = create_app()
