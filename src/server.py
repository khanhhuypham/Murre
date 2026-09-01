"""server.py — Điểm vào API service: tạo FastAPI app, lifespan, gắn router.

api/ là namespace package (không có __init__.py, giống core/, methods/, ...) nên
uvicorn cần một module phẳng như file này để nạp. Endpoint nằm hết trong
api/routers/; file này không chứa endpoint nào.

Trạng thái dùng chung nằm trong app.state (khởi tạo ở lifespan); router đọc qua
request.app.state nên không file nào phải import ngược lại đây.

CÁCH CHẠY — phải đứng trong src/ (config.py chdir lúc import):
  cd src && uvicorn server:app --host 0.0.0.0 --port 8000 --reload
  cd src && python -m server        (không reload, host/port lấy từ config.yaml)

Docs: http://localhost:8000/docs
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.dependencies import warmup_datasets
from api.routers import evaluate, health, pipeline, retrieve, sql
from config import cfg


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
    yield


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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
