"""api/jobs.py — Thân của một lần chạy pipeline dưới dạng job trong RAM.

`cfg` là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT job. CHẶN Ở HAI
TẦNG, cố ý:

    router /pipeline/run    từ chối sớm (409) khi còn job chưa kết thúc — người
                            gọi biết ngay, không phải đợi rồi mới thấy job FAILED.
    runner._RUN_LOCK        chốt thật, vì run_pipeline() còn được gọi từ CLI chứ
                            không riêng router; nó ném AppError.pipeline_busy().

Job chỉ nằm trong RAM (app.state.jobs) — restart service là mất. Kết quả thì đã
ghi ra đĩa, tra lại bằng /evaluate.
"""
from __future__ import annotations

from datetime import datetime, timezone

from starlette.datastructures import State

from api.evaluator import evaluate_run
from config import cfg
from enums import JobStatus
from models.errors import AppError
from pipeline.runner import run_pipeline
from schemas.pipeline import PipelineJob, PipelineRunRequest
from utils import logger


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_job(state: State, job_id: str, req: PipelineRunRequest) -> None:
    """Thân của một job — chạy trong thread riêng (không chạm event loop).

    KHÔNG ném ra ngoài: mọi lỗi được ghi vào `job.status` + `job.error` rồi nuốt.
    Chỗ gọi (`await task` trong /pipeline/run khi wait=true) vì vậy luôn chạy tới
    nơi; muốn biết chạy được hay không thì đọc `job.status`, đừng bắt exception.
    """
    job: PipelineJob = state.jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = now()

    def on_progress(done: int, total: int) -> None:
        job.processed = done
        job.total = total

    try:
        # Không truyền model: encoder do server quyết định (xem PipelineRunRequest).
        run_pipeline(dataset=req.dataset, limit=req.limit, on_progress=on_progress)
        # Đọc lại metric bằng đúng đường code của /evaluate → hai endpoint không thể
        # lệch số nhau, và cũng xác nhận file vừa ghi đọc được thật.
        job.result = evaluate_run(
            dataset=req.dataset,
            model=cfg.encoder_for(dataset=req.dataset).slug,
            k=req.k,
        )
        job.status = JobStatus.SUCCEEDED
    except AppError as e:
        logger.warning(f"[API] Job {job_id} thất bại: {e}")
        job.status = JobStatus.FAILED
        job.error = e.message
    except Exception as e:
        logger.exception(f"[API] Job {job_id} thất bại")
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = now()
