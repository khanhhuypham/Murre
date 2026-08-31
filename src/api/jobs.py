"""api/jobs.py — Thân của một lần chạy pipeline dưới dạng job trong RAM.

`cfg` là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT job; router
/pipeline/run là nơi chặn (409) khi đã có job chưa kết thúc.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from starlette.datastructures import State

from api.evaluator import evaluate_run
from config import cfg
from enums import JobStatus
from methods.runner import run_pipeline
from models.errors import AppError
from schemas.pipeline import PipelineJob, PipelineRunRequest
from utils import logger


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_job(state: State, job_id: str, req: PipelineRunRequest) -> None:
    """Thân của một job — chạy trong thread riêng (không chạm event loop)."""
    job: PipelineJob = state.jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = now()

    def on_progress(done: int, total: int) -> None:
        job.processed = done
        job.total = total

    try:
        # Khong truyen model: giu nguyen encoder.slug cua server (xem
        # PipelineRunRequest).
        run_pipeline(
            dataset=req.dataset,
            method=req.method,
            limit=req.limit,
            on_progress=on_progress,
        )
        # Đọc lại metric bằng đúng đường code của /evaluate → hai endpoint không thể
        # lệch số nhau, và cũng xác nhận file vừa ghi đọc được thật.
        job.result = evaluate_run(
            dataset=req.dataset,
            model=cfg.encoder.slug,
            method=req.method,
            k=req.k,
        )
        job.status = JobStatus.SUCCEEDED
    except AppError as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    except HTTPException as e:
        job.status = JobStatus.FAILED
        job.error = f"{e.status_code}: {e.detail}"
    except Exception as e:
        logger.exception(f"[API] Job {job_id} thất bại")
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = now()
