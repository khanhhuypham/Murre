"""api/jobs.py — Thân của một lần chạy pipeline dưới dạng job trong RAM.

Mỗi lúc chỉ chạy được MỘT job, chặn ở hai tầng: router /pipeline/run trả 409
sớm, và runner._RUN_LOCK là chốt thật (run_pipeline còn gọi từ CLI).

Job chỉ nằm trong RAM — restart service là mất; kết quả đã ghi ra đĩa.
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
    """Thân của một job — chạy trong thread riêng, KHÔNG ném lỗi ra ngoài.

    Mọi lỗi ghi vào `job.status` + `job.error`; chỗ gọi đọc status, đừng bắt.
    """
    job: PipelineJob = state.jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = now()

    def on_progress(done: int, total: int) -> None:
        job.processed = done
        job.total = total

    try:
        # Không truyền model: encoder do server quyết định (xem PipelineRunRequest).
        run_pipeline(
            dataset=req.dataset,
            limit=req.limit,
            on_progress=on_progress,
            verbose=req.verbose,
        )
        # Đọc lại metric bằng đúng đường code của /evaluate.
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
