"""api/routers/pipeline.py — CHẠY pipeline cho một dataset.

Encoder KHÔNG nằm trong API: server lấy từ encoder.model_name (config.yaml).
Vì vậy request lẫn response đều không có trường `model` — muốn biết model nào thì
đọc `result.result_file`, đường dẫn có sẵn nhãn model trong đó.

/evaluate chỉ ĐỌC metric của lần chạy đã có. Nhóm endpoint này mới là thứ TẠO ra
lần chạy đó: nó ghi `paths.result` + `paths.score`, nên chạy xong là /evaluate
tra được ngay với mọi k.

Mặc định POST chờ tới khi chạy xong rồi trả kết quả (200). Lần chạy dài hơn
timeout của client thì dùng `?wait=false` → trả job ngay (202), poll bằng
GET /pipeline/jobs/{job_id}.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Query, Request, Response

from api.dependencies import require_dataset
from api.jobs import run_job
from enums import JobStatus
from models.errors import AppError
from schemas.pipeline import PipelineJob, PipelineRunRequest

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post(
    "/run",
    response_model=PipelineJob,
    status_code=200,
    summary="Chạy pipeline trên cả dev.json rồi tính metric tại k",
)
async def pipeline_run(
    payload: PipelineRunRequest,
    request: Request,
    response: Response,
    wait: Annotated[bool, Query(
        description="true (mặc định) = chờ chạy xong rồi trả kết quả luôn → 200. "
                    "false = trả job ngay để poll /pipeline/jobs/{job_id} → 202; "
                    "dùng cho lần chạy dài (cả dev.json mất hàng giờ).",
    )] = True,
) -> PipelineJob:
    """Chạy pipeline rồi trả kết quả.

    Mặc định `wait=true`: request giữ mở tới khi chạy xong, response đã có sẵn
    `result` (metric tại k) — gọi một phát là có luôn. Đặt `wait=false` khi lần
    chạy dài hơn timeout của client/proxy; muốn thử nhanh thì đặt `limit` nhỏ.

    `cfg` là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT job; gọi khi
    đang có job khác sẽ nhận `409`.
    """
    require_dataset(ds_name=payload.dataset)

    state = request.app.state
    running: List[str] = [j.job_id for j in state.jobs.values() if not j.status.is_final]
    if running:
        raise AppError.conflict(
            message=f"Đang có job chạy: {running}. Đợi xong rồi thử lại."
        )

    job_id: str = uuid.uuid4().hex[:12]
    job = PipelineJob(
        job_id=job_id,
        status=JobStatus.QUEUED,
        dataset=payload.dataset,
        k=payload.k,
        limit=payload.limit,
    )
    state.jobs[job_id] = job

    # to_thread() chuyển tiếp nguyên keyword argument xuống run_job.
    task: asyncio.Task = asyncio.create_task(
        asyncio.to_thread(
            run_job,
            state=state,
            job_id=job_id,
            req=payload,
        )
    )
    state.job_tasks[job_id] = task

    if wait:
        await task          # chạy xong mới trả → job.result đã có metric
    else:
        # Chưa chạy xong, chỉ mới nhận việc → 202 Accepted mới đúng ngữ nghĩa.
        response.status_code = 202
    return job


@router.get("/jobs", response_model=List[PipelineJob], summary="Danh sách job đã tạo")
async def pipeline_jobs(request: Request) -> List[PipelineJob]:
    """Job chỉ nằm trong RAM — restart service là mất. Kết quả thì đã ghi ra đĩa,
    tra lại bằng /evaluate hoặc /evaluate/available."""
    return list(request.app.state.jobs.values())


@router.get(
    "/jobs/{job_id}",
    response_model=PipelineJob,
    summary="Trạng thái/tiến độ một lần chạy",
)
async def pipeline_job(job_id: str, request: Request) -> PipelineJob:
    job: Optional[PipelineJob] = request.app.state.jobs.get(job_id)
    if job is None:
        raise AppError.not_found(message=f"Không có job '{job_id}'.")
    return job
