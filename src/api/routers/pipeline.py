"""api/routers/pipeline.py — CHẠY pipeline cho một tổ hợp (dataset, method), cả 3 method.

Encoder KHÔNG nằm trong API: server lấy từ encoder.model_name (config.yaml).
Vì vậy request lẫn response đều không có trường `model` — muốn biết model nào thì
đọc `result.result_file`, đường dẫn có sẵn nhãn model trong đó.

/evaluate chỉ ĐỌC metric của lần chạy đã có. Nhóm endpoint này mới là thứ TẠO ra
lần chạy đó: nó ghi `paths.result` + `paths.score`, nên chạy xong là /evaluate
tra được ngay với mọi k.

Mặc định POST chờ tới khi chạy xong rồi trả kết quả (200). Lần chạy dài hơn
timeout của client thì dùng `?wait=false` → trả job ngay (202), poll bằng
GET /pipeline/jobs/{job_id}. Đo trên máy này, 658 câu mất khoảng:
  single_hop ~2.7 phút | crush ~32 phút | murre (beam 5 × hop 3) ~1.6 giờ
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response

from api.jobs import run_job
from enums import JobStatus
from schemas.pipeline import PipelineJob, PipelineRunRequest

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post(
    "/run",
    response_model=PipelineJob,
    status_code=200,
    summary="Chạy pipeline (murre | single_hop | crush) rồi tính metric tại k",
)
async def pipeline_run(
    payload: PipelineRunRequest,
    request: Request,
    response: Response,
    wait: Annotated[bool, Query(
        description="true (mặc định) = chờ chạy xong rồi trả kết quả luôn → 200. "
                    "false = trả job ngay để poll /pipeline/jobs/{job_id} → 202; "
                    "dùng cho lần chạy dài (murre đủ 658 câu mất khoảng 1.6 giờ).",
    )] = True,
) -> PipelineJob:
    """Chạy pipeline rồi trả kết quả.

    Mặc định `wait=true`: request giữ mở tới khi chạy xong, response đã có sẵn
    `result` (metric tại k) — gọi một phát là có luôn.

    Khi nào cần `wait=false`: lần chạy dài hơn timeout của client/proxy. Đo trên
    máy này (Ollama + qwen2.5:0.5b), chạy đủ 658 câu mất khoảng:
        single_hop ~2.7 phút | crush ~32 phút | murre (beam 5 × hop 3) ~1.6 giờ
    Muốn chạy nhanh để thử thì đặt `limit` nhỏ và cứ để mặc định.

    `cfg` là biến toàn cục của process nên mỗi lúc chỉ chạy được MỘT job; gọi khi
    đang có job khác sẽ nhận `409`.
    """
    state = request.app.state
    if any(not j.status.is_final for j in state.jobs.values()):
        running: List[str] = [
            j.job_id for j in state.jobs.values() if not j.status.is_final
        ]
        raise HTTPException(
            status_code=409,
            detail=f"Đang có job chạy: {running}. Đợi xong rồi thử lại.",
        )

    job_id: str = uuid.uuid4().hex[:12]
    job = PipelineJob(
        job_id=job_id,
        status=JobStatus.QUEUED,
        dataset=payload.dataset,
        method=payload.method,
        k=payload.k,
        limit=payload.limit,
    )
    state.jobs[job_id] = job

    task: asyncio.Task = asyncio.create_task(
        asyncio.to_thread(run_job, state, job_id, payload)
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
        raise HTTPException(status_code=404, detail=f"Không có job '{job_id}'.")
    return job
