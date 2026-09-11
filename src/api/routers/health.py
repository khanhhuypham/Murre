"""api/routers/health.py — /health và /config: xem trạng thái service, không nạp gì."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Request, Response

from api.dependencies import available_datasets
from config import cfg
from schemas.health import HealthStatus

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus, summary="Kiểm tra trạng thái service")
async def health(request: Request, response: Response) -> HealthStatus:
    """200 khi service đã sẵn sàng nhận request, 503 khi còn đang warm-up.

    Dùng luôn làm readiness probe của container orchestrator: với api.preload=true,
    nạp encoder + embeddings mất hàng chục giây và /retrieve chưa dùng được.
    """
    state = request.app.state
    if not state.ready:
        response.status_code = 503

    return HealthStatus(
        status="ok" if state.ready else "starting",
        ready=state.ready,
        datasets_available=available_datasets(),   # có tables.json trên đĩa
        datasets_loaded=list(state.datasets),      # đã nạp embeddings vào RAM
        encoders={
            str(d): cfg.encoder_for(dataset=d).model_name
            for d in available_datasets()
        },
        llm=cfg.llm.profiles[cfg.llm.active_profile].model_name,
        beam_size=cfg.pipeline.beam_size,
        max_hop=cfg.pipeline.max_hop,
    )


@router.get("/config", summary="Xem cấu hình hiện tại (api_key đã che)")
async def get_config() -> Dict[str, Any]:
    return cfg.to_dict()
