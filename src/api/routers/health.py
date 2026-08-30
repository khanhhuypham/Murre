"""api/routers/health.py — /health và /config: xem trạng thái service, không nạp gì."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Request

from api.dependencies import available_datasets
from config import cfg
from schemas.health import HealthStatus

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus, summary="Kiểm tra trạng thái service")
async def health(request: Request) -> HealthStatus:
    state = request.app.state
    loaded: List[str] = [str(d) for d in state.datasets]
    return HealthStatus(
        status="ok",
        method=cfg.pipeline.method,
        datasets_available=available_datasets(),   # có tables.json trên đĩa
        datasets_loaded=loaded,                    # đã nạp embeddings vào RAM
        beam_size=cfg.pipeline.beam_size,
        max_hop=cfg.pipeline.max_hop,
    )


@router.get("/config", summary="Xem cấu hình hiện tại")
async def get_config() -> Dict[str, Any]:
    return cfg.to_dict()
