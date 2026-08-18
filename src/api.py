"""api.py — FastAPI REST service để retrieve bảng qua HTTP.

Chạy: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import cfg
from core.encoder import SGPTEncoder
from core.factory import build_retriever
from core.llm import LLMGenerator
from dataset.loader import load_bird_tables, load_spider_tables
from utils import logger
from utils.schema import build_schema_corpus

load_dotenv()


class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi tự nhiên cần tìm bảng liên quan")
    dataset: str = Field(default="spider", description="Dataset: 'spider' hoặc 'bird'")
    top_n: int = Field(default=5, ge=1, le=20, description="Số lượng bảng trả về (1-20)")


class TableResult(BaseModel):
    rank: int = Field(..., description="Thứ hạng (bắt đầu từ 1)")
    table_schema: str = Field(..., description="Chuỗi schema bảng: db.table(col1, col2, ...)")
    score: float = Field(..., description="Điểm số tổng hợp (log-space) từ pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    encoder = SGPTEncoder()
    llm: Optional[LLMGenerator] = LLMGenerator() if cfg.pipeline.method.lower() in ("murre", "crush") else None

    datasets: Dict[str, Dict[str, Any]] = {}

    for ds_name in ("spider", "bird"):
        table_path: str = f"dataset/{ds_name}/tables.json"
        if not os.path.exists(table_path):
            continue

        loader = load_spider_tables if ds_name == "spider" else load_bird_tables
        tables = loader(path=table_path)
        corpus: List[str] = build_schema_corpus(tables=tables)

        cache_path: str = f"outputs/{ds_name}_embeddings.pt"
        embs: torch.Tensor
        if os.path.exists(cache_path):
            embs = torch.load(cache_path, weights_only=True)
        else:
            embs = encoder.encode(texts=corpus, is_query=False)
            os.makedirs("outputs", exist_ok=True)
            torch.save(obj=embs, f=cache_path)

        retriever = build_retriever(encoder=encoder, llm=llm)
        datasets[ds_name] = {"retriever": retriever, "corpus": corpus, "embs": embs}
        logger.info(f"[API] Đã nạp dataset '{ds_name}' ({len(corpus)} schemas)")

    app.state.datasets = datasets
    yield


app = FastAPI(
    title="MURRE — Multi-Hop Table Retrieval API",
    description=(
        "API cho hệ thống MURRE: retrieve bảng SQL liên quan từ câu hỏi tự nhiên.\n\n"
        "Dựa trên: MURRE: Multi-Hop Table Retrieval with Removal for Open-Domain "
        "Text-to-SQL (COLING 2025)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Kiểm tra trạng thái service")
async def health() -> Dict[str, Any]:
    datasets_loaded: List[str] = list(app.state.datasets.keys()) if hasattr(app.state, "datasets") else []
    return {
        "status": "ok",
        "method": cfg.pipeline.method,
        "datasets_loaded": datasets_loaded,
        "beam_size": cfg.pipeline.beam_size,
        "max_hop": cfg.pipeline.max_hop,
    }


@app.get("/config", summary="Xem cấu hình hiện tại")
async def get_config() -> Dict[str, Any]:
    return cfg.to_dict()


@app.post("/retrieve", response_model=List[TableResult], summary="Retrieve bảng cho câu hỏi")
async def retrieve_tables(payload: RetrieveRequest) -> List[TableResult]:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    datasets = app.state.datasets
    if payload.dataset not in datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{payload.dataset}' chưa được nạp. Có sẵn: {list(datasets.keys())}",
        )

    try:
        ds = datasets[payload.dataset]
        results = ds["retriever"].run(
            question=payload.question, corpus=ds["corpus"], schema_embeddings=ds["embs"],
        )
        return [
            TableResult(rank=i + 1, table_schema=r["schema"], score=r["score"])
            for i, r in enumerate(results[: payload.top_n])
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình retrieve: {str(e)}") from e
