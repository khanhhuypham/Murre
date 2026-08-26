"""api.py — FastAPI REST service để retrieve bảng qua HTTP.

Chạy: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from config import cfg, get_path
from core.encoder import SGPTEncoder
from core.factory import build_retriever
from core.llm import LLMGenerator
from dataset.loader import load_bird_tables, load_spider_tables
from schemas.evaluate import AvailableRun, EvalResult
from schemas.health import HealthStatus
from schemas.retrieve import RetrieveRequest, TableResult
from utils import logger
from utils.metrics import compute_res
from utils.schema import build_schema_corpus

load_dotenv()


def available_datasets() -> List[str]:
    """Dataset có sẵn tables.json trên đĩa (chưa chắc đã nạp vào RAM)."""
    return [d for d in ("spider", "bird") if os.path.exists(f"dataset/{d}/tables.json")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # NẠP LƯỜI (lazy): không encode corpus nào lúc khởi động.
    #
    # Encode corpus BirdUnion trên CPU mất nhiều phút, mà các endpoint tra cứu số
    # liệu (/evaluate) không cần model nào cả. Nạp sẵn ở đây sẽ chặn cả service
    # cho tới khi encode xong. Vì vậy encoder/LLM/embeddings chỉ được tạo ở lần gọi
    # /retrieve đầu tiên của từng dataset — đúng lúc thực sự cần.
    app.state.encoder = None
    app.state.llm = None
    app.state.datasets = {}
    app.state.load_lock = asyncio.Lock()
    logger.info(f"[API] Sẵn sàng (nạp lười). Dataset có sẵn: {available_datasets()}")
    yield


def _build_dataset(ds_name: str) -> Dict[str, Any]:
    """Nạp encoder + embeddings + retriever cho 1 dataset (chỉ gọi khi cần)."""
    if app.state.encoder is None:
        app.state.encoder = SGPTEncoder()
    encoder: SGPTEncoder = app.state.encoder

    needs_llm: bool = cfg.pipeline.method.lower() in ("murre", "crush")
    if needs_llm and app.state.llm is None:
        app.state.llm = LLMGenerator()
    llm: Optional[LLMGenerator] = app.state.llm

    # load_spider_tables()/load_bird_tables() KHÔNG nhận tham số — đường dẫn đã
    # hard-code bên trong dataset/loader.py.
    loader = load_spider_tables if ds_name == "spider" else load_bird_tables
    tables = loader()
    corpus: List[str] = build_schema_corpus(tables=tables)

    # Cache embeddings phải khớp template trong config.yaml (có cả {scale}), nếu
    # không sẽ nạp vector của scale khác → kết quả sai.
    cache_path: str = get_path(key="embeddings_cache", dataset=ds_name)
    embs: torch.Tensor
    if os.path.exists(cache_path):
        embs = torch.load(cache_path, weights_only=True)
    else:
        logger.info(f"[API] Chưa có cache, đang encode {len(corpus)} schemas của '{ds_name}' ...")
        embs = encoder.encode(texts=corpus, is_query=False)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(obj=embs, f=cache_path)

    retriever = build_retriever(encoder=encoder, llm=llm)
    logger.info(f"[API] Đã nạp dataset '{ds_name}' ({len(corpus)} schemas)")
    return {"retriever": retriever, "corpus": corpus, "embs": embs}


async def _ensure_dataset(ds_name: str) -> Dict[str, Any]:
    """Trả về dataset đã nạp, tự nạp ở lần gọi đầu. Lock để 2 request đồng thời
    không cùng encode một corpus hai lần."""
    if ds_name in app.state.datasets:
        return app.state.datasets[ds_name]

    async with app.state.load_lock:
        if ds_name not in app.state.datasets:   # kiểm tra lại sau khi giành được lock
            app.state.datasets[ds_name] = await asyncio.to_thread(_build_dataset, ds_name)
    return app.state.datasets[ds_name]


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


@app.get("/health", response_model=HealthStatus, summary="Kiểm tra trạng thái service")
async def health() -> HealthStatus:
    loaded: List[str] = list(app.state.datasets.keys()) if hasattr(app.state, "datasets") else []
    return HealthStatus(
        status="ok",
        method=cfg.pipeline.method,
        datasets_available=available_datasets(),   # có tables.json trên đĩa
        datasets_loaded=loaded,                    # đã nạp embeddings vào RAM
        beam_size=cfg.pipeline.beam_size,
        max_hop=cfg.pipeline.max_hop,
    )


@app.get("/config", summary="Xem cấu hình hiện tại")
async def get_config() -> Dict[str, Any]:
    return cfg.to_dict()


@app.post("/retrieve", response_model=List[TableResult], summary="Retrieve bảng cho câu hỏi")
async def retrieve_tables(payload: RetrieveRequest) -> List[TableResult]:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    if payload.dataset not in available_datasets():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{payload.dataset}' không có dataset/{payload.dataset}/tables.json. "
                   f"Có sẵn: {available_datasets()}",
        )

    try:
        # Lần gọi đầu của mỗi dataset sẽ mất thêm thời gian nạp model + embeddings.
        ds = await _ensure_dataset(payload.dataset)
        results = ds["retriever"].run(
            question=payload.question, corpus=ds["corpus"], schema_embeddings=ds["embs"],
        )
        return [
            TableResult(rank=i + 1, table_schema=r["schema"], score=r["score"])
            for i, r in enumerate(results[: payload.top_n])
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình retrieve: {str(e)}") from e


# =============================================================================
# /evaluate — TÍNH metric THẬT từ kết quả pipeline đã chạy trên máy này
# =============================================================================
_DATASET_CHOICES: List[str] = ["spider", "bird"]
_METHOD_CHOICES: List[str] = ["murre", "single_hop", "crush"]
_MODEL_ALIASES: Dict[str, str] = {
    # tên gọi thân thiện → giá trị general.scale trong config.yaml
    "sgpt-125m": "125m", "125m": "125m",
    "sgpt-1.3b": "1.3b", "1.3b": "1.3b",
    "sgpt-2.7b": "2.7b", "2.7b": "2.7b",
    "sgpt-5.8b": "5.8b", "5.8b": "5.8b",
}

def _choice(value: str, choices: List[str], field: str) -> str:
    v: str = value.strip().lower()
    if v not in choices:
        raise HTTPException(
            status_code=400, detail=f"{field}={value!r} không hợp lệ. Chỉ nhận: {choices}"
        )
    return v


def _resolve_model(value: str) -> str:
    v: str = value.strip().lower()
    return _MODEL_ALIASES.get(v, v)   # scale lạ vẫn cho qua, lỗi sẽ là 404 không có file


def evaluate_run(
    dataset: str,
    model: str,
    method: str,
    k: int,
) -> EvalResult:
    """Đọc kết quả retrieval đã lưu của một lần chạy rồi TÍNH LẠI metric tại k.

    Đây là số THẬT của máy này, không phải số trong paper. Metric được tính tại
    thời điểm gọi (không đọc lại score.json), nên k nào cũng được — miễn là
    k <= số bảng đã lưu cho mỗi câu.
    """
    ds: str = _choice(dataset, _DATASET_CHOICES, "dataset")
    mt: str = _choice(method, _METHOD_CHOICES, "method")
    sc: str = _resolve_model(model)

    result_file: str = get_path(
        key="result", dataset=ds, scale=sc, method=mt, max_hop=cfg.pipeline.max_hop
    )
    if not os.path.exists(result_file):
        cmd: str = "python -m steps.score   (sau khi chạy đủ chuỗi retrieve/rewrite)"
        raise HTTPException(
            status_code=404,
            detail=(
                f"Chưa có kết quả cho dataset={ds}, model={sc}, method={mt} "
                f"(không tìm thấy {result_file}). Chạy trước: {cmd}"
            ),
        )

    with open(result_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        raise HTTPException(status_code=409, detail=f"{result_file} rỗng.")

    depth: int = min(len(d.get("retrieved", [])) for d in data)
    if k > depth:
        # compute_recall_at_k() bỏ qua k > số bảng đã lưu → metric sẽ ra 0.0 mà
        # không báo gì. Chặn ở đây để không trả về số sai.
        raise HTTPException(
            status_code=400,
            detail=(
                f"k={k} lớn hơn số bảng đã lưu mỗi câu ({depth}). Tăng "
                f"pipeline.top_k_pool rồi chạy lại, hoặc dùng k <= {depth}."
            ),
        )

    metrics: Dict[str, Dict[int, float]] = compute_res(top_k=[k], data=data)
    return EvalResult(
        dataset=ds,
        model=sc,
        method=mt,
        k=k,
        recall=round(metrics["recall"][k] * 100, 2),
        complete_recall=round(metrics["complete_recall"][k] * 100, 2),
        num_questions=len(data),
        retrieved_depth=depth,
        result_file=result_file,
    )


@app.get(
    "/evaluate",
    response_model=List[EvalResult],
    summary="Tính recall@k / complete_recall@k THẬT từ kết quả đã chạy",
)
async def evaluate(
    dataset: str = Query(default="spider", description="spider | bird"),
    model: str = Query(default="125m", description="Scale encoder: 125m | 1.3b | 2.7b | 5.8b (nhận cả 'SGPT-125M')"),
    method: str = Query(default="murre", description="murre | single_hop | crush"),
    k: List[int] = Query(default=[5], description="Một hoặc nhiều k, ví dụ ?k=3&k=5&k=10"),
) -> List[EvalResult]:
    """Trả về metric THẬT tính từ file kết quả của lần chạy tương ứng trên máy này.

    Không có kết quả cho tổ hợp đó → `404` kèm lệnh cần chạy trước.
    """
    return [
        await asyncio.to_thread(evaluate_run, dataset, model, method, kk)
        for kk in k
    ]


@app.get("/evaluate/available", response_model=List[AvailableRun], summary="Các lần chạy đã có kết quả trên máy này")
async def evaluate_available() -> List[AvailableRun]:
    """Quét outputs/ để biết tổ hợp (dataset, model, method) nào đã chạy xong."""
    found: List[AvailableRun] = []
    for ds in _DATASET_CHOICES:
        for mt in _METHOD_CHOICES:
            for sc in sorted(set(_MODEL_ALIASES.values())):
                f: str = get_path(
                    key="result", dataset=ds, scale=sc, method=mt, max_hop=cfg.pipeline.max_hop
                )
                if not os.path.exists(f):
                    continue
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                found.append(AvailableRun(
                    dataset=ds,
                    model=sc,
                    method=mt,
                    num_questions=len(data),
                    retrieved_depth=min((len(d.get("retrieved", [])) for d in data), default=0),
                    result_file=f,
                ))
    return found
