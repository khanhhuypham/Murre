"""api.py — FastAPI REST service để retrieve bảng qua HTTP.

Chạy: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Type, TypeVar

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response

from config import cfg
from core.encoder import SGPTEncoder
from methods.factory import build_retriever
from core.llm import LLMGenerator
from methods.runner import PipelineBusyError, run_pipeline
from dataset.loader import load_bird_tables, load_spider_tables
from enums import BaseStrEnum, Dataset, JobStatus, Method, ModelScale
from models.records import ResultRecord
from schemas.evaluate import AvailableRun, EvalResult
from schemas.health import HealthStatus
from schemas.pipeline import PipelineJob, PipelineRunRequest
from schemas.retrieve import RetrieveRequest, TableResult
from utils import logger
from utils.metrics import compute_res
from utils.schema import build_schema_corpus

load_dotenv()


def available_datasets() -> List[Dataset]:
    """Dataset có sẵn tables.json trên đĩa (chưa chắc đã nạp vào RAM)."""
    return [d for d in Dataset if os.path.exists(f"dataset/{d}/tables.json")]


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
    app.state.jobs = {}          # job_id -> PipelineJob
    app.state.job_tasks = {}     # job_id -> asyncio.Task (giữ ref để không bị GC)
    logger.info(f"[API] Sẵn sàng (nạp lười). Dataset có sẵn: {available_datasets()}")
    yield


def _build_dataset(ds_name: Dataset) -> Dict[str, Any]:
    """Nạp encoder + embeddings + retriever cho 1 dataset (chỉ gọi khi cần)."""
    if app.state.encoder is None:
        app.state.encoder = SGPTEncoder()
    encoder: SGPTEncoder = app.state.encoder

    needs_llm: bool = Method(cfg.pipeline.method).needs_llm
    if needs_llm and app.state.llm is None:
        app.state.llm = LLMGenerator()
    llm: Optional[LLMGenerator] = app.state.llm

    # load_spider_tables()/load_bird_tables() KHÔNG nhận tham số — đường dẫn đã
    # hard-code bên trong dataset/loader.py.
    loader = {Dataset.SPIDER: load_spider_tables, Dataset.BIRD: load_bird_tables}[ds_name]
    tables = loader()
    corpus: List[str] = build_schema_corpus(tables=tables)

    # Cache embeddings phải khớp template paths.embeddings_cache trong src/config.py
    # (có cả {scale}), nếu
    # không sẽ nạp vector của scale khác → kết quả sai.
    cache_path: str = cfg.outputs.for_run(dataset=ds_name).embeddings_cache()
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


async def _ensure_dataset(ds_name: Dataset) -> Dict[str, Any]:
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
            TableResult(rank=i + 1, table_schema=r.schema, score=r.score)
            for i, r in enumerate(results[: payload.top_n])
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình retrieve: {str(e)}") from e


# =============================================================================
# /pipeline — CHẠY pipeline cho một tổ hợp (dataset, method), cả 3 method
#
# Scale encoder KHÔNG nằm trong API: server lấy từ general.scale (src/config.py).
# Vì vậy request lẫn response đều không có trường `model` — muốn biết scale nào thì
# đọc `result.result_file`, đường dẫn có sẵn scale trong đó.
#
# /evaluate chỉ ĐỌC metric của lần chạy đã có. Nhóm endpoint này mới là thứ TẠO ra
# lần chạy đó: nó ghi `paths.result` + `paths.score`, nên chạy xong là /evaluate
# tra được ngay với mọi k.
#
# Mặc định POST chờ tới khi chạy xong rồi trả kết quả (200). Lần chạy dài hơn
# timeout của client thì dùng `?wait=false` → trả job ngay (202), poll bằng
# GET /pipeline/jobs/{job_id}. Đo trên máy này, 658 câu mất khoảng:
#   single_hop ~2.7 phút | crush ~32 phút | murre (beam 5 × hop 3) ~1.6 giờ
# =============================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_job(job_id: str, req: PipelineRunRequest) -> None:
    """Thân của một job — chạy trong thread riêng (không chạm event loop)."""
    job: PipelineJob = app.state.jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = _now()

    def on_progress(done: int, total: int) -> None:
        job.processed = done
        job.total = total

    try:
        # Khong truyen scale: giu nguyen general.scale cua server (xem
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
            model=ModelScale(cfg.general.scale),
            method=req.method,
            k=req.k,
        )
        job.status = JobStatus.SUCCEEDED
    except PipelineBusyError as e:
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
        job.finished_at = _now()


@app.post(
    "/pipeline/run",
    response_model=PipelineJob,
    status_code=200,
    summary="Chạy pipeline (murre | single_hop | crush) rồi tính metric tại k",
)
async def pipeline_run(
    payload: PipelineRunRequest,
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
    if any(not j.status.is_final for j in app.state.jobs.values()):
        running: List[str] = [
            j.job_id for j in app.state.jobs.values() if not j.status.is_final
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
    app.state.jobs[job_id] = job

    task: asyncio.Task = asyncio.create_task(asyncio.to_thread(_run_job, job_id, payload))
    app.state.job_tasks[job_id] = task

    if wait:
        await task          # chạy xong mới trả → job.result đã có metric
    else:
        # Chưa chạy xong, chỉ mới nhận việc → 202 Accepted mới đúng ngữ nghĩa.
        response.status_code = 202
    return job


@app.get("/pipeline/jobs", response_model=List[PipelineJob], summary="Danh sách job đã tạo")
async def pipeline_jobs() -> List[PipelineJob]:
    """Job chỉ nằm trong RAM — restart service là mất. Kết quả thì đã ghi ra đĩa,
    tra lại bằng /evaluate hoặc /evaluate/available."""
    return list(app.state.jobs.values())


@app.get(
    "/pipeline/jobs/{job_id}",
    response_model=PipelineJob,
    summary="Trạng thái/tiến độ một lần chạy",
)
async def pipeline_job(job_id: str) -> PipelineJob:
    job: Optional[PipelineJob] = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Không có job '{job_id}'.")
    return job


# =============================================================================
# /evaluate — TÍNH metric THẬT từ kết quả pipeline đã chạy trên máy này
# =============================================================================
_E = TypeVar("_E", bound=BaseStrEnum)


def _parse_enum(enum_cls: Type[_E], value: Any, field: str) -> _E:
    """Đưa giá trị vào enum, sai thì 400 kèm danh sách giá trị hợp lệ.

    Endpoint đã khai báo tham số bằng enum nên FastAPI tự chặn trước (422); helper
    này lo trường hợp gọi trực tiếp từ script/test với chuỗi trần.
    """
    try:
        return enum_cls(value)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{field}={value!r} không hợp lệ. Chỉ nhận: {enum_cls.values()}",
        ) from None


def evaluate_run(
    dataset: Dataset | str,
    model: ModelScale | str,
    method: Method | str,
    k: int,
) -> EvalResult:
    """Đọc kết quả retrieval đã lưu của một lần chạy rồi TÍNH LẠI metric tại k.

    Đây là số THẬT của máy này, không phải số trong paper. Metric được tính tại
    thời điểm gọi (không đọc lại score.json), nên k nào cũng được — miễn là
    k <= số bảng đã lưu cho mỗi câu.
    """
    ds: Dataset = _parse_enum(Dataset, dataset, "dataset")
    mt: Method = _parse_enum(Method, method, "method")
    sc: ModelScale = _parse_enum(ModelScale, model, "model")

    result_file: str = cfg.outputs.for_run(
        dataset=ds, scale=sc, method=mt
    ).result()
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
        data: List[ResultRecord] = ResultRecord.from_list(items=json.load(f))

    if not data:
        raise HTTPException(status_code=409, detail=f"{result_file} rỗng.")

    depth: int = min(len(d.retrieved) for d in data)
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
    dataset: Dataset = Query(default=Dataset.SPIDER, description="spider | bird"),
    model: ModelScale = Query(default=ModelScale.M_125M, description="Scale encoder: 125m | 1.3b | 2.7b | 5.8b (nhận cả 'SGPT-125M')"),
    method: Method = Query(default=Method.MURRE, description="murre | single_hop | crush"),
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
    """Quét outputs/ để biết tổ hợp (dataset, method) nào đã chạy xong.

    Chỉ quét scale hiện tại của server (general.scale), không duyệt cả 4 scale —
    cùng quy ước với /pipeline/run: scale do server quyết định. Scale thực tế vẫn
    đọc được từ `result_file`.
    """
    found: List[AvailableRun] = []
    for ds in Dataset:
        for mt in Method:
            # Không truyền scale → for_run giữ nguyên general.scale của cfg.
            f: str = cfg.outputs.for_run(dataset=ds, method=mt).result()
            if not os.path.exists(f):
                continue
            with open(f, "r", encoding="utf-8") as fh:
                data = ResultRecord.from_list(items=json.load(fh))
            found.append(AvailableRun(
                dataset=ds,
                method=mt,
                num_questions=len(data),
                retrieved_depth=min((len(d.retrieved) for d in data), default=0),
                result_file=f,
            ))
    return found
