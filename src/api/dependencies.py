"""api/dependencies.py — Nạp encoder / LLM / embeddings cho từng dataset.

Mặc định nạp sẵn toàn bộ lúc khởi động (warmup_datasets, gọi từ lifespan) và ném
lỗi nếu thiếu thứ gì; ensure_dataset là đường lùi nạp lười khi api.preload=false.

Toàn bộ trạng thái dùng chung của service nằm trong `app.state` (khởi tạo ở
api/app.py::lifespan). Các hàm ở đây nhận thẳng `state` thay vì import `app`, để
router không phải import ngược lại app → tránh vòng import.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import List

import torch
from starlette.datastructures import State

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from dataset.loader import load_bird_tables, load_spider_tables
from enums import Dataset, Method
from methods.factory import RetrieverType, build_retriever
from utils import logger
from utils.schema import build_schema_corpus


@dataclass
class LoadedDataset:
    """Một dataset đã sẵn sàng phục vụ /retrieve.

    Ba thứ này luôn đi cùng nhau và phải khớp nhau: `embs[i]` là vector của
    `corpus[i]` (điều kiện được canh ở cuối build_dataset). Gói thành một object
    thay vì dict để không lỡ tay ghép embeddings dataset này với corpus dataset
    kia, và để chỗ dùng viết `ds.corpus` — gõ sai tên là IDE bắt được ngay.
    """

    retriever: RetrieverType
    corpus: List[str]
    embs: torch.Tensor


def available_datasets() -> List[Dataset]:
    """Dataset có sẵn tables.json trên đĩa (chưa chắc đã nạp vào RAM)."""
    return [d for d in Dataset if os.path.exists(f"dataset/{d}/tables.json")]


def build_dataset(state: State, ds_name: Dataset) -> LoadedDataset:
    """Nạp encoder + embeddings + retriever cho 1 dataset (chỉ gọi khi cần).

    encoder/llm dùng chung cho mọi dataset nên nằm ở state, nạp một lần rồi mọi
    dataset xài lại — chỉ corpus/embeddings/retriever là của riêng ds_name.
    """
    if state.encoder is None:
        state.encoder = SGPTEncoder()

    needs_llm: bool = Method(cfg.pipeline.method).needs_llm
    if needs_llm and state.llm is None:
        state.llm = LLMGenerator()

    loader = {Dataset.SPIDER: load_spider_tables, Dataset.BIRD: load_bird_tables}[ds_name]
    tables = loader()
    corpus: List[str] = build_schema_corpus(tables=tables)

    # Cache embeddings phải khớp template paths.embeddings_cache trong src/config.py
    # (có cả {model}), nếu không sẽ nạp vector của model khác → kết quả sai.
    cache_path: str = cfg.outputs.for_run(dataset=ds_name).embeddings_cache()
    embs: torch.Tensor
    if os.path.exists(cache_path):
        embs = torch.load(cache_path, weights_only=True)
    else:
        logger.info(f"[API] Chưa có cache, đang encode {len(corpus)} schemas của '{ds_name}' ...")
        embs = state.encoder.encode(texts=corpus, is_query=False)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(obj=embs, f=cache_path)

    # Cache cũ của model/corpus khác vẫn nạp được nhưng số vector sẽ lệch với số
    # schema → điểm số gán nhầm bảng. Bắt tại đây thay vì trả kết quả sai âm thầm.
    if embs.shape[0] != len(corpus):
        raise RuntimeError(
            f"Cache embeddings của '{ds_name}' không khớp corpus: {embs.shape[0]} vector "
            f"cho {len(corpus)} schema.\n"
            f"  Cache: {cache_path}\n"
            f"  Cách xử lý: xoá file cache đó rồi khởi động lại để encode lại."
        )

    retriever = build_retriever(encoder=state.encoder, llm=state.llm)
    logger.info(f"[API] Đã nạp dataset '{ds_name}' ({len(corpus)} schemas)")
    return LoadedDataset(retriever=retriever, corpus=corpus, embs=embs)


async def ensure_dataset(state: State, ds_name: Dataset) -> LoadedDataset:
    """Trả về dataset đã nạp, tự nạp nếu chưa có.

    Đường đi chính là warmup_datasets() lúc khởi động; hàm này giữ vai trò lưới an
    toàn cho các trường hợp còn lại (api.preload=false, dataset không nằm trong
    api.preload_datasets, hoặc lần nạp sẵn bị lỗi). Lock để 2 request đồng thời
    không cùng encode một corpus hai lần.
    """
    if ds_name in state.datasets:
        return state.datasets[ds_name]

    async with state.load_lock:
        if ds_name not in state.datasets:   # kiểm tra lại sau khi giành được lock
            state.datasets[ds_name] = await asyncio.to_thread(build_dataset, state, ds_name)
    return state.datasets[ds_name]


def datasets_to_preload() -> List[Dataset]:
    """Danh sách dataset BẮT BUỘC phải nạp xong lúc khởi động, theo cfg.api.

    `api.preload_datasets` rỗng = mọi dataset có tables.json. Tên sai hoặc thiếu
    tables.json thì ném lỗi luôn: đã yêu cầu nạp sẵn thì không được im lặng bỏ qua
    để rồi request đầu tiên mới báo lỗi.
    """
    if not cfg.api.preload:
        return []

    available: List[Dataset] = available_datasets()
    wanted: List[str] = list(cfg.api.preload_datasets)

    if not wanted:
        if not available:
            raise RuntimeError(
                "api.preload=true nhưng không có dataset nào để nạp: thiếu cả "
                f"{', '.join(f'dataset/{d}/tables.json' for d in Dataset.values())}."
            )
        return available

    selected: List[Dataset] = []
    for name in wanted:
        try:
            ds = Dataset(name)
        except ValueError as exc:
            raise RuntimeError(
                f"api.preload_datasets có '{name}' không hợp lệ. Hợp lệ: {Dataset.values()}"
            ) from exc
        if ds not in available:
            raise RuntimeError(
                f"api.preload_datasets có '{ds}' nhưng không tìm thấy dataset/{ds}/tables.json."
            )
        selected.append(ds)
    return selected


def verify_llm(llm: LLMGenerator) -> None:
    """Gọi thử LLM một phát cho chắc chắn endpoint sống.

    `LLMGenerator.__init__` chỉ ping TCP với endpoint local; api_key sai, model
    chưa pull hay endpoint remote chết thì vẫn phải sinh thật mới lộ. Ping ở đây
    để lỗi nổ lúc khởi động. Lỗi được `generate()` ném kèm hướng dẫn sẵn.
    """
    logger.info(f"[API] Kiểm tra LLM '{llm.model_name}' ...")
    llm.generate(prompt="ping")
    logger.info(f"[API] LLM '{llm.model_name}' phản hồi bình thường.")


async def warmup_datasets(state: State) -> None:
    """Nạp xong MỌI thứ cần thiết trước khi service nhận request (gọi trong lifespan).

    Chạy tuần tự để 2 dataset không cùng encode và tranh nhau RAM/GPU.

    Bất kỳ lỗi nào ở đây đều ném ra ngoài → uvicorn huỷ startup, service KHÔNG mở
    cổng. Đó là chủ đích: một service đã trả lời /health mà /retrieve vẫn hỏng thì
    tệ hơn hẳn một service không lên, vì lỗi chỉ lộ ra khi có người dùng thật.
    """
    targets: List[Dataset] = datasets_to_preload()
    if not targets:
        logger.warning(
            "[API] api.preload=false → nạp lười ở request đầu tiên. "
            "Service lên sớm nhưng KHÔNG bảo đảm /retrieve chạy được."
        )
        return

    logger.info(f"[API] Đang nạp sẵn {len(targets)} dataset: {[str(d) for d in targets]} ...")
    for ds in targets:
        started: float = time.perf_counter()
        await ensure_dataset(state, ds)
        logger.info(f"[API] Nạp sẵn '{ds}' xong sau {time.perf_counter() - started:.1f}s")

    if state.llm is not None:
        await asyncio.to_thread(verify_llm, state.llm)
