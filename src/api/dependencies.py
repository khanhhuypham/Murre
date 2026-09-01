"""api/dependencies.py — Nạp encoder / LLM / embeddings cho từng dataset."""
from __future__ import annotations

import asyncio
import os
import time
from typing import List

from starlette.datastructures import State

from config import cfg
from core.encoder import SGPTEncoder
from core.llm import LLMGenerator
from enums import Dataset, Method
from methods.build import LoadedDataset, build_dataset
from utils import logger


def _build_for_state(state: State, ds_name: Dataset) -> LoadedDataset:
    """Ráp 1 dataset bằng encoder/LLM của server — phần ráp ở methods/build.py.

    Việc riêng của server là VÒNG ĐỜI: encoder/LLM tạo một lần rồi giữ trong
    app.state cho MỌI dataset dùng chung, nên phải tạo ở đây rồi truyền xuống.
    """
    method: Method = Method(cfg.pipeline.method)

    if state.encoder is None:
        state.encoder = SGPTEncoder()
    if method.needs_llm and state.llm is None:
        state.llm = LLMGenerator()

    loaded: LoadedDataset = build_dataset(
        method=method, dataset=ds_name, encoder=state.encoder, llm=state.llm,
    )
    logger.info(f"[API] Đã nạp dataset '{ds_name}' ({len(loaded.corpus)} schemas)")
    return loaded


async def load_dataset_once(state: State, ds_name: Dataset) -> LoadedDataset:
    """Trả về dataset đã nạp, tự nạp nếu chưa có (mỗi dataset chỉ nạp một lần)."""
    if ds_name in state.datasets:
        return state.datasets[ds_name]

    async with state.load_lock:
        if ds_name not in state.datasets:   # kiểm tra lại sau khi giành được lock
            state.datasets[ds_name] = await asyncio.to_thread(_build_for_state, state, ds_name)
    return state.datasets[ds_name]


def available_datasets() -> List[Dataset]:
    """Dataset có sẵn tables.json trên đĩa (chưa chắc đã nạp vào RAM)."""
    return [d for d in Dataset if os.path.exists(f"dataset/{d}/tables.json")]


def datasets_to_preload() -> List[Dataset]:
    """Danh sách dataset phải nạp sẵn lúc khởi động, theo cfg.api.preload."""
    if not cfg.api.preload:
        return []

    available: List[Dataset] = available_datasets()
    if not available:
        raise RuntimeError(
            "api.preload=true nhưng không có dataset nào để nạp: thiếu cả "
            f"{', '.join(f'dataset/{d}/tables.json' for d in Dataset.values())}."
        )
    return available


def verify_llm(llm: LLMGenerator) -> None:
    """Gọi thử LLM một phát để chắc chắn endpoint còn sống."""
    logger.info(f"[API] Kiểm tra LLM '{llm.model_name}' ...")
    llm.generate(prompt="ping")
    logger.info(f"[API] LLM '{llm.model_name}' phản hồi bình thường.")


async def warmup_datasets(state: State) -> None:
    """Nạp sẵn mọi dataset cần thiết trước khi service nhận request."""
    targets: List[Dataset] = datasets_to_preload()
    if targets:
        logger.info(f"[API] Đang nạp sẵn {len(targets)} dataset: {[str(d) for d in targets]} ...")
        for ds in targets:
            started: float = time.perf_counter()
            await load_dataset_once(state, ds)
            logger.info(f"[API] Nạp sẵn '{ds}' xong sau {time.perf_counter() - started:.1f}s")

        if state.llm is not None:
            await asyncio.to_thread(verify_llm, state.llm)

    logger.info(f"[API] Sẵn sàng. Dataset đã nạp: {[str(d) for d in state.datasets]}")
