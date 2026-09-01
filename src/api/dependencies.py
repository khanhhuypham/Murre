"""api/dependencies.py — Nạp encoder / LLM / embeddings cho từng dataset."""
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
    """Một dataset đã sẵn sàng phục vụ /retrieve: retriever + corpus + embeddings."""

    retriever: RetrieverType
    corpus: List[str]
    embs: torch.Tensor


def build_dataset(state: State, ds_name: Dataset) -> LoadedDataset:
    """Nạp encoder + embeddings + retriever cho 1 dataset."""
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


async def load_dataset_once(state: State, ds_name: Dataset) -> LoadedDataset:
    """Trả về dataset đã nạp, tự nạp nếu chưa có (mỗi dataset chỉ nạp một lần)."""
    if ds_name in state.datasets:
        return state.datasets[ds_name]

    async with state.load_lock:
        if ds_name not in state.datasets:   # kiểm tra lại sau khi giành được lock
            state.datasets[ds_name] = await asyncio.to_thread(build_dataset, state, ds_name)
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
