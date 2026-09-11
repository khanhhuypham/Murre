"""api/dependencies.py — Vòng đời dataset của service: nạp một lần, dùng lại.

Giữ trong app.state một MurreRetriever cho mỗi dataset (retriever đã ngậm sẵn
corpus + embeddings) và MỘT LLMGenerator dùng chung cho mọi dataset.

Encoder KHÔNG nằm ở đây: SentenceEncoder.get() tự dùng lại instance theo tên
profile. Phần ráp cũng không nằm ở đây — nó ở MurreRetriever.for_dataset().

Ngoài vòng đời, file này còn trả lời "service được phép phục vụ dataset nào"
(configured_datasets / available_datasets / require_dataset) và chạy warm-up.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import List

from starlette.datastructures import State

from config import cfg
from core.llm import LLMGenerator
from pipeline.retriever import MurreRetriever
from enums import Dataset
from models.errors import AppError
from utils import logger


def _build_for_state(state: State, ds_name: Dataset) -> MurreRetriever:
    """Ráp 1 dataset bằng LLM của server — phần ráp ở MurreRetriever.for_dataset().

    Việc riêng của server là VÒNG ĐỜI của LLM: tạo một lần rồi giữ trong app.state
    cho MỌI dataset dùng chung, nên phải tạo ở đây rồi truyền xuống.

    Encoder KHÔNG cần giữ ở đây nữa: SentenceEncoder.get() đã dùng lại instance
    theo tên profile, nên for_dataset() tự gọi là đủ — cả ba dataset cùng trỏ
    `multilingual` vẫn chỉ nạp model một lần. Giữ thêm một dict trong app.state
    chỉ là cache thứ hai khoá y hệt cache thứ nhất.
    """
    if state.llm is None:
        state.llm = LLMGenerator()

    retriever: MurreRetriever = MurreRetriever.for_dataset(
        dataset=ds_name,
        llm=state.llm,
    )
    logger.info(
        f"[API] Đã nạp dataset '{ds_name}' ({len(retriever.corpus)} schemas, "
        f"encoder '{cfg.dataset_config(ds_name).encoder}')"
    )
    return retriever


async def load_dataset_once(state: State, ds_name: Dataset) -> MurreRetriever:
    """Trả về dataset đã nạp, tự nạp nếu chưa có (mỗi dataset chỉ nạp một lần)."""
    if ds_name in state.datasets:
        return state.datasets[ds_name]

    async with state.load_lock:
        if ds_name not in state.datasets:   # kiểm tra lại sau khi giành được lock
            state.datasets[ds_name] = await asyncio.to_thread(_build_for_state, state, ds_name)
    return state.datasets[ds_name]


def configured_datasets() -> List[Dataset]:
    """Dataset mà service này được phép phục vụ (api.datasets); rỗng = tất cả."""
    names: List[str] = cfg.api.datasets
    if not names:
        return list(Dataset)

    # Dataset(n) NÉM ValueError khi tra hụt (Enum xử lý _missing_ trả None như vậy),
    # nên phải bắt chứ không kiểm tra `is None` được.
    out: List[Dataset] = []
    unknown: List[str] = []
    for name in names:
        try:
            out.append(Dataset(name))
        except ValueError:
            unknown.append(name)

    if unknown:
        raise ValueError(
            f"api.datasets có giá trị không hợp lệ: {unknown}. Chỉ nhận: {Dataset.values()}"
        )
    return out


def available_datasets() -> List[Dataset]:
    """Dataset service phục vụ VÀ đã có tables.json trên đĩa (chưa chắc đã nạp RAM)."""
    return [d for d in configured_datasets() if os.path.exists(cfg.dataset_config(d).tables)]


def require_dataset(ds_name: Dataset) -> None:
    """Chặn sớm request hỏi dataset mà service này không phục vụ.

    Phân biệt hai lý do: KHÔNG KHAI trong api.datasets, hay khai rồi nhưng THIẾU
    FILE. Gộp làm một thì người vận hành đi tìm file đang nằm sẵn trên đĩa.
    """
    available: List[Dataset] = available_datasets()
    if ds_name in available:
        return

    have: str = ", ".join(str(d) for d in available) or "không có dataset nào"
    if ds_name not in configured_datasets():
        raise AppError.not_found(
            message=(
                f"Service này không phục vụ dataset '{ds_name}' (api.datasets trong "
                f"config.yaml). Đang phục vụ: {have}."
            )
        )
    raise AppError.not_found(
        message=(
            f"Dataset '{ds_name}' có khai trong api.datasets nhưng thiếu file "
            f"{cfg.dataset_config(ds_name).tables}. Đang phục vụ: {have}."
        )
    )


def datasets_to_preload() -> List[Dataset]:
    """Danh sách dataset phải nạp sẵn lúc khởi động, theo cfg.api.preload."""
    if not cfg.api.preload:
        return []

    available: List[Dataset] = available_datasets()
    if not available:
        raise RuntimeError(
            "api.preload=true nhưng không có dataset nào để nạp: thiếu cả "
            f"{', '.join(cfg.dataset_config(d).tables for d in configured_datasets())}."
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

    state.ready = True
    logger.info(f"[API] Sẵn sàng. Dataset đã nạp: {[str(d) for d in state.datasets]}")
