"""api/evaluator.py — TÍNH metric THẬT từ kết quả pipeline đã chạy trên máy này.

Tách khỏi router để dùng được trực tiếp trong script Python (không cần chạy API)
và để jobs.py đọc lại metric bằng đúng đường code của /evaluate.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Type, TypeVar

from fastapi import HTTPException

from config import cfg
from enums import BaseStrEnum, Dataset, Method
from models.records import ResultRecord
from schemas.evaluate import EvalResult
from utils.metrics import compute_res

_E = TypeVar("_E", bound=BaseStrEnum)


def parse_enum(enum_cls: Type[_E], value: Any, field: str) -> _E:
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


# Chuỗi giả để dò vị trí {scale} trong template đường dẫn.
_PROBE: str = "__scale__"


def scales_on_disk(dataset: Dataset, method: Method) -> List[str]:
    """Các scale đã thực sự có thư mục trong outputs/ — dùng cho thông báo lỗi.

    Không có danh sách scale hợp lệ khai ở đâu cả: scale chỉ là nhãn thư mục, nên
    thứ đáng nói khi tra hụt là "trên đĩa đang có những scale nào".
    """
    # Dựng đường dẫn với scale giả rồi cắt lấy phần đứng trước → thư mục cha chứa
    # các scale. Bám theo template nên đổi paths.result cũng không hỏng.
    probe: str = cfg.outputs.for_run(dataset=dataset, scale=_PROBE, method=method).result()
    parent: str = probe.split(_PROBE)[0]
    if not os.path.isdir(parent):
        return []
    return sorted(d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)))


def evaluate_run(
    dataset: Dataset | str,
    model: str,
    method: Method | str,
    k: int,
) -> EvalResult:
    """Đọc kết quả retrieval đã lưu của một lần chạy rồi TÍNH LẠI metric tại k.

    Đây là số THẬT của máy này, không phải số trong paper. Metric được tính tại
    thời điểm gọi (không đọc lại score.json), nên k nào cũng được — miễn là
    k <= số bảng đã lưu cho mỗi câu.
    """
    ds: Dataset = parse_enum(Dataset, dataset, "dataset")
    mt: Method = parse_enum(Method, method, "method")
    # scale là nhãn thư mục tự do — chỉ chuẩn hoá cho dễ gõ ('SGPT-125M' → '125m'),
    # không kiểm tra theo danh sách nào. Sai tên thì rơi vào nhánh 404 bên dưới,
    # ở đó có in ra các scale đang có thật trên đĩa.
    sc: str = str(model).strip().lower().removeprefix("sgpt-")

    result_file: str = cfg.outputs.for_run(
        dataset=ds, scale=sc, method=mt
    ).result()
    if not os.path.exists(result_file):
        cmd: str = "python -m steps.score   (sau khi chạy đủ chuỗi retrieve/rewrite)"
        have: List[str] = scales_on_disk(dataset=ds, method=mt)
        raise HTTPException(
            status_code=404,
            detail=(
                f"Chưa có kết quả cho dataset={ds}, model={sc}, method={mt} "
                f"(không tìm thấy {result_file}). "
                f"Scale đã có trên đĩa: {have or 'chưa có scale nào'}. "
                f"Chạy trước: {cmd}"
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
