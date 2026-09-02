"""api/evaluator.py — TÍNH metric THẬT từ kết quả pipeline đã chạy trên máy này.

Tách khỏi router để dùng được trực tiếp trong script Python (không cần chạy API)
và để jobs.py đọc lại metric bằng đúng đường code của /evaluate.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Type, TypeVar

from fastapi import HTTPException

from config import cfg, model_slug
from enums import BaseStrEnum, Dataset, Method
from models.metrics import MetricScores
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


# Chuỗi giả để dò vị trí {model} trong template đường dẫn.
_PROBE: str = "__model__"


def models_on_disk(dataset: Dataset, method: Method) -> List[str]:
    """Các nhãn model đã thực sự có thư mục trong outputs/ — dùng cho thông báo lỗi.

    Không có danh sách hợp lệ khai ở đâu cả: nhãn suy ra từ encoder.model_name của
    lần chạy, nên thứ đáng nói khi tra hụt là "trên đĩa đang có những nhãn nào".
    """
    # Dựng đường dẫn với nhãn giả rồi cắt lấy phần đứng trước → thư mục cha chứa
    # các nhãn. Bám theo template nên đổi paths.result cũng không hỏng.
    probe: str = cfg.outputs.for_run(dataset=dataset, model=_PROBE, method=method).result()
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
    # Chuẩn hoá qua đúng hàm sinh nhãn, nên gõ cả tên HuggingFace đầy đủ
    # ("Muennighoff/SGPT-125M-...") hay gõ sẵn nhãn đều ra cùng một thư mục.
    # Sai tên thì rơi vào nhánh 404 bên dưới, ở đó in ra các nhãn có thật trên đĩa.
    sc: str = model_slug(name=str(model))

    result_file: str = cfg.outputs.for_run(
        dataset=ds, model=sc, method=mt
    ).result()
    if not os.path.exists(result_file):
        cmd: str = (
            f"POST /pipeline/run với dataset={ds}, method={mt} "
            f"(hoặc `python -m main` với run_option.mode=batch)"
        )
        have: List[str] = models_on_disk(dataset=ds, method=mt)
        raise HTTPException(
            status_code=404,
            detail=(
                f"Chưa có kết quả cho dataset={ds}, model={sc}, method={mt} "
                f"(không tìm thấy {result_file}). "
                f"Model đã có trên đĩa: {have or 'chưa có model nào'}. "
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
        #
        # Cách nới `depth` KHÁC NHAU theo method: single_hop/crush cắt danh sách ở
        # pipeline.top_k_pool, còn murre thì theo §3.5 của paper chỉ xếp hạng bảng
        # nằm trên đường đi (tối đa B + (H-1)·B²) nên top_k_pool không có tác dụng.
        knob: str = (
            "pipeline.beam_size (và/hoặc pipeline.max_hop)"
            if mt is Method.MURRE
            else "pipeline.top_k_pool"
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"k={k} lớn hơn số bảng đã lưu mỗi câu ({depth}). Tăng "
                f"{knob} rồi chạy lại, hoặc dùng k <= {depth}."
            ),
        )

    metrics: MetricScores = compute_res(top_k=[k], data=data)
    return EvalResult(
        dataset=ds,
        method=mt,
        k=k,
        recall=round(metrics.recall_at(k) * 100, 2),
        complete_recall=round(metrics.complete_recall_at(k) * 100, 2),
        num_questions=len(data),
        retrieved_depth=depth,
        result_file=result_file,
    )
