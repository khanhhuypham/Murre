"""DTO của API: sai tên field hay sai giá trị phải BÁO LỖI, không im lặng.

Gõ nhầm một field rồi bị bỏ qua âm thầm là kiểu lỗi tốn thời gian nhất: response
vẫn 200, giá trị vẫn là mặc định, không có dấu hiệu nào cho thấy server không nhận
thứ mình gửi.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import cfg
from enums import Dataset
from schemas.common import QuestionRequest
from schemas.pipeline import PipelineRunRequest
from schemas.retrieve import RetrieveRequest
from schemas.sql import SqlRequest

# Hai endpoint nhận cùng bộ tham số, nên mọi test dưới đây chạy cho cả hai.
REQUESTS = [RetrieveRequest, SqlRequest]


@pytest.mark.parametrize("model", REQUESTS)
def test_both_endpoints_share_the_same_fields(model) -> None:
    """/retrieve và /sql phải nhận y hệt nhau — trước đây một bên top_n, một bên top_k."""
    assert set(model.model_fields) == set(QuestionRequest.model_fields)
    assert issubclass(model, QuestionRequest)


@pytest.mark.parametrize("model", REQUESTS)
def test_the_size_field_is_named_top_k(model) -> None:
    assert "top_k" in model.model_fields
    assert "top_n" not in model.model_fields


@pytest.mark.parametrize("model", REQUESTS)
def test_top_n_is_rejected_not_ignored(model) -> None:
    """Tên cũ phải báo 422 chỉ thẳng chỗ sai, không được bỏ qua âm thầm."""
    with pytest.raises(ValidationError, match="top_n"):
        model(question="q", top_n=5)


@pytest.mark.parametrize("model", REQUESTS)
def test_top_k_default_follows_config(model) -> None:
    assert model(question="q").top_k == cfg.pipeline.top_k_output


@pytest.mark.parametrize("model", REQUESTS)
def test_top_k_is_bounded(model) -> None:
    with pytest.raises(ValidationError):
        model(question="q", top_k=0)
    with pytest.raises(ValidationError):
        model(question="q", top_k=21)


@pytest.mark.parametrize("model", REQUESTS)
def test_a_misspelled_dataset_names_the_valid_ones(model) -> None:
    """Lỗi phải liệt kê giá trị hợp lệ: 'vitextsql' thiếu số 2 nhìn rất giống thật."""
    with pytest.raises(ValidationError, match="vitext2sql"):
        model(question="q", dataset="vitextsql")


@pytest.mark.parametrize("model", REQUESTS)
def test_vietnamese_dataset_is_accepted(model) -> None:
    assert model(question="q", dataset="vitext2sql").dataset is Dataset.VITEXT2SQL


@pytest.mark.parametrize("model", REQUESTS)
def test_an_empty_question_is_rejected(model) -> None:
    with pytest.raises(ValidationError):
        model(question="")


@pytest.mark.parametrize("model", REQUESTS)
def test_dataset_defaults_to_spider(model) -> None:
    assert model(question="q").dataset is Dataset.SPIDER


def test_pipeline_rejects_removed_fields() -> None:
    """Client cũ còn gửi method/model — phải báo, không được bỏ qua."""
    with pytest.raises(ValidationError, match="method"):
        PipelineRunRequest(method="murre")
