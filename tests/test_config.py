"""Cấu hình: sinh nhãn model và render đường dẫn đầu ra."""
from __future__ import annotations

import pytest

from config import cfg, model_slug


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
         "sgpt-125m-weightedmean-msmarco-specb-bitfit"),
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("org/Model Name!", "model-name"),
        ("", "unknown"),
    ],
)
def test_model_slug(name: str, expected: str) -> None:
    assert model_slug(name=name) == expected


def test_for_run_overrides_only_what_it_is_given() -> None:
    path = cfg.outputs.for_run(dataset="bird", model="tiny", max_hop=2).result()
    assert path == "outputs/bird/tiny/turn2/dev.json"


def test_for_run_does_not_touch_global_cfg() -> None:
    before = cfg.general.dataset
    cfg.outputs.for_run(dataset="bird").result()
    assert cfg.general.dataset == before


def test_sql_path_carries_k() -> None:
    assert cfg.outputs.for_run(dataset="spider", model="tiny", max_hop=3).sql(k=5) == (
        "outputs/spider/tiny/turn3/sql.5.txt"
    )


def test_to_dict_masks_api_keys() -> None:
    cfg.llm.profiles[cfg.llm.active_profile].api_key = "sk-secret"
    try:
        dumped = cfg.to_dict()
        assert dumped["llm"]["profiles"][cfg.llm.active_profile]["api_key"] == "***"
    finally:
        cfg.llm.profiles[cfg.llm.active_profile].api_key = ""


def test_unknown_dataset_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="spider"):
        cfg.dataset_config("mysql")


def test_each_dataset_resolves_its_own_encoder() -> None:
    """Cốt lõi của việc gộp hai config làm một: encoder đi theo dataset."""
    assert cfg.encoder_for("spider").type == "sgpt"
    assert cfg.encoder_for("vitext2sql").type == "sentence"


def test_vietnamese_dataset_uses_a_multilingual_model() -> None:
    """SGPT cho tiếng Việt ra recall gần như ngẫu nhiên — chốt lại bằng test."""
    assert "multilingual" in cfg.encoder_for("vitext2sql").model_name


def test_e5_prefixes_are_configured() -> None:
    """Thiếu tiền tố "query: "/"passage: " là điểm E5 tụt hẳn mà không báo gì."""
    profile = cfg.encoder_for("vitext2sql")
    assert profile.query_prefix == "query: "
    assert profile.doc_prefix == "passage: "


def test_output_path_model_label_follows_the_dataset_encoder() -> None:
    """Hai dataset dùng encoder khác nhau thì kết quả không được ghi đè nhau."""
    assert "sgpt" in cfg.outputs.for_run(dataset="spider").result()
    assert "e5" in cfg.outputs.for_run(dataset="vitext2sql").result()


def test_dataset_paths_follow_the_naming_convention() -> None:
    """Khai mỗi `encoder:` trong config.yaml, ba đường dẫn tự suy ra."""
    d = cfg.dataset_config("vitext2sql")
    assert d.tables == "dataset/vitext2sql/tables.json"
    assert d.dev == "dataset/vitext2sql/dev.json"
    assert d.prompt == "prompts/vitext2sql_rewrite.txt"


def test_a_dataset_pointing_at_a_missing_encoder_fails_loudly() -> None:
    saved = cfg.datasets.spider.encoder
    cfg.datasets.spider.encoder = "khong-ton-tai"
    try:
        with pytest.raises(ValueError, match="khong-ton-tai"):
            cfg.encoder_for("spider")
    finally:
        cfg.datasets.spider.encoder = saved
