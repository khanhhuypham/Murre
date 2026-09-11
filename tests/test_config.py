"""Cấu hình: sinh nhãn model và render đường dẫn đầu ra."""
from __future__ import annotations

import pytest

from pydantic import ValidationError

from config import EncoderProfileConfig, LLMProfileConfig, cfg, model_slug


@pytest.mark.parametrize(
    "name, expected",
    [
        ("intfloat/multilingual-E5-Base", "multilingual-e5-base"),
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


@pytest.mark.parametrize("bad_type", ["sgpt", "phobert", ""])
def test_config_rejects_any_encoder_type_but_sentence(bad_type: str) -> None:
    """`sgpt` nằm trong danh sách này: config.yaml của bản cũ phải nổ lỗi.

    Lớp SGPTEncoder đã bỏ khỏi nhánh này. Nếu `type: sgpt` bị bỏ qua âm thầm và
    rơi về encoder duy nhất còn lại thì chạy xong cả lượt vẫn không ai biết mình
    đã dùng model nào. Nổ ngay lúc nạp config.yaml, trước khi tải model.
    """
    with pytest.raises(ValidationError, match="sentence"):
        EncoderProfileConfig(model_name="intfloat/multilingual-e5-base", type=bad_type)


def test_encoder_type_defaults_to_sentence() -> None:
    """Khai thiếu `type` thì vẫn chạy — chỉ khai SAI mới bị chặn."""
    assert EncoderProfileConfig(model_name="intfloat/multilingual-e5-base").type == "sentence"


def test_each_dataset_resolves_its_own_encoder() -> None:
    """Cốt lõi của việc gộp hai config làm một: encoder đi theo dataset."""
    for ds in ("spider", "bird", "vitext2sql"):
        assert cfg.encoder_for(ds).type == "sentence"
        assert cfg.encoder_for(ds).model_name


def test_every_dataset_uses_a_multilingual_model() -> None:
    """Model chỉ học tiếng Anh cho recall gần như ngẫu nhiên trên corpus tiếng Việt
    (5.6 so với 82.2 ở r@5) — chốt lại bằng test, kể cả với dataset tiếng Anh vì
    cả ba đang dùng chung một profile."""
    for ds in ("spider", "bird", "vitext2sql"):
        assert "multilingual" in cfg.encoder_for(ds).model_name


def test_e5_prefixes_are_configured() -> None:
    """Thiếu tiền tố "query: "/"passage: " là điểm E5 tụt hẳn mà không báo gì."""
    profile = cfg.encoder_for("vitext2sql")
    assert profile.query_prefix == "query: "
    assert profile.doc_prefix == "passage: "


def test_output_path_carries_both_dataset_and_model() -> None:
    """Cùng một encoder cho mọi dataset, nên TÊN DATASET là thứ tách kết quả ra."""
    spider = cfg.outputs.for_run(dataset="spider").result()
    vitext = cfg.outputs.for_run(dataset="vitext2sql").result()

    assert "e5" in spider and "e5" in vitext
    assert spider != vitext


def test_murre_dataset_paths_follow_the_naming_convention() -> None:
    """Dataset dạng `murre` (spider, bird): ba đường dẫn tự suy ra theo tên."""
    d = cfg.dataset_config("spider")
    assert d.format == "murre"
    assert d.tables == "dataset/spider/tables.json"
    assert d.dev == "dataset/spider/dev.json"
    assert d.prompt == "prompts/spider_rewrite.txt"


def test_raw_dataset_paths_point_into_the_untouched_tree() -> None:
    """Dataset dạng `vitext2sql`: trỏ thẳng vào bản sao nguyên xi của upstream."""
    d = cfg.dataset_config("vitext2sql")
    assert d.format == "vitext2sql"
    assert d.tables == "dataset/vitext2sql/data/syllable-level/tables.json"
    assert d.dev == "dataset/vitext2sql/data/syllable-level/dev.json"
    assert d.prompt == "prompts/vitext2sql_rewrite.txt"


def test_a_dataset_pointing_at_a_missing_encoder_fails_loudly() -> None:
    saved = cfg.datasets.spider.encoder
    cfg.datasets.spider.encoder = "khong-ton-tai"
    try:
        with pytest.raises(ValueError, match="khong-ton-tai"):
            cfg.encoder_for("spider")
    finally:
        cfg.datasets.spider.encoder = saved


# --- LLM profile: khóa API ---------------------------------------------------
def _profile(name: str, **kw: object) -> LLMProfileConfig:
    """Profile rời, có đặt tên — tên là thứ mọi thông báo lỗi phải nói rõ."""
    p = LLMProfileConfig(model_name="gpt-4o", **kw)
    p._name = name
    return p


def test_each_profile_reads_its_own_env_var(monkeypatch) -> None:
    """Điểm chính của `api_key_env`: hai profile, hai biến môi trường khác nhau.

    Trước đây chỉ có một biến OPENAI_API_KEY và nó CHỈ áp cho profile đang
    active, nên `--llm-profile <tên khác>` báo thiếu khóa dù .env đã có.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    groq = _profile("groq", api_key_env="GROQ_API_KEY")
    openai = _profile("openai")          # mặc định OPENAI_API_KEY

    assert groq.resolve_api_key() == "gsk-groq"
    assert openai.resolve_api_key() == "sk-openai"


def test_env_wins_over_config_yaml(monkeypatch) -> None:
    """config.yaml nằm trong git nên khóa thật ở .env — .env phải thắng."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-tu-env")
    assert _profile("p", api_key="sk-trong-file").resolve_api_key() == "sk-tu-env"


def test_local_profile_needs_no_key(monkeypatch) -> None:
    """Ollama không kiểm khóa, nhưng SDK OpenAI đòi chuỗi non-empty."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local = _profile("ollama", base_url="http://localhost:11434/v1")

    assert local.is_local
    assert local.resolve_api_key() == "ollama"


def test_missing_key_names_the_right_env_var(monkeypatch) -> None:
    """Thông báo phải chỉ đúng biến CỦA PROFILE ĐÓ, không phải OPENAI_API_KEY.

    Báo sai tên biến là người dùng đi điền đúng biến rồi vẫn thấy báo thiếu.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq = _profile("groq", api_key_env="GROQ_API_KEY",
                    base_url="https://api.groq.com/openai/v1")

    with pytest.raises(ValueError, match="GROQ_API_KEY") as exc:
        groq.resolve_api_key()
    assert "groq" in str(exc.value)
