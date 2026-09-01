# src/config.py
"""Nạp và validate cấu hình từ config.yaml.

Mặc định của MỌI section nằm ngay trong file này (các class *Config bên dưới).
Section nào có trong config.yaml thì ghi đè mặc định tương ứng — trừ `llm`, bắt
buộc phải khai trong config.yaml. Trỏ file khác qua env `MURRE_CONFIG_PATH`.

    from config import cfg

    cfg.pipeline.method                # "murre"
    cfg.dataset_paths.tables           # dataset/spider/tables.json
    cfg.outputs.result()               # outputs/spider/sgpt-125m-.../murre/result/turn3/dev.json
    cfg.outputs.sql(k=5)               # outputs/spider/sgpt-125m-.../murre/result/turn3/sql.5.txt

.env ghi đè 3 giá trị đổi theo máy: OPENAI_API_KEY, OPENAI_BASE_URL,
ENCODER_MODEL_NAME (xem _apply_env_overrides).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Mọi đường dẫn trong config đều tương đối so với gốc project, nên chdir về gốc
# ngay khi import → chạy từ đâu cũng đúng (PyCharm, terminal trong src/steps/...).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

os.chdir(PROJECT_ROOT)

load_dotenv()


# ---------------------------------------------------------------------------
# Các schema con (map 1-1 với từng section trong config.yaml)
# ---------------------------------------------------------------------------
def model_slug(name: str) -> str:
    """Tên model → nhãn thư mục an toàn cho đường dẫn.

        Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit
        → sgpt-125m-weightedmean-msmarco-specb-bitfit

    Bỏ phần org trước "/", hạ chữ thường, ký tự lạ đổi thành "-". Nhờ vậy đổi
    encoder.model_name là outputs/ tự tách theo model, không cần khai thêm biến
    nào và không lo hai model ghi đè kết quả của nhau.
    """
    tail: str = name.strip().rstrip("/").split("/")[-1].lower()
    safe: str = "".join(c if c.isalnum() or c in "._-" else "-" for c in tail)
    return safe.strip("-") or "unknown"


class GeneralConfig(BaseModel):
    dataset: str = "spider"  # spider | bird
    top_k: List[int] = [3, 5, 10, 20]  # các k để tính recall (paper báo cáo 4 mức này)
    random_seed: int = 42


class EncoderConfig(BaseModel):
    # Tắt namespace `model_` của pydantic để dùng được tên field `model_name`.
    model_config = ConfigDict(protected_namespaces=())

    # Model THỰC SỰ nạp lên — cũng là thứ DUY NHẤT cần đổi khi muốn thử model khác:
    # nhãn thư mục outputs/ suy ra từ đây (xem `slug`) nên không thể quên đồng bộ.
    model_name: str = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
    batch_size: int = 256

    @property
    def slug(self) -> str:
        """Nhãn thư mục outputs/ suy ra từ model_name — placeholder {model}."""
        return model_slug(name=self.model_name)


class LLMProfileConfig(BaseModel):
    """Một model/endpoint LLM (OpenAI, Groq, Ollama...) khai trong llm.profiles."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    api_key: str = ""  # trống → điền qua .env: OPENAI_API_KEY
    base_url: str = ""  # trống → endpoint OpenAI mặc định
    temperature: float = 0.0

    connect_timeout: float = 3.0
    timeout: float = 120.0
    max_retries: int = 1  # 0 = không thử lại; lỗi kết nối thử lại cũng vô ích


class LLMConfig(BaseModel):
    active_profile: str
    profiles: Dict[str, LLMProfileConfig]


class AblationConfig(BaseModel):
    """Ba cờ ablation của paper (Table 4, §4.3) — false để tắt từng thành phần."""
    removal: bool = True     # false = nối câu hỏi + bảng đã tìm, thay vì Removal
    tabulation: bool = True  # false = Removal trả câu tự nhiên, không ép sang dạng schema
    early_stop: bool = True  # false = luôn chạy đủ max_hop


class PipelineConfig(BaseModel):
    method: str = "murre"  # murre | single_hop | crush
    beam_size: int = 5     # paper dùng 5. Tăng → tốt hơn nhưng tốn API call hơn
    max_hop: int = 3
    top_k_pool: int = 100  # hop 1+ chỉ search trong top_k_pool kết quả của hop 0
    top_n_output: int = 5  # số bảng truyền vào bước sinh SQL
    ablation: AblationConfig = Field(default_factory=AblationConfig)


class DatasetPathsConfig(BaseModel):
    """Đường dẫn dữ liệu đầu vào của MỘT dataset (paths.spider / paths.bird)."""

    tables: str
    dev: str
    gold: str
    prompt: str
    prompt_no_tabulation: str
    prompt_crush: str

    @classmethod
    def for_dataset(cls, name: str) -> "DatasetPathsConfig":
        """Dựng đủ 6 đường dẫn theo quy ước chung — thêm dataset mới chỉ cần đặt
        file đúng quy ước rồi thêm 1 field trong PathsConfig."""
        return cls(
            tables=f"dataset/{name}/tables.json",
            dev=f"dataset/{name}/dev.json",
            gold=f"dataset/{name}/gold.txt",
            prompt=f"prompts/{name}_rewrite.txt",
            prompt_no_tabulation=f"prompts/{name}_rewrite_no_tabulation.txt",
            prompt_crush=f"prompts/{name}_crush.txt",
        )


class PathsConfig(BaseModel):
    # --- Dữ liệu đầu vào ---
    spider: DatasetPathsConfig = Field(
        default_factory=lambda: DatasetPathsConfig.for_dataset("spider")
    )
    bird: DatasetPathsConfig = Field(
        default_factory=lambda: DatasetPathsConfig.for_dataset("bird")
    )

    # --- File đầu ra ---
    # TEMPLATE còn nguyên {placeholder}; đường dẫn thật lấy qua cfg.outputs.*().
    # {dataset} {model} {method} {max_hop} điền từ cfg; {k} do chỗ gọi truyền.
    # {model} = encoder.slug, suy ra từ encoder.model_name.
    # Không còn template file trung gian (turn{N}/, rewrite/outputs/): pipeline giữ
    # mọi thứ trong RAM kể từ khi chuỗi steps/{retrieve,rewrite,score}.py bị xoá.
    embeddings_cache: str = "outputs/{dataset}_{model}_embeddings.pt"
    result: str = "outputs/{dataset}/{model}/{method}/result/turn{max_hop}/dev.json"
    score: str = "outputs/{dataset}/{model}/{method}/result/turn{max_hop}/score.json"
    sql: str = "outputs/{dataset}/{model}/{method}/result/turn{max_hop}/sql.{k}.txt"


class OutputPaths:
    """Đường dẫn đầu ra ĐÃ ĐIỀN SẴN tham số — lấy qua `cfg.outputs`.

    Mỗi template trong PathsConfig có đúng một method ở đây: IDE gợi ý được tên, và
    placeholder bắt buộc (k) là tham số THẬT nên gõ thiếu là biết ngay lúc viết
    code, không phải KeyError lúc chạy.

        cfg.outputs.result()               → outputs/spider/sgpt-125m-.../murre/result/turn3/dev.json
        cfg.outputs.sql(k=5)               → outputs/spider/sgpt-125m-.../murre/result/turn3/sql.5.txt

    Cần đường dẫn của lần chạy KHÁC mà không ghi đè `cfg` toàn cục thì dùng for_run()
    — /evaluate làm đúng vậy:

        cfg.outputs.for_run(dataset="bird", model="sgpt-1.3b-...", method="crush").result()
    """

    def __init__(self, settings: Settings, **overrides: Any) -> None:
        self._settings: Settings = settings
        self._overrides: Dict[str, Any] = overrides

    def for_run(
        self,
        *,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        method: Optional[str] = None,
        max_hop: Optional[int] = None,
    ) -> OutputPaths:
        """Bản sao chỉ khác ở bốn giá trị này; để None thì giữ nguyên theo `cfg`."""
        merged: Dict[str, Any] = dict(self._overrides)
        for name, value in (
            ("dataset", dataset), ("model", model),
            ("method", method), ("max_hop", max_hop),
        ):
            if value is not None:
                merged[name] = value
        return OutputPaths(self._settings, **merged)

    def _render(self, template: str, **extra: Any) -> str:
        """Điền 4 giá trị từ cfg (đã tính override) + placeholder riêng của template."""
        s: Settings = self._settings
        values: Dict[str, Any] = {
            "dataset": s.general.dataset,
            "model": s.encoder.slug,
            "method": s.pipeline.method,
            "max_hop": s.pipeline.max_hop,
            **self._overrides,
            **extra,
        }
        return template.format(**values)

    # --- Một method cho mỗi template trong PathsConfig ------------------------
    def embeddings_cache(self) -> str:
        return self._render(self._settings.paths.embeddings_cache)

    def result(self) -> str:
        return self._render(self._settings.paths.result)

    def score(self) -> str:
        return self._render(self._settings.paths.score)

    def sql(self, k: int) -> str:
        return self._render(self._settings.paths.sql, k=k)


class LoggingConfig(BaseModel):
    level: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True  # true = ghi thêm ra file (append), vẫn in ra console
    log_dir: str = "outputs/logs"  # chỉ dùng khi log_to_file: true
    log_file: str = "murre.log"

class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    default_top_n: int = 5  # số bảng trả về tối đa qua /retrieve

    # true  → nạp encoder/LLM/embeddings và ping LLM TRƯỚC khi nhận request; thiếu gì
    #         thì startup hỏng luôn. Lên chậm, đổi lại /retrieve chắc chắn chạy.
    # false → nạp lười ở /retrieve đầu tiên; lên nhanh nhưng KHÔNG bảo đảm gì.
    # (xem api/dependencies.py::warmup_datasets)
    preload: bool = True


class RunOptionConfig(BaseModel):
    # one_question → chạy MỘT câu hỏi, in top-N ra terminal, không ghi file.
    # batch        → chạy CẢ dev.json, ghi result + score ra outputs/.
    mode: str = "batch"


class Settings(BaseModel):
    """Root config — mọi section có mặc định sẵn, trừ `llm` (bắt buộc khai trong YAML)."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    llm: LLMConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    run_option: RunOptionConfig = Field(default_factory=RunOptionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Config dạng dict thuần, JSON-safe — dùng cho endpoint /config."""
        return self.model_dump(mode="json")

    @property
    def outputs(self) -> OutputPaths:
        """Đường dẫn đầu ra đã điền sẵn tham số — xem OutputPaths.
        (`cfg.paths.*` là template; `cfg.outputs.*()` là đường dẫn thật.)"""
        return OutputPaths(self)

    @property
    def dataset_paths(self) -> DatasetPathsConfig:
        """Nhóm đường dẫn đầu vào của `general.dataset` đang chọn.

            cfg.dataset_paths.tables   → "dataset/spider/tables.json"
            cfg.dataset_paths.prompt   → "prompts/spider_rewrite.txt"
        """
        group: Any = getattr(self.paths, self.general.dataset, None)
        if not isinstance(group, DatasetPathsConfig):
            available: List[str] = [
                name
                for name in type(self.paths).model_fields
                if isinstance(getattr(self.paths, name), DatasetPathsConfig)
            ]
            raise ValueError(
                f"general.dataset='{self.general.dataset}' chưa khai báo đường dẫn "
                f"trong paths.\n  Dataset có sẵn: {available}"
            )
        return group


# ---------------------------------------------------------------------------
# Nạp config
# ---------------------------------------------------------------------------
def _apply_env_overrides(settings: Settings) -> None:
    """Cho .env ghi đè config.yaml ở 3 giá trị hay đổi theo máy.

    api_key/base_url chỉ ghi đè cho profile ĐANG ACTIVE — profile khác giữ nguyên.
    """
    name: str = settings.llm.active_profile
    if name not in settings.llm.profiles:
        raise ValueError(
            f"llm.active_profile='{name}' không tồn tại trong config.yaml (llm.profiles).\n"
            f"Các profile có sẵn: {list(settings.llm.profiles)}"
        )

    active: LLMProfileConfig = settings.llm.profiles[name]

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        active.api_key = api_key

    base_url: str = os.getenv("OPENAI_BASE_URL", "")
    if base_url:
        active.base_url = base_url

    model_name: str = os.getenv("ENCODER_MODEL_NAME", "")
    if model_name:
        settings.encoder.model_name = model_name


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Đọc config.yaml (hoặc file được chỉ định) và trả về Settings đã validate."""

    path: Path = config_path or Path(
        os.getenv("MURRE_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {path.resolve()}. "
            "Đặt config.yaml ở gốc project (MURRE_V2/) hoặc chỉ định qua biến môi "
            "trường MURRE_CONFIG_PATH."
        )

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    settings: Settings = Settings(**raw)
    _apply_env_overrides(settings=settings)
    return settings


# ---------------------------------------------------------------------------
# Helper đường dẫn & LLM profile
# ---------------------------------------------------------------------------
def list_llm_profiles() -> List[str]:
    """Tên mọi profile LLM trong config.yaml (llm.profiles) — xem HUONG_DAN.md mục 3b."""
    return list(cfg.llm.profiles)


def print_llm_profiles() -> None:
    """In các profile LLM, đánh dấu * vào cái đang dùng. Chạy: python -m config --llm"""
    active: str = cfg.llm.active_profile
    print(f"{'=' * 60}\n  LLM PROFILE trong config.yaml\n{'=' * 60}")
    for name in list_llm_profiles():
        mark: str = "*" if name == active else " "
        print(f"  {mark} {name}")
    print(f"{'=' * 60}\n  (* = llm.active_profile đang dùng)")


def get_llm_profile(profile_name: Optional[str] = None) -> LLMProfileConfig:
    """Cấu hình của 1 profile LLM. None → dùng cfg.llm.active_profile.

    Truyền profile_name để tạm dùng model local khác mà không sửa config.yaml.
    """
    name: str = profile_name or cfg.llm.active_profile
    if name not in cfg.llm.profiles:
        raise ValueError(
            f"LLM profile '{name}' không tồn tại trong config.yaml (llm.profiles).\n"
            f"Các profile có sẵn: {list_llm_profiles()}"
        )
    return cfg.llm.profiles[name]


def print_config() -> None:
    """In toàn bộ cấu hình hiện tại ra terminal để kiểm tra."""
    print(f"{'=' * 60}\n  CẤU HÌNH HIỆN TẠI\n{'=' * 60}")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print("=" * 60)


# Giá trị MẪU cho placeholder không suy ra được từ cfg — chỉ dùng trong print_paths().
_EXAMPLE_PLACEHOLDERS: Dict[str, Any] = {"hop": 1, "beam": 2, "k": 5}


def print_paths() -> None:
    """In mọi đường dẫn ĐÃ RESOLVE theo cfg hiện tại. Chạy: python -m config --paths"""
    header: str = (
        f"dataset={cfg.general.dataset} model={cfg.encoder.slug} "
        f"method={cfg.pipeline.method} max_hop={cfg.pipeline.max_hop}"
    )
    print(f"{'=' * 78}\n  ĐƯỜNG DẪN THỰC TẾ — {header}\n{'=' * 78}")

    print(f"\n--- Dữ liệu đầu vào (cfg.dataset_paths → paths.{cfg.general.dataset}) ---")
    ds_paths: DatasetPathsConfig = cfg.dataset_paths
    for name in DatasetPathsConfig.model_fields:
        print(f"  {name:22s} {getattr(ds_paths, name)}")

    print("\n--- File trung gian & đầu ra (cfg.outputs.*) ---")
    # Lặp theo TÊN template nên phải render động bằng _render(). Code ngoài config.py
    # thì luôn gọi method: cfg.outputs.result(), ...
    resolver: OutputPaths = cfg.outputs
    for name, template in cfg.paths.model_dump().items():
        # Bỏ qua spider/bird — đó là đường dẫn đầu vào, đã in ở khối trên.
        if not isinstance(template, str):
            continue

        extra: Dict[str, Any] = {
            field: _EXAMPLE_PLACEHOLDERS[field]
            for _, field, _, _ in Formatter().parse(template)
            if field in _EXAMPLE_PLACEHOLDERS
        }
        call: str = f"{name}({', '.join(f'{k}={v}' for k, v in extra.items())})"
        print(f"  {call:34s} {resolver._render(template, **extra)}")

    print("=" * 78)


# Singleton — nạp một lần khi module được import lần đầu, dùng chung cả project.
cfg: Settings = load_settings()


if __name__ == "__main__":
    #   python -m config          → toàn bộ cấu hình đang hiệu lực (JSON)
    #   python -m config --paths  → mọi đường dẫn đã resolve theo cfg hiện tại
    #   python -m config --llm    → danh sách profile LLM trong config.yaml
    args: List[str] = sys.argv[1:]
    if "--paths" in args:
        print_paths()
    elif "--llm" in args:
        print_llm_profiles()
    else:
        print_config()
