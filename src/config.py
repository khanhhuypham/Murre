# src/config.py
"""Nạp và validate cấu hình từ config.yaml — MỘT file duy nhất cho mọi dataset.

Mặc định của MỌI section nằm ngay trong file này (các class *Config bên dưới).
Section nào có trong config.yaml thì ghi đè mặc định tương ứng — trừ `encoders` và
`llm`, bắt buộc phải khai.

    from config import cfg

    cfg.dataset_paths.tables            # dataset/spider/tables.json
    cfg.encoder_for("vitext2sql")       # profile encoder của dataset tiếng Việt
    cfg.outputs.result()                # outputs/spider/sgpt-125m-.../turn3/dev.json
    cfg.outputs.sql(k=5)                # outputs/spider/sgpt-125m-.../turn3/sql.5.txt

MỖI DATASET KHAI ENCODER RIÊNG (`datasets.<ds>.encoder` → một khoá trong
`encoders`). Encoder gắn với ngôn ngữ của dataset, nên đổi dataset là encoder tự
đi theo — không cần file config thứ hai, không cần biến môi trường.

Chỉ BÍ MẬT mới đi qua .env, vì config.yaml nằm trong git:
    OPENAI_API_KEY, OPENAI_BASE_URL   (ghi đè profile LLM đang active)

Trỏ sang file config khác: `--config <đường dẫn>` hoặc env MURRE_CONFIG_PATH —
dùng khi mỗi môi trường triển khai có một file riêng.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Mọi đường dẫn trong config đều tương đối so với gốc project, nên chdir về gốc
# ngay khi import → chạy từ đâu cũng đúng (PyCharm, terminal trong src/core/...).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# Thư mục người dùng đang đứng lúc gọi lệnh — phải nhớ TRƯỚC khi chdir, để còn
# giải được đường dẫn tương đối của --config theo đúng cảm nhận của người gõ
# (đứng ở src/ mà gõ "../config.x.yaml" thì phải ra gốc project).
LAUNCH_DIR: Path = Path.cwd()

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
    model_name của encoder là outputs/ tự tách theo model, không cần khai thêm
    biến nào và không lo hai model ghi đè kết quả của nhau.
    """
    tail: str = name.strip().rstrip("/").split("/")[-1].lower()
    safe: str = "".join(c if c.isalnum() or c in "._-" else "-" for c in tail)
    return safe.strip("-") or "unknown"


class GeneralConfig(BaseModel):
    dataset: str = "spider"  # spider | bird
    top_k: List[int] = [3, 5, 10, 20]  # các k để tính recall (paper báo cáo 4 mức này)


class EncoderProfileConfig(BaseModel):
    """Một encoder khai trong `encoders` — mỗi dataset trỏ tới một profile ở đây."""

    # Tắt namespace `model_` của pydantic để dùng được tên field `model_name`.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    # Họ encoder — quyết định cách gộp token và cách phân biệt query/document.
    #   sgpt     : SGPT của paper (SPECB + weighted-mean). Chỉ dùng cho tiếng Anh.
    #   sentence : bi-encoder BERT/XLM-R (mean pooling + tiền tố). Bắt buộc cho
    #              corpus tiếng Việt — xem docstring core/encoder.py.
    type: str = "sgpt"
    batch_size: int = 256
    # Trần TOKEN cho một batch (số câu × độ dài câu dài nhất). Đây mới là thứ chặn
    # tràn bộ nhớ; `batch_size` chỉ đếm số câu nên không thấy được câu dài. Xem
    # core/encoder.py::plan_batches — schema dài 598 token với batch 256 từng làm
    # torch nổ ACCESS_VIOLATION trên máy 16 GB.
    max_batch_tokens: int = 16384
    # Cắt bớt chuỗi quá dài. Chỉ `sentence` dùng; SGPT theo đúng bản gốc, cắt theo
    # giới hạn của chính model.
    max_length: int = 512
    # Tiền tố theo vai trò, chỉ `sentence` dùng. E5 được huấn luyện với
    # "query: " / "passage: "; bỏ đi là điểm tụt hẳn.
    query_prefix: str = ""
    doc_prefix: str = ""

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


class PipelineConfig(BaseModel):
    # B của paper (§4.1). B vừa là số nhánh giữ lại, vừa là số bảng mỗi nhánh
    # retrieve ở mỗi hop (§3.3) → mỗi hop sinh tối đa B×B đường đi.
    beam_size: int = Field(default=5, ge=1)
    # H của paper (§4.1), ĐẾM CẢ hop 1: max_hop=3 → hop 1, 2, 3, tức chỉ 2 lượt
    # Removal. max_hop=1 nghĩa là single-hop.
    max_hop: int = Field(default=3, ge=1)
    # Số bảng mặc định lấy ra: mặc định của /retrieve, /sql và `cli ask`.
    # Tên là top_k cho khớp thuật ngữ top-K của paper; `general.top_k` là chuyện
    # khác — đó là DANH SÁCH các mức k để tính metric.
    top_k_output: int = Field(default=5, ge=1)
    # Số lần thử lại MỘT CÂU HỎI khi retriever ném lỗi (LLM timeout, 429, Ollama bận).
    # Một lượt chạy đầy đủ là hàng nghìn lần gọi LLM, gặp lỗi tạm thời là chắc chắn.
    question_retries: int = Field(default=3, ge=1)


class DatasetConfig(BaseModel):
    """Mọi thứ riêng của MỘT dataset: dữ liệu đầu vào + encoder dùng cho nó.

    `encoder` trỏ tới một khoá trong `encoders`. Encoder gắn với NGÔN NGỮ của
    dataset (SGPT chỉ hiểu tiếng Anh), nên nó thuộc về dataset chứ không phải là
    một giá trị toàn cục — khai ở đây thì đổi dataset là encoder tự đi theo, không
    cần file config riêng hay biến môi trường nào.

    Ba đường dẫn để trống thì DatasetsConfig tự điền theo quy ước, nên trong
    config.yaml chỉ cần khai đúng một dòng `encoder:`.
    """

    encoder: str                        # tên một profile trong `encoders`
    # Định dạng file TRÊN ĐĨA — dataset/loader.py rẽ theo khoá này:
    #   murre       đã tiền xử lý sẵn (tables.json có `schema`, dev.json có
    #               `utterance` + `rel_schema`) → đọc thẳng.
    #   vitext2sql  THÔ, giữ nguyên xi upstream → thích nghi lúc đọc.
    format: str = "murre"
    tables: Optional[str] = None        # schema của mọi database
    dev: Optional[str] = None           # split đánh giá
    prompt: Optional[str] = None        # prompt few-shot của pha Removal

    def with_defaults(self, name: str) -> "DatasetConfig":
        """Điền đường dẫn còn trống theo quy ước thư mục của từng định dạng."""
        if self.format == "vitext2sql":
            # Cây thư mục sao y upstream: dataset/vitext2sql/data/<mức>-level/.
            # Đổi sang word-level hay sang split test thì khai thẳng tables/dev.
            base: str = f"dataset/{name}/data/syllable-level"
            tables, dev = f"{base}/tables.json", f"{base}/dev.json"
        else:
            tables, dev = f"dataset/{name}/tables.json", f"dataset/{name}/dev.json"

        return self.model_copy(update={
            "tables": self.tables or tables,
            "dev": self.dev or dev,
            "prompt": self.prompt or f"prompts/{name}_rewrite.txt",
        })


class DatasetsConfig(BaseModel):
    """Khai báo từng dataset. Tên field phải khớp member của enum Dataset."""

    spider: DatasetConfig = Field(default_factory=lambda: DatasetConfig(encoder="sgpt"))
    bird: DatasetConfig = Field(default_factory=lambda: DatasetConfig(encoder="sgpt"))
    # Tiếng Việt. Dữ liệu dựng bằng scripts/prepare_vitext2sql.py, không có sẵn
    # trong repo. Bắt buộc encoder đa ngữ: SGPT chỉ học tiếng Anh, dùng nó cho
    # tiếng Việt thì recall gần như ngẫu nhiên (5.6 so với 82.2 ở r@5).
    vitext2sql: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(
            encoder="multilingual", format="vitext2sql",
        )
    )

    @model_validator(mode="after")
    def _fill_paths(self) -> "DatasetsConfig":
        """Điền đường dẫn theo TÊN FIELD — chỉ ở đây mới biết dataset tên gì."""
        for name in type(self).model_fields:
            setattr(self, name, getattr(self, name).with_defaults(name=name))
        return self


class PathsConfig(BaseModel):
    """Chỉ còn TEMPLATE file đầu ra; đường dẫn đầu vào nằm ở `datasets`.

    Template còn nguyên {placeholder}; đường dẫn thật lấy qua cfg.outputs.*().
    {dataset} {model} {max_hop} điền từ cfg; {k} do chỗ gọi truyền.
    {model} là nhãn suy ra từ model_name của encoder GẮN VỚI DATASET đó.
    """

    embeddings_cache: str = "outputs/{dataset}_{model}_embeddings.pt"
    result: str = "outputs/{dataset}/{model}/turn{max_hop}/dev.json"
    score: str = "outputs/{dataset}/{model}/turn{max_hop}/score.json"
    sql: str = "outputs/{dataset}/{model}/turn{max_hop}/sql.{k}.txt"
    # Ghi dần từng câu để chạy lại là tiếp tục được, không mất công đã chạy.
    checkpoint: str = "outputs/{dataset}/{model}/turn{max_hop}/checkpoint.jsonl"


class OutputPaths:
    """Đường dẫn đầu ra ĐÃ ĐIỀN SẴN tham số — lấy qua `cfg.outputs`.

    Mỗi template trong PathsConfig có đúng một method ở đây: IDE gợi ý được tên, và
    placeholder bắt buộc (k) là tham số THẬT nên gõ thiếu là biết ngay lúc viết
    code, không phải KeyError lúc chạy.

        cfg.outputs.result()               → outputs/spider/sgpt-125m-.../turn3/dev.json
        cfg.outputs.sql(k=5)               → outputs/spider/sgpt-125m-.../turn3/sql.5.txt

    Cần đường dẫn của lần chạy KHÁC mà không ghi đè `cfg` toàn cục thì dùng for_run()
    — /evaluate làm đúng vậy:

        cfg.outputs.for_run(dataset="bird", model="sgpt-1.3b-...").result()
    """

    def __init__(self, settings: "Settings", **overrides: Any) -> None:
        self._settings: Settings = settings
        self._overrides: Dict[str, Any] = overrides

    def for_run(
        self,
        *,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        max_hop: Optional[int] = None,
    ) -> "OutputPaths":
        """Bản sao chỉ khác ở ba giá trị này; để None thì giữ nguyên theo `cfg`."""
        merged: Dict[str, Any] = dict(self._overrides)
        for name, value in (
            ("dataset", dataset), ("model", model), ("max_hop", max_hop),
        ):
            if value is not None:
                merged[name] = value
        return OutputPaths(self._settings, **merged)

    def _render(self, template: str, **extra: Any) -> str:
        """Điền 3 giá trị từ cfg (đã tính override) + placeholder riêng của template."""
        s: Settings = self._settings
        dataset: str = self._overrides.get("dataset") or s.general.dataset
        values: Dict[str, Any] = {
            "dataset": dataset,
            # Nhãn model đi theo ĐÚNG DATASET đang render, không phải một encoder
            # toàn cục — spider và vitext2sql dùng encoder khác nhau.
            "model": s.encoder_for(dataset).slug,
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

    def checkpoint(self) -> str:
        return self._render(self._settings.paths.checkpoint)


class LoggingConfig(BaseModel):
    level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True  # true = ghi thêm ra file (append), vẫn in ra console
    log_dir: str = "outputs/logs"  # chỉ dùng khi log_to_file: true
    log_file: str = "murre.log"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

    # Dataset mà service này phục vụ. Rỗng = mọi dataset có tables.json trên đĩa.
    #
    # Nên khai tường minh: cả service dùng CHUNG MỘT encoder, mà encoder thì gắn với
    # ngôn ngữ (SGPT cho tiếng Anh, đa ngữ cho tiếng Việt). Để rỗng thì thêm dữ liệu
    # tiếng Việt vào đĩa là service tiếng Anh cũng nạp nó lúc khởi động — mã hoá cả
    # corpus bằng model không hiểu tiếng Việt, chậm mà lại vô dụng.
    datasets: List[str] = []

    # true  → nạp encoder/LLM/embeddings và ping LLM TRƯỚC khi nhận request; thiếu gì
    #         thì startup hỏng luôn. Lên chậm, đổi lại /retrieve chắc chắn chạy.
    # false → nạp lười ở /retrieve đầu tiên; lên nhanh nhưng KHÔNG bảo đảm gì.
    # (xem api/dependencies.py::warmup_datasets)
    preload: bool = True

    # Origin được phép gọi API từ trình duyệt. Rỗng = không bật CORS (mặc định:
    # service nội bộ, gọi từ backend khác chứ không từ browser).
    cors_origins: List[str] = []

    # true → response 500 kèm traceback. CHỈ bật khi debug, không bật ở production.
    debug_errors: bool = False


class Settings(BaseModel):
    """Root config — mọi section có mặc định sẵn, TRỪ `encoders` và `llm`.

    Hai section đó bắt buộc khai trong config.yaml vì chúng quyết định model nào
    được nạp/tải về. Có mặc định ở đây thì khai thiếu vẫn chạy được, chỉ là chạy
    bằng model khác thứ mình tưởng — im lặng và tốn vài GB băng thông.
    """

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    encoders: Dict[str, EncoderProfileConfig]
    llm: LLMConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Config dạng dict thuần, JSON-safe — dùng cho endpoint /config.

        api_key bị che: /config là endpoint đọc công khai của service.
        """
        data: Dict[str, Any] = self.model_dump(mode="json")
        for profile in data.get("llm", {}).get("profiles", {}).values():
            if profile.get("api_key"):
                profile["api_key"] = "***"
        return data

    @property
    def outputs(self) -> OutputPaths:
        """Đường dẫn đầu ra đã điền sẵn tham số — xem OutputPaths.
        (`cfg.paths.*` là template; `cfg.outputs.*()` là đường dẫn thật.)"""
        return OutputPaths(self)

    def dataset_config(self, dataset: Optional[str] = None) -> DatasetConfig:
        """Khai báo của `dataset`. None → `general.dataset` đang chọn.

        Nhận dataset tường minh để chỗ gọi (API phục vụ nhiều dataset cùng lúc)
        không phải ghi đè general.dataset — biến toàn cục, đổi là mọi request thấy.
        """
        name: str = dataset or self.general.dataset
        group: Any = getattr(self.datasets, name, None)
        if not isinstance(group, DatasetConfig):
            raise ValueError(
                f"dataset='{name}' chưa khai báo trong `datasets`.\n"
                f"  Dataset có sẵn: {list(type(self.datasets).model_fields)}"
            )
        return group

    def encoder_for(self, dataset: Optional[str] = None) -> EncoderProfileConfig:
        """Encoder của `dataset` — tra `datasets.<ds>.encoder` trong `encoders`.

        Đây là chỗ ràng dataset với encoder. Nhờ nó mà chỉ cần đổi
        `general.dataset` (hay truyền --dataset) là encoder tự đi theo đúng ngôn
        ngữ, không phải nhớ sửa thêm gì.
        """
        name: str = self.dataset_config(dataset).encoder
        profile: Optional[EncoderProfileConfig] = self.encoders.get(name)
        if profile is None:
            raise ValueError(
                f"dataset='{dataset or self.general.dataset}' trỏ tới encoder "
                f"'{name}' nhưng `encoders` không có profile đó.\n"
                f"  Profile có sẵn: {list(self.encoders)}"
            )
        return profile

    @property
    def dataset_paths(self) -> DatasetConfig:
        """Khai báo của `general.dataset` đang chọn.

            cfg.dataset_paths.tables   → "dataset/spider/tables.json"
            cfg.dataset_paths.prompt   → "prompts/spider_rewrite.txt"
        """
        return self.dataset_config()


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

    # KHÔNG có override cho encoder: giờ mỗi dataset khai encoder riêng nên một biến
    # môi trường đơn lẻ không nói được là ghi đè cái nào. Đổi encoder thì sửa thẳng
    # `encoders` trong config.yaml.


def config_path_from_argv(argv: Optional[List[str]] = None) -> Optional[str]:
    """Đọc `--config <đường dẫn>` (hoặc `--config=<đường dẫn>`) từ dòng lệnh.

    Có cờ này vì biến môi trường MURRE_CONFIG_PATH rất hay hụt trong IDE: đặt ở
    terminal này rồi chạy ở terminal khác, hoặc quên điền vào Run Configuration —
    server vẫn lên bình thường nhưng nạp nhầm config, và chỉ lòi ra ở lần gọi API
    đầu tiên. Cờ dòng lệnh thì nhìn thấy ngay trong lệnh đang chạy.

    Phải đọc thẳng sys.argv chứ không qua argparse: `cfg` được nạp ngay lúc import
    config.py, tức là trước khi bất kỳ parser nào kịp chạy.
    """
    args: List[str] = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def resolve_config_path(value: str) -> Path:
    """Đường dẫn config người dùng đưa vào → đường dẫn tuyệt đối.

    Thử theo thư mục đang đứng trước (LAUNCH_DIR), rồi mới theo gốc project. Nhờ
    vậy đứng ở src/ gõ "../config.prod.yaml" hay đứng ở gốc gõ "config.prod.yaml"
    đều ra cùng một file.
    """
    given: Path = Path(value).expanduser()
    if given.is_absolute():
        return given

    from_launch: Path = (LAUNCH_DIR / given).resolve()
    if from_launch.exists():
        return from_launch
    return (PROJECT_ROOT / given).resolve()


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Đọc config.yaml (hoặc file được chỉ định) và trả về Settings đã validate.

    Thứ tự ưu tiên: tham số hàm → `--config` → env MURRE_CONFIG_PATH → config.yaml.
    """
    if config_path is not None:
        path: Path = config_path
    else:
        chosen: Optional[str] = config_path_from_argv() or os.getenv("MURRE_CONFIG_PATH")
        path = resolve_config_path(value=chosen) if chosen else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {path}\n"
            f"  Đã thử theo thư mục đang đứng ({LAUNCH_DIR}) và theo gốc project "
            f"({PROJECT_ROOT}).\n"
            f"  Config có sẵn: "
            f"{', '.join(sorted(f.name for f in PROJECT_ROOT.glob('config*.yaml'))) or 'không có'}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    try:
        settings: Settings = Settings(**raw)
    except ValidationError as exc:
        # Lỗi thô của pydantic in ra một khối dài không nói rõ phải sửa file nào.
        problems: str = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"config.yaml không hợp lệ ({path.resolve()}):\n{problems}\n\n"
            f"Khối `encoder:` và `llm:` bắt buộc phải có. Ví dụ tối thiểu:\n"
            f"  encoder:\n"
            f"      model_name: Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit\n"
        ) from None

    _apply_env_overrides(settings=settings)
    return settings


def get_llm_profile(profile_name: Optional[str] = None) -> LLMProfileConfig:
    """Cấu hình của 1 profile LLM. None → dùng cfg.llm.active_profile.

    Truyền profile_name để tạm dùng model local khác mà không sửa config.yaml.
    """
    name: str = profile_name or cfg.llm.active_profile
    if name not in cfg.llm.profiles:
        raise ValueError(
            f"LLM profile '{name}' không tồn tại trong config.yaml (llm.profiles).\n"
            f"Các profile có sẵn: {list(cfg.llm.profiles)}"
        )
    return cfg.llm.profiles[name]


# Singleton — nạp một lần khi module được import lần đầu, dùng chung cả project.
cfg: Settings = load_settings()


if __name__ == "__main__":
    # python -m config → toàn bộ cấu hình đang hiệu lực (JSON)
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
