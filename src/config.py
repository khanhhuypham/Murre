# src/config.py
"""Nạp và validate cấu hình từ config.yaml.

Cấu hình của project chia làm hai chỗ:

  * `config.yaml` ở gốc project — encoder SGPT và LLM.
  * Chính file này — giá trị mặc định của `general`, `pipeline`, `paths`,
    `logging`, `api`, `run_option` (xem các class *Config bên dưới).

Section nào có trong config.yaml thì ghi đè mặc định tương ứng ở đây, nên vẫn có
thể chuyển một section về lại file YAML bất cứ lúc nào mà không sửa code.

Cách dùng ở nơi khác trong project:

    from config import cfg

    print(cfg.pipeline.method)                 # "murre"
    print(cfg.general.top_k)                   # [3, 5, 10, 20]
    cfg.dataset_paths.tables                   # dataset/spider/tables.json
    cfg.outputs.result()                       # outputs/spider/125m/murre/result/turn3/dev.json
    cfg.outputs.turn_n(hop=1, beam=2)          # outputs/spider/125m/murre/turn1/dev.2.json

Có thể trỏ tới file config khác bằng biến môi trường `MURRE_CONFIG_PATH`, hữu ích
khi chạy nhiều cấu hình song song (vd. thử nghiệm/production).

Ba giá trị hay đổi theo từng máy được .env ghi đè lên config.yaml, xem
`_apply_env_overrides()`: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ENCODER_MODEL_NAME`.
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

# ---------------------------------------------------------------------------
# Neo thư mục làm việc về GỐC PROJECT (nơi chứa config.yaml)
#
# File này nằm ở src/config.py → gốc project = thư mục cha của src/.
# Mọi đường dẫn trong cấu hình (outputs/, dataset/, prompts/) đều tương đối
# so với gốc project, nên ta chuyển cwd về gốc ngay khi import config. Nhờ vậy
# lệnh chạy đúng dù được gọi từ bất kỳ đâu (PyCharm right-click Run, terminal
# trong src/steps/, v.v.) — không còn lỗi "Không tìm thấy config.yaml".
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

os.chdir(PROJECT_ROOT)

# Tải biến môi trường từ file .env (nếu có) — đọc từ gốc project sau khi chdir
load_dotenv()


# ---------------------------------------------------------------------------
# Các schema con (map 1-1 với từng section trong config.yaml)
# ---------------------------------------------------------------------------
class GeneralConfig(BaseModel):
    """Tham số chung — đặt NGAY TẠI ĐÂY, không khai báo trong config.yaml.

    Thêm lại section `general:` vào config.yaml thì section đó ghi đè các mặc định này.
    """
    dataset: str = "spider"  # spider | bird
    # Model càng lớn → kết quả càng tốt nhưng cần RAM nhiều hơn
    scale: str = "125m"  # 125m | 1.3b | 2.7b | 5.8b
    # Số lượng top-K bảng để đánh giá recall — Table 2/3 của paper báo cáo tại 3, 5, 10, 20
    top_k: List[int] = [3, 5, 10, 20]
    random_seed: int = 42  # seed ngẫu nhiên để tái hiện kết quả

class EncoderConfig(BaseModel):
    # `model_` là namespace pydantic giữ riêng cho chính nó; tắt để dùng được tên
    # field `model_name` khớp với config.yaml mà không sinh UserWarning.
    model_config = ConfigDict(protected_namespaces=())

    # PHẢI khớp general.scale — lệch nhau là nạp vector của model khác → kết quả sai.
    model_name: str = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
    batch_size: int = 256


class LLMProfileConfig(BaseModel):
    """Một model/endpoint LLM (OpenAI, Groq, Ollama...) khai báo trong llm.profiles."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    api_key: str = ""  # để trống → điền qua .env: OPENAI_API_KEY
    base_url: str = ""  # để trống → endpoint OpenAI mặc định
    temperature: float = 0.0


class LLMConfig(BaseModel):
    active_profile: str
    profiles: Dict[str, LLMProfileConfig]


class AblationConfig(BaseModel):
    """Ba cờ ablation của paper (Table 4, §4.3) — đặt false để tắt từng thành phần."""

    # false = "w/o removal": nối (splice) câu hỏi + bảng đã tìm, thay vì Removal
    removal: bool = True
    # false = "w/o tabulation": Removal trả về câu hỏi tự nhiên, không ép sang dạng schema
    tabulation: bool = True
    # false = "w/o early stop": luôn chạy đủ max_hop, không cho phép dừng sớm
    early_stop: bool = True


class PipelineConfig(BaseModel):
    """Pipeline MURRE — đặt NGAY TẠI ĐÂY, không khai báo trong config.yaml.

    Thêm lại section `pipeline:` vào config.yaml thì section đó ghi đè các mặc định này.
    """
    method: str = "murre"  # murre | single_hop | crush — Bảng 2/3 của paper
    # Số beam duy trì tại mỗi hop (bài báo: 5). Tăng → kết quả tốt hơn nhưng tốn
    # API call hơn.
    beam_size: int = 5
    max_hop: int = 3
    # Số bảng tối đa lấy từ corpus tại mỗi hop để thu hẹp không gian tìm kiếm.
    # Hop 1+ chỉ search trong top_k_pool kết quả của hop 0.
    top_k_pool: int = 100
    # Số bảng đầu ra cuối cùng để truyền vào bước sinh SQL
    top_n_output: int = 5
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
        """Dựng đủ 6 đường dẫn theo quy ước đặt tên chung của spider và bird.

        Thêm dataset thứ ba chỉ cần đặt file đúng quy ước này rồi khai báo thêm một
        field trong PathsConfig — không phải gõ lại từng đường dẫn.
        """
        return cls(
            tables=f"dataset/{name}/tables.json",
            dev=f"dataset/{name}/dev.json",
            gold=f"dataset/{name}/gold.txt",
            prompt=f"prompts/{name}_rewrite.txt",
            prompt_no_tabulation=f"prompts/{name}_rewrite_no_tabulation.txt",
            prompt_crush=f"prompts/{name}_crush.txt",
        )


class PathsConfig(BaseModel):
    """Đường dẫn file — đặt NGAY TẠI ĐÂY, không khai báo trong config.yaml.

    Thêm lại section `paths:` vào config.yaml thì section đó ghi đè các mặc định này.
    """

    # --- Dữ liệu đầu vào ---
    spider: DatasetPathsConfig = Field(
        default_factory=lambda: DatasetPathsConfig.for_dataset("spider")
    )
    bird: DatasetPathsConfig = Field(
        default_factory=lambda: DatasetPathsConfig.for_dataset("bird")
    )

    # --- File trung gian và đầu ra ---
    # Đây là TEMPLATE còn nguyên {placeholder}; đường dẫn thật lấy qua cfg.outputs.*()
    # — {dataset}, {scale}, {method}, {max_hop} điền theo general.* / pipeline.*, còn
    # {hop}, {beam}, {k} do chỗ gọi truyền vào.
    # Cache embeddings — MỘT định dạng duy nhất (.pt), dùng chung cho Option 1,
    # Option 2 và API. Trước đây còn thêm embeddings.json theo format của tác giả:
    # cùng bộ vector nhưng lưu 2 nơi, encode 2 lần, nên đã bỏ.
    embeddings_cache: str = "outputs/{dataset}_{scale}_embeddings.pt"
    turn0: str = "outputs/{dataset}/{scale}/{method}/turn0/dev.json"
    rewrite_output: str = "outputs/{dataset}/{scale}/{method}/rewrite/outputs/turn{hop}"
    turn_n: str = "outputs/{dataset}/{scale}/{method}/turn{hop}/dev.{beam}.json"
    result: str = "outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/dev.json"
    score: str = "outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/score.json"
    sql: str = "outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/sql.{k}.txt"


class OutputPaths:
    """Đường dẫn file trung gian & đầu ra ĐÃ ĐIỀN SẴN tham số — lấy qua `cfg.outputs`.

    Mỗi template trong PathsConfig có đúng một method ở đây, thay cho việc tra bằng
    key dạng chuỗi: IDE gợi ý được tên, Ctrl+click nhảy tới method rồi tới template,
    và placeholder bắt buộc (hop/beam/k) là tham số THẬT — gõ thiếu hay gõ sai là biết
    ngay lúc viết code, không phải KeyError lúc chạy.

        cfg.outputs.result()               → outputs/spider/125m/murre/result/turn3/dev.json
        cfg.outputs.turn_n(hop=1, beam=2)  → outputs/spider/125m/murre/turn1/dev.2.json

    Cần đường dẫn của MỘT lần chạy khác mà không ghi đè `cfg` toàn cục thì dùng
    for_run() — /evaluate làm đúng như vậy:

        cfg.outputs.for_run(dataset="bird", scale="1.3b", method="crush").result()
    """

    def __init__(self, settings: Settings, **overrides: Any) -> None:
        self._settings: Settings = settings
        self._overrides: Dict[str, Any] = overrides

    def for_run(
        self,
        *,
        dataset: Optional[str] = None,
        scale: Optional[str] = None,
        method: Optional[str] = None,
        max_hop: Optional[int] = None,
    ) -> OutputPaths:
        """Bản sao chỉ khác ở bốn giá trị này; để None thì giữ nguyên theo `cfg`."""
        merged: Dict[str, Any] = dict(self._overrides)
        for name, value in (
            ("dataset", dataset), ("scale", scale),
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
            "scale": s.general.scale,
            "method": s.pipeline.method,
            "max_hop": s.pipeline.max_hop,
            **self._overrides,
            **extra,
        }
        return template.format(**values)

    # --- Một method cho mỗi template trong PathsConfig ------------------------
    def embeddings_cache(self) -> str:
        """outputs/{dataset}_{scale}_embeddings.pt"""
        return self._render(self._settings.paths.embeddings_cache)

    def turn0(self) -> str:
        """outputs/{dataset}/{scale}/{method}/turn0/dev.json"""
        return self._render(self._settings.paths.turn0)

    def rewrite_output(self, hop: int) -> str:
        """outputs/{dataset}/{scale}/{method}/rewrite/outputs/turn{hop}"""
        return self._render(self._settings.paths.rewrite_output, hop=hop)

    def turn_n(self, hop: int, beam: int) -> str:
        """outputs/{dataset}/{scale}/{method}/turn{hop}/dev.{beam}.json"""
        return self._render(self._settings.paths.turn_n, hop=hop, beam=beam)

    def result(self) -> str:
        """outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/dev.json"""
        return self._render(self._settings.paths.result)

    def score(self) -> str:
        """outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/score.json"""
        return self._render(self._settings.paths.score)

    def sql(self, k: int) -> str:
        """outputs/{dataset}/{scale}/{method}/result/turn{max_hop}/sql.{k}.txt"""
        return self._render(self._settings.paths.sql, k=k)


class LoggingConfig(BaseModel):
    """Cấu hình logging — đặt NGAY TẠI ĐÂY, không khai báo trong config.yaml.

    Đổi mức log hay bật/tắt ghi file thì sửa giá trị mặc định bên dưới. Section
    `logging:` vẫn được chấp nhận nếu bạn thêm lại vào config.yaml — khi đó nó ghi
    đè các mặc định này (xem utils/logger.py để biết cách đọc).
    """
    level: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True  # true = ghi thêm ra file (append), vẫn in ra console
    log_dir: str = "outputs/logs"  # chỉ dùng khi log_to_file: true
    log_file: str = "murre.log"


class ApiConfig(BaseModel):
    """Cấu hình API service (FastAPI) — đặt NGAY TẠI ĐÂY, không có trong config.yaml.

    Giống LoggingConfig: sửa trực tiếp giá trị mặc định bên dưới. Thêm lại section
    `api:` vào config.yaml thì section đó ghi đè các mặc định này.
    """
    host: str = "0.0.0.0"
    port: int = 8000
    default_top_n: int = 5  # số bảng trả về tối đa qua /retrieve


class RunOptionConfig(BaseModel):
    """Cách chạy pipeline — đặt NGAY TẠI ĐÂY, không khai báo trong config.yaml.

    Xem HUONG_DAN.md để biết cách chuyển đổi giữa hai option. Thêm lại section
    `run_option:` vào config.yaml thì section đó ghi đè mặc định này.
    """
    # Option 1: "offline" — chạy toàn bộ pipeline trong Python (dùng để debug/học)
    # Option 2: "batch"   — chạy từng bước riêng lẻ qua file steps/*.py (production)
    mode: str = "batch"  # offline | batch


class Settings(BaseModel):
    """Root config object — ánh xạ toàn bộ cấu hình của project.

    Mọi section đều có giá trị mặc định ngay trong file này, trừ `llm`: không thể
    đoán giúp bạn dùng endpoint/model LLM nào. Section nào xuất hiện trong
    config.yaml thì ghi đè mặc định tương ứng; không có cũng vẫn chạy.
    """

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    llm: LLMConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    run_option: RunOptionConfig = Field(default_factory=RunOptionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Toàn bộ config dạng dict thuần, JSON-safe — dùng cho endpoint /config."""
        return self.model_dump(mode="json")

    @property
    def outputs(self) -> OutputPaths:
        """Đường dẫn file trung gian & đầu ra đã điền sẵn tham số — xem OutputPaths.

        `cfg.paths.*` là TEMPLATE còn nguyên {placeholder}; `cfg.outputs.*()` là
        đường dẫn thật đã điền theo general.* / pipeline.*.
        """
        return OutputPaths(self)

    @property
    def dataset_paths(self) -> DatasetPathsConfig:
        """Nhóm đường dẫn đầu vào của `general.dataset` đang chọn.

        Dùng thay cho get_dataset_path() cũ — truy cập bằng thuộc tính nên IDE gợi ý
        được tên file và Ctrl+click nhảy thẳng tới khai báo:

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
    """Cho .env ghi đè config.yaml ở 3 giá trị hay đổi theo từng máy.

    api_key/base_url chỉ ghi đè cho PROFILE ĐANG ACTIVE — các profile khác giữ
    nguyên giá trị đã khai báo, để bạn vẫn chuyển sang model local khác bất cứ lúc nào.
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
    """Tên tất cả profile LLM đã khai báo trong config.yaml (llm.profiles).

    Dùng khi bạn có nhiều model local (Ollama) muốn chuyển đổi qua lại — mỗi model
    khai báo thành 1 profile trong config.yaml, xem HUONG_DAN.md mục 3b.
    """
    return list(cfg.llm.profiles)


def print_llm_profiles() -> None:
    """In các profile LLM đã khai báo, đánh dấu * vào cái đang dùng.

    Chạy: python -m config --llm
    """
    active: str = cfg.llm.active_profile
    print(f"{'=' * 60}\n  LLM PROFILE trong config.yaml\n{'=' * 60}")
    for name in list_llm_profiles():
        mark: str = "*" if name == active else " "
        print(f"  {mark} {name}")
    print(f"{'=' * 60}\n  (* = llm.active_profile đang dùng)")


def get_llm_profile(profile_name: Optional[str] = None) -> LLMProfileConfig:
    """Cấu hình của 1 profile LLM cụ thể (model_name/base_url/api_key/temperature).

    profile_name=None → dùng cfg.llm.active_profile (đặt trong config.yaml).
    Truyền profile_name để tạm dùng 1 model local khác mà không cần sửa config.yaml
    (ví dụ khi muốn so sánh nhiều model local trong cùng một lần chạy/script).
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


# Giá trị ví dụ cho các placeholder KHÔNG suy ra được từ cfg (do chỗ gọi truyền
# vào lúc chạy). Chỉ dùng để in đường dẫn MẪU trong print_paths().
_EXAMPLE_PLACEHOLDERS: Dict[str, Any] = {"hop": 1, "beam": 2, "k": 5}


def print_paths() -> None:
    """In mọi đường dẫn ĐÃ RESOLVE theo cfg hiện tại — để biết file thực sự nằm ở đâu.

    cfg.paths.* chỉ là template còn nguyên {placeholder}. Hàm này in ra bảng đối chiếu
    method → đường dẫn thật, kể cả sau khi đổi general.dataset / general.scale /
    pipeline.method.

    Chạy: python -m config --paths
    """
    header: str = (
        f"dataset={cfg.general.dataset} scale={cfg.general.scale} "
        f"method={cfg.pipeline.method} max_hop={cfg.pipeline.max_hop}"
    )
    print(f"{'=' * 78}\n  ĐƯỜNG DẪN THỰC TẾ — {header}\n{'=' * 78}")

    print(f"\n--- Dữ liệu đầu vào (cfg.dataset_paths → paths.{cfg.general.dataset}) ---")
    ds_paths: DatasetPathsConfig = cfg.dataset_paths
    for name in DatasetPathsConfig.model_fields:
        print(f"  {name:22s} {getattr(ds_paths, name)}")

    print("\n--- File trung gian & đầu ra (cfg.outputs.*) ---")
    # Lặp theo TÊN template nên phải render động — dùng _render() thay vì gọi từng
    # method. Code ngoài config.py thì luôn gọi method: cfg.outputs.result(), ...
    resolver: OutputPaths = cfg.outputs
    for name, template in cfg.paths.model_dump().items():
        # Bỏ qua spider/bird: đó là nhóm đường dẫn đầu vào, đã in ở khối trên.
        if not isinstance(template, str):
            continue

        # Placeholder nào không suy ra được từ cfg thì điền giá trị ví dụ, và nói rõ
        # đó là ví dụ để không ai tưởng đường dẫn cố định như vậy.
        extra: Dict[str, Any] = {
            field: _EXAMPLE_PLACEHOLDERS[field]
            for _, field, _, _ in Formatter().parse(template)
            if field in _EXAMPLE_PLACEHOLDERS
        }
        call: str = f"{name}({', '.join(f'{k}={v}' for k, v in extra.items())})"
        print(f"  {call:34s} {resolver._render(template, **extra)}")

    print("=" * 78)


# Singleton — import ở bất kỳ đâu trong project đều dùng chung một instance,
# nạp một lần duy nhất khi module được import lần đầu.
cfg: Settings = load_settings()


if __name__ == "__main__":
    # Mọi lệnh XEM cấu hình đều nằm ở đây, không nằm trong main.py — main.py chỉ để
    # chạy pipeline.
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
