# =============================================================================
# config.py — Module tải và quản lý cấu hình từ config.yaml
#
# Cách dùng trong bất kỳ file nào:
#   from config import cfg, get_path
#
# Sau đó truy cập tham số:
#   cfg.general.dataset       → "spider"
#   cfg.pipeline.beam_size    → 5
#   get_path("embeddings")    → "outputs/spider/125m/embeddings.json"
# =============================================================================
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from dotenv import load_dotenv

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()


class _Config:
    """
    Lớp bọc cho phép truy cập config dạng thuộc tính (dot notation).
    Ví dụ: cfg.pipeline.beam_size thay vì cfg["pipeline"]["beam_size"]
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                # Đệ quy: biến dict lồng nhau thành _Config
                setattr(self, key, _Config(data=value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển ngược _Config về dict thông thường."""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def _load_yaml(path: str = "config.yaml") -> _Config:
    """Đọc file config.yaml và trả về đối tượng _Config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {config_path.resolve()}\n"
            "Hãy chắc chắn bạn đang chạy lệnh từ thư mục gốc của project (MURRE_V2/)."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return _Config(raw)


# Tải config một lần duy nhất khi import module này
cfg: _Config = _load_yaml(path="config.yaml")

def get_path(key: str, **kwargs) -> str:
    """
    Lấy đường dẫn đã được điền sẵn tham số từ config.

    Tham số:
        key    : tên đường dẫn trong config.paths (vd: "embeddings", "turn0")
        kwargs : tham số bổ sung để format chuỗi (vd: hop=1, beam=0, k=5)

    Ví dụ:
        get_path("embeddings")           → "outputs/spider/125m/embeddings.json"
        get_path("turn_n", hop=1, beam=2)→ "outputs/spider/125m/turn1/dev.2.json"
        get_path("sql", hop=3, k=5)      → "outputs/spider/125m/result/turn3/sql.5.txt"
    """
    template: str = getattr(cfg.paths, key)

    # Điền các tham số mặc định từ config.general
    defaults: Dict[str, Any] = {
        "dataset": cfg.general.dataset,
        "scale": cfg.general.scale,
        "method": cfg.pipeline.method,
        "max_hop": cfg.pipeline.max_hop,
    }
    defaults.update(kwargs)

    return template.format(**defaults)

def get_dataset_path(key: str) -> str:
    """
    Lấy đường dẫn file dataset (tables, dev, gold, prompts).

    Ví dụ:
        get_dataset_path("tables") → "dataset/spider/tables.json"
        get_dataset_path("prompts") → "prompts/spider_rewrite.txt"
    """
    dataset_cfg = getattr(cfg.paths, cfg.general.dataset)
    return getattr(dataset_cfg, key)


def list_llm_profiles() -> List[str]:
    """Trả về tên tất cả profile LLM đã khai báo trong config.yaml (llm.profiles).

    Dùng khi bạn có nhiều model local (Ollama) muốn chuyển đổi qua lại — mỗi model
    khai báo thành 1 profile trong config.yaml, xem HUONG_DAN.md mục 4b.
    """
    return list(cfg.llm.profiles.__dict__.keys())

def get_llm_profile(profile_name: Optional[str] = None) -> _Config:
    """Lấy cấu hình của 1 profile LLM cụ thể (model_name/base_url/api_key/temperature).

    profile_name=None → dùng cfg.llm.active_profile (đặt trong config.yaml).
    Truyền profile_name để tạm dùng 1 model local khác mà không cần sửa config.yaml
    (ví dụ khi muốn so sánh nhiều model local trong cùng một lần chạy/script).
    """
    name: str = profile_name or cfg.llm.active_profile
    if not hasattr(cfg.llm.profiles, name):
        raise ValueError(
            f"LLM profile '{name}' không tồn tại trong config.yaml (llm.profiles).\n"
            f"Các profile có sẵn: {list_llm_profiles()}"
        )
    return getattr(cfg.llm.profiles, name)

# --- Override api_key/base_url của PROFILE ĐANG ACTIVE qua biến môi trường (.env) ---
# Chỉ áp dụng cho active_profile — các profile khác trong config.yaml giữ nguyên giá
# trị đã khai báo, để bạn vẫn có thể chuyển sang model local khác bất kỳ lúc nào.
_active_profile: _Config = get_llm_profile()

# Lấy API key: ưu tiên từ .env, sau đó mới đọc từ config.yaml
_api_key_from_env = os.getenv("OPENAI_API_KEY", "")
if _api_key_from_env:
    _active_profile.api_key = _api_key_from_env

_base_url_from_env = os.getenv("OPENAI_BASE_URL", "")
if _base_url_from_env:
    _active_profile.base_url = _base_url_from_env

_encoder_from_env = os.getenv("ENCODER_MODEL_NAME", "")
if _encoder_from_env:
    cfg.encoder.model_name = _encoder_from_env


def print_config() -> None:
    """In toàn bộ cấu hình hiện tại ra terminal để kiểm tra."""
    print(f"{'=' * 60}\n  CẤU HÌNH HIỆN TẠI\n{'=' * 60}")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    print_config()
