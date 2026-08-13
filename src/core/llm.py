    # =============================================================================
# core/llm.py — Bộ sinh văn bản dùng LLM (OpenAI-compatible)
#
# Hỗ trợ:
#   - OpenAI (gpt-3.5-turbo, gpt-4o-mini, gpt-4o)
#   - Groq (llama-3.1-8b-instant) — miễn phí, cần đổi base_url
#   - Ollama (qwen2.5, llama3.2, ...) — chạy LOCAL, hoàn toàn miễn phí
#   - Bất kỳ API nào tương thích với OpenAI SDK
# =============================================================================
from __future__ import annotations

from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from config import cfg, _Config, get_llm_profile
from utils import logger

class LLMGenerator:
    """
    Lớp bọc gọi LLM thông qua OpenAI SDK.
    Tự động đọc api_key, base_url, model_name từ config.yaml hoặc .env.

    Provider được xác định tự động qua base_url:
      - base_url rỗng / None        → OpenAI chính thức (cần api_key thật)
      - base_url chứa "localhost"
        hoặc "127.0.0.1"            → Ollama local (không cần api_key thật)
      - base_url khác                → API tương thích khác (Groq, v.v.)
    """

    def __init__(self, profile: Optional[str] = None) -> None:
        """profile=None → dùng cfg.llm.active_profile. Truyền tên profile khác để
        tạm dùng 1 model local khác (ví dụ LLMGenerator(profile="qwen2.5-14b"))."""
        profile_cfg: _Config = get_llm_profile(profile_name=profile)

        self.profile_name: str = profile or "active"
        self.model_name: str = profile_cfg.model_name
        base_url: str | None = profile_cfg.base_url if profile_cfg.base_url else None
        self.is_local: bool = bool(base_url) and ("localhost" in base_url or "127.0.0.1" in base_url)
        self.default_temperature: float = profile_cfg.temperature

       # Với Ollama/local: không bắt buộc phải có api_key thật,
       # SDK OpenAI vẫn yêu cầu 1 chuỗi non-empty nên dùng giá trị giả.
        api_key: str = profile_cfg.api_key
        if not api_key:
            if self.is_local:
                api_key = "ollama"  # giá trị giả, Ollama không kiểm tra
            else:
                raise ValueError(
                    f"Profile LLM '{self.model_name}' chưa có api_key và không phải local!\n"
                    "  1. Tạo file .env, thêm: OPENAI_API_KEY=sk-...\n"
                    "  2. Hoặc đổi llm.active_profile trong config.yaml sang 1 profile local (Ollama)."
                )

        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

        provider: str = "Ollama/local" if self.is_local else (
            "OpenAI" if base_url is None else f"Custom endpoint ({base_url})"
        )
        logger.info(f"[LLMGenerator] model='{self.model_name}' | provider={provider}")

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Gửi prompt đến LLM và trả về phản hồi dạng chuỗi.

        Tham số:
            prompt      : nội dung prompt gửi đến LLM
            temperature : nhiệt độ sinh text (None → dùng giá trị trong config)

        Trả về:
            Chuỗi phản hồi từ LLM, đã strip khoảng trắng
        """
        temp: float = temperature if temperature is not None else self.default_temperature

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if self.is_local:
                raise RuntimeError(
                    f"Không gọi được model local '{self.model_name}'.\n"
                    f"  1. Ollama đã chạy chưa? → ollama serve\n"
                    f"  2. Model đã pull chưa? → ollama pull {self.model_name}\n"
                    f"Lỗi gốc: {e}"
                ) from e
            raise
