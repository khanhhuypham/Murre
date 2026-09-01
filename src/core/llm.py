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

import socket
import subprocess
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError, APITimeoutError, NotFoundError, OpenAI
from openai.types.chat import ChatCompletion

from config import LLMProfileConfig, cfg, get_llm_profile
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
        profile_cfg: LLMProfileConfig = get_llm_profile(profile_name=profile)

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

        # Timeout PHẢI đặt tay: mặc định của SDK OpenAI là connect 5s + read 600s,
        # nhân thêm max_retries → đợi rất lâu mới biết endpoint chưa bật.
        self.base_url: Optional[str] = base_url
        self.connect_timeout: float = profile_cfg.connect_timeout
        self.client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(profile_cfg.timeout, connect=profile_cfg.connect_timeout),
            max_retries=profile_cfg.max_retries,
        )
        # Bật khi đã biết server local không chạy → các lần gọi sau lỗi ngay, không
        # chờ timeout lại từng hop (mỗi câu hỏi tốn beam_size × max_hop lần gọi).
        self._offline: bool = False
        # Kiểm tra ngay lúc dựng: chưa bật Ollama thì báo trước khi pipeline chạy.
        if self.is_local:
            self._check_local_server()

        provider: str = "Ollama/local" if self.is_local else (
            "OpenAI" if base_url is None else f"Custom endpoint ({base_url})"
        )
        logger.info(
            f"[LLMGenerator] model='{self.model_name}' | provider={provider} | "
            f"timeout={profile_cfg.timeout}s (connect {profile_cfg.connect_timeout}s) | "
            f"max_retries={profile_cfg.max_retries}"
        )

    # -- Kiểm tra endpoint local ---------------------------------------------
    def _endpoint(self) -> Tuple[str, int]:
        """(host, port) lấy từ base_url; thiếu port thì lấy 11434 của Ollama."""
        parsed = urlparse(url=self.base_url or "")
        return parsed.hostname or "localhost", parsed.port or 11434

    def _check_local_server(self) -> None:
        """Thử mở TCP tới endpoint local, không kết nối được thì raise ngay.

        Mở rồi đóng socket chỉ mất vài ms — nhanh và rõ ràng hơn là để request
        HTTP tự timeout khi Ollama chưa bật.
        """
        host, port = self._endpoint()
        try:
            with socket.create_connection(address=(host, port), timeout=self.connect_timeout):
                return
        except OSError as e:
            self._offline = True
            raise RuntimeError(
                f"Không kết nối được endpoint local {host}:{port} "
                f"(chờ {self.connect_timeout}s).\n"
                f"  1. Ollama đã chạy chưa? → ollama serve\n"
                f"  2. Model đã pull chưa?  → ollama pull {self.model_name}\n"
                f"Lỗi gốc: {e}"
            ) from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        max_tokens: int = 100,
    ) -> str:
        """
        Gửi prompt đến LLM và trả về phản hồi dạng chuỗi.

        Tham số:
            prompt      : nội dung prompt gửi đến LLM
            temperature : nhiệt độ sinh text (None → dùng giá trị trong config)
            stop        : chuỗi dừng. Prompt của paper là few-shot dạng COMPLETION
                          ("Question: ... / Database: ... / Completing Tables:"), model
                          phải điền nốt MỘT khối rồi ngừng. Không truyền stop thì model
                          instruct sẽ sinh tiếp sang khối "Question:" kế và tự trả lời
                          lại chính các ví dụ few-shot.
            max_tokens  : trần token sinh ra.

        Trả về:
            Chuỗi phản hồi từ LLM, đã strip khoảng trắng
        """
        temp: float = temperature if temperature is not None else self.default_temperature

        if self.is_local:
            host, port = self._endpoint()
            if self._offline:
                raise RuntimeError(
                    f"Endpoint local {host}:{port} đã xác định là không chạy ở lần gọi "
                    f"trước. Bật `ollama serve` rồi chạy lại."
                )
            self._check_local_server()

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                stop=stop,
            )
            return response.choices[0].message.content.strip()

        except (APIConnectionError, APITimeoutError) as e:
            # Server tắt giữa đường, hoặc model nạp/sinh lâu hơn timeout.
            if self.is_local:
                self._offline = isinstance(e, APIConnectionError)
                host, port = self._endpoint()
                raise RuntimeError(
                    f"Mất kết nối tới model local '{self.model_name}' ({host}:{port}).\n"
                    f"  1. Ollama đã chạy chưa? → ollama serve\n"
                    f"  2. Model đã pull chưa?  → ollama pull {self.model_name}\n"
                    f"  3. Nếu là timeout: tăng llm.profiles.*.timeout trong config.yaml\n"
                    f"Lỗi gốc: {e}"
                ) from e
            raise

        except NotFoundError as e:
            # Server SỐNG nhưng không có model này — lỗi cấu hình, không phải lỗi mạng.
            # Endpoint đang chạy nên hỏi luôn nó có sẵn model nào, đỡ phải đi tra.
            if self.is_local:
                raise RuntimeError(
                    f"Endpoint đang chạy nhưng KHÔNG CÓ model '{self.model_name}'.\n"
                    f"{self._available_models_hint()}"
                    f"  → Sửa llm.active_profile trong config.yaml sang một model có sẵn,\n"
                    f"    hoặc tải model này về: {self._pull_command()}\n"
                    f"Lỗi gốc: {e}"
                ) from e
            raise

        except Exception as e:
            if self.is_local:
                raise RuntimeError(
                    f"Không gọi được model local '{self.model_name}'.\n"
                    f"  1. Ollama đã chạy chưa? → ollama serve\n"
                    f"  2. Model đã pull chưa? → {self._pull_command()}\n"
                    f"Lỗi gốc: {e}"
                ) from e
            raise

    # -- Gợi ý khi cấu hình sai model -----------------------------------------
    def _available_models_hint(self) -> str:
        """Liệt kê model mà endpoint local đang có. Hỏi được thì hỏi, không thì thôi."""
        host, port = self._endpoint()
        try:
            resp = httpx.get(f"http://{host}:{port}/api/tags", timeout=3.0)
            names: List[str] = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return ""  # không hỏi được thì im, đừng làm nhiễu lỗi gốc

        if not names:
            return "  Endpoint chưa có model nào.\n"
        return "  Model đang có: " + ", ".join(sorted(names)) + "\n"

    def _pull_command(self) -> str:
        """Lệnh pull ĐÚNG với cách Ollama đang chạy.

        Ollama chạy trong Docker thì `ollama pull` trần trên máy host không có tác
        dụng (thường còn không có lệnh `ollama`) — phải gọi xuyên vào container.
        """
        try:
            out = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=ollama/ollama",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=3.0,
            ).stdout.strip()
        except Exception:
            out = ""

        container: str = out.splitlines()[0].strip() if out else ""
        if container:
            return f"docker exec {container} ollama pull {self.model_name}"
        return f"ollama pull {self.model_name}"
