# =============================================================================
# core/llm.py — Bộ sinh văn bản dùng LLM qua SDK OpenAI
#
# Chạy với mọi endpoint tương thích OpenAI: OpenAI, Groq, Ollama local. Provider
# suy ra từ base_url — rỗng là OpenAI, chứa localhost/127.0.0.1 là local.
# =============================================================================
from __future__ import annotations

import socket
import subprocess
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError, APITimeoutError, NotFoundError, OpenAI
from openai.types.chat import ChatCompletion

from config import LLMProfileConfig, get_llm_profile
from utils import logger


class LLMGenerator:
    """Gọi LLM qua SDK OpenAI, đọc api_key/base_url/model_name từ profile trong config.

    Endpoint local (Ollama) được đối xử riêng ở hai chỗ: không cần api_key thật, và
    mọi lỗi đều kèm gợi ý sửa — đó là chỗ hay quên bật server hoặc quên pull model.
    """

    def __init__(self, profile: Optional[str] = None) -> None:
        """profile=None → dùng cfg.llm.active_profile. Truyền tên khác để tạm dùng
        model khác, ví dụ LLMGenerator(profile="qwen2.5-14b")."""
        profile_cfg: LLMProfileConfig = get_llm_profile(profile_name=profile)

        self.model_name: str = profile_cfg.model_name
        self.default_temperature: float = profile_cfg.temperature
        self.connect_timeout: float = profile_cfg.connect_timeout
        self.base_url: Optional[str] = profile_cfg.base_url or None
        self.is_local: bool = bool(self.base_url) and (
            "localhost" in self.base_url or "127.0.0.1" in self.base_url
        )

        # SDK OpenAI đòi api_key non-empty; Ollama không kiểm nên đưa giá trị giả.
        api_key: str = profile_cfg.api_key
        if not api_key:
            if not self.is_local:
                raise ValueError(
                    f"Profile LLM '{self.model_name}' chưa có api_key và không phải local!\n"
                    "  1. Tạo file .env, thêm: OPENAI_API_KEY=sk-...\n"
                    "  2. Hoặc đổi llm.active_profile trong config.yaml sang profile local (Ollama)."
                )
            api_key = "ollama"

        # Timeout PHẢI đặt tay: mặc định của SDK là connect 5s + read 600s, nhân thêm
        # max_retries → đợi rất lâu mới biết endpoint chưa bật.
        self.client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(profile_cfg.timeout, connect=profile_cfg.connect_timeout),
            max_retries=profile_cfg.max_retries,
        )

        # Bật khi đã biết server local không chạy → lần gọi sau lỗi ngay, không chờ
        # timeout lại từng hop (một câu hỏi tốn (max_hop-1) × beam_size lần gọi).
        self._offline: bool = False
        if self.is_local:
            self._check_local_server()

        provider: str = "Ollama/local" if self.is_local else (
            "OpenAI" if self.base_url is None else f"Custom endpoint ({self.base_url})"
        )
        logger.info(
            f"[LLMGenerator] model='{self.model_name}' | provider={provider} | "
            f"timeout={profile_cfg.timeout}s (connect {profile_cfg.connect_timeout}s) | "
            f"max_retries={profile_cfg.max_retries}"
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        max_tokens: int = 100,
    ) -> str:
        """Gửi prompt, trả về phản hồi đã strip khoảng trắng.

            temperature : None → lấy từ profile trong config.yaml.
            stop        : chuỗi cắt output. Prompt few-shot dạng completion BẮT BUỘC
                          phải có, không thì model sinh tiếp cả khối ví dụ kế.
        """
        if self._offline:
            host, port = self._endpoint()
            raise RuntimeError(
                f"Endpoint local {host}:{port} đã xác định là không chạy ở lần gọi "
                f"trước. Bật `ollama serve` rồi chạy lại."
            )

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return response.choices[0].message.content.strip()

        except (APIConnectionError, APITimeoutError) as e:
            # Server tắt giữa đường, hoặc model nạp/sinh lâu hơn timeout.
            if not self.is_local:
                raise
            self._offline = isinstance(e, APIConnectionError)
            host, port = self._endpoint()
            raise RuntimeError(
                f"Mất kết nối tới model local '{self.model_name}' ({host}:{port}).\n"
                f"{self._local_hint()}"
                f"  3. Nếu là timeout: tăng llm.profiles.*.timeout trong config.yaml\n"
                f"Lỗi gốc: {e}"
            ) from e

        except NotFoundError as e:
            # Server SỐNG nhưng không có model này — lỗi cấu hình, không phải lỗi mạng.
            # Endpoint đang chạy nên hỏi luôn nó có sẵn model nào, đỡ phải đi tra.
            if not self.is_local:
                raise
            raise RuntimeError(
                f"Endpoint đang chạy nhưng KHÔNG CÓ model '{self.model_name}'.\n"
                f"{self._available_models_hint()}"
                f"  → Sửa llm.active_profile trong config.yaml sang model có sẵn,\n"
                f"    hoặc tải model này về: {self._pull_command()}\n"
                f"Lỗi gốc: {e}"
            ) from e

        except Exception as e:
            if not self.is_local:
                raise
            raise RuntimeError(
                f"Không gọi được model local '{self.model_name}'.\n"
                f"{self._local_hint()}Lỗi gốc: {e}"
            ) from e

    # -- Endpoint local: kiểm tra trước và gợi ý khi lỗi -----------------------
    def _endpoint(self) -> Tuple[str, int]:
        """(host, port) lấy từ base_url; thiếu port thì lấy 11434 của Ollama."""
        parsed = urlparse(url=self.base_url or "")
        return parsed.hostname or "localhost", parsed.port or 11434

    def _check_local_server(self) -> None:
        """Mở thử TCP tới endpoint local — vài ms, rõ hơn là để request tự timeout."""
        host, port = self._endpoint()
        try:
            with socket.create_connection(address=(host, port), timeout=self.connect_timeout):
                return
        except OSError as e:
            self._offline = True
            raise RuntimeError(
                f"Không kết nối được endpoint local {host}:{port} "
                f"(chờ {self.connect_timeout}s).\n"
                f"{self._local_hint()}Lỗi gốc: {e}"
            ) from e

    def _local_hint(self) -> str:
        """Hai dòng gợi ý dùng chung cho mọi lỗi của endpoint local."""
        return (
            f"  1. Ollama đã chạy chưa? → ollama serve\n"
            f"  2. Model đã pull chưa?  → {self._pull_command()}\n"
        )

    def _available_models_hint(self) -> str:
        """Liệt kê model endpoint local đang có. Hỏi được thì hỏi, không thì thôi."""
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

        Ollama trong Docker thì `ollama pull` trần trên host không có tác dụng
        (thường còn không có lệnh `ollama`) — phải gọi xuyên vào container.
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
