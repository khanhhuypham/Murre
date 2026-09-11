# =============================================================================
# core/rewriter.py — pha Removal của MURRE (§3.4)
#
# LLM nói ra BẢNG CÒN THIẾU dựa trên câu hỏi gốc + bảng đã tìm được, rồi lấy
# chuỗi đó đi retrieve hop kế. Trả "None" → nhánh đó dừng sớm (Early Stop).
#
#   Câu hỏi: "Which airlines fly to AHD?"
#   Đã có:   flight_2.flights(airline, source, destination)
#   LLM trả: flight_2.airlines(airline id, airline name, country)
# =============================================================================
from __future__ import annotations

from typing import List, Optional

from config import cfg
from core.llm import LLMGenerator
from utils import logger


class QueryRewriter:
    """Pha Removal: (câu hỏi gốc + bảng đã có) → bảng còn thiếu, hoặc Early Stop.

    Ba hằng số dưới đây là tham số của pha Removal, ghi đè được bằng kế thừa.
    """

    # Các mẫu LLM dùng để báo "đã đủ bảng". Khớp từ ĐẦU DÒNG — xem is_early_stop().
    _EARLY_STOP_INDICATORS: List[str] = [
        "There is no",
        "None of the given tables",
        "No additional tables",
        "No completion needed",
        "None",
    ]

    # Nhãn model hay chép lại trước khi trả lời; cắt đi kẻo lẫn vào query.
    _ECHO_PREFIXES: List[str] = ["Completing Tables:", "Rewritten Question:"]

    # Prompt few-shot dạng completion: thiếu stop này thì model sinh tiếp khối
    # "Question:" kế rồi tự bịa và tự trả lời thêm.
    _STOP_SEQUENCES: List[str] = ["\n\n"]

    _MAX_TOKENS: int = 256

    def __init__(self, llm: LLMGenerator, dataset: Optional[str] = None) -> None:
        """dataset=None → prompt của general.dataset đang chọn.

        Prompt phải khớp dataset của corpus, nên API luôn truyền tường minh.
        """
        self.llm: LLMGenerator = llm

        prompt_path: str = cfg.dataset_config(dataset).prompt
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template: str = "\n".join(line.rstrip("\n") for line in f)

        logger.info(f"[QueryRewriter] Đã tải prompt Removal từ: {prompt_path}")

    def rewrite(self, question: str, retrieved_schemas: List[str]) -> str:
        """Gọi LLM dự đoán bảng còn thiếu (Completing Tables) — §3.4.

            question          : CÂU HỎI GỐC, không đổi qua các hop.
            retrieved_schemas : TOÀN BỘ bảng trên đường đi, tích luỹ từ hop 1.

        Trả về chuỗi thô; phán dừng sớm là việc của is_early_stop().
        """
        # Dấu ngăn là 2 ký tự \ và n VIẾT RA, không phải xuống dòng: cả khối
        # `Database:` nằm trên MỘT dòng.
        database_field: str = " \\n ".join(retrieved_schemas)

        prompt: str = self.prompt_template.format(question=question, database=database_field)
        raw_output: str = self.llm.generate(
            prompt=prompt,
            stop=self._STOP_SEQUENCES,
            max_tokens=self._MAX_TOKENS,
        )
        return self._strip_echo(text=raw_output)

    @classmethod
    def is_early_stop(cls, rewrite_output: str) -> bool:
        """Output của LLM có phải tín hiệu dừng sớm không — §3.4.

        Chỉ xét DÒNG ĐẦU và khớp từ đầu dòng, kẻo danh sách bảng hợp lệ có chữ
        "None" ở giữa cũng bị hiểu là dừng sớm.
        """
        cleaned: str = cls._strip_echo(text=rewrite_output)
        lines: List[str] = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        first_line: str = lines[0] if lines else ""
        return any(first_line.startswith(indicator) for indicator in cls._EARLY_STOP_INDICATORS)

    @classmethod
    def _strip_echo(cls, text: str) -> str:
        """Bỏ nhãn prompt mà model chép lại, giữ nguyên phần còn lại (kể cả đa dòng)."""
        out: str = text.strip()
        for prefix in cls._ECHO_PREFIXES:
            if out.startswith(prefix):
                out = out[len(prefix):].lstrip()
        return out
