# =============================================================================
# core/rewriter.py — pha Removal của MURRE (§3.4)
#
# Khác CRUSH ở chỗ: thay vì THÊM bảng đã tìm được vào câu hỏi, MURRE bắt LLM nói ra
# BẢNG CÒN THIẾU dựa trên bảng đã có, rồi lấy chuỗi đó đi retrieve hop kế.
#
#   Câu hỏi: "Which airlines fly to AHD?"
#   Đã có:   flight_2.flights(airline, source, destination)
#   LLM trả: flight_2.airlines(airline id, airline name, country)
#
# LLM trả "None" → đã đủ bảng, nhánh đó dừng sớm (Early Stop).
# =============================================================================

from typing import List

from config import cfg
from core.llm import LLMGenerator
from utils import logger


class QueryRewriter:
    """Pha Removal: (câu hỏi gốc + bảng đã có) → bảng còn thiếu, hoặc mẫu Early Stop.

    Bốn hằng số dưới đây là tham số của pha Removal. Kế thừa class rồi ghi đè chúng
    là đổi được hành vi, không phải sửa thân hàm.
    """

    # Các mẫu LLM dùng để báo "đã đủ bảng". Khớp từ ĐẦU DÒNG — xem is_early_stop().
    _EARLY_STOP_INDICATORS: List[str] = [
        "There is no",
        "None of the given tables",
        "No additional tables",
        "No completion needed",
        "None",
    ]

    # Model hay chép lại nhãn cuối prompt trước khi trả lời; cắt đi kẻo lẫn vào query.
    _ECHO_PREFIXES: List[str] = ["Completing Tables:", "Rewritten Question:"]

    # Prompt là few-shot dạng completion, các ví dụ cách nhau một dòng trống. Thiếu
    # stop này thì model sinh tiếp khối "Question:" kế rồi tự bịa và tự trả lời thêm.
    _STOP_SEQUENCES: List[str] = ["\n\n"]


    def __init__(self, llm: LLMGenerator) -> None:
        self.llm: LLMGenerator = llm
        self.use_tabulation: bool = cfg.pipeline.ablation.tabulation

        ds_paths = cfg.dataset_paths
        prompt_path: str = ds_paths.prompt if self.use_tabulation else ds_paths.prompt_no_tabulation
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template: str = "\n".join(line.rstrip("\n") for line in f)

        mode_desc: str = "Tabulation" if self.use_tabulation else "w/o Tabulation (natural language)"
        logger.info(f"[QueryRewriter] Đã tải prompt ({mode_desc}) từ: {prompt_path}")

    def rewrite(self, question: str, retrieved_schemas: List[str]) -> str:
        """Gọi LLM dự đoán bảng còn thiếu (Completing Tables) — §3.4.

            question          : CÂU HỎI GỐC, không đổi qua các hop.
            retrieved_schemas : TOÀN BỘ bảng trên đường đi, tích luỹ từ hop 1.

        Trả về chuỗi thô của LLM; việc phán có dừng sớm là của is_early_stop().
        """
        # Dấu ngăn là 2 ký tự \ và n VIẾT RA, không phải xuống dòng — cả khối
        # `Database:` nằm trên MỘT dòng. Sửa " \\n " thành " \n " là prompt vỡ ngay.
        database_field: str = " \\n ".join(retrieved_schemas)

        prompt: str = self.prompt_template.format(
            question=question,
            database=database_field,
        )
        raw_output: str = self.llm.generate(
            prompt=prompt,
            stop=self._STOP_SEQUENCES,
            max_tokens=256,
        )
        return self._strip_echo(text=raw_output)

    @classmethod
    def is_early_stop(cls, rewrite_output: str) -> bool:
        """Output của LLM có phải tín hiệu dừng sớm không — §3.4.

        Chỉ xét DÒNG ĐẦU và phải khớp từ đầu dòng: quét cả output bằng `in` thì một
        danh sách bảng hợp lệ có chữ "None" nằm đâu đó ở giữa cũng bị hiểu là dừng
        sớm — model nhỏ dính lỗi này liên tục.
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
