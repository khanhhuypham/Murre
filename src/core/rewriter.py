# =============================================================================
# core/rewriter.py — Bộ viết lại câu hỏi theo thuật toán "Completing Tables"
#
# Đây là thành phần cốt lõi phân biệt MURRE với các phương pháp trước:
#
# Thay vì THÊM thông tin đã retrieve vào câu hỏi (như CRUSH),
# MURRE YÊU CẦU LLM ĐIỀN BẢNG CÒN THIẾU dựa trên bảng đã có.
#
# Ví dụ:
#   Câu hỏi: "Which airlines fly to AHD?"
#   Đã có:   flight_2.flights(airline, source, destination)
#   LLM trả: flight_2.airlines(airline_id, airline_name, country)
#            → dùng chuỗi này để retrieve hop tiếp theo
#
# Nếu LLM trả "None" → đã đủ bảng, dừng sớm (Early Stop)
# =============================================================================

from typing import List
from utils import logger
from config import get_dataset_path, cfg
from core.llm import LLMGenerator

"""
Các chuỗi báo hiệu dừng sớm:
    "None"                    → không còn bảng nào cần tìm
    "There is no"             → không có bảng liên quan
    "None of the given tables"→ tất cả bảng đã đủ
    "No additional tables"    → không cần thêm bảng
    "No completion needed"    → không cần hoàn thành thêm
"""
_EARLY_STOP_INDICATORS: List[str] = [
    "There is no",
    "None of the given tables",
    "No additional tables",
    "No completion needed",
    "None",
]
def _judge_early_stop(text: str) -> bool:
    return any(indicator in text for indicator in _EARLY_STOP_INDICATORS)


class QueryRewriter:
    """
    Thực hiện bước Removal/Completing Tables của MURRE.

    Nhận vào:
        - Câu hỏi gốc
        - Danh sách schema bảng đã retrieve được (selected_database)

    Trả về:
        - Chuỗi schema bảng còn thiếu (dùng để retrieve hop tiếp)
        - Hoặc "None" nếu đã đủ bảng (Early Stop)
    """

    def __init__(self, llm: LLMGenerator) -> None:
        self.llm: LLMGenerator = llm
        self.use_tabulation: bool = cfg.pipeline.ablation.tabulation

        # Đọc file prompts few-shot từ đường dẫn trong config
        prompt_key: str = "prompt" if self.use_tabulation else "prompt_no_tabulation"
        prompt_path: str = get_dataset_path(key=prompt_key)
        with open(prompt_path, "r", encoding="utf-8") as f:
            # Giữ nguyên format của tác giả: join các dòng lại
            self.prompt_template:str = "\n".join(line.rstrip("\n") for line in f)

        mode_desc: str = "Tabulation" if self.use_tabulation else "w/o Tabulation (natural language)"
        logger.info(f"[QueryRewriter] Đã tải prompt ({mode_desc}) từ: {prompt_path}")

    @staticmethod
    def _build_database_field(schemas: List[str]) -> str:
        """
        Ghép các schema đã retrieve thành chuỗi database field.
        Format của tác giả: "db.table1(col...) \\n db.table2(col...)"
        """
        if len(schemas) == 1:
            return schemas[0]
        # Dùng " \\n " làm dấu phân cách giữa nhiều bảng (theo format tác giả)
        return " \\n ".join(schemas)

    def rewrite(self, question: str, retrieved_schemas: List[str]) -> str:
        """
        Gọi LLM để dự đoán bảng còn thiếu (Completing Tables).

        Tham số:
            question          : câu hỏi gốc của người dùng
            retrieved_schemas : danh sách schema bảng đã tìm được ở hop trước

        Trả về:
            Chuỗi schema bảng còn thiếu, hoặc "None" nếu đủ rồi
        """
        database_field:str = self._build_database_field(schemas=retrieved_schemas)

        # Điền câu hỏi và database vào template few-shot
        prompt: str = self.prompt_template.format(question=question, database=database_field)

        return self.llm.generate(prompt)

    @staticmethod
    def is_early_stop(rewrite_output: str) -> bool:
        """Kiểm tra output của LLM có phải tín hiệu dừng sớm không."""
        return _judge_early_stop(text=rewrite_output)
