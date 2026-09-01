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
from config import cfg
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

# Model hay chép lại nhãn cuối prompt trước khi trả lời ("Completing Tables: ...").
# Cắt đi để phần còn lại là đúng nội dung Removal, không lẫn nhãn vào query retrieve.
_ECHO_PREFIXES: List[str] = ["Completing Tables:", "Rewritten Question:"]


def _strip_echo(text: str) -> str:
    """Bỏ nhãn prompt mà model chép lại, giữ nguyên phần còn lại (kể cả đa dòng)."""
    out: str = text.strip()
    for prefix in _ECHO_PREFIXES:
        if out.startswith(prefix):
            out = out[len(prefix):].lstrip()
    return out


def _judge_early_stop(text: str) -> bool:
    """Chỉ xét DÒNG ĐẦU của output, và phải khớp từ đầu dòng.

    §3.4 của paper: LLM sinh một "special mark" (ví dụ "None") để báo đã đủ bảng.
    Quét cả output bằng `in` thì một danh sách bảng hợp lệ có chữ "None" nằm đâu đó
    ở giữa cũng bị hiểu là dừng sớm — model nhỏ dính lỗi này liên tục.
    """
    first: str = next((ln.strip() for ln in _strip_echo(text).splitlines() if ln.strip()), "")
    return any(first.startswith(indicator) for indicator in _EARLY_STOP_INDICATORS)


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
        ds_paths = cfg.dataset_paths
        prompt_path: str = ds_paths.prompt if self.use_tabulation else ds_paths.prompt_no_tabulation
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template:str = "\n".join(line.rstrip("\n") for line in f)

        mode_desc: str = "Tabulation" if self.use_tabulation else "w/o Tabulation (natural language)"
        logger.info(f"[QueryRewriter] Đã tải prompt ({mode_desc}) từ: {prompt_path}")

    @staticmethod
    def _build_database_field(schemas: List[str]) -> str:
        """
        Ghép các schema đã retrieve thành trường `Database:` của prompt.

        Dấu ngăn là chuỗi " \\n " — dấu gạch chéo + chữ n VIẾT RA, không phải ký tự
        xuống dòng. Cả prompt gốc lẫn rewrite/rewrite.py của tác giả đều vậy:

            Database: car_1.model list(model id, maker, model) \\n car_1.cars data(...)

        Table 6 của paper trình bày chỗ này thành nhiều dòng chỉ vì khổ giấy — đừng
        sửa theo, cả khối `Database:` nằm trên MỘT dòng.
        """
        if len(schemas) == 1:
            return schemas[0]
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

        # Prompt kết thúc bằng nhãn "Completing Tables:" — model chỉ được điền nốt MỘT
        # khối rồi ngừng. Các ví dụ few-shot cách nhau bằng một dòng trống nên "\n\n"
        # là ranh giới; đúng bằng config/35turbo.json của tác giả.
        return _strip_echo(
            self.llm.generate(prompt, stop=["\n\n"], max_tokens=256)
        )

    @staticmethod
    def is_early_stop(rewrite_output: str) -> bool:
        """Kiểm tra output của LLM có phải tín hiệu dừng sớm không."""
        return _judge_early_stop(text=rewrite_output)
