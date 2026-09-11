"""models/diagnosis.py — Chẩn đoán lỗi bằng BẢNG LUẬT, dùng chung cho mọi miền.

CHƯA NỐI VÀO ĐÂU, VÀ CHƯA CÓ MIỀN NÀO DÙNG. File này mới chỉ là cơ chế để xem
trước; core/llm.py, pipeline/runner.py, api/ vẫn chạy nguyên như cũ, không file
nào import ở đây. Bảng luật cho LLM từng viết thử ở core/llm_errors.py nhưng đã
bỏ, nên phần "phần RIÊNG" dưới đây hiện chưa có bản thật nào — ví dụ trong
docstring này là mã minh hoạ, không phải mã đang chạy.

Ý TƯỞNG
-------
Ở đâu cũng lặp lại một hình dạng: bắt exception của thư viện bên dưới → đoán xem
nó nghĩa là gì → ráp một thông báo kèm gợi ý → ném lại. Viết tay ở từng chỗ thì
mỗi chỗ một kiểu, và thông tin quan trọng nhất (CÓ ĐÁNG THỬ LẠI KHÔNG) bị mất
ngay lúc ném lại.

Ở đây tách làm hai phần:

    phần CHUNG (file này)   — cơ chế: khớp luật, dựng thông báo, mang theo phân
                              loại. Không biết gì về LLM, dataset hay encoder.
    phần RIÊNG (từng miền)  — chỉ là DỮ LIỆU: một enum loại lỗi + một bảng luật.
                              Xem ví dụ ngay bên dưới.

Thêm một miền mới (nạp dataset, tải encoder từ HuggingFace, chạy SQL...) = thêm
một enum và một bảng, KHÔNG thêm cơ chế.

    class DatasetErrorKind(str, Enum):
        MISSING_FILE = "missing_file"
        ...

    DATASET_ERRORS = RuleTable(
        name="dataset",
        rules=(Rule(exc=FileNotFoundError, kind=..., retryable=False, ...),),
        fallback=Rule(...),
    )

    try:
        ...
    except Exception as e:
        raise DATASET_ERRORS.explain(exc=e, context={...}) from e

BA THỨ CƠ CHẾ NÀY LO
--------------------
1. KHỚP LUẬT THEO THỨ TỰ, và TỰ KIỂM thứ tự đó lúc dựng bảng. Exception của thư
   viện hay kế thừa lồng nhau (APITimeoutError < APIConnectionError), đặt lớp cha
   lên trước là lớp con không bao giờ khớp — sai âm thầm, không có lỗi nào nổ.
   RuleTable.__init__ phát hiện và nổ NGAY lúc import, nên không cần một dòng
   chú thích "ĐỪNG SẮP LẠI" mà vẫn an toàn.
2. MANG THEO `retryable`. Chỗ gọi hỏi thẳng thay vì suy từ tên class exception.
3. DỊCH SANG HTTP một lần (`status` + to_app_error), để tầng API không phải tự
   đoán lỗi lõi tương ứng status nào.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from models.errors import AppError

# Mỗi miền tự khai enum loại lỗi của mình; cơ chế chỉ cần biết "là một Enum".
K = TypeVar("K", bound=Enum)

ExcTypes = Union[Type[BaseException], Tuple[Type[BaseException], ...]]

# Khóa của biến thể mặc định trong Rule.hints — miền nào không chia biến thể thì
# chỉ khai đúng khóa này.
DEFAULT_VARIANT = ""


class Diagnosis(RuntimeError, Generic[K]):
    """Lỗi đã được chẩn đoán: có thông báo cho người, và phân loại cho máy.

    Vẫn là RuntimeError để chỗ nào đang `except RuntimeError` không hỏng khi nối
    vào. Phần thêm:

        kind      : loại lỗi theo enum CỦA MIỀN đó
        retryable : thử lại có ích không — chỗ gọi hỏi cái này, không đoán
        status    : HTTP tương ứng, dùng khi dịch sang AppError
        context   : dữ liệu đã dùng để dựng thông báo, giữ lại để ghi log
    """

    def __init__(
        self,
        message: str,
        *,
        kind: K,
        retryable: bool,
        status: int = 500,
        cause: Optional[BaseException] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.kind: K = kind
        self.retryable: bool = retryable
        self.status: int = status
        self.cause: Optional[BaseException] = cause
        self.context: Dict[str, Any] = dict(context or {})

    def to_app_error(self, message: Optional[str] = None) -> AppError:
        """Sang AppError để tầng API trả đúng HTTP status.

        MẶC ĐỊNH ĐƯA NGUYÊN `message` ra client, vì message ở đây là hướng dẫn
        sửa cấu hình cho chính người vận hành service này. Nếu service mở ra
        ngoài, truyền `message` ngắn gọn vào đây — phần chi tiết vẫn còn trong
        log qua `cause`.
        """
        return AppError(message or self.message, self.status, self.cause)

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Rule(Generic[K]):
    """Một dòng của bảng: exception của thư viện → nghĩa + gợi ý.

    exc       : một class exception, hoặc tuple nhiều class cùng nghĩa
    kind      : loại lỗi (enum của miền)
    retryable : thử lại có ích không
    status    : HTTP tương ứng, chỉ dùng khi lỗi đi ra tới tầng API
    headline  : câu đầu tiên; chuỗi format, điền bằng `context`
    hints     : biến thể gợi ý → chuỗi format. Miền không chia biến thể thì khai
                đúng một khóa DEFAULT_VARIANT. Thiếu khóa đang cần thì rơi về
                DEFAULT_VARIANT, thiếu nữa thì để trống — gợi ý là phần thêm,
                không được phép làm hỏng việc báo lỗi.

    Để dạng DỮ LIỆU (chuỗi format) chứ không phải hàm, để cả bảng đọc được trong
    một màn hình. Phần gợi ý phải đi hỏi mới biết (liệt kê model đang có, lệnh
    pull đúng...) thì truyền qua tham số `extra` của explain(), không nhét vào
    bảng — nhờ vậy bảng không bao giờ tự gọi mạng.
    """

    exc: ExcTypes
    kind: K
    retryable: bool
    headline: str
    hints: Mapping[str, str] = field(default_factory=dict)
    status: int = 500

    @property
    def types(self) -> Tuple[Type[BaseException], ...]:
        return self.exc if isinstance(self.exc, tuple) else (self.exc,)


class UnreachableRule(ValueError):
    """Bảng khai sai thứ tự: có luật không bao giờ khớp được."""


class RuleTable(Generic[K]):
    """Bảng luật của MỘT miền. Dựng ở mức module, dùng lại mãi.

    name     : tên miền, chỉ để thông báo lỗi nói rõ bảng nào sai
    rules    : theo THỨ TỰ ƯU TIÊN — khớp đầu tiên thắng
    fallback : luật cuối, dùng khi không khớp gì. BẮT BUỘC: không có nó thì chỗ
               gọi nhận lại exception chưa phân loại và không có `retryable` để
               hỏi — đúng cái vấn đề bảng này sinh ra để giải.
    variant  : context → khóa biến thể gợi ý (ví dụ "local" / "remote"). Bỏ
               trống thì mọi lỗi dùng DEFAULT_VARIANT.
    """

    def __init__(
        self,
        *,
        name: str,
        rules: Sequence[Rule[K]],
        fallback: Rule[K],
        variant: Optional[Any] = None,
    ) -> None:
        self.name: str = name
        self.rules: Tuple[Rule[K], ...] = tuple(rules)
        self.fallback: Rule[K] = fallback
        self._variant = variant
        self._check_order()

    def _check_order(self) -> None:
        """Nổ nếu có luật đứng sau bị luật đứng trước che hoàn toàn.

        isinstance khớp cả lớp cha, nên đặt lớp cha trước lớp con là lớp con
        thành code chết. Bắt ở đây, lúc import, thay vì để nó lặng lẽ chẩn đoán
        sai suốt đời chương trình.
        """
        for i, earlier in enumerate(self.rules):
            for later in self.rules[i + 1:]:
                shadowed: Iterable[bool] = (
                    any(issubclass(t, parent) for parent in earlier.types)
                    for t in later.types
                )
                if all(shadowed):
                    raise UnreachableRule(
                        f"Bảng luật '{self.name}' sắp sai thứ tự: "
                        f"{_names(later.types)} nằm SAU {_names(earlier.types)} "
                        f"mà lại là lớp con của nó → luật "
                        f"'{later.kind}' không bao giờ khớp.\n"
                        f"  Sửa: đưa {_names(later.types)} lên TRƯỚC "
                        f"{_names(earlier.types)}."
                    )

    def match(self, exc: BaseException) -> Rule[K]:
        """Luật KHỚP ĐẦU TIÊN; không có thì `fallback`."""
        for rule in self.rules:
            if isinstance(exc, rule.types):
                return rule
        return self.fallback

    def explain(
        self,
        exc: BaseException,
        context: Mapping[str, Any],
        extra: str = "",
    ) -> Diagnosis[K]:
        """exception thô → Diagnosis đã phân loại, kèm gợi ý đúng biến thể.

        `extra` là phần gợi ý phải đi HỎI mới biết (liệt kê model đang có, lệnh
        pull đúng với Docker hay không...). Truyền từ ngoài vào để hàm này không
        tự gọi mạng — test gọi được mà không cần server nào.
        """
        rule: Rule[K] = self.match(exc=exc)
        key: str = self._variant(context) if self._variant else DEFAULT_VARIANT
        hint: str = rule.hints.get(key, rule.hints.get(DEFAULT_VARIANT, ""))

        message: str = (
            rule.headline.format(**context)
            + "\n"
            + hint.format(**context)
            + extra
            + f"Lỗi gốc: {exc}"
        )
        return Diagnosis(
            message,
            kind=rule.kind,
            retryable=rule.retryable,
            status=rule.status,
            cause=exc,
            context=context,
        )


def is_retryable(exc: BaseException, default: bool = False) -> bool:
    """Có nên thử lại `exc` không — MỘT chỗ quyết định cho mọi vòng lặp retry.

    `default` là câu trả lời cho exception CHƯA qua bảng nào (lỗi lạ từ thư viện
    khác, bug trong code ta). Vòng lặp nào đang `except Exception` rộng thì để
    default=True cho giống hành vi cũ, rồi siết dần.
    """
    if isinstance(exc, Diagnosis):
        return exc.retryable
    return default


def _names(types: Tuple[Type[BaseException], ...]) -> str:
    return "/".join(t.__name__ for t in types)
