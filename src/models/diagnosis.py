"""models/diagnosis.py — Chẩn đoán lỗi bằng BẢNG LUẬT, dùng chung cho mọi miền.

CHƯA NỐI VÀO ĐÂU, VÀ CHƯA CÓ MIỀN NÀO DÙNG. File này mới chỉ là cơ chế để xem
trước; core/llm.py, pipeline/runner.py, api/ vẫn chạy nguyên như cũ, không file
nào import ở đây. Bảng luật cho LLM từng viết thử ở core/llm_errors.py nhưng đã
bỏ, nên phần "phần RIÊNG" dưới đây hiện chưa có bản thật nào — ví dụ trong
docstring này là mã minh hoạ, không phải mã đang chạy.

Tách làm hai phần:

    phần CHUNG (file này)   — cơ chế: khớp luật, dựng thông báo, mang phân loại
    phần RIÊNG (từng miền)  — dữ liệu: một enum loại lỗi + một bảng luật

Thêm một miền mới = thêm một enum và một bảng, KHÔNG thêm cơ chế.

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

Cơ chế lo ba việc:

1. Khớp luật theo THỨ TỰ, và tự kiểm thứ tự đó lúc dựng bảng — exception của thư
   viện hay kế thừa lồng nhau, đặt lớp cha trước là lớp con không bao giờ khớp.
2. Mang theo `retryable` để chỗ gọi hỏi thẳng.
3. Dịch sang HTTP một lần (`status` + to_app_error).
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
    """Lỗi đã chẩn đoán: thông báo cho người + phân loại cho máy.

        kind      : loại lỗi theo enum của miền đó
        retryable : thử lại có ích không
        status    : HTTP tương ứng, dùng khi dịch sang AppError
        context   : dữ liệu đã dùng để dựng thông báo
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

        Mặc định đưa nguyên `message` ra client; service mở ra ngoài thì truyền
        `message` ngắn gọn vào đây.
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
    status    : HTTP tương ứng, chỉ dùng khi lỗi ra tới tầng API
    headline  : câu đầu tiên; chuỗi format, điền bằng `context`
    hints     : biến thể gợi ý → chuỗi format; thiếu khóa thì rơi về
                DEFAULT_VARIANT, thiếu nữa thì để trống
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

    name     : tên miền, để thông báo lỗi nói rõ bảng nào sai
    rules    : theo THỨ TỰ ƯU TIÊN — khớp đầu tiên thắng
    fallback : luật cuối, dùng khi không khớp gì (bắt buộc)
    variant  : context → khóa biến thể gợi ý; bỏ trống thì dùng DEFAULT_VARIANT
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

        isinstance khớp cả lớp cha, nên lớp cha đặt trước là lớp con thành code
        chết. Bắt lúc import thay vì để chẩn đoán sai lúc chạy.
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

        `extra` là gợi ý phải đi hỏi mới biết, truyền từ ngoài vào để hàm này
        không tự gọi mạng.
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
    """Có nên thử lại `exc` không — một chỗ quyết định cho mọi vòng lặp retry.

    `default` dùng cho exception chưa qua bảng nào.
    """
    if isinstance(exc, Diagnosis):
        return exc.retryable
    return default


def _names(types: Tuple[Type[BaseException], ...]) -> str:
    return "/".join(t.__name__ for t in types)
