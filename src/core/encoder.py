"""core/encoder.py — Bi-Encoder: văn bản → vector, dùng cho retrieval (§3.2).

Một họ encoder, chọn bằng `type` của profile trong config.yaml:

    sentence  Bi-encoder kiểu sentence-transformers (BERT/XLM-R): mean pooling
              có mask, phân biệt vai trò bằng TIỀN TỐ văn bản nếu model đòi
              (E5: "query: " / "passage: ").

Nhánh này dùng MỘT model đa ngữ cho mọi dataset. SGPT của paper (SPECB bọc chuỗi
token bằng "[ ]" / "{ }" + weighted-mean pooling) đã bỏ khỏi nhánh production:
nó chỉ học tiếng Anh, nên corpus tiếng Việt (ViText2SQL) bị BPE tiếng Anh băm ra
82 token mỗi schema so với 24 token của bản tiếng Anh — vừa chậm gấp mấy lần vừa
mất nghĩa (r@5 = 5.6 so với 82.2 của multilingual-e5-base). Cần đối chiếu với
Bảng 2 của paper thì xem lịch sử git, không dựng lại ở đây.

`type` giữ lại dù chỉ còn một giá trị, và EncoderProfileConfig vẫn CHẶN mọi giá
trị khác ngay lúc nạp config.yaml: config cũ khai `type: sgpt` phải nổ lỗi chứ
không được lặng lẽ rơi về encoder này — không ai muốn chạy cả lượt rồi mới đoán
mình đã dùng model nào.

Ba hàm nằm NGOÀI class (_load, plan_batches, _encode_batched) là có chủ ý, không
phải sót lại: cả ba không đọc `self` gì cả, nên test gọi được thẳng bằng dữ liệu
bịa — plan_batches bằng một list số nguyên, _encode_batched bằng một `forward`
giả. Đó là cách duy nhất kiểm tra được việc XẾP LẠI THỨ TỰ mà không cần model
thật, và lỗi ở chỗ đó thì không nổ ra gì cả (xem _encode_batched). Phần nào thực
sự cần tokenizer của instance thì nằm trong class (_token_lengths).

    encoder = SentenceEncoder.get(dataset="vitext2sql")   # dùng lại, không nạp lại
    vectors = encoder.encode(texts=[...], is_query=False)   # (n, hidden_dim)
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Sequence

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import EncoderProfileConfig, cfg
from utils import logger


class SentenceEncoder:
    """Bi-encoder kiểu sentence-transformers: mean pooling + tiền tố theo vai trò.

    Encoder duy nhất của nhánh này, dùng cho mọi dataset — model đa ngữ
    (intfloat/multilingual-e5-*, BAAI/bge-m3, ...) xử lý được cả tiếng Anh
    (spider, bird) và tiếng Việt (vitext2sql).

    Hai chỗ bắt buộc phải đúng thì vector mới có nghĩa:
      - Gộp token bằng MEAN CÓ MASK: token padding không được kéo lệch trung bình.
      - Phân biệt query/document bằng TIỀN TỐ VĂN BẢN. E5 được huấn luyện với
        "query: " và "passage: "; thiếu tiền tố thì điểm tụt hẳn mà không báo gì.
        Khai ở <profile>.query_prefix / <profile>.doc_prefix.
    """

    # Một instance cho mỗi PROFILE, dựng lúc cần chứ không lúc import. Khóa theo
    # tên profile (`datasets.<ds>.encoder`) chứ không theo dataset: cả ba dataset
    # hiện cùng trỏ `multilingual`, nên thực tế chỉ có ĐÚNG MỘT SentenceEncoder
    # trong tiến trình (~1.1GB) — thêm một profile tiếng Việt riêng thì tự tách.
    _instances: Dict[str, SentenceEncoder] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, dataset: Optional[str] = None) -> SentenceEncoder:
        """Encoder CỦA `dataset`, dùng lại nếu đã dựng — LỐI VÀO của mọi module.

        dataset=None → dataset đang chọn (general.dataset). Profile tra qua
        cfg.encoder_for(), tức là qua `datasets.<ds>.encoder` → `encoders.<tên>`.
        Gọi hai lần cùng một profile trả về CÙNG MỘT object.

        Dựng lúc gọi chứ KHÔNG sẵn ở mức module: `SentenceEncoder(...)` đặt ngoài
        class sẽ chạy _load() ngay lúc `import core.encoder`, mà core/corpus.py
        import file này CHỈ để annotate. Khi đó `python -m cli config` (in JSON)
        cũng phải tải model, server.ensure_port_free() mất tác dụng vì model đã
        nạp xong trước khi nó kịp thử cổng, và test không kịp monkeypatch _load.

        Gọi thẳng `SentenceEncoder(profile=...)` vẫn được — dựng bản RIÊNG, không
        đụng cache. Chỉ nên dùng khi thật sự cần hai bản khác cấu hình.
        """
        profile_name: str = cfg.dataset_config(dataset).encoder

        # Giữ khóa suốt lúc nạp model (có thể vài phút) là CỐ Ý: hai luồng cùng
        # hỏi một profile thì luồng sau đợi rồi dùng chung, chứ không nạp thêm
        # một bản 1.1GB nữa vào RAM.
        with cls._lock:
            encoder: Optional[SentenceEncoder] = cls._instances.get(profile_name)
            if encoder is None:
                # encoder_for() mới là chỗ nổ lỗi khi profile không tồn tại.
                encoder = cls(profile=cfg.encoder_for(dataset))
                cls._instances[profile_name] = encoder
            return encoder

    @classmethod
    def reset_cache(cls) -> None:
        """Xoá cache instance — CHỈ dùng cho test.

        Cache sống hết đời tiến trình, nên test nào đếm số lần nạp model phải gọi
        cái này trước, không thì nó đo nhờ kết quả của test chạy trước đó.
        """
        with cls._lock:
            cls._instances.clear()

    def __init__(self, profile: EncoderProfileConfig) -> None:
        self.model_name: str = profile.model_name
        self.batch_size: int = profile.batch_size
        self.max_batch_tokens: int = profile.max_batch_tokens
        self.max_length: int = profile.max_length
        self.query_prefix: str = profile.query_prefix
        self.doc_prefix: str = profile.doc_prefix
        self.tokenizer, self.model, self.device = _load(model_name=self.model_name)

    @staticmethod
    def _mean_pooling(
        tokens: Dict[str, torch.Tensor], hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
            Trung bình các token embedding, BỎ QUA token padding.
            embedding = Σ(hidden_i × mask_i) / Σ(mask_i)
        """
        mask: torch.Tensor = (
            tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        )
        summed: torch.Tensor = torch.sum(hidden * mask, dim=1)
        # clamp: một chuỗi rỗng hoàn toàn sẽ chia cho 0 → NaN lan ra cả ma trận điểm.
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _token_lengths(self, texts: List[str]) -> List[int]:
        """Số token của từng văn bản, chưa padding — chỉ để chia batch.

        Không truncate theo `max_length` ở đây: mục đích là ĐO để plan_batches()
        biết câu nào dài, còn cắt thật thì làm ở encode(). Đo dài hơn thực tế chỉ
        khiến batch nhỏ hơn cần thiết, không sai kết quả.
        """
        if not texts:
            return []
        encoded = self.tokenizer(text=texts, padding=False, truncation=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Mã hóa danh sách văn bản → tensor (len(texts), hidden_dim) trên CPU."""
        prefix: str = self.query_prefix if is_query else self.doc_prefix
        prefixed: List[str] = [prefix + t for t in texts] if prefix else list(texts)

        def run(batch: List[str]) -> torch.Tensor:
            tokens: Dict[str, torch.Tensor] = self.tokenizer(
                text=batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                hidden: torch.Tensor = self.model(**tokens).last_hidden_state
            return self._mean_pooling(tokens=tokens, hidden=hidden)

        return _encode_batched(
            texts=prefixed,
            lengths=self._token_lengths(texts=prefixed),
            max_items=self.batch_size,
            max_tokens=self.max_batch_tokens,
            forward=run,
        )



def _load(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
    """Tải tokenizer + model, đặt lên đúng thiết bị và chuyển sang chế độ eval."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Encoder] Đang tải '{model_name}' trên {device} ...")
    logger.info("[Encoder] Lần đầu tiên sẽ tải model từ HuggingFace (~vài phút) ...")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model.to(device)
    model.eval()  # Tắt dropout để kết quả ổn định
    logger.info(f"[Encoder] Tải xong '{model_name}'.")
    return tokenizer, model, device


def plan_batches(
    lengths: Sequence[int], max_items: int, max_tokens: int,
) -> List[List[int]]:
    """Chia chỉ số thành các batch theo NGÂN SÁCH TOKEN, không theo số lượng cố định.

    Đây là chỗ từng làm sập tiến trình. Bộ nhớ attention tăng theo
    `số_câu × độ_dài_nhất²`, mà `batch_size` chỉ đếm SỐ CÂU — một schema dài bất
    thường sẽ kéo cả batch phải pad tới độ dài của nó. Đo thật trên BIRD: schema
    dài nhất 598 token, batch 256 câu → riêng một tensor attention đã ~4.4 GB,
    RSS đỉnh 8.4 GB, và trên máy 16 GB thì torch nổ ACCESS_VIOLATION (0xC0000005)
    ngay trong code C, không phải MemoryError của Python.

    Hai việc ở đây, cùng giải quyết chuyện đó:
      1. Sắp theo độ dài rồi mới gom → câu ngắn nằm cùng nhau, gần như không pad.
      2. Cắt batch khi `số_câu × độ_dài_nhất` vượt `max_tokens` → độ dài càng lớn
         thì batch càng nhỏ, đỉnh bộ nhớ không còn phụ thuộc câu dài nhất corpus.

    Trả về danh sách batch, mỗi batch là các CHỈ SỐ trong `lengths` (chỗ gọi phải
    tự xếp kết quả về đúng vị trí cũ — xem _encode_batched).
    """
    order: List[int] = sorted(range(len(lengths)), key=lambda i: lengths[i])

    batches: List[List[int]] = []
    current: List[int] = []
    for idx in order:
        # order tăng dần nên `idx` luôn là câu dài nhất của batch nếu thêm vào.
        longest: int = max(lengths[idx], 1)
        if current and (
            len(current) + 1 > max_items or (len(current) + 1) * longest > max_tokens
        ):
            batches.append(current)
            current = []
        current.append(idx)

    if current:
        batches.append(current)
    return batches

def _encode_batched(
    texts: List[str],
    lengths: List[int],
    max_items: int,
    max_tokens: int,
    forward: Callable[[List[str]], torch.Tensor],
) -> torch.Tensor:
    """Chạy `forward` theo từng batch rồi XẾP LẠI ĐÚNG THỨ TỰ đầu vào.

    plan_batches() sắp lại theo độ dài để bớt padding, nên kết quả trả về so le so
    với `texts`. Xếp lại là bắt buộc: embs[i] phải là vector của corpus[i], lệch
    một chỗ là điểm số gán nhầm bảng trong toàn bộ corpus mà không có lỗi nào.
    """
    if not texts:
        return torch.empty(0, 0)

    batches: List[List[int]] = plan_batches(
        lengths=lengths, max_items=max_items, max_tokens=max_tokens,
    )

    parts: List[torch.Tensor] = []
    positions: List[int] = []
    for batch in batches:
        parts.append(forward([texts[i] for i in batch]).cpu())  # CPU: tiết kiệm VRAM
        positions.extend(batch)

    stacked: torch.Tensor = torch.cat(parts, dim=0)
    out: torch.Tensor = torch.empty_like(stacked)
    out[torch.tensor(positions, dtype=torch.long)] = stacked
    return out