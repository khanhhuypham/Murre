"""core/encoder.py — Bi-Encoder: văn bản → vector, dùng cho retrieval (§3.2).

Hai họ encoder, chọn bằng `encoder.type` trong config.yaml:

    sgpt      SGPT của paper. Phân biệt vai trò input bằng cặp ngoặc SPECB bọc
              quanh chuỗi token — query "[ ... ]", document "{ ... }" — rồi
              weighted-mean pooling (token càng về sau càng nặng).

    sentence  Bi-encoder kiểu sentence-transformers (BERT/XLM-R): mean pooling
              có mask, phân biệt vai trò bằng TIỀN TỐ văn bản nếu model đòi
              (E5: "query: " / "passage: ").

Vì sao cần loại thứ hai: SGPT chỉ học tiếng Anh. Corpus tiếng Việt (ViText2SQL)
bị BPE tiếng Anh băm ra 82 token mỗi schema so với 24 token của bản tiếng Anh —
vừa chậm gấp mấy lần vừa mất nghĩa. Model đa ngữ (multilingual-e5) tokenize cùng
nội dung đó hết 32 token và thực sự có biểu diễn cho tiếng Việt.

Cả hai lớp có chung giao diện, chỗ gọi không cần biết đang dùng loại nào:

    encoder = build_encoder(dataset="vitext2sql")
    vectors = encoder.encode(texts=[...], is_query=False)   # (n, hidden_dim)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import EncoderProfileConfig, cfg
from utils import logger


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


def _token_lengths(tokenizer: PreTrainedTokenizerBase, texts: List[str]) -> List[int]:
    """Số token của từng văn bản, chưa padding — chỉ để chia batch."""
    if not texts:
        return []
    return [len(ids) for ids in tokenizer(text=texts, padding=False, truncation=True)["input_ids"]]


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


class SGPTEncoder:
    """Mã hoá câu hỏi / schema bảng thành vector, theo một profile trong `encoders`."""

    def __init__(self, profile: EncoderProfileConfig) -> None:
        self.model_name: str = profile.model_name
        self.batch_size: int = profile.batch_size
        self.max_batch_tokens: int = profile.max_batch_tokens
        self.tokenizer, self.model, self.device = _load(model_name=self.model_name)

        # ─── Token đặc biệt SPECB (SPEcial Character Bracketing) ───────────────────
        # Bọc mỗi input bằng 1 token đầu (BOS) + 1 token cuối (EOS) khác nhau cho
        # query và document, nhờ vậy cùng một nội dung vẫn ra hai vector hơi lệch
        # nhau và cosine similarity query-vs-document chính xác hơn.
        # Ở đây chỉ lấy ID; việc chèn vào chuỗi diễn ra trong _tokenize().
        self.SPECB_QUE_BOS: int = self._encode_single_char_as_token_id(char="[")
        self.SPECB_QUE_EOS: int = self._encode_single_char_as_token_id(char="]")

        self.SPECB_DOC_BOS: int = self._encode_single_char_as_token_id(char="{")
        self.SPECB_DOC_EOS: int = self._encode_single_char_as_token_id(char="}")


    # --------------------------------------------------------------------------
    # Các phương thức nội bộ
    # --------------------------------------------------------------------------
    def _encode_single_char_as_token_id(self, char: str) -> int:
        """ID token của một ký tự đơn, dùng cho SPECB.

        add_special_tokens=False để không lẫn token mặc định của model ([CLS], [SEP]).
        """
        token_ids: List[int] = self.tokenizer.encode(text=char, add_special_tokens=False)

        # BPE/WordPiece có thể tách 1 ký tự thành nhiều token — như vậy là hỏng logic
        # SPECB, nên chặn ngay tại đây.
        assert len(token_ids) == 1, (
            f"Ký tự '{char}' bị tokenizer tách thành {len(token_ids)} token "
            f"({token_ids}), không thể dùng làm token SPECB (yêu cầu đúng 1 token)."
        )
        return token_ids[0]

    def _tokenize(self, texts: List[str], is_query: bool) -> Dict[str, torch.Tensor]:
        """Tokenize rồi bọc SPECB: is_query=True → [ ], False → { }."""
        # Chưa padding ở bước này để còn chèn được token vào đầu/cuối từng chuỗi.
        batch: Dict[str, List[List[int]]] = self.tokenizer(
            text=texts, padding=False, truncation=True
        )

        bos: int = self.SPECB_QUE_BOS if is_query else self.SPECB_DOC_BOS
        eos: int = self.SPECB_QUE_EOS if is_query else self.SPECB_DOC_EOS

        for ids, att in zip(batch["input_ids"], batch["attention_mask"]):
            ids.insert(0, bos)
            ids.append(eos)
            att.insert(0, 1)  # mask = 1 → model chú ý vào token vừa chèn
            att.append(1)

        # Giờ mới pad cả batch về cùng độ dài và chuyển sang tensor.
        return self.tokenizer.pad(encoded_inputs=batch, padding=True, return_tensors="pt")

    @staticmethod
    def _weighted_mean_pooling(tokens: Dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        """Gộp token embedding thành 1 vector, token càng về sau càng nặng.

            embedding = Σ(hidden_i × mask_i × i) / Σ(mask_i × i)
        """
        seq_len:int = hidden.shape[1]

        # Trọng số = vị trí token (1, 2, 3, ...)
        weights:torch.Tensor = (
            torch.arange(1, seq_len + 1)
            .unsqueeze(0)        # (1, seq_len)
            .unsqueeze(-1)       # (1, seq_len, 1)
            .expand(hidden.size())  # (batch, seq_len, hidden_dim)
            .float()
            .to(hidden.device)
        )

        # Mask để bỏ qua token padding
        mask:torch.Tensor = (
            tokens["attention_mask"]
            .unsqueeze(-1)       # (batch, seq_len, 1)
            .expand(hidden.size())
            .float()
            .to(hidden.device)
        )

        summed: torch.Tensor = torch.sum(hidden * mask * weights, dim=1)
        norm: torch.Tensor = torch.sum(mask * weights, dim=1)
        return summed / norm

    # --------------------------------------------------------------------------
    # API công khai
    # --------------------------------------------------------------------------
    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Mã hóa danh sách văn bản → tensor (len(texts), hidden_dim) trên CPU.

        is_query : True = câu hỏi (bọc [ ]), False = schema bảng (bọc { }).
        Chia batch theo ngân sách token — xem plan_batches().
        """
        def run(batch: List[str]) -> torch.Tensor:
            tokens: Dict[str, torch.Tensor] = self._tokenize(texts=batch, is_query=is_query)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                hidden: torch.Tensor = self.model(**tokens).last_hidden_state
            return self._weighted_mean_pooling(tokens=tokens, hidden=hidden)

        return _encode_batched(
            texts=texts,
            lengths=_token_lengths(tokenizer=self.tokenizer, texts=texts),
            max_items=self.batch_size,
            max_tokens=self.max_batch_tokens,
            forward=run,
        )


class SentenceEncoder:
    """Bi-encoder kiểu sentence-transformers: mean pooling + tiền tố theo vai trò.

    Dùng cho mọi model KHÔNG phải SGPT — cụ thể là các model đa ngữ cần thiết cho
    corpus tiếng Việt (intfloat/multilingual-e5-*, BAAI/bge-m3, ...).

    Khác SGPT ở hai chỗ, và cả hai đều bắt buộc phải đúng thì vector mới có nghĩa:
      - Gộp token bằng MEAN có mask, không phải weighted-mean theo vị trí.
      - Phân biệt query/document bằng TIỀN TỐ VĂN BẢN, không phải token SPECB.
        E5 được huấn luyện với "query: " và "passage: "; thiếu tiền tố thì điểm
        tụt hẳn. Khai ở encoder.query_prefix / encoder.doc_prefix.
    """

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
        """Trung bình các token embedding, BỎ QUA token padding.

            embedding = Σ(hidden_i × mask_i) / Σ(mask_i)
        """
        mask: torch.Tensor = (
            tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        )
        summed: torch.Tensor = torch.sum(hidden * mask, dim=1)
        # clamp: một chuỗi rỗng hoàn toàn sẽ chia cho 0 → NaN lan ra cả ma trận điểm.
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

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
            lengths=_token_lengths(tokenizer=self.tokenizer, texts=prefixed),
            max_items=self.batch_size,
            max_tokens=self.max_batch_tokens,
            forward=run,
        )


Encoder = Union[SGPTEncoder, SentenceEncoder]

_ENCODERS: Dict[str, type] = {"sgpt": SGPTEncoder, "sentence": SentenceEncoder}


def build_encoder(dataset: Optional[str] = None) -> Encoder:
    """Dựng encoder CỦA `dataset` — NƠI DUY NHẤT chọn lớp encoder.

    dataset=None → dataset đang chọn (general.dataset). Profile tra qua
    cfg.encoder_for(), tức là qua `datasets.<ds>.encoder` → `encoders.<tên>`.

    Chọn nhầm lớp không nổ lỗi, chỉ ra vector vô nghĩa (SPECB áp lên model BERT,
    hay bỏ tiền tố "query: " của E5), nên việc chọn gom về đúng một chỗ này.
    """
    profile_name: str = cfg.dataset_config(dataset).encoder
    profile: EncoderProfileConfig = cfg.encoder_for(dataset)
    kind: str = profile.type
    cls: Optional[type] = _ENCODERS.get(kind)
    if cls is None:
        raise ValueError(
            f"encoders.{profile_name}.type={kind!r} không hợp lệ. "
            f"Chỉ nhận: {sorted(_ENCODERS)}"
        )
    return cls(profile=profile)
