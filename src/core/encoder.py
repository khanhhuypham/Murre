"""core/encoder.py — Bi-Encoder SGPT: văn bản → vector, dùng cho retrieval (§3.2).

SGPT phân biệt vai trò của input bằng cặp ngoặc SPECB bọc quanh chuỗi token:

    query    (câu hỏi)   → [ ... ]
    document (schema)    → { ... }

Vector cuối = weighted-mean pooling trên các token embedding.
"""

import math
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from utils import logger
from config import cfg


class SGPTEncoder:
    """Mã hóa câu hỏi / schema bảng thành vector. model_name=None → cfg.encoder."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = model_name or cfg.encoder.model_name
        self.batch_size: int = cfg.encoder.batch_size
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[SGPTEncoder] Đang tải model '{self.model_name}' trên {self.device} ...")
        logger.info("[SGPTEncoder] Lần đầu tiên sẽ tải model từ HuggingFace (~vài phút) ...")

        # AutoTokenizer/AutoModel tự nhận đúng kiến trúc theo tên model. Lần đầu tải
        # từ HuggingFace Hub rồi cache lại; tokenizer chỉ đổi văn bản thành ID, model
        # mới là phần tính embedding.
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )

        self.model.to(self.device)
        self.model.eval()  # Tắt dropout để kết quả ổn định

        # ─── Token đặc biệt SPECB (SPEcial Character Bracketing) ───────────────────
        # Bọc mỗi input bằng 1 token đầu (BOS) + 1 token cuối (EOS) khác nhau cho
        # query và document, nhờ vậy cùng một nội dung vẫn ra hai vector hơi lệch
        # nhau và cosine similarity query-vs-document chính xác hơn.
        # Ở đây chỉ lấy ID; việc chèn vào chuỗi diễn ra trong _tokenize().
        self.SPECB_QUE_BOS: int = self._encode_single_char_as_token_id(char="[")
        self.SPECB_QUE_EOS: int = self._encode_single_char_as_token_id(char="]")

        self.SPECB_DOC_BOS: int = self._encode_single_char_as_token_id(char="{")
        self.SPECB_DOC_EOS: int = self._encode_single_char_as_token_id(char="}")

        logger.info("[SGPTEncoder] Tải xong! Sẵn sàng mã hóa.")

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
        Chạy theo batch cfg.encoder.batch_size để không vỡ RAM/VRAM.
        """
        all_embeddings: List[torch.Tensor] = []
        num_batches:int = math.ceil(len(texts) / self.batch_size)

        for b in range(num_batches):
            batch_texts:List[str] = texts[b * self.batch_size: (b + 1) * self.batch_size]
            tokens: Dict[str, torch.Tensor] = self._tokenize(texts=batch_texts, is_query=is_query)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                hidden:torch.Tensor = self.model(**tokens).last_hidden_state

            emb: torch.Tensor = self._weighted_mean_pooling(tokens=tokens, hidden=hidden)
            all_embeddings.append(emb.cpu())  # Luôn trả về CPU để tiết kiệm VRAM

        return torch.cat(all_embeddings, dim=0)
