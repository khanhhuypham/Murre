# =============================================================================
# core/encoder.py — Bộ mã hóa văn bản SGPT với SPECB brackets
#
# SGPT dùng ký hiệu đặc biệt:
#   Query (câu hỏi): [câu hỏi]
#   Document (bảng): {tên bảng và cột}
#
# Sau đó tính weighted-mean pooling trên các token embedding
# để tạo ra vector đại diện cho toàn bộ câu/văn bản.
# =============================================================================

import math
from typing import List, Dict

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
    """Bộ mã hóa câu hỏi / schema bảng thành vector, dùng cho Bi-Encoder retrieval (§3.2)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        # Ưu tiên tham số truyền vào, sau đó dùng config
        self.model_name: str = model_name or cfg.encoder.model_name
        self.batch_size: int = cfg.encoder.batch_size

        # Dùng GPU nếu có, ngược lại dùng CPU
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[SGPTEncoder] Đang tải model '{self.model_name}' trên {self.device} ...")
        logger.info("[SGPTEncoder] Lần đầu tiên sẽ tải model từ HuggingFace (~vài phút) ...")

        # Tải tokenizer tương ứng với model_name — AutoTokenizer tự nhận diện đúng loại
        # tokenizer (BPE/WordPiece/...) dựa trên tên model, không cần khai báo trước.
        # Lần đầu chạy sẽ tải về từ HuggingFace Hub và cache lại cho các lần sau.
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        # Tải model SGPT đã huấn luyện sẵn — đây là phần tính toán embedding thực sự,
        # tokenizer ở trên chỉ chuẩn bị đầu vào (chuyển văn bản thành ID) cho model này.
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )

        self.model.to(self.device) # Chuyển model sang GPU/CPU theo self.device
        self.model.eval()  # Tắt dropout để kết quả ổn định

        # ─── Token đặc biệt SPECB (SPEcial Character Bracketing) ───────────────────
        # SGPT dùng một cặp ký tự riêng để "đánh dấu" đầu vào là câu hỏi (query)
        # hay là tài liệu/bảng (document), giúp model tạo ra hai loại embedding
        # có xu hướng khác nhau một chút dù cùng nội dung — nhờ vậy khi so khớp
        # query-vs-document bằng cosine similarity sẽ chính xác hơn so với việc
        # mã hóa cả hai như văn bản thông thường, không phân biệt vai trò.
        #
        # Cách hoạt động: mỗi input trước khi đưa vào model sẽ được "bọc" thêm
        # 1 token ở đầu (BOS - Beginning Of Sequence) và 1 token ở cuối
        # (EOS - End Of Sequence). Token thật sự được chèn vào chuỗi input diễn ra
        # sau, ở hàm _tokenize() — ở đây chỉ lấy ra ID tương ứng để dùng cho bước đó.

        # Query (câu hỏi) được bọc bởi cặp ngoặc vuông "[" "]"
        self.SPECB_QUE_BOS: int = self._encode_single_char_as_token_id(char="[")
        self.SPECB_QUE_EOS: int = self._encode_single_char_as_token_id(char="]")

        # Document (schema bảng) được bọc bởi cặp ngoặc nhọn "{" "}"
        self.SPECB_DOC_BOS: int = self._encode_single_char_as_token_id(char="{")
        self.SPECB_DOC_EOS: int = self._encode_single_char_as_token_id(char="}")

        logger.info("[SGPTEncoder] Tải xong! Sẵn sàng mã hóa.")

    # --------------------------------------------------------------------------
    # Các phương thức nội bộ
    # --------------------------------------------------------------------------
    def _encode_single_char_as_token_id(self, char: str) -> int:
        """Mã hóa một ký tự đơn thành đúng 1 token ID, dùng cho token SPECB.

        add_special_tokens=False: không tự động thêm token đặc biệt mặc định
        của model (ví dụ [CLS], [SEP]) — chỉ lấy ID của riêng ký tự `char`.
        """
        token_ids: List[int] = self.tokenizer.encode(text=char, add_special_tokens=False)

        # Đảm bảo ký tự char biến thành ĐÚNG 1 token ID duy nhất.
        # Một số tokenizer (như BPE, WordPiece) có thể cắt 1 ký tự lạ/unicode/emoji
        # thành 2 hoặc nhiều token nhỏ hơn. Nếu len(token_ids) != 1, báo lỗi lập tức để tránh
        # làm hỏng logic của token đánh dấu đặc biệt (SPECB).
        assert len(token_ids) == 1, (
            f"Ký tự '{char}' bị tokenizer tách thành {len(token_ids)} token "
            f"({token_ids}), không thể dùng làm token SPECB (yêu cầu đúng 1 token)."
        )
        return token_ids[0]

    def _tokenize(self, texts: List[str], is_query: bool) -> Dict[str, torch.Tensor]:
        """
        Tokenize văn bản và thêm token SPECB vào đầu/cuối.
        is_query=True  → thêm [ ] (dành cho câu hỏi)
        is_query=False → thêm { } (dành cho schema bảng)
        """
        # Tokenize không padding trước để chèn token đặc biệt
        batch: Dict[str, List[List[int]]] = self.tokenizer(
            text=texts, padding=False, truncation=True
        )

        # Xác định token đặc biệt theo loại văn bản
        bos: int = self.SPECB_QUE_BOS if is_query else self.SPECB_DOC_BOS
        eos: int = self.SPECB_QUE_EOS if is_query else self.SPECB_DOC_EOS

        for ids, att in zip(batch["input_ids"], batch["attention_mask"]):
            # Chèn token đặc biệt vào đầu và cuối mỗi chuỗi
            ids.insert(0, bos)
            ids.append(eos)
            att.insert(0, 1)  # Attention mask = 1 → chú ý vào token này
            att.append(1)

        # Padding toàn bộ batch về cùng độ dài rồi chuyển sang tensor
        return self.tokenizer.pad(encoded_inputs=batch, padding=True, return_tensors="pt")

    @staticmethod
    def _weighted_mean_pooling(tokens: Dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        """
        Tính weighted-mean pooling: token ở vị trí sau được trọng số lớn hơn.
        Công thức: embedding = Σ(hidden_i × mask_i × weight_i) / Σ(mask_i × weight_i)
        weight_i = vị trí của token (1, 2, 3, ...)
        """
        seq_len:int = hidden.shape[1]

        # Trọng số theo vị trí: token ở vị trí cuối được trọng số cao hơn
        weights:torch.Tensor = (
            torch.arange(1, seq_len + 1)
            .unsqueeze(0)        # (1, seq_len)
            .unsqueeze(-1)       # (1, seq_len, 1)
            .expand(hidden.size())  # (batch, seq_len, hidden_dim)
            .float()
            .to(hidden.device)
        )

        # Mask: chỉ tính trên các token thực (không phải padding)
        mask:torch.Tensor = (
            tokens["attention_mask"]
            .unsqueeze(-1)       # (batch, seq_len, 1)
            .expand(hidden.size())
            .float()
            .to(hidden.device)
        )

        # Tính trung bình có trọng số
        summed: torch.Tensor = torch.sum(hidden * mask * weights, dim=1)
        norm: torch.Tensor = torch.sum(mask * weights, dim=1)
        return summed / norm

    # --------------------------------------------------------------------------
    # API công khai
    # --------------------------------------------------------------------------
    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """
        Mã hóa danh sách văn bản thành ma trận embedding.

        Tham số:
            texts    : danh sách chuỗi cần mã hóa
            is_query : True nếu là câu hỏi, False nếu là schema bảng

        Trả về:
            Tensor shape (len(texts), hidden_dim) trên CPU
        """
        all_embeddings: List[torch.Tensor] = []
        num_batches:int = math.ceil(len(texts) / self.batch_size)

        for b in range(num_batches):
            # Lấy từng batch
            batch_texts:List[str] = texts[b * self.batch_size: (b + 1) * self.batch_size]
            tokens: Dict[str, torch.Tensor] = self._tokenize(texts=batch_texts, is_query=is_query)

            # Chuyển sang thiết bị (GPU/CPU)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                # Lấy hidden state từ model
                hidden:torch.Tensor = self.model(**tokens).last_hidden_state

            # Tính weighted-mean pooling
            emb: torch.Tensor = self._weighted_mean_pooling(tokens=tokens, hidden=hidden)
            all_embeddings.append(emb.cpu())  # Luôn trả về CPU để tiết kiệm VRAM

        return torch.cat(all_embeddings, dim=0)
