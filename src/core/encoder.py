"""core/encoder.py — Bi-Encoder: văn bản → vector, dùng cho retrieval (§3.2).

    encoder = SentenceEncoder.get(dataset="vitext2sql")   # dùng lại instance
    vectors = encoder.encode(texts=[...], is_query=False)  # (n, hidden_dim)

Mean pooling có mask, phân biệt query/document bằng tiền tố văn bản nếu model
đòi (E5: "query: " / "passage: ", khai ở <profile>.query_prefix / doc_prefix).
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence

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
    """Bi-encoder sentence-transformers: mean pooling + tiền tố theo vai trò.

    Model đa ngữ (multilingual-e5-*, bge-m3, ...), dùng chung cho mọi dataset.
    """

    # Một instance cho mỗi PROFILE, dựng lúc gọi get() chứ không lúc import.
    _instances: Dict[str, SentenceEncoder] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, dataset: Optional[str] = None) -> SentenceEncoder:
        """Encoder của `dataset`, dùng lại instance nếu đã dựng.

        dataset=None → dataset đang chọn (general.dataset). Profile tra qua
        cfg.encoder_for(). Gọi hai lần cùng một profile trả về CÙNG MỘT object;
        gọi thẳng `SentenceEncoder(profile=...)` thì dựng bản riêng, không cache.
        """
        profile_name: str = cfg.dataset_config(dataset).encoder

        # Giữ khóa suốt lúc nạp model: luồng thứ hai đợi rồi dùng chung.
        with cls._lock:
            encoder: Optional[SentenceEncoder] = cls._instances.get(profile_name)
            if encoder is None:
                encoder = cls(profile=cfg.encoder_for(dataset))
                cls._instances[profile_name] = encoder
            return encoder

    @classmethod
    def reset_cache(cls) -> None:
        """Xoá cache instance — CHỈ dùng cho test."""
        with cls._lock:
            cls._instances.clear()

    def __init__(self, profile: EncoderProfileConfig) -> None:
        self.model_name: str = profile.model_name
        self.batch_size: int = profile.batch_size
        self.max_batch_tokens: int = profile.max_batch_tokens
        self.max_length: int = profile.max_length
        self.query_prefix: str = profile.query_prefix
        self.doc_prefix: str = profile.doc_prefix
        self.tokenizer, self.model, self.device = type(self)._load(
            model_name=self.model_name,
        )

    @staticmethod
    def _mean_pooling(
        tokens: Dict[str, torch.Tensor], hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Trung bình token embedding, bỏ qua padding: Σ(h_i × m_i) / Σ(m_i)."""
        mask: torch.Tensor = (
            tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        )
        summed: torch.Tensor = torch.sum(hidden * mask, dim=1)
        # clamp: chuỗi rỗng hoàn toàn sẽ chia cho 0 → NaN.
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _token_lengths(self, texts: List[str]) -> List[int]:
        """Số token của từng văn bản, chưa padding — dùng để chia batch."""
        if not texts:
            return []
        encoded = self.tokenizer(text=texts, padding=False, truncation=True)
        return [len(ids) for ids in encoded["input_ids"]]

    @staticmethod
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
        model.eval()
        logger.info(f"[Encoder] Tải xong '{model_name}'.")
        return tokenizer, model, device

    @staticmethod
    def plan_batches(
        lengths: Sequence[int], max_items: int, max_tokens: int,
    ) -> List[List[int]]:
        """Chia chỉ số thành batch theo NGÂN SÁCH TOKEN, không theo số lượng.

        Sắp theo độ dài rồi gom, cắt batch khi `số_câu × độ_dài_nhất` vượt
        `max_tokens` — câu càng dài thì batch càng nhỏ, chặn tràn bộ nhớ.

        Trả về danh sách batch, mỗi batch là các CHỈ SỐ trong `lengths`; chỗ gọi
        phải tự xếp kết quả về vị trí cũ (xem _encode_batched).
        """
        order: List[int] = sorted(range(len(lengths)), key=lambda i: lengths[i])

        batches: List[List[int]] = []
        current: List[int] = []
        for idx in order:
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

    def _encode_batched(self, texts: List[str]) -> torch.Tensor:
        """Chia lô, nhúng từng lô bằng _embed_batch, rồi XẾP LẠI ĐÚNG THỨ TỰ.

        Độ dài (số token) đo bằng tokenizer của chính encoder; ngân sách batch
        lấy từ profile (batch_size / max_batch_tokens).

        plan_batches() xáo thứ tự theo độ dài, nên phải xếp lại: kết quả[i] phải
        là vector của texts[i].
        """
        if not texts:
            return torch.empty(0, 0)

        batches: List[List[int]] = self.plan_batches(
            lengths=self._token_lengths(texts=texts),
            max_items=self.batch_size,
            max_tokens=self.max_batch_tokens,
        )

        parts: List[torch.Tensor] = []
        positions: List[int] = []
        for batch in batches:
            parts.append(self._embed_batch([texts[i] for i in batch]).cpu())
            positions.extend(batch)

        stacked: torch.Tensor = torch.cat(parts, dim=0)
        out: torch.Tensor = torch.empty_like(stacked)
        out[torch.tensor(positions, dtype=torch.long)] = stacked
        return out

    def encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Mã hóa danh sách văn bản → tensor (len(texts), hidden_dim) trên CPU."""
        prefix: str = self.query_prefix if is_query else self.doc_prefix
        prefixed: List[str] = [prefix + t for t in texts] if prefix else list(texts)

        return self._encode_batched(texts=prefixed)

    def _embed_batch(self, batch: List[str]) -> torch.Tensor:
        """MỘT lô văn bản → ma trận vector: tokenize → model → mean pooling.

        Không chia lô, không xếp lại thứ tự — đó là việc của _encode_batched.
        """
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
