# =============================================================================
# steps/embed.py — BƯỚC 1: Mã hóa toàn bộ corpus schema thành vector embedding
#
# Tương đương với retrieve/embed.py của tác giả.
#
# CÁCH CHẠY (Option 2 — Batch mode):
#   python -m steps.embed
#
# Kết quả lưu tại: outputs/{dataset}_{scale}_embeddings.pt  (paths.embeddings_cache)
#
# Bước này KHÔNG bắt buộc: mọi chỗ cần embeddings đều gọi core/corpus.prepare(),
# và prepare() tự encode rồi lưu cache nếu chưa có. Chạy riêng ở đây hữu ích khi
# muốn mã hóa trước một lần cho xong (BirdUnion mất vài phút trên CPU) rồi mới bắt
# đầu chuỗi retrieve/rewrite dài.
#
# Trước đây bước này ghi ra outputs/{dataset}/{scale}/embeddings.json theo format
# của tác giả — cùng một bộ vector nhưng lưu thành 2 định dạng, encode 2 lần. Nay
# chỉ còn cache .pt duy nhất, dùng chung cho cả Option 1, Option 2 và API.
# =============================================================================

import torch

from config import cfg
from core.corpus import prepare
from utils import logger


def run_embed() -> None:
    """Mã hóa corpus rồi lưu cache .pt (bỏ qua nếu cache đã có)."""
    logger.info(f"\n{'=' * 60}\n  BƯỚC 1: MÃ HÓA CORPUS SCHEMA (EMBED)\n{'=' * 60}")

    cache_path: str = cfg.outputs.embeddings_cache()

    # prepare() tự lo: có cache thì nạp, chưa có thì encode rồi ghi cache.
    _, corpus, embeddings = prepare()
    embeddings: torch.Tensor

    logger.info(
        f"[Embed] Xong {len(corpus)} schema | embeddings {tuple(embeddings.shape)} "
        f"→ {cache_path}"
    )
    logger.info("=" * 60)


# =============================================================================
# ĐIỂM CHẠY ĐỘC LẬP (Option 2 — Batch mode)
# =============================================================================
if __name__ == "__main__":
    run_embed()
