# =============================================================================
# MURRE API service. Build:  docker build -t murre .
#
# Torch lấy từ index CPU-only (~200MB thay vì ~2.5GB bản CUDA). Cần GPU thì bỏ
# dòng --index-url ở lớp cài torch và chạy container với `--gpus all`.
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model HuggingFace tải về đây — mount volume vào để không tải lại mỗi lần chạy.
    HF_HOME=/models

WORKDIR /app

# Cài dependency trước, copy code sau: sửa code không phải build lại lớp nặng này.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3" \
 && pip install -r requirements.txt

COPY config.yaml ./
COPY prompts/ ./prompts/
COPY dataset/ ./dataset/
COPY src/ ./src/

# Chạy bằng user thường; /models và /app/outputs là hai chỗ duy nhất cần ghi.
RUN useradd --create-home --uid 10001 murre \
 && mkdir -p /models /app/outputs \
 && chown -R murre:murre /models /app/outputs
USER murre

EXPOSE 8000

# config.py chdir về gốc project lúc import, nên chạy uvicorn từ src/.
WORKDIR /app/src
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
