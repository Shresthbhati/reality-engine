# Multi-stage production Dockerfile for Reality Engine API backend
# Supports Python 3.11/3.12 with OpenCV, PyTorch CPU, and headless GL

# ----------------------------------------------------------------------------
# Stage 1: Base runtime environment
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8100 \
    STORAGE_ROOT=/data/reality/artifacts \
    DATABASE_URL=sqlite+aiosqlite:////data/reality/reality.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root application user
RUN groupadd -g 10001 reality && \
    useradd -u 10001 -g reality -m -s /bin/bash appuser

WORKDIR /app

# ----------------------------------------------------------------------------
# Stage 2: Dependencies builder
# ----------------------------------------------------------------------------
FROM runtime-base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install .[api,export]

# ----------------------------------------------------------------------------
# Stage 3: Final application container
# ----------------------------------------------------------------------------
FROM runtime-base AS runner

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:reality . /app

# Ensure data directory exists and is writable by appuser
RUN mkdir -p /data/reality/artifacts && chown -R appuser:reality /data/reality

USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8100/healthz || exit 1

ENTRYPOINT ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
