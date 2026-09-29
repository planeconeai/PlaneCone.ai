# ============================================================================
# PlaneConeAI Mammo AI — Production Dockerfile
# Multi-stage · CPU-only PyTorch · Non-root · Health-checked
# ============================================================================

# ── Stage 1: Build wheel cache ────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools for native extensions (pylibjpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels into /build/wheels — reused in production stage
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.10-slim AS production

LABEL org.opencontainers.image.title="planeconeai-mammo-ai" \
      org.opencontainers.image.description="PlaneConeAI Mammography AI Inference Service" \
      org.opencontainers.image.vendor="PlaneCone AI"

# Security: create non-root user
RUN groupadd -g 1001 appgroup \
 && useradd  -u 1001 -g appgroup -s /bin/false appuser

# Runtime system dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from builder (no compilation in prod image)
COPY --from=builder /build/wheels /tmp/wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
 && rm -rf /tmp/wheels

# Copy application source
COPY --chown=appuser:appgroup app.py constants.py inference.py preprocessing.py schemas.py ./

# Minimal resource usage for CPU inference
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=2 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

USER appuser

# tini for proper PID 1 signal handling
ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
