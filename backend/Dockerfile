# DeltaScribe Edge Backend
# Multi-stage build for production

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (cmake needed for llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install uv

# Copy dependency files (README.md needed by hatchling build)
COPY pyproject.toml .
RUN touch README.md

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache ".[gguf,retrieval,audio]"

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies (ffmpeg needed for audio transcription)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/

# Create non-root user with HF cache directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Set HF cache location
ENV HF_HOME="/home/appuser/.cache/huggingface"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
