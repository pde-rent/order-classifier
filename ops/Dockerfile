# Lean CPU-only build for rule-based Order Classifier
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies - minimal set for CPU-only classification
RUN uv sync --frozen

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY src ./src
COPY run_server.py ./

# Fix permissions
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=40012 \
    HOST=0.0.0.0

EXPOSE $PORT

# Health check - use environment variable for port
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run server
CMD ["uv", "run", "python", "run_server.py"]