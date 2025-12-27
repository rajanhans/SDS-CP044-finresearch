# =============================================================================
# FinResearch AI - Hugging Face Spaces Dockerfile
# Multi-Stage Build: React Frontend + FastAPI Backend
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Frontend Builder (Node.js)
# Build the React application with Vite
# -----------------------------------------------------------------------------
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --silent

# Copy frontend source
COPY frontend/ ./

# Build the production bundle
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python Runtime
# FastAPI backend serving React static files
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face Spaces port
    PORT=7860 \
    HOST=0.0.0.0 \
    # App settings
    PYTHONPATH=/app \
    OUTPUT_DIR=/app/outputs \
    CHROMA_PERSIST_DIR=/app/.chroma_db \
    # Disable telemetry
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Using pip directly for simpler deployment
COPY pyproject.toml ./

# Install poetry and export requirements, then install with pip
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi \
    && pip uninstall -y poetry

# Copy backend source code
COPY src/ ./src/
COPY main.py ./

# Copy built frontend from Stage 1 into static/ directory
COPY --from=frontend-builder /app/frontend/dist ./static/

# Create necessary directories
RUN mkdir -p /app/outputs /app/.chroma_db

# Expose the Hugging Face Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Run the FastAPI server on HF Spaces port
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
