# syntax=docker/dockerfile:1.7

# ─── Stage 1: build the frontend (Vite → static dist/) ────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/react-app/package.json frontend/react-app/package-lock.json ./
RUN npm ci
COPY frontend/react-app/ ./
RUN npm run build

# ─── Stage 2: Python runtime (FastAPI + PyTorch) ──────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/backend

# OpenCV needs libGL + libglib for image I/O even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies. By default we pull CPU-only PyTorch wheels;
# override TORCH_INDEX_URL at build time for CUDA (see README).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url ${TORCH_INDEX_URL} \
        opencv-python-headless matplotlib uvicorn[standard] \
    && pip install --extra-index-url ${TORCH_INDEX_URL} -r /tmp/requirements.txt

# Copy application source
COPY backend/   /app/backend/
COPY utils/     /app/utils/
COPY analysis/  /app/analysis/
COPY thesis_configs/ /app/thesis_configs/

# Copy the built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist /app/frontend/react-app/dist

# Persistent volumes for uploads / generated results
RUN mkdir -p /app/backend/uploads /app/backend/results

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
