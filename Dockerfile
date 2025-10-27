# syntax=docker/dockerfile:1

# Multi-stage build for optimized image size
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/tmp/.torch

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Runtime with minimal dependencies
# ============================================
FROM base AS runtime

# Copy requirements and install Python dependencies
COPY requirements.txt ./
# Install core scientific Python stack first from PyPI
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    scipy==1.11.4 \
    scikit-learn \
    torchsummary \
    imbalanced-learn \
    shap \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    notebook \
    ipython

# Install PyTorch CPU wheels compatible with Python 3.11
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies from requirements.txt if needed
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output

# Set default working directory
WORKDIR /app

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('PyTorch OK')" || exit 1

# Default command: Run the hybrid model help
CMD ["python", "scripts/hybrid_cnn_rnn_attention.py", "-h"]

# ============================================
# Alternative entry points
# ============================================
# For running Jupyter:
# docker run -p 8888:8888 -v $(pwd)/output:/app/output hybrid-model jupyter lab --ip=0.0.0.0 --allow-root

# For running the hybrid model:
# docker run -v $(pwd)/data:/app/data -v $(pwd)/index:/app/index -v $(pwd)/output:/app/output hybrid-model \
#     python scripts/hybrid_cnn_rnn_attention.py --data-dir data --index-dir index --output-dir output

# For running with GPU support, use:
# docker run --gpus all -v $(pwd)/output:/app/output hybrid-model python scripts/hybrid_cnn_rnn_attention.py
