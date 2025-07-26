# Dockerfile for IDP Pipeline
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install detectron2 separately for layoutparser
RUN pip3 install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/uploads /tmp/uploads /var/log/idp

# Set permissions
RUN chmod +x /app/start.sh || true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app /tmp/uploads /var/log/idp
USER app

# Environment variables
ENV MODEL_CACHE_DIR=/app/models
ENV UPLOAD_FOLDER=/tmp/uploads
ENV LOG_LEVEL=INFO
ENV DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["uvicorn", "fastapi_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
