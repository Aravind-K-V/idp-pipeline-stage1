#!/bin/bash
# start.sh - Startup script for IDP Pipeline

set -e

echo "🚀 Starting IDP Pipeline..."

# Check if running in container
if [ -f /.dockerenv ]; then
    echo "📦 Running in Docker container"
    DOCKER_MODE=true
else
    echo "🖥️  Running on host system"
    DOCKER_MODE=false
fi

# Create necessary directories
mkdir -p /app/models /tmp/uploads /var/log/idp

# Set permissions
if [ "$DOCKER_MODE" = false ]; then
    sudo chown -R $USER:$USER /tmp/uploads /var/log/idp 2>/dev/null || true
fi

# Download models if needed
echo "📥 Checking model files..."
python3 -c "
import os
from transformers import AutoTokenizer, AutoProcessor, DonutProcessor
from ultralytics import YOLO

model_cache = os.environ.get('MODEL_CACHE_DIR', '/app/models')
os.makedirs(model_cache, exist_ok=True)

print('Downloading Donut model...')
try:
    DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2', cache_dir=model_cache)
    print('✅ Donut model ready')
except Exception as e:
    print(f'❌ Donut download failed: {e}')

print('Downloading YOLO model...')
try:
    yolo = YOLO('yolov8s.pt')
    print('✅ YOLO model ready')
except Exception as e:
    print(f'❌ YOLO download failed: {e}')

print('Downloading Table Transformer...')
try:
    AutoProcessor.from_pretrained('microsoft/table-transformer-detection', cache_dir=model_cache)
    print('✅ Table Transformer ready')
except Exception as e:
    print(f'❌ Table Transformer download failed: {e}')

print('Model initialization complete!')
"

# Check if Redis is running
echo "🔍 Checking Redis connection..."
python3 -c "
import redis
import sys
import time

max_retries = 10
for i in range(max_retries):
    try:
        r = redis.from_url('redis://localhost:6379/0')
        r.ping()
        print('✅ Redis connection successful')
        break
    except Exception as e:
        print(f'⏳ Redis connection attempt {i+1}/{max_retries} failed: {e}')
        if i == max_retries - 1:
            print('❌ Redis connection failed after all retries')
            sys.exit(1)
        time.sleep(2)
"

echo "✅ All checks passed!"

# Start the application
if [ "$1" = "api" ]; then
    echo "🌐 Starting FastAPI server..."
    exec uvicorn fastapi_main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1}
elif [ "$1" = "worker" ]; then
    queue=${2:-default}
    echo "👷 Starting Celery worker for queue: $queue"
    exec celery -A celery_tasks worker --loglevel=info --queues=$queue --concurrency=${CONCURRENCY:-2}
elif [ "$1" = "flower" ]; then
    echo "🌸 Starting Flower monitoring..."
    exec celery -A celery_tasks flower --port=5555
elif [ "$1" = "beat" ]; then
    echo "🥁 Starting Celery beat scheduler..."
    exec celery -A celery_tasks beat --loglevel=info
else
    echo "📋 Usage: $0 [api|worker|flower|beat] [queue_name]"
    echo "  api     - Start FastAPI server"
    echo "  worker  - Start Celery worker (specify queue as second argument)"
    echo "  flower  - Start Flower monitoring"
    echo "  beat    - Start Celery beat scheduler"
    exit 1
fi
