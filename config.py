# config.py - Environment-driven configuration
import os
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Configuration
    API_TITLE: str = "IDP Pipeline API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Intelligent Document Processing Pipeline with Parallel Services"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")

    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")

    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".pdf"], env="ALLOWED_EXTENSIONS")
    UPLOAD_FOLDER: str = Field(default="/tmp/uploads", env="UPLOAD_FOLDER")

    # Processing Configuration
    MAX_PAGES: int = Field(default=50, env="MAX_PAGES")
    DPI: int = Field(default=150, env="DPI")
    HIGH_DPI: int = Field(default=300, env="HIGH_DPI")  # For retry attempts

    # Model Configuration
    MODEL_CACHE_DIR: str = Field(default="/app/models", env="MODEL_CACHE_DIR")
    DEVICE: str = Field(default="cuda", env="DEVICE")  # cuda, cpu, or mps

    # Text Service Configuration
    DONUT_MODEL: str = Field(default="naver-clova-ix/donut-base-finetuned-cord-v2", env="DONUT_MODEL")
    OCR_ENGINE: str = Field(default="paddleocr", env="OCR_ENGINE")  # paddleocr, easyocr, tesseract

    # Table Service Configuration
    TABLE_DETECTOR_MODEL: str = Field(default="microsoft/table-transformer-detection", env="TABLE_DETECTOR_MODEL")
    TABLE_STRUCTURE_MODEL: str = Field(default="microsoft/table-transformer-structure-recognition", env="TABLE_STRUCTURE_MODEL")

    # Checkbox Service Configuration
    CHECKBOX_MODEL: str = Field(default="yolov8s", env="CHECKBOX_MODEL")
    CHECKBOX_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="CHECKBOX_CONFIDENCE_THRESHOLD")

    # Handwriting Service Configuration
    HANDWRITING_MODEL: str = Field(default="Qwen/Qwen2-VL-7B-Instruct", env="HANDWRITING_MODEL")
    HANDWRITING_MIN_PIXELS: int = Field(default=256*28*28, env="HANDWRITING_MIN_PIXELS")

    # Quality Validation Configuration
    MIN_CONFIDENCE_THRESHOLD: float = Field(default=0.8, env="MIN_CONFIDENCE_THRESHOLD")
    RETRY_THRESHOLD: float = Field(default=0.6, env="RETRY_THRESHOLD")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=10, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_PERIOD: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Environment-specific configurations
def get_database_url() -> str:
    """Get database URL for different environments"""
    if settings.DEBUG:
        return f"sqlite:///./test.db"
    else:
        return f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def get_model_paths() -> dict:
    """Get model paths based on environment"""
    return {
        "text_model": os.path.join(settings.MODEL_CACHE_DIR, "donut"),
        "table_model": os.path.join(settings.MODEL_CACHE_DIR, "table_transformer"),
        "checkbox_model": os.path.join(settings.MODEL_CACHE_DIR, "yolo"),
        "handwriting_model": os.path.join(settings.MODEL_CACHE_DIR, "qwen2vl"),
    }

def setup_directories():
    """Create necessary directories"""
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    # Create model subdirectories
    model_paths = get_model_paths()
    for path in model_paths.values():
        os.makedirs(path, exist_ok=True)
