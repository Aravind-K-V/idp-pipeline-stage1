# main.py - Main FastAPI Application
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
import os

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from pydantic import BaseModel, Field
import aiofiles

from celery_app import celery_app
from tasks import process_document_task
from config import settings
from utils import setup_logging, validate_pdf, generate_document_id

# Setup structured logging
logger = setup_logging()

# Pydantic models for request/response
class DocumentStatus(BaseModel):
    document_id: str
    status: str
    pages_processed: int
    total_pages: int
    progress_percentage: float
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

class DocumentResult(BaseModel):
    document_id: str
    status: str
    processing_time: float
    total_pages: int
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: str

# Global exception handler
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    logger.error(
        f"Global exception handler triggered",
        extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "url": str(request.url),
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        ).dict()
    )

# Custom HTTP exception handler
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "url": str(request.url),
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        ).dict()
    )

# Validation error handler
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={
            "request_id": request_id,
            "validation_errors": exc.errors(),
            "url": str(request.url),
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": exc.errors()},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        ).dict()
    )

# Request ID middleware
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = datetime.utcnow()

    logger.info(
        f"Request started: {request.method} {request.url}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("user-agent"),
            "start_time": start_time.isoformat()
        }
    )

    try:
        response = await call_next(request)
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Request completed: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "processing_time": processing_time
            }
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        logger.error(
            f"Request failed: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "processing_time": processing_time
            }
        )
        raise

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting IDP Pipeline Application")

    # Initialize Redis for rate limiting
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await FastAPILimiter.init(redis_client)
        logger.info("Redis connection established for rate limiting")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

    # Test Celery connection
    try:
        celery_app.control.inspect().ping()
        logger.info("Celery connection established")
    except Exception as e:
        logger.warning(f"Celery connection test failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down IDP Pipeline Application")
    await FastAPILimiter.close()

# Create FastAPI application
app = FastAPI(
    title="IDP Pipeline API",
    description="Intelligent Document Processing Pipeline for Complex Proposal Forms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.middleware("http")(add_request_id_middleware)

# Add exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"

    # Check Celery connection
    try:
        celery_app.control.inspect().ping()
        celery_status = "healthy"
    except Exception:
        celery_status = "unhealthy"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": redis_status,
            "celery": celery_status
        }
    }

# Main document processing endpoint
@app.post(
    "/process-document",
    response_model=DocumentStatus,
    tags=["Document Processing"],
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def process_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    extract_handwriting: bool = True,
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0),
    enable_table_detection: bool = True,
    enable_checkbox_detection: bool = True
):
    """
    Process a PDF document through the IDP pipeline

    - **file**: PDF file to process (max 50MB)
    - **extract_handwriting**: Enable handwriting extraction using Qwen2.5VL
    - **confidence_threshold**: Minimum confidence threshold for extractions
    - **enable_table_detection**: Enable table detection and extraction
    - **enable_checkbox_detection**: Enable checkbox detection
    """
    request_id = request.state.request_id

    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE/1024/1024}MB"
            )

        # Generate document ID
        document_id = generate_document_id()

        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{document_id}.pdf"

        async with aiofiles.open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)

        # Validate PDF structure
        page_count = validate_pdf(temp_file_path)

        if page_count > settings.MAX_PAGES:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Document has {page_count} pages, maximum allowed is {settings.MAX_PAGES}"
            )

        # Create processing task
        task = process_document_task.delay(
            document_id=document_id,
            file_path=temp_file_path,
            options={
                "extract_handwriting": extract_handwriting,
                "confidence_threshold": confidence_threshold,
                "enable_table_detection": enable_table_detection,
                "enable_checkbox_detection": enable_checkbox_detection
            }
        )

        logger.info(
            f"Document processing task created",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "task_id": task.id,
                "filename": file.filename,
                "pages": page_count
            }
        )

        return DocumentStatus(
            document_id=document_id,
            status="processing",
            pages_processed=0,
            total_pages=page_count,
            progress_percentage=0.0,
            estimated_completion=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing document upload",
            extra={
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process document upload"
        )

# Document status endpoint
@app.get(
    "/document-status/{document_id}",
    response_model=DocumentStatus,
    tags=["Document Processing"]
)
async def get_document_status(document_id: str, request: Request):
    """Get the processing status of a document"""
    request_id = request.state.request_id

    try:
        # Get task status from Redis
        redis_client = redis.from_url(settings.REDIS_URL)
        status_data = await redis_client.hgetall(f"document:{document_id}")

        if not status_data:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        # Convert bytes to strings
        status_dict = {k.decode(): v.decode() for k, v in status_data.items()}

        return DocumentStatus(
            document_id=document_id,
            status=status_dict.get("status", "unknown"),
            pages_processed=int(status_dict.get("pages_processed", 0)),
            total_pages=int(status_dict.get("total_pages", 0)),
            progress_percentage=float(status_dict.get("progress_percentage", 0.0)),
            estimated_completion=status_dict.get("estimated_completion"),
            error_message=status_dict.get("error_message")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting document status",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to get document status"
        )

# Document results endpoint
@app.get(
    "/document-result/{document_id}",
    response_model=DocumentResult,
    tags=["Document Processing"]
)
async def get_document_result(document_id: str, request: Request):
    """Get the final results of a processed document"""
    request_id = request.state.request_id

    try:
        # Get results from Redis
        redis_client = redis.from_url(settings.REDIS_URL)
        result_data = await redis_client.get(f"result:{document_id}")

        if not result_data:
            raise HTTPException(
                status_code=404,
                detail="Document results not found or still processing"
            )

        import json
        result_dict = json.loads(result_data)

        return DocumentResult(**result_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting document result",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to get document result"
        )

# List recent documents endpoint
@app.get("/documents", tags=["Document Processing"])
async def list_recent_documents(
    request: Request,
    limit: int = Field(10, ge=1, le=100),
    offset: int = Field(0, ge=0)
):
    """List recently processed documents"""
    request_id = request.state.request_id

    try:
        redis_client = redis.from_url(settings.REDIS_URL)

        # Get list of document IDs (this would be implemented based on your storage strategy)
        # For now, return a placeholder response

        return {
            "documents": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(
            f"Error listing documents",
            extra={
                "request_id": request_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None  # We handle logging ourselves
    )
