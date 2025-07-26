# utils.py - Shared utilities for IDP Pipeline
import logging
import sys
import uuid
import hashlib
import json
import time
from typing import Dict, Any, Optional
import structlog
from pathlib import Path
import PyPDF2
import io

def setup_logging() -> logging.Logger:
    """Setup structured logging for the application"""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    return structlog.get_logger()

def generate_document_id() -> str:
    """Generate a unique document ID"""
    return str(uuid.uuid4())

def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())[:8]

def validate_pdf(file_content: bytes) -> Dict[str, Any]:
    """
    Validate PDF file content

    Args:
        file_content: Raw PDF file content

    Returns:
        Dictionary with validation results
    """
    try:
        # Basic file size check (already done at FastAPI level)
        if len(file_content) == 0:
            return {"valid": False, "error": "Empty file"}

        # Try to read as PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))

        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            return {"valid": False, "error": "Encrypted PDFs are not supported"}

        # Get page count
        page_count = len(pdf_reader.pages)
        if page_count == 0:
            return {"valid": False, "error": "PDF has no pages"}

        # Check for reasonable page limit
        if page_count > 50:  # Configurable limit
            return {"valid": False, "error": f"PDF has too many pages ({page_count}). Maximum allowed: 50"}

        # Extract basic metadata
        metadata = {}
        if pdf_reader.metadata:
            metadata = {
                "title": pdf_reader.metadata.get("/Title", ""),
                "author": pdf_reader.metadata.get("/Author", ""),
                "creator": pdf_reader.metadata.get("/Creator", ""),
                "producer": pdf_reader.metadata.get("/Producer", ""),
                "creation_date": str(pdf_reader.metadata.get("/CreationDate", "")),
                "modification_date": str(pdf_reader.metadata.get("/ModDate", ""))
            }

        return {
            "valid": True,
            "page_count": page_count,
            "file_size": len(file_content),
            "metadata": metadata
        }

    except PyPDF2.errors.PdfReadError as e:
        return {"valid": False, "error": f"Invalid PDF format: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"PDF validation failed: {str(e)}"}

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

def estimate_processing_time(page_count: int) -> float:
    """
    Estimate processing time based on page count

    Args:
        page_count: Number of pages in the document

    Returns:
        Estimated processing time in seconds
    """
    # Base processing times per page (in seconds)
    # These are rough estimates based on benchmarks
    time_per_page = {
        "text": 0.35,
        "table": 1.10,
        "checkbox": 0.05,
        "handwriting": 2.50
    }

    # Since services run in parallel, use the slowest service
    max_time_per_page = max(time_per_page.values())

    # Add overhead for orchestration and I/O
    overhead_per_page = 0.1

    # Total estimate
    estimated_time = page_count * (max_time_per_page + overhead_per_page)

    # Add base overhead for document setup
    base_overhead = 5.0

    return estimated_time + base_overhead

def format_processing_status(
    current_page: int, 
    total_pages: int, 
    start_time: float,
    estimated_total_time: Optional[float] = None
) -> Dict[str, Any]:
    """Format processing status for API responses"""

    progress_percentage = (current_page / total_pages) * 100 if total_pages > 0 else 0
    elapsed_time = time.time() - start_time

    # Estimate completion time
    estimated_completion = None
    if estimated_total_time and progress_percentage > 0:
        remaining_time = estimated_total_time - elapsed_time
        if remaining_time > 0:
            estimated_completion = time.time() + remaining_time

    return {
        "pages_processed": current_page,
        "total_pages": total_pages,
        "progress_percentage": round(progress_percentage, 1),
        "elapsed_time": round(elapsed_time, 2),
        "estimated_completion": estimated_completion
    }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    sanitized = filename

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 255 - len(ext) - 1
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')

    return sanitized

def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "timestamp": time.time(),
        "request_id": request_id or generate_request_id()
    }

def validate_confidence_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Validate and normalize confidence scores"""
    validated = {}

    for service, score in scores.items():
        # Ensure score is between 0 and 1
        normalized_score = max(0.0, min(1.0, float(score)))
        validated[service] = round(normalized_score, 3)

    return validated

def merge_service_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from different services into a unified format"""
    merged = {
        "text_data": results.get("text", {}).get("text_blocks", []),
        "tables": results.get("table", {}).get("tables", []),
        "checkboxes": results.get("checkbox", {}).get("checkboxes", []),
        "handwriting": results.get("handwriting", {}).get("handwriting", []),
        "confidence_scores": {
            "text": results.get("text", {}).get("confidence", 0.0),
            "table": results.get("table", {}).get("confidence", 0.0),
            "checkbox": results.get("checkbox", {}).get("confidence", 0.0),
            "handwriting": results.get("handwriting", {}).get("confidence", 0.0)
        },
        "processing_times": {
            "text": results.get("text", {}).get("processing_time", 0.0),
            "table": results.get("table", {}).get("processing_time", 0.0),
            "checkbox": results.get("checkbox", {}).get("processing_time", 0.0),
            "handwriting": results.get("handwriting", {}).get("processing_time", 0.0)
        }
    }

    return merged

class PerformanceMonitor:
    """Simple performance monitoring utility"""

    def __init__(self):
        self.metrics = {}

    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.metrics[timer_id] = {"start": time.time(), "operation": operation}
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """End timing and return duration"""
        if timer_id in self.metrics:
            duration = time.time() - self.metrics[timer_id]["start"]
            self.metrics[timer_id]["duration"] = duration
            return duration
        return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.metrics.copy()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
