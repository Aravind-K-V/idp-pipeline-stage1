# tasks.py - Celery Tasks for Document Processing
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import tempfile

from celery import Celery
from celery.utils.log import get_task_logger
import redis
from pdf2image import convert_from_path
import cv2
import numpy as np

from services.text_service import TextExtractionService
from services.table_service import TableExtractionService
from services.checkbox_service import CheckboxDetectionService
from services.handwriting_service import HandwritingExtractionService
from services.quality_validator import QualityValidator
from services.json_assembler import JSONSchemaAssembler
from config import settings

# Setup Celery
celery_app = Celery(
    'idp_pipeline',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
)

logger = get_task_logger(__name__)

class DocumentProcessor:
    """Main document processor orchestrating all services"""

    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.text_service = TextExtractionService()
        self.table_service = TableExtractionService()
        self.checkbox_service = CheckboxDetectionService()
        self.handwriting_service = HandwritingExtractionService()
        self.quality_validator = QualityValidator()
        self.json_assembler = JSONSchemaAssembler()

    def update_progress(self, document_id: str, status: str, pages_processed: int, 
                       total_pages: int, error_message: str = None):
        """Update document processing progress in Redis"""
        progress_percentage = (pages_processed / total_pages) * 100 if total_pages > 0 else 0

        # Estimate completion time based on current progress
        if pages_processed > 0 and status == "processing":
            avg_time_per_page = 1.5  # seconds (from specs)
            remaining_pages = total_pages - pages_processed
            estimated_completion = (datetime.utcnow() + 
                                  timedelta(seconds=remaining_pages * avg_time_per_page)).isoformat()
        else:
            estimated_completion = None

        status_data = {
            "status": status,
            "pages_processed": str(pages_processed),
            "total_pages": str(total_pages),
            "progress_percentage": str(progress_percentage),
            "estimated_completion": estimated_completion or "",
            "error_message": error_message or "",
            "last_updated": datetime.utcnow().isoformat()
        }

        self.redis_client.hset(f"document:{document_id}", mapping=status_data)
        self.redis_client.expire(f"document:{document_id}", 3600)  # 1 hour TTL

    def classify_document(self, file_path: str) -> Dict[str, Any]:
        """Classify document type and determine processing strategy"""
        try:
            # Convert first page to analyze document structure
            pages = convert_from_path(file_path, first_page=1, last_page=1, dpi=150)
            first_page = np.array(pages[0])

            # Simple classification logic - can be enhanced with ML models
            classification = {
                "document_type": "proposal_form",  # Default assumption
                "complexity": "high",  # Assume complex until proven otherwise
                "has_tables": self._detect_tables_present(first_page),
                "has_checkboxes": self._detect_checkboxes_present(first_page),
                "has_handwriting": True,  # Assume present for proposal forms
                "processing_strategy": "parallel"
            }

            logger.info(f"Document classified: {classification}")
            return classification

        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return {
                "document_type": "unknown",
                "complexity": "high",
                "has_tables": True,
                "has_checkboxes": True,
                "has_handwriting": True,
                "processing_strategy": "parallel"
            }

    def _detect_tables_present(self, image: np.ndarray) -> bool:
        """Quick detection if tables are present in the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Look for horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

            # If we find significant horizontal and vertical lines, likely has tables
            return np.sum(horizontal_lines > 128) > 1000 and np.sum(vertical_lines > 128) > 1000
        except Exception:
            return True  # Default to assuming tables are present

    def _detect_checkboxes_present(self, image: np.ndarray) -> bool:
        """Quick detection if checkboxes are present in the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Look for square-like contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            square_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Reasonable checkbox size
                    # Check if it's roughly square
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 <= aspect_ratio <= 1.2:
                        square_count += 1

            return square_count > 5  # If we find several square-like shapes
        except Exception:
            return True  # Default to assuming checkboxes are present

@celery_app.task(bind=True, name='process_document_task')
def process_document_task(self, document_id: str, file_path: str, options: Dict[str, Any]):
    """
    Main task to process a document through the IDP pipeline
    """
    processor = DocumentProcessor()

    try:
        logger.info(f"Starting document processing for {document_id}")

        # Initialize progress
        processor.update_progress(document_id, "initializing", 0, 1)

        # Step 1: Classify document
        classification = processor.classify_document(file_path)

        # Step 2: Convert PDF to images
        logger.info(f"Converting PDF to images: {file_path}")
        pages = convert_from_path(file_path, dpi=200, fmt='RGB')
        total_pages = len(pages)

        processor.update_progress(document_id, "processing", 0, total_pages)

        # Step 3: Process each page in parallel
        page_results = []

        for page_num, page_image in enumerate(pages, 1):
            try:
                logger.info(f"Processing page {page_num}/{total_pages}")

                # Convert PIL image to numpy array
                page_array = np.array(page_image)

                # Create tasks for parallel processing
                page_tasks = []

                # Text extraction task
                text_task = extract_text_from_page.delay(
                    document_id, page_num, page_array.tolist(), 
                    options.get('confidence_threshold', 0.8)
                )
                page_tasks.append(('text', text_task))

                # Table extraction task (if enabled)
                if options.get('enable_table_detection', True) and classification['has_tables']:
                    table_task = extract_tables_from_page.delay(
                        document_id, page_num, page_array.tolist()
                    )
                    page_tasks.append(('tables', table_task))

                # Checkbox detection task (if enabled)
                if options.get('enable_checkbox_detection', True) and classification['has_checkboxes']:
                    checkbox_task = detect_checkboxes_on_page.delay(
                        document_id, page_num, page_array.tolist()
                    )
                    page_tasks.append(('checkboxes', checkbox_task))

                # Handwriting extraction task (if enabled)
                if options.get('extract_handwriting', True) and classification['has_handwriting']:
                    handwriting_task = extract_handwriting_from_page.delay(
                        document_id, page_num, page_array.tolist()
                    )
                    page_tasks.append(('handwriting', handwriting_task))

                # Collect results from all tasks
                page_result = {
                    'page_number': page_num,
                    'text': [],
                    'tables': [],
                    'checkboxes': [],
                    'handwriting': []
                }

                # Wait for all tasks to complete with timeout
                for task_type, task in page_tasks:
                    try:
                        result = task.get(timeout=300)  # 5 minute timeout per task
                        page_result[task_type] = result
                    except Exception as e:
                        logger.error(f"Task {task_type} failed for page {page_num}: {e}")
                        page_result[task_type] = []

                page_results.append(page_result)

                # Update progress
                processor.update_progress(document_id, "processing", page_num, total_pages)

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                # Continue with other pages
                continue

        # Step 4: Quality validation
        logger.info("Performing quality validation")
        validated_results = processor.quality_validator.validate_results(
            page_results, options.get('confidence_threshold', 0.8)
        )

        # Step 5: Assemble final JSON
        logger.info("Assembling final JSON")
        final_result = processor.json_assembler.assemble_document_json(
            document_id=document_id,
            page_results=validated_results,
            classification=classification,
            processing_metadata={
                'total_pages': total_pages,
                'processing_time': 0,  # Will be calculated
                'options': options
            }
        )

        # Step 6: Store final result
        processor.redis_client.set(
            f"result:{document_id}", 
            json.dumps(final_result), 
            ex=86400  # 24 hour TTL
        )

        # Update final status
        processor.update_progress(document_id, "completed", total_pages, total_pages)

        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.info(f"Document processing completed for {document_id}")
        return final_result

    except Exception as e:
        error_message = f"Document processing failed: {str(e)}"
        logger.error(f"Error in process_document_task: {error_message}")
        logger.error(traceback.format_exc())

        # Update error status
        processor.update_progress(document_id, "failed", 0, 1, error_message)

        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(name='extract_text_from_page')
def extract_text_from_page(document_id: str, page_num: int, page_array: List, confidence_threshold: float):
    """Extract text from a single page"""
    try:
        # Convert list back to numpy array
        page_image = np.array(page_array, dtype=np.uint8)

        # Initialize text service
        text_service = TextExtractionService()

        # Extract text
        result = text_service.extract_text(page_image, confidence_threshold)

        logger.info(f"Text extraction completed for page {page_num} of document {document_id}")
        return result

    except Exception as e:
        logger.error(f"Text extraction failed for page {page_num}: {e}")
        return []

@celery_app.task(name='extract_tables_from_page')
def extract_tables_from_page(document_id: str, page_num: int, page_array: List):
    """Extract tables from a single page"""
    try:
        page_image = np.array(page_array, dtype=np.uint8)

        table_service = TableExtractionService()
        result = table_service.extract_tables(page_image)

        logger.info(f"Table extraction completed for page {page_num} of document {document_id}")
        return result

    except Exception as e:
        logger.error(f"Table extraction failed for page {page_num}: {e}")
        return []

@celery_app.task(name='detect_checkboxes_on_page')
def detect_checkboxes_on_page(document_id: str, page_num: int, page_array: List):
    """Detect checkboxes on a single page"""
    try:
        page_image = np.array(page_array, dtype=np.uint8)

        checkbox_service = CheckboxDetectionService()
        result = checkbox_service.detect_checkboxes(page_image)

        logger.info(f"Checkbox detection completed for page {page_num} of document {document_id}")
        return result

    except Exception as e:
        logger.error(f"Checkbox detection failed for page {page_num}: {e}")
        return []

@celery_app.task(name='extract_handwriting_from_page')  
def extract_handwriting_from_page(document_id: str, page_num: int, page_array: List):
    """Extract handwriting from a single page using Qwen2.5VL"""
    try:
        page_image = np.array(page_array, dtype=np.uint8)

        handwriting_service = HandwritingExtractionService()
        result = handwriting_service.extract_handwriting(page_image)

        logger.info(f"Handwriting extraction completed for page {page_num} of document {document_id}")
        return result

    except Exception as e:
        logger.error(f"Handwriting extraction failed for page {page_num}: {e}")
        return []
