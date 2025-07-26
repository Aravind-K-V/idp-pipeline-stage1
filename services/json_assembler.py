# services/json_assembler.py - JSON Schema Assembly Service
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class JSONSchemaAssembler:
    """Service for assembling extraction results into standardized JSON schema"""

    def __init__(self):
        self.schema_version = "1.0.0"

    def assemble_document_results(
        self, 
        document_id: str,
        page_results: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assemble complete document results into standardized JSON schema

        Args:
            document_id: Unique document identifier
            page_results: List of page-level extraction results
            document_metadata: Document metadata (file info, etc.)
            processing_metadata: Processing statistics and metadata

        Returns:
            Complete document results in standardized JSON format
        """
        try:
            # Sort pages by page number
            sorted_pages = sorted(page_results, key=lambda x: x.get('page_number', 0))

            # Assemble the complete document structure
            document_json = {
                "schema_version": self.schema_version,
                "document_id": document_id,
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "document_metadata": self._process_document_metadata(document_metadata),
                "processing_metadata": self._process_processing_metadata(processing_metadata),
                "summary": self._generate_document_summary(sorted_pages),
                "pages": [self._assemble_page_results(page) for page in sorted_pages],
                "quality_report": self._generate_quality_report(sorted_pages)
            }

            # Validate the assembled JSON
            validation_result = self._validate_json_schema(document_json)
            if not validation_result['valid']:
                logger.warning(f"JSON schema validation issues: {validation_result['issues']}")

            return document_json

        except Exception as e:
            logger.error(f"JSON assembly failed for document {document_id}: {e}")
            return self._create_error_response(document_id, str(e))

    def _process_document_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and standardize document metadata"""
        return {
            "filename": metadata.get("filename", "unknown"),
            "file_size": metadata.get("file_size", 0),
            "file_hash": metadata.get("file_hash", ""),
            "mime_type": metadata.get("mime_type", "application/pdf"),
            "total_pages": metadata.get("page_count", 0),
            "pdf_metadata": metadata.get("pdf_metadata", {}),
            "upload_timestamp": metadata.get("upload_timestamp", datetime.utcnow().isoformat())
        }

    def _process_processing_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and standardize processing metadata"""
        return {
            "total_processing_time": metadata.get("total_processing_time", 0.0),
            "services_used": metadata.get("services_used", []),
            "retry_count": metadata.get("retry_count", 0),
            "processing_mode": metadata.get("processing_mode", "parallel"),
            "model_versions": metadata.get("model_versions", {}),
            "processing_parameters": metadata.get("processing_parameters", {}),
            "worker_info": metadata.get("worker_info", {})
        }

    def _generate_document_summary(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level document summary"""
        try:
            total_text_blocks = sum(len(page.get('text', {}).get('text_blocks', [])) for page in pages)
            total_tables = sum(page.get('table', {}).get('table_count', 0) for page in pages)
            total_checkboxes = sum(page.get('checkbox', {}).get('checkbox_count', 0) for page in pages)
            total_handwriting = sum(page.get('handwriting', {}).get('handwriting_count', 0) for page in pages)

            # Calculate average confidence scores
            avg_confidences = {}
            for service in ['text', 'table', 'checkbox', 'handwriting']:
                confidences = [
                    page.get(service, {}).get('confidence', 0) 
                    for page in pages 
                    if page.get(service, {}).get('confidence', 0) > 0
                ]
                avg_confidences[service] = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "total_pages": len(pages),
                "content_summary": {
                    "text_blocks": total_text_blocks,
                    "tables": total_tables,
                    "checkboxes": total_checkboxes,
                    "handwriting_regions": total_handwriting
                },
                "average_confidence_scores": avg_confidences,
                "overall_confidence": sum(avg_confidences.values()) / len(avg_confidences) if avg_confidences else 0.0
            }

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}

    def _assemble_page_results(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble results for a single page"""
        try:
            page_number = page_data.get('page_number', 0)

            assembled_page = {
                "page_number": page_number,
                "processing_timestamp": page_data.get('timestamp', datetime.utcnow().isoformat()),
                "content": {
                    "text": self._process_text_results(page_data.get('text', {})),
                    "tables": self._process_table_results(page_data.get('table', {})),
                    "checkboxes": self._process_checkbox_results(page_data.get('checkbox', {})),
                    "handwriting": self._process_handwriting_results(page_data.get('handwriting', {}))
                },
                "confidence_scores": {
                    "text": page_data.get('text', {}).get('confidence', 0.0),
                    "table": page_data.get('table', {}).get('confidence', 0.0),
                    "checkbox": page_data.get('checkbox', {}).get('confidence', 0.0),
                    "handwriting": page_data.get('handwriting', {}).get('confidence', 0.0)
                },
                "processing_times": {
                    "text": page_data.get('text', {}).get('processing_time', 0.0),
                    "table": page_data.get('table', {}).get('processing_time', 0.0),
                    "checkbox": page_data.get('checkbox', {}).get('processing_time', 0.0),
                    "handwriting": page_data.get('handwriting', {}).get('processing_time', 0.0)
                },
                "quality_metrics": page_data.get('quality_validation', {}),
                "retry_count": page_data.get('retry_count', 0)
            }

            return assembled_page

        except Exception as e:
            logger.error(f"Page assembly failed for page {page_data.get('page_number', 'unknown')}: {e}")
            return {
                "page_number": page_data.get('page_number', 0),
                "error": str(e),
                "content": {"text": [], "tables": [], "checkboxes": [], "handwriting": []}
            }

    def _process_text_results(self, text_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process text extraction results"""
        text_blocks = text_data.get('text_blocks', [])
        processed_blocks = []

        for block in text_blocks:
            processed_block = {
                "text": block.get('text', '').strip(),
                "confidence": round(block.get('confidence', 0.0), 3),
                "bbox": self._normalize_bbox(block.get('bbox', {})),
                "properties": {
                    "font_size": block.get('font_size'),
                    "font_type": block.get('font_type'),
                    "is_bold": block.get('is_bold', False),
                    "is_italic": block.get('is_italic', False)
                }
            }

            # Remove None values
            processed_block['properties'] = {k: v for k, v in processed_block['properties'].items() if v is not None}

            if processed_block['text']:  # Only include non-empty text
                processed_blocks.append(processed_block)

        return processed_blocks

    def _process_table_results(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process table extraction results"""
        tables = table_data.get('tables', [])
        processed_tables = []

        for i, table in enumerate(tables):
            processed_table = {
                "table_id": f"table_{i+1}",
                "bbox": self._normalize_bbox(table.get('bbox', {})),
                "confidence": round(table.get('confidence', 0.0), 3),
                "structure": {
                    "rows": len(table.get('content', [])),
                    "columns": len(table.get('content', [[]])[0]) if table.get('content') else 0,
                    "has_header": table.get('has_header', False)
                },
                "content": table.get('content', []),
                "csv_data": table.get('csv_data', ''),
                "extraction_method": table.get('method', 'unknown')
            }

            processed_tables.append(processed_table)

        return processed_tables

    def _process_checkbox_results(self, checkbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process checkbox detection results"""
        checkboxes = checkbox_data.get('checkboxes', [])
        processed_checkboxes = []

        for i, checkbox in enumerate(checkboxes):
            processed_checkbox = {
                "checkbox_id": f"checkbox_{i+1}",
                "bbox": self._normalize_bbox(checkbox.get('bbox', {})),
                "state": checkbox.get('state', 'unknown'),
                "confidence": round(checkbox.get('confidence', 0.0), 3),
                "associated_text": checkbox.get('associated_text', '').strip(),
                "text_confidence": round(checkbox.get('text_confidence', 0.0), 3),
                "text_distance": checkbox.get('text_distance', 0),
                "detection_method": checkbox.get('method', 'unknown')
            }

            processed_checkboxes.append(processed_checkbox)

        return processed_checkboxes

    def _process_handwriting_results(self, handwriting_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process handwriting extraction results"""
        handwriting = handwriting_data.get('handwriting', [])
        processed_handwriting = []

        for i, hw in enumerate(handwriting):
            processed_hw = {
                "handwriting_id": f"handwriting_{i+1}",
                "text": hw.get('text', '').strip(),
                "confidence": round(hw.get('confidence', 0.0), 3),
                "bbox": self._normalize_bbox(hw.get('bbox', {})),
                "extraction_method": hw.get('method', 'unknown'),
                "language": hw.get('language', 'en'),
                "region_score": hw.get('region_score', 0.0)
            }

            if processed_hw['text']:  # Only include non-empty handwriting
                processed_handwriting.append(processed_hw)

        return processed_handwriting

    def _normalize_bbox(self, bbox: Dict[str, Any]) -> Dict[str, int]:
        """Normalize bounding box format"""
        return {
            "x": int(bbox.get('x', 0)),
            "y": int(bbox.get('y', 0)),
            "width": int(bbox.get('width', 0)),
            "height": int(bbox.get('height', 0))
        }

    def _generate_quality_report(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall quality report for the document"""
        try:
            page_qualities = []
            service_issues = []
            retry_pages = []

            for page in pages:
                page_num = page.get('page_number', 0)
                quality_data = page.get('quality_validation', {})

                if quality_data:
                    page_qualities.append(quality_data.get('overall_quality', 'unknown'))

                    if quality_data.get('quality_issues'):
                        service_issues.extend([
                            f"Page {page_num}: {issue}" 
                            for issue in quality_data['quality_issues']
                        ])

                    if quality_data.get('retry_recommended'):
                        retry_pages.append(page_num)

            # Calculate quality distribution
            quality_counts = {}
            for quality in page_qualities:
                quality_counts[quality] = quality_counts.get(quality, 0) + 1

            return {
                "total_pages_processed": len(pages),
                "quality_distribution": quality_counts,
                "pages_requiring_retry": retry_pages,
                "total_issues": len(service_issues),
                "service_issues": service_issues[:50],  # Limit to first 50 issues
                "overall_document_quality": self._determine_document_quality(page_qualities)
            }

        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            return {"error": str(e)}

    def _determine_document_quality(self, page_qualities: List[str]) -> str:
        """Determine overall document quality"""
        if not page_qualities:
            return 'unknown'

        quality_scores = {
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1,
            'error': 0,
            'unknown': 0
        }

        avg_score = sum(quality_scores.get(q, 0) for q in page_qualities) / len(page_qualities)

        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'fair'
        else:
            return 'poor'

    def _validate_json_schema(self, document_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the assembled JSON schema"""
        validation = {'valid': True, 'issues': []}

        try:
            # Check required fields
            required_fields = ['document_id', 'pages', 'summary']
            for field in required_fields:
                if field not in document_json:
                    validation['issues'].append(f"Missing required field: {field}")
                    validation['valid'] = False

            # Validate pages
            if 'pages' in document_json:
                for i, page in enumerate(document_json['pages']):
                    if 'page_number' not in page:
                        validation['issues'].append(f"Page {i}: Missing page_number")
                        validation['valid'] = False

                    if 'content' not in page:
                        validation['issues'].append(f"Page {i}: Missing content")
                        validation['valid'] = False

            # Check JSON serializability
            try:
                json.dumps(document_json)
            except (TypeError, ValueError) as e:
                validation['issues'].append(f"JSON serialization error: {str(e)}")
                validation['valid'] = False

        except Exception as e:
            validation['valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")

        return validation

    def _create_error_response(self, document_id: str, error_message: str) -> Dict[str, Any]:
        """Create error response for failed assembly"""
        return {
            "schema_version": self.schema_version,
            "document_id": document_id,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "status": "error",
            "error_message": error_message,
            "pages": [],
            "summary": {
                "total_pages": 0,
                "content_summary": {
                    "text_blocks": 0,
                    "tables": 0,
                    "checkboxes": 0,
                    "handwriting_regions": 0
                },
                "overall_confidence": 0.0
            }
        }

    def create_minimal_schema(self, document_id: str, error: str) -> Dict[str, Any]:
        """Create minimal schema for failed processing"""
        return {
            "schema_version": self.schema_version,
            "document_id": document_id,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": error,
            "pages": [],
            "summary": {"total_pages": 0, "overall_confidence": 0.0}
        }
