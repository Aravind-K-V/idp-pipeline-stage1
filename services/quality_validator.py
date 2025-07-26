# services/quality_validator.py - Quality Validation Service
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re

logger = logging.getLogger(__name__)

class QualityValidator:
    """Service for validating extraction quality and determining retry needs"""

    def __init__(self):
        self.min_confidence_threshold = 0.8
        self.retry_threshold = 0.6
        self.max_retries = 3

    def validate_extraction_results(self, results: Dict[str, Any], page_number: int) -> Dict[str, Any]:
        """
        Validate extraction results for a single page

        Args:
            results: Dictionary containing results from all services
            page_number: Page number being validated

        Returns:
            Dictionary with validation results and retry recommendations
        """
        try:
            validation_report = {
                'page_number': page_number,
                'overall_quality': 'good',
                'confidence_scores': {},
                'quality_issues': [],
                'retry_recommended': False,
                'retry_services': [],
                'validation_details': {}
            }

            # Validate each service result
            for service_name, service_result in results.items():
                if service_result and isinstance(service_result, dict):
                    service_validation = self._validate_service_result(service_name, service_result)
                    validation_report['confidence_scores'][service_name] = service_validation['confidence']
                    validation_report['validation_details'][service_name] = service_validation

                    # Check if retry is needed
                    if service_validation['needs_retry']:
                        validation_report['retry_recommended'] = True
                        validation_report['retry_services'].append(service_name)

                    # Collect quality issues
                    validation_report['quality_issues'].extend(service_validation.get('issues', []))

            # Determine overall quality
            validation_report['overall_quality'] = self._determine_overall_quality(
                validation_report['confidence_scores']
            )

            return validation_report

        except Exception as e:
            logger.error(f"Validation failed for page {page_number}: {e}")
            return {
                'page_number': page_number,
                'overall_quality': 'error',
                'error': str(e),
                'retry_recommended': True,
                'retry_services': list(results.keys())
            }

    def _validate_service_result(self, service_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate results from a specific service"""
        try:
            validation = {
                'service': service_name,
                'confidence': result.get('confidence', 0.0),
                'needs_retry': False,
                'issues': [],
                'metrics': {}
            }

            # Service-specific validation
            if service_name == 'text':
                validation.update(self._validate_text_results(result))
            elif service_name == 'table':
                validation.update(self._validate_table_results(result))
            elif service_name == 'checkbox':
                validation.update(self._validate_checkbox_results(result))
            elif service_name == 'handwriting':
                validation.update(self._validate_handwriting_results(result))

            # General confidence check
            if validation['confidence'] < self.min_confidence_threshold:
                if validation['confidence'] >= self.retry_threshold:
                    validation['needs_retry'] = True
                    validation['issues'].append(f"Low confidence: {validation['confidence']:.2f}")
                else:
                    validation['issues'].append(f"Very low confidence: {validation['confidence']:.2f}")

            return validation

        except Exception as e:
            logger.error(f"Service validation failed for {service_name}: {e}")
            return {
                'service': service_name,
                'confidence': 0.0,
                'needs_retry': True,
                'issues': [f"Validation error: {str(e)}"],
                'metrics': {}
            }

    def _validate_text_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text extraction results"""
        validation = {'issues': [], 'metrics': {}}

        text_blocks = result.get('text_blocks', [])

        # Check if any text was found
        if not text_blocks:
            validation['issues'].append("No text blocks extracted")
            validation['confidence'] = 0.0
            validation['needs_retry'] = True
        else:
            # Calculate metrics
            total_chars = sum(len(block.get('text', '')) for block in text_blocks)
            avg_confidence = np.mean([block.get('confidence', 0) for block in text_blocks])

            validation['metrics'] = {
                'text_blocks_count': len(text_blocks),
                'total_characters': total_chars,
                'average_confidence': avg_confidence
            }

            # Quality checks
            if total_chars < 10:
                validation['issues'].append("Very little text extracted")

            if avg_confidence < 0.5:
                validation['issues'].append("Low average text confidence")
                validation['needs_retry'] = True

        return validation

    def _validate_table_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate table extraction results"""
        validation = {'issues': [], 'metrics': {}}

        tables = result.get('tables', [])

        # Metrics
        validation['metrics'] = {
            'tables_found': len(tables),
            'total_cells': sum(len(table.get('content', [])) for table in tables if table.get('content'))
        }

        # Quality checks
        for i, table in enumerate(tables):
            if not table.get('content'):
                validation['issues'].append(f"Table {i+1}: No content extracted")

            table_confidence = table.get('confidence', 0)
            if table_confidence < 0.6:
                validation['issues'].append(f"Table {i+1}: Low confidence ({table_confidence:.2f})")

        return validation

    def _validate_checkbox_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate checkbox detection results"""
        validation = {'issues': [], 'metrics': {}}

        checkboxes = result.get('checkboxes', [])

        # Metrics
        checked_count = sum(1 for cb in checkboxes if cb.get('state') == 'checked')
        unchecked_count = sum(1 for cb in checkboxes if cb.get('state') == 'unchecked')
        unknown_count = sum(1 for cb in checkboxes if cb.get('state') == 'unknown')

        validation['metrics'] = {
            'checkboxes_found': len(checkboxes),
            'checked': checked_count,
            'unchecked': unchecked_count,
            'unknown_state': unknown_count
        }

        # Quality checks
        if unknown_count > len(checkboxes) * 0.3:  # More than 30% unknown
            validation['issues'].append("High percentage of checkboxes with unknown state")
            validation['needs_retry'] = True

        # Check for reasonable text associations
        unassociated_count = sum(1 for cb in checkboxes if not cb.get('associated_text', '').strip())
        if unassociated_count > len(checkboxes) * 0.5:  # More than 50% without text
            validation['issues'].append("Many checkboxes without associated text")

        return validation

    def _validate_handwriting_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate handwriting extraction results"""
        validation = {'issues': [], 'metrics': {}}

        handwriting = result.get('handwriting', [])

        # Metrics
        total_chars = sum(len(hw.get('text', '')) for hw in handwriting)
        avg_confidence = np.mean([hw.get('confidence', 0) for hw in handwriting]) if handwriting else 0

        validation['metrics'] = {
            'handwriting_regions': len(handwriting),
            'total_characters': total_chars,
            'average_confidence': avg_confidence
        }

        # Quality checks
        for i, hw in enumerate(handwriting):
            text = hw.get('text', '').strip()
            confidence = hw.get('confidence', 0)

            # Check for suspicious patterns that might indicate OCR errors
            if self._contains_ocr_artifacts(text):
                validation['issues'].append(f"Handwriting {i+1}: Contains OCR artifacts")

            if confidence < 0.4:
                validation['issues'].append(f"Handwriting {i+1}: Very low confidence ({confidence:.2f})")
                validation['needs_retry'] = True

        return validation

    def _contains_ocr_artifacts(self, text: str) -> bool:
        """Check if text contains common OCR artifacts"""
        if not text:
            return False

        # Common OCR artifacts
        artifacts = [
            r'[|]{3,}',  # Multiple vertical bars
            r'[.]{5,}',  # Multiple dots
            r'[#]{3,}',  # Multiple hashes
            r'[?]{3,}',  # Multiple question marks
            r'[\\/]{3,}',  # Multiple slashes
        ]

        for pattern in artifacts:
            if re.search(pattern, text):
                return True

        # Check for very high ratio of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0 and special_chars / len(text) > 0.5:
            return True

        return False

    def _determine_overall_quality(self, confidence_scores: Dict[str, float]) -> str:
        """Determine overall quality based on service confidence scores"""
        if not confidence_scores:
            return 'poor'

        avg_confidence = np.mean(list(confidence_scores.values()))
        min_confidence = min(confidence_scores.values())

        if avg_confidence >= 0.9 and min_confidence >= 0.7:
            return 'excellent'
        elif avg_confidence >= 0.8 and min_confidence >= 0.6:
            return 'good'
        elif avg_confidence >= 0.6 and min_confidence >= 0.4:
            return 'fair'
        else:
            return 'poor'

    def should_retry_page(self, validation_report: Dict[str, Any], current_retry_count: int) -> bool:
        """Determine if a page should be retried"""
        if current_retry_count >= self.max_retries:
            return False

        if validation_report.get('retry_recommended', False):
            return True

        # Check overall quality
        overall_quality = validation_report.get('overall_quality', 'poor')
        if overall_quality in ['poor', 'error']:
            return True

        return False

    def get_retry_configuration(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for retry attempt"""
        retry_config = {
            'use_high_dpi': True,  # Use higher DPI for retry
            'services_to_retry': validation_report.get('retry_services', []),
            'preprocessing_enhanced': True,
            'confidence_threshold_relaxed': True
        }

        # Service-specific retry configurations
        for service in retry_config['services_to_retry']:
            if service == 'text':
                retry_config['text_ocr_engine_fallback'] = True
            elif service == 'handwriting':
                retry_config['handwriting_enhanced_preprocessing'] = True
            elif service == 'table':
                retry_config['table_fallback_method'] = True

        return retry_config
