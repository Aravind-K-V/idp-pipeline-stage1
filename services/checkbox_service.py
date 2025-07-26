# services/checkbox_service.py - Checkbox Detection Service
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import time
from PIL import Image
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed")
    YOLO_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract not installed")
    TESSERACT_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)

class CheckboxDetectionService:
    """Service for detecting checkboxes and their states in document images"""

    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = getattr(settings, 'CHECKBOX_CONFIDENCE_THRESHOLD', 0.5)

        # Initialize models
        self._initialize_models()

        logger.info(f"CheckboxDetectionService initialized with device: {self.device}")

    def _initialize_models(self):
        """Initialize checkbox detection models"""
        try:
            # Method 1: YOLO for checkbox detection
            if YOLO_AVAILABLE:
                try:
                    # Try to load pre-trained YOLO model
                    # In production, you would use a fine-tuned model for checkboxes
                    self.yolo_model = YOLO('yolov8s.pt')  # Start with base model
                    self.use_yolo = True
                    logger.info("YOLO model initialized successfully")
                except Exception as e:
                    logger.warning(f"YOLO initialization failed: {e}")
                    self.use_yolo = False
            else:
                self.use_yolo = False

            # Method 2: OpenCV contour detection (fallback)
            self.use_opencv = True

            # For associating checkboxes with text
            self.min_checkbox_size = 10
            self.max_checkbox_size = 50
            self.checkbox_aspect_ratio_range = (0.5, 2.0)

        except Exception as e:
            logger.error(f"Error initializing checkbox models: {e}")
            raise

    def detect_checkboxes(self, image: np.ndarray, page_number: int = 1) -> Dict[str, Any]:
        """
        Detect checkboxes and their states in the image

        Args:
            image: Input image as numpy array
            page_number: Page number for metadata

        Returns:
            Dictionary containing detected checkboxes with states and associated text
        """
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Detect checkboxes using available methods
            checkboxes = []

            # Try YOLO first if available
            if self.use_yolo:
                yolo_checkboxes = self._detect_with_yolo(image)
                checkboxes.extend(yolo_checkboxes)

            # If no checkboxes found or YOLO not available, use OpenCV
            if len(checkboxes) == 0 and self.use_opencv:
                opencv_checkboxes = self._detect_with_opencv(processed_image)
                checkboxes.extend(opencv_checkboxes)

            # Determine checkbox states (checked/unchecked)
            for checkbox in checkboxes:
                checkbox['state'] = self._determine_checkbox_state(
                    processed_image, checkbox['bbox']
                )

            # Associate checkboxes with nearby text
            text_associations = self._associate_with_text(image, checkboxes)

            # Merge checkbox data with text associations
            for i, checkbox in enumerate(checkboxes):
                if i < len(text_associations):
                    checkbox.update(text_associations[i])

            processing_time = time.time() - start_time

            result = {
                'checkboxes': checkboxes,
                'checkbox_count': len(checkboxes),
                'page_number': page_number,
                'processing_time': processing_time,
                'timestamp': time.time()
            }

            logger.info(f"Checkbox detection completed in {processing_time:.2f}s, found {len(checkboxes)} checkboxes")
            return result

        except Exception as e:
            logger.error(f"Error in checkbox detection: {e}")
            return {
                'checkboxes': [],
                'checkbox_count': 0,
                'error': str(e),
                'page_number': page_number,
                'processing_time': time.time() - start_time
            }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better checkbox detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            return thresh

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect checkboxes using YOLO model"""
        try:
            if not self.use_yolo:
                return []

            # Convert to PIL Image for YOLO
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')

            # Run YOLO inference
            results = self.yolo_model(pil_image, conf=self.confidence_threshold)

            checkboxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Filter for checkbox-like objects
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 1

                        if (self.min_checkbox_size <= width <= self.max_checkbox_size and
                            self.min_checkbox_size <= height <= self.max_checkbox_size and
                            self.checkbox_aspect_ratio_range[0] <= aspect_ratio <= self.checkbox_aspect_ratio_range[1]):

                            checkbox = {
                                'bbox': {
                                    'x': int(x1),
                                    'y': int(y1),
                                    'width': int(width),
                                    'height': int(height)
                                },
                                'confidence': confidence,
                                'method': 'yolo',
                                'class_id': class_id
                            }
                            checkboxes.append(checkbox)

            return checkboxes

        except Exception as e:
            logger.warning(f"YOLO checkbox detection failed: {e}")
            return []

    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect checkboxes using OpenCV contour detection"""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            checkboxes = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size and aspect ratio
                aspect_ratio = w / h if h > 0 else 1
                area = cv2.contourArea(contour)
                rect_area = w * h

                # Check if it looks like a checkbox
                if (self.min_checkbox_size <= w <= self.max_checkbox_size and
                    self.min_checkbox_size <= h <= self.max_checkbox_size and
                    self.checkbox_aspect_ratio_range[0] <= aspect_ratio <= self.checkbox_aspect_ratio_range[1] and
                    area > 50 and  # Minimum area
                    area / rect_area > 0.5):  # Should be reasonably filled

                    # Additional shape analysis
                    # Check if contour is roughly rectangular
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                    if len(approx) >= 4:  # Should have at least 4 corners
                        checkbox = {
                            'bbox': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            },
                            'confidence': min(0.8, area / (w * h)),  # Confidence based on fill ratio
                            'method': 'opencv',
                            'contour_area': float(area),
                            'corners': len(approx)
                        }
                        checkboxes.append(checkbox)

            # Remove duplicate detections (overlapping bounding boxes)
            checkboxes = self._remove_duplicate_checkboxes(checkboxes)

            return checkboxes

        except Exception as e:
            logger.error(f"OpenCV checkbox detection failed: {e}")
            return []

    def _remove_duplicate_checkboxes(self, checkboxes: List[Dict[str, Any]], overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Remove duplicate checkbox detections"""
        try:
            if len(checkboxes) <= 1:
                return checkboxes

            # Sort by confidence
            checkboxes.sort(key=lambda x: x['confidence'], reverse=True)

            filtered_checkboxes = []
            for checkbox in checkboxes:
                is_duplicate = False

                for existing in filtered_checkboxes:
                    # Calculate IoU
                    iou = self._calculate_iou(checkbox['bbox'], existing['bbox'])
                    if iou > overlap_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_checkboxes.append(checkbox)

            return filtered_checkboxes

        except Exception as e:
            logger.warning(f"Duplicate removal failed: {e}")
            return checkboxes

    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        try:
            # Calculate intersection
            x1 = max(bbox1['x'], bbox2['x'])
            y1 = max(bbox1['y'], bbox2['y'])
            x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
            y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)

            # Calculate union
            area1 = bbox1['width'] * bbox1['height']
            area2 = bbox2['width'] * bbox2['height']
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.warning(f"IoU calculation failed: {e}")
            return 0.0

    def _determine_checkbox_state(self, image: np.ndarray, bbox: Dict[str, int]) -> str:
        """Determine if a checkbox is checked or unchecked"""
        try:
            # Extract checkbox region
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            checkbox_roi = image[y:y+h, x:x+w]

            if checkbox_roi.size == 0:
                return 'unknown'

            # Method 1: Analyze pixel density in the center region
            center_margin = max(2, min(w, h) // 4)
            center_roi = checkbox_roi[
                center_margin:h-center_margin,
                center_margin:w-center_margin
            ]

            if center_roi.size > 0:
                # Count dark pixels (assuming checkmarks are dark)
                dark_pixels = np.sum(center_roi < 128)
                total_pixels = center_roi.size
                fill_ratio = dark_pixels / total_pixels

                # Threshold for determining checked state
                if fill_ratio > 0.3:  # 30% dark pixels indicates checked
                    return 'checked'
                elif fill_ratio < 0.1:  # Less than 10% indicates unchecked
                    return 'unchecked'
                else:
                    return 'partially_checked'

            # Method 2: Edge density analysis
            edges = cv2.Canny(checkbox_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density > 0.1:  # High edge density might indicate checkmark
                return 'checked'
            else:
                return 'unchecked'

        except Exception as e:
            logger.warning(f"Checkbox state determination failed: {e}")
            return 'unknown'

    def _associate_with_text(self, image: np.ndarray, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Associate checkboxes with nearby text"""
        try:
            if not TESSERACT_AVAILABLE:
                return [{'associated_text': '', 'text_confidence': 0.0} for _ in checkboxes]

            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6'
            )

            # Extract text blocks
            text_blocks = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and int(ocr_data['conf'][i]) > 30:  # Minimum confidence
                    text_blocks.append({
                        'text': text,
                        'bbox': {
                            'x': ocr_data['left'][i],
                            'y': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        },
                        'confidence': int(ocr_data['conf'][i]) / 100.0
                    })

            # Associate each checkbox with nearby text
            associations = []
            for checkbox in checkboxes:
                associated_text = self._find_nearest_text(checkbox, text_blocks)
                associations.append(associated_text)

            return associations

        except Exception as e:
            logger.warning(f"Text association failed: {e}")
            return [{'associated_text': '', 'text_confidence': 0.0} for _ in checkboxes]

    def _find_nearest_text(self, checkbox: Dict[str, Any], text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the nearest text to a checkbox"""
        try:
            checkbox_center_x = checkbox['bbox']['x'] + checkbox['bbox']['width'] // 2
            checkbox_center_y = checkbox['bbox']['y'] + checkbox['bbox']['height'] // 2

            best_text = ""
            best_confidence = 0.0
            min_distance = float('inf')

            for text_block in text_blocks:
                # Calculate distance from checkbox to text
                text_center_x = text_block['bbox']['x'] + text_block['bbox']['width'] // 2
                text_center_y = text_block['bbox']['y'] + text_block['bbox']['height'] // 2

                distance = np.sqrt(
                    (checkbox_center_x - text_center_x) ** 2 + 
                    (checkbox_center_y - text_center_y) ** 2
                )

                # Prefer text to the right or below the checkbox
                x_diff = text_center_x - checkbox_center_x
                y_diff = text_center_y - checkbox_center_y

                # Weight the distance based on position preference
                if x_diff > 0 and abs(y_diff) < 50:  # Text to the right
                    distance *= 0.5
                elif y_diff > 0 and abs(x_diff) < 100:  # Text below
                    distance *= 0.7

                # Consider only text within reasonable distance
                if distance < min_distance and distance < 200:  # Max 200 pixels
                    min_distance = distance
                    best_text = text_block['text']
                    best_confidence = text_block['confidence']

            return {
                'associated_text': best_text,
                'text_confidence': best_confidence,
                'text_distance': min_distance if min_distance != float('inf') else 0
            }

        except Exception as e:
            logger.warning(f"Nearest text search failed: {e}")
            return {'associated_text': '', 'text_confidence': 0.0, 'text_distance': 0}

    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'checkbox_detection',
            'status': 'healthy',
            'device': self.device,
            'models_loaded': {
                'yolo': self.use_yolo,
                'opencv': self.use_opencv,
                'tesseract': TESSERACT_AVAILABLE
            },
            'confidence_threshold': self.confidence_threshold
        }
