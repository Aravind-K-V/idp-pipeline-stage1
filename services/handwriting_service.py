# services/handwriting_service.py - Handwriting Extraction Service
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import time
from PIL import Image
import json
import base64
import io

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    print("Warning: Qwen2VL not available - install transformers>=4.37.0 and qwen-vl-utils")
    QWEN_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract not installed")
    TESSERACT_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)

class HandwritingExtractionService:
    """Service for extracting handwritten text using Qwen2.5-VL-7B"""

    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.model_name = getattr(settings, 'HANDWRITING_MODEL', 'Qwen/Qwen2-VL-7B-Instruct')
        self.min_pixels = getattr(settings, 'HANDWRITING_MIN_PIXELS', 256*28*28)

        # Initialize models
        self._initialize_models()

        logger.info(f"HandwritingExtractionService initialized with device: {self.device}")

    def _initialize_models(self):
        """Initialize Qwen2.5-VL model for handwriting recognition"""
        try:
            if QWEN_AVAILABLE:
                logger.info(f"Loading Qwen2VL model: {self.model_name}")

                # Load model with optimizations for inference
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
                )

                if self.device != "cuda":
                    self.model.to(self.device)

                self.processor = AutoProcessor.from_pretrained(self.model_name)

                self.model.eval()
                self.use_qwen = True
                logger.info("Qwen2VL model loaded successfully")

                # Set generation config
                self.generation_config = {
                    'max_new_tokens': 512,
                    'do_sample': False,  # Deterministic for OCR
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'repetition_penalty': 1.1
                }

            else:
                self.use_qwen = False
                logger.warning("Qwen2VL not available, using fallback methods")

            # Fallback to Tesseract for handwriting (less accurate but available)
            self.use_tesseract_fallback = TESSERACT_AVAILABLE

        except Exception as e:
            logger.error(f"Error initializing handwriting models: {e}")
            self.use_qwen = False
            self.use_tesseract_fallback = TESSERACT_AVAILABLE

    def extract_handwriting(self, image: np.ndarray, page_number: int = 1) -> Dict[str, Any]:
        """
        Extract handwritten text from image

        Args:
            image: Input image as numpy array
            page_number: Page number for metadata

        Returns:
            Dictionary containing extracted handwriting with positions and confidence
        """
        start_time = time.time()

        try:
            # Detect handwriting regions first
            handwriting_regions = self._detect_handwriting_regions(image)

            # Extract text from each region
            handwriting_results = []
            for i, region in enumerate(handwriting_regions):
                try:
                    if self.use_qwen:
                        result = self._extract_with_qwen(region['image'], region['bbox'])
                    else:
                        result = self._extract_with_fallback(region['image'], region['bbox'])

                    if result and result.get('text', '').strip():
                        result['region_index'] = i
                        result.update(region['bbox'])
                        handwriting_results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to extract handwriting from region {i}: {e}")
                    continue

            processing_time = time.time() - start_time

            result = {
                'handwriting': handwriting_results,
                'handwriting_count': len(handwriting_results),
                'regions_detected': len(handwriting_regions),
                'page_number': page_number,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'model_used': 'qwen2vl' if self.use_qwen else 'fallback'
            }

            logger.info(f"Handwriting extraction completed in {processing_time:.2f}s, found {len(handwriting_results)} handwritten texts")
            return result

        except Exception as e:
            logger.error(f"Error in handwriting extraction: {e}")
            return {
                'handwriting': [],
                'handwriting_count': 0,
                'error': str(e),
                'page_number': page_number,
                'processing_time': time.time() - start_time
            }

    def _detect_handwriting_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regions that likely contain handwriting"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply preprocessing for handwriting detection
            # Different from printed text preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use adaptive threshold to handle varying lighting
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
            )

            # Morphological operations to connect handwritten strokes
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

            # Detect horizontal text lines
            horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal)
            # Detect vertical strokes (for cursive connections)
            vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)

            # Combine horizontal and vertical features
            combined = cv2.addWeighted(horizontal, 0.7, vertical, 0.3, 0)

            # Find contours for handwriting regions
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            min_region_area = 500  # Minimum area for handwriting region

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_region_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0

                # Handwriting regions tend to be wider than tall
                if aspect_ratio > 0.5 and w > 50 and h > 20:
                    # Add padding around the region
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)

                    # Extract region image
                    region_image = image[y:y+h, x:x+w].copy()

                    # Score region based on characteristics
                    score = self._score_handwriting_region(region_image)

                    if score > 0.3:  # Minimum score threshold
                        regions.append({
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'image': region_image,
                            'area': area,
                            'score': score,
                            'aspect_ratio': aspect_ratio
                        })

            # Sort regions by score (highest first)
            regions.sort(key=lambda x: x['score'], reverse=True)

            # Remove overlapping regions
            regions = self._remove_overlapping_regions(regions)

            logger.info(f"Detected {len(regions)} potential handwriting regions")
            return regions

        except Exception as e:
            logger.error(f"Handwriting region detection failed: {e}")
            return []

    def _score_handwriting_region(self, region_image: np.ndarray) -> float:
        """Score a region based on handwriting characteristics"""
        try:
            if region_image.size == 0:
                return 0.0

            # Convert to grayscale if needed
            if len(region_image.shape) == 3:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_image.copy()

            score = 0.0

            # 1. Edge density (handwriting has more curves)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            score += min(edge_density * 2, 0.3)  # Max 0.3 for edge density

            # 2. Contour complexity (handwriting has more complex contours)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                avg_contour_complexity = np.mean([len(c) for c in contours])
                complexity_score = min(avg_contour_complexity / 100, 0.2)  # Max 0.2
                score += complexity_score

            # 3. Stroke width variation (handwriting has more variation)
            # Simple approximation using morphological operations
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(gray, kernel, iterations=1)
            dilated = cv2.dilate(gray, kernel, iterations=1)
            stroke_variation = np.std(dilated - eroded) / 255.0
            score += min(stroke_variation, 0.3)  # Max 0.3

            # 4. Text line irregularity (handwriting lines are less straight)
            # Project pixels horizontally
            horizontal_projection = np.sum(gray < 128, axis=1)
            if len(horizontal_projection) > 1:
                projection_std = np.std(horizontal_projection) / np.mean(horizontal_projection + 1)
                score += min(projection_std / 2, 0.2)  # Max 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.warning(f"Region scoring failed: {e}")
            return 0.5  # Default neutral score

    def _remove_overlapping_regions(self, regions: List[Dict[str, Any]], overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Remove overlapping handwriting regions"""
        try:
            if len(regions) <= 1:
                return regions

            filtered_regions = []
            for region in regions:
                is_overlapping = False

                for existing in filtered_regions:
                    # Calculate IoU
                    iou = self._calculate_bbox_iou(region['bbox'], existing['bbox'])
                    if iou > overlap_threshold:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    filtered_regions.append(region)

            return filtered_regions

        except Exception as e:
            logger.warning(f"Overlap removal failed: {e}")
            return regions

    def _calculate_bbox_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """Calculate IoU between two bounding boxes"""
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
            return 0.0

    def _extract_with_qwen(self, region_image: np.ndarray, bbox: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Extract handwriting using Qwen2.5-VL"""
        try:
            if not self.use_qwen:
                return None

            # Convert to PIL Image
            if len(region_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(region_image).convert('RGB')

            # Prepare the vision input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": 1280*28*28,
                        },
                        {
                            "type": "text", 
                            "text": "Please transcribe the handwritten text in this image. Return only the text content, no additional formatting or explanation."
                        }
                    ],
                }
            ]

            # Process the input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = inputs.to(self.device)

            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **self.generation_config)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]

            # Clean up the output
            cleaned_text = output_text.strip()

            # Calculate confidence based on text characteristics
            confidence = self._calculate_transcription_confidence(cleaned_text, region_image)

            return {
                'text': cleaned_text,
                'confidence': confidence,
                'method': 'qwen2vl',
                'bbox': bbox
            }

        except Exception as e:
            logger.error(f"Qwen handwriting extraction failed: {e}")
            return None

    def _extract_with_fallback(self, region_image: np.ndarray, bbox: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Fallback handwriting extraction using Tesseract"""
        try:
            if not self.use_tesseract_fallback:
                return None

            # Preprocess for handwriting OCR
            processed = self._preprocess_for_handwriting_ocr(region_image)

            # Use Tesseract with specific config for handwriting
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}"'-+= '

            text = pytesseract.image_to_string(processed, config=custom_config)
            confidence_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(conf) for conf in confidence_data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

            # Reduce confidence for fallback method
            adjusted_confidence = avg_confidence * 0.7  # Tesseract is less reliable for handwriting

            return {
                'text': text.strip(),
                'confidence': adjusted_confidence,
                'method': 'tesseract_fallback',
                'bbox': bbox
            }

        except Exception as e:
            logger.error(f"Fallback handwriting extraction failed: {e}")
            return None

    def _preprocess_for_handwriting_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for handwriting OCR"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Slight Gaussian blur to smooth out noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.warning(f"Handwriting preprocessing failed: {e}")
            return image

    def _calculate_transcription_confidence(self, text: str, image: np.ndarray) -> float:
        """Calculate confidence score for transcribed text"""
        try:
            base_confidence = 0.7  # Base confidence for Qwen2VL

            # Adjust based on text characteristics
            if not text or len(text.strip()) == 0:
                return 0.0

            # Longer text generally indicates better recognition
            length_bonus = min(len(text) / 50, 0.2)  # Max 0.2 bonus

            # Check for common OCR artifacts
            artifact_penalty = 0.0
            artifacts = ['|||', '...', '???', '###']
            for artifact in artifacts:
                if artifact in text:
                    artifact_penalty += 0.1

            # Calculate final confidence
            confidence = base_confidence + length_bonus - artifact_penalty
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'handwriting_extraction',
            'status': 'healthy',
            'device': self.device,
            'models_loaded': {
                'qwen2vl': self.use_qwen,
                'tesseract_fallback': self.use_tesseract_fallback
            },
            'model_name': self.model_name if self.use_qwen else 'fallback'
        }
