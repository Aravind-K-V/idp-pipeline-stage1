# services/text_service.py - Text Extraction Service
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import time
from PIL import Image
import json

try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Warning: transformers not installed")

try:
    import paddleocr
except ImportError:
    print("Warning: paddleocr not installed")

try:
    import easyocr
except ImportError:
    print("Warning: easyocr not installed")

try:
    import pytesseract
except ImportError:
    print("Warning: pytesseract not installed")

from config import settings

logger = logging.getLogger(__name__)

class TextExtractionService:
    """Service for extracting text from document images using multiple OCR engines"""

    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.ocr_engine = settings.OCR_ENGINE
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD

        # Initialize models
        self._initialize_models()

        logger.info(f"TextExtractionService initialized with device: {self.device}")

    def _initialize_models(self):
        """Initialize OCR models based on configuration"""
        try:
            # Initialize Donut for form understanding
            if hasattr(settings, 'DONUT_MODEL'):
                logger.info(f"Loading Donut model: {settings.DONUT_MODEL}")
                self.donut_processor = DonutProcessor.from_pretrained(settings.DONUT_MODEL)
                self.donut_model = VisionEncoderDecoderModel.from_pretrained(settings.DONUT_MODEL)
                self.donut_model.to(self.device)
                self.donut_model.eval()
            else:
                self.donut_processor = None
                self.donut_model = None

            # Initialize PaddleOCR
            if self.ocr_engine == "paddleocr":
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available(),
                    show_log=False
                )

            # Initialize EasyOCR
            elif self.ocr_engine == "easyocr":
                self.easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

            # Tesseract is initialized per call

        except Exception as e:
            logger.error(f"Error initializing OCR models: {e}")
            raise

    def extract_text(self, image: np.ndarray, page_number: int = 1) -> Dict[str, Any]:
        """
        Extract text from image using the configured OCR engine

        Args:
            image: Input image as numpy array
            page_number: Page number for metadata

        Returns:
            Dictionary containing extracted text with bounding boxes and confidence scores
        """
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Extract text based on configured engine
            if self.ocr_engine == "paddleocr":
                result = self._extract_with_paddle(processed_image)
            elif self.ocr_engine == "easyocr":
                result = self._extract_with_easyocr(processed_image)
            elif self.ocr_engine == "tesseract":
                result = self._extract_with_tesseract(processed_image)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")

            # If Donut is available, also extract form structure
            if self.donut_model is not None:
                form_data = self._extract_form_structure(processed_image)
                result['form_structure'] = form_data

            processing_time = time.time() - start_time

            # Add metadata
            result.update({
                'page_number': page_number,
                'processing_time': processing_time,
                'ocr_engine': self.ocr_engine,
                'timestamp': time.time()
            })

            logger.info(f"Text extraction completed in {processing_time:.2f}s for page {page_number}")
            return result

        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return {
                'text_blocks': [],
                'confidence': 0.0,
                'error': str(e),
                'page_number': page_number,
                'processing_time': time.time() - start_time
            }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # Threshold for better text recognition
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return thresh

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    def _extract_with_paddle(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using PaddleOCR"""
        try:
            result = self.paddle_ocr.ocr(image, cls=True)

            text_blocks = []
            overall_confidence = 0.0
            confidence_count = 0

            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]  # Bounding box coordinates
                        text_info = line[1]  # (text, confidence)

                        if text_info and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = float(text_info[1])

                            # Convert bbox to standard format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]

                            text_block = {
                                'text': text,
                                'confidence': confidence,
                                'bbox': {
                                    'x': min(x_coords),
                                    'y': min(y_coords),
                                    'width': max(x_coords) - min(x_coords),
                                    'height': max(y_coords) - min(y_coords)
                                },
                                'polygon': bbox
                            }

                            text_blocks.append(text_block)
                            overall_confidence += confidence
                            confidence_count += 1

            if confidence_count > 0:
                overall_confidence /= confidence_count

            return {
                'text_blocks': text_blocks,
                'confidence': overall_confidence,
                'total_blocks': len(text_blocks)
            }

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {'text_blocks': [], 'confidence': 0.0, 'error': str(e)}

    def _extract_with_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        try:
            result = self.easy_reader.readtext(image)

            text_blocks = []
            overall_confidence = 0.0

            for detection in result:
                bbox = detection[0]  # Bounding box coordinates
                text = detection[1]  # Text
                confidence = float(detection[2])  # Confidence

                # Convert bbox to standard format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                text_block = {
                    'text': text,
                    'confidence': confidence,
                    'bbox': {
                        'x': min(x_coords),
                        'y': min(y_coords),
                        'width': max(x_coords) - min(x_coords),
                        'height': max(y_coords) - min(y_coords)
                    },
                    'polygon': bbox
                }

                text_blocks.append(text_block)
                overall_confidence += confidence

            if len(text_blocks) > 0:
                overall_confidence /= len(text_blocks)

            return {
                'text_blocks': text_blocks,
                'confidence': overall_confidence,
                'total_blocks': len(text_blocks)
            }

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text_blocks': [], 'confidence': 0.0, 'error': str(e)}

    def _extract_with_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using Tesseract OCR"""
        try:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6'
            )

            text_blocks = []
            overall_confidence = 0.0
            confidence_count = 0

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 range

                    text_block = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    }

                    text_blocks.append(text_block)
                    overall_confidence += confidence
                    confidence_count += 1

            if confidence_count > 0:
                overall_confidence /= confidence_count

            # Also get plain text
            plain_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')

            return {
                'text_blocks': text_blocks,
                'confidence': overall_confidence,
                'total_blocks': len(text_blocks),
                'full_text': plain_text.strip()
            }

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text_blocks': [], 'confidence': 0.0, 'error': str(e)}

    def _extract_form_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract form structure using Donut model"""
        try:
            if self.donut_model is None or self.donut_processor is None:
                return {}

            # Convert numpy array to PIL Image
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(image)

            # Process with Donut
            pixel_values = self.donut_processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate form structure
            with torch.no_grad():
                decoder_input_ids = torch.tensor([[self.donut_model.config.decoder_start_token_id]], device=self.device)
                outputs = self.donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.donut_model.decoder.config.max_position_embeddings,
                    pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                    eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            # Decode the output
            sequence = self.donut_processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.donut_processor.tokenizer.eos_token, "").replace(self.donut_processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

            # Try to parse as JSON
            try:
                form_data = json.loads(sequence)
                return form_data
            except json.JSONDecodeError:
                return {'raw_output': sequence}

        except Exception as e:
            logger.error(f"Donut form extraction failed: {e}")
            return {'error': str(e)}

    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'text_extraction',
            'status': 'healthy',
            'ocr_engine': self.ocr_engine,
            'device': self.device,
            'models_loaded': {
                'donut': self.donut_model is not None,
                'paddle_ocr': hasattr(self, 'paddle_ocr'),
                'easy_ocr': hasattr(self, 'easy_reader')
            }
        }
