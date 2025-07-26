# services/table_service.py - Table Extraction Service
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import time
from PIL import Image
import pandas as pd
import json

try:
    import layoutparser as lp
    from layoutparser.models import Detectron2LayoutModel
except ImportError:
    print("Warning: layoutparser not installed")

try:
    import paddleocr
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: paddleocr not installed")

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    from transformers import DetrImageProcessor, DetrForObjectDetection
except ImportError:
    print("Warning: transformers not installed")

from config import settings

logger = logging.getLogger(__name__)

class TableExtractionService:
    """Service for detecting and extracting tables from document images"""

    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD

        # Initialize models
        self._initialize_models()

        logger.info(f"TableExtractionService initialized with device: {self.device}")

    def _initialize_models(self):
        """Initialize table detection and structure recognition models"""
        try:
            # Method 1: LayoutParser with PubLayNet
            try:
                self.layout_model = Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                self.use_layoutparser = True
                logger.info("LayoutParser initialized successfully")
            except Exception as e:
                logger.warning(f"LayoutParser initialization failed: {e}")
                self.use_layoutparser = False

            # Method 2: Microsoft Table Transformer
            try:
                self.table_processor = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.table_model = TableTransformerForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.table_model.to(self.device)
                self.table_model.eval()

                # Structure recognition model
                self.structure_processor = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-structure-recognition"  
                )
                self.structure_model = TableTransformerForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-structure-recognition"
                )
                self.structure_model.to(self.device)
                self.structure_model.eval()

                self.use_table_transformer = True
                logger.info("Table Transformer initialized successfully")
            except Exception as e:
                logger.warning(f"Table Transformer initialization failed: {e}")
                self.use_table_transformer = False

            # OCR for table content
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )

        except Exception as e:
            logger.error(f"Error initializing table models: {e}")
            raise

    def extract_tables(self, image: np.ndarray, page_number: int = 1) -> Dict[str, Any]:
        """
        Extract tables from image

        Args:
            image: Input image as numpy array
            page_number: Page number for metadata

        Returns:
            Dictionary containing detected tables with structure and content
        """
        start_time = time.time()

        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = image

            # Detect tables
            table_regions = self._detect_tables(pil_image, image)

            # Extract content from each table
            tables = []
            for i, region in enumerate(table_regions):
                table_data = self._extract_table_content(region, i)
                if table_data:
                    tables.append(table_data)

            processing_time = time.time() - start_time

            result = {
                'tables': tables,
                'table_count': len(tables),
                'page_number': page_number,
                'processing_time': processing_time,
                'timestamp': time.time()
            }

            logger.info(f"Table extraction completed in {processing_time:.2f}s, found {len(tables)} tables")
            return result

        except Exception as e:
            logger.error(f"Error in table extraction: {e}")
            return {
                'tables': [],
                'table_count': 0,
                'error': str(e),
                'page_number': page_number,
                'processing_time': time.time() - start_time
            }

    def _detect_tables(self, pil_image: Image.Image, np_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect table regions in the image"""
        table_regions = []

        # Method 1: LayoutParser
        if self.use_layoutparser:
            try:
                layout = self.layout_model.detect(pil_image)
                tables = layout.filter_by(layout, filter_type='Table')

                for table in tables:
                    bbox = table.block
                    region = {
                        'bbox': {
                            'x': int(bbox.x_1),
                            'y': int(bbox.y_1),
                            'width': int(bbox.x_2 - bbox.x_1),
                            'height': int(bbox.y_2 - bbox.y_1)
                        },
                        'confidence': table.score,
                        'method': 'layoutparser',
                        'image_crop': np_image[int(bbox.y_1):int(bbox.y_2), int(bbox.x_1):int(bbox.x_2)]
                    }
                    table_regions.append(region)

            except Exception as e:
                logger.warning(f"LayoutParser detection failed: {e}")

        # Method 2: Table Transformer
        if self.use_table_transformer and len(table_regions) == 0:
            try:
                inputs = self.table_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.table_model(**inputs)

                # Process outputs
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                results = self.table_processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.7
                )[0]

                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score > 0.7:  # Confidence threshold
                        box = [round(i, 2) for i in box.tolist()]
                        x1, y1, x2, y2 = box

                        region = {
                            'bbox': {
                                'x': int(x1),
                                'y': int(y1),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': float(score),
                            'method': 'table_transformer',
                            'image_crop': np_image[int(y1):int(y2), int(x1):int(x2)]
                        }
                        table_regions.append(region)

            except Exception as e:
                logger.warning(f"Table Transformer detection failed: {e}")

        # Fallback: Simple contour detection
        if len(table_regions) == 0:
            contour_regions = self._detect_tables_contours(np_image)
            table_regions.extend(contour_regions)

        return table_regions

    def _detect_tables_contours(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback table detection using OpenCV contours"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply threshold
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            table_regions = []
            for contour in contours:
                # Filter by area and aspect ratio
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Tables typically have certain aspect ratios
                    if 0.3 < aspect_ratio < 10 and w > 100 and h > 50:
                        region = {
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'confidence': 0.6,  # Lower confidence for contour detection
                            'method': 'contours',
                            'image_crop': image[y:y+h, x:x+w]
                        }
                        table_regions.append(region)

            return table_regions

        except Exception as e:
            logger.error(f"Contour table detection failed: {e}")
            return []

    def _extract_table_content(self, region: Dict[str, Any], table_index: int) -> Optional[Dict[str, Any]]:
        """Extract content from a detected table region"""
        try:
            image_crop = region['image_crop']
            if image_crop is None or image_crop.size == 0:
                return None

            # Enhance image for better OCR
            enhanced_crop = self._enhance_table_image(image_crop)

            # Method 1: Try structure recognition first
            if self.use_table_transformer:
                structure_result = self._extract_table_structure(enhanced_crop)
                if structure_result:
                    # OCR the cells
                    content_result = self._ocr_table_cells(enhanced_crop, structure_result)
                    if content_result:
                        return {
                            'table_index': table_index,
                            'bbox': region['bbox'],
                            'confidence': region['confidence'],
                            'method': f"{region['method']}_structured",
                            'structure': structure_result,
                            'content': content_result,
                            'csv_data': self._convert_to_csv(content_result)
                        }

            # Method 2: Fallback to line detection
            fallback_result = self._extract_table_fallback(enhanced_crop)
            if fallback_result:
                return {
                    'table_index': table_index,
                    'bbox': region['bbox'],
                    'confidence': region['confidence'] * 0.8,  # Lower confidence for fallback
                    'method': f"{region['method']}_fallback",
                    'content': fallback_result,
                    'csv_data': self._convert_to_csv(fallback_result)
                }

            return None

        except Exception as e:
            logger.error(f"Table content extraction failed: {e}")
            return None

    def _enhance_table_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance table image for better OCR"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur to remove noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Remove noise with morphological operations
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _extract_table_structure(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract table structure using Table Transformer"""
        try:
            if not self.use_table_transformer:
                return None

            # Convert to PIL
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            inputs = self.structure_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.structure_model(**inputs)

            # Process structure outputs
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.structure_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.6
            )[0]

            # Extract cells and headers
            cells = []
            headers = []

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.6:
                    box = [round(i, 2) for i in box.tolist()]
                    x1, y1, x2, y2 = box

                    cell_info = {
                        'bbox': {'x': int(x1), 'y': int(y1), 'width': int(x2-x1), 'height': int(y2-y1)},
                        'confidence': float(score),
                        'label': self.structure_model.config.id2label[label.item()]
                    }

                    if 'header' in cell_info['label'].lower():
                        headers.append(cell_info)
                    else:
                        cells.append(cell_info)

            return {
                'cells': cells,
                'headers': headers,
                'total_elements': len(cells) + len(headers)
            }

        except Exception as e:
            logger.warning(f"Structure extraction failed: {e}")
            return None

    def _ocr_table_cells(self, image: np.ndarray, structure: Dict[str, Any]) -> Optional[List[List[str]]]:
        """OCR individual table cells"""
        try:
            all_elements = structure['cells'] + structure['headers']

            # Sort elements by position (top to bottom, left to right)
            all_elements.sort(key=lambda x: (x['bbox']['y'], x['bbox']['x']))

            # Group into rows based on y-coordinate
            rows = []
            current_row = []
            current_y = -1
            y_tolerance = 20  # Pixels tolerance for same row

            for element in all_elements:
                y = element['bbox']['y']

                if current_y == -1 or abs(y - current_y) <= y_tolerance:
                    current_row.append(element)
                    current_y = y
                else:
                    if current_row:
                        # Sort current row by x-coordinate
                        current_row.sort(key=lambda x: x['bbox']['x'])
                        rows.append(current_row)
                    current_row = [element]
                    current_y = y

            # Add the last row
            if current_row:
                current_row.sort(key=lambda x: x['bbox']['x'])
                rows.append(current_row)

            # OCR each cell
            table_data = []
            for row in rows:
                row_data = []
                for cell in row:
                    bbox = cell['bbox']
                    cell_image = image[
                        bbox['y']:bbox['y']+bbox['height'],
                        bbox['x']:bbox['x']+bbox['width']
                    ]

                    if cell_image.size > 0:
                        # OCR the cell
                        ocr_result = self.ocr.ocr(cell_image, cls=False)
                        cell_text = ""

                        if ocr_result and ocr_result[0]:
                            texts = [line[1][0] for line in ocr_result[0] if line[1][0].strip()]
                            cell_text = " ".join(texts)

                        row_data.append(cell_text.strip())
                    else:
                        row_data.append("")

                if any(cell.strip() for cell in row_data):  # Only add non-empty rows
                    table_data.append(row_data)

            return table_data

        except Exception as e:
            logger.error(f"Cell OCR failed: {e}")
            return None

    def _extract_table_fallback(self, image: np.ndarray) -> Optional[List[List[str]]]:
        """Fallback table extraction using line detection"""
        try:
            # Use PaddleOCR on the entire table region
            ocr_result = self.ocr.ocr(image, cls=True)

            if not ocr_result or not ocr_result[0]:
                return None

            # Extract text with positions
            text_blocks = []
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]

                    if text_info and text_info[0].strip():
                        # Get center coordinates
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]

                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)

                        text_blocks.append({
                            'text': text_info[0].strip(),
                            'x': center_x,
                            'y': center_y,
                            'confidence': text_info[1]
                        })

            if not text_blocks:
                return None

            # Group into rows and columns
            # Sort by y-coordinate first
            text_blocks.sort(key=lambda x: x['y'])

            # Group into rows
            rows = []
            current_row = []
            current_y = -1
            y_tolerance = 30

            for block in text_blocks:
                if current_y == -1 or abs(block['y'] - current_y) <= y_tolerance:
                    current_row.append(block)
                    current_y = block['y']
                else:
                    if current_row:
                        current_row.sort(key=lambda x: x['x'])  # Sort by x within row
                        rows.append([block['text'] for block in current_row])
                    current_row = [block]
                    current_y = block['y']

            # Add the last row
            if current_row:
                current_row.sort(key=lambda x: x['x'])
                rows.append([block['text'] for block in current_row])

            return rows if rows else None

        except Exception as e:
            logger.error(f"Fallback table extraction failed: {e}")
            return None

    def _convert_to_csv(self, table_data: List[List[str]]) -> str:
        """Convert table data to CSV format"""
        try:
            if not table_data:
                return ""

            # Create DataFrame
            df = pd.DataFrame(table_data)

            # Convert to CSV string
            return df.to_csv(index=False, header=False)

        except Exception as e:
            logger.warning(f"CSV conversion failed: {e}")
            return ""

    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'table_extraction',
            'status': 'healthy',
            'device': self.device,
            'models_loaded': {
                'layoutparser': self.use_layoutparser,
                'table_transformer': self.use_table_transformer,
                'ocr': hasattr(self, 'ocr')
            }
        }
