"""
OCR Engine module for text extraction.
Supports both EasyOCR and Tesseract with Arabic and English.
"""
import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import cv2

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not available. Install with: pip install pytesseract")

from config.settings import OCR_LANGUAGES, OCR_GPU


class OCRResult:
    """Container for OCR results with confidence scores."""

    def __init__(self, text: str, confidence: float, bbox: Optional[List] = None):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox  # Bounding box coordinates

    def __repr__(self):
        return f"OCRResult(text='{self.text}', confidence={self.confidence:.3f})"


class OCREngine:
    """
    OCR Engine that supports both EasyOCR and Tesseract.
    Prioritizes EasyOCR for better Arabic support.
    """

    def __init__(self, engine: str = "auto", languages: List[str] = None):
        """
        Initialize OCR engine.

        Args:
            engine: 'easyocr', 'tesseract', or 'auto' (tries EasyOCR first)
            languages: List of language codes (default: ['ar', 'en'])
        """
        self.languages = languages or OCR_LANGUAGES
        self.engine_type = engine
        self.reader = None

        # Initialize the appropriate engine
        if engine == "auto":
            if EASYOCR_AVAILABLE:
                self._init_easyocr()
            elif TESSERACT_AVAILABLE:
                self._init_tesseract()
            else:
                raise RuntimeError("No OCR engine available. Install easyocr or pytesseract.")
        elif engine == "easyocr":
            self._init_easyocr()
        elif engine == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _init_easyocr(self):
        """Initialize EasyOCR reader."""
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR is not installed")

        print(f"Initializing EasyOCR with languages: {self.languages}")
        self.reader = easyocr.Reader(
            self.languages,
            gpu=OCR_GPU,
            verbose=False
        )
        self.engine_type = "easyocr"
        print("EasyOCR initialized successfully")

    def _init_tesseract(self):
        """Initialize Tesseract."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract is not installed")

        print(f"Initializing Tesseract with languages: {self.languages}")
        self.engine_type = "tesseract"
        print("Tesseract initialized successfully")

    def read_image(
            self,
            image: np.ndarray,
            detail: int = 1
    ) -> List[OCRResult]:
        """
        Extract text from image.

        Args:
            image: Input image (numpy array)
            detail: Detail level (0=simple, 1=normal, 2=detailed)

        Returns:
            List of OCRResult objects
        """
        if self.engine_type == "easyocr":
            return self._read_easyocr(image, detail)
        else:
            return self._read_tesseract(image)

    def _read_easyocr(self, image: np.ndarray, detail: int = 1) -> List[OCRResult]:
        """
        Read text using EasyOCR.

        Args:
            image: Input image
            detail: 0 = speed, 1 = balanced, 2 = accuracy

        Returns:
            List of OCRResult objects
        """
        # EasyOCR returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
        results = self.reader.readtext(
            image,
            detail=detail,
            paragraph=False  # Keep individual text blocks separate
        )

        ocr_results = []
        for bbox, text, confidence in results:
            # Clean up the text
            text = self._clean_text(text)
            if text:  # Only include non-empty results
                ocr_results.append(OCRResult(text, confidence, bbox))

        return ocr_results

    def _read_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """
        Read text using Tesseract.

        Args:
            image: Input image

        Returns:
            List of OCRResult objects
        """
        # Configure Tesseract for Arabic and English
        lang = '+'.join(self.languages)  # 'ar+en'

        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )

        ocr_results = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            if confidence > 0:  # Filter out low confidence
                text = data['text'][i]
                text = self._clean_text(text)

                if text:
                    # Normalize confidence to 0-1
                    conf_normalized = confidence / 100.0

                    # Create bounding box
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                    ocr_results.append(OCRResult(text, conf_normalized, bbox))

        return ocr_results

    def read_roi(
            self,
            image: np.ndarray,
            x: int,
            y: int,
            width: int,
            height: int,
            padding: int = 5
    ) -> List[OCRResult]:
        """
        Read text from a specific region of interest.

        Args:
            image: Full image
            x, y: Top-left coordinates
            width, height: ROI dimensions
            padding: Extra padding around ROI

        Returns:
            List of OCRResult objects
        """
        h, w = image.shape[:2]

        # Add padding and ensure within bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + width + padding)
        y2 = min(h, y + height + padding)

        roi = image[y1:y2, x1:x2]

        return self.read_image(roi)

    def extract_field_value(
            self,
            image: np.ndarray,
            field_region: Dict,
            expected_pattern: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Extract a specific field value from a defined region.

        Args:
            image: Input image
            field_region: Dict with 'x', 'y', 'width', 'height'
            expected_pattern: Optional regex pattern for validation

        Returns:
            Tuple of (extracted_text, confidence)
        """
        # Extract ROI
        results = self.read_roi(
            image,
            field_region['x'],
            field_region['y'],
            field_region['width'],
            field_region['height']
        )

        if not results:
            return "", 0.0

        # If pattern is provided, try to find matching text
        if expected_pattern:
            pattern = re.compile(expected_pattern)
            for result in results:
                if pattern.match(result.text):
                    return result.text, result.confidence

        # Otherwise, return the result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result.text, best_result.confidence

    def get_all_text(self, image: np.ndarray) -> str:
        """
        Get all text from image as a single string.

        Args:
            image: Input image

        Returns:
            Concatenated text
        """
        results = self.read_image(image)
        return ' '.join([r.text for r in results])

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize OCR output.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common OCR artifacts
        text = text.replace('|', '')
        text = text.replace('_', '')

        # Fix common number confusions
        replacements = {
            'O': '0',  # Letter O -> Zero (in numeric contexts)
            'l': '1',  # Lowercase L -> One (in numeric contexts)
            'I': '1',  # Capital I -> One (in numeric contexts)
        }

        # Only apply number fixes if the text looks numeric
        if any(c.isdigit() for c in text):
            for old, new in replacements.items():
                text = text.replace(old, new)

        return text.strip()

    def visualize_results(
            self,
            image: np.ndarray,
            results: List[OCRResult],
            output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and text on image for debugging.

        Args:
            image: Input image
            results: List of OCRResult objects
            output_path: Optional path to save visualization

        Returns:
            Annotated image
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        for result in results:
            if result.bbox:
                # Draw bounding box
                bbox = np.array(result.bbox, dtype=np.int32)
                cv2.polylines(annotated, [bbox], True, (0, 255, 0), 2)

                # Draw text and confidence
                x, y = bbox[0]
                label = f"{result.text} ({result.confidence:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        if output_path:
            cv2.imwrite(str(output_path), annotated)

        return annotated


# Convenience function
def create_ocr_engine(engine: str = "auto") -> OCREngine:
    """
    Factory function to create OCR engine.

    Args:
        engine: 'easyocr', 'tesseract', or 'auto'

    Returns:
        Initialized OCREngine
    """
    return OCREngine(engine=engine)