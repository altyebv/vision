"""
Simple OCR engine wrapper.
Supports EasyOCR and Tesseract.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass
class OCRText:
    """OCR result."""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x,y], [x,y], [x,y], [x,y]]


class OCREngine:
    """Simple OCR engine."""

    def __init__(self, engine: str = "auto", languages: List[str] = None):
        """
        Initialize OCR.

        Args:
            engine: 'easyocr', 'tesseract', or 'auto'
            languages: Language codes (default: ['ar', 'en'])
        """
        self.languages = languages or ['ar', 'en']
        self.reader = None

        # Initialize
        if engine == "auto":
            if EASYOCR_AVAILABLE:
                self._init_easyocr()
            elif TESSERACT_AVAILABLE:
                self._init_tesseract()
            else:
                raise RuntimeError("No OCR engine available")
        elif engine == "easyocr":
            self._init_easyocr()
        elif engine == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _init_easyocr(self):
        """Initialize EasyOCR."""
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR not installed")

        print(f"Initializing EasyOCR ({self.languages})...")
        self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        self.engine_type = "easyocr"
        print("✓ EasyOCR ready")

    def _init_tesseract(self):
        """Initialize Tesseract."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not installed")

        print(f"Initializing Tesseract ({self.languages})...")
        self.engine_type = "tesseract"
        print("✓ Tesseract ready")

    def read(self, image: np.ndarray) -> List[OCRText]:
        """
        Read text from image.

        Args:
            image: Image as numpy array

        Returns:
            List of OCRText objects
        """
        if self.engine_type == "easyocr":
            return self._read_easyocr(image)
        else:
            return self._read_tesseract(image)

    def _read_easyocr(self, image: np.ndarray) -> List[OCRText]:
        """Read with EasyOCR."""
        results = self.reader.readtext(image, detail=1, paragraph=False)

        ocr_results = []
        for bbox, text, confidence in results:
            if text.strip():
                ocr_results.append(OCRText(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox
                ))

        return ocr_results

    def _read_tesseract(self, image: np.ndarray) -> List[OCRText]:
        """Read with Tesseract."""
        import cv2

        lang = '+'.join(self.languages)
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )

        ocr_results = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            if confidence > 0:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = (data['left'][i], data['top'][i],
                                  data['width'][i], data['height'][i])
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                    ocr_results.append(OCRText(
                        text=text,
                        confidence=confidence / 100.0,
                        bbox=bbox
                    ))

        return ocr_results

    def read_region(self, image: np.ndarray, x: int, y: int,
                    width: int, height: int, padding: int = 5) -> List[OCRText]:
        """
        Read text from a specific region.

        Args:
            image: Full image
            x, y: Top-left coordinates
            width, height: Region size
            padding: Extra padding

        Returns:
            List of OCRText objects
        """
        h, w = image.shape[:2]

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + width + padding)
        y2 = min(h, y + height + padding)

        roi = image[y1:y2, x1:x2]
        return self.read(roi)