"""
Field extraction module.
Uses templates to extract specific fields from receipt images.
Now with hybrid approach: label-based + template-based + text correction.
"""
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from config.settings import TEMPLATES_DIR
from core.models import (
    ReceiptType,
    TransactionData,
    FieldConfidence
)
from core.ocr_engine import OCREngine
from core.text_correction import TextCorrector
from core.label_based_extractor import LabelBasedExtractor


class FieldExtractor:
    """Extracts fields from receipts using hybrid approach."""

    def __init__(self, ocr_engine: OCREngine):
        """
        Initialize field extractor.

        Args:
            ocr_engine: Initialized OCR engine
        """
        self.ocr_engine = ocr_engine
        self.text_corrector = TextCorrector()
        self.label_extractor = LabelBasedExtractor(ocr_engine)
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """Load all template definitions from JSON files."""
        template_files = {
            ReceiptType.GREEN: TEMPLATES_DIR / "green_template.json",
            ReceiptType.WHITE: TEMPLATES_DIR / "white_template.json"
        }

        for receipt_type, template_path in template_files.items():
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.templates[receipt_type] = json.load(f)
                print(f"Loaded template: {template_path.name}")
            else:
                print(f"Warning: Template not found: {template_path}")

    def extract_fields(
        self,
        image: np.ndarray,
        receipt_type: ReceiptType,
        use_label_based: bool = True
    ) -> TransactionData:
        """
        Extract all fields from a receipt image using hybrid approach.

        Args:
            image: Preprocessed receipt image
            receipt_type: Type of receipt (GREEN or WHITE)
            use_label_based: Whether to try label-based extraction first

        Returns:
            TransactionData object with extracted fields
        """
        extracted_data = {}

        # Strategy 1: Try label-based extraction first (more flexible)
        if use_label_based:
            label_results = self.label_extractor.extract_fields_by_labels(
                image,
                receipt_type.value
            )

            # Process each field with corrections
            for field_name, (value, confidence) in label_results.items():
                corrected_value = self._apply_corrections(value, field_name)

                if corrected_value:
                    extracted_data[field_name] = FieldConfidence(
                        value=corrected_value,
                        confidence=confidence,
                        raw_text=value
                    )

        # Strategy 2: Fallback to template-based for missing fields
        if receipt_type in self.templates:
            template = self.templates[receipt_type]
            fields_config = template['fields']

            for field_name, field_config in fields_config.items():
                # Skip if already extracted with good confidence
                if field_name in extracted_data and extracted_data[field_name].confidence > 0.7:
                    continue

                # Try template-based extraction
                value, confidence = self._extract_single_field_template(
                    image,
                    field_config
                )

                if value:
                    corrected_value = self._apply_corrections(value, field_name)

                    # Use this if better than existing or if missing
                    if field_name not in extracted_data or confidence > extracted_data[field_name].confidence:
                        extracted_data[field_name] = FieldConfidence(
                            value=corrected_value,
                            confidence=confidence,
                            raw_text=value
                        )

        return TransactionData(**extracted_data)

    def _apply_corrections(self, value: str, field_name: str) -> str:
        """
        Apply field-specific corrections.

        Args:
            value: Raw extracted value
            field_name: Name of the field

        Returns:
            Corrected value
        """
        if not value:
            return value

        # Apply corrections based on field type
        if field_name == 'datetime':
            return self.text_corrector.correct_date_time(value)

        elif field_name in ['from_account', 'to_account']:
            return self.text_corrector.correct_account_number(value)

        elif field_name == 'transaction_id':
            return self.text_corrector.correct_transaction_id(value)

        elif field_name == 'amount':
            return self.text_corrector.correct_amount(value)

        elif field_name in ['receiver_name', 'comment']:
            return self.text_corrector.clean_arabic_text(value)

        return value

    def _extract_single_field_template(
        self,
        image: np.ndarray,
        field_config: Dict
    ) -> Tuple[str, float]:
        """
        Extract a single field using template-based approach.

        Args:
            image: Receipt image
            field_config: Field configuration from template

        Returns:
            Tuple of (extracted_value, confidence)
        """
        region = field_config['region']
        expected_pattern = field_config.get('expected_pattern')
        preprocessing = field_config.get('preprocessing', {})

        # Extract the region of interest
        roi = self._extract_roi(image, region)

        # Apply field-specific preprocessing only if really needed
        if preprocessing.get('enhance_contrast') and self._needs_enhancement(roi):
            roi = self._enhance_roi(roi)

        # Perform OCR on the ROI
        ocr_results = self.ocr_engine.read_image(roi)

        if not ocr_results:
            return "", 0.0

        # Filter out field labels (Arabic text) to get only values
        filtered_results = self._filter_field_labels(ocr_results, field_config)

        if not filtered_results:
            filtered_results = ocr_results  # Fallback to all results

        # Try to find text matching the expected pattern
        if expected_pattern:
            value, confidence = self._find_pattern_match(
                filtered_results,
                expected_pattern
            )
            if value:
                return value, confidence

        # Combine all text from OCR results
        combined_text = ' '.join([r.text for r in filtered_results])
        avg_confidence = sum([r.confidence for r in filtered_results]) / len(filtered_results)

        return combined_text, avg_confidence

    def _needs_enhancement(self, roi: np.ndarray) -> bool:
        """Check if ROI needs enhancement based on contrast."""
        import cv2

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Calculate contrast (standard deviation)
        contrast = gray.std()

        # If contrast is low, enhancement might help
        return contrast < 50

    def _filter_field_labels(self, ocr_results, field_config: Dict):
        """Filter out Arabic field labels to extract only values."""
        label_ar = field_config.get('label_ar', '')

        filtered = []
        for result in ocr_results:
            # Skip if text contains Arabic label
            if label_ar and label_ar in result.text:
                continue

            # Skip if text is mostly Arabic (likely a label)
            arabic_chars = sum(1 for c in result.text if '\u0600' <= c <= '\u06FF')
            if len(result.text) > 0 and arabic_chars / len(result.text) > 0.5:
                continue

            filtered.append(result)

        return filtered

    def _extract_roi(self, image: np.ndarray, region: Dict) -> np.ndarray:
        """
        Extract region of interest from image.

        Args:
            image: Full image
            region: Dict with x, y, width, height

        Returns:
            ROI as numpy array
        """
        x = region['x']
        y = region['y']
        width = region['width']
        height = region['height']

        h, w = image.shape[:2]

        # Ensure coordinates are within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        roi = image[y1:y2, x1:x2]
        return roi

    def _enhance_roi(self, roi: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement to ROI."""
        import cv2

        # If already grayscale, use it; otherwise convert
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _binarize_roi(self, roi: np.ndarray) -> np.ndarray:
        """Apply binarization to ROI."""
        import cv2

        # Ensure grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return binary

    def _find_pattern_match(
        self,
        ocr_results,
        pattern: str
    ) -> Tuple[str, float]:
        """
        Find text matching the expected pattern.

        Args:
            ocr_results: List of OCRResult objects
            pattern: Regex pattern to match

        Returns:
            Tuple of (matched_text, confidence)
        """
        compiled_pattern = re.compile(pattern)

        # First, try to find exact matches
        for result in ocr_results:
            if compiled_pattern.match(result.text):
                return result.text, result.confidence

        # If no exact match, try to extract pattern from combined text
        combined = ' '.join([r.text for r in ocr_results])
        match = compiled_pattern.search(combined)

        if match:
            matched_text = match.group(0)
            # Use average confidence of all results
            avg_conf = sum([r.confidence for r in ocr_results]) / len(ocr_results)
            return matched_text, avg_conf

        return "", 0.0

    def _clean_field_value(self, value: str, pattern: Optional[str] = None) -> str:
        """
        Clean and normalize field value.

        Args:
            value: Raw extracted value
            pattern: Expected pattern (optional)

        Returns:
            Cleaned value
        """
        if not value:
            return ""

        # Remove extra whitespace
        value = ' '.join(value.split())

        # Pattern-specific cleaning
        if pattern:
            # Account numbers - ensure spaces between groups
            if 'account' in pattern or '\\d{4}' in pattern:
                # Remove all spaces first, then add them back
                digits_only = re.sub(r'\s+', '', value)
                if len(digits_only) == 16:
                    value = ' '.join([digits_only[i:i+4] for i in range(0, 16, 4)])

            # Transaction ID - remove spaces
            if pattern == r'^\d{11}$':
                value = re.sub(r'\s+', '', value)

            # Amount - standardize format
            if 'amount' in pattern.lower() or ',\\d{3}' in pattern:
                # Remove spaces, keep commas and dots
                value = value.replace(' ', '')

        return value.strip()

    def extract_field_by_name(
        self,
        image: np.ndarray,
        receipt_type: ReceiptType,
        field_name: str
    ) -> Optional[FieldConfidence]:
        """
        Extract a specific field by name.

        Args:
            image: Receipt image
            receipt_type: Type of receipt
            field_name: Name of field to extract

        Returns:
            FieldConfidence object or None
        """
        if receipt_type not in self.templates:
            return None

        template = self.templates[receipt_type]

        if field_name not in template['fields']:
            return None

        field_config = template['fields'][field_name]
        value, confidence = self._extract_single_field(image, field_config)

        if value:
            return FieldConfidence(
                value=value,
                confidence=confidence,
                raw_text=value
            )

        return None

    def get_template_info(self, receipt_type: ReceiptType) -> Optional[Dict]:
        """
        Get template information for a receipt type.

        Args:
            receipt_type: Type of receipt

        Returns:
            Template dictionary or None
        """
        return self.templates.get(receipt_type)

    def list_available_templates(self) -> list:
        """List all available template types."""
        return list(self.templates.keys())


# Convenience function
def create_field_extractor(ocr_engine: OCREngine) -> FieldExtractor:
    """
    Factory function to create a field extractor.

    Args:
        ocr_engine: Initialized OCR engine

    Returns:
        FieldExtractor instance
    """
    return FieldExtractor(ocr_engine)