"""
Label-based field extraction.
Finds Arabic field labels first, then extracts values relative to them.
This makes extraction flexible across different screen sizes.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.ocr_engine import OCREngine, OCRResult


@dataclass
class LabelPosition:
    """Position of a detected field label."""
    label: str
    bbox: List
    confidence: float
    center_x: int
    center_y: int


class LabelBasedExtractor:
    """
    Extracts field values by first finding labels, then extracting values relative to them.
    This approach is robust to screen size variations.
    """

    def __init__(self, ocr_engine: OCREngine):
        """
        Initialize label-based extractor.

        Args:
            ocr_engine: Initialized OCR engine
        """
        self.ocr_engine = ocr_engine

        # Arabic labels to search for
        self.label_map = {
            'transaction_id': ['رقم العملية', 'رقمالعملية'],
            'datetime': ['التاريخ والوقت', 'التاريخ و الزمن', 'التاريخوالوقت'],
            'from_account': ['من حساب', 'منحساب', 'من'],
            'to_account': ['إلى حساب', 'الى حساب', 'إلىحساب', 'إلى', 'الى'],
            'receiver_name': ['اسم المرسل اليه', 'اسمالمرسلاليه', 'اسم المرسل'],
            'comment': ['التعليق'],
            'amount': ['المبلغ'],
        }

    def extract_fields_by_labels(
            self,
            image: np.ndarray,
            receipt_type: str
    ) -> Dict[str, Tuple[str, float]]:
        """
        Extract all fields by finding labels first.

        Args:
            image: Receipt image
            receipt_type: Type of receipt (for layout hints)

        Returns:
            Dict of {field_name: (value, confidence)}
        """
        # Step 1: OCR the entire image to find all text
        all_ocr_results = self.ocr_engine.read_image(image, detail=1)

        # Step 2: Find all field labels
        label_positions = self._find_labels(all_ocr_results)

        # Step 3: Extract value for each field based on its label position
        extracted_fields = {}

        for field_name, labels in self.label_map.items():
            # Find this field's label position
            label_pos = self._get_label_position(label_positions, labels)

            if label_pos:
                # Extract value based on label position and receipt type
                value, confidence = self._extract_value_near_label(
                    image,
                    label_pos,
                    field_name,
                    receipt_type,
                    all_ocr_results
                )

                if value:
                    extracted_fields[field_name] = (value, confidence)

        return extracted_fields

    def _find_labels(self, ocr_results: List[OCRResult]) -> Dict[str, LabelPosition]:
        """
        Find all field labels in OCR results.

        Args:
            ocr_results: All OCR results from image

        Returns:
            Dict of {label_text: LabelPosition}
        """
        label_positions = {}

        for result in ocr_results:
            # Check if this text matches any known label
            for field_name, label_variants in self.label_map.items():
                for label in label_variants:
                    if self._text_matches_label(result.text, label):
                        # Calculate center of bounding box
                        bbox = result.bbox
                        if bbox:
                            center_x = int(np.mean([p[0] for p in bbox]))
                            center_y = int(np.mean([p[1] for p in bbox]))

                            label_positions[field_name] = LabelPosition(
                                label=label,
                                bbox=bbox,
                                confidence=result.confidence,
                                center_x=center_x,
                                center_y=center_y
                            )
                        break

        return label_positions

    def _text_matches_label(self, text: str, label: str) -> bool:
        """Check if OCR text matches a label (fuzzy match)."""
        # Remove spaces for comparison
        text_clean = text.replace(' ', '')
        label_clean = label.replace(' ', '')

        # Exact match
        if label_clean in text_clean or text_clean in label_clean:
            return True

        # Fuzzy match - at least 80% characters match
        if len(label_clean) > 3:
            matches = sum(1 for a, b in zip(text_clean, label_clean) if a == b)
            if matches / len(label_clean) > 0.8:
                return True

        return False

    def _get_label_position(
            self,
            label_positions: Dict[str, LabelPosition],
            label_variants: List[str]
    ) -> Optional[LabelPosition]:
        """Get label position for a field."""
        for field_name, pos in label_positions.items():
            if pos.label in label_variants:
                return pos
        return None

    def _extract_value_near_label(
            self,
            image: np.ndarray,
            label_pos: LabelPosition,
            field_name: str,
            receipt_type: str,
            all_results: List[OCRResult]
    ) -> Tuple[str, float]:
        """
        Extract field value based on label position.

        Args:
            image: Receipt image
            label_pos: Position of the field label
            field_name: Name of the field
            receipt_type: Receipt type (affects layout)
            all_results: All OCR results

        Returns:
            Tuple of (value, confidence)
        """
        h, w = image.shape[:2]

        # Determine value position based on receipt type
        # Green receipts: values on right side
        # White receipts: values on left side
        is_green = 'green' in receipt_type.lower() or 'confirmation' in receipt_type.lower()

        # Define search region based on label position
        if is_green:
            # Value is to the RIGHT of label
            search_x_min = 0
            search_x_max = label_pos.center_x - 50  # Stop before label
        else:
            # Value is to the LEFT of label
            search_x_min = label_pos.center_x + 50  # Start after label
            search_x_max = w

        # Vertical search region (same row as label, with some tolerance)
        search_y_min = max(0, label_pos.center_y - 30)
        search_y_max = min(h, label_pos.center_y + 30)

        # For multi-line fields (comments, names), expand vertically
        if field_name in ['comment', 'receiver_name']:
            search_y_max = min(h, label_pos.center_y + 80)

        # Find OCR results in the search region
        candidates = []
        for result in all_results:
            if not result.bbox:
                continue

            # Check if result is in search region
            bbox = result.bbox
            result_center_x = int(np.mean([p[0] for p in bbox]))
            result_center_y = int(np.mean([p[1] for p in bbox]))

            if (search_x_min <= result_center_x <= search_x_max and
                    search_y_min <= result_center_y <= search_y_max):

                # Skip if this is the label itself
                if not self._text_matches_label(result.text, label_pos.label):
                    candidates.append(result)

        if not candidates:
            return "", 0.0

        # Combine all candidate texts
        # Sort by vertical position (top to bottom)
        candidates.sort(key=lambda r: np.mean([p[1] for p in r.bbox]))

        combined_text = ' '.join([r.text for r in candidates])
        avg_confidence = sum([r.confidence for r in candidates]) / len(candidates)

        return combined_text, avg_confidence

    def extract_single_field(
            self,
            image: np.ndarray,
            field_name: str,
            receipt_type: str
    ) -> Tuple[str, float]:
        """
        Extract a single field by finding its label.

        Args:
            image: Receipt image
            field_name: Field to extract
            receipt_type: Receipt type

        Returns:
            Tuple of (value, confidence)
        """
        all_fields = self.extract_fields_by_labels(image, receipt_type)
        return all_fields.get(field_name, ("", 0.0))


# Convenience function
def create_label_based_extractor(ocr_engine: OCREngine) -> LabelBasedExtractor:
    """Factory function to create label-based extractor."""
    return LabelBasedExtractor(ocr_engine)