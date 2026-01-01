"""
Hybrid field extractor combining smart slicing and box detection.
Uses slicing for speed and rough positioning, box detection for precision.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

from core.smart_slicer import SmartFieldSlicer, FieldSlice
from core.box_detector import BoxDetector, FieldBox
from core.ocr_engine import OCREngine
from core.preprocessing import ImagePreprocessor
from core.text_correction import TextCorrector
from core.models import TransactionData, FieldConfidence


class HybridFieldExtractor:
    """
    Combines smart slicing and box detection for optimal extraction.

    Strategy:
    1. Use smart slicer to get approximate field regions (fast)
    2. Within each slice, detect the exact box (precise)
    3. Extract text only from the detected box (clean)
    """

    def __init__(self, ocr_engine: OCREngine):
        """
        Initialize hybrid extractor.

        Args:
            ocr_engine: Initialized OCR engine
        """
        self.ocr_engine = ocr_engine
        self.slicer = SmartFieldSlicer(ocr_engine)
        self.box_detector = BoxDetector()
        self.preprocessor = ImagePreprocessor()
        self.text_corrector = TextCorrector()

    def extract_fields(
            self,
            image: np.ndarray,
            receipt_type: str,
            debug: bool = False
    ) -> TransactionData:
        """
        Extract all fields using hybrid approach.

        Args:
            image: Receipt image
            receipt_type: Type of receipt
            debug: Whether to save debug visualizations

        Returns:
            TransactionData with extracted fields
        """
        extracted_data = {}

        # Step 1: Get field slices (fast, approximate)
        field_slices = self.slicer.create_field_slices(image, debug=False)

        if debug:
            print("\n  [Hybrid] Processing fields with box detection...")

        # Step 2: For each slice, find the box and extract
        for field_name in self.slicer.target_fields:
            if field_name not in field_slices:
                continue

            field_slice = field_slices[field_name]

            # Try hybrid extraction
            value, confidence = self._extract_field_hybrid(
                image,
                field_slice,
                field_name,
                debug
            )

            if value:
                # Apply corrections
                corrected_value = self._apply_corrections(value, field_name)

                extracted_data[field_name] = FieldConfidence(
                    value=corrected_value,
                    confidence=confidence,
                    raw_text=value
                )

                if debug:
                    print(f"    {field_name}: '{corrected_value}' (conf: {confidence:.2f})")

        return TransactionData(**extracted_data)

    def _extract_field_hybrid(
            self,
            image: np.ndarray,
            field_slice: FieldSlice,
            field_name: str,
            debug: bool
    ) -> Tuple[str, float]:
        """
        Extract field using hybrid approach:
        1. Get slice ROI
        2. Enhance for box detection
        3. Find box within slice
        4. Extract text from box

        Args:
            image: Full receipt image
            field_slice: Field slice definition
            field_name: Name of field
            debug: Debug mode

        Returns:
            Tuple of (value, confidence)
        """
        # Get slice ROI
        slice_roi = field_slice.get_roi(image)

        # Try to detect box within this slice
        box_enhanced = self.preprocessor.enhance_for_box_detection(slice_roi)
        boxes = self.box_detector.detect_field_boxes(box_enhanced, debug=False)

        # Strategy: If we found a box, use it; otherwise use the whole slice
        if boxes and len(boxes) > 0:
            # Use the largest box (most likely the field box)
            box = max(boxes, key=lambda b: b.area)

            # Extract box ROI from original slice
            box_roi = box.get_roi(slice_roi, padding=5)

            if debug and box_roi.size > 0:
                # Save debug image of the box
                from config.settings import OUTPUT_DIR
                debug_path = OUTPUT_DIR / f"debug_box_{field_name}.png"
                cv2.imwrite(str(debug_path), box_roi)
        else:
            # Fallback: use entire slice
            box_roi = slice_roi

        # OCR the refined ROI
        if box_roi.size == 0:
            return "", 0.0

        ocr_results = self.ocr_engine.read_image(box_roi, detail=1)

        if not ocr_results:
            return "", 0.0

        # Filter out labels
        filtered = self._filter_labels(ocr_results, field_name)

        if not filtered:
            filtered = ocr_results

        # Combine text
        combined = ' '.join([r.text for r in filtered])
        avg_conf = sum([r.confidence for r in filtered]) / len(filtered)

        return combined, avg_conf

    def _filter_labels(self, ocr_results: List, field_name: str) -> List:
        """Filter out Arabic field labels."""
        labels_to_remove = [
            'رقم العملية', 'التاريخ', 'الزمن', 'من حساب', 'من',
            'الى حساب', 'إلى', 'الى', 'اسم المرسل', 'رقم الموبايل',
            'التعليق', 'المبلغ', 'اسم', 'حساب'
        ]

        filtered = []
        for result in ocr_results:
            text = result.text.replace(' ', '')

            # Check if text contains any label
            is_label = False
            for label in labels_to_remove:
                label_clean = label.replace(' ', '')
                if label_clean in text:
                    is_label = True
                    break

            if not is_label:
                filtered.append(result)

        return filtered

    def _apply_corrections(self, value: str, field_name: str) -> str:
        """Apply field-specific text corrections."""
        if not value:
            return value

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


# Convenience function
def create_hybrid_extractor(ocr_engine: OCREngine) -> HybridFieldExtractor:
    """Factory function to create hybrid extractor."""
    return HybridFieldExtractor(ocr_engine)