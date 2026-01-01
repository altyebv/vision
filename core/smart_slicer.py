"""
Smart field slicer for Bankak receipts.
Uses anchor detection and fixed layout to slice fields perfectly.
Much faster and more reliable than box detection.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from core.ocr_engine import OCREngine


@dataclass
class FieldSlice:
    """Represents a horizontal slice for a field."""
    field_name: str
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    height: int
    width: int

    def get_roi(self, image: np.ndarray) -> np.ndarray:
        """Extract ROI from image."""
        return image[self.y_start:self.y_end, self.x_start:self.x_end]


class SmartFieldSlicer:
    """
    Slices receipt into fields using anchor detection and fixed layout.
    Fast, accurate, and works across all screen sizes.
    """

    def __init__(self, ocr_engine: OCREngine):
        """
        Initialize smart slicer.

        Args:
            ocr_engine: OCR engine for anchor detection
        """
        self.ocr_engine = ocr_engine

        # Field order (top to bottom)
        self.field_order = [
            'transaction_id',
            'datetime',
            'from_account',
            'to_account',
            'receiver_name',
            'mobile_number',  # "رقم الموبايل" (often N/A)
            'comment',
            'amount'
        ]

        # We only care about these fields
        self.target_fields = [
            'transaction_id',
            'datetime',
            'from_account',
            'to_account',
            'receiver_name',
            'comment',
            'amount'
        ]

    def detect_field_region(
            self,
            image: np.ndarray,
            debug: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Detect the field region boundaries by finding anchor texts.

        Args:
            image: Receipt image
            debug: Whether to print debug info

        Returns:
            Tuple of (y_start, y_end, x_start, x_end) for field region
        """
        h, w = image.shape[:2]

        # OCR the image to find anchors
        ocr_results = self.ocr_engine.read_image(image, detail=1)

        # Find "تحويلات" (top anchor)
        top_y = None
        for result in ocr_results:
            if 'تحويلات' in result.text or 'تحويلات' in result.text.replace(' ', ''):
                if result.bbox:
                    # Get bottom of this text (fields start below it)
                    top_y = int(max([p[1] for p in result.bbox])) + 20
                    if debug:
                        print(f"  Found top anchor 'تحويلات' at y={top_y}")
                    break

        # Find "موافق" button (bottom anchor)
        bottom_y = None
        for result in ocr_results:
            if 'موافق' in result.text or 'موافق' in result.text.replace(' ', ''):
                if result.bbox:
                    # Get top of this text (fields end above it)
                    bottom_y = int(min([p[1] for p in result.bbox])) - 20
                    if debug:
                        print(f"  Found bottom anchor 'موافق' at y={bottom_y}")
                    break

        # Fallback to proportions if anchors not found
        if top_y is None:
            top_y = int(h * 0.15)  # Start at 15% of height
            if debug:
                print(f"  Using fallback top_y={top_y}")

        if bottom_y is None:
            bottom_y = int(h * 0.65)  # End at 65% of height
            if debug:
                print(f"  Using fallback bottom_y={bottom_y}")

        # Horizontal bounds - fields are centered with margins
        margin_percent = 0.05  # 5% margin on each side
        x_start = int(w * margin_percent)
        x_end = int(w * (1 - margin_percent))

        if debug:
            print(f"  Field region: y=[{top_y}, {bottom_y}], x=[{x_start}, {x_end}]")
            print(f"  Total field height: {bottom_y - top_y}px")

        return top_y, bottom_y, x_start, x_end

    def create_field_slices(
            self,
            image: np.ndarray,
            debug: bool = False
    ) -> Dict[str, FieldSlice]:
        """
        Create field slices by dividing the field region.

        Args:
            image: Receipt image
            debug: Whether to save debug visualization

        Returns:
            Dict of {field_name: FieldSlice}
        """
        h, w = image.shape[:2]

        # Detect field region
        top_y, bottom_y, x_start, x_end = self.detect_field_region(image, debug=debug)

        # Calculate slice height
        total_height = bottom_y - top_y
        num_fields = len(self.field_order)
        slice_height = total_height // num_fields

        if debug:
            print(f"  Slice height: {slice_height}px per field")

        # Create slices
        slices = {}
        for i, field_name in enumerate(self.field_order):
            y_start = top_y + (i * slice_height)
            y_end = top_y + ((i + 1) * slice_height)

            # Add small overlap to avoid cutting off text
            y_start = max(0, y_start - 2)
            y_end = min(h, y_end + 2)

            slices[field_name] = FieldSlice(
                field_name=field_name,
                y_start=y_start,
                y_end=y_end,
                x_start=x_start,
                x_end=x_end,
                height=y_end - y_start,
                width=x_end - x_start
            )

            if debug:
                print(f"  {field_name}: y=[{y_start}, {y_end}]")

        # Visualize if debug
        if debug:
            self._visualize_slices(image, slices)

        return slices

    def extract_field_value(
            self,
            image: np.ndarray,
            field_slice: FieldSlice,
            field_name: str
    ) -> Tuple[str, float]:
        """
        Extract value from a field slice.

        Args:
            image: Receipt image
            field_slice: Field slice definition
            field_name: Name of field (for filtering)

        Returns:
            Tuple of (value, confidence)
        """
        # Extract ROI
        roi = field_slice.get_roi(image)

        # OCR the slice
        ocr_results = self.ocr_engine.read_image(roi, detail=1)

        if not ocr_results:
            return "", 0.0

        # Filter out field labels (Arabic text on the right)
        # We want the VALUE on the left side of the slice
        filtered = []

        for result in ocr_results:
            text = result.text

            # Skip if this looks like an Arabic label
            if self._is_arabic_label(text):
                continue

            # Skip very short text (likely noise)
            if len(text.strip()) < 2:
                continue

            filtered.append(result)

        if not filtered:
            return "", 0.0

        # Combine all non-label text (the actual value)
        # Sort left-to-right for proper order
        filtered.sort(key=lambda r: r.bbox[0][0] if r.bbox else 0)

        combined_text = ' '.join([r.text for r in filtered])
        avg_confidence = sum([r.confidence for r in filtered]) / len(filtered)

        return combined_text, avg_confidence

    def _is_arabic_label(self, text: str) -> bool:
        """Check if text is likely an Arabic field label."""
        # Common labels
        labels = [
            'رقم العملية', 'التاريخ', 'الزمن', 'من حساب', 'من',
            'الى حساب', 'إلى', 'الى', 'اسم المرسل', 'اسم',
            'رقم الموبايل', 'التعليق', 'المبلغ'
        ]

        text_clean = text.replace(' ', '')

        for label in labels:
            label_clean = label.replace(' ', '')
            if label_clean in text_clean:
                return True

        # Check if mostly Arabic characters
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        if len(text) > 0 and arabic_chars / len(text) > 0.6:
            return True

        return False

    def extract_all_fields(
            self,
            image: np.ndarray,
            debug: bool = False
    ) -> Dict[str, Tuple[str, float]]:
        """
        Extract all fields using smart slicing.

        Args:
            image: Receipt image
            debug: Whether to show debug info

        Returns:
            Dict of {field_name: (value, confidence)}
        """
        # Create slices
        slices = self.create_field_slices(image, debug=debug)

        # Extract each target field
        extracted = {}
        for field_name in self.target_fields:
            if field_name in slices:
                value, confidence = self.extract_field_value(
                    image,
                    slices[field_name],
                    field_name
                )
                extracted[field_name] = (value, confidence)

        return extracted

    def _visualize_slices(self, image: np.ndarray, slices: Dict[str, FieldSlice]):
        """Draw field slices on image for debugging."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0)
        ]

        for i, (field_name, slice_obj) in enumerate(slices.items()):
            color = colors[i % len(colors)]

            # Draw rectangle
            cv2.rectangle(
                vis,
                (slice_obj.x_start, slice_obj.y_start),
                (slice_obj.x_end, slice_obj.y_end),
                color,
                2
            )

            # Draw label
            cv2.putText(
                vis,
                field_name[:10],  # Truncate long names
                (slice_obj.x_start + 5, slice_obj.y_start + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Save visualization
        from config.settings import OUTPUT_DIR
        output_path = OUTPUT_DIR / "debug_slices.png"
        cv2.imwrite(str(output_path), vis)
        print(f"  [DEBUG] Slice visualization saved: {output_path}")


# Convenience function
def create_smart_slicer(ocr_engine: OCREngine) -> SmartFieldSlicer:
    """Factory function to create smart slicer."""
    return SmartFieldSlicer(ocr_engine)