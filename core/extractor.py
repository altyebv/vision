"""
Slice-based extractor - divides cropped box into 8 horizontal slices.
Simple, fast, and accurate.
"""
import re
import numpy as np
from typing import Dict, Tuple, List
import cv2

from models import TransactionData, FieldResult, ReceiptType
from ocr_engine import OCREngine, OCRText


class SliceBasedExtractor:
    """
    Simple slice-based extraction.
    Divides the cropped box into 8 equal horizontal slices.
    Each slice corresponds to one field in order.
    """

    # Field order in receipt (top to bottom)
    FIELD_ORDER = [
        'transaction_id',
        'datetime',
        'from_account',
        'to_account',
        'receiver_name',
        'mobile_number',
        'comment',
        'amount'
    ]

    # Fields we care about (excluding mobile_number)
    TARGET_FIELDS = [
        'transaction_id', 'datetime', 'from_account',
        'to_account', 'receiver_name', 'comment', 'amount'
    ]

    # Patterns for validation
    PATTERNS = {
        'transaction_id': r'^\d{10,11}$',
        'account': r'^\d{16}$',
        'amount': r'^\d{1,10}[,.]?\d{0,3}\.?\d{0,2}$',
        'datetime': r'\d{1,2}[-/](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[-/]\d{4}'
    }

    LABEL_KEYWORDS = {
        'transaction_id': ['رقم', 'العملية'],
        'datetime': ['التاريخ', 'الزمن', 'الوقت'],
        'from_account': ['من', 'حساب'],
        'to_account': ['الى', 'إلى', 'حساب'],
        'receiver_name': ['اسم', 'المرسل', 'اليه', 'إليه'],
        'comment': ['التعليق', 'تعليق'],
        'amount': ['المبلغ', 'مبلغ']
    }

    def __init__(self, ocr_engine: OCREngine):
        self.ocr = ocr_engine

    def extract(self, image: np.ndarray, receipt_type: ReceiptType = None) -> TransactionData:
        """
        Extract using slice-based approach.

        Args:
            image: Preprocessed (cropped) image
            receipt_type: Receipt type

        Returns:
            TransactionData with extracted fields
        """
        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h}")

        # Divide into 8 equal slices
        slice_height = h / 8
        print(f"  Slice height: {slice_height:.1f}px")

        extracted = {}

        # Process each field
        for i, field_name in enumerate(self.FIELD_ORDER):
            # Skip mobile_number
            if field_name not in self.TARGET_FIELDS:
                continue

            # Calculate slice boundaries
            y_start = int(i * slice_height)
            y_end = int((i + 1) * slice_height)

            # Add small overlap to avoid cutting text
            overlap = int(slice_height * 0.1)
            y_start = max(0, y_start - overlap)
            y_end = min(h, y_end + overlap)

            # Extract slice
            slice_img = image[y_start:y_end, :]

            # OCR the slice
            ocr_results = self.ocr.read(slice_img)

            if not ocr_results:
                print(f"  ⚠️  {field_name}: No OCR results in slice")
                continue

            # Extract value from OCR results
            value, confidence, raw = self._extract_from_slice(
                ocr_results, field_name
            )

            if value:
                needs_review = self._should_review(field_name, value, confidence)

                extracted[field_name] = FieldResult(
                    value=value,
                    confidence=confidence,
                    raw_text=raw,
                    needs_review=needs_review
                )

                status = "⚠️" if needs_review else "✓"
                print(f"  {status} {field_name}: {value[:30]} (conf: {confidence:.2%})")
            else:
                print(f"  ⚠️  {field_name}: Could not extract value")

        return TransactionData(**extracted)

    def _extract_from_slice(self, ocr_results: List[OCRText],
                           field_name: str) -> Tuple[str, float, str]:
        """
        Extract value from OCR results in a single slice.

        Args:
            ocr_results: OCR results from the slice
            field_name: Name of the field

        Returns:
            (cleaned_value, confidence, raw_text)
        """
        if not ocr_results:
            return "", 0.0, ""

        # Sort by X position (left to right) to handle RTL properly
        ocr_results.sort(key=lambda r: r.bbox[0][0])

        # Filter out labels (Arabic text on the right side)
        values = []
        for result in ocr_results:
            # Skip if it's a label
            if self._is_label(result.text, field_name):
                continue

            # Skip very short text
            if len(result.text.strip()) < 2:
                continue

            values.append(result)

        # If we filtered everything, be more lenient
        if not values:
            # For numeric fields, grab anything with digits
            if field_name in ['transaction_id', 'from_account', 'to_account', 'amount']:
                for result in ocr_results:
                    if any(c.isdigit() for c in result.text):
                        values.append(result)
            # For datetime, grab anything with date pattern
            elif field_name == 'datetime':
                for result in ocr_results:
                    if re.search(r'\d{1,2}[-/]', result.text):
                        values.append(result)
            # For text fields, grab longer text
            else:
                for result in ocr_results:
                    if len(result.text.strip()) >= 3:
                        values.append(result)

        if not values:
            return "", 0.0, ""

        # Combine text (left to right, which is correct for both English and Arabic digits)
        combined_text = ' '.join([v.text for v in values])
        avg_confidence = sum([v.confidence for v in values]) / len(values)

        # Clean the value
        cleaned = self._correct_value(combined_text, field_name)

        return cleaned, avg_confidence, combined_text

    def _is_label(self, text: str, field_name: str) -> bool:
        """Check if text is a label (Arabic field name)."""
        text_clean = text.replace(' ', '').lower()

        # Check if it contains label keywords
        keywords = self.LABEL_KEYWORDS.get(field_name, [])
        for keyword in keywords:
            if keyword.replace(' ', '').lower() in text_clean:
                return True

        # Check if mostly Arabic text (labels are in Arabic)
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars > 0 and arabic_chars / total_chars > 0.7:
            # For text fields (receiver_name, comment), Arabic is the data, not label
            if field_name not in ['receiver_name', 'comment']:
                return True

        return False

    def _correct_value(self, value: str, field_name: str) -> str:
        """Apply field-specific corrections."""
        if not value:
            return value

        if field_name in ['transaction_id', 'amount', 'from_account', 'to_account']:
            return self._correct_numeric(value, field_name)
        elif field_name == 'datetime':
            return self._correct_datetime(value)
        else:
            # For text fields, just clean up whitespace
            return ' '.join(value.split())

    def _correct_numeric(self, value: str, field_name: str) -> str:
        """Correct numeric values."""
        # Common OCR mistakes in numbers
        replacements = {
            'O': '0', 'o': '0', 'l': '1', 'I': '1',
            'S': '5', 's': '5', 'Z': '2', 'z': '2',
            'B': '8', 'b': '6'
        }

        corrected = value
        for wrong, right in replacements.items():
            corrected = corrected.replace(wrong, right)

        if field_name == 'transaction_id':
            # Extract only digits
            digits = re.sub(r'\D', '', corrected)
            return digits

        elif field_name in ['from_account', 'to_account']:
            # Extract digits and format as groups of 4
            digits = re.sub(r'\D', '', corrected)
            if len(digits) == 16:
                return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"
            return digits

        elif field_name == 'amount':
            # Keep only digits, commas, and dots
            cleaned = re.sub(r'[^\d,.]', '', corrected)
            return cleaned

        return corrected

    def _correct_datetime(self, value: str) -> str:
        """Correct date/time format."""
        # Common month OCR mistakes
        month_fixes = {
            'M3y': 'May', 'M@y': 'May', 'J@n': 'Jan',
            '0ct': 'Oct', 'Dcc': 'Dec'
        }

        corrected = value
        for wrong, right in month_fixes.items():
            corrected = corrected.replace(wrong, right)

        # Ensure space between date and time
        corrected = re.sub(r'(\d{4})(\d{2}:\d{2}:\d{2})', r'\1 \2', corrected)

        return corrected

    def _should_review(self, field_name: str, value: str, confidence: float) -> bool:
        """Determine if field needs human review."""
        # Critical fields: transaction_id and amount
        if field_name in ['transaction_id', 'amount']:
            if confidence < 0.90:
                return True

            # Validate pattern
            if field_name == 'transaction_id':
                value_clean = value.replace(' ', '')
                if not re.match(r'^\d{10,11}$', value_clean):
                    return True

            elif field_name == 'amount':
                value_clean = value.replace(' ', '')
                if not re.search(r'\d', value_clean):
                    return True

        # Account fields
        elif field_name in ['from_account', 'to_account']:
            if confidence < 0.85:
                return True

            value_clean = value.replace(' ', '')
            if not re.match(r'^\d{16}$', value_clean):
                return True

        # Other fields: lower threshold
        elif confidence < 0.70:
            return True

        return False


# Alias for backwards compatibility
SmartHybridExtractor = SliceBasedExtractor