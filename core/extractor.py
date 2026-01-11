"""
Smart hybrid extractor - uses pattern recognition as anchors.
"""
import re
import numpy as np
from typing import Dict, Tuple, List, Optional
import cv2

from models import TransactionData, FieldResult, ReceiptType
from ocr_engine import OCREngine, OCRText


class SmartHybridExtractor:
    """Uses pattern-based anchoring for reliable field mapping."""

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

    # Patterns for anchor detection
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
        """Extract using pattern-anchored approach."""
        ocr_results = self.ocr.read(image)

        if not ocr_results:
            return TransactionData()

        if receipt_type is None:
            receipt_type = self._detect_type(image)

        print(f"  Receipt type: {receipt_type.value}")

        # Sort by Y position
        sorted_results = sorted(ocr_results, key=lambda r: r.bbox[0][1])

        # Find field region boundaries
        field_start_y, field_end_y = self._find_field_region(sorted_results, image.shape)
        print(f"  Field region: Y {field_start_y} to {field_end_y}")

        # Filter to field region
        field_results = [
            r for r in sorted_results
            if field_start_y <= r.bbox[0][1] <= field_end_y
        ]

        print(f"  Found {len(field_results)} OCR results in field region")

        # Extract using pattern-based anchors
        extracted = self._extract_with_anchors(field_results, field_start_y, field_end_y)

        return TransactionData(**extracted)

    def _detect_type(self, image: np.ndarray) -> ReceiptType:
        """Detect receipt type."""
        if len(image.shape) == 2:
            return ReceiptType.UNKNOWN

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percent = np.count_nonzero(mask) / mask.size

        return ReceiptType.GREEN if green_percent > 0.3 else ReceiptType.WHITE

    def _find_field_region(self, sorted_results: List[OCRText], image_shape: Tuple) -> Tuple[int, int]:
        """Find field boundaries."""
        h, w = image_shape[:2]
        start_y = None
        end_y = None

        # Find start (header or first transaction_id)
        for result in sorted_results:
            text = result.text.replace(' ', '')
            if 'تحويلات' in text or 'تفاصيل' in text or 'المعاملة' in text:
                start_y = result.bbox[2][1] + 20
                break
            if not start_y and re.match(r'^\d{10,11}$', text):
                start_y = result.bbox[0][1] - 10
                break

        # Find end (footer buttons or date)
        for result in reversed(sorted_results):
            text = result.text.replace(' ', '')
            if 'موافق' in text or 'تحويل' in text or 'تحميل' in text or '2024' in text or '2025' in text:
                end_y = result.bbox[0][1] - 20
                break

        # Fallback
        if start_y is None:
            start_y = int(h * 0.15)
        if end_y is None:
            end_y = int(h * 0.75)

        return start_y, end_y

    def _extract_with_anchors(self, field_results: List[OCRText],
                             start_y: int, end_y: int) -> Dict[str, FieldResult]:
        """Extract using pattern-based anchors."""

        # Step 1: Find anchor fields by pattern
        anchors = self._find_anchor_fields(field_results)

        print(f"\n  Anchors found:")
        for field, (y_pos, value) in anchors.items():
            print(f"    {field}: Y={y_pos}, value={value[:20]}")

        # Step 2: Calculate expected Y positions for all 8 fields
        field_positions = self._calculate_field_positions(anchors, start_y, end_y)

        print(f"\n  Expected field positions:")
        for field, y_pos in zip(self.FIELD_ORDER, field_positions):
            print(f"    {field}: Y≈{y_pos}")

        # Step 3: Extract each field from its expected region
        extracted = {}

        for i, field_name in enumerate(self.FIELD_ORDER):
            if field_name not in self.TARGET_FIELDS:
                continue

            # Get region for this field (±tolerance from expected Y)
            expected_y = field_positions[i]
            tolerance = 35  # pixels

            # Get OCR results in this region
            region_results = [
                r for r in field_results
                if abs(r.bbox[0][1] - expected_y) < tolerance
            ]

            if not region_results:
                print(f"  ⚠️  No OCR results near {field_name} (Y={expected_y})")
                continue

            # Extract value
            value, confidence, raw = self._extract_from_region(
                region_results, field_name
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

        return extracted

    def _find_anchor_fields(self, field_results: List[OCRText]) -> Dict[str, Tuple[int, str]]:
        """Find anchor fields by pattern matching."""
        anchors = {}
        accounts_found = []
        datetime_candidates = []

        for result in field_results:
            text_clean = result.text.replace(' ', '').replace('-', '')
            y_pos = result.bbox[0][1]

            # Transaction ID: 10-11 digits (must be at top)
            if 'transaction_id' not in anchors:
                if re.match(r'^\d{10,11}$', text_clean):
                    anchors['transaction_id'] = (y_pos, text_clean)
                    continue

            # Datetime: Contains date pattern
            if re.search(self.PATTERNS['datetime'], result.text):
                datetime_candidates.append((y_pos, result.text))
                continue

            # Accounts: 16 digits
            if re.match(r'^\d{16}$', text_clean):
                accounts_found.append((y_pos, text_clean))
                continue

            # Amount: number with comma/decimal (must be at bottom)
            if re.search(r'\d{1,3}[,]\d{3}', result.text) or re.match(r'^\d{4,}\.?\d{0,2}$', text_clean):
                # Prefer results with commas, or larger Y position
                if 'amount' not in anchors or ',' in result.text or y_pos > anchors['amount'][0]:
                    anchors['amount'] = (y_pos, result.text)

        # Pick earliest datetime candidate
        if datetime_candidates:
            datetime_candidates.sort(key=lambda x: x[0])
            anchors['datetime'] = datetime_candidates[0]

        # Assign accounts: from_account comes before to_account
        if len(accounts_found) >= 2:
            accounts_found.sort(key=lambda x: x[0])  # Sort by Y
            anchors['from_account'] = accounts_found[0]
            anchors['to_account'] = accounts_found[1]
        elif len(accounts_found) == 1:
            anchors['from_account'] = accounts_found[0]

        return anchors

    def _calculate_field_positions(self, anchors: Dict[str, Tuple[int, str]],
                                   start_y: int, end_y: int) -> List[int]:
        """Calculate expected Y position for each of 8 fields."""
        positions = [None] * 8

        # Map anchors to their indices
        field_indices = {name: i for i, name in enumerate(self.FIELD_ORDER)}

        # Fill known positions from anchors
        for field_name, (y_pos, _) in anchors.items():
            idx = field_indices.get(field_name)
            if idx is not None:
                positions[idx] = y_pos

        # Interpolate missing positions
        known_indices = [i for i, pos in enumerate(positions) if pos is not None]

        if len(known_indices) >= 2:
            # Interpolate between pairs
            for i in range(len(known_indices) - 1):
                start_idx = known_indices[i]
                end_idx = known_indices[i + 1]
                start_pos = positions[start_idx]
                end_pos = positions[end_idx]

                gap = end_idx - start_idx
                if gap > 1:
                    step = (end_pos - start_pos) / gap
                    for j in range(start_idx + 1, end_idx):
                        positions[j] = int(start_pos + ((j - start_idx) * step))

            # Fill start if needed
            if positions[0] is None:
                first_idx = known_indices[0]
                first_pos = positions[first_idx]
                if len(known_indices) > 1:
                    avg_step = (positions[known_indices[1]] - first_pos) / (known_indices[1] - first_idx)
                else:
                    avg_step = 50
                for j in range(first_idx):
                    positions[j] = int(first_pos - ((first_idx - j) * avg_step))

            # Fill end if needed
            if positions[7] is None:
                last_idx = known_indices[-1]
                last_pos = positions[last_idx]
                if len(known_indices) > 1:
                    avg_step = (last_pos - positions[known_indices[-2]]) / (last_idx - known_indices[-2])
                else:
                    avg_step = 50
                for j in range(last_idx + 1, 8):
                    positions[j] = int(last_pos + ((j - last_idx) * avg_step))

        else:
            # Not enough anchors: divide evenly
            step = (end_y - start_y) / 8
            for i in range(8):
                if positions[i] is None:
                    positions[i] = int(start_y + (i * step))

        # Ensure all are integers
        positions = [int(p) if p is not None else int(start_y + (i * (end_y - start_y) / 8))
                    for i, p in enumerate(positions)]

        return positions

    def _extract_from_region(self, region_results: List[OCRText],
                            field_name: str) -> Tuple[str, float, str]:
        """Extract value from OCR results in a region."""
        if not region_results:
            return "", 0.0, ""

        # Sort by X position (left to right)
        region_results.sort(key=lambda r: r.bbox[0][0])

        # Filter out labels
        values = []
        for result in region_results:
            if self._is_label(result.text, field_name):
                continue
            if len(result.text.strip()) < 2:
                continue
            values.append(result)

        # Fallback if we filtered everything
        if not values:
            if field_name in ['transaction_id', 'from_account', 'to_account', 'amount']:
                for result in region_results:
                    if any(c.isdigit() for c in result.text):
                        values.append(result)
            elif field_name == 'datetime':
                for result in region_results:
                    if re.search(r'\d{1,2}[-/]', result.text):
                        values.append(result)
            else:
                for result in region_results:
                    if len(result.text.strip()) >= 3:
                        values.append(result)

        if not values:
            return "", 0.0, ""

        # Combine text
        combined_text = ' '.join([v.text for v in values])
        avg_confidence = sum([v.confidence for v in values]) / len(values)

        # Clean
        cleaned = self._correct_value(combined_text, field_name)

        return cleaned, avg_confidence, combined_text

    def _is_label(self, text: str, field_name: str) -> bool:
        """Check if text is a label."""
        text_clean = text.replace(' ', '').lower()

        # Check keywords
        keywords = self.LABEL_KEYWORDS.get(field_name, [])
        for keyword in keywords:
            if keyword.replace(' ', '').lower() in text_clean:
                return True

        # Check if mostly Arabic (except for text fields)
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        if len(text) > 0 and arabic_chars / len(text) > 0.7:
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
            return ' '.join(value.split())

    def _correct_numeric(self, value: str, field_name: str) -> str:
        """Correct numeric values."""
        replacements = {
            'O': '0', 'o': '0', 'l': '1', 'I': '1',
            'S': '5', 's': '5', 'Z': '2', 'z': '2', 'B': '8'
        }

        corrected = value
        for wrong, right in replacements.items():
            corrected = corrected.replace(wrong, right)

        if field_name == 'transaction_id':
            return re.sub(r'\D', '', corrected)

        elif field_name in ['from_account', 'to_account']:
            digits = re.sub(r'\D', '', corrected)
            if len(digits) == 16:
                return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"
            return digits

        elif field_name == 'amount':
            return re.sub(r'[^\d,.]', '', corrected)

        return corrected

    def _correct_datetime(self, value: str) -> str:
        """Correct date/time."""
        month_fixes = {
            'M3y': 'May', 'M@y': 'May', 'J@n': 'Jan',
            'Feb': 'Feb', 'M@r': 'Mar', 'Apr': 'Apr',
            'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Aug',
            'Sep': 'Sep', 'Oct': 'Oct', '0ct': 'Oct',
            'Nov': 'Nov', 'Dec': 'Dec'
        }

        corrected = value
        for wrong, right in month_fixes.items():
            corrected = corrected.replace(wrong, right)

        # Add space between date and time if missing
        corrected = re.sub(r'(\d{4})(\d{2}:\d{2}:\d{2})', r'\1 \2', corrected)

        return corrected

    def _should_review(self, field_name: str, value: str, confidence: float) -> bool:
        """Determine if field needs human review."""
        # Critical fields: high standards
        if field_name in ['transaction_id', 'amount']:
            if confidence < 0.95:
                return True
            pattern = self.PATTERNS.get(field_name)
            if pattern and not re.match(pattern, value.replace(' ', '')):
                return True

        elif field_name in ['from_account', 'to_account']:
            if confidence < 0.90:
                return True
            value_norm = value.replace(' ', '')
            if not re.match(self.PATTERNS['account'], value_norm):
                return True

        elif confidence < 0.70:
            return True

        return False