"""
Text correction and post-processing for OCR results.
Handles common OCR mistakes and field-specific cleaning.
"""
import re
from typing import Optional, Tuple
from datetime import datetime


class TextCorrector:
    """Corrects common OCR mistakes and cleans extracted text."""

    # Common OCR character confusions
    CHAR_CORRECTIONS = {
        # Numbers
        'O': '0', 'o': '0',  # Letter O → Zero
        'l': '1', 'I': '1',  # Letter l/I → One
        'S': '5', 's': '5',  # Letter S → Five (in numeric context)
        'Z': '2',  # Letter Z → Two (in numeric context)
        'B': '8',  # Letter B → Eight (in numeric context)

        # Month name corrections (case-insensitive handled separately)
        'M3y': 'May', 'M@y': 'May', 'M4y': 'May',
        'J@n': 'Jan', 'J4n': 'Jan',
        'Feb': 'Feb', 'F3b': 'Feb',
        'M@r': 'Mar', 'M4r': 'Mar',
        'Apr': 'Apr', '@pr': 'Apr',
        'Jun': 'Jun', 'Ju1': 'Jun',
        'Jul': 'Jul', 'Ju7': 'Jul',
        'Aug': 'Aug', '@ug': 'Aug',
        'Sep': 'Sep', 'S3p': 'Sep',
        'Oct': 'Oct', '0ct': 'Oct',
        'Nov': 'Nov', 'N0v': 'Nov',
        'Dec': 'Dec', 'D3c': 'Dec',
    }

    # Valid month names
    VALID_MONTHS = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    def __init__(self):
        """Initialize text corrector."""
        pass

    def correct_date_time(self, text: str) -> str:
        """
        Correct common OCR mistakes in date/time strings.
        Expected format: DD-MMM-YYYY HH:MM:SS

        Args:
            text: Raw date/time text

        Returns:
            Corrected date/time string
        """
        if not text:
            return text

        # Pattern: DD-MMM-YYYY HH:MM:SS
        date_pattern = r'(\d{2})-([A-Za-z0-9@]{3})-(\d{4})\s*(\d{2}):(\d{2}):(\d{2})'
        match = re.search(date_pattern, text)

        if match:
            day, month, year, hour, minute, second = match.groups()

            # Correct month name
            month_corrected = self._correct_month_name(month)

            # Ensure year is 4 digits (fix cases like 202516 → 2025)
            if len(year) > 4:
                year = year[:4]

            # Reconstruct
            corrected = f"{day}-{month_corrected}-{year} {hour}:{minute}:{second}"
            return corrected

        # If pattern doesn't match, try fuzzy correction
        return self._fuzzy_date_correction(text)

    def _correct_month_name(self, month: str) -> str:
        """
        Correct month name using multiple strategies.

        Args:
            month: Raw month text (e.g., 'M3y', 'J@n')

        Returns:
            Corrected month name
        """
        # Direct lookup
        if month in self.CHAR_CORRECTIONS:
            return self.CHAR_CORRECTIONS[month]

        # Check if already valid
        if month in self.VALID_MONTHS:
            return month

        # Fuzzy match - find closest valid month
        month_lower = month.lower()
        for valid_month in self.VALID_MONTHS:
            if self._fuzzy_match(month_lower, valid_month.lower()):
                return valid_month

        # Last resort - try common substitutions
        corrected = month
        for wrong, right in self.CHAR_CORRECTIONS.items():
            corrected = corrected.replace(wrong, right)

        # Check if corrected version is valid
        if corrected in self.VALID_MONTHS:
            return corrected

        # Give up, return original
        return month

    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy string matching."""
        if len(text1) != len(text2):
            return False

        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        similarity = matches / len(text1)
        return similarity >= threshold

    def _fuzzy_date_correction(self, text: str) -> str:
        """Attempt to fix malformed date strings."""
        # Fix cases like "01-May-202518:00:25" (missing space before time)
        text = re.sub(r'(\d{4})(\d{2}:\d{2}:\d{2})', r'\1 \2', text)

        # Fix month names
        for wrong, right in self.CHAR_CORRECTIONS.items():
            text = text.replace(wrong, right)

        return text

    def correct_account_number(self, text: str) -> str:
        """
        Correct and format account number.
        Expected: 16 digits, formatted as XXXX XXXX XXXX XXXX

        Args:
            text: Raw account number text

        Returns:
            Corrected and formatted account number
        """
        if not text:
            return text

        # Extract only digits
        digits = re.sub(r'\D', '', text)

        # Apply character corrections for numbers
        corrected_digits = self._correct_numeric_characters(digits)

        # Ensure exactly 16 digits
        if len(corrected_digits) == 16:
            # Format as XXXX XXXX XXXX XXXX
            formatted = ' '.join([
                corrected_digits[0:4],
                corrected_digits[4:8],
                corrected_digits[8:12],
                corrected_digits[12:16]
            ])
            return formatted

        return corrected_digits

    def correct_transaction_id(self, text: str) -> str:
        """
        Correct transaction ID.
        Expected: 11 digits, no spaces

        Args:
            text: Raw transaction ID text

        Returns:
            Corrected transaction ID
        """
        if not text:
            return text

        # Extract only digits
        digits = re.sub(r'\D', '', text)

        # Apply character corrections
        corrected = self._correct_numeric_characters(digits)

        # Ensure exactly 11 digits
        if len(corrected) == 11:
            return corrected

        return corrected

    def correct_amount(self, text: str) -> str:
        """
        Correct and format amount.
        Expected: Numbers with optional commas and decimal point

        Args:
            text: Raw amount text

        Returns:
            Corrected amount string
        """
        if not text:
            return text

        # Remove all spaces
        text = text.replace(' ', '')

        # Extract digits, commas, and decimal points only
        cleaned = re.sub(r'[^\d,.]', '', text)

        # Apply numeric corrections
        corrected = self._correct_numeric_characters(cleaned)

        # Ensure proper format (handle multiple decimal points)
        parts = corrected.split('.')
        if len(parts) > 2:
            # Multiple decimal points - keep only first one
            corrected = parts[0] + '.' + ''.join(parts[1:])

        return corrected

    def _correct_numeric_characters(self, text: str) -> str:
        """
        Correct common OCR mistakes in numeric strings.
        Only applies to strings that should be numbers.

        Args:
            text: Text containing numbers

        Returns:
            Corrected text
        """
        corrections = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'B': '8',
        }

        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

        return text

    def clean_arabic_text(self, text: str) -> str:
        """
        Clean Arabic text (for names and comments).

        Args:
            text: Raw Arabic text

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common OCR artifacts
        text = text.replace('|', '').replace('_', '')

        return text.strip()

    def extract_digits_only(self, text: str) -> str:
        """Extract only digit characters from text."""
        return re.sub(r'\D', '', text)

    def is_numeric(self, text: str) -> bool:
        """Check if text is primarily numeric."""
        if not text:
            return False

        digits = sum(1 for c in text if c.isdigit())
        return digits / len(text) > 0.7


# Convenience function
def create_text_corrector() -> TextCorrector:
    """Factory function to create text corrector."""
    return TextCorrector()