"""
Validator for extracted data.
Focuses on critical numeric fields.
"""
import re
from typing import List, Tuple
from datetime import datetime
from dateutil import parser as date_parser

from models import TransactionData, ValidationIssue, ExtractionResult, ReceiptType


class Validator:
    """Validates extracted transaction data."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(self, data: TransactionData, receipt_type: ReceiptType,
                 filename: str, processing_time: float) -> ExtractionResult:
        """
        Validate extracted data.

        Args:
            data: Extracted transaction data
            receipt_type: Type of receipt
            filename: Original filename
            processing_time: Processing time in seconds

        Returns:
            ExtractionResult with validation results
        """
        issues = []

        # Validate critical numeric fields
        issues.extend(self._validate_transaction_id(data))
        issues.extend(self._validate_amount(data))
        issues.extend(self._validate_accounts(data))

        # Validate secondary fields
        issues.extend(self._validate_datetime(data))

        # Check if already flagged by extractor
        needs_review = data.needs_review()

        # Also flag if we found errors
        if any(issue.severity == 'error' for issue in issues):
            needs_review = True

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(data)

        return ExtractionResult(
            filename=filename,
            receipt_type=receipt_type,
            data=data,
            issues=issues,
            overall_confidence=overall_confidence,
            needs_review=needs_review,
            processing_time=processing_time
        )

    def _validate_transaction_id(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate transaction ID."""
        issues = []

        if not data.transaction_id:
            issues.append(ValidationIssue(
                field="transaction_id",
                severity="error",
                message="Transaction ID missing"
            ))
            return issues

        value = data.transaction_id.value

        # Must be 10-11 digits
        if not re.match(r'^\d{10,11}$', value):
            issues.append(ValidationIssue(
                field="transaction_id",
                severity="error",
                message=f"Invalid format: '{value}' (expected 10-11 digits)"
            ))

        # Check confidence
        if data.transaction_id.confidence < 0.95:
            issues.append(ValidationIssue(
                field="transaction_id",
                severity="warning",
                message=f"Low confidence: {data.transaction_id.confidence:.2f}"
            ))

        return issues

    def _validate_amount(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate amount - CRITICAL."""
        issues = []

        if not data.amount:
            issues.append(ValidationIssue(
                field="amount",
                severity="error",
                message="Amount missing - CRITICAL"
            ))
            return issues

        value = data.amount.value

        # Try to parse as number
        try:
            amount_num = float(value.replace(',', ''))

            # Check reasonable range
            if amount_num <= 0:
                issues.append(ValidationIssue(
                    field="amount",
                    severity="error",
                    message=f"Amount is zero or negative: {value}"
                ))
            elif amount_num > 10000000:
                issues.append(ValidationIssue(
                    field="amount",
                    severity="warning",
                    message=f"Unusually large amount: {value} - verify"
                ))
        except ValueError:
            issues.append(ValidationIssue(
                field="amount",
                severity="error",
                message=f"Cannot parse amount: '{value}'"
            ))

        # Check confidence
        if data.amount.confidence < 0.95:
            issues.append(ValidationIssue(
                field="amount",
                severity="error",
                message=f"Low confidence on CRITICAL field: {data.amount.confidence:.2f}"
            ))

        return issues

    def _validate_accounts(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate account numbers."""
        issues = []

        # From account
        if not data.from_account:
            issues.append(ValidationIssue(
                field="from_account",
                severity="error",
                message="Source account missing"
            ))
        else:
            value = data.from_account.value.replace(' ', '')
            if not re.match(r'^\d{16}$', value):
                issues.append(ValidationIssue(
                    field="from_account",
                    severity="error",
                    message=f"Invalid format: '{data.from_account.value}' (expected 16 digits)"
                ))

        # To account
        if not data.to_account:
            issues.append(ValidationIssue(
                field="to_account",
                severity="error",
                message="Destination account missing"
            ))
        else:
            value = data.to_account.value.replace(' ', '')
            if not re.match(r'^\d{16}$', value):
                issues.append(ValidationIssue(
                    field="to_account",
                    severity="error",
                    message=f"Invalid format: '{data.to_account.value}' (expected 16 digits)"
                ))

        # Check if same (possible error)
        if data.from_account and data.to_account:
            from_norm = data.from_account.value.replace(' ', '')
            to_norm = data.to_account.value.replace(' ', '')
            if from_norm == to_norm:
                issues.append(ValidationIssue(
                    field="accounts",
                    severity="warning",
                    message="Source and destination accounts are identical"
                ))

        return issues

    def _validate_datetime(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate date/time."""
        issues = []

        if not data.datetime:
            issues.append(ValidationIssue(
                field="datetime",
                severity="warning",
                message="Date/time missing"
            ))
            return issues

        value = data.datetime.value

        # Try to parse
        parsed = self._parse_date(value)

        if parsed is None:
            issues.append(ValidationIssue(
                field="datetime",
                severity="warning",
                message=f"Cannot parse date: '{value}'"
            ))
        else:
            # Check if reasonable
            now = datetime.now()
            if parsed > now:
                issues.append(ValidationIssue(
                    field="datetime",
                    severity="warning",
                    message=f"Date is in future: {value}"
                ))

        return issues

    def _parse_date(self, date_str: str) -> datetime:
        """Try to parse date string."""
        # Common formats
        formats = [
            "%d-%b-%Y %H:%M:%S",
            "%d-%B-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        # Fuzzy parse
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except:
            return None

    def _calculate_confidence(self, data: TransactionData) -> float:
        """Calculate overall confidence score."""
        critical_fields = ['transaction_id', 'amount', 'from_account', 'to_account']

        confidences = []
        for field_name in critical_fields:
            conf = data.get_confidence(field_name)
            if conf > 0:
                confidences.append(conf)

        return sum(confidences) / len(confidences) if confidences else 0.0