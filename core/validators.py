"""
Validation module for extracted receipt data.
Validates field formats and flags issues for human review.
"""
import re
from datetime import datetime
from typing import List, Optional, Tuple
from dateutil import parser as date_parser

from config.settings import VALIDATION
from core.models import (
    TransactionData,
    ValidationIssue,
    ReceiptResult,
    ReceiptType
)


class DataValidator:
    """Validates extracted transaction data."""

    def __init__(self):
        """Initialize validator with configuration."""
        self.config = VALIDATION
        self.confidence_threshold = self.config['confidence_threshold']

    def validate_receipt(
        self,
        data: TransactionData,
        receipt_type: ReceiptType,
        filename: str
    ) -> ReceiptResult:
        """
        Validate extracted receipt data.

        Args:
            data: Extracted transaction data
            receipt_type: Type of receipt
            filename: Original filename

        Returns:
            ReceiptResult with validation issues and flagged status
        """
        issues = []

        # Validate each field
        issues.extend(self._validate_transaction_id(data))
        issues.extend(self._validate_datetime(data))
        issues.extend(self._validate_accounts(data))
        issues.extend(self._validate_amount(data))
        issues.extend(self._validate_confidence_scores(data))

        # Determine if receipt should be flagged
        flagged = self._should_flag(issues, data)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(data)

        return ReceiptResult(
            filename=filename,
            receipt_type=receipt_type,
            data=data,
            overall_confidence=overall_confidence,
            flagged=flagged,
            validation_issues=issues
        )

    def _validate_transaction_id(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate transaction ID field."""
        issues = []

        if not data.transaction_id:
            issues.append(ValidationIssue(
                field="transaction_id",
                issue_type="missing",
                message="Transaction ID is missing",
                severity="error"
            ))
            return issues

        value = data.transaction_id.value
        pattern = self.config['transaction_id_pattern']

        if not re.match(pattern, value):
            issues.append(ValidationIssue(
                field="transaction_id",
                issue_type="invalid_format",
                message=f"Transaction ID '{value}' does not match expected format (11 digits)",
                severity="error"
            ))

        return issues

    def _validate_datetime(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate datetime field."""
        issues = []

        if not data.datetime:
            issues.append(ValidationIssue(
                field="datetime",
                issue_type="missing",
                message="Date/time is missing",
                severity="error"
            ))
            return issues

        value = data.datetime.value

        # Try to parse the date
        parsed_date = self._parse_date(value)

        if parsed_date is None:
            issues.append(ValidationIssue(
                field="datetime",
                issue_type="invalid_format",
                message=f"Could not parse date/time: '{value}'",
                severity="error"
            ))
        else:
            # Check if date is reasonable (not in future, not too old)
            now = datetime.now()
            if parsed_date > now:
                issues.append(ValidationIssue(
                    field="datetime",
                    issue_type="invalid_value",
                    message=f"Date is in the future: {value}",
                    severity="warning"
                ))

            # Check if older than 10 years
            years_old = (now - parsed_date).days / 365
            if years_old > 10:
                issues.append(ValidationIssue(
                    field="datetime",
                    issue_type="invalid_value",
                    message=f"Date is more than 10 years old: {value}",
                    severity="warning"
                ))

        return issues

    def _validate_accounts(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate account number fields."""
        issues = []
        pattern = self.config['account_number_pattern']

        # Validate from_account
        if not data.from_account:
            issues.append(ValidationIssue(
                field="from_account",
                issue_type="missing",
                message="Source account is missing",
                severity="error"
            ))
        else:
            value = data.from_account.value
            # Remove spaces for pattern matching
            value_normalized = value.replace(' ', '')
            if not re.match(pattern.replace(r'\s?', ''), value_normalized):
                issues.append(ValidationIssue(
                    field="from_account",
                    issue_type="invalid_format",
                    message=f"Source account '{value}' does not match expected format (16 digits)",
                    severity="error"
                ))

        # Validate to_account
        if not data.to_account:
            issues.append(ValidationIssue(
                field="to_account",
                issue_type="missing",
                message="Destination account is missing",
                severity="error"
            ))
        else:
            value = data.to_account.value
            # Remove spaces for pattern matching
            value_normalized = value.replace(' ', '')
            if not re.match(pattern.replace(r'\s?', ''), value_normalized):
                issues.append(ValidationIssue(
                    field="to_account",
                    issue_type="invalid_format",
                    message=f"Destination account '{value}' does not match expected format (16 digits)",
                    severity="error"
                ))

        # Check if accounts are the same (possible OCR error)
        if data.from_account and data.to_account:
            from_normalized = data.from_account.value.replace(' ', '')
            to_normalized = data.to_account.value.replace(' ', '')
            if from_normalized == to_normalized:
                issues.append(ValidationIssue(
                    field="accounts",
                    issue_type="invalid_value",
                    message="Source and destination accounts are identical",
                    severity="warning"
                ))

        return issues

    def _validate_amount(self, data: TransactionData) -> List[ValidationIssue]:
        """Validate amount field - CRITICAL FIELD."""
        issues = []

        if not data.amount:
            issues.append(ValidationIssue(
                field="amount",
                issue_type="missing",
                message="Amount is missing - CRITICAL",
                severity="error"
            ))
            return issues

        value = data.amount.value

        # Try to parse amount as number
        amount_numeric = self._parse_amount(value)

        if amount_numeric is None:
            issues.append(ValidationIssue(
                field="amount",
                issue_type="invalid_format",
                message=f"Could not parse amount: '{value}' - CRITICAL",
                severity="error"
            ))
        else:
            # Check for unreasonable amounts
            if amount_numeric <= 0:
                issues.append(ValidationIssue(
                    field="amount",
                    issue_type="invalid_value",
                    message=f"Amount is zero or negative: {value} - CRITICAL",
                    severity="error"
                ))
            elif amount_numeric > 10000000:  # More than 10 million
                issues.append(ValidationIssue(
                    field="amount",
                    issue_type="invalid_value",
                    message=f"Amount is unusually large: {value} - Please verify",
                    severity="warning"
                ))
            elif amount_numeric < 0.01:  # Less than 1 cent
                issues.append(ValidationIssue(
                    field="amount",
                    issue_type="invalid_value",
                    message=f"Amount is suspiciously small: {value}",
                    severity="warning"
                ))

        # CRITICAL: Amount must have high confidence
        if data.amount.confidence < 0.85:
            issues.append(ValidationIssue(
                field="amount",
                issue_type="low_confidence",
                message=f"Low confidence on CRITICAL field amount: {data.amount.confidence:.2f}",
                severity="error"
            ))

        return issues

    def _validate_confidence_scores(self, data: TransactionData) -> List[ValidationIssue]:
        """Check confidence scores for all fields."""
        issues = []

        # Critical fields that must have high confidence
        critical_fields = ['transaction_id', 'amount', 'from_account', 'to_account']

        for field_name in critical_fields:
            field = getattr(data, field_name, None)
            if field and field.confidence < self.confidence_threshold:
                issues.append(ValidationIssue(
                    field=field_name,
                    issue_type="low_confidence",
                    message=f"Low confidence ({field.confidence:.2f}) for {field_name}: '{field.value}'",
                    severity="warning"
                ))

        return issues

    def _should_flag(self, issues: List[ValidationIssue], data: TransactionData) -> bool:
        """
        Determine if receipt should be flagged for human review.

        Args:
            issues: List of validation issues
            data: Transaction data

        Returns:
            True if should be flagged
        """
        # Flag if any error-severity issues
        if any(issue.severity == "error" for issue in issues):
            return True

        # Flag if multiple warnings
        if len([i for i in issues if i.severity == "warning"]) >= 2:
            return True

        # Flag if overall confidence is too low
        overall_confidence = self._calculate_overall_confidence(data)
        if overall_confidence < self.confidence_threshold:
            return True

        return False

    def _calculate_overall_confidence(self, data: TransactionData) -> float:
        """
        Calculate overall confidence score.

        Args:
            data: Transaction data

        Returns:
            Average confidence of critical fields
        """
        critical_fields = ['transaction_id', 'datetime', 'from_account',
                          'to_account', 'amount']

        confidences = []
        for field_name in critical_fields:
            conf = data.get_field_confidence(field_name)
            if conf > 0:
                confidences.append(conf)

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string using multiple formats.

        Args:
            date_str: Date string

        Returns:
            Parsed datetime or None
        """
        # Try configured formats first
        for fmt in self.config['date_formats']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try fuzzy parsing as fallback
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except:
            return None

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """
        Parse amount string to float.

        Args:
            amount_str: Amount string (may contain commas)

        Returns:
            Parsed amount or None
        """
        try:
            # Remove commas and spaces
            cleaned = amount_str.replace(',', '').replace(' ', '')
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def validate_batch(
        self,
        results: List[ReceiptResult]
    ) -> Tuple[List[ReceiptResult], List[ReceiptResult]]:
        """
        Separate flagged and successful receipts.

        Args:
            results: List of receipt results

        Returns:
            Tuple of (successful_receipts, flagged_receipts)
        """
        successful = [r for r in results if not r.flagged]
        flagged = [r for r in results if r.flagged]

        return successful, flagged


# Convenience function
def create_validator() -> DataValidator:
    """
    Factory function to create a validator.

    Returns:
        DataValidator instance
    """
    return DataValidator()