"""
Core data models for receipt OCR system.
Simplified and focused.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ReceiptType(str, Enum):
    """Receipt type."""
    GREEN = "green"
    WHITE = "white"
    UNKNOWN = "unknown"


@dataclass
class FieldResult:
    """Result of extracting a single field."""
    value: str
    confidence: float  # 0.0 to 1.0
    raw_text: str  # Before corrections
    needs_review: bool = False  # Flag for manual review


@dataclass
class TransactionData:
    """Extracted transaction data."""
    # Critical numeric fields (must be 95%+ accurate)
    transaction_id: Optional[FieldResult] = None
    amount: Optional[FieldResult] = None
    from_account: Optional[FieldResult] = None
    to_account: Optional[FieldResult] = None

    # Secondary fields (okay to flag for review)
    datetime: Optional[FieldResult] = None
    receiver_name: Optional[FieldResult] = None
    comment: Optional[FieldResult] = None

    def get_value(self, field_name: str) -> Optional[str]:
        """Get field value."""
        field = getattr(self, field_name, None)
        return field.value if field else None

    def get_confidence(self, field_name: str) -> float:
        """Get field confidence."""
        field = getattr(self, field_name, None)
        return field.confidence if field else 0.0

    def needs_review(self) -> bool:
        """Check if any field needs review."""
        for field_name in ['transaction_id', 'amount', 'from_account',
                           'to_account', 'datetime', 'receiver_name', 'comment']:
            field = getattr(self, field_name, None)
            if field and field.needs_review:
                return True
        return False


@dataclass
class ValidationIssue:
    """A validation issue."""
    field: str
    severity: str  # 'error' or 'warning'
    message: str


@dataclass
class ExtractionResult:
    """Complete result of processing a receipt."""
    filename: str
    receipt_type: ReceiptType
    data: TransactionData
    issues: List[ValidationIssue]
    overall_confidence: float
    needs_review: bool
    processing_time: float  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for export."""
        return {
            'filename': self.filename,
            'receipt_type': self.receipt_type.value,
            'transaction_id': self.data.get_value('transaction_id'),
            'datetime': self.data.get_value('datetime'),
            'from_account': self.data.get_value('from_account'),
            'to_account': self.data.get_value('to_account'),
            'receiver_name': self.data.get_value('receiver_name'),
            'comment': self.data.get_value('comment'),
            'amount': self.data.get_value('amount'),
            'confidence': round(self.overall_confidence, 3),
            'needs_review': self.needs_review,
            'issues': '; '.join([f"{i.field}: {i.message}" for i in self.issues])
        }