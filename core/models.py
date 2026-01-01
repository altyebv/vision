"""
Data models for the receipt OCR system.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ReceiptType(str, Enum):
    """Receipt type enumeration."""
    GREEN = "confirmation"
    WHITE = "transaction_log"
    UNKNOWN = "unknown"


class FieldConfidence(BaseModel):
    """Confidence score for an extracted field."""
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: Optional[str] = None  # Original OCR text before cleanup

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "value": "20009131470",
                "confidence": 0.95,
                "raw_text": "20009131470"
            }]
        }
    }


class TransactionData(BaseModel):
    """Extracted transaction data from receipt."""
    transaction_id: Optional[FieldConfidence] = None
    datetime: Optional[FieldConfidence] = None
    from_account: Optional[FieldConfidence] = None
    to_account: Optional[FieldConfidence] = None
    receiver_name: Optional[FieldConfidence] = None
    comment: Optional[FieldConfidence] = None
    amount: Optional[FieldConfidence] = None

    def get_field_value(self, field_name: str) -> Optional[str]:
        """Get the value of a field if it exists."""
        field = getattr(self, field_name, None)
        return field.value if field else None

    def get_field_confidence(self, field_name: str) -> float:
        """Get the confidence score of a field."""
        field = getattr(self, field_name, None)
        return field.confidence if field else 0.0


class ValidationIssue(BaseModel):
    """A validation issue found in the extracted data."""
    field: str
    issue_type: str  # 'missing', 'invalid_format', 'low_confidence'
    message: str
    severity: str = "warning"  # 'warning' or 'error'


class ReceiptResult(BaseModel):
    """Complete result of processing a receipt."""
    filename: str
    receipt_type: ReceiptType
    data: TransactionData
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    flagged: bool = False
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    processing_time: Optional[float] = None  # seconds

    def model_post_init(self, __context):
        """Calculate overall confidence after initialization."""
        if self.overall_confidence == 0.0:
            confidences = []
            for field_name in ['transaction_id', 'datetime', 'from_account',
                               'to_account', 'amount']:
                conf = self.data.get_field_confidence(field_name)
                if conf > 0:
                    confidences.append(conf)

            self.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'filename': self.filename,
            'receipt_type': self.receipt_type.value,
            'transaction_id': self.data.get_field_value('transaction_id'),
            'datetime': self.data.get_field_value('datetime'),
            'from_account': self.data.get_field_value('from_account'),
            'to_account': self.data.get_field_value('to_account'),
            'receiver_name': self.data.get_field_value('receiver_name'),
            'comment': self.data.get_field_value('comment'),
            'amount': self.data.get_field_value('amount'),
            'confidence_score': round(self.overall_confidence, 3),
            'flagged': self.flagged,
            'flag_reason': '; '.join([issue.message for issue in self.validation_issues])
        }


class BatchResult(BaseModel):
    """Result of processing a batch of receipts."""
    total_processed: int
    successful: int
    flagged: int
    failed: int
    results: List[ReceiptResult]
    processing_time: float  # total seconds

    def get_flagged_receipts(self) -> List[ReceiptResult]:
        """Get all flagged receipts."""
        return [r for r in self.results if r.flagged]

    def get_successful_receipts(self) -> List[ReceiptResult]:
        """Get all successfully processed receipts."""
        return [r for r in self.results if not r.flagged]