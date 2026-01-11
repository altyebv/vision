"""
Configuration for receipt OCR system.
"""
import os
from pathlib import Path

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# OCR Settings
OCR_ENGINE = "auto"  # 'auto', 'easyocr', or 'tesseract'
OCR_LANGUAGES = ['ar', 'en']
OCR_GPU = False  # Set True if you have CUDA GPU

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'transaction_id': 0.95,  # Critical - must be 95%+
    'amount': 0.95,          # Critical - must be 95%+
    'from_account': 0.90,    # Important - 90%+
    'to_account': 0.90,      # Important - 90%+
    'datetime': 0.80,        # Secondary - 80%+
    'receiver_name': 0.70,   # Text field - 70% ok
    'comment': 0.70          # Text field - 70% ok
}

# Validation Patterns
PATTERNS = {
    'transaction_id': r'^\d{10,11}$',
    'amount': r'^\d{1,10}([.,]\d{2})?$',
    'account': r'^\d{4}\s?\d{4}\s?\d{4}\s?\d{4}$'
}

# Database
DB_PATH = BASE_DIR / "receipts.db"

# Export Settings
EXPORT_COLUMNS = [
    'filename', 'receipt_type', 'transaction_id', 'datetime',
    'from_account', 'to_account', 'receiver_name', 'comment',
    'amount', 'confidence', 'needs_review', 'issues'
]