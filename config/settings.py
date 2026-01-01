"""
Configuration settings for the receipt OCR system.
"""
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_RECEIPTS_DIR = DATA_DIR / "sample_receipts"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

# OCR Settings
OCR_LANGUAGES = ['ar', 'en']  # Arabic and English
OCR_GPU = False  # Set to True if you have CUDA-capable GPU

# Image preprocessing settings
IMAGE_PREPROCESSING = {
    'target_dpi': 300,
    'denoise_strength': 10,
    'contrast_alpha': 1.5,  # Contrast control (1.0-3.0)
    'brightness_beta': 0,   # Brightness control (0-100)
}

# Template detection thresholds
TEMPLATE_DETECTION = {
    'green_hsv_lower': (35, 100, 100),   # HSV lower bound for green
    'green_hsv_upper': (85, 255, 255),   # HSV upper bound for green
    'green_threshold': 0.15,  # Minimum 15% green pixels to classify as green receipt
}

# Validation settings
VALIDATION = {
    'confidence_threshold': 0.80,  # 80% confidence minimum
    'transaction_id_pattern': r'^\d{11}$',
    'account_number_pattern': r'^\d{4}\s?\d{4}\s?\d{4}\s?\d{4}$',
    'amount_pattern': r'^\d{1,3}(,\d{3})*(\.\d{2})?$',
    'date_formats': [
        '%d-%b-%Y %H:%M:%S',  # 12-May-2025 10:42:11
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
    ]
}

# Field names mapping (Arabic to English)
FIELD_MAPPING = {
    'رقم العملية': 'transaction_id',
    'التاريخ والوقت': 'datetime',
    'التاريخ و الزمن': 'datetime',
    'من حساب': 'from_account',
    'من': 'from_account',
    'الى حساب': 'to_account',
    'إلى': 'to_account',
    'اسم المرسل اليه': 'receiver_name',
    'التعليق': 'comment',
    'المبلغ': 'amount',
}

# Receipt types
RECEIPT_TYPES = {
    'GREEN': 'confirmation',
    'WHITE': 'transaction_log',
}

# Export settings
EXPORT_SETTINGS = {
    'excel_columns': [
        'transaction_id',
        'datetime',
        'from_account',
        'to_account',
        'receiver_name',
        'comment',
        'amount',
        'confidence_score',
        'flagged',
        'flag_reason',
    ],
    'date_format': '%Y-%m-%d %H:%M:%S',
}