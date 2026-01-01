"""
Image preprocessing module for receipt OCR.
Handles image enhancement and template detection.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image

from config.settings import (
    IMAGE_PREPROCESSING,
    TEMPLATE_DETECTION
)
from core.models import ReceiptType


class ImagePreprocessor:
    """Handles image preprocessing and enhancement."""

    def __init__(self):
        self.config = IMAGE_PREPROCESSING
        self.template_config = TEMPLATE_DETECTION

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from path.

        Args:
            image_path: Path to the image file

        Returns:
            Image as numpy array (BGR format)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def detect_receipt_type(self, image: np.ndarray) -> ReceiptType:
        """
        Detect receipt type based on background color.
        Green background = confirmation receipt
        White background = transaction log

        Args:
            image: Input image in BGR format

        Returns:
            ReceiptType enum
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for green color
        lower_green = np.array(self.template_config['green_hsv_lower'])
        upper_green = np.array(self.template_config['green_hsv_upper'])

        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Calculate percentage of green pixels
        green_percentage = np.count_nonzero(mask) / mask.size

        # Classify based on threshold
        if green_percentage > self.template_config['green_threshold']:
            return ReceiptType.GREEN
        else:
            return ReceiptType.WHITE

    def enhance_for_box_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Special enhancement to make field boxes stand out.
        Designed for white/light borders on colored backgrounds.

        Args:
            image: Input image

        Returns:
            Enhanced image with prominent box edges
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Enhance contrast to make boxes pop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Apply bilateral filter to smooth while preserving edges
        smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 3. Threshold to isolate light boxes on dark background
        # Invert so boxes become black on white (better for edge detection)
        _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresh)

        # 4. Morphological operations to close gaps and strengthen edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed

    def enhance_image(self, image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Apply image enhancement techniques.

        Args:
            image: Input image in BGR format
            aggressive: Whether to apply aggressive enhancement (for difficult images)

        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if not aggressive:
            # Gentle enhancement - better for clean mobile screenshots
            # Just apply slight denoising
            denoised = cv2.fastNlMeansDenoising(gray, h=5)

            # Mild contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            return enhanced
        else:
            # Aggressive enhancement - for poor quality images
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(
                gray,
                h=self.config['denoise_strength']
            )

            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Apply additional contrast and brightness adjustment
            enhanced = cv2.convertScaleAbs(
                enhanced,
                alpha=self.config['contrast_alpha'],
                beta=self.config['brightness_beta']
            )

            # Apply binary threshold
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            return binary

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew.

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return image

        # Calculate the dominant angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)

        # Get median angle
        if angles:
            median_angle = np.median(angles)

            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(median_angle) > 0.5:
                # Get image center
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)

                # Perform rotation
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated

        return image

    def extract_roi(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        padding: int = 5
    ) -> np.ndarray:
        """
        Extract Region of Interest (ROI) from image.

        Args:
            image: Input image
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: ROI width
            height: ROI height
            padding: Extra padding around ROI

        Returns:
            Cropped ROI
        """
        h, w = image.shape[:2]

        # Add padding and ensure within bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + width + padding)
        y2 = min(h, y + height + padding)

        roi = image[y1:y2, x1:x2]
        return roi

    def preprocess_for_ocr(
        self,
        image_path: Path,
        enhance: bool = True,
        deskew: bool = False,
        aggressive: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, ReceiptType]:
        """
        Complete preprocessing pipeline for OCR.

        Args:
            image_path: Path to the image
            enhance: Whether to apply enhancement
            deskew: Whether to deskew the image
            aggressive: Whether to apply aggressive enhancement

        Returns:
            Tuple of (original_image, processed_image, receipt_type)
        """
        # Load image
        original = self.load_image(image_path)

        # Detect receipt type
        receipt_type = self.detect_receipt_type(original)

        # Make a copy for processing
        processed = original.copy()

        # Apply deskewing if requested (usually not needed for mobile screenshots)
        if deskew:
            processed = self.deskew_image(processed)

        # Apply enhancement if requested
        if enhance:
            processed = self.enhance_image(processed, aggressive=aggressive)

        return original, processed, receipt_type

    def save_debug_image(self, image: np.ndarray, output_path: Path):
        """Save image for debugging purposes."""
        cv2.imwrite(str(output_path), image)


# Convenience function
def preprocess_receipt(image_path: Path) -> Tuple[np.ndarray, np.ndarray, ReceiptType]:
    """
    Convenience function to preprocess a receipt image.

    Args:
        image_path: Path to the receipt image

    Returns:
        Tuple of (original_image, processed_image, receipt_type)
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_for_ocr(image_path)