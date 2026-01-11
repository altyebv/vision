"""
Main processor - orchestrates the extraction pipeline.
"""
import config  # Import first to fix OpenMP
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List

from models import ExtractionResult, ReceiptType
from ocr_engine import OCREngine
from extractor import SmartHybridExtractor
from validator import Validator


class ReceiptProcessor:
    """Main receipt processor."""

    def __init__(self, ocr_engine: str = "auto"):
        """
        Initialize processor.

        Args:
            ocr_engine: 'easyocr', 'tesseract', or 'auto'
        """
        print("Initializing Receipt Processor...")

        self.ocr = OCREngine(engine=ocr_engine)
        self.extractor = SmartHybridExtractor(self.ocr)
        self.validator = Validator()

        print("✓ Processor ready\n")

    def process(self, image_path: Union[str, Path],
               save_debug: bool = False) -> ExtractionResult:
        """
        Process a single receipt.

        Args:
            image_path: Path to receipt image
            save_debug: Save debug visualization

        Returns:
            ExtractionResult
        """
        start_time = time.time()
        image_path = Path(image_path)

        print(f"Processing: {image_path.name}")

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            # Detect receipt type
            receipt_type = self._detect_type(image)
            print(f"  Type: {receipt_type.value}")

            # Extract fields
            data = self.extractor.extract(image, receipt_type)
            print(f"  Extracted fields")

            # Validate
            processing_time = time.time() - start_time
            result = self.validator.validate(
                data,
                receipt_type,
                image_path.name,
                processing_time
            )

            # Print summary
            status = "⚠️  REVIEW" if result.needs_review else "✅ OK"
            print(f"  {status} (confidence: {result.overall_confidence:.2%}, "
                  f"time: {processing_time:.2f}s)")

            if result.issues:
                print(f"  Issues: {len(result.issues)}")
                for issue in result.issues[:3]:  # Show first 3
                    print(f"    - {issue.field}: {issue.message}")

            print()

            return result

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            # Return failed result
            from models import TransactionData, ValidationIssue

            return ExtractionResult(
                filename=image_path.name,
                receipt_type=ReceiptType.UNKNOWN,
                data=TransactionData(),
                issues=[ValidationIssue(
                    field="processing",
                    severity="error",
                    message=str(e)
                )],
                overall_confidence=0.0,
                needs_review=True,
                processing_time=time.time() - start_time
            )

    def process_batch(self, image_paths: List[Union[str, Path]],
                     save_debug: bool = False) -> List[ExtractionResult]:
        """
        Process multiple receipts.

        Args:
            image_paths: List of image paths
            save_debug: Save debug visualizations

        Returns:
            List of ExtractionResult
        """
        print(f"Processing {len(image_paths)} receipts...\n")
        print("=" * 60)

        results = []
        for path in image_paths:
            result = self.process(path, save_debug)
            results.append(result)

        # Summary
        print("=" * 60)
        total = len(results)
        needs_review = sum(1 for r in results if r.needs_review)
        ok = total - needs_review

        print(f"\nBatch complete:")
        print(f"  Total: {total}")
        print(f"  OK: {ok}")
        print(f"  Needs review: {needs_review}")

        if total > 0:
            avg_conf = sum(r.overall_confidence for r in results) / total
            avg_time = sum(r.processing_time for r in results) / total
            print(f"  Avg confidence: {avg_conf:.2%}")
            print(f"  Avg time: {avg_time:.2f}s")

        print()

        return results

    def process_directory(self, directory: Union[str, Path],
                         pattern: str = "*.*") -> List[ExtractionResult]:
        """
        Process all images in a directory.

        Args:
            directory: Directory path
            pattern: File pattern (default: *.*)

        Returns:
            List of ExtractionResult
        """
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Find images
        valid_exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        all_files = sorted(directory.glob(pattern))
        image_files = [f for f in all_files if f.suffix in valid_exts]

        if not image_files:
            print(f"No images found in {directory}")
            return []

        print(f"Found {len(image_files)} images in {directory}\n")

        return self.process_batch(image_files)

    def _detect_type(self, image: np.ndarray) -> ReceiptType:
        """Detect receipt type (green or white)."""
        if len(image.shape) == 2:
            return ReceiptType.UNKNOWN

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percent = np.count_nonzero(mask) / mask.size

        if green_percent > 0.3:
            return ReceiptType.GREEN
        else:
            return ReceiptType.WHITE