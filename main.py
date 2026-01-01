"""
Main entry point for receipt OCR system.
Use this to test the complete pipeline.
"""
import os
# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import sys

from core.processor import create_processor
from core.box_detector import create_box_detector
from core.smart_slicer import create_smart_slicer
from core.ocr_engine import create_ocr_engine
from core.preprocessing import ImagePreprocessor
from config.settings import SAMPLE_RECEIPTS_DIR, OUTPUT_DIR


def test_hybrid_extraction(image_path: str):
    """
    Test hybrid extraction (smart slicing + box detection) on a single receipt.

    Args:
        image_path: Path to receipt image
    """
    print("\n" + "=" * 60)
    print("HYBRID EXTRACTION TEST")
    print("=" * 60 + "\n")

    from pathlib import Path
    image_path = Path(image_path)

    print(f"Testing on: {image_path.name}\n")

    # Process with hybrid approach
    processor = create_processor(ocr_engine="auto")  # Uses hybrid by default
    result = processor.process_receipt(image_path, save_debug=True)

    # Display detailed results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"\nFilename: {result.filename}")
    print(f"Receipt Type: {result.receipt_type.value}")
    print(f"Overall Confidence: {result.overall_confidence:.2%}")
    print(f"Flagged: {'⚠️  Yes' if result.flagged else '✅ No'}")

    print("\nExtracted Fields:")
    print("-" * 60)
    for field_name in ['transaction_id', 'datetime', 'from_account',
                       'to_account', 'receiver_name', 'comment', 'amount']:
        value = result.data.get_field_value(field_name)
        conf = result.data.get_field_confidence(field_name)
        if value:
            status = "✅" if conf > 0.8 else "⚠️ "
            print(f"  {status} {field_name:20s}: {value:35s} ({conf:.1%})")
        else:
            print(f"  ❌ {field_name:20s}: [NOT EXTRACTED]")

    if result.validation_issues:
        print("\n⚠️  Validation Issues:")
        print("-" * 60)
        for issue in result.validation_issues:
            symbol = "❌" if issue.severity == "error" else "⚠️ "
            print(f"  {symbol} {issue.field}: {issue.message}")

    print(f"\n✓ Processing time: {result.processing_time:.2f}s")
    print(f"✓ Debug images saved to: {OUTPUT_DIR}\n")


def test_smart_slicing(image_path: str):
    """
    Test smart field slicing on a single receipt.

    Args:
        image_path: Path to receipt image
    """
    print("\n" + "=" * 60)
    print("SMART SLICING TEST")
    print("=" * 60 + "\n")

    from pathlib import Path
    image_path = Path(image_path)

    print(f"Testing smart slicing on: {image_path.name}\n")

    # Load and preprocess image
    preprocessor = ImagePreprocessor()
    original = preprocessor.load_image(image_path)
    receipt_type = preprocessor.detect_receipt_type(original)

    print(f"Receipt type: {receipt_type.value}\n")

    # Create smart slicer
    print("Initializing OCR and slicer...")
    ocr_engine = create_ocr_engine()
    slicer = create_smart_slicer(ocr_engine)

    # Extract fields
    print("\nExtracting fields...\n")
    extracted = slicer.extract_all_fields(original, debug=True)

    print("\n" + "=" * 60)
    print("EXTRACTED FIELDS")
    print("=" * 60)
    for field_name, (value, confidence) in extracted.items():
        print(f"\n{field_name}:")
        print(f"  Value: {value}")
        print(f"  Confidence: {confidence:.2%}")

    print(f"\n✓ Visualization saved to: {OUTPUT_DIR / 'debug_slices.png'}")
    print("  Check the colored boxes - each field should be in its own slice!\n")


def test_smart_slicing_batch():
    """Test smart slicing on all sample receipts."""
    print("\n" + "=" * 60)
    print("BATCH SMART SLICING TEST")
    print("=" * 60 + "\n")

    from pathlib import Path

    # Find all sample receipts
    sample_dir = SAMPLE_RECEIPTS_DIR
    sample_files = (
        list(sample_dir.glob("*.png")) +
        list(sample_dir.glob("*.jpg")) +
        list(sample_dir.glob("*.jpeg"))
    )

    if not sample_files:
        print("❌ No sample receipts found!")
        return

    print(f"Found {len(sample_files)} receipt(s)\n")

    # Initialize once
    print("Initializing OCR...")
    ocr_engine = create_ocr_engine()
    preprocessor = ImagePreprocessor()
    slicer = create_smart_slicer(ocr_engine)
    print("✓ Ready\n")

    for i, img_path in enumerate(sample_files, 1):
        print(f"[{i}/{len(sample_files)}] {img_path.name}")

        # Load image
        original = preprocessor.load_image(img_path)

        # Extract fields (with visualization)
        slices = slicer.create_field_slices(original, debug=False)

        # Visualize for each
        slicer._visualize_slices(original, slices)

        # Rename output
        import shutil
        from config.settings import OUTPUT_DIR
        shutil.move(
            str(OUTPUT_DIR / "debug_slices.png"),
            str(OUTPUT_DIR / f"slices_{img_path.stem}.png")
        )

        print(f"  ✓ Created {len(slices)} slices")
        print(f"  ✓ Saved: slices_{img_path.stem}.png\n")

    print(f"✓ All visualizations saved to: {OUTPUT_DIR}\n")


def test_box_detection(image_path: str):
    """
    Test box detection on a single receipt.
    Visualizes detected boxes for debugging.

    Args:
        image_path: Path to receipt image
    """
    print("\n" + "=" * 60)
    print("BOX DETECTION TEST")
    print("=" * 60 + "\n")

    from pathlib import Path
    image_path = Path(image_path)

    print(f"Testing box detection on: {image_path.name}\n")

    # Load and preprocess image
    preprocessor = ImagePreprocessor()
    original = preprocessor.load_image(image_path)
    receipt_type = preprocessor.detect_receipt_type(original)

    print(f"Receipt type: {receipt_type.value}")

    # Detect boxes
    box_detector = create_box_detector()
    boxes = box_detector.detect_field_boxes(original, debug=True)

    print(f"\n✓ Detected {len(boxes)} field boxes")
    print("\nBox Details:")
    print("-" * 60)
    for i, box in enumerate(boxes):
        print(f"  Box {i}: ({box.x}, {box.y}) size={box.width}x{box.height} area={box.area}")

    print(f"\n✓ Visualization saved to: {OUTPUT_DIR / 'debug_boxes.png'}")
    print("  Open this image to see the detected boxes!\n")


def test_box_detection_batch():
    """Test box detection on all sample receipts."""
    print("\n" + "=" * 60)
    print("BATCH BOX DETECTION TEST")
    print("=" * 60 + "\n")

    from pathlib import Path

    # Find all sample receipts
    sample_dir = SAMPLE_RECEIPTS_DIR
    sample_files = (
        list(sample_dir.glob("*.png")) +
        list(sample_dir.glob("*.jpg")) +
        list(sample_dir.glob("*.jpeg"))
    )

    if not sample_files:
        print("❌ No sample receipts found!")
        return

    print(f"Found {len(sample_files)} receipt(s)\n")

    preprocessor = ImagePreprocessor()
    box_detector = create_box_detector()

    for i, img_path in enumerate(sample_files, 1):
        print(f"[{i}/{len(sample_files)}] {img_path.name}")

        # Load image
        original = preprocessor.load_image(img_path)

        # Detect boxes
        boxes = box_detector.detect_field_boxes(original, debug=False)

        print(f"  ✓ Detected {len(boxes)} boxes")

        # Save visualization for each
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()

        import cv2
        for j, box in enumerate(boxes):
            cv2.rectangle(
                vis,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 255, 0),
                2
            )
            cv2.putText(
                vis,
                str(j),
                (box.x + 5, box.y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        output_path = OUTPUT_DIR / f"boxes_{img_path.stem}.png"
        cv2.imwrite(str(output_path), vis)
        print(f"  ✓ Saved: {output_path.name}\n")

    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("  Review the images to verify box detection!\n")


def test_single_receipt(image_path: str, save_debug: bool = False):
    """
    Test processing a single receipt.

    Args:
        image_path: Path to receipt image
        save_debug: Whether to save debug images
    """
    print("\n" + "=" * 60)
    print("SINGLE RECEIPT TEST")
    print("=" * 60 + "\n")

    # Create processor
    processor = create_processor(ocr_engine="auto")

    # Process the receipt
    result = processor.process_receipt(image_path, save_debug=save_debug)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFilename: {result.filename}")
    print(f"Receipt Type: {result.receipt_type.value}")
    print(f"Overall Confidence: {result.overall_confidence:.2%}")
    print(f"Flagged: {'Yes' if result.flagged else 'No'}")
    print(f"Processing Time: {result.processing_time:.2f}s")

    print("\nExtracted Data:")
    print("-" * 60)
    for field_name in ['transaction_id', 'datetime', 'from_account',
                       'to_account', 'receiver_name', 'comment', 'amount']:
        value = result.data.get_field_value(field_name)
        conf = result.data.get_field_confidence(field_name)
        if value:
            print(f"  {field_name:20s}: {value:30s} (conf: {conf:.2%})")
        else:
            print(f"  {field_name:20s}: [NOT EXTRACTED]")

    if result.validation_issues:
        print("\nValidation Issues:")
        print("-" * 60)
        for issue in result.validation_issues:
            print(f"  [{issue.severity.upper()}] {issue.field}: {issue.message}")

    print("\n")


def test_batch_processing(directory: str = None, pattern: str = "*.png", save_debug: bool = False):
    """
    Test processing multiple receipts.

    Args:
        directory: Directory containing receipts (default: data/sample_receipts)
        pattern: File pattern to match
        save_debug: Whether to save debug images
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TEST")
    print("=" * 60 + "\n")

    if directory is None:
        directory = SAMPLE_RECEIPTS_DIR

    # Create processor
    processor = create_processor(ocr_engine="auto")

    # Process directory
    batch_result = processor.process_directory(
        directory,
        pattern=pattern,
        save_debug=save_debug
    )

    # Export results
    if batch_result.total_processed > 0:
        excel_path = processor.export_to_excel(batch_result)

        if batch_result.flagged > 0:
            report_path = processor.export_flagged_report(batch_result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total receipts: {batch_result.total_processed}")
    print(f"Successful: {batch_result.successful}")
    print(f"Flagged: {batch_result.flagged}")
    print(f"Failed: {batch_result.failed}")

    if batch_result.total_processed > 0:
        print(f"Success rate: {batch_result.successful/batch_result.total_processed*100:.1f}%")
        print(f"Total time: {batch_result.processing_time:.2f}s")
        print(f"Avg time per receipt: {batch_result.processing_time/batch_result.total_processed:.2f}s")

    # Show some examples
    if batch_result.successful > 0:
        print("\n✓ Sample successful extraction:")
        successful = batch_result.get_successful_receipts()[0]
        print(f"  File: {successful.filename}")
        print(f"  Transaction ID: {successful.data.get_field_value('transaction_id')}")
        print(f"  Amount: {successful.data.get_field_value('amount')}")
        print(f"  Confidence: {successful.overall_confidence:.2%}")

    if batch_result.flagged > 0:
        print("\n⚠ Sample flagged receipt:")
        flagged = batch_result.get_flagged_receipts()[0]
        print(f"  File: {flagged.filename}")
        print(f"  Confidence: {flagged.overall_confidence:.2%}")
        print(f"  Issues: {len(flagged.validation_issues)}")
        if flagged.validation_issues:
            print(f"  First issue: {flagged.validation_issues[0].message}")

    print("\n")


def quick_test():
    """
    Quick test with sample receipts in data/sample_receipts.
    """
    print("\n" + "=" * 60)
    print("QUICK TEST - Sample Receipts")
    print("=" * 60 + "\n")

    # Check if sample receipts exist
    sample_dir = SAMPLE_RECEIPTS_DIR
    sample_files = (
        list(sample_dir.glob("*.png")) +
        list(sample_dir.glob("*.jpg")) +
        list(sample_dir.glob("*.jpeg"))
    )

    if not sample_files:
        print("❌ No sample receipts found!")
        print(f"Please add receipt images to: {sample_dir}")
        print(f"Expected file formats: .png, .jpg, .jpeg")
        return

    print(f"Found {len(sample_files)} sample receipt(s):")
    for f in sample_files:
        print(f"  - {f.name}")
    print()

    # Process them
    # Update pattern to include jpg files
    test_batch_processing(directory=sample_dir, pattern="*.*", save_debug=True)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Receipt OCR System - Extract data from mobile banking receipts"
    )

    parser.add_argument(
        'mode',
        choices=['quick', 'single', 'batch', 'test-boxes', 'test-boxes-batch', 'test-slices', 'test-slices-batch'],
        help='Processing mode'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file (for single mode) or directory (for batch mode)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='File pattern for batch mode (default: *.png)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug images'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'quick':
            quick_test()

        elif args.mode == 'test-boxes':
            if not args.input:
                print("❌ Error: --input required for test-boxes mode")
                print("Usage: python main.py test-boxes --input path/to/receipt.jpg")
                sys.exit(1)

            if not Path(args.input).exists():
                print(f"❌ Error: File not found: {args.input}")
                sys.exit(1)

            test_box_detection(args.input)

        elif args.mode == 'test-boxes-batch':
            test_box_detection_batch()

        elif args.mode == 'test-slices':
            if not args.input:
                print("❌ Error: --input required for test-slices mode")
                sys.exit(1)

            if not Path(args.input).exists():
                print(f"❌ Error: File not found: {args.input}")
                sys.exit(1)

            test_smart_slicing(args.input)

        elif args.mode == 'test-slices-batch':
            test_smart_slicing_batch()

        elif args.mode == 'single':
            if not args.input:
                print("❌ Error: --input required for single mode")
                sys.exit(1)

            if not Path(args.input).exists():
                print(f"❌ Error: File not found: {args.input}")
                sys.exit(1)

            test_single_receipt(args.input, save_debug=args.debug)

        elif args.mode == 'batch':
            if not args.input:
                print("❌ Error: --input required for batch mode")
                sys.exit(1)

            if not Path(args.input).exists():
                print(f"❌ Error: Directory not found: {args.input}")
                sys.exit(1)

            test_batch_processing(
                directory=args.input,
                pattern=args.pattern,
                save_debug=args.debug
            )

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, show menu
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("RECEIPT OCR SYSTEM - Test Menu")
        print("=" * 60)
        print("\nAvailable tests:")
        print("  1. Quick test (process sample receipts)")
        print("  2. Test hybrid extraction (RECOMMENDED - slicing + boxes)")
        print("  3. Test smart slicing only")
        print("  4. Test box detection only")
        print("  5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            quick_test()
        elif choice == "2":
            # Find first sample and test hybrid
            sample_files = (
                list(SAMPLE_RECEIPTS_DIR.glob("*.png")) +
                list(SAMPLE_RECEIPTS_DIR.glob("*.jpg")) +
                list(SAMPLE_RECEIPTS_DIR.glob("*.jpeg"))
            )
            if sample_files:
                test_hybrid_extraction(str(sample_files[0]))
            else:
                print("❌ No sample receipts found!")
        elif choice == "3":
            test_smart_slicing_batch()
        elif choice == "4":
            test_box_detection_batch()
        elif choice == "5":
            sys.exit(0)
        else:
            print("Invalid choice. Use --help for more options.")
    else:
        main()