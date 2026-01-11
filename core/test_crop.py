"""
Quick test script to verify cropping works.
Run this BEFORE full extraction to debug preprocessing.
"""
import cv2
import sys
from pathlib import Path
from preprocessor import preprocess_receipt


def test_crop(image_path: str):
    """Test cropping on a single image."""
    print(f"\nTesting crop on: {image_path}")
    print("=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Original size: {w}x{h}")

    # Preprocess with debug mode
    print("\nPreprocessing...")
    processed = preprocess_receipt(image, target_height=800, debug=True)

    ph, pw = processed.shape[:2]
    print(f"\nProcessed size: {pw}x{ph}")

    # Save comparison
    output_path = f"test_crop_result_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, processed)
    print(f"\n✅ Saved result to: {output_path}")
    print(f"✅ Saved debug images: debug_box_detection.jpg, debug_cropped.jpg")

    print("\n" + "=" * 60)
    print("Check the output images:")
    print("  - debug_box_detection.jpg: Shows detected box (green rectangle)")
    print("  - debug_cropped.jpg: The cropped region")
    print(f"  - {output_path}: Final resized image")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_crop.py <image_path>")
        print("Example: python test_crop.py image1.jpg")
        sys.exit(1)

    test_crop(sys.argv[1])