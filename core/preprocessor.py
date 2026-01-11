"""
Image preprocessor for receipt extraction.
Crops to the white data box and normalizes size.
ROBUST VERSION: Multiple fallback strategies.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class ReceiptPreprocessor:
    """Preprocessor to crop and normalize receipts."""

    def __init__(self, target_height: int = 800):
        """
        Initialize preprocessor.

        Args:
            target_height: Target height for normalized receipts (default: 800px)
        """
        self.target_height = target_height

    def preprocess(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Preprocess receipt image.

        Args:
            image: Input image
            debug: If True, show debug visualizations

        Returns:
            Preprocessed image (cropped and normalized)
        """
        h, w = image.shape[:2]

        # Step 1: Find the white data box
        box_coords = self._find_white_box_robust(image)

        if box_coords is None:
            print("  ⚠️  Could not find white box, using intelligent fallback")
            # Fallback: crop to estimated region
            x = int(w * 0.03)
            y = int(h * 0.25)
            box_w = int(w * 0.94)
            box_h = int(h * 0.50)
            box_coords = (x, y, box_w, box_h)

        x, y, box_w, box_h = box_coords
        print(f"  ✓ Crop region: x={x}, y={y}, w={box_w}, h={box_h}")

        # Step 2: Crop to the box
        cropped = image[y:y+box_h, x:x+box_w]

        if debug:
            # Show the detected box
            debug_img = image.copy()
            cv2.rectangle(debug_img, (x, y), (x+box_w, y+box_h), (0, 255, 0), 3)
            cv2.imwrite("debug_box_detection.jpg", debug_img)
            cv2.imwrite("debug_cropped.jpg", cropped)
            print(f"  Saved: debug_box_detection.jpg, debug_cropped.jpg")

        # Step 3: Resize to standard height (maintaining aspect ratio)
        if self.target_height > 0:
            crop_h, crop_w = cropped.shape[:2]
            scale = self.target_height / crop_h
            new_w = int(crop_w * scale)
            resized = cv2.resize(cropped, (new_w, self.target_height),
                               interpolation=cv2.INTER_LINEAR)
            print(f"  ✓ Resized: {crop_w}x{crop_h} → {new_w}x{self.target_height}")
            return resized

        return cropped

    def _find_white_box_robust(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the white data box using multiple strategies.

        Args:
            image: Input image

        Returns:
            (x, y, width, height) or None if not found
        """
        h, w = image.shape[:2]

        # Strategy 1: Horizontal projection (most reliable for this layout)
        box = self._find_by_projection(image)
        if box:
            print(f"  ✓ Found box using projection method")
            return box

        # Strategy 2: Edge-based detection
        box = self._find_by_edges(image)
        if box:
            print(f"  ✓ Found box using edge detection")
            return box

        # Strategy 3: Color-based (white region in green background)
        box = self._find_by_color(image)
        if box:
            print(f"  ✓ Found box using color detection")
            return box

        print(f"  ⚠️  All detection methods failed")
        return None

    def _find_by_projection(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Find box by analyzing horizontal brightness projection.
        The white box will have high brightness, green areas will be dark.
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate horizontal projection (average brightness per row)
        projection = np.mean(gray, axis=1)

        # Smooth the projection
        kernel_size = max(5, h // 100)
        projection_smooth = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')

        # Find the brightest continuous region (white box)
        threshold = np.mean(projection_smooth) + 20

        # Find start and end of bright region
        bright_rows = projection_smooth > threshold

        # Find continuous segments
        segments = []
        start = None
        for i, is_bright in enumerate(bright_rows):
            if is_bright and start is None:
                start = i
            elif not is_bright and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(bright_rows)))

        if not segments:
            return None

        # Pick the largest segment
        largest = max(segments, key=lambda s: s[1] - s[0])
        y_start, y_end = largest

        # Must be reasonably sized
        box_height = y_end - y_start
        if box_height < h * 0.2 or box_height > h * 0.7:
            return None

        # Find horizontal boundaries (vertical projection)
        roi = gray[y_start:y_end, :]
        v_projection = np.mean(roi, axis=0)
        v_projection_smooth = np.convolve(v_projection, np.ones(kernel_size)/kernel_size, mode='same')

        v_threshold = np.mean(v_projection_smooth) + 10
        bright_cols = v_projection_smooth > v_threshold

        # Find leftmost and rightmost bright columns
        bright_indices = np.where(bright_cols)[0]
        if len(bright_indices) < w * 0.3:
            return None

        x_start = bright_indices[0]
        x_end = bright_indices[-1]

        # Add small margin
        margin_y = int(box_height * 0.02)
        margin_x = int((x_end - x_start) * 0.01)

        y_start = max(0, y_start - margin_y)
        y_end = min(h, y_end + margin_y)
        x_start = max(0, x_start - margin_x)
        x_end = min(w, x_end + margin_x)

        return (x_start, y_start, x_end - x_start, y_end - y_start)

    def _find_by_edges(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find box by detecting the rounded rectangle border."""
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find rectangular contours
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            # Filter
            if area < (w * h * 0.15):  # Too small
                continue
            if area > (w * h * 0.95):  # Too large
                continue
            if x < w * 0.01 or y < h * 0.1:  # Too close to edge
                continue
            if x + cw > w * 0.99:
                continue

            # Check if it looks like a rectangle
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity > 0.85:  # Pretty rectangular
                    candidates.append((area, (x, y, cw, ch)))

        if candidates:
            # Return the largest valid box
            candidates.sort(reverse=True)
            return candidates[0][1]

        return None

    def _find_by_color(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find white box by detecting non-green regions."""
        if len(image.shape) != 3:
            return None

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert to get non-green (white box + others)
        non_green = cv2.bitwise_not(green_mask)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        non_green = cv2.morphologyEx(non_green, cv2.MORPH_CLOSE, kernel, iterations=2)
        non_green = cv2.morphologyEx(non_green, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(non_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest non-green region that's reasonably positioned
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            # Filter
            if area < (w * h * 0.15):
                continue
            if area > (w * h * 0.95):
                continue

            # Check position (should be in middle portion)
            center_y = y + ch / 2
            if center_y < h * 0.2 or center_y > h * 0.8:
                continue

            candidates.append((area, (x, y, cw, ch)))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

        return None


def preprocess_receipt(image: np.ndarray, target_height: int = 800,
                      debug: bool = False) -> np.ndarray:
    """
    Convenience function to preprocess a receipt.

    Args:
        image: Input image
        target_height: Target height for output (default: 800px, 0 = no resize)
        debug: Show debug visualizations

    Returns:
        Preprocessed image
    """
    preprocessor = ReceiptPreprocessor(target_height=target_height)
    return preprocessor.preprocess(image, debug=debug)