"""
Box detector for receipt field isolation.
Detects the rectangular boxes around each field for precise extraction.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FieldBox:
    """Represents a detected field box."""
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    area: int

    def contains_point(self, x: int, y: int, padding: int = 0) -> bool:
        """Check if a point is inside this box."""
        return (self.x - padding <= x <= self.x + self.width + padding and
                self.y - padding <= y <= self.y + self.height + padding)

    def get_roi(self, image: np.ndarray, padding: int = 5) -> np.ndarray:
        """Extract ROI from image with optional padding."""
        h, w = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w, self.x + self.width + padding)
        y2 = min(h, self.y + self.height + padding)
        return image[y1:y2, x1:x2]


class BoxDetector:
    """
    Detects rectangular boxes around fields in receipts.
    Uses computer vision to find field boundaries.
    """

    def __init__(self):
        """Initialize box detector."""
        # Minimum box dimensions (to filter out noise)
        # More lenient for hybrid approach (working within slices)
        self.min_width = 80
        self.min_height = 20
        self.max_width = 800
        self.max_height = 200

        # Aspect ratio constraints (more flexible)
        self.min_aspect_ratio = 1.5  # width / height
        self.max_aspect_ratio = 30.0

    def detect_field_boxes(
        self,
        image: np.ndarray,
        debug: bool = False
    ) -> List[FieldBox]:
        """
        Detect all field boxes in the receipt image.

        Args:
            image: Receipt image (color or grayscale)
            debug: Whether to save debug visualization

        Returns:
            List of detected FieldBox objects, sorted top-to-bottom
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and convert contours to FieldBox objects
        boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Apply size filters
            if not self._is_valid_field_box(w, h):
                continue

            # Create FieldBox
            box = FieldBox(
                x=x,
                y=y,
                width=w,
                height=h,
                center_x=x + w // 2,
                center_y=y + h // 2,
                area=w * h
            )
            boxes.append(box)

        # Sort boxes top-to-bottom (by y-coordinate)
        boxes.sort(key=lambda b: b.y)

        # Merge overlapping boxes
        boxes = self._merge_overlapping_boxes(boxes)

        if debug:
            self._visualize_boxes(image, boxes)

        return boxes

    def _is_valid_field_box(self, width: int, height: int) -> bool:
        """Check if dimensions match field box criteria."""
        # Size constraints
        if width < self.min_width or width > self.max_width:
            return False
        if height < self.min_height or height > self.max_height:
            return False

        # Aspect ratio constraints
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False

        return True

    def _merge_overlapping_boxes(
        self,
        boxes: List[FieldBox],
        iou_threshold: float = 0.3
    ) -> List[FieldBox]:
        """
        Merge boxes that overlap significantly.
        Handles cases where one field has multiple detected boxes.
        """
        if not boxes:
            return boxes

        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            # Find all boxes that overlap with this one
            overlapping = [box1]
            for j, box2 in enumerate(boxes):
                if i != j and j not in used:
                    if self._calculate_iou(box1, box2) > iou_threshold:
                        overlapping.append(box2)
                        used.add(j)

            # Merge overlapping boxes
            if len(overlapping) > 1:
                merged_box = self._merge_boxes(overlapping)
                merged.append(merged_box)
            else:
                merged.append(box1)

            used.add(i)

        return merged

    def _calculate_iou(self, box1: FieldBox, box2: FieldBox) -> float:
        """Calculate Intersection over Union between two boxes."""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_boxes(self, boxes: List[FieldBox]) -> FieldBox:
        """Merge multiple boxes into one encompassing box."""
        x_min = min(b.x for b in boxes)
        y_min = min(b.y for b in boxes)
        x_max = max(b.x + b.width for b in boxes)
        y_max = max(b.y + b.height for b in boxes)

        width = x_max - x_min
        height = y_max - y_min

        return FieldBox(
            x=x_min,
            y=y_min,
            width=width,
            height=height,
            center_x=x_min + width // 2,
            center_y=y_min + height // 2,
            area=width * height
        )

    def _visualize_boxes(self, image: np.ndarray, boxes: List[FieldBox]):
        """Draw detected boxes on image for debugging."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for i, box in enumerate(boxes):
            # Draw rectangle
            cv2.rectangle(
                vis,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 255, 0),
                2
            )

            # Draw box number
            cv2.putText(
                vis,
                str(i),
                (box.x + 5, box.y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Save visualization
        from config.settings import OUTPUT_DIR
        output_path = OUTPUT_DIR / "debug_boxes.png"
        cv2.imwrite(str(output_path), vis)
        print(f"  [DEBUG] Box visualization saved: {output_path}")

    def map_labels_to_boxes(
        self,
        boxes: List[FieldBox],
        ocr_results: List,
        label_map: Dict[str, List[str]]
    ) -> Dict[str, FieldBox]:
        """
        Map field labels to their containing boxes.

        Args:
            boxes: Detected field boxes
            ocr_results: OCR results with bounding boxes
            label_map: Dict of {field_name: [label_variants]}

        Returns:
            Dict of {field_name: FieldBox}
        """
        field_boxes = {}

        for field_name, label_variants in label_map.items():
            # Find OCR result matching this label
            for ocr_result in ocr_results:
                for label in label_variants:
                    if self._text_matches_label(ocr_result.text, label):
                        # Find which box contains this OCR result
                        if ocr_result.bbox:
                            center_x = int(np.mean([p[0] for p in ocr_result.bbox]))
                            center_y = int(np.mean([p[1] for p in ocr_result.bbox]))

                            # Find containing box
                            for box in boxes:
                                if box.contains_point(center_x, center_y, padding=10):
                                    field_boxes[field_name] = box
                                    break
                        break

        return field_boxes

    def _text_matches_label(self, text: str, label: str) -> bool:
        """Check if OCR text matches a label (fuzzy match)."""
        text_clean = text.replace(' ', '')
        label_clean = label.replace(' ', '')

        if label_clean in text_clean or text_clean in label_clean:
            return True

        if len(label_clean) > 3:
            matches = sum(1 for a, b in zip(text_clean, label_clean) if a == b)
            if matches / len(label_clean) > 0.8:
                return True

        return False


# Convenience function
def create_box_detector() -> BoxDetector:
    """Factory function to create box detector."""
    return BoxDetector()