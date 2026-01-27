"""
Geometric utilities for poultry vision system.

This module provides:
- Bounding box operations (center, width, overlap)
- Homography transformations (camera to world coordinates)
- Spatial checks (point in circle, polygon containment)
- Color palette for visualization

Merged from utils.py and utilss.py.
Implements algorithms from MDPI paper on poultry welfare assessment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import cv2


# --- Color Palette (BGR format for OpenCV) ---
COLORS = {
    'feeder': (0, 140, 255),      # Orange
    'waterer': (235, 206, 135),   # Sky Blue
    'hen': (0, 255, 0),           # Green (Idle)
    'hen_eating': (0, 0, 255),    # Red
    'hen_drinking': (255, 0, 0),  # Blue
    'text': (255, 255, 255),      # White
    'text_bg': (0, 0, 0),         # Black
    'trail': (100, 100, 100),     # Grey
    'pen_boundary': (0, 0, 255),  # Red
}


# =============================================================================
# Bounding Box Operations
# =============================================================================

def get_box_center(box: tuple) -> Tuple[float, float]:
    """
    Calculate the center (x, y) of a bounding box.

    Implements Paper Equation (5).

    Args:
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_box_center_int(box: tuple) -> Tuple[int, int]:
    """
    Calculate the center (x, y) of a bounding box as integers.

    Args:
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Tuple of (center_x, center_y) as integers
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_box_width(box: tuple) -> int:
    """
    Calculate the width of a bounding box.

    Used to estimate radius of feeder/waterer from detection.

    Args:
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Width of the box in pixels
    """
    return int(abs(box[2] - box[0]))


def get_box_height(box: tuple) -> int:
    """
    Calculate the height of a bounding box.

    Args:
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Height of the box in pixels
    """
    return int(abs(box[3] - box[1]))


def get_box_area(box: tuple) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Area of the box in square pixels
    """
    return abs(box[2] - box[0]) * abs(box[3] - box[1])


# =============================================================================
# Overlap Detection (Paper Algorithms 2 & 3)
# =============================================================================

def calculate_overlap_area(box1: tuple, box2: tuple) -> float:
    """
    Calculate the intersection area between two bounding boxes.

    Implements Algorithm 3 from the MDPI paper.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        Intersection area (0 if no overlap)
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # Calculate overlap dimensions
    x_overlap = max(0, min(right1, right2) - max(left1, left2))
    y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))

    return x_overlap * y_overlap


def check_overlap(box1: tuple, box2: tuple) -> Tuple[bool, float]:
    """
    Check if two bounding boxes overlap and return the intersection area.

    Implements Algorithm 2 from the MDPI paper.

    Args:
        box1: First bounding box (x1, y1, x2, y2), e.g., hen
        box2: Second bounding box (x1, y1, x2, y2), e.g., feeder/waterer

    Returns:
        Tuple of (is_overlapping, overlap_area)
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # Check if boxes are disjoint (Algorithm 2 logic)
    if (left1 >= right2) or (right1 <= left2) or \
       (top1 >= bottom2) or (bottom1 <= top2):
        return False, 0.0

    # Calculate area if they overlap
    area = calculate_overlap_area(box1, box2)
    return True, area


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    intersection = calculate_overlap_area(box1, box2)
    if intersection == 0:
        return 0.0

    area1 = get_box_area(box1)
    area2 = get_box_area(box2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# =============================================================================
# Homography Transformations
# =============================================================================

def get_homography_matrix(
    src_pts: np.ndarray,
    dst_width: float,
    dst_height: float
) -> np.ndarray:
    """
    Compute perspective transformation matrix from source points to destination.

    Args:
        src_pts: 4 corner points in source image (camera frame) as (4, 2) array
                 Order: top-left, top-right, bottom-right, bottom-left
        dst_width: Width of destination space (e.g., pen width in cm)
        dst_height: Height of destination space (e.g., pen height in cm)

    Returns:
        3x3 homography matrix
    """
    dst_pts = np.array([
        [0, 0],
        [dst_width, 0],
        [dst_width, dst_height],
        [0, dst_height]
    ], dtype=np.float32)

    src_pts_float = src_pts.astype(np.float32)
    return cv2.getPerspectiveTransform(src_pts_float, dst_pts)


def transform_point(
    point: Tuple[float, float],
    matrix: Optional[np.ndarray]
) -> Tuple[float, float]:
    """
    Apply homography transformation to a single point.

    Transforms camera coordinates to world coordinates.

    Args:
        point: (x, y) coordinates in source space
        matrix: 3x3 homography matrix (or None for identity transform)

    Returns:
        Transformed (x, y) coordinates
    """
    if matrix is None:
        return point

    p = np.array([point[0], point[1], 1], dtype=np.float32)
    warped = matrix @ p

    # Perspective divide
    if warped[2] != 0:
        warped = warped / warped[2]

    return float(warped[0]), float(warped[1])


def transform_point_int(
    point: Tuple[float, float],
    matrix: Optional[np.ndarray]
) -> Tuple[int, int]:
    """
    Apply homography transformation and return integer coordinates.

    Args:
        point: (x, y) coordinates in source space
        matrix: 3x3 homography matrix

    Returns:
        Transformed (x, y) coordinates as integers
    """
    x, y = transform_point(point, matrix)
    return int(x), int(y)


def get_inverse_homography(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a homography matrix.

    Useful for transforming world coordinates back to camera space.

    Args:
        matrix: 3x3 homography matrix

    Returns:
        Inverse 3x3 homography matrix
    """
    return np.linalg.inv(matrix)


# =============================================================================
# Spatial Checks
# =============================================================================

def is_point_in_circle(
    point: Tuple[float, float],
    center: Optional[Tuple[float, float]],
    radius: float
) -> bool:
    """
    Check if a point is inside a circle.

    Used for interaction detection (hen near feeder/waterer).

    Args:
        point: (x, y) coordinates of the point
        center: (x, y) coordinates of the circle center
        radius: Radius of the circle

    Returns:
        True if point is inside (or on) the circle
    """
    if center is None:
        return False

    dist = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return dist <= radius


def euclidean_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Distance between the points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def is_point_in_polygon(
    point: Tuple[float, float],
    polygon: np.ndarray
) -> bool:
    """
    Check if a point is inside a polygon using OpenCV.

    Args:
        point: (x, y) coordinates
        polygon: Polygon vertices as numpy array

    Returns:
        True if point is inside the polygon
    """
    result = cv2.pointPolygonTest(polygon.astype(np.int32), point, False)
    return result >= 0


# =============================================================================
# Coordinate Conversion Utilities
# =============================================================================

def pixels_to_cm(pixels: float, scale_factor: float = 0.2) -> float:
    """
    Convert pixel measurement to centimeters.

    Args:
        pixels: Measurement in pixels
        scale_factor: Conversion factor (cm/pixel), default 0.2

    Returns:
        Measurement in centimeters
    """
    return pixels * scale_factor


def cm_to_pixels(cm: float, scale_factor: float = 0.2) -> float:
    """
    Convert centimeter measurement to pixels.

    Args:
        cm: Measurement in centimeters
        scale_factor: Conversion factor (cm/pixel), default 0.2

    Returns:
        Measurement in pixels
    """
    return cm / scale_factor if scale_factor > 0 else 0


def scale_point(
    point: Tuple[float, float],
    scale: float
) -> Tuple[int, int]:
    """
    Scale a point by a factor (for minimap rendering).

    Args:
        point: (x, y) coordinates
        scale: Scale factor

    Returns:
        Scaled (x, y) coordinates as integers
    """
    return int(point[0] * scale), int(point[1] * scale)


# =============================================================================
# Dataclass for Homography Configuration
# =============================================================================

@dataclass
class HomographyConfig:
    """Configuration for homography transformation."""
    pen_corners: np.ndarray  # 4x2 array of corner points (pixels)
    pen_width_cm: float = 120.0
    pen_height_cm: float = 200.0

    def get_matrix(self) -> np.ndarray:
        """Compute and return the homography matrix."""
        return get_homography_matrix(
            self.pen_corners,
            self.pen_width_cm,
            self.pen_height_cm
        )
