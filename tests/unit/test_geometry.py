"""
Unit tests for geometry module.
"""

import pytest
import numpy as np
import math

from src.core.geometry import (
    get_box_center,
    get_box_center_int,
    get_box_width,
    get_box_height,
    get_box_area,
    check_overlap,
    calculate_overlap_area,
    calculate_iou,
    is_point_in_circle,
    euclidean_distance,
    get_homography_matrix,
    transform_point,
    pixels_to_cm,
    cm_to_pixels,
)


class TestBoxOperations:
    """Tests for bounding box operations."""

    def test_get_box_center(self):
        """Test box center calculation."""
        box = (100, 200, 300, 400)
        cx, cy = get_box_center(box)
        assert cx == 200.0
        assert cy == 300.0

    def test_get_box_center_int(self):
        """Test box center as integers."""
        box = (100, 200, 301, 401)
        cx, cy = get_box_center_int(box)
        assert cx == 200
        assert cy == 300
        assert isinstance(cx, int)
        assert isinstance(cy, int)

    def test_get_box_width(self):
        """Test box width calculation."""
        box = (100, 200, 350, 400)
        assert get_box_width(box) == 250

    def test_get_box_height(self):
        """Test box height calculation."""
        box = (100, 200, 300, 500)
        assert get_box_height(box) == 300

    def test_get_box_area(self):
        """Test box area calculation."""
        box = (0, 0, 100, 50)
        assert get_box_area(box) == 5000.0


class TestOverlapDetection:
    """Tests for overlap detection (Paper Algorithm 2 & 3)."""

    def test_overlapping_boxes(self, overlapping_boxes):
        """Test detection of overlapping boxes."""
        box1, box2 = overlapping_boxes
        is_overlap, area = check_overlap(box1, box2)
        assert is_overlap is True
        assert area == 2500  # 50x50 overlap

    def test_non_overlapping_boxes(self, non_overlapping_boxes):
        """Test detection of non-overlapping boxes."""
        box1, box2 = non_overlapping_boxes
        is_overlap, area = check_overlap(box1, box2)
        assert is_overlap is False
        assert area == 0

    def test_calculate_overlap_area_partial(self):
        """Test partial overlap area calculation."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        area = calculate_overlap_area(box1, box2)
        assert area == 2500  # 50x50

    def test_calculate_overlap_area_contained(self):
        """Test when one box is inside another."""
        outer = (0, 0, 200, 200)
        inner = (50, 50, 100, 100)
        area = calculate_overlap_area(outer, inner)
        assert area == 2500  # Area of inner box

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes."""
        box = (100, 100, 200, 200)
        iou = calculate_iou(box, box)
        assert iou == 1.0

    def test_iou_no_overlap(self, non_overlapping_boxes):
        """Test IoU with no overlap."""
        box1, box2 = non_overlapping_boxes
        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        # Overlap: 50x100 = 5000
        # Union: 100*100 + 100*100 - 5000 = 15000
        iou = calculate_iou(box1, box2)
        assert abs(iou - (5000 / 15000)) < 0.001


class TestCircleContainment:
    """Tests for point-in-circle checks."""

    def test_point_inside_circle(self):
        """Test point inside circle."""
        assert is_point_in_circle((50, 50), (50, 50), 10) is True

    def test_point_on_circle_edge(self):
        """Test point on circle edge."""
        assert is_point_in_circle((60, 50), (50, 50), 10) is True

    def test_point_outside_circle(self):
        """Test point outside circle."""
        assert is_point_in_circle((100, 100), (50, 50), 10) is False

    def test_point_with_none_center(self):
        """Test with None center."""
        assert is_point_in_circle((50, 50), None, 10) is False


class TestDistanceCalculations:
    """Tests for distance calculations."""

    def test_euclidean_distance_same_point(self):
        """Test distance to same point."""
        assert euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_euclidean_distance_horizontal(self):
        """Test horizontal distance."""
        assert euclidean_distance((0, 0), (10, 0)) == 10.0

    def test_euclidean_distance_diagonal(self):
        """Test diagonal distance (3-4-5 triangle)."""
        assert euclidean_distance((0, 0), (3, 4)) == 5.0


class TestHomographyTransform:
    """Tests for homography transformation."""

    def test_homography_matrix_creation(self, sample_pen_corners):
        """Test homography matrix creation."""
        matrix = get_homography_matrix(sample_pen_corners, 120.0, 200.0)
        assert matrix.shape == (3, 3)
        assert matrix is not None

    def test_transform_point_with_none_matrix(self):
        """Test transform with None matrix returns original point."""
        point = (100, 200)
        result = transform_point(point, None)
        assert result == point

    def test_transform_point_identity(self):
        """Test transform with corners at destination."""
        # When corners match destination, transform should be identity-like
        corners = np.array([
            [0, 0], [120, 0], [120, 200], [0, 200]
        ], dtype=np.float32)
        matrix = get_homography_matrix(corners, 120.0, 200.0)

        # Transform a point
        result = transform_point((60, 100), matrix)
        assert abs(result[0] - 60) < 1.0
        assert abs(result[1] - 100) < 1.0


class TestUnitConversions:
    """Tests for unit conversions."""

    def test_pixels_to_cm_default_scale(self):
        """Test pixel to cm conversion with default scale."""
        assert pixels_to_cm(100, 0.2) == 20.0

    def test_cm_to_pixels_default_scale(self):
        """Test cm to pixel conversion with default scale."""
        assert cm_to_pixels(20.0, 0.2) == 100.0

    def test_cm_to_pixels_zero_scale(self):
        """Test cm to pixel with zero scale (edge case)."""
        assert cm_to_pixels(20.0, 0) == 0

    def test_roundtrip_conversion(self):
        """Test pixels -> cm -> pixels roundtrip."""
        original = 500
        scale = 0.2
        cm = pixels_to_cm(original, scale)
        back = cm_to_pixels(cm, scale)
        assert back == original
