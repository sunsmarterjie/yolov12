"""
Integration tests for multi-view fusion.
"""

import pytest
import numpy as np

from src.core.fusion import (
    HomographyConfig,
    HomographyTransform,
    MultiViewFusion,
    FusedDetection,
    load_homography,
)
from src.core.inference import Detection


class TestHomographyTransform:
    """Tests for HomographyTransform class."""

    def test_camera_to_world_center(self, sample_pen_corners):
        """Test camera to world transformation at pen center."""
        config = HomographyConfig(
            pen_corners=sample_pen_corners,
            pen_width_cm=120.0,
            pen_height_cm=200.0
        )
        transform = HomographyTransform(config)

        # Center of pen in camera coords
        camera_center = (400.0, 300.0)
        world_pos = transform.camera_to_world(camera_center)

        # Should be approximately center of world space
        assert 55 < world_pos[0] < 65  # ~60 cm
        assert 95 < world_pos[1] < 105  # ~100 cm

    def test_camera_to_world_corners(self, sample_pen_corners):
        """Test transformation at corners."""
        config = HomographyConfig(
            pen_corners=sample_pen_corners,
            pen_width_cm=120.0,
            pen_height_cm=200.0
        )
        transform = HomographyTransform(config)

        # Top-left corner
        tl = transform.camera_to_world((100.0, 100.0))
        assert abs(tl[0] - 0) < 1.0
        assert abs(tl[1] - 0) < 1.0

        # Bottom-right corner
        br = transform.camera_to_world((700.0, 500.0))
        assert abs(br[0] - 120) < 1.0
        assert abs(br[1] - 200) < 1.0

    def test_world_to_camera_roundtrip(self, sample_pen_corners):
        """Test camera->world->camera roundtrip."""
        config = HomographyConfig(
            pen_corners=sample_pen_corners,
            pen_width_cm=120.0,
            pen_height_cm=200.0
        )
        transform = HomographyTransform(config)

        original = (400.0, 300.0)
        world = transform.camera_to_world(original)
        back = transform.world_to_camera(world)

        assert abs(back[0] - original[0]) < 1.0
        assert abs(back[1] - original[1]) < 1.0

    def test_is_inside_pen(self, sample_pen_corners):
        """Test pen boundary detection."""
        config = HomographyConfig(
            pen_corners=sample_pen_corners,
            pen_width_cm=120.0,
            pen_height_cm=200.0
        )
        transform = HomographyTransform(config)

        # Point inside
        assert transform.is_inside_pen((400, 300)) is True

        # Point outside
        assert transform.is_inside_pen((50, 50)) is False
        assert transform.is_inside_pen((800, 600)) is False

    def test_matrix_properties(self, sample_pen_corners):
        """Test matrix is properly computed."""
        config = HomographyConfig(
            pen_corners=sample_pen_corners,
            pen_width_cm=120.0,
            pen_height_cm=200.0
        )
        transform = HomographyTransform(config)

        assert transform.matrix.shape == (3, 3)
        assert transform.inverse_matrix.shape == (3, 3)


class TestMultiViewFusion:
    """Tests for MultiViewFusion class."""

    def test_fuse_single_detection(self, sample_pen_corners):
        """Test fusing a single detection."""
        config = HomographyConfig(pen_corners=sample_pen_corners)
        homography = HomographyTransform(config)
        fusion = MultiViewFusion(homography)

        top_det = Detection(
            box=(350.0, 250.0, 450.0, 350.0),
            class_id=1,
            class_name="hen",
            confidence=0.9,
            track_id=1
        )

        fused = fusion.fuse([top_det], [])

        assert len(fused) == 1
        assert fused[0].track_id == 1
        assert fused[0].class_name == "hen"
        assert fused[0].world_position[0] > 0
        assert fused[0].world_position[1] > 0

    def test_fuse_filters_outside_pen(self, sample_pen_corners):
        """Test that detections outside pen are filtered."""
        config = HomographyConfig(pen_corners=sample_pen_corners)
        homography = HomographyTransform(config)
        fusion = MultiViewFusion(homography)

        # Detection outside pen boundary
        outside_det = Detection(
            box=(10.0, 10.0, 50.0, 50.0),
            class_id=1,
            class_name="hen",
            confidence=0.9,
            track_id=1
        )

        fused = fusion.fuse([outside_det], [])

        assert len(fused) == 0

    def test_fuse_matches_side_by_track_id(self, sample_pen_corners):
        """Test that side detections are matched by track_id."""
        config = HomographyConfig(pen_corners=sample_pen_corners)
        homography = HomographyTransform(config)
        fusion = MultiViewFusion(homography)

        top_det = Detection(
            box=(350.0, 250.0, 450.0, 350.0),
            class_id=1, class_name="hen",
            confidence=0.9, track_id=5
        )

        side_det = Detection(
            box=(100.0, 100.0, 200.0, 200.0),
            class_id=1, class_name="hen",
            confidence=0.85, track_id=5
        )

        fused = fusion.fuse([top_det], [side_det])

        assert len(fused) == 1
        assert fused[0].side_detection is not None
        assert fused[0].side_detection.track_id == 5

    def test_fuse_to_dicts(self, sample_pen_corners):
        """Test conversion to dictionary format."""
        config = HomographyConfig(pen_corners=sample_pen_corners)
        homography = HomographyTransform(config)
        fusion = MultiViewFusion(homography)

        top_det = Detection(
            box=(350.0, 250.0, 450.0, 350.0),
            class_id=1, class_name="hen",
            confidence=0.9, track_id=1
        )

        dicts = fusion.fuse_to_dicts([top_det], [])

        assert len(dicts) == 1
        assert "id" in dicts[0]
        assert "type" in dicts[0]
        assert "pos" in dicts[0]
        assert dicts[0]["type"] == "hen"


class TestFusedDetection:
    """Tests for FusedDetection class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fused = FusedDetection(
            track_id=1,
            class_name="hen",
            world_position=(50.0, 60.0),
            world_radius=10.0,
            confidence=0.9
        )

        d = fused.to_dict()
        assert d["id"] == 1
        assert d["type"] == "hen"
        assert d["pos"] == (50.0, 60.0)
        assert d["radius"] == 10.0
        assert d["confidence"] == 0.9
