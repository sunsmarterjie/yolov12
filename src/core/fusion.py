"""
Multi-view fusion combining top and side camera perspectives.

This module provides:
- Homography-based coordinate transformation
- Multi-view detection fusion
- Cross-camera track matching

Designed for combining overhead and side-angle camera views.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np
import cv2

from .geometry import (
    get_homography_matrix, transform_point, is_point_in_polygon,
    get_box_center, pixels_to_cm, calculate_iou
)
from .inference import Detection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FusedDetection:
    """
    Detection result fused from multiple camera views.

    Attributes:
        track_id: Persistent tracking ID
        class_name: Object class name
        world_position: Position in world coordinates (cm)
        world_radius: Estimated radius in world coordinates (cm)
        confidence: Combined confidence score
        top_detection: Original detection from top camera
        side_detection: Matched detection from side camera (if any)
        side_box: Bounding box from side view (if available)
    """
    track_id: int
    class_name: str
    world_position: Tuple[float, float]
    world_radius: float
    confidence: float = 0.0
    top_detection: Optional[Detection] = None
    side_detection: Optional[Detection] = None
    side_box: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for state machine compatibility."""
        return {
            'id': self.track_id,
            'type': self.class_name,
            'pos': self.world_position,
            'radius': self.world_radius,
            'confidence': self.confidence
        }


@dataclass
class HomographyConfig:
    """
    Configuration for homography transformation.

    Attributes:
        pen_corners: 4 corner points in camera frame (pixels)
        pen_width_cm: Physical pen width in centimeters
        pen_height_cm: Physical pen height in centimeters
    """
    pen_corners: np.ndarray  # Shape: (4, 2)
    pen_width_cm: float = 120.0
    pen_height_cm: float = 200.0

    @classmethod
    def from_file(cls, path: str, width_cm: float = 120.0, height_cm: float = 200.0):
        """Load configuration from numpy file."""
        corners = np.load(path)
        return cls(pen_corners=corners, pen_width_cm=width_cm, pen_height_cm=height_cm)


# =============================================================================
# Homography Transform
# =============================================================================

class HomographyTransform:
    """
    Handles perspective transformation between camera and world coordinates.

    Provides bidirectional transformation and boundary checking.
    """

    def __init__(self, config: HomographyConfig):
        """
        Initialize homography transform.

        Args:
            config: Homography configuration with pen corners and dimensions
        """
        self.config = config
        self._matrix: Optional[np.ndarray] = None
        self._inv_matrix: Optional[np.ndarray] = None
        self._compute_homography()

    def _compute_homography(self) -> None:
        """Compute forward and inverse homography matrices."""
        self._matrix = get_homography_matrix(
            self.config.pen_corners,
            self.config.pen_width_cm,
            self.config.pen_height_cm
        )
        self._inv_matrix = np.linalg.inv(self._matrix)

    def camera_to_world(
        self,
        point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Transform camera coordinates to world coordinates.

        Args:
            point: (x, y) in camera frame (pixels)

        Returns:
            (x, y) in world coordinates (centimeters)
        """
        return transform_point(point, self._matrix)

    def world_to_camera(
        self,
        point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Transform world coordinates to camera coordinates.

        Args:
            point: (x, y) in world coordinates (centimeters)

        Returns:
            (x, y) in camera frame (pixels)
        """
        return transform_point(point, self._inv_matrix)

    def is_inside_pen(self, camera_point: Tuple[float, float]) -> bool:
        """
        Check if a camera point is inside the pen boundary.

        Args:
            camera_point: (x, y) in camera frame

        Returns:
            True if point is inside pen polygon
        """
        return is_point_in_polygon(camera_point, self.config.pen_corners)

    @property
    def matrix(self) -> np.ndarray:
        """Get forward homography matrix."""
        return self._matrix

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Get inverse homography matrix."""
        return self._inv_matrix


# =============================================================================
# Multi-View Fusion
# =============================================================================

class MultiViewFusion:
    """
    Fuses detections from top and side camera views.

    Strategy:
    1. Top camera provides accurate world position via homography
    2. Side camera provides additional features (pose, height info)
    3. Track IDs are matched across views using class + proximity/IoU
    """

    def __init__(
        self,
        homography: HomographyTransform,
        match_threshold: float = 0.5,
        pixels_to_cm_scale: float = 0.2
    ):
        """
        Initialize multi-view fusion.

        Args:
            homography: Homography transform for top camera
            match_threshold: IoU threshold for cross-view matching
            pixels_to_cm_scale: Conversion factor for radius estimation
        """
        self.homography = homography
        self.match_threshold = match_threshold
        self.scale = pixels_to_cm_scale

    def fuse(
        self,
        top_detections: List[Detection],
        side_detections: Optional[List[Detection]] = None
    ) -> List[FusedDetection]:
        """
        Fuse detections from both camera views.

        Args:
            top_detections: Detections from overhead camera
            side_detections: Optional detections from side camera

        Returns:
            List of fused detections with world coordinates
        """
        side_detections = side_detections or []
        fused = []

        for top_det in top_detections:
            # Get box center in camera coordinates
            center_cam = get_box_center(top_det.box)

            # Filter by pen boundary
            if not self.homography.is_inside_pen(center_cam):
                continue

            # Transform to world coordinates
            world_pos = self.homography.camera_to_world(center_cam)

            # Estimate radius from box width
            width_pixels = abs(top_det.box[2] - top_det.box[0])
            radius_cm = pixels_to_cm(width_pixels / 2, self.scale)

            # Try to find matching side detection
            matched_side = self._find_side_match(top_det, side_detections)

            fused_det = FusedDetection(
                track_id=top_det.track_id or -1,
                class_name=top_det.class_name,
                world_position=world_pos,
                world_radius=radius_cm,
                confidence=top_det.confidence,
                top_detection=top_det,
                side_detection=matched_side,
                side_box=matched_side.box if matched_side else None
            )
            fused.append(fused_det)

        return fused

    def _find_side_match(
        self,
        top_det: Detection,
        side_detections: List[Detection]
    ) -> Optional[Detection]:
        """
        Find matching detection in side view.

        Uses track_id if available, otherwise falls back to
        class + confidence matching.

        Args:
            top_det: Detection from top camera
            side_detections: List of detections from side camera

        Returns:
            Matching side detection or None
        """
        if not side_detections:
            return None

        # First try exact track_id match
        if top_det.track_id is not None:
            for side_det in side_detections:
                if (side_det.track_id == top_det.track_id and
                        side_det.class_name == top_det.class_name):
                    return side_det

        # Fallback: Find best class match by confidence
        candidates = [
            d for d in side_detections
            if d.class_name == top_det.class_name
        ]

        if candidates:
            return max(candidates, key=lambda d: d.confidence)

        return None

    def fuse_to_dicts(
        self,
        top_detections: List[Detection],
        side_detections: Optional[List[Detection]] = None
    ) -> List[dict]:
        """
        Fuse detections and return as dictionary list.

        Convenience method for compatibility with state machine.

        Args:
            top_detections: Detections from overhead camera
            side_detections: Optional detections from side camera

        Returns:
            List of detection dictionaries
        """
        fused = self.fuse(top_detections, side_detections)
        return [f.to_dict() for f in fused]


# =============================================================================
# Factory Functions
# =============================================================================

def create_fusion_pipeline(
    pen_config_path: str,
    pen_width_cm: float = 120.0,
    pen_height_cm: float = 200.0,
    match_threshold: float = 0.5
) -> MultiViewFusion:
    """
    Factory function to create configured fusion pipeline.

    Args:
        pen_config_path: Path to pen calibration file (pen_config.npy)
        pen_width_cm: Physical pen width
        pen_height_cm: Physical pen height
        match_threshold: Cross-view matching threshold

    Returns:
        Configured MultiViewFusion instance
    """
    config = HomographyConfig.from_file(
        pen_config_path,
        pen_width_cm,
        pen_height_cm
    )
    homography = HomographyTransform(config)
    return MultiViewFusion(homography, match_threshold)


def load_homography(
    pen_config_path: str,
    pen_width_cm: float = 120.0,
    pen_height_cm: float = 200.0
) -> HomographyTransform:
    """
    Load homography transform from calibration file.

    Args:
        pen_config_path: Path to pen_config.npy
        pen_width_cm: Physical pen width
        pen_height_cm: Physical pen height

    Returns:
        Configured HomographyTransform instance
    """
    config = HomographyConfig.from_file(
        pen_config_path,
        pen_width_cm,
        pen_height_cm
    )
    return HomographyTransform(config)
