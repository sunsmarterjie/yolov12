"""
Inference engine with support for multiple backends.

This module provides:
- Backend-agnostic inference abstraction
- PyTorch/Ultralytics backend for desktop/GPU
- Hailo-8L backend stub for Raspberry Pi 5
- CameraView class for frame processing with tracking

Refactored from perception.py with added backend abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import time
import numpy as np
import cv2

from .geometry import (
    COLORS, get_box_center_int, get_box_width,
    transform_point, is_point_in_polygon
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Detection:
    """
    Single detection result from inference.

    Attributes:
        box: Bounding box as (x1, y1, x2, y2)
        class_id: Numeric class identifier
        class_name: Human-readable class name
        confidence: Detection confidence score (0-1)
        track_id: Optional persistent tracking ID
    """
    box: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    def get_center(self) -> Tuple[int, int]:
        """Get box center as integer coordinates."""
        return get_box_center_int(self.box)

    def get_width(self) -> int:
        """Get box width in pixels."""
        return get_box_width(self.box)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'id': self.track_id,
            'type': self.class_name,
            'box': self.box,
            'confidence': self.confidence,
            'class_id': self.class_id
        }


@dataclass
class InferenceResult:
    """
    Complete inference result for a frame.

    Attributes:
        detections: List of Detection objects
        frame_id: Frame sequence number
        inference_time_ms: Time taken for inference
        annotated_frame: Optional annotated frame image
    """
    detections: List[Detection]
    frame_id: int = 0
    inference_time_ms: float = 0.0
    annotated_frame: Optional[np.ndarray] = None

    def get_by_class(self, class_name: str) -> List[Detection]:
        """Filter detections by class name."""
        return [d for d in self.detections if d.class_name == class_name]


# =============================================================================
# Backend Abstraction
# =============================================================================

class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    Implementations should handle model loading, inference,
    and optionally tracking.
    """

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """
        Load a model from disk.

        Args:
            model_path: Path to model file (.pt, .hef, .onnx, etc.)
        """
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def track(
        self,
        frame: np.ndarray,
        persist: bool = True
    ) -> List[Detection]:
        """
        Run inference with object tracking.

        Args:
            frame: BGR image as numpy array
            persist: Maintain track IDs across frames

        Returns:
            List of Detection objects with track_id populated
        """
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to names."""
        pass


class PyTorchBackend(InferenceBackend):
    """
    PyTorch/Ultralytics backend for desktop and GPU inference.

    Uses the custom ultralytics fork with YOLOv12 support.
    """

    def __init__(self, tracker_config: str = "bytetrack.yaml"):
        """
        Initialize PyTorch backend.

        Args:
            tracker_config: Path to tracker configuration YAML
        """
        self._model = None
        self._class_names: Dict[int, str] = {}
        self.tracker_config = tracker_config
        self._track_config = {
            "persist": True,
            "verbose": False,
            "tracker": tracker_config
        }

    def load_model(self, model_path: Path) -> None:
        """Load Ultralytics YOLO model."""
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        self._class_names = dict(self._model.names)

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference without tracking."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model(frame, verbose=False)
        return self._parse_results(results[0], with_tracking=False)

    def track(
        self,
        frame: np.ndarray,
        persist: bool = True
    ) -> List[Detection]:
        """Run inference with ByteTrack tracking."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = self._track_config.copy()
        config["persist"] = persist

        results = self._model.track(frame, **config)
        return self._parse_results(results[0], with_tracking=True)

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    def _parse_results(
        self,
        result: Any,
        with_tracking: bool = False
    ) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        track_ids = None
        if with_tracking and boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()

        for i in range(len(xyxy)):
            det = Detection(
                box=tuple(xyxy[i]),
                class_id=int(classes[i]),
                class_name=self._class_names.get(int(classes[i]), "unknown"),
                confidence=float(confs[i]),
                track_id=int(track_ids[i]) if track_ids is not None else None
            )
            detections.append(det)

        return detections


class HailoBackend(InferenceBackend):
    """
    Hailo-8L backend for Raspberry Pi 5 deployment.

    Uses HailoRT for NPU-accelerated inference.
    Note: This is a stub implementation - actual Hailo integration
    requires the hailo_platform package and compiled .hef model.
    """

    def __init__(self):
        self._hef = None
        self._vdevice = None
        self._network_group = None
        self._input_info = None
        self._output_info = None
        self._class_names: Dict[int, str] = {
            0: 'feeder',
            1: 'hen',
            2: 'waterer'
        }
        self._input_shape = (640, 640)

    def load_model(self, model_path: Path) -> None:
        """
        Load compiled Hailo model (.hef file).

        Args:
            model_path: Path to .hef model file
        """
        try:
            from hailo_platform import VDevice, HailoStreamInterface
            from hailo_platform import InputVStreamParams, OutputVStreamParams

            self._vdevice = VDevice()
            self._hef = self._vdevice.create_hef(str(model_path))
            self._network_group = self._vdevice.configure(self._hef)[0]

            # Get stream info
            self._input_info = self._hef.get_input_vstream_infos()[0]
            self._output_info = self._hef.get_output_vstream_infos()[0]

        except ImportError:
            raise RuntimeError(
                "Hailo platform not installed. "
                "Install with: pip install hailo_platform"
            )

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on Hailo-8L NPU."""
        if self._network_group is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        input_data = self._preprocess(frame)

        # Run inference
        with self._network_group.activate():
            output = self._network_group.infer({
                self._input_info.name: input_data
            })

        # Postprocess
        return self._postprocess(output, frame.shape)

    def track(
        self,
        frame: np.ndarray,
        persist: bool = True
    ) -> List[Detection]:
        """
        Track with ByteTrack on Hailo detections.

        Note: Tracking is implemented on CPU side.
        """
        detections = self.infer(frame)
        # TODO: Implement ByteTrack integration
        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo input."""
        # Resize to model input size
        resized = cv2.resize(frame, self._input_shape)
        # Normalize to 0-1 range
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)

    def _postprocess(
        self,
        output: dict,
        original_shape: tuple
    ) -> List[Detection]:
        """Postprocess Hailo output to detections."""
        # TODO: Implement NMS and detection parsing
        # This depends on the specific model output format
        return []


# =============================================================================
# Backend Factory
# =============================================================================

def get_inference_backend(backend_type: str = "pytorch") -> InferenceBackend:
    """
    Factory function to create inference backends.

    Args:
        backend_type: Backend type ("pytorch", "hailo")

    Returns:
        Configured InferenceBackend instance
    """
    backends = {
        "pytorch": PyTorchBackend,
        "hailo": HailoBackend,
    }

    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. "
            f"Available: {list(backends.keys())}"
        )

    return backends[backend_type]()


# =============================================================================
# Camera View (Refactored from perception.py)
# =============================================================================

class CameraView:
    """
    Camera perception module with YOLO detection and tracking.

    Handles:
    - Model inference with tracking
    - Pen boundary filtering
    - Coordinate transformation (camera to world)
    - Frame annotation

    Refactored from perception.py.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        pen_polygon: np.ndarray,
        homography_matrix: Optional[np.ndarray] = None,
        backend: str = "pytorch",
        tracker_config: str = "bytetrack.yaml"
    ):
        """
        Initialize camera view.

        Args:
            name: Camera identifier
            model_path: Path to YOLO model
            pen_polygon: 4-point polygon defining pen boundaries
            homography_matrix: Optional 3x3 perspective transform matrix
            backend: Inference backend ("pytorch" or "hailo")
            tracker_config: Path to tracker configuration
        """
        self.name = name
        self.pen_polygon = pen_polygon.astype(np.int32)
        self.homography = homography_matrix

        # Initialize backend
        if backend == "pytorch":
            self._backend = PyTorchBackend(tracker_config)
        else:
            self._backend = get_inference_backend(backend)

        self._backend.load_model(Path(model_path))
        self._frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        transform_coords: bool = True,
        annotate: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process a frame with detection, tracking, and transformation.

        Args:
            frame: BGR image as numpy array
            transform_coords: Apply homography transformation
            annotate: Draw annotations on returned frame

        Returns:
            Tuple of (annotated_frame, detections_list)
            Detection dict format:
            {
                'id': track_id,
                'type': class_name,
                'pos': (x_world, y_world),  # World coords if transformed
                'radius': radius_cm,
                'box': (x1, y1, x2, y2)  # Camera coords
            }
        """
        start_time = time.time()

        # Run tracking
        raw_detections = self._backend.track(frame, persist=True)

        # Filter and transform detections
        detections = []
        for det in raw_detections:
            if det.track_id is None:
                continue

            # Get center in camera coordinates
            center_cam = det.get_center()

            # Filter by pen boundary
            if not is_point_in_polygon(center_cam, self.pen_polygon):
                continue

            # Transform to world coordinates
            if transform_coords and self.homography is not None:
                center_world = transform_point(center_cam, self.homography)
            else:
                center_world = center_cam

            # Estimate radius from box width
            radius_pixels = det.get_width() / 2
            radius_cm = radius_pixels * 0.2  # Configurable scale factor

            detections.append({
                'id': det.track_id,
                'type': det.class_name,
                'pos': center_world,
                'radius': radius_cm,
                'box': det.box,
                'confidence': det.confidence
            })

        # Annotate frame
        if annotate:
            annotated = self._annotate_frame(frame, detections)
        else:
            annotated = frame

        self._frame_count += 1
        inference_time = (time.time() - start_time) * 1000

        return annotated, detections

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[dict]
    ) -> np.ndarray:
        """Draw annotations on frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            color = COLORS.get(det['type'], (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{det['type']} {det['id']}"
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - 20), (x1 + w, y1), color, -1
            )
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Draw pen boundary
        cv2.polylines(
            annotated, [self.pen_polygon], True,
            COLORS['pen_boundary'], 2
        )

        return annotated

    @property
    def class_names(self) -> Dict[int, str]:
        """Get class names from backend."""
        return self._backend.class_names
