"""
Poultry Vision Core Module.

This package provides the runtime engine for poultry monitoring:
- Camera ingestion and frame synchronization
- YOLO-based inference with tracking
- Multi-view fusion and coordinate transformation
- Behavioral state machine and monitoring

Usage:
    from src.core import (
        CameraManager, CameraConfig,
        FrameSynchronizer,
        CameraView, PyTorchBackend,
        MultiViewFusion, HomographyTransform,
        PenStateMachine, BehaviorState,
        HenBehaviorMonitor
    )
"""

# Geometry utilities
from .geometry import (
    COLORS,
    get_box_center,
    get_box_center_int,
    get_box_width,
    get_box_height,
    get_box_area,
    check_overlap,
    calculate_overlap_area,
    calculate_iou,
    get_homography_matrix,
    transform_point,
    transform_point_int,
    is_point_in_circle,
    is_point_in_polygon,
    euclidean_distance,
    pixels_to_cm,
    cm_to_pixels,
    HomographyConfig,
)

# Camera ingestion
from .ingestion import (
    CameraConfig,
    TimestampedFrame,
    USBCamera,
    RTSPCamera,
    FileCamera,
    CameraThread,
    CameraManager,
)

# Frame synchronization
from .synchronization import (
    SyncConfig,
    SyncedFramePair,
    RingBuffer,
    FrameSynchronizer,
)

# Inference
from .inference import (
    Detection,
    InferenceResult,
    InferenceBackend,
    PyTorchBackend,
    HailoBackend,
    get_inference_backend,
    CameraView,
)

# Multi-view fusion
from .fusion import (
    FusedDetection,
    HomographyTransform,
    MultiViewFusion,
    create_fusion_pipeline,
    load_homography,
)

# State machine
from .state_machine import (
    BehaviorState,
    EntityType,
    Entity,
    PenStateMachine,
    PenState,  # Legacy compatibility
)

# Behavior monitoring
from .behavior import (
    BehaviorType,
    HenStats,
    HenBehaviorMonitor,
)

__all__ = [
    # Geometry
    'COLORS',
    'get_box_center',
    'get_box_center_int',
    'get_box_width',
    'get_box_height',
    'get_box_area',
    'check_overlap',
    'calculate_overlap_area',
    'calculate_iou',
    'get_homography_matrix',
    'transform_point',
    'transform_point_int',
    'is_point_in_circle',
    'is_point_in_polygon',
    'euclidean_distance',
    'pixels_to_cm',
    'cm_to_pixels',
    'HomographyConfig',
    # Ingestion
    'CameraConfig',
    'TimestampedFrame',
    'USBCamera',
    'RTSPCamera',
    'FileCamera',
    'CameraThread',
    'CameraManager',
    # Synchronization
    'SyncConfig',
    'SyncedFramePair',
    'RingBuffer',
    'FrameSynchronizer',
    # Inference
    'Detection',
    'InferenceResult',
    'InferenceBackend',
    'PyTorchBackend',
    'HailoBackend',
    'get_inference_backend',
    'CameraView',
    # Fusion
    'FusedDetection',
    'HomographyTransform',
    'MultiViewFusion',
    'create_fusion_pipeline',
    'load_homography',
    # State Machine
    'BehaviorState',
    'EntityType',
    'Entity',
    'PenStateMachine',
    'PenState',
    # Behavior
    'BehaviorType',
    'HenStats',
    'HenBehaviorMonitor',
]
