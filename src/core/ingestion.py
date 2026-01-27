"""
Multi-threaded camera ingestion with ring buffer support.

This module provides:
- Camera configuration dataclasses
- USB camera capture
- RTSP stream capture with reconnection
- Threaded capture with output buffers
- Camera manager for multi-camera setups

Designed for Raspberry Pi 5 with USB + RTSP camera setup.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
from threading import Thread, Event
from queue import Queue, Empty
import time
import os
import cv2
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CameraConfig:
    """
    Camera configuration.

    Attributes:
        name: Unique camera identifier
        source: Device index (int), RTSP URL, or file path
        camera_type: "usb", "rtsp", or "file"
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Target frames per second
        rotation: Rotation angle (0, 90, 180, 270)
        buffer_size: Ring buffer size for frame storage
    """
    name: str
    source: str
    camera_type: str = "usb"  # "usb", "rtsp", "file"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    rotation: int = 0
    buffer_size: int = 30


@dataclass
class TimestampedFrame:
    """
    Frame with capture metadata.

    Attributes:
        frame: BGR image as numpy array
        timestamp_ms: Capture timestamp in milliseconds
        camera_name: Source camera identifier
        frame_id: Sequential frame number
    """
    frame: np.ndarray
    timestamp_ms: float
    camera_name: str
    frame_id: int


# =============================================================================
# Camera Sources
# =============================================================================

class CameraSource:
    """Base class for camera sources."""

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        raise NotImplementedError

    def release(self) -> None:
        """Release the camera resource."""
        raise NotImplementedError

    def is_opened(self) -> bool:
        """Check if source is active."""
        raise NotImplementedError


class USBCamera(CameraSource):
    """
    USB camera capture.

    Handles built-in and USB webcams.
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize USB camera.

        Args:
            config: Camera configuration
        """
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._setup_capture()

    def _setup_capture(self) -> None:
        """Initialize OpenCV capture."""
        source = int(self.config.source) if self.config.source.isdigit() else 0
        self._cap = cv2.VideoCapture(source)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from USB camera."""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        """Release USB camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._cap is not None and self._cap.isOpened()


class RTSPCamera(CameraSource):
    """
    RTSP camera capture with reconnection support.

    Handles network cameras and RTSP streams.
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize RTSP camera.

        Args:
            config: Camera configuration with RTSP URL as source
        """
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 30.0
        self._setup_capture()

    def _setup_capture(self) -> None:
        """Initialize RTSP capture with optimized settings."""
        # Set FFmpeg options for low latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp;fflags;nobuffer;flags;low_delay"
        )

        self._cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from RTSP stream."""
        if self._cap is None:
            return False, None

        success, frame = self._cap.read()

        if not success:
            # Attempt reconnection
            self._reconnect()
            return False, None

        return True, frame

    def _reconnect(self) -> None:
        """Attempt to reconnect to RTSP stream."""
        print(f"[{self.config.name}] Reconnecting to RTSP stream...")

        if self._cap is not None:
            self._cap.release()

        time.sleep(self._reconnect_delay)
        self._setup_capture()

        # Exponential backoff
        self._reconnect_delay = min(
            self._reconnect_delay * 1.5,
            self._max_reconnect_delay
        )

    def release(self) -> None:
        """Release RTSP capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check if RTSP stream is connected."""
        return self._cap is not None and self._cap.isOpened()


class FileCamera(CameraSource):
    """
    Video file playback as camera source.

    Useful for testing and offline processing.
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize file camera.

        Args:
            config: Camera configuration with file path as source
        """
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._loop = True
        self._setup_capture()

    def _setup_capture(self) -> None:
        """Initialize video file capture."""
        self._cap = cv2.VideoCapture(self.config.source)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from video file."""
        if self._cap is None:
            return False, None

        success, frame = self._cap.read()

        if not success and self._loop:
            # Loop back to start
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._cap.read()

        return success, frame

    def release(self) -> None:
        """Release video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self._cap is not None and self._cap.isOpened()


# =============================================================================
# Threaded Camera Capture
# =============================================================================

class CameraThread(Thread):
    """
    Threaded camera capture with output queue.

    Runs capture loop in background thread, pushing
    timestamped frames to a queue for consumption.
    """

    def __init__(
        self,
        config: CameraConfig,
        output_queue: Queue,
        on_frame_callback: Optional[Callable[[TimestampedFrame], None]] = None
    ):
        """
        Initialize camera thread.

        Args:
            config: Camera configuration
            output_queue: Queue to push captured frames
            on_frame_callback: Optional callback for each frame
        """
        super().__init__(daemon=True)
        self.config = config
        self.output_queue = output_queue
        self.on_frame_callback = on_frame_callback

        self._stop_event = Event()
        self._frame_count = 0
        self._camera: Optional[CameraSource] = None

        # Initialize camera based on type
        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize appropriate camera source."""
        if self.config.camera_type == "usb":
            self._camera = USBCamera(self.config)
        elif self.config.camera_type == "rtsp":
            self._camera = RTSPCamera(self.config)
        elif self.config.camera_type == "file":
            self._camera = FileCamera(self.config)
        else:
            raise ValueError(f"Unknown camera type: {self.config.camera_type}")

    def run(self) -> None:
        """Main capture loop."""
        while not self._stop_event.is_set():
            if self._camera is None or not self._camera.is_opened():
                time.sleep(0.1)
                continue

            success, frame = self._camera.read()

            if not success or frame is None:
                continue

            # Get capture timestamp
            timestamp_ms = time.time() * 1000

            # Apply rotation if configured
            if self.config.rotation != 0:
                frame = self._rotate(frame, self.config.rotation)

            # Create timestamped frame
            ts_frame = TimestampedFrame(
                frame=frame,
                timestamp_ms=timestamp_ms,
                camera_name=self.config.name,
                frame_id=self._frame_count
            )

            # Manage buffer - drop oldest if full
            if self.output_queue.qsize() >= self.config.buffer_size:
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass

            # Push to queue
            self.output_queue.put(ts_frame)
            self._frame_count += 1

            # Optional callback
            if self.on_frame_callback is not None:
                try:
                    self.on_frame_callback(ts_frame)
                except Exception as e:
                    print(f"Frame callback error: {e}")

    def _rotate(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Rotate frame by specified angle."""
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def stop(self) -> None:
        """Stop capture thread."""
        self._stop_event.set()
        if self._camera is not None:
            self._camera.release()

    def is_running(self) -> bool:
        """Check if thread is running."""
        return not self._stop_event.is_set()


# =============================================================================
# Camera Manager
# =============================================================================

class CameraManager:
    """
    Manages multiple camera threads.

    Provides unified interface for multi-camera setups.
    """

    def __init__(self, configs: Optional[list[CameraConfig]] = None):
        """
        Initialize camera manager.

        Args:
            configs: Optional list of camera configurations
        """
        self.configs: list[CameraConfig] = configs or []
        self.queues: Dict[str, Queue] = {}
        self.threads: Dict[str, CameraThread] = {}
        self._running = False

    def add_camera(self, config: CameraConfig) -> None:
        """Add a camera configuration."""
        self.configs.append(config)

    def start_all(self) -> None:
        """Start all camera threads."""
        for config in self.configs:
            queue = Queue(maxsize=config.buffer_size)
            thread = CameraThread(config, queue)

            self.queues[config.name] = queue
            self.threads[config.name] = thread
            thread.start()

        self._running = True
        print(f"[CameraManager] Started {len(self.threads)} camera(s)")

    def stop_all(self) -> None:
        """Stop all camera threads."""
        for name, thread in self.threads.items():
            thread.stop()
            print(f"[CameraManager] Stopped camera: {name}")

        self._running = False

    def get_frame(
        self,
        camera_name: str,
        timeout: float = 0.1
    ) -> Optional[TimestampedFrame]:
        """
        Get the next frame from a camera.

        Args:
            camera_name: Camera identifier
            timeout: Wait timeout in seconds

        Returns:
            TimestampedFrame or None if unavailable
        """
        queue = self.queues.get(camera_name)
        if queue is None:
            return None

        try:
            return queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_frame(self, camera_name: str) -> Optional[TimestampedFrame]:
        """
        Get the most recent frame from a camera (drops older frames).

        Args:
            camera_name: Camera identifier

        Returns:
            Most recent TimestampedFrame or None
        """
        queue = self.queues.get(camera_name)
        if queue is None:
            return None

        latest = None
        while not queue.empty():
            try:
                latest = queue.get_nowait()
            except Empty:
                break

        return latest

    def get_all_latest(self) -> Dict[str, Optional[TimestampedFrame]]:
        """Get latest frames from all cameras."""
        return {
            name: self.get_latest_frame(name)
            for name in self.queues.keys()
        }

    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all cameras."""
        stats = {}
        for name, thread in self.threads.items():
            stats[name] = {
                "running": thread.is_running(),
                "frame_count": thread._frame_count,
                "queue_size": self.queues[name].qsize()
            }
        return stats
