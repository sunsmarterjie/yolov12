"""
Frame synchronization for dual-camera setup.

This module provides:
- Ring buffer for timestamped frames
- Frame synchronizer for matching frames across cameras
- Sync statistics tracking

Designed for USB + RTSP dual camera setup where RTSP
has higher latency than USB.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque
import numpy as np

from .ingestion import TimestampedFrame


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SyncConfig:
    """
    Synchronization configuration.

    Attributes:
        tolerance_ms: Maximum timestamp difference for frame matching
        buffer_size: Ring buffer size for each camera
        primary_camera: Name of the primary (faster) camera
    """
    tolerance_ms: float = 50.0
    buffer_size: int = 30
    primary_camera: str = "top"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SyncedFramePair:
    """
    Synchronized frame pair from dual cameras.

    Attributes:
        primary_frame: Frame from primary (faster) camera
        secondary_frame: Frame from secondary (slower) camera
        timestamp_diff_ms: Time difference between frames
        sync_id: Sequential sync pair identifier
    """
    primary_frame: TimestampedFrame
    secondary_frame: TimestampedFrame
    timestamp_diff_ms: float
    sync_id: int

    @property
    def top_frame(self) -> TimestampedFrame:
        """Alias for primary frame (top camera)."""
        return self.primary_frame

    @property
    def side_frame(self) -> TimestampedFrame:
        """Alias for secondary frame (side camera)."""
        return self.secondary_frame


# =============================================================================
# Ring Buffer
# =============================================================================

class RingBuffer:
    """
    Thread-safe ring buffer for timestamped frames.

    Provides efficient storage and timestamp-based retrieval.
    """

    def __init__(self, maxlen: int = 30):
        """
        Initialize ring buffer.

        Args:
            maxlen: Maximum buffer size
        """
        self._buffer: deque[TimestampedFrame] = deque(maxlen=maxlen)

    def append(self, frame: TimestampedFrame) -> None:
        """
        Add frame to buffer.

        Oldest frame is automatically removed if buffer is full.

        Args:
            frame: Timestamped frame to add
        """
        self._buffer.append(frame)

    def get_closest(
        self,
        target_timestamp_ms: float,
        tolerance_ms: float
    ) -> Optional[TimestampedFrame]:
        """
        Find frame closest to target timestamp within tolerance.

        Args:
            target_timestamp_ms: Target timestamp in milliseconds
            tolerance_ms: Maximum acceptable time difference

        Returns:
            Closest matching frame or None if no match within tolerance
        """
        if not self._buffer:
            return None

        best_frame = None
        best_diff = float('inf')

        for frame in self._buffer:
            diff = abs(frame.timestamp_ms - target_timestamp_ms)
            if diff < best_diff and diff <= tolerance_ms:
                best_diff = diff
                best_frame = frame

        return best_frame

    def get_frames_in_range(
        self,
        start_ms: float,
        end_ms: float
    ) -> List[TimestampedFrame]:
        """
        Get all frames within a timestamp range.

        Args:
            start_ms: Range start timestamp
            end_ms: Range end timestamp

        Returns:
            List of frames within range
        """
        return [
            f for f in self._buffer
            if start_ms <= f.timestamp_ms <= end_ms
        ]

    def get_latest(self) -> Optional[TimestampedFrame]:
        """Get most recent frame."""
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Get number of frames in buffer."""
        return len(self._buffer)


# =============================================================================
# Frame Synchronizer
# =============================================================================

class FrameSynchronizer:
    """
    Synchronizes frames from dual cameras using ring buffers.

    Design:
    - Primary camera (faster, e.g., USB) drives the sync
    - Secondary camera (slower, e.g., RTSP) frames are matched
    - Frames outside tolerance are dropped

    Usage:
    ```python
    sync = FrameSynchronizer(SyncConfig(tolerance_ms=50))

    # Add frames from cameras
    sync.add_frame(usb_frame)
    sync.add_frame(rtsp_frame)

    # Get synchronized pair
    pair = sync.get_synced_pair()
    if pair:
        process_pair(pair.primary_frame, pair.secondary_frame)
    ```
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """
        Initialize frame synchronizer.

        Args:
            config: Synchronization configuration
        """
        self.config = config or SyncConfig()

        # Separate buffers for each camera
        self._primary_buffer = RingBuffer(self.config.buffer_size)
        self._secondary_buffer = RingBuffer(self.config.buffer_size)

        self._sync_count = 0

        # Statistics
        self.frames_synced = 0
        self.frames_dropped = 0
        self._last_sync_diff_ms = 0.0

    def add_frame(self, frame: TimestampedFrame) -> None:
        """
        Add frame to appropriate buffer based on camera name.

        Args:
            frame: Timestamped frame from either camera
        """
        if frame.camera_name == self.config.primary_camera:
            self._primary_buffer.append(frame)
        else:
            self._secondary_buffer.append(frame)

    def add_primary_frame(self, frame: TimestampedFrame) -> None:
        """Add frame to primary buffer."""
        self._primary_buffer.append(frame)

    def add_secondary_frame(self, frame: TimestampedFrame) -> None:
        """Add frame to secondary buffer."""
        self._secondary_buffer.append(frame)

    def get_synced_pair(self) -> Optional[SyncedFramePair]:
        """
        Attempt to get a synchronized frame pair.

        Uses primary camera timestamp to find matching secondary frame.

        Returns:
            SyncedFramePair if match found, None otherwise
        """
        # Get latest primary frame
        primary_frame = self._primary_buffer.get_latest()
        if primary_frame is None:
            return None

        # Find matching secondary frame
        secondary_frame = self._secondary_buffer.get_closest(
            primary_frame.timestamp_ms,
            self.config.tolerance_ms
        )

        if secondary_frame is None:
            self.frames_dropped += 1
            return None

        # Calculate timestamp difference
        timestamp_diff = abs(
            primary_frame.timestamp_ms - secondary_frame.timestamp_ms
        )

        self._sync_count += 1
        self.frames_synced += 1
        self._last_sync_diff_ms = timestamp_diff

        return SyncedFramePair(
            primary_frame=primary_frame,
            secondary_frame=secondary_frame,
            timestamp_diff_ms=timestamp_diff,
            sync_id=self._sync_count
        )

    def try_sync(
        self,
        primary_frame: TimestampedFrame
    ) -> Optional[SyncedFramePair]:
        """
        Try to synchronize a specific primary frame.

        Useful for on-demand synchronization.

        Args:
            primary_frame: Frame to find match for

        Returns:
            SyncedFramePair if match found, None otherwise
        """
        # Add to buffer
        self._primary_buffer.append(primary_frame)

        # Find match
        secondary_frame = self._secondary_buffer.get_closest(
            primary_frame.timestamp_ms,
            self.config.tolerance_ms
        )

        if secondary_frame is None:
            self.frames_dropped += 1
            return None

        timestamp_diff = abs(
            primary_frame.timestamp_ms - secondary_frame.timestamp_ms
        )

        self._sync_count += 1
        self.frames_synced += 1
        self._last_sync_diff_ms = timestamp_diff

        return SyncedFramePair(
            primary_frame=primary_frame,
            secondary_frame=secondary_frame,
            timestamp_diff_ms=timestamp_diff,
            sync_id=self._sync_count
        )

    def get_sync_stats(self) -> dict:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with sync metrics
        """
        total = self.frames_synced + self.frames_dropped
        sync_rate = self.frames_synced / total if total > 0 else 0.0

        return {
            "frames_synced": self.frames_synced,
            "frames_dropped": self.frames_dropped,
            "sync_rate": sync_rate,
            "last_diff_ms": self._last_sync_diff_ms,
            "primary_buffer_size": len(self._primary_buffer),
            "secondary_buffer_size": len(self._secondary_buffer),
            "tolerance_ms": self.config.tolerance_ms
        }

    def reset_stats(self) -> None:
        """Reset synchronization statistics."""
        self.frames_synced = 0
        self.frames_dropped = 0
        self._last_sync_diff_ms = 0.0

    def clear_buffers(self) -> None:
        """Clear all frame buffers."""
        self._primary_buffer.clear()
        self._secondary_buffer.clear()


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_latency(
    frames: List[TimestampedFrame],
    reference_time_ms: float
) -> float:
    """
    Estimate average latency from a series of frames.

    Args:
        frames: List of timestamped frames
        reference_time_ms: Reference timestamp

    Returns:
        Average latency in milliseconds
    """
    if not frames:
        return 0.0

    latencies = [
        reference_time_ms - f.timestamp_ms
        for f in frames
    ]
    return sum(latencies) / len(latencies)
