"""
Integration tests for frame synchronization.
"""

import pytest
import numpy as np
import time

from src.core.synchronization import (
    SyncConfig,
    RingBuffer,
    FrameSynchronizer,
    SyncedFramePair,
)
from src.core.ingestion import TimestampedFrame


class TestRingBuffer:
    """Tests for RingBuffer class."""

    def test_buffer_maxlen(self):
        """Test that buffer respects maxlen."""
        buffer = RingBuffer(maxlen=5)

        for i in range(10):
            frame = TimestampedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33.0,
                camera_name="test",
                frame_id=i
            )
            buffer.append(frame)

        assert len(buffer) == 5

    def test_get_closest_within_tolerance(self):
        """Test finding closest frame within tolerance."""
        buffer = RingBuffer(maxlen=10)

        for i in range(10):
            frame = TimestampedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33.0,
                camera_name="test",
                frame_id=i
            )
            buffer.append(frame)

        # Find frame closest to 165ms (should be frame 5: 165ms)
        result = buffer.get_closest(165.0, tolerance_ms=50.0)
        assert result is not None
        assert result.frame_id == 5

    def test_get_closest_outside_tolerance(self):
        """Test that no frame returned when outside tolerance."""
        buffer = RingBuffer(maxlen=10)

        for i in range(5):
            frame = TimestampedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33.0,
                camera_name="test",
                frame_id=i
            )
            buffer.append(frame)

        # Look for frame at 1000ms (way outside range)
        result = buffer.get_closest(1000.0, tolerance_ms=50.0)
        assert result is None

    def test_get_latest(self):
        """Test getting latest frame."""
        buffer = RingBuffer(maxlen=10)

        for i in range(5):
            frame = TimestampedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33.0,
                camera_name="test",
                frame_id=i
            )
            buffer.append(frame)

        latest = buffer.get_latest()
        assert latest is not None
        assert latest.frame_id == 4

    def test_empty_buffer_returns_none(self):
        """Test that empty buffer returns None."""
        buffer = RingBuffer(maxlen=10)
        assert buffer.get_latest() is None
        assert buffer.get_closest(100.0, 50.0) is None


class TestFrameSynchronizer:
    """Tests for FrameSynchronizer class."""

    def test_sync_within_tolerance(self):
        """Test synchronization when frames are within tolerance."""
        config = SyncConfig(tolerance_ms=50.0, primary_camera="top")
        sync = FrameSynchronizer(config)

        # Add primary (top) frame
        top_frame = TimestampedFrame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=1000.0,
            camera_name="top",
            frame_id=1
        )
        sync.add_frame(top_frame)

        # Add secondary (side) frame within tolerance
        side_frame = TimestampedFrame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=1030.0,  # 30ms difference
            camera_name="side",
            frame_id=1
        )
        sync.add_frame(side_frame)

        pair = sync.get_synced_pair()
        assert pair is not None
        assert pair.timestamp_diff_ms == 30.0
        assert pair.primary_frame.camera_name == "top"
        assert pair.secondary_frame.camera_name == "side"

    def test_sync_outside_tolerance(self):
        """Test that frames outside tolerance are not synced."""
        config = SyncConfig(tolerance_ms=50.0, primary_camera="top")
        sync = FrameSynchronizer(config)

        top_frame = TimestampedFrame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=1000.0,
            camera_name="top",
            frame_id=1
        )
        sync.add_frame(top_frame)

        # Side frame outside tolerance
        side_frame = TimestampedFrame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=1100.0,  # 100ms difference
            camera_name="side",
            frame_id=1
        )
        sync.add_frame(side_frame)

        pair = sync.get_synced_pair()
        assert pair is None

    def test_sync_stats(self):
        """Test synchronization statistics tracking."""
        config = SyncConfig(tolerance_ms=50.0, primary_camera="top")
        sync = FrameSynchronizer(config)

        # Add synced pair
        sync.add_frame(TimestampedFrame(
            frame=np.zeros((480, 640, 3)),
            timestamp_ms=1000.0, camera_name="top", frame_id=1
        ))
        sync.add_frame(TimestampedFrame(
            frame=np.zeros((480, 640, 3)),
            timestamp_ms=1020.0, camera_name="side", frame_id=1
        ))
        sync.get_synced_pair()

        # Add unsynced pair (outside tolerance)
        sync.add_frame(TimestampedFrame(
            frame=np.zeros((480, 640, 3)),
            timestamp_ms=2000.0, camera_name="top", frame_id=2
        ))
        sync.get_synced_pair()

        stats = sync.get_sync_stats()
        assert stats["frames_synced"] == 1
        assert stats["frames_dropped"] == 1
        assert stats["sync_rate"] == 0.5

    def test_sync_rate_calculation(self):
        """Test sync rate is calculated correctly."""
        config = SyncConfig(tolerance_ms=50.0, primary_camera="top")
        sync = FrameSynchronizer(config)

        # Add 10 frame pairs, all synced
        for i in range(10):
            sync.add_frame(TimestampedFrame(
                frame=np.zeros((480, 640, 3)),
                timestamp_ms=i * 100.0, camera_name="top", frame_id=i
            ))
            sync.add_frame(TimestampedFrame(
                frame=np.zeros((480, 640, 3)),
                timestamp_ms=i * 100.0 + 20.0, camera_name="side", frame_id=i
            ))
            sync.get_synced_pair()

        stats = sync.get_sync_stats()
        assert stats["sync_rate"] == 1.0
        assert stats["frames_synced"] == 10

    def test_frame_aliases(self):
        """Test that top_frame and side_frame aliases work."""
        config = SyncConfig(tolerance_ms=50.0, primary_camera="top")
        sync = FrameSynchronizer(config)

        sync.add_frame(TimestampedFrame(
            frame=np.zeros((480, 640, 3)),
            timestamp_ms=1000.0, camera_name="top", frame_id=1
        ))
        sync.add_frame(TimestampedFrame(
            frame=np.zeros((480, 640, 3)),
            timestamp_ms=1020.0, camera_name="side", frame_id=1
        ))

        pair = sync.get_synced_pair()
        assert pair.top_frame == pair.primary_frame
        assert pair.side_frame == pair.secondary_frame
