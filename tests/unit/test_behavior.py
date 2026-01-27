"""
Unit tests for behavior monitoring module.
"""

import pytest

from src.core.behavior import (
    BehaviorType,
    HenStats,
    HenBehaviorMonitor,
)


class TestHenStats:
    """Tests for HenStats class."""

    def test_stats_creation(self):
        """Test stats creation with defaults."""
        stats = HenStats(hen_id=1)
        assert stats.hen_id == 1
        assert stats.feeding_time_s == 0.0
        assert stats.drinking_time_s == 0.0
        assert stats.current_behavior == BehaviorType.IDLE
        assert stats.feeding_events == 0
        assert stats.drinking_events == 0

    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = HenStats(hen_id=1)
        stats.feeding_time_s = 10.555
        stats.drinking_time_s = 5.123

        d = stats.to_dict()
        assert d["hen_id"] == 1
        assert d["feeding_time_s"] == 10.56  # Rounded
        assert d["drinking_time_s"] == 5.12  # Rounded
        assert d["current_behavior"] == "idle"


class TestHenBehaviorMonitor:
    """Tests for HenBehaviorMonitor class."""

    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = HenBehaviorMonitor(fps=30.0)
        assert monitor.fps == 30.0
        assert monitor._frame_count == 0
        assert len(monitor.tracked_hen_ids) == 0

    def test_update_creates_stats(self):
        """Test that update creates stats for new hens."""
        monitor = HenBehaviorMonitor(fps=30.0)

        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = []
        waterers = []

        monitor.update(hens, feeders, waterers)

        assert 1 in monitor.tracked_hen_ids
        stats = monitor.get_stats(1)
        assert stats["current_behavior"] == "idle"

    def test_feeding_detection(self):
        """Test feeding behavior detection via overlap."""
        monitor = HenBehaviorMonitor(fps=30.0)

        # Hen box overlaps with feeder box
        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = [{"box": (150, 150, 250, 250)}]  # Overlaps
        waterers = []

        monitor.update(hens, feeders, waterers)

        stats = monitor.get_stats(1)
        assert stats["current_behavior"] == "feeding"
        assert stats["feeding_time_s"] > 0

    def test_drinking_detection(self):
        """Test drinking behavior detection via overlap."""
        monitor = HenBehaviorMonitor(fps=30.0)

        # Hen box overlaps with waterer box
        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = []
        waterers = [{"box": (150, 150, 250, 250)}]  # Overlaps

        monitor.update(hens, feeders, waterers)

        stats = monitor.get_stats(1)
        assert stats["current_behavior"] == "drinking"
        assert stats["drinking_time_s"] > 0

    def test_feeding_priority_over_drinking(self):
        """Test that feeding has priority when both overlap."""
        monitor = HenBehaviorMonitor(fps=30.0)

        # Hen overlaps with both feeder and waterer
        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = [{"box": (150, 150, 250, 250)}]
        waterers = [{"box": (150, 150, 250, 250)}]

        monitor.update(hens, feeders, waterers)

        stats = monitor.get_stats(1)
        assert stats["current_behavior"] == "feeding"

    def test_no_overlap_stays_idle(self):
        """Test that no overlap results in idle state."""
        monitor = HenBehaviorMonitor(fps=30.0)

        # Hen doesn't overlap with feeder or waterer
        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = [{"box": (400, 400, 500, 500)}]
        waterers = [{"box": (600, 600, 700, 700)}]

        monitor.update(hens, feeders, waterers)

        stats = monitor.get_stats(1)
        assert stats["current_behavior"] == "idle"

    def test_time_accumulation(self):
        """Test that feeding/drinking time accumulates."""
        monitor = HenBehaviorMonitor(fps=30.0)

        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = [{"box": (150, 150, 250, 250)}]

        # Simulate 30 frames of feeding
        for _ in range(30):
            monitor.update(hens, feeders, [])

        stats = monitor.get_stats(1)
        # 30 frames at 30fps = 1 second
        assert abs(stats["feeding_time_s"] - 1.0) < 0.01

    def test_behavior_change_callback(self):
        """Test behavior change callback is fired."""
        monitor = HenBehaviorMonitor(fps=30.0)

        events = []

        def callback(hen_id, old_behavior, new_behavior, duration):
            events.append((hen_id, old_behavior, new_behavior))

        monitor.register_callback(callback)

        # Start idle
        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        monitor.update(hens, [], [])

        # Start feeding
        feeders = [{"box": (150, 150, 250, 250)}]
        monitor.update(hens, feeders, [])

        assert len(events) == 1
        assert events[0][0] == 1
        assert events[0][1] == BehaviorType.IDLE
        assert events[0][2] == BehaviorType.FEEDING

    def test_feeding_events_count(self):
        """Test that feeding events are counted."""
        monitor = HenBehaviorMonitor(fps=30.0)

        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        feeders = [{"box": (150, 150, 250, 250)}]

        # Idle -> Feeding (event 1)
        monitor.update(hens, [], [])
        monitor.update(hens, feeders, [])

        # Feeding -> Idle -> Feeding (event 2)
        monitor.update(hens, [], [])
        monitor.update(hens, feeders, [])

        stats = monitor.get_stats(1)
        assert stats["feeding_events"] == 2

    def test_get_summary(self):
        """Test getting summary statistics."""
        monitor = HenBehaviorMonitor(fps=30.0)

        # Add two hens
        hens = [
            {"id": 1, "box": (100, 100, 200, 200)},
            {"id": 2, "box": (300, 300, 400, 400)}
        ]
        feeders = [{"box": (150, 150, 250, 250)}]  # Only hen 1 overlaps

        # Run a few frames
        for _ in range(30):
            monitor.update(hens, feeders, [])

        summary = monitor.get_summary()
        assert summary["total_hens"] == 2
        assert summary["total_feeding_time_s"] > 0
        assert summary["frames_processed"] == 30

    def test_reset(self):
        """Test resetting all statistics."""
        monitor = HenBehaviorMonitor(fps=30.0)

        hens = [{"id": 1, "box": (100, 100, 200, 200)}]
        monitor.update(hens, [], [])

        assert len(monitor.tracked_hen_ids) == 1

        monitor.reset()

        assert len(monitor.tracked_hen_ids) == 0
        assert monitor._frame_count == 0

    def test_get_current_behaviors(self):
        """Test getting current behaviors for all hens."""
        monitor = HenBehaviorMonitor(fps=30.0)

        hens = [
            {"id": 1, "box": (100, 100, 200, 200)},
            {"id": 2, "box": (300, 300, 400, 400)}
        ]
        feeders = [{"box": (150, 150, 250, 250)}]

        monitor.update(hens, feeders, [])

        behaviors = monitor.get_current_behaviors()
        assert behaviors[1] == "feeding"
        assert behaviors[2] == "idle"
