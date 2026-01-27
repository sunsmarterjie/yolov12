"""
Temporal behavior tracking for hens.

This module provides:
- Per-hen behavior statistics accumulation
- Bounding box overlap-based behavior detection
- Event callbacks for behavior changes
- Statistics export

Refactored from behavior_monitor.py with event system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from enum import Enum
import time

from .geometry import check_overlap, get_box_center


# =============================================================================
# Types and Enums
# =============================================================================

class BehaviorType(Enum):
    """Types of detectable behaviors."""
    IDLE = "idle"
    FEEDING = "feeding"
    DRINKING = "drinking"


# Callback type: (hen_id, old_behavior, new_behavior, duration_s) -> None
BehaviorChangeCallback = Callable[[int, BehaviorType, BehaviorType, float], None]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HenStats:
    """
    Statistics for a single hen.

    Attributes:
        hen_id: Tracking identifier
        feeding_time_s: Cumulative feeding time in seconds
        drinking_time_s: Cumulative drinking time in seconds
        current_behavior: Current behavioral state
        behavior_start_time: When current behavior started
        feeding_events: Number of feeding events
        drinking_events: Number of drinking events
    """
    hen_id: int
    feeding_time_s: float = 0.0
    drinking_time_s: float = 0.0
    current_behavior: BehaviorType = BehaviorType.IDLE
    behavior_start_time: float = field(default_factory=time.time)
    feeding_events: int = 0
    drinking_events: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hen_id": self.hen_id,
            "feeding_time_s": round(self.feeding_time_s, 2),
            "drinking_time_s": round(self.drinking_time_s, 2),
            "current_behavior": self.current_behavior.value,
            "feeding_events": self.feeding_events,
            "drinking_events": self.drinking_events
        }


# =============================================================================
# Behavior Monitor
# =============================================================================

class HenBehaviorMonitor:
    """
    Monitors and tracks hen behaviors over time.

    Uses bounding box overlap to detect feeding and drinking.
    Accumulates time spent in each behavior state.

    Implements Algorithm 2 from the MDPI paper.

    Usage:
    ```python
    monitor = HenBehaviorMonitor(fps=30.0)

    # Each frame
    monitor.update(hen_detections, feeder_detections, waterer_detections)

    # Get statistics
    stats = monitor.get_stats(hen_id=1)
    all_stats = monitor.get_all_stats()
    ```
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_overlap_area: float = 0.0
    ):
        """
        Initialize behavior monitor.

        Args:
            fps: Frame rate for time calculations
            min_overlap_area: Minimum overlap area to trigger behavior
        """
        self.fps = fps
        self.min_overlap_area = min_overlap_area
        self._time_step = 1.0 / fps

        # Per-hen statistics
        self._stats: Dict[int, HenStats] = {}

        # Event callbacks
        self._callbacks: List[BehaviorChangeCallback] = []

        # Frame counter
        self._frame_count = 0

    # =========================================================================
    # Event System
    # =========================================================================

    def register_callback(self, callback: BehaviorChangeCallback) -> None:
        """
        Register callback for behavior changes.

        Args:
            callback: Function(hen_id, old_behavior, new_behavior, duration)
        """
        self._callbacks.append(callback)

    def unregister_callback(self, callback: BehaviorChangeCallback) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire_behavior_change(
        self,
        hen_id: int,
        old_behavior: BehaviorType,
        new_behavior: BehaviorType,
        duration_s: float
    ) -> None:
        """Fire all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(hen_id, old_behavior, new_behavior, duration_s)
            except Exception as e:
                print(f"Behavior callback error: {e}")

    # =========================================================================
    # Core Update Logic
    # =========================================================================

    def update(
        self,
        hens: List[dict],
        feeders: List[dict],
        waterers: List[dict]
    ) -> None:
        """
        Update behavior states from current frame detections.

        Detection dict format:
        {
            'id': int,
            'box': (x1, y1, x2, y2),
            ...
        }

        Args:
            hens: List of hen detection dictionaries
            feeders: List of feeder detection dictionaries
            waterers: List of waterer detection dictionaries
        """
        self._frame_count += 1

        for hen in hens:
            hen_id = hen.get('id')
            if hen_id is None:
                continue

            hen_box = hen.get('box')
            if hen_box is None:
                continue

            # Ensure stats exist for this hen
            if hen_id not in self._stats:
                self._stats[hen_id] = HenStats(hen_id=hen_id)

            stats = self._stats[hen_id]
            old_behavior = stats.current_behavior
            new_behavior = BehaviorType.IDLE

            # Check feeding (Algorithm 2)
            for feeder in feeders:
                feeder_box = feeder.get('box')
                if feeder_box is None:
                    continue

                is_overlap, area = check_overlap(hen_box, feeder_box)
                if is_overlap and area > self.min_overlap_area:
                    new_behavior = BehaviorType.FEEDING
                    stats.feeding_time_s += self._time_step
                    break

            # Check drinking (only if not feeding)
            if new_behavior == BehaviorType.IDLE:
                for waterer in waterers:
                    waterer_box = waterer.get('box')
                    if waterer_box is None:
                        continue

                    is_overlap, area = check_overlap(hen_box, waterer_box)
                    if is_overlap and area > self.min_overlap_area:
                        new_behavior = BehaviorType.DRINKING
                        stats.drinking_time_s += self._time_step
                        break

            # Handle behavior transitions
            if new_behavior != old_behavior:
                # Calculate duration of previous behavior
                duration = time.time() - stats.behavior_start_time

                # Update event counts
                if new_behavior == BehaviorType.FEEDING:
                    stats.feeding_events += 1
                elif new_behavior == BehaviorType.DRINKING:
                    stats.drinking_events += 1

                # Fire callback
                self._fire_behavior_change(
                    hen_id, old_behavior, new_behavior, duration
                )

                # Update state
                stats.current_behavior = new_behavior
                stats.behavior_start_time = time.time()

    def update_from_boxes(
        self,
        hen_boxes: Dict[int, Tuple],
        feeder_boxes: List[Tuple],
        waterer_boxes: List[Tuple]
    ) -> None:
        """
        Update from raw bounding boxes.

        Convenience method for simpler interfaces.

        Args:
            hen_boxes: Dict mapping hen_id to (x1, y1, x2, y2)
            feeder_boxes: List of feeder boxes
            waterer_boxes: List of waterer boxes
        """
        hens = [{'id': h_id, 'box': box} for h_id, box in hen_boxes.items()]
        feeders = [{'box': box} for box in feeder_boxes]
        waterers = [{'box': box} for box in waterer_boxes]

        self.update(hens, feeders, waterers)

    # =========================================================================
    # Statistics Access
    # =========================================================================

    def get_stats(self, hen_id: int) -> Optional[dict]:
        """
        Get statistics for a specific hen.

        Args:
            hen_id: Hen tracking ID

        Returns:
            Statistics dictionary or None if hen not tracked
        """
        stats = self._stats.get(hen_id)
        return stats.to_dict() if stats else None

    def get_all_stats(self) -> Dict[int, dict]:
        """Get statistics for all tracked hens."""
        return {
            hen_id: stats.to_dict()
            for hen_id, stats in self._stats.items()
        }

    def get_summary(self) -> dict:
        """
        Get summary statistics across all hens.

        Returns:
            Summary dictionary with totals and averages
        """
        if not self._stats:
            return {
                "total_hens": 0,
                "total_feeding_time_s": 0,
                "total_drinking_time_s": 0,
                "avg_feeding_time_s": 0,
                "avg_drinking_time_s": 0,
                "total_feeding_events": 0,
                "total_drinking_events": 0
            }

        total_feeding = sum(s.feeding_time_s for s in self._stats.values())
        total_drinking = sum(s.drinking_time_s for s in self._stats.values())
        total_feeding_events = sum(s.feeding_events for s in self._stats.values())
        total_drinking_events = sum(s.drinking_events for s in self._stats.values())

        n = len(self._stats)

        return {
            "total_hens": n,
            "total_feeding_time_s": round(total_feeding, 2),
            "total_drinking_time_s": round(total_drinking, 2),
            "avg_feeding_time_s": round(total_feeding / n, 2),
            "avg_drinking_time_s": round(total_drinking / n, 2),
            "total_feeding_events": total_feeding_events,
            "total_drinking_events": total_drinking_events,
            "frames_processed": self._frame_count
        }

    def get_current_behaviors(self) -> Dict[int, str]:
        """Get current behavior for all hens."""
        return {
            hen_id: stats.current_behavior.value
            for hen_id, stats in self._stats.items()
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
        self._frame_count = 0

    def reset_hen(self, hen_id: int) -> None:
        """Reset statistics for a specific hen."""
        if hen_id in self._stats:
            self._stats[hen_id] = HenStats(hen_id=hen_id)

    def export_csv(self, filepath: str) -> None:
        """
        Export statistics to CSV file.

        Args:
            filepath: Output file path
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'hen_id', 'feeding_time_s', 'drinking_time_s',
                'current_behavior', 'feeding_events', 'drinking_events'
            ])
            writer.writeheader()
            for stats in self._stats.values():
                writer.writerow(stats.to_dict())

    @property
    def tracked_hen_ids(self) -> List[int]:
        """Get list of tracked hen IDs."""
        return list(self._stats.keys())

    @property
    def frame_count(self) -> int:
        """Get number of frames processed."""
        return self._frame_count
