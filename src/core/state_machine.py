"""
State machine for pen entities and behavior detection.

This module manages the spatial world model of a poultry pen:
- Tracks positions of hens, feeders, and waterers in world coordinates
- Detects behavioral states (idle, eating, drinking)
- Generates minimap visualizations
- Fires callbacks on behavior state changes

Refactored from pen_state.py with event-driven architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from enum import Enum
from collections import deque
import numpy as np
import cv2

from .geometry import COLORS, is_point_in_circle, scale_point


# =============================================================================
# Enums and Types
# =============================================================================

class BehaviorState(Enum):
    """Hen behavior states."""
    IDLE = "idle"
    EATING = "eating"
    DRINKING = "drinking"
    MOVING = "moving"


class EntityType(Enum):
    """Types of entities in the pen."""
    HEN = "hen"
    FEEDER = "feeder"
    WATERER = "waterer"


# Callback type for behavior state changes
# Signature: (hen_id, old_state, new_state) -> None
BehaviorCallback = Callable[[int, BehaviorState, BehaviorState], None]


# =============================================================================
# Entity Classes
# =============================================================================

@dataclass
class Entity:
    """
    Represents a tracked entity in the pen (hen, feeder, or waterer).

    Attributes:
        entity_id: Unique tracking identifier
        entity_type: Type of entity (hen, feeder, waterer)
        position: Current position in world coordinates (cm)
        radius: Estimated radius in cm (for feeders/waterers)
        behavior: Current behavioral state (for hens only)
        history: Position history for trail visualization
        last_seen_ms: Timestamp of last detection (milliseconds)
    """
    entity_id: int
    entity_type: EntityType
    position: Tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    behavior: BehaviorState = BehaviorState.IDLE
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    last_seen_ms: float = 0.0

    # Behavior timing accumulators (for hens)
    eating_time_s: float = 0.0
    drinking_time_s: float = 0.0

    def update(
        self,
        position: Tuple[float, float],
        timestamp_ms: float,
        radius: Optional[float] = None
    ) -> None:
        """
        Update entity position and metadata.

        Args:
            position: New position in world coordinates (cm)
            timestamp_ms: Current timestamp
            radius: Optional new radius estimate
        """
        if position[0] > 0 and position[1] > 0:
            self.history.append(self.position)
            self.position = position
            self.last_seen_ms = timestamp_ms
            if radius is not None:
                self.radius = radius

    def get_stats(self) -> dict:
        """Return statistics dictionary for this entity."""
        return {
            "id": self.entity_id,
            "type": self.entity_type.value,
            "position": self.position,
            "behavior": self.behavior.value,
            "eating_time_s": self.eating_time_s,
            "drinking_time_s": self.drinking_time_s,
            "history_length": len(self.history)
        }


# Legacy Entity class for backwards compatibility
class LegacyEntity:
    """
    Legacy Entity class for backwards compatibility with pen_state.py.

    Deprecated: Use Entity dataclass instead.
    """

    def __init__(self, e_id: int, e_type: str):
        self.id = e_id
        self.type = e_type
        self.position = (0, 0)
        self.radius = 0
        self.history = deque(maxlen=40)
        self.state = "Idle"
        self.last_seen = 0

    def update(self, pos: Tuple, timestamp: float, radius: Optional[float] = None):
        if pos[0] > 0 and pos[1] > 0:
            self.history.append(self.position)
            self.position = pos
            self.last_seen = timestamp
            if radius:
                self.radius = radius


# =============================================================================
# Pen State Machine
# =============================================================================

class PenStateMachine:
    """
    Manages the complete state of a poultry pen.

    Features:
    - Entity storage for hens, feeders, and waterers
    - Interaction detection (eating/drinking zones)
    - Event-driven behavior state changes with callbacks
    - Minimap generation for visualization
    - Position history tracking for trails

    Attributes:
        width: Pen width in centimeters
        height: Pen height in centimeters
        interaction_radius: Buffer distance for interaction detection (cm)
        hens: Dictionary of tracked hens
        feeders: Dictionary of tracked feeders
        waterers: Dictionary of tracked waterers
    """

    def __init__(
        self,
        width_cm: float,
        height_cm: float,
        interaction_radius_cm: float = 15.0
    ):
        """
        Initialize pen state machine.

        Args:
            width_cm: Physical pen width in centimeters
            height_cm: Physical pen height in centimeters
            interaction_radius_cm: Buffer radius for feeder/waterer zones
        """
        self.width = width_cm
        self.height = height_cm
        self.interaction_radius = interaction_radius_cm

        # Entity storage
        self.hens: Dict[int, Entity] = {}
        self.feeders: Dict[int, Entity] = {}
        self.waterers: Dict[int, Entity] = {}

        # Event callbacks
        self._behavior_callbacks: List[BehaviorCallback] = []

        # Frame timing
        self._last_update_ms: float = 0.0

    # =========================================================================
    # Event System
    # =========================================================================

    def register_behavior_callback(self, callback: BehaviorCallback) -> None:
        """
        Register a callback for behavior state changes.

        Callback will be invoked whenever a hen's behavior changes.

        Args:
            callback: Function with signature (hen_id, old_state, new_state)
        """
        self._behavior_callbacks.append(callback)

    def unregister_behavior_callback(self, callback: BehaviorCallback) -> None:
        """Remove a previously registered callback."""
        if callback in self._behavior_callbacks:
            self._behavior_callbacks.remove(callback)

    def _fire_behavior_change(
        self,
        hen_id: int,
        old_state: BehaviorState,
        new_state: BehaviorState
    ) -> None:
        """Fire all registered behavior change callbacks."""
        for callback in self._behavior_callbacks:
            try:
                callback(hen_id, old_state, new_state)
            except Exception as e:
                # Log but don't crash on callback errors
                print(f"Behavior callback error: {e}")

    # =========================================================================
    # Entity Management
    # =========================================================================

    def update_entity(
        self,
        entity_type: str,
        entity_id: int,
        position: Tuple[float, float],
        timestamp_ms: float,
        radius: Optional[float] = None
    ) -> None:
        """
        Update or create an entity from a detection.

        Args:
            entity_type: Type string ("hen", "feeder", "waterer")
            entity_id: Tracking ID
            position: World coordinates (cm)
            timestamp_ms: Current frame timestamp
            radius: Optional radius estimate (for feeders/waterers)
        """
        # Route to appropriate container
        target_dict: Optional[Dict[int, Entity]] = None
        etype: Optional[EntityType] = None

        if entity_type == "hen":
            target_dict = self.hens
            etype = EntityType.HEN
        elif entity_type == "feeder":
            target_dict = self.feeders
            etype = EntityType.FEEDER
        elif entity_type == "waterer":
            target_dict = self.waterers
            etype = EntityType.WATERER

        if target_dict is None or etype is None:
            return

        # Create entity if new
        if entity_id not in target_dict:
            target_dict[entity_id] = Entity(
                entity_id=entity_id,
                entity_type=etype
            )

        # Update entity state
        target_dict[entity_id].update(position, timestamp_ms, radius)

    def update_from_detections(
        self,
        detections: List[dict],
        timestamp_ms: float
    ) -> None:
        """
        Batch update from a list of detection dictionaries.

        Expected detection format:
        {
            'id': int,
            'type': str,  # "hen", "feeder", "waterer"
            'pos': (x, y),  # World coordinates
            'radius': float  # Optional
        }

        Args:
            detections: List of detection dictionaries
            timestamp_ms: Current frame timestamp
        """
        delta_s = 0.0
        if self._last_update_ms > 0:
            delta_s = (timestamp_ms - self._last_update_ms) / 1000.0
        self._last_update_ms = timestamp_ms

        # Update all entities
        for det in detections:
            self.update_entity(
                entity_type=det.get('type', ''),
                entity_id=det.get('id', -1),
                position=det.get('pos', (0, 0)),
                timestamp_ms=timestamp_ms,
                radius=det.get('radius')
            )

        # Check interactions and update behaviors
        self._check_interactions(delta_s)

    def get_hen_stats(self, hen_id: int) -> Optional[dict]:
        """
        Get statistics for a specific hen.

        Args:
            hen_id: Tracking ID of the hen

        Returns:
            Statistics dictionary or None if hen not found
        """
        hen = self.hens.get(hen_id)
        return hen.get_stats() if hen else None

    def get_all_stats(self) -> dict:
        """Get statistics for all entities."""
        return {
            "hens": {h_id: h.get_stats() for h_id, h in self.hens.items()},
            "feeders": len(self.feeders),
            "waterers": len(self.waterers),
            "total_eating_time": sum(h.eating_time_s for h in self.hens.values()),
            "total_drinking_time": sum(h.drinking_time_s for h in self.hens.values())
        }

    # =========================================================================
    # Interaction Detection
    # =========================================================================

    def _check_interactions(self, delta_s: float = 0.0) -> None:
        """
        Check hen interactions with feeders and waterers.

        Updates behavior states and accumulates timing.

        Args:
            delta_s: Time delta since last update (seconds)
        """
        for hen_id, hen in self.hens.items():
            old_behavior = hen.behavior
            new_behavior = BehaviorState.IDLE

            # Check feeders first (priority)
            for feeder in self.feeders.values():
                interaction_radius = feeder.radius + self.interaction_radius
                if is_point_in_circle(hen.position, feeder.position, interaction_radius):
                    new_behavior = BehaviorState.EATING
                    hen.eating_time_s += delta_s
                    break

            # Check waterers (only if not eating)
            if new_behavior == BehaviorState.IDLE:
                for waterer in self.waterers.values():
                    interaction_radius = waterer.radius + self.interaction_radius
                    if is_point_in_circle(hen.position, waterer.position, interaction_radius):
                        new_behavior = BehaviorState.DRINKING
                        hen.drinking_time_s += delta_s
                        break

            # Update behavior and fire events
            hen.behavior = new_behavior
            if new_behavior != old_behavior:
                self._fire_behavior_change(hen_id, old_behavior, new_behavior)

    # =========================================================================
    # Visualization
    # =========================================================================

    def generate_minimap(self, render_scale: int = 3) -> np.ndarray:
        """
        Generate a minimap visualization of the pen state.

        Creates a FIFA-style tactical view with:
        - Feeders (orange circles)
        - Waterers (blue circles)
        - Hens (colored by behavior state with trails)

        Args:
            render_scale: Pixels per centimeter

        Returns:
            BGR image as numpy array
        """
        # Check interactions before drawing
        self._check_interactions()

        # Create canvas
        w = int(self.width * render_scale)
        h = int(self.height * render_scale)
        minimap = np.zeros((h, w, 3), dtype=np.uint8)
        minimap[:] = (30, 30, 30)  # Dark grey background

        # Helper for coordinate conversion
        def to_pix(pt: Tuple[float, float]) -> Tuple[int, int]:
            return scale_point(pt, render_scale)

        def sc(val: float) -> int:
            return int(val * render_scale)

        # Layer 0: Draw feeders
        for feeder in self.feeders.values():
            center = to_pix(feeder.position)
            rad = sc(feeder.radius if feeder.radius > 10 else 25)
            cv2.circle(minimap, center, rad, COLORS['feeder'], -1)
            cv2.putText(
                minimap, "Feeder",
                (center[0] - 20, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Layer 0: Draw waterers
        for waterer in self.waterers.values():
            center = to_pix(waterer.position)
            rad = sc(waterer.radius if waterer.radius > 10 else 20)
            cv2.circle(minimap, center, rad, COLORS['waterer'], -1)
            cv2.putText(
                minimap, "Water",
                (center[0] - 20, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Layer 1: Draw hens
        for hen in self.hens.values():
            if hen.position == (0, 0):
                continue

            # Draw trail
            if len(hen.history) > 1:
                pts = np.array(
                    [to_pix(pt) for pt in hen.history],
                    np.int32
                ).reshape((-1, 1, 2))
                cv2.polylines(minimap, [pts], False, COLORS['trail'], 1)

            # Determine color by behavior state
            if hen.behavior == BehaviorState.EATING:
                color = COLORS['hen_eating']
            elif hen.behavior == BehaviorState.DRINKING:
                color = COLORS['hen_drinking']
            else:
                color = COLORS['hen']

            # Draw hen dot
            center = to_pix(hen.position)
            cv2.circle(minimap, center, sc(10), color, -1)

            # ID label
            cv2.putText(
                minimap, str(hen.entity_id),
                (center[0] - 5, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

        return minimap


# =============================================================================
# Legacy Class for Backwards Compatibility
# =============================================================================

class PenState:
    """
    Legacy PenState class for backwards compatibility.

    Deprecated: Use PenStateMachine instead.
    """

    def __init__(self, phys_width_cm: float, phys_length_cm: float):
        self.width = phys_width_cm
        self.height = phys_length_cm
        self.hens: Dict[int, LegacyEntity] = {}
        self.feeders: Dict[int, LegacyEntity] = {}
        self.waterers: Dict[int, LegacyEntity] = {}

    def update_entity(
        self,
        e_type: str,
        e_id: int,
        pos: Tuple,
        timestamp: float,
        radius: Optional[float] = None
    ) -> None:
        target_dict = None
        if e_type == 'hen':
            target_dict = self.hens
        elif e_type == 'feeder':
            target_dict = self.feeders
        elif e_type == 'waterer':
            target_dict = self.waterers

        if target_dict is not None:
            if e_id not in target_dict:
                target_dict[e_id] = LegacyEntity(e_id, e_type)
            target_dict[e_id].update(pos, timestamp, radius)

    def _check_interactions(self) -> None:
        for h_id, hen in self.hens.items():
            hen.state = "Idle"
            for f in self.feeders.values():
                if is_point_in_circle(hen.position, f.position, f.radius + 15):
                    hen.state = "Eating"
                    break
            if hen.state == "Idle":
                for w in self.waterers.values():
                    if is_point_in_circle(hen.position, w.position, w.radius + 15):
                        hen.state = "Drinking"
                        break

    def generate_minimap(self, render_scale: int = 3) -> np.ndarray:
        self._check_interactions()
        w, h = int(self.width * render_scale), int(self.height * render_scale)
        minimap = np.zeros((h, w, 3), dtype=np.uint8)
        minimap[:] = (30, 30, 30)

        def to_pix(pt):
            return (int(pt[0] * render_scale), int(pt[1] * render_scale))

        def sc(val):
            return int(val * render_scale)

        for f in self.feeders.values():
            center = to_pix(f.position)
            rad = sc(f.radius if f.radius > 10 else 25)
            cv2.circle(minimap, center, rad, COLORS['feeder'], -1)
            cv2.putText(minimap, "Feeder", (center[0] - 20, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for w in self.waterers.values():
            center = to_pix(w.position)
            rad = sc(w.radius if w.radius > 10 else 20)
            cv2.circle(minimap, center, rad, COLORS['waterer'], -1)
            cv2.putText(minimap, "Water", (center[0] - 20, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for hen in self.hens.values():
            if hen.position == (0, 0):
                continue
            if len(hen.history) > 1:
                pts = np.array([to_pix(pt) for pt in hen.history], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(minimap, [pts], False, (100, 100, 100), 1)

            color = COLORS['hen']
            if hen.state == "Eating":
                color = COLORS['hen_eating']
            elif hen.state == "Drinking":
                color = COLORS['waterer']

            center = to_pix(hen.position)
            cv2.circle(minimap, center, sc(10), color, -1)
            cv2.putText(minimap, str(hen.id), (center[0] - 5, center[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return minimap
