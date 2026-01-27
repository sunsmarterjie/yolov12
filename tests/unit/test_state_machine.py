"""
Unit tests for state machine module.
"""

import pytest
import numpy as np

from src.core.state_machine import (
    Entity,
    EntityType,
    BehaviorState,
    PenStateMachine,
)


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test entity creation with defaults."""
        entity = Entity(entity_id=1, entity_type=EntityType.HEN)
        assert entity.entity_id == 1
        assert entity.entity_type == EntityType.HEN
        assert entity.position == (0.0, 0.0)
        assert entity.behavior == BehaviorState.IDLE
        assert len(entity.history) == 0

    def test_entity_update_position(self):
        """Test entity position update."""
        entity = Entity(entity_id=1, entity_type=EntityType.HEN)
        entity.update((50.0, 60.0), timestamp_ms=1000.0)

        assert entity.position == (50.0, 60.0)
        assert entity.last_seen_ms == 1000.0
        assert len(entity.history) == 1

    def test_entity_update_skips_zero_position(self):
        """Test that zero position updates are skipped."""
        entity = Entity(entity_id=1, entity_type=EntityType.HEN)
        entity.update((50.0, 60.0), timestamp_ms=1000.0)
        entity.update((0.0, 0.0), timestamp_ms=2000.0)

        # Position should not change
        assert entity.position == (50.0, 60.0)
        assert entity.last_seen_ms == 1000.0

    def test_entity_history_limit(self):
        """Test entity history is limited to maxlen."""
        entity = Entity(entity_id=1, entity_type=EntityType.HEN)

        # Add more than maxlen positions
        for i in range(50):
            entity.update((float(i), float(i)), timestamp_ms=float(i * 100))

        # History should be capped at 40 (default maxlen)
        assert len(entity.history) == 40

    def test_entity_get_stats(self):
        """Test entity statistics retrieval."""
        entity = Entity(entity_id=1, entity_type=EntityType.HEN)
        entity.position = (50.0, 60.0)
        entity.behavior = BehaviorState.EATING
        entity.eating_time_s = 10.5

        stats = entity.get_stats()
        assert stats["id"] == 1
        assert stats["type"] == "hen"
        assert stats["position"] == (50.0, 60.0)
        assert stats["behavior"] == "eating"
        assert stats["eating_time_s"] == 10.5


class TestPenStateMachine:
    """Tests for PenStateMachine class."""

    def test_pen_state_creation(self):
        """Test pen state machine creation."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)
        assert pen.width == 120.0
        assert pen.height == 200.0
        assert len(pen.hens) == 0
        assert len(pen.feeders) == 0
        assert len(pen.waterers) == 0

    def test_update_entity_creates_hen(self):
        """Test that update_entity creates new hen."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)
        pen.update_entity("hen", 1, (50.0, 60.0), timestamp_ms=1000.0)

        assert 1 in pen.hens
        assert pen.hens[1].position == (50.0, 60.0)

    def test_update_entity_creates_feeder(self):
        """Test that update_entity creates new feeder."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)
        pen.update_entity("feeder", 1, (30.0, 40.0), timestamp_ms=1000.0, radius=25.0)

        assert 1 in pen.feeders
        assert pen.feeders[1].radius == 25.0

    def test_update_entity_updates_existing(self):
        """Test that update_entity updates existing entity."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)
        pen.update_entity("hen", 1, (50.0, 60.0), timestamp_ms=1000.0)
        pen.update_entity("hen", 1, (55.0, 65.0), timestamp_ms=2000.0)

        assert pen.hens[1].position == (55.0, 65.0)
        assert len(pen.hens[1].history) == 1

    def test_interaction_detection_eating(self):
        """Test eating behavior detection."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0, interaction_radius_cm=15.0)

        # Add feeder at (50, 50) with radius 20
        pen.update_entity("feeder", 1, (50.0, 50.0), timestamp_ms=1000.0, radius=20.0)

        # Add hen near feeder (within radius + interaction_radius)
        pen.update_entity("hen", 1, (55.0, 55.0), timestamp_ms=1000.0)

        # Check interactions
        pen._check_interactions(delta_s=0.033)

        assert pen.hens[1].behavior == BehaviorState.EATING

    def test_interaction_detection_drinking(self):
        """Test drinking behavior detection."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0, interaction_radius_cm=15.0)

        # Add waterer at (100, 100) with radius 15
        pen.update_entity("waterer", 1, (100.0, 100.0), timestamp_ms=1000.0, radius=15.0)

        # Add hen near waterer
        pen.update_entity("hen", 1, (105.0, 105.0), timestamp_ms=1000.0)

        # Check interactions
        pen._check_interactions(delta_s=0.033)

        assert pen.hens[1].behavior == BehaviorState.DRINKING

    def test_interaction_eating_priority(self):
        """Test that eating has priority over drinking."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0, interaction_radius_cm=15.0)

        # Place feeder and waterer at same location
        pen.update_entity("feeder", 1, (50.0, 50.0), timestamp_ms=1000.0, radius=20.0)
        pen.update_entity("waterer", 1, (50.0, 50.0), timestamp_ms=1000.0, radius=15.0)

        # Add hen near both
        pen.update_entity("hen", 1, (55.0, 55.0), timestamp_ms=1000.0)

        pen._check_interactions(delta_s=0.033)

        # Eating should have priority
        assert pen.hens[1].behavior == BehaviorState.EATING

    def test_behavior_callback(self):
        """Test behavior change callback firing."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0, interaction_radius_cm=15.0)

        callback_events = []

        def callback(hen_id, old_state, new_state):
            callback_events.append((hen_id, old_state, new_state))

        pen.register_behavior_callback(callback)

        # Add feeder
        pen.update_entity("feeder", 1, (50.0, 50.0), timestamp_ms=1000.0, radius=20.0)

        # Add hen (starts idle)
        pen.update_entity("hen", 1, (100.0, 100.0), timestamp_ms=1000.0)
        pen._check_interactions()

        # Move hen to feeder (triggers eating)
        pen.update_entity("hen", 1, (55.0, 55.0), timestamp_ms=2000.0)
        pen._check_interactions()

        assert len(callback_events) == 1
        assert callback_events[0][0] == 1  # hen_id
        assert callback_events[0][1] == BehaviorState.IDLE
        assert callback_events[0][2] == BehaviorState.EATING

    def test_minimap_generation(self):
        """Test minimap generation returns valid image."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)

        # Add some entities
        pen.update_entity("feeder", 1, (30.0, 30.0), timestamp_ms=1000.0, radius=25.0)
        pen.update_entity("hen", 1, (60.0, 100.0), timestamp_ms=1000.0)

        minimap = pen.generate_minimap(render_scale=3)

        assert minimap is not None
        assert minimap.shape == (600, 360, 3)  # 200*3 x 120*3 x 3 channels

    def test_get_all_stats(self):
        """Test getting all statistics."""
        pen = PenStateMachine(width_cm=120.0, height_cm=200.0)
        pen.update_entity("hen", 1, (50.0, 60.0), timestamp_ms=1000.0)
        pen.update_entity("hen", 2, (70.0, 80.0), timestamp_ms=1000.0)
        pen.update_entity("feeder", 1, (30.0, 30.0), timestamp_ms=1000.0)

        stats = pen.get_all_stats()
        assert len(stats["hens"]) == 2
        assert stats["feeders"] == 1
        assert stats["waterers"] == 0
