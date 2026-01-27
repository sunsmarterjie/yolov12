"""
Core module for PoseLabeler annotation tool.
"""

# Annotation classes
from .annotation import (
    Keypoint,
    BoundingBox,
    Instance,
    FrameAnnotation
)

# Project management
from .project import (
    Project,
    ProjectManager
)

# Schema definitions
from .schema import (
    KeypointDef,
    Schema,
    create_poultry_schema,
    get_builtin_schema,
    list_builtin_schemas
)

__all__ = [
    # Annotation
    'Keypoint',
    'BoundingBox',
    'Instance',
    'FrameAnnotation',
    # Project
    'Project',
    'ProjectManager',
    # Schema
    'KeypointDef',
    'Schema',
    'create_poultry_schema',
    'get_builtin_schema',
    'list_builtin_schemas',
]
