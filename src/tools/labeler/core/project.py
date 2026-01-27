"""
Project management for pose annotation projects.
Handles loading, saving, autosave, and YOLO export.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import shutil
import yaml
from datetime import datetime

from .annotation import FrameAnnotation, Instance, Keypoint
from .schema import Schema, create_poultry_schema


@dataclass
class Project:
    """Represents a pose annotation project."""
    name: str
    path: Path
    schema: Schema
    image_folder: Path
    frames: dict[str, FrameAnnotation] = field(default_factory=dict)
    current_frame_index: int = 0
    image_files: list[Path] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if not self.image_files:
            self.scan_images()
    
    def scan_images(self) -> None:
        """Scan image folder for supported image files."""
        if not self.image_folder.exists():
            self.image_files = []
            return
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = sorted([
            f for f in self.image_folder.iterdir()
            if f.suffix.lower() in extensions
        ])
    
    def get_current_frame(self) -> Optional[FrameAnnotation]:
        """Get the current frame annotation."""
        if not self.image_files:
            return None
        
        if self.current_frame_index >= len(self.image_files):
            self.current_frame_index = len(self.image_files) - 1
        
        image_path = self.image_files[self.current_frame_index]
        key = str(image_path)
        
        if key not in self.frames:
            self.frames[key] = FrameAnnotation(image_path=key)
        
        return self.frames[key]
    
    def get_frame_by_path(self, image_path: str) -> Optional[FrameAnnotation]:
        """Get frame annotation by image path."""
        if image_path not in self.frames:
            self.frames[image_path] = FrameAnnotation(image_path=image_path)
        return self.frames[image_path]
    
    def next_frame(self) -> Optional[FrameAnnotation]:
        """Move to and return the next frame."""
        if self.current_frame_index < len(self.image_files) - 1:
            self.current_frame_index += 1
        return self.get_current_frame()
    
    def prev_frame(self) -> Optional[FrameAnnotation]:
        """Move to and return the previous frame."""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
        return self.get_current_frame()
    
    def go_to_frame(self, index: int) -> Optional[FrameAnnotation]:
        """Go to a specific frame index."""
        if 0 <= index < len(self.image_files):
            self.current_frame_index = index
        return self.get_current_frame()
    
    def get_progress(self) -> tuple[int, int, int]:
        """Return (current_index, total_frames, annotated_frames)."""
        annotated = sum(1 for f in self.frames.values() if f.instances)
        return (self.current_frame_index, len(self.image_files), annotated)
    
    def mark_modified(self) -> None:
        """Update modification timestamp."""
        self.modified_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Serialize project to dictionary."""
        return {
            "name": self.name,
            "schema": self.schema.to_dict(),
            "image_folder": str(self.image_folder),
            "current_frame_index": self.current_frame_index,
            "frames": {k: v.to_dict() for k, v in self.frames.items()},
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict, project_path: Path) -> "Project":
        """Deserialize project from dictionary."""
        schema = Schema.from_dict(data["schema"])
        
        project = cls(
            name=data["name"],
            path=project_path,
            schema=schema,
            image_folder=Path(data["image_folder"]),
            current_frame_index=data.get("current_frame_index", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat())
        )
        
        for key, frame_data in data.get("frames", {}).items():
            project.frames[key] = FrameAnnotation.from_dict(frame_data)
        
        return project
    
    def save(self) -> None:
        """Save project to file."""
        self.mark_modified()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Project":
        """Load project from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data, path)
    
    @classmethod
    def create_new(cls, name: str, project_dir: Path, image_folder: Path, 
                   schema: Optional[Schema] = None) -> "Project":
        """Create a new project."""
        if schema is None:
            schema = create_poultry_schema()
        
        project_path = project_dir / f"{name}.poselabel"
        
        project = cls(
            name=name,
            path=project_path,
            schema=schema,
            image_folder=image_folder
        )
        
        project.save()
        return project
    
    def export_yolo(self, output_dir: Path, train_split: float = 0.8) -> dict:
        """
        Export annotations to YOLO format.
        
        Returns dict with export statistics.
        """
        output_dir = Path(output_dir)
        
        # Create directory structure
        images_train = output_dir / "images" / "train"
        images_val = output_dir / "images" / "val"
        labels_train = output_dir / "labels" / "train"
        labels_val = output_dir / "labels" / "val"
        
        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Get annotated frames
        annotated_frames = [
            (path, frame) for path, frame in self.frames.items()
            if frame.instances
        ]
        
        # Split into train/val
        split_idx = int(len(annotated_frames) * train_split)
        train_frames = annotated_frames[:split_idx]
        val_frames = annotated_frames[split_idx:]
        
        stats = {
            "total_frames": len(annotated_frames),
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "total_instances": 0
        }
        
        keypoint_names = self.schema.get_keypoint_names()
        
        # Export train set
        for img_path, frame in train_frames:
            self._export_frame(
                frame, Path(img_path), images_train, labels_train, keypoint_names
            )
            stats["total_instances"] += len(frame.instances)
        
        # Export val set
        for img_path, frame in val_frames:
            self._export_frame(
                frame, Path(img_path), images_val, labels_val, keypoint_names
            )
            stats["total_instances"] += len(frame.instances)
        
        # Create data.yaml
        yaml_config = self.schema.to_yolo_config(str(output_dir))
        yaml_path = output_dir / "data.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        stats["yaml_path"] = str(yaml_path)
        
        return stats
    
    def _export_frame(self, frame: FrameAnnotation, src_image: Path,
                      images_dir: Path, labels_dir: Path, 
                      keypoint_names: list[str]) -> None:
        """Export a single frame's image and labels."""
        # Copy image
        if src_image.exists():
            dst_image = images_dir / src_image.name
            shutil.copy2(src_image, dst_image)
        
        # Write label file
        label_name = src_image.stem + ".txt"
        label_path = labels_dir / label_name
        
        yolo_content = frame.to_yolo_format(keypoint_names, num_dims=3)
        
        with open(label_path, 'w') as f:
            f.write(yolo_content)


class ProjectManager:
    """Manages project operations including autosave."""
    
    def __init__(self):
        self.current_project: Optional[Project] = None
        self._autosave_enabled = True
        self._autosave_interval = 60  # seconds
        self._last_save_time: Optional[datetime] = None
        self._unsaved_changes = False
    
    def create_project(self, name: str, project_dir: Path, image_folder: Path,
                       schema: Optional[Schema] = None) -> Project:
        """Create and set a new project as current."""
        self.current_project = Project.create_new(name, project_dir, image_folder, schema)
        self._unsaved_changes = False
        self._last_save_time = datetime.now()
        return self.current_project
    
    def open_project(self, path: Path) -> Project:
        """Open an existing project."""
        self.current_project = Project.load(path)
        self._unsaved_changes = False
        self._last_save_time = datetime.now()
        return self.current_project
    
    def save_project(self) -> bool:
        """Save current project."""
        if self.current_project:
            self.current_project.save()
            self._unsaved_changes = False
            self._last_save_time = datetime.now()
            return True
        return False
    
    def mark_changed(self) -> None:
        """Mark that unsaved changes exist."""
        self._unsaved_changes = True
        if self.current_project:
            self.current_project.mark_modified()
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def should_autosave(self) -> bool:
        """Check if autosave should trigger."""
        if not self._autosave_enabled or not self._unsaved_changes:
            return False
        
        if self._last_save_time is None:
            return True
        
        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        return elapsed >= self._autosave_interval
    
    def autosave_if_needed(self) -> bool:
        """Perform autosave if conditions are met."""
        if self.should_autosave():
            return self.save_project()
        return False
    
    def set_autosave_interval(self, seconds: int) -> None:
        """Set autosave interval in seconds."""
        self._autosave_interval = max(10, seconds)
    
    def enable_autosave(self, enabled: bool = True) -> None:
        """Enable or disable autosave."""
        self._autosave_enabled = enabled
    
    def close_project(self) -> bool:
        """Close current project. Returns True if closed successfully."""
        if self._unsaved_changes:
            # Caller should prompt user to save
            return False
        self.current_project = None
        return True
