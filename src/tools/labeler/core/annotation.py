"""
Core data models for pose annotation.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import uuid


@dataclass
class Keypoint:
    """Represents a single keypoint annotation."""
    name: str
    x: float = 0.0  # Normalized 0-1
    y: float = 0.0  # Normalized 0-1
    visibility: int = 0  # 0=unlabeled, 1=occluded, 2=visible
    
    def is_labeled(self) -> bool:
        """Check if keypoint has been labeled."""
        return self.visibility > 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "visibility": self.visibility
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Keypoint":
        return cls(
            name=data["name"],
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            visibility=data.get("visibility", 0)
        )


@dataclass
class BoundingBox:
    """Represents a bounding box (normalized coordinates)."""
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if bounding box has been drawn."""
        return self.width > 0 and self.height > 0
    
    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        """Create from corner coordinates (normalized)."""
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return cls(x_center, y_center, width, height)
    
    def get_corners(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) corner coordinates."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return (x1, y1, x2, y2)
    
    def to_dict(self) -> dict:
        return {
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox":
        return cls(
            x_center=data.get("x_center", 0.0),
            y_center=data.get("y_center", 0.0),
            width=data.get("width", 0.0),
            height=data.get("height", 0.0)
        )


@dataclass
class Instance:
    """Represents a single annotated instance (e.g., one hen)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_id: int = 0
    class_name: str = "hen"
    bbox: BoundingBox = field(default_factory=BoundingBox)
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    visible: bool = True  # UI visibility toggle
    
    def get_labeled_keypoints(self) -> list[Keypoint]:
        """Return only keypoints that have been labeled."""
        return [kp for kp in self.keypoints.values() if kp.is_labeled()]
    
    def is_complete(self, required_keypoints: list[str]) -> bool:
        """Check if all required keypoints are labeled."""
        for kp_name in required_keypoints:
            if kp_name not in self.keypoints or not self.keypoints[kp_name].is_labeled():
                return False
        return True
    
    def compute_bbox_from_keypoints(self) -> None:
        """Auto-compute bounding box from labeled keypoints."""
        labeled = self.get_labeled_keypoints()
        if not labeled:
            return
        
        xs = [kp.x for kp in labeled]
        ys = [kp.y for kp in labeled]
        
        padding = 0.02  # 2% padding
        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(1, max(xs) + padding)
        y2 = min(1, max(ys) + padding)
        
        self.bbox = BoundingBox.from_corners(x1, y1, x2, y2)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "keypoints": {name: kp.to_dict() for name, kp in self.keypoints.items()},
            "visible": self.visible
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        instance = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            class_id=data.get("class_id", 0),
            class_name=data.get("class_name", "hen"),
            bbox=BoundingBox.from_dict(data.get("bbox", {})),
            visible=data.get("visible", True)
        )
        for name, kp_data in data.get("keypoints", {}).items():
            instance.keypoints[name] = Keypoint.from_dict(kp_data)
        return instance


@dataclass
class FrameAnnotation:
    """Represents annotations for a single frame/image."""
    image_path: str
    image_width: int = 0
    image_height: int = 0
    instances: list[Instance] = field(default_factory=list)
    
    def add_instance(self, instance: Instance) -> None:
        self.instances.append(instance)
    
    def remove_instance(self, instance_id: str) -> None:
        self.instances = [i for i in self.instances if i.id != instance_id]
    
    def get_instance(self, instance_id: str) -> Optional[Instance]:
        for instance in self.instances:
            if instance.id == instance_id:
                return instance
        return None
    
    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "instances": [i.to_dict() for i in self.instances]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FrameAnnotation":
        frame = cls(
            image_path=data["image_path"],
            image_width=data.get("image_width", 0),
            image_height=data.get("image_height", 0)
        )
        for inst_data in data.get("instances", []):
            frame.instances.append(Instance.from_dict(inst_data))
        return frame
    
    def to_yolo_format(self, keypoint_names: list[str], num_dims: int = 3) -> str:
        """
        Export frame annotations to YOLO pose format.
        
        Format: <class-index> <x> <y> <width> <height> <px1> <py1> <v1> ... <pxn> <pyn> <vn>
        """
        lines = []
        
        for instance in self.instances:
            if not instance.bbox.is_valid():
                continue
            
            parts = [
                str(instance.class_id),
                f"{instance.bbox.x_center:.6f}",
                f"{instance.bbox.y_center:.6f}",
                f"{instance.bbox.width:.6f}",
                f"{instance.bbox.height:.6f}"
            ]
            
            # Add keypoints in order
            for kp_name in keypoint_names:
                kp = instance.keypoints.get(kp_name)
                if kp and kp.is_labeled():
                    parts.append(f"{kp.x:.6f}")
                    parts.append(f"{kp.y:.6f}")
                    if num_dims == 3:
                        parts.append(str(kp.visibility))
                else:
                    # Unlabeled keypoint
                    parts.append("0.000000")
                    parts.append("0.000000")
                    if num_dims == 3:
                        parts.append("0")
            
            lines.append(" ".join(parts))
        
        return "\n".join(lines)
