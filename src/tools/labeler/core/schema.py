"""
Schema definitions for pose estimation configurations.
Supports loading keypoint definitions from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class KeypointDef:
    """Definition of a keypoint type."""
    name: str
    color: str = "#FF0000"
    index: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": self.color,
            "index": self.index
        }


@dataclass
class Schema:
    """Schema defining keypoints, skeleton, and classes for a domain."""
    name: str
    version: str = "1.0"
    classes: dict[int, str] = field(default_factory=dict)
    keypoints: list[KeypointDef] = field(default_factory=list)
    skeleton: list[tuple[str, str]] = field(default_factory=list)
    flip_pairs: list[tuple[str, str]] = field(default_factory=list)
    
    def get_keypoint_names(self) -> list[str]:
        """Return ordered list of keypoint names."""
        return [kp.name for kp in self.keypoints]
    
    def get_keypoint_colors(self) -> dict[str, str]:
        """Return mapping of keypoint name to color."""
        return {kp.name: kp.color for kp in self.keypoints}
    
    def get_keypoint_index(self, name: str) -> int:
        """Get the index of a keypoint by name."""
        for i, kp in enumerate(self.keypoints):
            if kp.name == name:
                return i
        return -1
    
    def get_flip_idx(self) -> list[int]:
        """Generate flip_idx array for YOLO config."""
        names = self.get_keypoint_names()
        flip_idx = list(range(len(names)))
        
        for left, right in self.flip_pairs:
            left_idx = self.get_keypoint_index(left)
            right_idx = self.get_keypoint_index(right)
            if left_idx >= 0 and right_idx >= 0:
                flip_idx[left_idx] = right_idx
                flip_idx[right_idx] = left_idx
        
        return flip_idx
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "skeleton": self.skeleton,
            "flip_pairs": self.flip_pairs
        }
    
    def to_yolo_config(self, dataset_path: str, train_path: str = "images/train", 
                       val_path: str = "images/val") -> dict:
        """Generate YOLO dataset config dictionary."""
        return {
            "path": dataset_path,
            "train": train_path,
            "val": val_path,
            "kpt_shape": [len(self.keypoints), 3],  # [num_keypoints, dims (x,y,visibility)]
            "flip_idx": self.get_flip_idx(),
            "names": self.classes,
            "kpt_names": {
                class_id: self.get_keypoint_names() 
                for class_id in self.classes.keys()
            }
        }
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Schema":
        """Load schema from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Schema":
        """Create schema from dictionary."""
        keypoints = []
        for i, kp_data in enumerate(data.get("keypoints", [])):
            if isinstance(kp_data, dict):
                keypoints.append(KeypointDef(
                    name=kp_data["name"],
                    color=kp_data.get("color", "#FF0000"),
                    index=i
                ))
            else:
                keypoints.append(KeypointDef(name=kp_data, index=i))
        
        skeleton = []
        for conn in data.get("skeleton", []):
            if isinstance(conn, (list, tuple)) and len(conn) == 2:
                skeleton.append((conn[0], conn[1]))
        
        flip_pairs = []
        for pair in data.get("flip_pairs", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                flip_pairs.append((pair[0], pair[1]))
        
        classes = data.get("classes", {0: "object"})
        if isinstance(classes, dict):
            classes = {int(k): v for k, v in classes.items()}
        
        return cls(
            name=data.get("name", "Custom"),
            version=data.get("version", "1.0"),
            classes=classes,
            keypoints=keypoints,
            skeleton=skeleton,
            flip_pairs=flip_pairs
        )
    
    def save_yaml(self, path: Path) -> None:
        """Save schema to YAML file."""
        data = {
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "keypoints": [
                {"name": kp.name, "color": kp.color}
                for kp in self.keypoints
            ],
            "skeleton": [list(conn) for conn in self.skeleton],
            "flip_pairs": [list(pair) for pair in self.flip_pairs]
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_poultry_schema() -> Schema:
    """Create the default poultry/hen pose schema based on your DeepLabCut config."""
    return Schema(
        name="Poultry Pose",
        version="1.0",
        classes={0: "hen"},
        keypoints=[
            KeypointDef(name="body_center", color="#FF6B6B", index=0),
            KeypointDef(name="body_tail", color="#4ECDC4", index=1),
            KeypointDef(name="body_knee_left", color="#45B7D1", index=2),
            KeypointDef(name="body_knee_right", color="#96CEB4", index=3),
            KeypointDef(name="body_heel_left", color="#FFEAA7", index=4),
            KeypointDef(name="body_heel_right", color="#DDA0DD", index=5),
            KeypointDef(name="eye_left", color="#98D8C8", index=6),
            KeypointDef(name="eye_right", color="#F7DC6F", index=7),
            KeypointDef(name="comb", color="#BB8FCE", index=8),
            KeypointDef(name="beak", color="#F8B500", index=9),
        ],
        skeleton=[
            ("body_center", "body_tail"),
            ("body_center", "body_knee_left"),
            ("body_center", "body_knee_right"),
            ("body_center", "eye_left"),
            ("body_center", "eye_right"),
            ("body_knee_left", "body_heel_left"),
            ("body_knee_right", "body_heel_right"),
            ("eye_left", "comb"),
            ("eye_left", "beak"),
            ("eye_right", "comb"),
            ("eye_right", "beak"),
        ],
        flip_pairs=[
            ("body_knee_left", "body_knee_right"),
            ("body_heel_left", "body_heel_right"),
            ("eye_left", "eye_right"),
        ]
    )


# Built-in schemas
BUILTIN_SCHEMAS = {
    "poultry": create_poultry_schema,
}


def get_builtin_schema(name: str) -> Optional[Schema]:
    """Get a built-in schema by name."""
    factory = BUILTIN_SCHEMAS.get(name.lower())
    if factory:
        return factory()
    return None


def list_builtin_schemas() -> list[str]:
    """List available built-in schema names."""
    return list(BUILTIN_SCHEMAS.keys())
