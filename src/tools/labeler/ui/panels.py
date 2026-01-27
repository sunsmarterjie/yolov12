"""
Side panel widgets for instance management, keypoint selection, and navigation.
"""

from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSlider, QSpinBox, QGroupBox, QCheckBox,
    QScrollArea, QFrame, QComboBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush

from core import FrameAnnotation, Instance, Keypoint, Schema, Project


class ColorSwatch(QWidget):
    """Small colored square widget."""
    
    def __init__(self, color: str, size: int = 16, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(size, size)
    
    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 3, 3)


class InstancePanel(QWidget):
    """Panel for managing instances (hens) in the current frame."""
    
    instance_selected = pyqtSignal(str)  # instance_id
    instance_added = pyqtSignal(object)  # Instance
    instance_removed = pyqtSignal(str)  # instance_id
    instance_visibility_changed = pyqtSignal(str, bool)  # instance_id, visible
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._frame: Optional[FrameAnnotation] = None
        self._schema: Optional[Schema] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("<b>Instances</b>")
        layout.addWidget(header)
        
        # Instance list
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("+ Add Hen")
        self.btn_add.clicked.connect(self.add_instance)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.delete_selected)
        btn_layout.addWidget(self.btn_remove)
        
        layout.addLayout(btn_layout)
        
        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.info_label)
    
    def set_frame(self, frame: FrameAnnotation, schema: Schema) -> None:
        """Set the current frame and schema."""
        self._frame = frame
        self._schema = schema
        self._refresh_list()
    
    def _refresh_list(self) -> None:
        """Refresh the instance list."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        
        if not self._frame:
            self.list_widget.blockSignals(False)
            return
        
        for i, instance in enumerate(self._frame.instances):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, instance.id)
            item.setCheckState(
                Qt.CheckState.Checked if instance.visible else Qt.CheckState.Unchecked
            )
            
            # Count labeled keypoints
            labeled = len([kp for kp in instance.keypoints.values() if kp.is_labeled()])
            total = len(self._schema.keypoints) if self._schema else 0
            
            text = f"Hen #{i + 1} ({labeled}/{total} pts)"
            if instance.bbox.is_valid():
                text += " ✓"
            
            item.setText(text)
            self.list_widget.addItem(item)
        
        self.list_widget.blockSignals(False)
        
        self.info_label.setText(f"{len(self._frame.instances)} instance(s)")
    
    def add_instance(self) -> None:
        """Add a new instance."""
        if not self._frame or not self._schema:
            return
        
        instance = Instance(
            class_id=0,
            class_name=list(self._schema.classes.values())[0] if self._schema.classes else "object"
        )
        
        # Initialize keypoints
        for kp_def in self._schema.keypoints:
            instance.keypoints[kp_def.name] = Keypoint(name=kp_def.name)
        
        self._frame.add_instance(instance)
        self._refresh_list()
        
        # Select the new instance
        self.list_widget.setCurrentRow(len(self._frame.instances) - 1)
        
        self.instance_added.emit(instance)
    
    def delete_selected(self) -> None:
        """Delete the selected instance."""
        item = self.list_widget.currentItem()
        if not item or not self._frame:
            return
        
        instance_id = item.data(Qt.ItemDataRole.UserRole)
        self._frame.remove_instance(instance_id)
        self._refresh_list()
        
        self.instance_removed.emit(instance_id)
    
    def refresh(self) -> None:
        """Refresh the panel."""
        self._refresh_list()

    def select_instance(self, instance_id: str) -> None:
        """Select an instance by ID."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == instance_id:
                self.list_widget.setCurrentItem(item)
                break

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle instance selection."""
        instance_id = item.data(Qt.ItemDataRole.UserRole)
        self.instance_selected.emit(instance_id)
    
    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Handle visibility checkbox change."""
        instance_id = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        
        if self._frame:
            instance = self._frame.get_instance(instance_id)
            if instance:
                instance.visible = visible
        
        self.instance_visibility_changed.emit(instance_id, visible)


class KeypointPanel(QWidget):
    """Panel for selecting and managing keypoints."""
    
    keypoint_selected = pyqtSignal(str)  # keypoint_name
    visibility_changed = pyqtSignal(str, int)  # keypoint_name, visibility
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._schema: Optional[Schema] = None
        self._instance: Optional[Instance] = None
        self._current_index = 0
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("<b>Keypoints</b>")
        layout.addWidget(header)
        
        # Scroll area for keypoint list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.keypoints_widget = QWidget()
        self.keypoints_layout = QVBoxLayout(self.keypoints_widget)
        self.keypoints_layout.setContentsMargins(0, 0, 0, 0)
        self.keypoints_layout.setSpacing(2)
        self.keypoints_layout.addStretch()
        
        scroll.setWidget(self.keypoints_widget)
        layout.addWidget(scroll)
        
        # Current keypoint info
        self.current_label = QLabel("Select an instance first")
        self.current_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.current_label)
        
        # Visibility controls
        vis_group = QGroupBox("Visibility")
        vis_layout = QHBoxLayout(vis_group)
        
        self.btn_visible = QPushButton("Visible")
        self.btn_visible.setCheckable(True)
        self.btn_visible.clicked.connect(lambda: self._set_visibility(2))
        vis_layout.addWidget(self.btn_visible)
        
        self.btn_occluded = QPushButton("Occluded")
        self.btn_occluded.setCheckable(True)
        self.btn_occluded.clicked.connect(lambda: self._set_visibility(1))
        vis_layout.addWidget(self.btn_occluded)
        
        layout.addWidget(vis_group)
        
        self.keypoint_items: list[QWidget] = []
    
    def set_schema(self, schema: Schema) -> None:
        """Set the keypoint schema."""
        self._schema = schema
        self._rebuild_keypoint_list()
    
    def set_instance(self, instance: Optional[Instance]) -> None:
        """Set the current instance."""
        self._instance = instance
        self._update_keypoint_states()
        
        if instance:
            self.current_label.setText(f"Instance: {instance.id[:8]}")
        else:
            self.current_label.setText("Select an instance first")
    
    def _rebuild_keypoint_list(self) -> None:
        """Rebuild the keypoint list from schema."""
        # Clear existing
        for item in self.keypoint_items:
            item.deleteLater()
        self.keypoint_items.clear()
        
        if not self._schema:
            return
        
        colors = self._schema.get_keypoint_colors()
        
        for i, kp_def in enumerate(self._schema.keypoints):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(4, 2, 4, 2)
            item_layout.setSpacing(4)
            
            # Color swatch
            swatch = ColorSwatch(kp_def.color)
            item_layout.addWidget(swatch)
            
            # Keypoint button
            btn = QPushButton(f"{i + 1}. {kp_def.name}")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            btn.setProperty("kp_name", kp_def.name)
            btn.setProperty("kp_index", i)
            btn.clicked.connect(lambda checked, name=kp_def.name: self._on_keypoint_clicked(name))
            item_layout.addWidget(btn)
            
            # Status indicator
            status = QLabel("○")
            status.setFixedWidth(20)
            status.setProperty("kp_name", kp_def.name)
            item_layout.addWidget(status)
            
            # Store references
            item_widget.setProperty("btn", btn)
            item_widget.setProperty("status", status)
            item_widget.setProperty("kp_name", kp_def.name)
            
            self.keypoint_items.append(item_widget)
            self.keypoints_layout.insertWidget(i, item_widget)
    
    def _update_keypoint_states(self) -> None:
        """Update keypoint status indicators based on instance."""
        for item in self.keypoint_items:
            btn = item.property("btn")
            status = item.property("status")
            kp_name = item.property("kp_name")
            
            if self._instance and kp_name in self._instance.keypoints:
                kp = self._instance.keypoints[kp_name]
                if kp.visibility == 2:
                    status.setText("●")
                    status.setStyleSheet("color: #00FF00;")
                elif kp.visibility == 1:
                    status.setText("◐")
                    status.setStyleSheet("color: #FFAA00;")
                else:
                    status.setText("○")
                    status.setStyleSheet("color: #888888;")
            else:
                status.setText("○")
                status.setStyleSheet("color: #888888;")
        
        self._update_visibility_buttons()
    
    def _update_visibility_buttons(self) -> None:
        """Update visibility button states."""
        if not self._instance or not self._schema:
            self.btn_visible.setChecked(False)
            self.btn_occluded.setChecked(False)
            return
        
        kp_name = self._schema.keypoints[self._current_index].name if self._current_index < len(self._schema.keypoints) else None
        
        if kp_name and kp_name in self._instance.keypoints:
            kp = self._instance.keypoints[kp_name]
            self.btn_visible.setChecked(kp.visibility == 2)
            self.btn_occluded.setChecked(kp.visibility == 1)
        else:
            self.btn_visible.setChecked(False)
            self.btn_occluded.setChecked(False)
    
    def _on_keypoint_clicked(self, name: str) -> None:
        """Handle keypoint selection."""
        # Update button states
        for item in self.keypoint_items:
            btn = item.property("btn")
            btn.setChecked(btn.property("kp_name") == name)
            if btn.property("kp_name") == name:
                self._current_index = btn.property("kp_index")
        
        self._update_visibility_buttons()
        self.keypoint_selected.emit(name)
    
    def _set_visibility(self, visibility: int) -> None:
        """Set visibility of current keypoint."""
        if not self._instance or not self._schema:
            return
        
        if self._current_index >= len(self._schema.keypoints):
            return
        
        kp_name = self._schema.keypoints[self._current_index].name
        
        if kp_name in self._instance.keypoints:
            self._instance.keypoints[kp_name].visibility = visibility
            self._update_keypoint_states()
            self.visibility_changed.emit(kp_name, visibility)
    
    def select_by_index(self, index: int) -> None:
        """Select a keypoint by index."""
        if not self._schema or index >= len(self._schema.keypoints):
            return
        
        kp_name = self._schema.keypoints[index].name
        self._on_keypoint_clicked(kp_name)
    
    def select_next(self) -> None:
        """Select the next keypoint."""
        if not self._schema:
            return
        
        next_index = (self._current_index + 1) % len(self._schema.keypoints)
        self.select_by_index(next_index)
    
    def select_first(self) -> None:
        """Select the first keypoint."""
        self.select_by_index(0)

    def select_keypoint(self, keypoint_name: str) -> None:
        """Select a keypoint by name."""
        self._on_keypoint_clicked(keypoint_name)

    def refresh(self) -> None:
        """Refresh the panel."""
        self._update_keypoint_states()


class NavigationPanel(QWidget):
    """Panel for frame navigation."""
    
    frame_changed = pyqtSignal(int)  # frame_index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._project: Optional[Project] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("<b>Navigation</b>")
        layout.addWidget(header)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider)
        
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.valueChanged.connect(self._on_spin_changed)
        slider_layout.addWidget(self.frame_spin)
        
        layout.addLayout(slider_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("← Previous (A)")
        self.btn_prev.clicked.connect(self._prev_frame)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next → (D)")
        self.btn_next.clicked.connect(self._next_frame)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        # Frame info
        self.frame_info = QLabel("No frames")
        self.frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_info)
        
        # Quick jump
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Jump to:"))
        
        self.jump_combo = QComboBox()
        self.jump_combo.addItems(["First", "Last", "First Unlabeled", "Next Unlabeled"])
        jump_layout.addWidget(self.jump_combo)
        
        self.btn_jump = QPushButton("Go")
        self.btn_jump.clicked.connect(self._do_jump)
        jump_layout.addWidget(self.btn_jump)
        
        layout.addLayout(jump_layout)
        
        layout.addStretch()
    
    def set_project(self, project: Project) -> None:
        """Set the current project."""
        self._project = project
        self._update_controls()
    
    def _update_controls(self) -> None:
        """Update navigation controls."""
        if not self._project:
            self.slider.setMaximum(0)
            self.frame_spin.setMaximum(1)
            self.frame_info.setText("No frames")
            return
        
        total = len(self._project.image_files)
        current = self._project.current_frame_index
        
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, total - 1))
        self.slider.setValue(current)
        self.slider.blockSignals(False)
        
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(1, total))
        self.frame_spin.setValue(current + 1)
        self.frame_spin.blockSignals(False)
        
        # Update info
        annotated = sum(1 for f in self._project.frames.values() if f.instances)
        self.frame_info.setText(f"Frame {current + 1} of {total} ({annotated} annotated)")
    
    def _on_slider_changed(self, value: int) -> None:
        """Handle slider change."""
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value + 1)
        self.frame_spin.blockSignals(False)
        
        self.frame_changed.emit(value)
    
    def _on_spin_changed(self, value: int) -> None:
        """Handle spinbox change."""
        self.slider.blockSignals(True)
        self.slider.setValue(value - 1)
        self.slider.blockSignals(False)
        
        self.frame_changed.emit(value - 1)
    
    def _prev_frame(self) -> None:
        """Go to previous frame."""
        if self._project and self._project.current_frame_index > 0:
            self.frame_changed.emit(self._project.current_frame_index - 1)
    
    def _next_frame(self) -> None:
        """Go to next frame."""
        if self._project and self._project.current_frame_index < len(self._project.image_files) - 1:
            self.frame_changed.emit(self._project.current_frame_index + 1)
    
    def _do_jump(self) -> None:
        """Perform jump based on combo selection."""
        if not self._project:
            return
        
        action = self.jump_combo.currentText()
        
        if action == "First":
            self.frame_changed.emit(0)
        
        elif action == "Last":
            self.frame_changed.emit(len(self._project.image_files) - 1)
        
        elif action == "First Unlabeled":
            for i, img_path in enumerate(self._project.image_files):
                key = str(img_path)
                if key not in self._project.frames or not self._project.frames[key].instances:
                    self.frame_changed.emit(i)
                    return
        
        elif action == "Next Unlabeled":
            start = self._project.current_frame_index + 1
            for i in range(start, len(self._project.image_files)):
                key = str(self._project.image_files[i])
                if key not in self._project.frames or not self._project.frames[key].instances:
                    self.frame_changed.emit(i)
                    return
            # Wrap around
            for i in range(0, start):
                key = str(self._project.image_files[i])
                if key not in self._project.frames or not self._project.frames[key].instances:
                    self.frame_changed.emit(i)
                    return
