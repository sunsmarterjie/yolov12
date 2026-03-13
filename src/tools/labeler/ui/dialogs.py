"""
Dialog windows for project creation, export, and schema editing.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QLabel, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QDialogButtonBox, QColorDialog, QCheckBox, QScrollArea, QWidget,
    QStackedWidget, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QPixmap, QPainter, QPen, QBrush

from core import Schema, KeypointDef, create_poultry_schema, list_builtin_schemas, get_builtin_schema


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("New Project")
        self.setMinimumWidth(500)
        
        self._config = {}
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Project settings
        form = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., poultry-dataset-v1")
        form.addRow("Project Name:", self.name_edit)
        
        # Project directory
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_edit)
        
        self.btn_dir = QPushButton("Browse...")
        self.btn_dir.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.btn_dir)
        
        form.addRow("Project Folder:", dir_layout)
        
        # Image folder
        img_layout = QHBoxLayout()
        self.img_edit = QLineEdit()
        self.img_edit.setReadOnly(True)
        img_layout.addWidget(self.img_edit)
        
        self.btn_img = QPushButton("Browse...")
        self.btn_img.clicked.connect(self._browse_images)
        img_layout.addWidget(self.btn_img)
        
        form.addRow("Images Folder:", img_layout)
        
        layout.addLayout(form)
        
        # Schema selection
        schema_group = QGroupBox("Keypoint Schema")
        schema_layout = QVBoxLayout(schema_group)
        
        self.schema_combo = QComboBox()
        self.schema_combo.addItem("Poultry (Built-in)", "poultry")
        self.schema_combo.addItem("Load from file...", "custom")
        self.schema_combo.currentIndexChanged.connect(self._on_schema_changed)
        schema_layout.addWidget(self.schema_combo)
        
        # Custom schema file
        custom_layout = QHBoxLayout()
        self.schema_edit = QLineEdit()
        self.schema_edit.setReadOnly(True)
        self.schema_edit.setEnabled(False)
        custom_layout.addWidget(self.schema_edit)
        
        self.btn_schema = QPushButton("Browse...")
        self.btn_schema.clicked.connect(self._browse_schema)
        self.btn_schema.setEnabled(False)
        custom_layout.addWidget(self.btn_schema)
        
        schema_layout.addLayout(custom_layout)
        
        # Schema preview
        self.schema_preview = QLabel()
        self.schema_preview.setStyleSheet("color: gray; font-size: 11px;")
        self._update_schema_preview()
        schema_layout.addWidget(self.schema_preview)
        
        layout.addWidget(schema_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if path:
            self.dir_edit.setText(path)
    
    def _browse_images(self):
        path = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if path:
            self.img_edit.setText(path)
            
            # Count images
            img_path = Path(path)
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            count = sum(1 for f in img_path.iterdir() if f.suffix.lower() in extensions)
            
            self.schema_preview.setText(f"Found {count} images")
    
    def _browse_schema(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Schema File",
            "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.schema_edit.setText(path)
            self._update_schema_preview()
    
    def _on_schema_changed(self, index: int):
        is_custom = self.schema_combo.currentData() == "custom"
        self.schema_edit.setEnabled(is_custom)
        self.btn_schema.setEnabled(is_custom)
        self._update_schema_preview()
    
    def _update_schema_preview(self):
        schema_type = self.schema_combo.currentData()
        
        if schema_type == "poultry":
            schema = create_poultry_schema()
            self.schema_preview.setText(
                f"10 keypoints: {', '.join(schema.get_keypoint_names()[:5])}..."
            )
        elif schema_type == "custom" and self.schema_edit.text():
            try:
                schema = Schema.from_yaml(Path(self.schema_edit.text()))
                self.schema_preview.setText(
                    f"{len(schema.keypoints)} keypoints"
                )
            except Exception as e:
                self.schema_preview.setText(f"Error: {str(e)}")
    
    def _validate_and_accept(self):
        errors = []
        
        if not self.name_edit.text().strip():
            errors.append("Project name is required")
        
        if not self.dir_edit.text():
            errors.append("Project folder is required")
        
        if not self.img_edit.text():
            errors.append("Images folder is required")
        elif not Path(self.img_edit.text()).exists():
            errors.append("Images folder does not exist")
        
        if self.schema_combo.currentData() == "custom" and not self.schema_edit.text():
            errors.append("Schema file is required for custom schema")
        
        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(errors))
            return
        
        # Build config
        self._config = {
            "name": self.name_edit.text().strip(),
            "project_dir": self.dir_edit.text(),
            "image_folder": self.img_edit.text(),
        }
        
        # Load schema
        if self.schema_combo.currentData() == "poultry":
            self._config["schema"] = create_poultry_schema()
        else:
            self._config["schema"] = Schema.from_yaml(Path(self.schema_edit.text()))
        
        self.accept()
    
    def get_config(self) -> dict:
        return self._config


class ExportDialog(QDialog):
    """Dialog for exporting to YOLO format."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Export to YOLO Format")
        self.setMinimumWidth(450)
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Output directory
        form = QFormLayout()
        
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        dir_layout.addWidget(self.dir_edit)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.btn_browse)
        
        form.addRow("Output Folder:", dir_layout)
        
        # Train/val split
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.99)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(0.8)
        self.split_spin.setDecimals(2)
        form.addRow("Train Split:", self.split_spin)
        
        layout.addLayout(form)
        
        # Info
        info = QLabel(
            "Export will create:\n"
            "  • images/train/ and images/val/\n"
            "  • labels/train/ and labels/val/\n"
            "  • data.yaml for training"
        )
        info.setStyleSheet("color: gray; font-size: 11px; padding: 10px;")
        layout.addWidget(info)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.dir_edit.setText(path)
    
    def _validate_and_accept(self):
        if not self.dir_edit.text():
            QMessageBox.warning(self, "Error", "Output folder is required")
            return
        
        output_path = Path(self.dir_edit.text())
        
        # Check if directory has existing content
        if output_path.exists() and any(output_path.iterdir()):
            reply = QMessageBox.question(
                self, "Directory Not Empty",
                "The output directory is not empty. Files may be overwritten. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.accept()
    
    def get_output_dir(self) -> str:
        return self.dir_edit.text()
    
    def get_train_split(self) -> float:
        return self.split_spin.value()


class SchemaEditorDialog(QDialog):
    """Dialog for editing keypoint schema."""
    
    def __init__(self, schema: Schema, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Edit Schema")
        self.setMinimumSize(600, 500)
        
        self._schema = Schema(
            name=schema.name,
            version=schema.version,
            classes=dict(schema.classes),
            keypoints=[KeypointDef(kp.name, kp.color, kp.index) for kp in schema.keypoints],
            skeleton=list(schema.skeleton),
            flip_pairs=list(schema.flip_pairs)
        )
        
        self._setup_ui()
        self._load_schema()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Schema name
        form = QFormLayout()
        
        self.name_edit = QLineEdit()
        form.addRow("Schema Name:", self.name_edit)
        
        layout.addLayout(form)
        
        # Keypoints table
        kp_group = QGroupBox("Keypoints")
        kp_layout = QVBoxLayout(kp_group)
        
        self.kp_table = QTableWidget()
        self.kp_table.setColumnCount(3)
        self.kp_table.setHorizontalHeaderLabels(["Name", "Color", ""])
        self.kp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.kp_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.kp_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.kp_table.setColumnWidth(1, 80)
        self.kp_table.setColumnWidth(2, 60)
        kp_layout.addWidget(self.kp_table)
        
        # Keypoint buttons
        kp_btn_layout = QHBoxLayout()
        
        self.btn_add_kp = QPushButton("Add Keypoint")
        self.btn_add_kp.clicked.connect(self._add_keypoint)
        kp_btn_layout.addWidget(self.btn_add_kp)
        
        self.btn_remove_kp = QPushButton("Remove Selected")
        self.btn_remove_kp.clicked.connect(self._remove_keypoint)
        kp_btn_layout.addWidget(self.btn_remove_kp)
        
        self.btn_move_up = QPushButton("↑ Move Up")
        self.btn_move_up.clicked.connect(self._move_up)
        kp_btn_layout.addWidget(self.btn_move_up)
        
        self.btn_move_down = QPushButton("↓ Move Down")
        self.btn_move_down.clicked.connect(self._move_down)
        kp_btn_layout.addWidget(self.btn_move_down)
        
        kp_layout.addLayout(kp_btn_layout)
        
        layout.addWidget(kp_group)
        
        # Skeleton connections (simplified - just show count)
        skel_group = QGroupBox("Skeleton Connections")
        skel_layout = QVBoxLayout(skel_group)
        
        self.skeleton_list = QListWidget()
        self.skeleton_list.setMaximumHeight(100)
        skel_layout.addWidget(self.skeleton_list)
        
        self.skel_info = QLabel()
        skel_layout.addWidget(self.skel_info)
        
        layout.addWidget(skel_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_schema(self):
        """Load schema into UI."""
        self.name_edit.setText(self._schema.name)
        
        # Load keypoints
        self.kp_table.setRowCount(len(self._schema.keypoints))
        
        for i, kp in enumerate(self._schema.keypoints):
            # Name
            name_item = QTableWidgetItem(kp.name)
            self.kp_table.setItem(i, 0, name_item)
            
            # Color button
            color_btn = QPushButton()
            color_btn.setStyleSheet(f"background-color: {kp.color};")
            color_btn.setProperty("color", kp.color)
            color_btn.clicked.connect(lambda checked, row=i: self._pick_color(row))
            self.kp_table.setCellWidget(i, 1, color_btn)
            
            # Index display
            idx_item = QTableWidgetItem(str(i))
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.kp_table.setItem(i, 2, idx_item)
        
        # Load skeleton
        self._update_skeleton_display()
    
    def _update_skeleton_display(self):
        """Update skeleton connections display."""
        self.skeleton_list.clear()
        
        for conn in self._schema.skeleton:
            self.skeleton_list.addItem(f"{conn[0]} → {conn[1]}")
        
        self.skel_info.setText(f"{len(self._schema.skeleton)} connections")
    
    def _add_keypoint(self):
        """Add a new keypoint."""
        row = self.kp_table.rowCount()
        self.kp_table.insertRow(row)
        
        name_item = QTableWidgetItem(f"keypoint_{row}")
        self.kp_table.setItem(row, 0, name_item)
        
        color_btn = QPushButton()
        color_btn.setStyleSheet("background-color: #FF0000;")
        color_btn.setProperty("color", "#FF0000")
        color_btn.clicked.connect(lambda: self._pick_color(row))
        self.kp_table.setCellWidget(row, 1, color_btn)
        
        idx_item = QTableWidgetItem(str(row))
        idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.kp_table.setItem(row, 2, idx_item)
    
    def _remove_keypoint(self):
        """Remove selected keypoint."""
        row = self.kp_table.currentRow()
        if row >= 0:
            self.kp_table.removeRow(row)
            self._update_indices()
    
    def _move_up(self):
        """Move selected keypoint up."""
        row = self.kp_table.currentRow()
        if row > 0:
            self._swap_rows(row, row - 1)
            self.kp_table.setCurrentCell(row - 1, 0)
    
    def _move_down(self):
        """Move selected keypoint down."""
        row = self.kp_table.currentRow()
        if row < self.kp_table.rowCount() - 1:
            self._swap_rows(row, row + 1)
            self.kp_table.setCurrentCell(row + 1, 0)
    
    def _swap_rows(self, row1: int, row2: int):
        """Swap two rows in the table."""
        name1 = self.kp_table.item(row1, 0).text()
        name2 = self.kp_table.item(row2, 0).text()
        
        color1 = self.kp_table.cellWidget(row1, 1).property("color")
        color2 = self.kp_table.cellWidget(row2, 1).property("color")
        
        self.kp_table.item(row1, 0).setText(name2)
        self.kp_table.item(row2, 0).setText(name1)
        
        self.kp_table.cellWidget(row1, 1).setProperty("color", color2)
        self.kp_table.cellWidget(row1, 1).setStyleSheet(f"background-color: {color2};")
        
        self.kp_table.cellWidget(row2, 1).setProperty("color", color1)
        self.kp_table.cellWidget(row2, 1).setStyleSheet(f"background-color: {color1};")
    
    def _update_indices(self):
        """Update index column after changes."""
        for i in range(self.kp_table.rowCount()):
            self.kp_table.item(i, 2).setText(str(i))
    
    def _pick_color(self, row: int):
        """Open color picker for a keypoint."""
        btn = self.kp_table.cellWidget(row, 1)
        current = QColor(btn.property("color"))
        
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            btn.setProperty("color", color.name())
            btn.setStyleSheet(f"background-color: {color.name()};")
    
    def _save_and_accept(self):
        """Save schema and accept."""
        # Build keypoints list
        keypoints = []
        for i in range(self.kp_table.rowCount()):
            name = self.kp_table.item(i, 0).text()
            color = self.kp_table.cellWidget(i, 1).property("color")
            keypoints.append(KeypointDef(name=name, color=color, index=i))
        
        self._schema.name = self.name_edit.text()
        self._schema.keypoints = keypoints
        
        # Update skeleton - remove connections with invalid keypoints
        valid_names = {kp.name for kp in keypoints}
        self._schema.skeleton = [
            conn for conn in self._schema.skeleton
            if conn[0] in valid_names and conn[1] in valid_names
        ]
        
        self.accept()
    
    def get_schema(self) -> Schema:
        return self._schema


class TutorialDialog(QDialog):
    """Step-by-step tutorial dialog for new users."""

    STEPS = [
        {
            "title": "Welcome to PoseLabeler",
            "content": (
                "<h2>Welcome to PoseLabeler!</h2>"
                "<p>This tool lets you annotate poultry images with <b>bounding boxes</b> "
                "and <b>keypoints</b> to create pose estimation datasets.</p>"
                "<p><b>Layout:</b></p>"
                "<ul>"
                "<li><b>Left panel</b> — Instance list (your hens)</li>"
                "<li><b>Center</b> — Image canvas (click here to annotate)</li>"
                "<li><b>Right panel</b> — Keypoint list and frame navigation</li>"
                "<li><b>Toolbar</b> — Tool selection and quick actions</li>"
                "</ul>"
                "<p>Click <b>Next</b> to learn the labeling workflow.</p>"
            ),
        },
        {
            "title": "Step 1 — Add a Hen",
            "content": (
                "<h2>Step 1: Add a Hen</h2>"
                "<p>Press <b>N</b> or click <b>\"+ Add Hen (N)\"</b> in the toolbar.</p>"
                "<p>A new hen entry appears in the left panel and is automatically selected.</p>"
                "<p><b>Tips:</b></p>"
                "<ul>"
                "<li>You can add multiple hens per image</li>"
                "<li>Click a hen in the left panel to select it</li>"
                "<li>Toggle the eye icon to show/hide a hen's annotations</li>"
                "</ul>"
            ),
        },
        {
            "title": "Step 2 — Draw Bounding Box",
            "content": (
                "<h2>Step 2: Draw a Bounding Box</h2>"
                "<p>After adding a hen, the tool automatically switches to <b>Draw Box (B)</b>.</p>"
                "<p><b>Click and drag</b> on the canvas to draw a rectangle around the hen.</p>"
                "<p>Release the mouse to confirm the box.</p>"
                "<p><b>Tips:</b></p>"
                "<ul>"
                "<li>Press <b>B</b> at any time to switch to the bbox tool</li>"
                "<li>After drawing, the tool automatically switches to keypoint mode</li>"
                "<li>Use <b>Ctrl+Z</b> to undo if you drew the wrong box</li>"
                "</ul>"
            ),
        },
        {
            "title": "Step 3 — Place Keypoints",
            "content": (
                "<h2>Step 3: Place Keypoints</h2>"
                "<p>After the bbox, the tool switches to <b>Place Point (K)</b> and selects "
                "the first keypoint automatically.</p>"
                "<p><b>Click on the image</b> to place each keypoint at the correct body part location.</p>"
                "<p>After placing a point, the next keypoint is automatically selected.</p>"
                "<p><b>Tips:</b></p>"
                "<ul>"
                "<li>Press <b>1–0</b> to jump to a specific keypoint</li>"
                "<li>Press <b>K</b> to manually switch to keypoint mode</li>"
                "<li>Press <b>V</b> then click a point to drag and reposition it</li>"
                "<li>Use <b>Ctrl+Z</b> to undo a misplaced point</li>"
                "</ul>"
            ),
        },
        {
            "title": "Step 4 — Fix Mistakes",
            "content": (
                "<h2>Step 4: Fixing Mistakes</h2>"
                "<p>Made an error? Here are your options:</p>"
                "<p><b>Eraser Tool (E)</b> — Press E to activate. "
                "Click any placed keypoint to delete it. The cursor becomes a red circle.</p>"
                "<p><b>Delete / Backspace</b> — Select a keypoint first (press V, click the point), "
                "then press Delete or Backspace to remove it.</p>"
                "<p><b>Ctrl+Z</b> — Undo the last action (supports up to 50 steps).</p>"
                "<p><b>Right-click a keypoint</b> — Cycles between: "
                "<i>visible</i> → <i>occluded</i> → <i>unlabeled</i>.</p>"
            ),
        },
        {
            "title": "Step 5 — Navigate & Save",
            "content": (
                "<h2>Step 5: Navigate Frames &amp; Save</h2>"
                "<p><b>A / D</b> — Move to previous / next frame.</p>"
                "<p><b>Ctrl+S</b> — Save your project at any time.</p>"
                "<p><b>Ctrl+E</b> — Export annotations to YOLO pose format.</p>"
                "<p>The status bar at the bottom shows your annotation progress.</p>"
                "<p><b>The project autosaves every 30 seconds.</b></p>"
                "<hr>"
                "<p>You're ready to start labeling! Press <b>Close</b> to begin.</p>"
                "<p><i>You can reopen this tutorial at any time from Help → Tutorial (F1).</i></p>"
            ),
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PoseLabeler Tutorial")
        self.setMinimumSize(560, 420)
        self.setMaximumWidth(700)

        self._current_step = 0
        self._setup_ui()
        self._update_step()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Step indicator
        self._step_label = QLabel()
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(9)
        self._step_label.setFont(font)
        self._step_label.setStyleSheet("color: #888;")
        layout.addWidget(self._step_label)

        # Content area (stacked pages)
        self._stack = QStackedWidget()
        for step in self.STEPS:
            page = QLabel(step["content"])
            page.setWordWrap(True)
            page.setTextFormat(Qt.TextFormat.RichText)
            page.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            page.setContentsMargins(16, 8, 16, 8)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._stack.addWidget(scroll)
        layout.addWidget(self._stack, stretch=1)

        # Navigation buttons
        btn_row = QHBoxLayout()
        self._btn_prev = QPushButton("← Back")
        self._btn_prev.clicked.connect(self._prev_step)
        btn_row.addWidget(self._btn_prev)

        btn_row.addStretch()

        self._btn_next = QPushButton("Next →")
        self._btn_next.setDefault(True)
        self._btn_next.clicked.connect(self._next_step)
        btn_row.addWidget(self._btn_next)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_close)

        layout.addLayout(btn_row)

    def _update_step(self):
        total = len(self.STEPS)
        self._stack.setCurrentIndex(self._current_step)
        self._step_label.setText(
            f"Step {self._current_step + 1} of {total}  —  {self.STEPS[self._current_step]['title']}"
        )
        self._btn_prev.setEnabled(self._current_step > 0)
        is_last = self._current_step == total - 1
        self._btn_next.setVisible(not is_last)
        self._btn_close.setText("Close" if is_last else "Skip Tutorial")

    def _next_step(self):
        if self._current_step < len(self.STEPS) - 1:
            self._current_step += 1
            self._update_step()

    def _prev_step(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step()
