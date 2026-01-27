"""
Main application window for PoseLabeler.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QInputDialog, QLabel, QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QAction, QKeySequence, QIcon

from core import Project, ProjectManager, Schema, create_poultry_schema, get_builtin_schema
from .canvas import AnnotationCanvas
from .panels import InstancePanel, KeypointPanel, NavigationPanel
from .dialogs import NewProjectDialog, ExportDialog, SchemaEditorDialog


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.project_manager = ProjectManager()
        self.settings = QSettings("PoultryVision", "PoseLabeler")
        
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_shortcuts()
        self._setup_autosave_timer()
        
        self._load_settings()
        self._update_title()
        self._update_ui_state()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("PoseLabeler")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left panel: Instance list
        self.instance_panel = InstancePanel()
        self.instance_panel.setMinimumWidth(200)
        self.instance_panel.setMaximumWidth(350)
        self.splitter.addWidget(self.instance_panel)
        
        # Center: Canvas
        self.canvas = AnnotationCanvas()
        self.splitter.addWidget(self.canvas)
        
        # Right panel: Keypoints + Navigation
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        
        self.keypoint_panel = KeypointPanel()
        self.navigation_panel = NavigationPanel()
        
        right_layout.addWidget(self.keypoint_panel, stretch=2)
        right_layout.addWidget(self.navigation_panel, stretch=1)
        
        right_panel.setMinimumWidth(220)
        right_panel.setMaximumWidth(350)
        self.splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        self.splitter.setSizes([250, 700, 250])
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect panel and canvas signals."""
        # Instance panel signals
        self.instance_panel.instance_selected.connect(self._on_instance_selected)
        self.instance_panel.instance_added.connect(self._on_instance_added)
        self.instance_panel.instance_removed.connect(self._on_instance_removed)
        self.instance_panel.instance_visibility_changed.connect(self._on_instance_visibility_changed)
        
        # Keypoint panel signals
        self.keypoint_panel.keypoint_selected.connect(self._on_keypoint_selected)
        self.keypoint_panel.visibility_changed.connect(self._on_keypoint_visibility_changed)
        
        # Navigation panel signals
        self.navigation_panel.frame_changed.connect(self._on_frame_changed)
        
        # Canvas signals
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        self.canvas.keypoint_placed.connect(self._on_keypoint_placed)
        self.canvas.keypoint_clicked.connect(self._on_keypoint_clicked)
        self.canvas.bbox_drawn.connect(self._on_bbox_drawn)
    
    def _setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        self.action_new = QAction("&New Project...", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.triggered.connect(self._new_project)
        file_menu.addAction(self.action_new)
        
        self.action_open = QAction("&Open Project...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._open_project)
        file_menu.addAction(self.action_open)
        
        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_project)
        file_menu.addAction(self.action_save)
        
        file_menu.addSeparator()
        
        self.action_export = QAction("&Export to YOLO...", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.triggered.connect(self._export_yolo)
        file_menu.addAction(self.action_export)
        
        file_menu.addSeparator()
        
        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        self.action_edit_schema = QAction("Edit &Schema...", self)
        self.action_edit_schema.triggered.connect(self._edit_schema)
        edit_menu.addAction(self.action_edit_schema)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.action_zoom_in = QAction("Zoom &In", self)
        self.action_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        self.action_zoom_in.triggered.connect(lambda: self.canvas.zoom(1.2))
        view_menu.addAction(self.action_zoom_in)
        
        self.action_zoom_out = QAction("Zoom &Out", self)
        self.action_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        self.action_zoom_out.triggered.connect(lambda: self.canvas.zoom(0.8))
        view_menu.addAction(self.action_zoom_out)
        
        self.action_zoom_fit = QAction("&Fit to Window", self)
        self.action_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.action_zoom_fit.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(self.action_zoom_fit)
        
        view_menu.addSeparator()
        
        self.action_show_skeleton = QAction("Show &Skeleton", self)
        self.action_show_skeleton.setCheckable(True)
        self.action_show_skeleton.setChecked(True)
        self.action_show_skeleton.triggered.connect(self._toggle_skeleton)
        view_menu.addAction(self.action_show_skeleton)
        
        self.action_show_bbox = QAction("Show &Bounding Boxes", self)
        self.action_show_bbox.setCheckable(True)
        self.action_show_bbox.setChecked(True)
        self.action_show_bbox.triggered.connect(self._toggle_bbox)
        view_menu.addAction(self.action_show_bbox)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        self.action_shortcuts = QAction("Keyboard &Shortcuts", self)
        self.action_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(self.action_shortcuts)
        
        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
    
    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Tool selection
        self.action_tool_select = QAction("Select (V)", self)
        self.action_tool_select.setCheckable(True)
        self.action_tool_select.setChecked(True)
        self.action_tool_select.triggered.connect(lambda: self._set_tool("select"))
        toolbar.addAction(self.action_tool_select)
        
        self.action_tool_bbox = QAction("Draw Box (B)", self)
        self.action_tool_bbox.setCheckable(True)
        self.action_tool_bbox.triggered.connect(lambda: self._set_tool("bbox"))
        toolbar.addAction(self.action_tool_bbox)
        
        self.action_tool_keypoint = QAction("Place Point (K)", self)
        self.action_tool_keypoint.setCheckable(True)
        self.action_tool_keypoint.triggered.connect(lambda: self._set_tool("keypoint"))
        toolbar.addAction(self.action_tool_keypoint)
        
        toolbar.addSeparator()
        
        # Navigation
        self.action_prev_frame = QAction("← Prev (A)", self)
        self.action_prev_frame.triggered.connect(self._prev_frame)
        toolbar.addAction(self.action_prev_frame)
        
        self.action_next_frame = QAction("Next → (D)", self)
        self.action_next_frame.triggered.connect(self._next_frame)
        toolbar.addAction(self.action_next_frame)
        
        toolbar.addSeparator()
        
        # Instance actions
        self.action_add_instance = QAction("+ Add Hen (N)", self)
        self.action_add_instance.triggered.connect(self._add_instance)
        toolbar.addAction(self.action_add_instance)
    
    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Frame counter
        self.frame_label = QLabel("No project")
        self.statusbar.addWidget(self.frame_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # Autosave indicator
        self.autosave_label = QLabel("")
        self.statusbar.addPermanentWidget(self.autosave_label)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Navigation
        self.shortcut_next = QAction(self)
        self.shortcut_next.setShortcut(QKeySequence("D"))
        self.shortcut_next.triggered.connect(self._next_frame)
        self.addAction(self.shortcut_next)
        
        self.shortcut_prev = QAction(self)
        self.shortcut_prev.setShortcut(QKeySequence("A"))
        self.shortcut_prev.triggered.connect(self._prev_frame)
        self.addAction(self.shortcut_prev)
        
        # Tools
        self.shortcut_select = QAction(self)
        self.shortcut_select.setShortcut(QKeySequence("V"))
        self.shortcut_select.triggered.connect(lambda: self._set_tool("select"))
        self.addAction(self.shortcut_select)
        
        self.shortcut_bbox = QAction(self)
        self.shortcut_bbox.setShortcut(QKeySequence("B"))
        self.shortcut_bbox.triggered.connect(lambda: self._set_tool("bbox"))
        self.addAction(self.shortcut_bbox)
        
        self.shortcut_keypoint = QAction(self)
        self.shortcut_keypoint.setShortcut(QKeySequence("K"))
        self.shortcut_keypoint.triggered.connect(lambda: self._set_tool("keypoint"))
        self.addAction(self.shortcut_keypoint)
        
        # Instance
        self.shortcut_new_instance = QAction(self)
        self.shortcut_new_instance.setShortcut(QKeySequence("N"))
        self.shortcut_new_instance.triggered.connect(self._add_instance)
        self.addAction(self.shortcut_new_instance)
        
        # Delete
        self.shortcut_delete = QAction(self)
        self.shortcut_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self.shortcut_delete.triggered.connect(self._delete_selected)
        self.addAction(self.shortcut_delete)
        
        # Keypoint number shortcuts (1-0 for first 10 keypoints)
        for i in range(10):
            action = QAction(self)
            action.setShortcut(QKeySequence(str((i + 1) % 10)))  # 1,2,3...9,0
            action.triggered.connect(lambda checked, idx=i: self._select_keypoint_by_index(idx))
            self.addAction(action)
    
    def _setup_autosave_timer(self):
        """Setup autosave timer."""
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start(30000)  # Check every 30 seconds
    
    def _load_settings(self):
        """Load application settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_settings(self):
        """Save application settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def _update_title(self):
        """Update window title."""
        project = self.project_manager.current_project
        if project:
            modified = "*" if self.project_manager.has_unsaved_changes() else ""
            self.setWindowTitle(f"PoseLabeler - {project.name}{modified}")
        else:
            self.setWindowTitle("PoseLabeler")
    
    def _update_ui_state(self):
        """Update UI enabled states based on project."""
        has_project = self.project_manager.current_project is not None
        
        self.action_save.setEnabled(has_project)
        self.action_export.setEnabled(has_project)
        self.action_edit_schema.setEnabled(has_project)
        
        self.canvas.setEnabled(has_project)
        self.instance_panel.setEnabled(has_project)
        self.keypoint_panel.setEnabled(has_project)
        self.navigation_panel.setEnabled(has_project)
        
        self._update_status()
    
    def _update_status(self):
        """Update status bar."""
        project = self.project_manager.current_project
        if project:
            current, total, annotated = project.get_progress()
            self.frame_label.setText(f"Frame {current + 1} / {total}")
            
            if total > 0:
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(annotated)
                self.progress_bar.setFormat(f"{annotated}/{total} annotated")
            else:
                self.progress_bar.setMaximum(1)
                self.progress_bar.setValue(0)
        else:
            self.frame_label.setText("No project")
            self.progress_bar.setValue(0)
    
    def _load_current_frame(self):
        """Load the current frame into the canvas and panels."""
        project = self.project_manager.current_project
        if not project:
            return
        
        frame = project.get_current_frame()
        if not frame:
            return
        
        # Load image into canvas
        self.canvas.load_image(Path(frame.image_path))
        self.canvas.set_frame(frame)
        self.canvas.set_schema(project.schema)
        
        # Update panels
        self.instance_panel.set_frame(frame, project.schema)
        self.keypoint_panel.set_schema(project.schema)
        self.navigation_panel.set_project(project)
        
        self._update_status()
    
    # ===== Actions =====
    
    def _new_project(self):
        """Create a new project."""
        dialog = NewProjectDialog(self)
        if dialog.exec():
            config = dialog.get_config()
            
            try:
                project = self.project_manager.create_project(
                    name=config["name"],
                    project_dir=Path(config["project_dir"]),
                    image_folder=Path(config["image_folder"]),
                    schema=config.get("schema")
                )
                
                self._load_current_frame()
                self._update_title()
                self._update_ui_state()
                
                self.statusbar.showMessage(f"Created project: {config['name']}", 3000)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create project:\n{str(e)}")
    
    def _open_project(self):
        """Open an existing project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "PoseLabeler Projects (*.poselabel);;All Files (*)"
        )
        
        if path:
            try:
                self.project_manager.open_project(Path(path))
                self._load_current_frame()
                self._update_title()
                self._update_ui_state()
                
                self.statusbar.showMessage(f"Opened project: {path}", 3000)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project:\n{str(e)}")
    
    def _save_project(self):
        """Save the current project."""
        if self.project_manager.save_project():
            self._update_title()
            self.statusbar.showMessage("Project saved", 2000)
            self.autosave_label.setText("Saved")
    
    def _autosave(self):
        """Perform autosave if needed."""
        if self.project_manager.autosave_if_needed():
            self._update_title()
            self.autosave_label.setText("Autosaved")
        else:
            self.autosave_label.setText("")
    
    def _export_yolo(self):
        """Export to YOLO format."""
        project = self.project_manager.current_project
        if not project:
            return
        
        dialog = ExportDialog(self)
        if dialog.exec():
            output_dir = dialog.get_output_dir()
            train_split = dialog.get_train_split()
            
            try:
                stats = project.export_yolo(Path(output_dir), train_split)
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported successfully!\n\n"
                    f"Total frames: {stats['total_frames']}\n"
                    f"Training: {stats['train_frames']}\n"
                    f"Validation: {stats['val_frames']}\n"
                    f"Total instances: {stats['total_instances']}\n\n"
                    f"Output: {output_dir}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
    
    def _edit_schema(self):
        """Open schema editor dialog."""
        project = self.project_manager.current_project
        if not project:
            return
        
        dialog = SchemaEditorDialog(project.schema, self)
        if dialog.exec():
            project.schema = dialog.get_schema()
            self.project_manager.mark_changed()
            self._load_current_frame()
    
    def _set_tool(self, tool: str):
        """Set the current tool."""
        self.action_tool_select.setChecked(tool == "select")
        self.action_tool_bbox.setChecked(tool == "bbox")
        self.action_tool_keypoint.setChecked(tool == "keypoint")
        
        self.canvas.set_tool(tool)
    
    def _toggle_skeleton(self, checked: bool):
        """Toggle skeleton visibility."""
        self.canvas.set_show_skeleton(checked)
    
    def _toggle_bbox(self, checked: bool):
        """Toggle bounding box visibility."""
        self.canvas.set_show_bbox(checked)
    
    def _next_frame(self):
        """Go to next frame."""
        project = self.project_manager.current_project
        if project:
            project.next_frame()
            self._load_current_frame()
    
    def _prev_frame(self):
        """Go to previous frame."""
        project = self.project_manager.current_project
        if project:
            project.prev_frame()
            self._load_current_frame()
    
    def _add_instance(self):
        """Add a new instance to current frame."""
        project = self.project_manager.current_project
        if not project:
            return
        
        frame = project.get_current_frame()
        if not frame:
            return
        
        self.instance_panel.add_instance()
    
    def _delete_selected(self):
        """Delete selected keypoint or instance."""
        # First try to delete explicitly selected keypoint (clicked on canvas)
        if self.canvas.delete_selected_keypoint():
            self.keypoint_panel.refresh()
            self.instance_panel.refresh()
            return

        # Then try to delete hovered keypoint
        if self.canvas.delete_hovered_keypoint():
            self.keypoint_panel.refresh()
            self.instance_panel.refresh()
            return

        # Then try to delete current keypoint (selected in panel)
        if self.canvas.delete_current_keypoint():
            self.keypoint_panel.refresh()
            self.instance_panel.refresh()
            return

        # Finally, delete selected instance
        self.instance_panel.delete_selected()
    
    def _select_keypoint_by_index(self, index: int):
        """Select a keypoint by its index."""
        self.keypoint_panel.select_by_index(index)
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>Navigation</b></td><td></td></tr>
<tr><td>A</td><td>Previous frame</td></tr>
<tr><td>D</td><td>Next frame</td></tr>
<tr><td><b>Tools</b></td><td></td></tr>
<tr><td>V</td><td>Select tool</td></tr>
<tr><td>B</td><td>Bounding box tool</td></tr>
<tr><td>K</td><td>Keypoint tool</td></tr>
<tr><td><b>Annotation</b></td><td></td></tr>
<tr><td>N</td><td>New instance (hen)</td></tr>
<tr><td>1-0</td><td>Select keypoint 1-10</td></tr>
<tr><td>Delete</td><td>Delete selected</td></tr>
<tr><td><b>View</b></td><td></td></tr>
<tr><td>Ctrl+=</td><td>Zoom in</td></tr>
<tr><td>Ctrl+-</td><td>Zoom out</td></tr>
<tr><td>Ctrl+0</td><td>Fit to window</td></tr>
<tr><td><b>File</b></td><td></td></tr>
<tr><td>Ctrl+S</td><td>Save</td></tr>
<tr><td>Ctrl+E</td><td>Export</td></tr>
</table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About PoseLabeler",
            "<h2>PoseLabeler</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Multi-Animal Pose Annotation Tool</p>"
            "<p>Creates YOLO-format pose estimation datasets.</p>"
            "<hr>"
            "<p>Built for the Poultry Vision project.</p>"
        )
    
    # ===== Signal Handlers =====
    
    def _on_instance_selected(self, instance_id: str):
        """Handle instance selection."""
        project = self.project_manager.current_project
        if not project:
            return
        
        frame = project.get_current_frame()
        if not frame:
            return
        
        instance = frame.get_instance(instance_id)
        if instance:
            self.canvas.set_current_instance(instance)
            self.keypoint_panel.set_instance(instance)
    
    def _on_instance_added(self, instance):
        """Handle new instance added."""
        self.project_manager.mark_changed()
        self.canvas.set_current_instance(instance)
        self.keypoint_panel.set_instance(instance)
        self._set_tool("bbox")  # Switch to bbox tool for new instance
        self._update_title()
    
    def _on_instance_removed(self, instance_id: str):
        """Handle instance removed."""
        self.project_manager.mark_changed()
        self.canvas.refresh()
        self._update_title()
    
    def _on_instance_visibility_changed(self, instance_id: str, visible: bool):
        """Handle instance visibility change."""
        self.canvas.refresh()
    
    def _on_keypoint_selected(self, keypoint_name: str):
        """Handle keypoint selection."""
        self.canvas.set_current_keypoint(keypoint_name)
        self._set_tool("keypoint")
    
    def _on_keypoint_visibility_changed(self, keypoint_name: str, visibility: int):
        """Handle keypoint visibility change."""
        self.project_manager.mark_changed()
        self.canvas.refresh()
        self._update_title()
    
    def _on_frame_changed(self, index: int):
        """Handle frame change from navigation panel."""
        project = self.project_manager.current_project
        if project:
            project.go_to_frame(index)
            self._load_current_frame()
    
    def _on_annotation_changed(self):
        """Handle annotation change from canvas."""
        self.project_manager.mark_changed()
        self._update_title()
        
        # Update panels
        project = self.project_manager.current_project
        if project:
            frame = project.get_current_frame()
            if frame:
                self.instance_panel.refresh()
                self.keypoint_panel.refresh()
    
    def _on_keypoint_placed(self, keypoint_name: str):
        """Handle keypoint placed on canvas."""
        self.project_manager.mark_changed()
        self._update_title()

        # Auto-advance to next keypoint
        self.keypoint_panel.select_next()

    def _on_keypoint_clicked(self, instance_id: str, keypoint_name: str):
        """Handle keypoint clicked on canvas - sync with panels."""
        project = self.project_manager.current_project
        if not project:
            return

        frame = project.get_current_frame()
        if not frame:
            return

        instance = frame.get_instance(instance_id)
        if instance:
            # Update instance panel selection
            self.instance_panel.select_instance(instance_id)
            # Update keypoint panel
            self.keypoint_panel.set_instance(instance)
            self.keypoint_panel.select_keypoint(keypoint_name)

    def _on_bbox_drawn(self):
        """Handle bounding box drawn."""
        self.project_manager.mark_changed()
        self._update_title()
        
        # Switch to keypoint tool after bbox
        self._set_tool("keypoint")
        self.keypoint_panel.select_first()
    
    # ===== Events =====
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.project_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
                return
        
        self._save_settings()
        event.accept()
