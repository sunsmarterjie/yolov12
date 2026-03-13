"""
Annotation canvas for displaying images and handling keypoint/bbox interactions.
"""

from pathlib import Path
from typing import Optional
import math

from PyQt6.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QLabel, QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont, QImage,
    QMouseEvent, QWheelEvent, QPainterPath, QTransform
)

from core import FrameAnnotation, Instance, Keypoint, Schema


class AnnotationCanvas(QWidget):
    """Canvas widget for displaying and annotating images."""
    
    # Signals
    annotation_changed = pyqtSignal()
    keypoint_placed = pyqtSignal(str)  # keypoint name
    keypoint_clicked = pyqtSignal(str, str)  # instance_id, keypoint_name
    bbox_drawn = pyqtSignal()
    
    # Constants
    POINT_RADIUS = 8
    POINT_RADIUS_SELECTED = 10
    HIT_RADIUS = 15
    SKELETON_WIDTH = 2
    BBOX_WIDTH = 2
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)

        # Image state
        self._pixmap: Optional[QPixmap] = None
        self._image_path: Optional[Path] = None
        self._image_size = QSize(1, 1)

        # View state
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._dragging_view = False
        self._drag_start = QPoint()

        # Annotation state
        self._frame: Optional[FrameAnnotation] = None
        self._schema: Optional[Schema] = None
        self._current_instance: Optional[Instance] = None
        self._current_keypoint: Optional[str] = None

        # Tool state
        self._tool = "select"  # select, bbox, keypoint, eraser
        self._show_skeleton = True
        self._show_bbox = True

        # Interaction state
        self._hovered_point: Optional[tuple[str, str]] = None  # (instance_id, keypoint_name)
        self._selected_point: Optional[tuple[str, str]] = None  # (instance_id, keypoint_name) - persistent selection
        self._dragging_point: Optional[tuple[str, str]] = None
        self._drawing_bbox = False
        self._bbox_start: Optional[QPointF] = None
        self._bbox_current: Optional[QPointF] = None

        # Undo stack: list of serialized FrameAnnotation dicts
        self._undo_stack: list[dict] = []
        self._MAX_UNDO = 50

        # View rotation (degrees CW, 0/90/180/270 only)
        self._rotation_angle: int = 0
    
    def load_image(self, path: Path) -> bool:
        """Load an image from path."""
        if not path.exists():
            self._pixmap = None
            self._image_path = None
            self.update()
            return False
        
        self._pixmap = QPixmap(str(path))
        if self._pixmap.isNull():
            self._pixmap = None
            self._image_path = None
            self.update()
            return False
        
        self._image_path = path
        self._image_size = self._pixmap.size()
        
        # Update frame dimensions
        if self._frame:
            self._frame.image_width = self._image_size.width()
            self._frame.image_height = self._image_size.height()
        
        self.fit_to_window()
        return True
    
    def set_frame(self, frame: FrameAnnotation) -> None:
        """Set the current frame annotation."""
        self._frame = frame

        if self._pixmap:
            frame.image_width = self._image_size.width()
            frame.image_height = self._image_size.height()

        self._current_instance = None
        self._current_keypoint = None
        self._selected_point = None
        self._rotation_angle = 0

        self.update()
    
    def set_schema(self, schema: Schema) -> None:
        """Set the keypoint schema."""
        self._schema = schema
        self.update()
    
    def set_current_instance(self, instance: Optional[Instance]) -> None:
        """Set the currently selected instance."""
        self._current_instance = instance
        self.update()
    
    def set_current_keypoint(self, keypoint_name: Optional[str]) -> None:
        """Set the currently selected keypoint to place."""
        self._current_keypoint = keypoint_name
        self.update()
    
    def set_tool(self, tool: str) -> None:
        """Set the current tool (select, bbox, keypoint)."""
        self._tool = tool
        self._update_cursor()
        self.update()
    
    def set_show_skeleton(self, show: bool) -> None:
        """Toggle skeleton visibility."""
        self._show_skeleton = show
        self.update()
    
    def set_show_bbox(self, show: bool) -> None:
        """Toggle bounding box visibility."""
        self._show_bbox = show
        self.update()

    # ===== Undo =====

    def _push_undo(self) -> None:
        """Snapshot current frame state onto the undo stack."""
        if not self._frame:
            return
        self._undo_stack.append(self._frame.to_dict())
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore the previous frame state. Returns True if undo was performed."""
        if not self._undo_stack or not self._frame:
            return False
        state = self._undo_stack.pop()
        restored = FrameAnnotation.from_dict(state)
        # Restore instances in-place so panel references stay consistent
        self._frame.instances = restored.instances
        # Sync current instance reference
        if self._current_instance:
            self._current_instance = self._frame.get_instance(self._current_instance.id)
        self._selected_point = None
        self._hovered_point = None
        self.annotation_changed.emit()
        self.update()
        return True

    def delete_current_keypoint(self) -> bool:
        """Delete (reset) the currently selected keypoint. Returns True if deleted."""
        if not self._current_instance or not self._current_keypoint:
            return False

        if self._current_keypoint in self._current_instance.keypoints:
            kp = self._current_instance.keypoints[self._current_keypoint]
            if kp.is_labeled():
                self._push_undo()
                kp.x = 0.0
                kp.y = 0.0
                kp.visibility = 0
                self.annotation_changed.emit()
                self.update()
                return True
        return False

    def delete_selected_keypoint(self) -> bool:
        """Delete the explicitly selected keypoint (clicked in select mode). Returns True if deleted."""
        if not self._selected_point or not self._frame:
            return False

        instance_id, kp_name = self._selected_point
        instance = self._frame.get_instance(instance_id)

        if instance and kp_name in instance.keypoints:
            kp = instance.keypoints[kp_name]
            if kp.is_labeled():
                self._push_undo()
                kp.x = 0.0
                kp.y = 0.0
                kp.visibility = 0
                self._selected_point = None
                self.annotation_changed.emit()
                self.update()
                return True
        return False

    def erase_point_at(self, view_pos: QPointF) -> bool:
        """Erase (delete) a keypoint at the given view position. Used by eraser tool."""
        point = self._find_point_at(view_pos)
        if not point or not self._frame:
            return False
        instance_id, kp_name = point
        instance = self._frame.get_instance(instance_id)
        if instance and kp_name in instance.keypoints:
            kp = instance.keypoints[kp_name]
            if kp.is_labeled():
                self._push_undo()
                kp.x = 0.0
                kp.y = 0.0
                kp.visibility = 0
                self._hovered_point = None
                self._selected_point = None
                self.annotation_changed.emit()
                self.update()
                return True
        return False
    
    def refresh(self) -> None:
        """Force refresh of the canvas."""
        self.update()
    
    def zoom(self, factor: float) -> None:
        """Zoom by a factor."""
        new_scale = self._scale * factor
        new_scale = max(0.1, min(10.0, new_scale))
        
        # Zoom towards center
        center = QPointF(self.width() / 2, self.height() / 2)
        image_center = self._view_to_image(center)
        
        self._scale = new_scale
        
        new_view_center = self._image_to_view(image_center)
        self._offset += center - new_view_center
        
        self.update()
    
    def fit_to_window(self) -> None:
        """Fit image to window."""
        if not self._pixmap:
            return
        
        padding = 40
        available_w = self.width() - padding
        available_h = self.height() - padding
        
        scale_w = available_w / self._image_size.width()
        scale_h = available_h / self._image_size.height()
        
        self._scale = min(scale_w, scale_h, 1.0)
        
        # Center image
        scaled_w = self._image_size.width() * self._scale
        scaled_h = self._image_size.height() * self._scale
        
        self._offset = QPointF(
            (self.width() - scaled_w) / 2,
            (self.height() - scaled_h) / 2
        )
        
        self.update()
    
    def rotate_view(self, degrees: int) -> None:
        """Rotate the view by degrees (positive = CW). Snaps to 90° steps."""
        self._rotation_angle = (self._rotation_angle + degrees) % 360
        self.update()

    def _image_to_view(self, point: QPointF) -> QPointF:
        """Convert image coordinates to view coordinates (includes rotation)."""
        vx = point.x() * self._scale + self._offset.x()
        vy = point.y() * self._scale + self._offset.y()
        if self._rotation_angle != 0:
            cx = self._image_size.width() * self._scale / 2 + self._offset.x()
            cy = self._image_size.height() * self._scale / 2 + self._offset.y()
            vx -= cx
            vy -= cy
            angle = math.radians(self._rotation_angle)
            vx, vy = (
                vx * math.cos(angle) - vy * math.sin(angle),
                vx * math.sin(angle) + vy * math.cos(angle),
            )
            vx += cx
            vy += cy
        return QPointF(vx, vy)

    def _view_to_image(self, point: QPointF) -> QPointF:
        """Convert view coordinates to image coordinates (includes inverse rotation)."""
        px, py = point.x(), point.y()
        if self._rotation_angle != 0:
            cx = self._image_size.width() * self._scale / 2 + self._offset.x()
            cy = self._image_size.height() * self._scale / 2 + self._offset.y()
            px -= cx
            py -= cy
            angle = math.radians(-self._rotation_angle)
            px, py = (
                px * math.cos(angle) - py * math.sin(angle),
                px * math.sin(angle) + py * math.cos(angle),
            )
            px += cx
            py += cy
        return QPointF(
            (px - self._offset.x()) / self._scale,
            (py - self._offset.y()) / self._scale,
        )
    
    def _image_to_normalized(self, point: QPointF) -> tuple[float, float]:
        """Convert image coordinates to normalized (0-1)."""
        return (
            point.x() / self._image_size.width(),
            point.y() / self._image_size.height()
        )
    
    def _normalized_to_image(self, x: float, y: float) -> QPointF:
        """Convert normalized coordinates to image coordinates."""
        return QPointF(
            x * self._image_size.width(),
            y * self._image_size.height()
        )
    
    def _update_cursor(self) -> None:
        """Update cursor based on tool and state."""
        if self._tool == "bbox":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._tool == "keypoint":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._tool == "eraser":
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
        elif self._hovered_point:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _find_point_at(self, view_pos: QPointF) -> Optional[tuple[str, str]]:
        """Find a keypoint at the given view position."""
        if not self._frame or not self._schema:
            return None
        
        hit_radius = self.HIT_RADIUS / self._scale
        
        for instance in self._frame.instances:
            if not instance.visible:
                continue
            
            for kp_name, kp in instance.keypoints.items():
                if not kp.is_labeled():
                    continue
                
                kp_image = self._normalized_to_image(kp.x, kp.y)
                cursor_image = self._view_to_image(view_pos)
                
                dist = math.sqrt(
                    (kp_image.x() - cursor_image.x()) ** 2 +
                    (kp_image.y() - cursor_image.y()) ** 2
                )
                
                if dist < hit_radius:
                    return (instance.id, kp_name)
        
        return None
    
    def _place_keypoint(self, view_pos: QPointF) -> None:
        """Place or move a keypoint at the given position."""
        if not self._current_instance or not self._current_keypoint:
            return

        self._push_undo()

        image_pos = self._view_to_image(view_pos)
        norm_x, norm_y = self._image_to_normalized(image_pos)

        # Clamp to image bounds
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))

        # Create or update keypoint
        if self._current_keypoint not in self._current_instance.keypoints:
            self._current_instance.keypoints[self._current_keypoint] = Keypoint(
                name=self._current_keypoint,
                x=norm_x,
                y=norm_y,
                visibility=2  # Default to visible
            )
        else:
            kp = self._current_instance.keypoints[self._current_keypoint]
            kp.x = norm_x
            kp.y = norm_y
            if kp.visibility == 0:
                kp.visibility = 2

        self.annotation_changed.emit()
        self.keypoint_placed.emit(self._current_keypoint)
        self.update()
    
    def _start_bbox(self, view_pos: QPointF) -> None:
        """Start drawing a bounding box."""
        if not self._current_instance:
            return
        
        self._drawing_bbox = True
        self._bbox_start = view_pos
        self._bbox_current = view_pos
    
    def _update_bbox(self, view_pos: QPointF) -> None:
        """Update bounding box being drawn."""
        if self._drawing_bbox:
            self._bbox_current = view_pos
            self.update()
    
    def _finish_bbox(self) -> None:
        """Finish drawing bounding box."""
        if not self._drawing_bbox or not self._current_instance:
            return

        if self._bbox_start and self._bbox_current:
            # Convert to normalized coordinates
            start_image = self._view_to_image(self._bbox_start)
            end_image = self._view_to_image(self._bbox_current)

            x1, y1 = self._image_to_normalized(start_image)
            x2, y2 = self._image_to_normalized(end_image)

            # Ensure valid bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Clamp to bounds
            x1 = max(0, min(1, x1))
            x2 = max(0, min(1, x2))
            y1 = max(0, min(1, y1))
            y2 = max(0, min(1, y2))

            # Check minimum size
            if x2 - x1 > 0.01 and y2 - y1 > 0.01:
                self._push_undo()
                from core import BoundingBox
                self._current_instance.bbox = BoundingBox.from_corners(x1, y1, x2, y2)
                self.annotation_changed.emit()
                self.bbox_drawn.emit()

        self._drawing_bbox = False
        self._bbox_start = None
        self._bbox_current = None
        self.update()
    
    # ===== Painting =====
    
    def paintEvent(self, event):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        
        if not self._pixmap:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return
        
        # Draw image (with optional rotation around image center)
        painter.save()
        if self._rotation_angle != 0:
            cx = self._image_size.width() * self._scale / 2 + self._offset.x()
            cy = self._image_size.height() * self._scale / 2 + self._offset.y()
            painter.translate(cx, cy)
            painter.rotate(self._rotation_angle)
            painter.translate(-cx, -cy)
        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()
        
        # Draw annotations
        if self._frame and self._schema:
            self._draw_annotations(painter)
        
        # Draw bbox being drawn
        if self._drawing_bbox and self._bbox_start and self._bbox_current:
            self._draw_temp_bbox(painter)
    
    def _draw_annotations(self, painter: QPainter) -> None:
        """Draw all annotations for the current frame."""
        if not self._frame:
            return
        
        for instance in self._frame.instances:
            if not instance.visible:
                continue
            
            is_selected = (instance == self._current_instance)
            
            # Draw bounding box
            if self._show_bbox and instance.bbox.is_valid():
                self._draw_bbox(painter, instance, is_selected)
            
            # Draw skeleton
            if self._show_skeleton:
                self._draw_skeleton(painter, instance, is_selected)
            
            # Draw keypoints
            self._draw_keypoints(painter, instance, is_selected)
    
    def _draw_bbox(self, painter: QPainter, instance: Instance, is_selected: bool) -> None:
        """Draw bounding box for an instance."""
        x1, y1, x2, y2 = instance.bbox.get_corners()
        
        p1 = self._image_to_view(self._normalized_to_image(x1, y1))
        p2 = self._image_to_view(self._normalized_to_image(x2, y2))
        
        rect = QRectF(p1, p2)
        
        color = QColor("#00FF00") if is_selected else QColor("#AAAAAA")
        pen = QPen(color, self.BBOX_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine if not is_selected else Qt.PenStyle.SolidLine)
        
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        
        # Draw instance label
        painter.setPen(color)
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        label = f"Hen #{instance.id[:4]}"
        painter.drawText(p1 + QPointF(4, -4), label)
    
    def _draw_skeleton(self, painter: QPainter, instance: Instance, is_selected: bool) -> None:
        """Draw skeleton connections for an instance."""
        if not self._schema:
            return
        
        alpha = 200 if is_selected else 100
        
        for conn in self._schema.skeleton:
            kp1_name, kp2_name = conn
            
            kp1 = instance.keypoints.get(kp1_name)
            kp2 = instance.keypoints.get(kp2_name)
            
            if not kp1 or not kp2 or not kp1.is_labeled() or not kp2.is_labeled():
                continue
            
            p1 = self._image_to_view(self._normalized_to_image(kp1.x, kp1.y))
            p2 = self._image_to_view(self._normalized_to_image(kp2.x, kp2.y))
            
            # Get color from first keypoint
            colors = self._schema.get_keypoint_colors()
            color = QColor(colors.get(kp1_name, "#FFFFFF"))
            color.setAlpha(alpha)
            
            pen = QPen(color, self.SKELETON_WIDTH)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
    
    def _draw_keypoints(self, painter: QPainter, instance: Instance, is_selected: bool) -> None:
        """Draw keypoints for an instance."""
        if not self._schema:
            return

        colors = self._schema.get_keypoint_colors()

        for kp_name, kp in instance.keypoints.items():
            if not kp.is_labeled():
                continue

            pos = self._image_to_view(self._normalized_to_image(kp.x, kp.y))

            is_current = (
                is_selected and
                self._current_keypoint == kp_name
            )

            is_hovered = (
                self._hovered_point and
                self._hovered_point[0] == instance.id and
                self._hovered_point[1] == kp_name
            )

            # Check if this keypoint is explicitly selected (for deletion)
            is_explicitly_selected = (
                self._selected_point and
                self._selected_point[0] == instance.id and
                self._selected_point[1] == kp_name
            )

            # Determine appearance
            radius = self.POINT_RADIUS_SELECTED if (is_current or is_hovered or is_explicitly_selected) else self.POINT_RADIUS

            color = QColor(colors.get(kp_name, "#FFFFFF"))

            # Draw selection ring for explicitly selected keypoint
            if is_explicitly_selected:
                painter.setPen(QPen(QColor("#FF6B6B"), 3))  # Red selection ring
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(pos, radius + 4, radius + 4)

            # Draw outer circle
            if is_selected:
                painter.setPen(QPen(QColor("#FFFFFF"), 2))
            else:
                painter.setPen(QPen(color.darker(150), 1))

            painter.setBrush(QBrush(color))
            painter.drawEllipse(pos, radius, radius)

            # Draw visibility indicator
            if kp.visibility == 1:  # Occluded
                painter.setPen(QPen(QColor("#000000"), 2))
                painter.drawLine(
                    pos - QPointF(radius * 0.5, radius * 0.5),
                    pos + QPointF(radius * 0.5, radius * 0.5)
                )
                painter.drawLine(
                    pos - QPointF(-radius * 0.5, radius * 0.5),
                    pos + QPointF(-radius * 0.5, radius * 0.5)
                )

            # Draw keypoint name tooltip on hover or selection
            if is_hovered or is_current or is_explicitly_selected:
                self._draw_tooltip(painter, pos, kp_name, color, radius)
    
    def _draw_tooltip(self, painter: QPainter, pos: QPointF, text: str, color: QColor, radius: float) -> None:
        """Draw a tooltip-style label near a keypoint."""
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        # Calculate text dimensions
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        
        # Tooltip position (above and to the right of the point)
        padding = 6
        tooltip_x = pos.x() + radius + 8
        tooltip_y = pos.y() - text_height - padding
        
        # Background rectangle
        bg_rect = QRectF(
            tooltip_x - padding,
            tooltip_y - padding / 2,
            text_width + padding * 2,
            text_height + padding
        )
        
        # Draw background with slight transparency
        bg_color = QColor(40, 40, 40, 220)
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(bg_rect, 4, 4)
        
        # Draw text
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(QPointF(tooltip_x, tooltip_y + text_height - padding / 2), text)
    
    def _draw_temp_bbox(self, painter: QPainter) -> None:
        """Draw temporary bounding box being drawn."""
        rect = QRectF(self._bbox_start, self._bbox_current)
        
        pen = QPen(QColor("#00FF00"), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
    
    # ===== Mouse Events =====
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        pos = QPointF(event.position())
        
        # Middle button: pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_view = True
            self._drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        
        # Right click: cycle visibility (visible → occluded → unlabeled)
        if event.button() == Qt.MouseButton.RightButton:
            point = self._find_point_at(pos)
            if point:
                instance_id, kp_name = point
                if self._frame:
                    instance = self._frame.get_instance(instance_id)
                    if instance and kp_name in instance.keypoints:
                        kp = instance.keypoints[kp_name]
                        self._push_undo()
                        # Cycle: visible(2) → occluded(1) → unlabeled(0) → visible(2)
                        kp.visibility = (kp.visibility - 1) % 3
                        if kp.visibility == 0:
                            # Unlabeled: clear coordinates so point disappears
                            kp.x = 0.0
                            kp.y = 0.0
                        self.annotation_changed.emit()
                        self.update()
            return
        
        # Left click handling
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == "select":
                # Check for point dragging
                point = self._find_point_at(pos)
                if point:
                    self._push_undo()  # Capture state before drag begins
                    self._dragging_point = point
                    self._selected_point = point  # Set persistent selection
                    instance_id, kp_name = point

                    # Select this instance
                    if self._frame:
                        instance = self._frame.get_instance(instance_id)
                        if instance:
                            self._current_instance = instance
                            self._current_keypoint = kp_name
                            self.keypoint_clicked.emit(instance_id, kp_name)
                            self.update()
                else:
                    # Clear selection and start view drag if not on a point
                    self._selected_point = None
                    self._dragging_view = True
                    self._drag_start = event.pos()
                    self.update()
            
            elif self._tool == "bbox":
                self._start_bbox(pos)

            elif self._tool == "keypoint":
                self._place_keypoint(pos)

            elif self._tool == "eraser":
                self.erase_point_at(pos)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        pos = QPointF(event.position())
        
        # View dragging
        if self._dragging_view:
            delta = event.pos() - self._drag_start
            self._offset += QPointF(delta.x(), delta.y())
            self._drag_start = event.pos()
            self.update()
            return
        
        # Point dragging
        if self._dragging_point:
            instance_id, kp_name = self._dragging_point
            if self._frame:
                instance = self._frame.get_instance(instance_id)
                if instance and kp_name in instance.keypoints:
                    image_pos = self._view_to_image(pos)
                    norm_x, norm_y = self._image_to_normalized(image_pos)
                    norm_x = max(0, min(1, norm_x))
                    norm_y = max(0, min(1, norm_y))
                    
                    instance.keypoints[kp_name].x = norm_x
                    instance.keypoints[kp_name].y = norm_y
                    
                    self.update()
            return
        
        # Bbox drawing
        if self._drawing_bbox:
            self._update_bbox(pos)
            return
        
        # Hover detection
        old_hovered = self._hovered_point
        self._hovered_point = self._find_point_at(pos)
        
        if old_hovered != self._hovered_point:
            self._update_cursor()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_view = False
            self._update_cursor()
        
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_view:
                self._dragging_view = False
                self._update_cursor()
            
            if self._dragging_point:
                self.annotation_changed.emit()
                self._dragging_point = None
            
            if self._drawing_bbox:
                self._finish_bbox()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.zoom(1.15)
        else:
            self.zoom(0.85)
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        # Optionally re-fit on resize
        # self.fit_to_window()
