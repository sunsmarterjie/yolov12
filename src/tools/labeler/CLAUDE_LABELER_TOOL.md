# CLAUDE_LABELER_TOOL.md
## Task Instruction File — Pose Labeler Improvements
> Always refer to this file. Update it when instructed or when the user rejects changes.

---

## Active Feature Branch
`feature/labeler-eraser-undo-tutorial`

---

## Labeling Workflow (User's Vision)
1. **Add Hen** — Press N or click "+ Add Hen (N)". Hen appears in left panel, auto-selected.
2. **Select Hen** — Click hen in left panel (if not already selected).
3. **Draw BBox** — Tool auto-switches to "Draw Box (B)". Click-drag around the hen body. Release to confirm.
4. **Select Keypoint** — Keypoint panel (right) auto-selects first keypoint. User can also press 1-0 to pick one.
5. **Place Keypoint** — Tool auto-switches to "Place Point (K)". Click on image to place. Auto-advances to next keypoint.
6. **Repeat** for all keypoints.

---

## Problems Found & Fixes Required

### Bug 1: Delete Mechanism Unreliable
- `_delete_selected()` in `main_window.py` tries three states: `_selected_point`, `_hovered_point`, `_current_keypoint`
- Ambiguous — the wrong keypoint often gets deleted
- **Fix:** Clear priority chain: selected_point → current_keypoint in panel → delete whole instance

### Bug 2: Right-Click Visibility Toggle Broken
- `canvas.py:634-635` skips `visibility=0`, so right-click can never truly unlabel a keypoint
- **Fix:** Remove the "skip 0" guard. Cycle: visible(2) → occluded(1) → unlabeled(0)

### Bug 3: No Undo
- Ctrl+Z is not bound. No undo history.
- **Fix:** Snapshot-based undo stack (max 50), push before every destructive action

### Missing Feature: Eraser Tool
- No dedicated erase mode
- **Fix:** E key activates eraser tool — click a keypoint to delete it (ForbiddenCursor)

### Missing Feature: Tutorial
- Only a bare shortcuts dialog exists
- **Fix:** 6-step tutorial dialog, F1 key, auto-shows on first launch

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Feature branch created | ✅ Done | `feature/labeler-eraser-undo-tutorial` |
| CLAUDE_LABELER_TOOL.md | ✅ Done | This file |
| Fix right-click bug | ✅ Done | Removed skip-0 guard; right-click now unlabels (clears x/y) |
| Undo stack (Ctrl+Z) | ✅ Done | Snapshot-based, max 50 steps, push before every destructive op |
| Eraser tool (E key) | ✅ Done | `erase_point_at()`, ForbiddenCursor, E shortcut + toolbar button |
| Simplify delete logic | ✅ Done | 2-step priority: selected_point → current_keypoint → instance |
| Tutorial dialog (F1) | ✅ Done | 6-step TutorialDialog, F1 key, auto-shows on first launch |
| Image rotation (R / Shift+R) | ✅ Done | View-only rotation, resets on frame change, toolbar buttons |

---

## Key Files

| File | Role |
|------|------|
| `src/tools/labeler/ui/canvas.py` | Drawing canvas, all mouse/keyboard events, tool state machine |
| `src/tools/labeler/ui/main_window.py` | App orchestration, toolbar, menus, shortcuts, signal routing |
| `src/tools/labeler/ui/panels.py` | Instance list (left), Keypoint list (right), Navigation (bottom-right) |
| `src/tools/labeler/ui/dialogs.py` | Project creation, export, schema editor, tutorial (to be added) |
| `src/tools/labeler/core/annotation.py` | Data models: Keypoint, BoundingBox, Instance, FrameAnnotation |
| `src/tools/labeler/core/project.py` | Project management and persistence |
| `src/tools/labeler/core/schema.py` | Keypoint schema definitions |

---

## Tool Modes (canvas._tool)
- `"select"` — V key, drag keypoints, click to select
- `"bbox"` — B key, click-drag to draw bounding box
- `"keypoint"` — K key, click to place keypoint
- `"eraser"` — E key (NEW), click to delete a keypoint

---

## Undo Stack Design
- `canvas._undo_stack: list[dict]` — list of serialized FrameAnnotation dicts
- `canvas._push_undo()` — serialize current frame state, push to stack (max 50)
- `canvas.undo()` — pop last state, restore frame from dict, emit annotation_changed
- Push before: `_place_keypoint()`, `_finish_bbox()`, `delete_*_keypoint()`, drag-release, instance delete

---

## Keyboard Shortcuts (Current + Planned)
| Key | Action |
|-----|--------|
| V | Select tool |
| B | Bounding box tool |
| K | Keypoint tool |
| E | Eraser tool (NEW) |
| N | Add new hen |
| Delete / Backspace | Delete selected keypoint or instance |
| Ctrl+Z | Undo (NEW) |
| A | Previous frame |
| D | Next frame |
| 1-0 | Select keypoints 1-10 |
| Ctrl+S | Save |
| Ctrl+E | Export to YOLO |
| Ctrl+= | Zoom in |
| Ctrl+- | Zoom out |
| Ctrl+0 | Fit to window |
| R | Rotate view 90° CW |
| Shift+R | Rotate view 90° CCW |
| F1 | Tutorial |

---

## User Feedback Log
> Update this section when the user rejects changes or requests corrections.

*(No rejections yet — initial implementation phase)*
