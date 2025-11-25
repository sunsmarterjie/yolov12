# live_inference.py
"""
Live poultry vision inference script.

- Designed for Raspberry Pi 5 deployment (but works on desktop too).
- Connects to a video source (USB cam, Pi cam, RTSP stream, etc.).
- Runs YOLOv12 tracking + HenBehaviorMonitor in an infinite loop.
- Only stops when:
    - 'q' is pressed in the display window, or
    - the process receives Ctrl+C (KeyboardInterrupt).

Usage examples:
    python live_inference.py
    python live_inference.py --source 0
    python live_inference.py --source rtsp://user:pass@camera-ip:554/stream
"""

import argparse
import time
import cv2
from ultralytics import YOLO
from behavior_monitor import HenBehaviorMonitor

# --- CONFIGURATION ---
MODEL_PATH = "models/poultry-yolov12n-v1.pt"

# Default source:
#   0          -> /dev/video0 (USB cam) on Pi or default webcam on laptop
#   "0" or "1" -> also treated as numeric indices
#   "rtsp://..." -> treated as RTSP URL
DEFAULT_SOURCE = 0

# Class Map
CLASS_MAP = {
    0: 'feeder',
    1: 'hen',
    2: 'waterer'
}

# Visualization Colors (BGR)
COLOR_HEN_IDLE = (0, 255, 0)       # Green
COLOR_FEEDER = (0, 140, 255)       # Orange
COLOR_WATERER = (235, 206, 135)    # Light-ish
# ---------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Live Poultry Vision Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to YOLO model .pt file"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Video source: index like '0' or '1', or RTSP/URL"
    )
    return parser.parse_args()


def open_capture(source_str):
    """
    Open a cv2.VideoCapture from a string that may be an int index or URL.
    """
    # Try to interpret as integer (camera index)
    try:
        src = int(source_str)
    except ValueError:
        src = source_str  # treat as URL / file path

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[WARN] Failed to open video source: {source_str}")
        return None

    # Try to read FPS; fallback if not provided
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN or 0
        fps = 30.0

    print(f"[INFO] Opened source={source_str}, reported FPS={fps:.1f}")
    return cap, fps


def main():
    args = parse_args()

    print(f"[INFO] Loading model from {args.model}...")
    model = YOLO(args.model)

    # Try to open capture
    cap_info = open_capture(args.source)
    while cap_info is None:
        print("[ERROR] Could not open source. Retrying in 5 seconds...")
        time.sleep(5)
        cap_info = open_capture(args.source)

    cap, fps = cap_info
    monitor = HenBehaviorMonitor(fps=fps)

    print(f"[INFO] Starting LIVE inference from source={args.source}")
    print("[INFO] Press 'q' in the window or Ctrl+C in the terminal to stop.")

    cv2.namedWindow("Poultry Vision Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("[WARN] Frame grab failed. Attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap_info = open_capture(args.source)
                if cap_info is None:
                    print("[ERROR] Reconnect failed. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                cap, fps = cap_info
                monitor = HenBehaviorMonitor(fps=fps)
                continue

            # Run YOLO tracking
            results = model.track(frame, persist=True, verbose=False)

            current_hens = []
            current_feeders = []
            current_waterers = []

            # Extract detections
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    class_name = CLASS_MAP.get(int(cls_id))

                    if class_name == 'hen':
                        current_hens.append({'id': int(track_id), 'box': box})
                    elif class_name == 'feeder':
                        current_feeders.append(box)
                    elif class_name == 'waterer':
                        current_waterers.append(box)

            # Update behavior logic
            monitor.update(current_hens, current_feeders, current_waterers)

            # --- VISUALIZATION ---

            # 1. Draw feeders/waterers
            for box in current_feeders:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FEEDER, 2)
                cv2.putText(
                    frame, "Feeder", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FEEDER, 2, cv2.LINE_AA
                )

            for box in current_waterers:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WATERER, 2)
                cv2.putText(
                    frame, "Waterer", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WATERER, 2, cv2.LINE_AA
                )

            # 2. Draw hens + stats
            for hen in current_hens:
                hen_id = hen['id']
                stats = monitor.get_stats(hen_id)

                if stats:
                    x1, y1, x2, y2 = map(int, hen['box'])

                    action = stats['current_action']
                    box_color = COLOR_HEN_IDLE
                    if action == 'Feeding':
                        box_color = COLOR_FEEDER
                    elif action == 'Drinking':
                        box_color = COLOR_WATERER

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    label_main = f"Hen {hen_id}"
                    label_sub = f"{action}"
                    label_stats = (
                        f"Feeding: {stats['food_time']:.1f}s | "
                        f"Drinking: {stats['water_time']:.1f}s"
                    )

                    # Draw a filled rectangle above the box
                    cv2.rectangle(frame, (x1, y1 - 45), (x2, y1), box_color, -1)

                    # Text rows
                    cv2.putText(
                        frame, label_main, (x1 + 5, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, label_sub, (x1 + 85, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, label_stats, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
                    )

            cv2.imshow("Poultry Vision Live", frame)

            # Only exit on explicit 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down...")

    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Live inference stopped cleanly.")


if __name__ == "__main__":
    main()