#!/usr/bin/env python3
"""
Main entry point for poultry vision inference.

This script provides a unified interface for running inference
on video files, USB cameras, or RTSP streams.

Usage:
    python scripts/run_inference.py --source 0                    # USB camera
    python scripts/run_inference.py --source rtsp://...           # RTSP stream
    python scripts/run_inference.py --source video.mp4            # Video file
    python scripts/run_inference.py --config config/system_config.yaml
"""

import argparse
import sys
from pathlib import Path
import time
import cv2
import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inference import CameraView
from src.core.state_machine import PenStateMachine
from src.core.behavior import HenBehaviorMonitor
from src.core.geometry import get_homography_matrix


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pen_corners(calibration_file: str) -> np.ndarray:
    """Load pen calibration from file."""
    if Path(calibration_file).exists():
        return np.load(calibration_file)
    else:
        print(f"Warning: Calibration file not found: {calibration_file}")
        print("Using default pen corners. Run calibration first.")
        return np.array([
            [100, 100],
            [1820, 100],
            [1820, 980],
            [100, 980]
        ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Poultry Vision Inference System"
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='Video source (camera index, RTSP URL, or file path)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/poultry-yolov12n-v1.pt',
        help='Path to YOLO model'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/system_config.yaml',
        help='Path to system configuration file'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default='pen_config.npy',
        help='Path to pen calibration file'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display (headless mode)'
    )
    parser.add_argument(
        '--save-video',
        type=str,
        default=None,
        help='Save output to video file'
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

    pen_config = config.get('pen', {})
    pen_width = pen_config.get('width_cm', 120)
    pen_height = pen_config.get('height_cm', 200)

    # Load calibration
    calibration_file = args.calibration
    if 'calibration_file' in pen_config:
        calibration_file = pen_config['calibration_file']

    pen_corners = load_pen_corners(calibration_file)

    # Compute homography
    homography = get_homography_matrix(pen_corners, pen_width, pen_height)

    # Initialize components
    print(f"Loading model: {args.model}")
    camera = CameraView(
        name="main",
        model_path=args.model,
        pen_polygon=pen_corners,
        homography_matrix=homography
    )

    pen_state = PenStateMachine(
        width_cm=pen_width,
        height_cm=pen_height,
        interaction_radius_cm=pen_config.get('interaction_radius_cm', 15)
    )

    behavior_monitor = HenBehaviorMonitor(fps=30.0)

    # Open video source
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"Video FPS: {fps}")

    # Video writer
    writer = None
    if args.save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        print(f"Saving output to: {args.save_video}")

    print("\n=== Poultry Vision System ===")
    print("Press 'q' to quit, 's' to save screenshot")
    print("=" * 30 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.startswith('rtsp'):
                    break  # End of video file
                continue  # RTSP dropout, retry

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms <= 0:
                timestamp_ms = frame_count * (1000 / fps)

            # Process frame
            annotated, detections = camera.process_frame(frame)

            # Update state machine
            pen_state.update_from_detections(detections, timestamp_ms)

            # Update behavior monitor
            hens = [d for d in detections if d['type'] == 'hen']
            feeders = [d for d in detections if d['type'] == 'feeder']
            waterers = [d for d in detections if d['type'] == 'waterer']
            behavior_monitor.update(hens, feeders, waterers)

            # Generate minimap
            minimap = pen_state.generate_minimap(render_scale=3)

            # Display
            if not args.no_display:
                # Add stats overlay
                stats = behavior_monitor.get_summary()
                cv2.putText(
                    annotated,
                    f"Hens: {stats['total_hens']} | "
                    f"Feeding: {stats['total_feeding_time_s']:.1f}s | "
                    f"Drinking: {stats['total_drinking_time_s']:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

                cv2.imshow("Poultry Vision", annotated)
                cv2.imshow("Pen State", minimap)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count:06d}.png"
                    cv2.imwrite(screenshot_path, annotated)
                    print(f"Saved: {screenshot_path}")

            # Save video
            if writer:
                writer.write(annotated)

            frame_count += 1

            # Print progress periodically
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"Frame {frame_count} | FPS: {current_fps:.1f} | "
                      f"Hens: {len(pen_state.hens)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print final statistics
        elapsed = time.time() - start_time
        print(f"\n=== Session Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {frame_count / elapsed:.1f}")

        summary = behavior_monitor.get_summary()
        print(f"\nBehavior Summary:")
        print(f"  Total hens tracked: {summary['total_hens']}")
        print(f"  Total feeding time: {summary['total_feeding_time_s']:.1f}s")
        print(f"  Total drinking time: {summary['total_drinking_time_s']:.1f}s")
        print(f"  Feeding events: {summary['total_feeding_events']}")
        print(f"  Drinking events: {summary['total_drinking_events']}")


if __name__ == "__main__":
    main()
