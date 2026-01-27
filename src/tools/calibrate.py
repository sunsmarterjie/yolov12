"""
Interactive pen calibration tool.

This tool allows users to define the pen geometry by clicking
4 corner points on a video frame. The calibration is saved
to a numpy file for use in coordinate transformation.

Usage:
    python -m src.tools.calibrate --video path/to/video.mp4 --output pen_config.npy

Refactored from calibrate_pen.py with improved UI and config output.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import yaml


class PenCalibrator:
    """
    Interactive pen calibration using OpenCV.

    Click 4 corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
    """

    def __init__(self, video_path: str, output_path: str = "pen_config.npy"):
        """
        Initialize calibrator.

        Args:
            video_path: Path to video file or camera index
            output_path: Output path for calibration file
        """
        self.video_path = video_path
        self.output_path = output_path
        self.points: List[Tuple[int, int]] = []
        self.frame: Optional[np.ndarray] = None
        self.window_name = "Pen Calibration - Click 4 Corners"

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                self._draw_overlay()

    def _draw_overlay(self) -> None:
        """Draw calibration overlay on frame."""
        if self.frame is None:
            return

        display = self.frame.copy()

        # Draw existing points
        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 8, (0, 0, 255), -1)
            cv2.circle(display, pt, 10, (255, 255, 255), 2)
            label = ["TL", "TR", "BR", "BL"][i] if i < 4 else str(i)
            cv2.putText(
                display, label, (pt[0] + 15, pt[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # Draw lines connecting points
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32)
            cv2.polylines(display, [pts], len(self.points) == 4, (0, 255, 0), 2)

        # Draw instructions
        instructions = [
            "Click 4 corners: TL -> TR -> BR -> BL",
            f"Points: {len(self.points)}/4",
            "Press 'r' to reset, 's' to save, 'q' to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                display, text, (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        cv2.imshow(self.window_name, display)

    def calibrate(self) -> Optional[np.ndarray]:
        """
        Run interactive calibration.

        Returns:
            4x2 numpy array of corner points, or None if cancelled
        """
        # Open video source
        if self.video_path.isdigit():
            cap = cv2.VideoCapture(int(self.video_path))
        else:
            cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.video_path}")
            return None

        # Read first frame
        ret, self.frame = cap.read()
        cap.release()

        if not ret or self.frame is None:
            print("Error: Could not read frame from video source")
            return None

        # Setup window and callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Initial display
        self._draw_overlay()

        print("\n=== Pen Calibration ===")
        print("Click 4 corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
        print("Press 'r' to reset, 's' to save, 'q' to quit\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Calibration cancelled")
                break

            elif key == ord('r'):
                self.points = []
                print("Points reset")
                self._draw_overlay()

            elif key == ord('s') or len(self.points) == 4:
                if len(self.points) == 4:
                    corners = np.array(self.points, dtype=np.float32)
                    self._save_calibration(corners)
                    cv2.destroyAllWindows()
                    return corners
                else:
                    print(f"Need 4 points, have {len(self.points)}")

        cv2.destroyAllWindows()
        return None

    def _save_calibration(self, corners: np.ndarray) -> None:
        """Save calibration to file."""
        # Save numpy format
        np.save(self.output_path, corners)
        print(f"\nCalibration saved to: {self.output_path}")

        # Also save as YAML for human readability
        yaml_path = Path(self.output_path).with_suffix('.yaml')
        config = {
            'pen_corners': {
                'top_left': corners[0].tolist(),
                'top_right': corners[1].tolist(),
                'bottom_right': corners[2].tolist(),
                'bottom_left': corners[3].tolist()
            },
            'source_video': str(self.video_path)
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Human-readable config saved to: {yaml_path}")

        # Print corners
        print("\nCorner coordinates (pixels):")
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for label, pt in zip(labels, corners):
            print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")


def main():
    """Main entry point for calibration tool."""
    parser = argparse.ArgumentParser(
        description="Interactive pen calibration tool"
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default='samplevideos/poultry-vid-01.mp4',
        help='Video file path or camera index'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='pen_config.npy',
        help='Output calibration file path'
    )

    args = parser.parse_args()

    calibrator = PenCalibrator(args.video, args.output)
    result = calibrator.calibrate()

    if result is not None:
        print("\nCalibration complete!")
    else:
        print("\nCalibration failed or cancelled")


if __name__ == "__main__":
    main()
