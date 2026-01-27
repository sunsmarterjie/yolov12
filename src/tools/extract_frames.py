#!/usr/bin/env python3
"""
Frame Extraction Script for PoseLabeler
Extracts frames from video files for pose annotation.

Usage:
    python extract_frames.py --video path/to/video.mp4 --output path/to/frames --interval 30
    
Or use in a Jupyter notebook by importing the functions.
"""

import argparse
import cv2
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    interval: int = 30,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize: Optional[tuple[int, int]] = None,
    format: str = "jpg",
    quality: int = 95,
    prefix: Optional[str] = None,
) -> dict:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Extract every Nth frame (default: 30, i.e., 1 frame per second at 30fps)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        max_frames: Maximum number of frames to extract (optional)
        resize: Resize frames to (width, height) if specified
        format: Output format ('jpg', 'png')
        quality: JPEG quality (1-100, default: 95)
        prefix: Prefix for output filenames (default: video filename)
    
    Returns:
        Dictionary with extraction statistics
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f}s ({total_frames} frames)")
    
    # Calculate frame range
    start_frame = 0
    if start_time is not None:
        start_frame = int(start_time * fps)
    
    end_frame = total_frames
    if end_time is not None:
        end_frame = min(int(end_time * fps), total_frames)
    
    # Set up prefix
    if prefix is None:
        prefix = video_path.stem
    
    # Extraction
    frame_count = 0
    saved_count = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate total iterations for progress bar
    total_iterations = (end_frame - start_frame) // interval
    if max_frames:
        total_iterations = min(total_iterations, max_frames)
    
    pbar = tqdm(total=total_iterations, desc="Extracting frames")
    
    for frame_idx in range(start_frame, end_frame, interval):
        if max_frames and saved_count >= max_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        
        # Generate filename with timestamp
        timestamp = frame_idx / fps
        filename = f"{prefix}_frame{frame_idx:06d}_t{timestamp:.2f}s.{format}"
        output_path = output_dir / filename
        
        # Save frame
        if format == "jpg":
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(output_path), frame)
        
        saved_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    stats = {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "total_video_frames": total_frames,
        "extracted_frames": saved_count,
        "interval": interval,
        "fps": fps,
        "duration": duration,
    }
    
    print(f"\nExtracted {saved_count} frames to {output_dir}")
    
    return stats


def extract_frames_uniform(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_frames: int = 100,
    **kwargs
) -> dict:
    """
    Extract a uniform number of frames across the video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        num_frames: Number of frames to extract (uniformly distributed)
        **kwargs: Additional arguments passed to extract_frames
    
    Returns:
        Extraction statistics
    """
    video_path = Path(video_path)
    
    # Get total frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate interval
    interval = max(1, total_frames // num_frames)
    
    return extract_frames(video_path, output_dir, interval=interval, max_frames=num_frames, **kwargs)


def batch_extract(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path],
    interval: int = 30,
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
    **kwargs
) -> list[dict]:
    """
    Extract frames from all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Base output directory (subdirectories created per video)
        interval: Frame interval
        extensions: Video file extensions to process
        **kwargs: Additional arguments passed to extract_frames
    
    Returns:
        List of extraction statistics for each video
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]
    
    print(f"Found {len(videos)} videos to process")
    
    all_stats = []
    
    for video_path in videos:
        video_output = output_dir / video_path.stem
        
        try:
            stats = extract_frames(video_path, video_output, interval=interval, **kwargs)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            all_stats.append({"video": str(video_path), "error": str(e)})
    
    return all_stats


# ============================================================
# Jupyter Notebook Example Usage
# ============================================================

NOTEBOOK_EXAMPLE = """
# Frame Extraction for PoseLabeler
# Run this notebook to extract frames from your poultry videos

# %% [markdown]
# ## Setup

# %%
from extract_frames import extract_frames, extract_frames_uniform, batch_extract
from pathlib import Path

# %% [markdown]
# ## Configuration

# %%
# Single video extraction
VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_DIR = "path/to/output/frames"

# Extraction settings
INTERVAL = 30  # Extract every 30th frame (1 frame/second at 30fps)
# Or use these for time-based extraction:
START_TIME = None  # Start from beginning (or set to seconds, e.g., 60.0)
END_TIME = None    # Extract to end (or set to seconds, e.g., 300.0)
MAX_FRAMES = None  # No limit (or set maximum number of frames)

# %% [markdown]
# ## Extract Frames from Single Video

# %%
stats = extract_frames(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    interval=INTERVAL,
    start_time=START_TIME,
    end_time=END_TIME,
    max_frames=MAX_FRAMES,
    format="jpg",
    quality=95
)

print(f"Extracted {stats['extracted_frames']} frames")

# %% [markdown]
# ## Alternative: Extract Uniform Number of Frames

# %%
# This extracts exactly 200 frames uniformly distributed across the video
stats = extract_frames_uniform(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    num_frames=200
)

# %% [markdown]
# ## Batch Process Multiple Videos

# %%
VIDEO_DIR = "path/to/video/folder"
OUTPUT_BASE = "path/to/output"

all_stats = batch_extract(
    video_dir=VIDEO_DIR,
    output_dir=OUTPUT_BASE,
    interval=60  # Every 60 frames
)

for s in all_stats:
    if "error" not in s:
        print(f"{Path(s['video']).name}: {s['extracted_frames']} frames")
    else:
        print(f"{Path(s['video']).name}: ERROR - {s['error']}")
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for pose annotation")
    
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for frames")
    parser.add_argument("--interval", "-i", type=int, default=30, 
                        help="Extract every Nth frame (default: 30)")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--max-frames", "-m", type=int, help="Maximum frames to extract")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="Resize frames to WxH")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Output format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    
    args = parser.parse_args()
    
    resize = tuple(args.resize) if args.resize else None
    
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval=args.interval,
        start_time=args.start,
        end_time=args.end,
        max_frames=args.max_frames,
        resize=resize,
        format=args.format,
        quality=args.quality,
    )
