from pathlib import Path
from typing import Optional

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def list_frame_paths(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in FRAME_EXTENSIONS
    )


def read_video_frames(path: Path, max_frames: Optional[int] = None) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 30.0

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames, fps


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float, codec: str = "mp4v") -> None:
    if not frames_rgb:
        raise ValueError("frames_rgb must contain at least one frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame_rgb in frames_rgb:
            if frame_rgb.shape[:2] != (height, width):
                raise ValueError("All video frames must have the same spatial size")
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_frames(directory: Path, frames_rgb: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx, frame_rgb in enumerate(frames_rgb):
        out_path = directory / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def make_temporal_window(frames, center_idx: int, num_frames: int) -> list[np.ndarray]:
    if num_frames <= 0 or num_frames % 2 == 0:
        raise ValueError("num_frames must be odd and >= 1")
    if not frames:
        raise ValueError("frames must contain at least one frame")

    radius = num_frames // 2
    last = len(frames) - 1
    return [frames[min(max(center_idx + offset, 0), last)] for offset in range(-radius, radius + 1)]
