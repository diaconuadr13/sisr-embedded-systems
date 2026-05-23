import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic video SR dataset.")
    parser.add_argument("--output", type=str, default="data/video_toy")
    parser.add_argument("--num_clips", type=int, default=4)
    parser.add_argument("--frames_per_clip", type=int, default=12)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    return parser.parse_args()


def make_frame(clip_idx: int, frame_idx: int, height: int, width: int) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = (x + 0.08 * clip_idx) % 1.0
    base[..., 1] = (y + 0.05 * frame_idx) % 1.0
    base[..., 2] = 0.5 * (np.sin((x + y) * np.pi * (clip_idx + 1)) + 1.0)

    square_size = max(10, min(height, width) // 6)
    sx = (frame_idx * 7 + clip_idx * 13) % max(1, width - square_size)
    sy = (frame_idx * 5 + clip_idx * 11) % max(1, height - square_size)
    base[sy:sy + square_size, sx:sx + square_size] = np.array([1.0, 0.15, 0.1], dtype=np.float32)

    center = ((frame_idx * 9 + clip_idx * 17) % width, (height // 2 + int(np.sin(frame_idx * 0.7) * height * 0.25)) % height)
    cv2.circle(base, center, max(5, min(height, width) // 12), (0.1, 0.9, 0.35), thickness=-1)

    rng = np.random.default_rng(seed=clip_idx * 1000 + frame_idx)
    noise = rng.normal(0.0, 0.025, size=base.shape).astype(np.float32)
    img = np.clip(base + noise, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def write_split(output: Path, split: str, num_clips: int, frames_per_clip: int, height: int, width: int) -> None:
    for clip_idx in range(num_clips):
        clip_dir = output / split / f"clip_{clip_idx:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(frames_per_clip):
            frame = make_frame(clip_idx + (100 if split == "val" else 0), frame_idx, height, width)
            cv2.imwrite(str(clip_dir / f"frame_{frame_idx:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    write_split(output, "train", args.num_clips, args.frames_per_clip, args.height, args.width)
    write_split(output, "val", max(1, args.num_clips // 2), args.frames_per_clip, args.height, args.width)
    print(f"Created toy video dataset at {output}")


if __name__ == "__main__":
    main()
