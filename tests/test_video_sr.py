import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from models.video_espcn import VideoESPCN, VideoESPCNMicro
from utils.video_dataset import VideoSISRDataset
from utils.video_io import make_temporal_window


def _write_frame(path: Path, value: int, shape=(64, 64, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full(shape, value, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def test_video_espcn_output_shape() -> None:
    model = VideoESPCN(scale_factor=2, num_channels=3, num_frames=3, hidden_channels=8)
    y = model(torch.randn(2, 9, 16, 20))
    assert y.shape == (2, 3, 32, 40)


def test_video_espcn_micro_output_shape() -> None:
    model = VideoESPCNMicro(scale_factor=2)
    y = model(torch.randn(2, 3, 16, 20))
    assert y.shape == (2, 1, 32, 40)


def test_video_dataset_shapes(tmp_path: Path) -> None:
    clip = tmp_path / "clip_000"
    for idx in range(5):
        _write_frame(clip / f"frame_{idx:03d}.png", idx * 20)
    dataset = VideoSISRDataset(str(tmp_path), scale=2, patch_size=32, num_frames=3, random_crop=False, augment=False)
    lr, hr = dataset[0]
    assert lr.shape == (9, 16, 16)
    assert hr.shape == (3, 32, 32)


def test_temporal_window_clamps_boundaries() -> None:
    frames = [np.array([idx], dtype=np.uint8) for idx in range(3)]
    left = make_temporal_window(frames, 0, 5)
    right = make_temporal_window(frames, 2, 5)
    assert [int(item[0]) for item in left] == [0, 0, 0, 1, 2]
    assert [int(item[0]) for item in right] == [0, 1, 2, 2, 2]


def test_toy_dataset_generator_creates_files(tmp_path: Path) -> None:
    output = tmp_path / "video_toy"
    script = Path(__file__).resolve().parents[1] / "tools" / "create_toy_video_dataset.py"
    subprocess.run([sys.executable, str(script), "--output", str(output), "--num_clips", "2", "--frames_per_clip", "4"], check=True)
    assert (output / "train" / "clip_000" / "frame_000.png").exists()
    assert (output / "val" / "clip_000" / "frame_000.png").exists()
