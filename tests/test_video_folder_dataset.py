from pathlib import Path

import cv2
import numpy as np

from utils.dataset import VideoFolderSRDataset


def _write_clip(root: Path, name: str, frames: int = 4, width: int = 32, height: int = 24) -> None:
    clip_dir = root / name
    clip_dir.mkdir(parents=True)
    for idx in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[..., 0] = idx * 20
        img[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
        img[..., 2] = np.arange(height, dtype=np.uint8)[:, None]
        cv2.imwrite(str(clip_dir / f"frame_{idx:03d}.png"), img)


def test_video_folder_dataset_returns_padded_temporal_windows(tmp_path: Path) -> None:
    _write_clip(tmp_path, "clip_a")
    dataset = VideoFolderSRDataset(str(tmp_path), scale=2, num_frames=5, grayscale=True)

    lr, hr = dataset[0]

    assert len(dataset) == 4
    assert tuple(lr.shape) == (5, 1, 12, 16)
    assert tuple(hr.shape) == (1, 24, 32)
    assert dataset.sample_info(0)["clip_name"] == "clip_a"
