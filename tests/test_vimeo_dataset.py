from pathlib import Path

import cv2
import numpy as np

from utils.dataset import RawVimeo90KVideoSRDataset


def _write_vimeo_clip(root: Path, width: int = 448, height: int = 256) -> None:
    clip_dir = root / "sequence" / "00001" / "0001"
    clip_dir.mkdir(parents=True)
    (root / "sep_trainlist.txt").write_text("00001/0001\n", encoding="utf-8")
    (root / "sep_testlist.txt").write_text("00001/0001\n", encoding="utf-8")

    for idx in range(1, 8):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = idx * 20
        frame[..., 1] = np.arange(width, dtype=np.uint16)[None, :] % 256
        frame[..., 2] = np.arange(height, dtype=np.uint16)[:, None] % 256
        cv2.imwrite(str(clip_dir / f"im{idx}.png"), frame)


def test_raw_vimeo_preserves_hr_resolution_and_downscales_lr(tmp_path: Path) -> None:
    _write_vimeo_clip(tmp_path)
    dataset = RawVimeo90KVideoSRDataset(
        root_dir=str(tmp_path),
        split="train",
        scale=2,
        num_frames=3,
        patch_size=96,
        grayscale=True,
        random_crop=True,
        augment=True,
    )

    lr, hr = dataset[0]

    assert tuple(hr.shape) == (1, 256, 448)
    assert tuple(lr.shape) == (3, 1, 128, 224)
