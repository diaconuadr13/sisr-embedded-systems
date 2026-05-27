from pathlib import Path

import cv2
import numpy as np

from tools.download_vsr_datasets import DatasetSpec, normalize_dataset, validate_normalized_dataset


def _write_source_clip(root: Path, name: str, frames: int = 3) -> None:
    clip_dir = root / "GT" / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(frames):
        img = np.full((16, 20, 3), idx * 30, dtype=np.uint8)
        cv2.imwrite(str(clip_dir / f"img_{idx}.png"), img)


def test_downloader_normalizes_synthetic_video_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_source_clip(source, "alpha")
    _write_source_clip(source, "beta")
    spec = DatasetSpec(
        canonical_name="ToyVSR",
        aliases=("toy",),
        expected_clips=2,
        min_frames_per_clip=3,
        min_size=(16, 16),
        size_label="test",
    )

    dataset_root = normalize_dataset(source, output, spec)
    summary = validate_normalized_dataset(dataset_root, spec)

    assert summary == {"clips": 2, "frames": 6}
    assert (dataset_root / "alpha" / "frame_000.png").exists()
    assert (dataset_root / "beta" / "frame_002.png").exists()
