import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from models import get_model


def _write_dataset(root: Path) -> None:
    for clip in ("clip_a", "clip_b"):
        clip_dir = root / clip
        clip_dir.mkdir(parents=True)
        for idx in range(3):
            img = np.full((16, 16, 3), idx * 40, dtype=np.uint8)
            cv2.imwrite(str(clip_dir / f"frame_{idx:03d}.png"), img)


def test_evaluate_vsr_on_synthetic_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "ToyVSR"
    out_dir = tmp_path / "report"
    checkpoint = tmp_path / "model.pth"
    _write_dataset(data_root)

    model = get_model(
        "VideoESPCN",
        scale=2,
        device=torch.device("cpu"),
        num_channels=1,
        num_frames=3,
        hidden_channels=4,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "VideoESPCN",
            "scale": 2,
            "config": {"arch": "VideoESPCN", "hidden_channels": 4, "num_frames": 3, "grayscale": True},
        },
        checkpoint,
    )

    subprocess.run(
        [
            sys.executable,
            "evaluate_vsr.py",
            "--checkpoint",
            str(checkpoint),
            "--video-root",
            str(data_root),
            "--dataset-name",
            "ToyVSR",
            "--scale",
            "2",
            "--num-frames",
            "3",
            "--arch",
            "VideoESPCN",
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert any((out_dir / "samples").glob("*.png"))
