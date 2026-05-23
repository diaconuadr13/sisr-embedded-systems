from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoSISRDataset(Dataset):
    """Video SISR dataset where temporal frames are concatenated as channels."""

    def __init__(
        self,
        root_dir: str,
        scale: int,
        patch_size: int,
        num_frames: int = 3,
        grayscale: bool = False,
        split: str = "train",
        random_crop: bool = True,
        augment: bool = True,
        samples_per_epoch: Optional[int] = None,
        cache_in_memory: bool = False,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        self.num_frames = int(num_frames)
        self.grayscale = bool(grayscale)
        self.split = split
        self.random_crop = bool(random_crop)
        self.augment = bool(augment)
        self.samples_per_epoch = samples_per_epoch
        self.extensions = tuple(ext.lower() for ext in extensions)

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.patch_size % self.scale != 0:
            raise ValueError("patch_size must be divisible by scale")
        if self.num_frames < 1 or self.num_frames % 2 == 0:
            raise ValueError("num_frames must be odd and >= 1")
        if samples_per_epoch is not None and int(samples_per_epoch) <= 0:
            raise ValueError("samples_per_epoch must be > 0")

        self.clips = self._discover_clips()
        if not self.clips:
            raise FileNotFoundError(f"No usable video clips found in: {self.root_dir}")

        self.index: list[tuple[int, int]] = [
            (clip_idx, frame_idx)
            for clip_idx, clip in enumerate(self.clips)
            for frame_idx in range(len(clip))
        ]
        if not self.index:
            raise FileNotFoundError(f"No usable frames found in: {self.root_dir}")

        self._cache: Optional[list[list[np.ndarray]]] = None
        if cache_in_memory:
            print(f"[video_dataset] Caching {len(self.clips)} clips from {self.root_dir}...", flush=True)
            self._cache = [[self._read_frame(path) for path in clip] for clip in self.clips]
            print("[video_dataset] Cache ready.", flush=True)

    def _discover_clips(self) -> list[list[Path]]:
        def list_frames(directory: Path) -> list[Path]:
            return sorted(
                p
                for p in directory.iterdir()
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in self.extensions
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Video dataset directory does not exist: {self.root_dir}")

        flat_frames = list_frames(self.root_dir)
        if flat_frames:
            return [flat_frames]

        clips: list[list[Path]] = []
        for directory in sorted(p for p in self.root_dir.rglob("*") if p.is_dir() and not p.name.startswith(".")):
            frames = list_frames(directory)
            if frames:
                clips.append(frames)
        return clips

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return int(self.samples_per_epoch)
        return len(self.index)

    def _read_frame(self, path: Path) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        if self.grayscale:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_frame(self, clip_idx: int, frame_idx: int) -> np.ndarray:
        if self._cache is not None:
            return self._cache[clip_idx][frame_idx]
        return self._read_frame(self.clips[clip_idx][frame_idx])

    def _load_window(self, clip_idx: int, center_idx: int) -> list[np.ndarray]:
        clip = self.clips[clip_idx]
        last = len(clip) - 1
        radius = self.num_frames // 2
        frames = []
        for offset in range(-radius, radius + 1):
            idx = min(max(center_idx + offset, 0), last)
            frames.append(self._get_frame(clip_idx, idx))
        return frames

    def _crop_window(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        h, w = frames[0].shape[:2]
        if h < self.patch_size or w < self.patch_size:
            new_h = max(h, self.patch_size)
            new_w = max(w, self.patch_size)
            frames = [
                cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                for frame in frames
            ]
            h, w = frames[0].shape[:2]

        if self.random_crop:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        return [frame[top : top + self.patch_size, left : left + self.patch_size] for frame in frames]

    def _augment_window(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not self.augment:
            return frames
        if np.random.rand() < 0.5:
            frames = [np.flip(frame, axis=1) for frame in frames]
        if np.random.rand() < 0.5:
            frames = [np.flip(frame, axis=0) for frame in frames]
        k = int(np.random.randint(0, 4))
        if k:
            frames = [np.rot90(frame, k=k, axes=(0, 1)) for frame in frames]
        return frames

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(img)
        if self.grayscale:
            return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_idx, frame_idx = self.index[index % len(self.index)]
        hr_window = self._augment_window(self._crop_window(self._load_window(clip_idx, frame_idx)))
        center = self.num_frames // 2
        hr_target = hr_window[center]

        lr_size = self.patch_size // self.scale
        lr_frames = [
            cv2.resize(frame, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
            for frame in hr_window
        ]

        lr_input = torch.cat([self._to_tensor(frame) for frame in lr_frames], dim=0)
        hr_tensor = self._to_tensor(hr_target)
        return lr_input, hr_tensor
