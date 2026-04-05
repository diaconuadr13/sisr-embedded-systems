from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SISRDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int, patch_size: int,
                 cache_in_memory: bool = True) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        self.patch_size = int(patch_size)

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.patch_size % self.scale != 0:
            raise ValueError("patch_size must be divisible by scale")

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self.image_paths: List[Path] = sorted(
            p for p in self.hr_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.hr_dir}")

        # Pre-decode all images into RAM to avoid repeated disk I/O + PNG decode
        self._cache: Optional[List[np.ndarray]] = None
        if cache_in_memory:
            self._cache = []
            for p in self.image_paths:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {p}")
                self._cache.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            hr_img = self._cache[index]
        else:
            hr_path = self.image_paths[index]
            hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
            if hr_img is None:
                raise RuntimeError(f"Failed to read image: {hr_path}")
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        h, w = hr_img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            new_h = max(h, self.patch_size)
            new_w = max(w, self.patch_size)
            hr_img = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = hr_img.shape[:2]

        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        hr_crop = hr_img[top : top + self.patch_size, left : left + self.patch_size]

        lr_size = self.patch_size // self.scale
        lr_img = cv2.resize(hr_crop, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)

        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_crop)).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor
