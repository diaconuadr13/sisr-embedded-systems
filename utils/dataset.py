from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_image_paths(image_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_grayscale_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise RuntimeError(f"Unsupported image shape for grayscale conversion: {path} {img.shape}")


class SISRDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int, patch_size: int,
                 cache_in_memory: bool = True,
                 grayscale: bool = False) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        self.grayscale = grayscale

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.patch_size % self.scale != 0:
            raise ValueError("patch_size must be divisible by scale")

        self.image_paths = _list_image_paths(self.hr_dir)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.hr_dir}")

        # Pre-decode images into RAM. For large datasets, downsample to
        # CACHE_MAX_SIDE before storing so the cache stays within RAM limits
        # while still providing patches much larger than the training crop.
        CACHE_MAX_SIDE = 384
        self._cache: Optional[List[np.ndarray]] = None
        n = len(self.image_paths)
        if cache_in_memory:
            print(f"[dataset] Caching {n} images from {self.hr_dir.name} (max side {CACHE_MAX_SIDE}px)...", flush=True)
            self._cache = []
            for i, p in enumerate(self.image_paths):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {p}")
                h, w = img.shape[:2]
                if max(h, w) > CACHE_MAX_SIDE:
                    scale_factor = CACHE_MAX_SIDE / max(h, w)
                    img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
                if self.grayscale:
                    self._cache.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                else:
                    self._cache.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if (i + 1) % 200 == 0:
                    print(f"[dataset]   {i + 1}/{n}", flush=True)
            print(f"[dataset] Cache ready.", flush=True)

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
            if self.grayscale:
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
            else:
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

        if self.grayscale:
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).unsqueeze(0).float() / 255.0
            hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_crop)).unsqueeze(0).float() / 255.0
        else:
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float() / 255.0
            hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_crop)).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor


class ThermalFullFrameSISRDataset(Dataset):
    """Full-frame grayscale SISR dataset for native 32x24 thermal images.

    Each HR target is the original thermal frame resized, if needed, to
    hr_height x hr_width. The LR input is generated by downsampling the full HR
    frame by scale, preserving the rectangular aspect ratio.
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int,
        hr_height: int = 24,
        hr_width: int = 32,
        split: Optional[str] = None,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        cache_in_memory: bool = True,
    ) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        self.hr_height = int(hr_height)
        self.hr_width = int(hr_width)

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.hr_height <= 0 or self.hr_width <= 0:
            raise ValueError("hr_height and hr_width must be > 0")
        if self.hr_height % self.scale != 0 or self.hr_width % self.scale != 0:
            raise ValueError("thermal HR dimensions must be divisible by scale")

        split_name = split.lower() if split is not None else None
        if split_name not in {None, "train", "val"}:
            raise ValueError("split must be one of: None, 'train', 'val'")
        if not 0.0 < float(val_fraction) < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")

        paths = _list_image_paths(self.hr_dir)
        if not paths:
            raise FileNotFoundError(f"No images found in: {self.hr_dir}")

        if split_name is not None:
            if len(paths) < 2:
                raise ValueError("At least two images are required for train/val splitting")
            rng = np.random.default_rng(int(split_seed))
            indices = np.arange(len(paths))
            rng.shuffle(indices)
            val_count = int(round(len(paths) * float(val_fraction)))
            val_count = min(max(val_count, 1), len(paths) - 1)
            val_indices = set(int(i) for i in indices[:val_count])
            if split_name == "val":
                paths = [p for i, p in enumerate(paths) if i in val_indices]
            else:
                paths = [p for i, p in enumerate(paths) if i not in val_indices]

        self.image_paths = paths
        self.lr_height = self.hr_height // self.scale
        self.lr_width = self.hr_width // self.scale

        self._cache: Optional[List[np.ndarray]] = None
        if cache_in_memory:
            print(
                f"[dataset] Caching {len(self.image_paths)} thermal frames from "
                f"{self.hr_dir.name} as {self.hr_width}x{self.hr_height} grayscale...",
                flush=True,
            )
            self._cache = [self._load_hr_image(path) for path in self.image_paths]
            print("[dataset] Thermal cache ready.", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_hr_image(self, path: Path) -> np.ndarray:
        img = _read_grayscale_image(path)
        h, w = img.shape[:2]
        if h != self.hr_height or w != self.hr_width:
            interpolation = cv2.INTER_AREA if h > self.hr_height or w > self.hr_width else cv2.INTER_CUBIC
            img = cv2.resize(img, (self.hr_width, self.hr_height), interpolation=interpolation)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            hr_img = self._cache[index]
        else:
            hr_img = self._load_hr_image(self.image_paths[index])

        lr_img = cv2.resize(
            hr_img,
            (self.lr_width, self.lr_height),
            interpolation=cv2.INTER_AREA,
        )

        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).unsqueeze(0).float() / 255.0
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_img)).unsqueeze(0).float() / 255.0
        return lr_tensor, hr_tensor
