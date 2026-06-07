from __future__ import annotations

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


def _read_color_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if grayscale:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


class PairedImageSISRDataset(Dataset):
    """Paired LR/HR image dataset for real-world image super-resolution.

    The dataset expects matching filenames or stems in separate LR and HR
    folders. In full-frame mode, the LR image and HR target are used at their
    original sizes. A shape mismatch raises an error instead of resizing data.
    """

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        scale: int,
        patch_size: int,
        grayscale: bool = False,
        random_crop: bool = True,
        augment: bool = True,
        full_frame: bool = False,
        cache_in_memory: bool = False,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        self.grayscale = bool(grayscale)
        self.random_crop = bool(random_crop)
        self.augment = bool(augment)
        self.full_frame = bool(full_frame)
        self.extensions = tuple(ext.lower() for ext in extensions)

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if not self.full_frame and self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if not self.full_frame and self.patch_size % self.scale != 0:
            raise ValueError("patch_size must be divisible by scale")
        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory does not exist: {self.lr_dir}")
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {self.hr_dir}")

        lr_paths = self._list_paths(self.lr_dir)
        hr_paths = self._list_paths(self.hr_dir)
        self.pairs = self._match_pairs(lr_paths, hr_paths)
        if not self.pairs:
            raise FileNotFoundError(f"No matching LR/HR image pairs found in {self.lr_dir} and {self.hr_dir}")

        self._cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        if cache_in_memory:
            print(f"[dataset] Caching {len(self.pairs)} paired LR/HR images...", flush=True)
            self._cache = [(self._load_image(lr), self._load_image(hr)) for lr, hr in self.pairs]
            print("[dataset] Paired cache ready.", flush=True)

    def _list_paths(self, directory: Path) -> List[Path]:
        return sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in self.extensions
        )

    def _match_pairs(self, lr_paths: List[Path], hr_paths: List[Path]) -> List[Tuple[Path, Path]]:
        hr_by_rel = {p.relative_to(self.hr_dir).with_suffix("").as_posix(): p for p in hr_paths}
        pairs = []
        for lr_path in lr_paths:
            key = lr_path.relative_to(self.lr_dir).with_suffix("").as_posix()
            if key in hr_by_rel:
                pairs.append((lr_path, hr_by_rel[key]))
        if pairs:
            return pairs

        hr_by_stem = {p.stem: p for p in hr_paths}
        pairs = [(lr_path, hr_by_stem[lr_path.stem]) for lr_path in lr_paths if lr_path.stem in hr_by_stem]
        if pairs:
            return pairs

        if len(lr_paths) == len(hr_paths):
            return list(zip(lr_paths, hr_paths))
        return []

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if self.grayscale:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _augment_pair(
        self,
        lr_img: np.ndarray,
        hr_img: np.ndarray,
        rotate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return lr_img, hr_img
        if np.random.rand() < 0.5:
            lr_img = np.flip(lr_img, axis=1)
            hr_img = np.flip(hr_img, axis=1)
        if np.random.rand() < 0.5:
            lr_img = np.flip(lr_img, axis=0)
            hr_img = np.flip(hr_img, axis=0)
        if rotate:
            k = int(np.random.randint(0, 4))
            if k:
                lr_img = np.rot90(lr_img, k=k, axes=(0, 1))
                hr_img = np.rot90(hr_img, k=k, axes=(0, 1))
        return lr_img, hr_img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(img)
        if self.grayscale:
            return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            lr_img, hr_img = self._cache[index]
        else:
            lr_path, hr_path = self.pairs[index]
            lr_img = self._load_image(lr_path)
            hr_img = self._load_image(hr_path)

        if self.full_frame:
            lr_h, lr_w = lr_img.shape[:2]
            target_h = lr_h * self.scale
            target_w = lr_w * self.scale
            hr_h, hr_w = hr_img.shape[:2]
            if (hr_h, hr_w) != (target_h, target_w):
                raise ValueError(
                    "Full-frame paired SR requires HR dimensions to equal LR dimensions "
                    f"multiplied by scale. Got LR={lr_w}x{lr_h}, HR={hr_w}x{hr_h}, "
                    f"scale={self.scale}, expected HR={target_w}x{target_h}."
                )
            lr_img, hr_img = self._augment_pair(lr_img, hr_img, rotate=False)
            return self._to_tensor(lr_img), self._to_tensor(hr_img)

        hr_h, hr_w = hr_img.shape[:2]
        if hr_h < self.patch_size or hr_w < self.patch_size:
            new_h = max(hr_h, self.patch_size)
            new_w = max(hr_w, self.patch_size)
            hr_img = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            hr_h, hr_w = hr_img.shape[:2]

        if self.random_crop:
            top = np.random.randint(0, hr_h - self.patch_size + 1)
            left = np.random.randint(0, hr_w - self.patch_size + 1)
        else:
            top = (hr_h - self.patch_size) // 2
            left = (hr_w - self.patch_size) // 2
        hr_crop = hr_img[top : top + self.patch_size, left : left + self.patch_size]

        lr_h, lr_w = lr_img.shape[:2]
        lr_top = int(round(top * lr_h / hr_h))
        lr_left = int(round(left * lr_w / hr_w))
        lr_crop_h = max(1, int(round(self.patch_size * lr_h / hr_h)))
        lr_crop_w = max(1, int(round(self.patch_size * lr_w / hr_w)))
        lr_top = min(max(lr_top, 0), max(lr_h - lr_crop_h, 0))
        lr_left = min(max(lr_left, 0), max(lr_w - lr_crop_w, 0))
        lr_crop = lr_img[lr_top : lr_top + lr_crop_h, lr_left : lr_left + lr_crop_w]

        lr_size = self.patch_size // self.scale
        lr_crop = cv2.resize(lr_crop, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        lr_crop, hr_crop = self._augment_pair(lr_crop, hr_crop)
        return self._to_tensor(lr_crop), self._to_tensor(hr_crop)


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


class RawVimeo90KVideoSRDataset(Dataset):
    """Read Vimeo-90K septuplets directly and synthesize LR clips in memory.

    The raw PNG files are never copied, resized, overwritten, or otherwise
    modified on disk. Each sample returns an LR frame window with shape
    T,C,H,W and the HR center-frame target at the decoded Vimeo frame size.
    HR frames are never cropped or resized in memory; only LR frames are
    downscaled from the full-resolution HR frames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        scale: int,
        num_frames: int = 5,
        patch_size: int = 96,
        grayscale: bool = False,
        random_crop: bool = True,
        augment: bool = True,
        samples_per_epoch: Optional[int] = None,
        split_list: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.scale = int(scale)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.grayscale = bool(grayscale)
        self.random_crop = bool(random_crop)
        self.augment = bool(augment)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else None

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.num_frames <= 0 or self.num_frames % 2 == 0:
            raise ValueError("num_frames must be a positive odd number")
        if self.patch_size < 0:
            raise ValueError("patch_size must be >= 0")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Vimeo-90K root does not exist: {self.root_dir}")

        self.sequence_dir = self._find_sequence_dir(self.root_dir)
        self.clip_dirs = self._load_clip_dirs(split_list)
        if not self.clip_dirs:
            raise FileNotFoundError(f"No valid Vimeo-90K clips found for split={self.split} in {self.root_dir}")

    def _find_sequence_dir(self, root_dir: Path) -> Path:
        candidates = [root_dir / "sequences", root_dir / "sequence"]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not find Vimeo-90K sequence directory under {root_dir}. "
            "Expected 'sequences' or Kaggle mirror 'sequence'."
        )

    def _default_split_list(self) -> Path:
        if self.split == "train":
            return self.root_dir / "sep_trainlist.txt"
        return self.root_dir / "sep_testlist.txt"

    def _load_clip_dirs(self, split_list: Optional[str]) -> List[Path]:
        list_path = Path(split_list) if split_list else self._default_split_list()
        clip_dirs: List[Path] = []
        if list_path.exists():
            with list_path.open("r", encoding="utf-8") as f:
                rel_paths = [line.strip() for line in f if line.strip()]
            for rel in rel_paths:
                clip_dir = self.sequence_dir / rel
                if clip_dir.is_dir() and len(self._frame_paths(clip_dir)) >= self.num_frames:
                    clip_dirs.append(clip_dir)
            return clip_dirs

        for clip_dir in sorted(p for p in self.sequence_dir.glob("*/*") if p.is_dir()):
            if len(self._frame_paths(clip_dir)) >= self.num_frames:
                clip_dirs.append(clip_dir)
        return clip_dirs

    def _frame_paths(self, clip_dir: Path) -> List[Path]:
        def key(path: Path) -> Tuple[int, str]:
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            return (int(digits) if digits else 0, path.name)

        return sorted(
            (p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=key,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.clip_dirs)

    def _select_clip_dir(self, index: int) -> Path:
        if self.samples_per_epoch:
            if self.random_crop:
                return self.clip_dirs[int(np.random.randint(0, len(self.clip_dirs)))]
            return self.clip_dirs[index % len(self.clip_dirs)]
        return self.clip_dirs[index]

    def _load_frame_window(self, clip_dir: Path) -> List[np.ndarray]:
        frames = self._frame_paths(clip_dir)
        if len(frames) < self.num_frames:
            raise RuntimeError(f"Clip {clip_dir} has fewer than {self.num_frames} frames")
        start = (len(frames) - self.num_frames) // 2
        selected = frames[start : start + self.num_frames]
        return [_read_color_image(path, grayscale=self.grayscale) for path in selected]

    def _validate_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        h, w = frames[0].shape[:2]
        for frame in frames[1:]:
            if frame.shape[:2] != (h, w):
                raise ValueError(
                    "All frames in a Vimeo-90K clip must have the same dimensions. "
                    f"Got {w}x{h} and {frame.shape[1]}x{frame.shape[0]}."
                )
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Vimeo-90K frame has invalid dimensions: {w}x{h}."
            )
        return frames

    def _augment_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        return frames

    def _downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        lr_h = h // self.scale
        lr_w = w // self.scale
        return cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    def _to_frame_tensor(self, img: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(img)
        if self.grayscale:
            return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_dir = self._select_clip_dir(index)
        hr_frames = self._augment_clip(self._validate_clip(self._load_frame_window(clip_dir)))
        lr_frames = [self._downscale_frame(frame) for frame in hr_frames]

        lr_tensor = torch.stack([self._to_frame_tensor(frame) for frame in lr_frames], dim=0)
        center = self.num_frames // 2
        hr_tensor = self._to_frame_tensor(hr_frames[center])
        return lr_tensor, hr_tensor


class VideoFolderSRDataset(Dataset):
    """Video-folder evaluation dataset with synthetic bicubic LR inputs.

    Expected layout:
        root/clip_name/frame_000.png
        root/clip_name/frame_001.png

    Each sample returns an LR temporal window shaped T,C,H,W and the HR center
    frame shaped C,H,W. Windows are edge-padded at clip boundaries so every HR
    frame can be evaluated.
    """

    def __init__(
        self,
        root_dir: str,
        scale: int,
        num_frames: int = 5,
        grayscale: bool = True,
        patch_size: int = 0,
        random_crop: bool = False,
        include_all_frames: bool = True,
        min_size: int = 16,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.scale = int(scale)
        self.num_frames = int(num_frames)
        self.grayscale = bool(grayscale)
        self.patch_size = int(patch_size)
        self.random_crop = bool(random_crop)
        self.include_all_frames = bool(include_all_frames)
        self.min_size = int(min_size)

        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.num_frames <= 0 or self.num_frames % 2 == 0:
            raise ValueError("num_frames must be a positive odd number")
        if self.patch_size < 0:
            raise ValueError("patch_size must be >= 0")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Video folder root does not exist: {self.root_dir}")

        self.clips = self._load_clips()
        self.samples: List[Tuple[int, int]] = []
        for clip_idx, (_clip_name, frames) in enumerate(self.clips):
            if include_all_frames:
                self.samples.extend((clip_idx, frame_idx) for frame_idx in range(len(frames)))
            else:
                self.samples.append((clip_idx, len(frames) // 2))
        if not self.samples:
            raise FileNotFoundError(f"No video frames found under: {self.root_dir}")

    def _natural_key(self, path: Path) -> Tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 0, path.name.lower())

    def _load_clips(self) -> List[Tuple[str, List[Path]]]:
        clips: List[Tuple[str, List[Path]]] = []
        for clip_dir in sorted((p for p in self.root_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            frames = sorted(
                (
                    p
                    for p in clip_dir.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=self._natural_key,
            )
            if frames:
                clips.append((clip_dir.name, frames))
        return clips

    def __len__(self) -> int:
        return len(self.samples)

    def sample_info(self, index: int) -> dict:
        clip_idx, frame_idx = self.samples[index]
        clip_name, frames = self.clips[clip_idx]
        return {
            "clip_name": clip_name,
            "frame_index": frame_idx,
            "frame_path": str(frames[frame_idx]),
        }

    def _window_indices(self, center_idx: int, clip_len: int) -> List[int]:
        half = self.num_frames // 2
        return [min(max(i, 0), clip_len - 1) for i in range(center_idx - half, center_idx + half + 1)]

    def _validate_frames(self, frames: List[np.ndarray]) -> None:
        h, w = frames[0].shape[:2]
        if h < self.min_size or w < self.min_size:
            raise ValueError(f"Frame is too small for evaluation: {w}x{h}, min_size={self.min_size}")
        if h // self.scale <= 0 or w // self.scale <= 0:
            raise ValueError(f"Frame is too small for scale x{self.scale}: {w}x{h}")
        for frame in frames[1:]:
            if frame.shape[:2] != (h, w):
                raise ValueError("All frames in a clip must have identical dimensions")

    def _crop_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        h, w = frames[0].shape[:2]
        crop_h = h - (h % self.scale)
        crop_w = w - (w % self.scale)
        if self.patch_size > 0:
            crop_h = min(crop_h, self.patch_size)
            crop_w = min(crop_w, self.patch_size)
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"Invalid crop for scale x{self.scale}: {w}x{h}")
        if self.random_crop and (h > crop_h or w > crop_w):
            top = int(np.random.randint(0, h - crop_h + 1))
            left = int(np.random.randint(0, w - crop_w + 1))
        else:
            top = max(0, (h - crop_h) // 2)
            left = max(0, (w - crop_w) // 2)
        return [frame[top : top + crop_h, left : left + crop_w] for frame in frames]

    def _downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(img)
        if self.grayscale:
            return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_idx, frame_idx = self.samples[index]
        _clip_name, paths = self.clips[clip_idx]
        indices = self._window_indices(frame_idx, len(paths))
        hr_frames = [_read_color_image(paths[i], grayscale=self.grayscale) for i in indices]
        self._validate_frames(hr_frames)
        hr_frames = self._crop_frames(hr_frames)
        lr_frames = [self._downscale_frame(frame) for frame in hr_frames]
        center = self.num_frames // 2
        return (
            torch.stack([self._to_tensor(frame) for frame in lr_frames], dim=0),
            self._to_tensor(hr_frames[center]),
        )
