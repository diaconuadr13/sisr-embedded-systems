"""On-demand super-resolution inference for uploaded images.

PyTorch imports are deferred to the first call so the webapp can start fast
and does not initialise CUDA unless the user actually runs inference.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import UPLOADS_DIR
from backend.services import run_scanner

_model_cache: dict[tuple[str, str], Any] = {}


@dataclass
class InferenceOutput:
    scale: int
    arch: str
    device: str
    amp_used: bool
    inference_ms: float

    lr_url: str
    bicubic_url: str
    sr_url: str
    hr_url: str | None

    lr_size: tuple[int, int]  # (w, h)
    sr_size: tuple[int, int]

    psnr_bicubic: float | None
    psnr_sr: float | None
    ssim_bicubic: float | None
    ssim_sr: float | None


def _load_torch():
    import cv2  # noqa: F401
    import numpy as np
    import torch

    return torch, np


def _resolve_checkpoint_metadata(ckpt_path: Path, ckpt: Any) -> tuple[str, int]:
    arch = None
    scale = None
    if isinstance(ckpt, dict):
        arch = ckpt.get("arch")
        scale = ckpt.get("scale")
    if arch is None or scale is None:
        # Fall back to training config.json in the run dir (one level up from checkpoint).
        cfg_path = ckpt_path.parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            arch = arch or cfg.get("arch")
            scale = scale or cfg.get("scale")
    if arch is None or scale is None:
        raise ValueError("Could not determine arch/scale from checkpoint or config.json")
    return str(arch), int(scale)


def _load_model(ckpt_path: Path, device_str: str):
    from models import get_model
    from utils.device import resolve_device

    torch, _ = _load_torch()
    cache_key = (str(ckpt_path), device_str)
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch, scale = _resolve_checkpoint_metadata(ckpt_path, ckpt)

    device = resolve_device(device_str)
    model = get_model(arch, scale=scale, device=device)

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[cache_key] = (model, arch, scale, device)
    return _model_cache[cache_key]


def _read_image(data: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a readable image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_rgb(img_rgb, dest: Path) -> None:
    import cv2

    dest.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dest), bgr)


def _tensor_from_rgb(img_rgb, device):
    torch, _ = _load_torch()
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t.to(device, non_blocking=True)


def _tensor_to_rgb(t):
    _, np = _load_torch()
    arr = t.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _autocast(use_amp: bool):
    import torch

    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _bicubic_upscale(lr_rgb, scale: int):
    import cv2

    h, w = lr_rgb.shape[:2]
    return cv2.resize(lr_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _bicubic_downscale(hr_rgb, scale: int):
    import cv2

    h, w = hr_rgb.shape[:2]
    # Crop so downscale divides evenly; the model output is a multiple of scale.
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    hr_rgb = hr_rgb[:h2, :w2]
    return cv2.resize(hr_rgb, (w2 // scale, h2 // scale), interpolation=cv2.INTER_CUBIC), hr_rgb


def _psnr_ssim(a_rgb, b_rgb) -> tuple[float, float]:
    """a, b are uint8 HxWx3 aligned to the same shape."""
    from utils.metrics import calculate_psnr, calculate_ssim

    torch, _ = _load_torch()
    # Shapes can differ by a pixel due to rounding — crop to common size.
    h = min(a_rgb.shape[0], b_rgb.shape[0])
    w = min(a_rgb.shape[1], b_rgb.shape[1])
    a = torch.from_numpy(a_rgb[:h, :w]).permute(2, 0, 1).float().div(255.0)
    b = torch.from_numpy(b_rgb[:h, :w]).permute(2, 0, 1).float().div(255.0)
    return calculate_psnr(a, b), calculate_ssim(a, b)


def run_inference(
    run_id: str,
    checkpoint_which: str,
    image_bytes: bytes,
    upload_kind: str,  # "lr" or "hr"
    device_str: str = "auto",
    use_amp: bool = False,
) -> InferenceOutput:
    """Run SR inference for the given run + checkpoint on an uploaded image.

    upload_kind="lr": image is already low-res; apply model directly.
    upload_kind="hr": treat as ground truth, bicubic-downscale by scale, then apply model;
                      report PSNR/SSIM against the original HR.
    """
    ckpt_path = run_scanner.checkpoint_path(run_id, checkpoint_which)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint '{checkpoint_which}' not found for run '{run_id}'.")

    torch, _np = _load_torch()
    model, arch, scale, device = _load_model(ckpt_path, device_str)
    amp_active = use_amp and device.type == "cuda"

    rgb = _read_image(image_bytes)

    hr_reference = None
    if upload_kind == "hr":
        lr_rgb, hr_reference = _bicubic_downscale(rgb, scale)
    elif upload_kind == "lr":
        lr_rgb = rgb
    else:
        raise ValueError("upload_kind must be 'lr' or 'hr'.")

    bicubic_rgb = _bicubic_upscale(lr_rgb, scale)

    lr_t = _tensor_from_rgb(lr_rgb, device)
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with _autocast(amp_active):
            sr_t = model(lr_t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    sr_rgb = _tensor_to_rgb(sr_t.float())

    psnr_b = ssim_b = psnr_s = ssim_s = None
    if hr_reference is not None:
        psnr_b, ssim_b = _psnr_ssim(bicubic_rgb, hr_reference)
        psnr_s, ssim_s = _psnr_ssim(sr_rgb, hr_reference)

    uid = uuid.uuid4().hex[:12]
    out_dir = UPLOADS_DIR / uid
    _save_rgb(lr_rgb, out_dir / "lr.png")
    _save_rgb(bicubic_rgb, out_dir / "bicubic.png")
    _save_rgb(sr_rgb, out_dir / "sr.png")
    hr_url = None
    if hr_reference is not None:
        _save_rgb(hr_reference, out_dir / "hr.png")
        hr_url = f"/inference/files/{uid}/hr.png"

    return InferenceOutput(
        scale=scale,
        arch=arch,
        device=str(device),
        amp_used=amp_active,
        inference_ms=elapsed_ms,
        lr_url=f"/inference/files/{uid}/lr.png",
        bicubic_url=f"/inference/files/{uid}/bicubic.png",
        sr_url=f"/inference/files/{uid}/sr.png",
        hr_url=hr_url,
        lr_size=(lr_rgb.shape[1], lr_rgb.shape[0]),
        sr_size=(sr_rgb.shape[1], sr_rgb.shape[0]),
        psnr_bicubic=psnr_b,
        psnr_sr=psnr_s,
        ssim_bicubic=ssim_b,
        ssim_sr=ssim_s,
    )


def cleanup_old_uploads(max_age_hours: float = 24.0) -> int:
    import shutil

    if not UPLOADS_DIR.exists():
        return 0
    removed = 0
    cutoff = time.time() - max_age_hours * 3600.0
    for child in UPLOADS_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def read_upload_file(uid: str, filename: str) -> Path | None:
    if "/" in uid or ".." in uid or "/" in filename or ".." in filename:
        return None
    path = UPLOADS_DIR / uid / filename
    return path if path.is_file() else None
