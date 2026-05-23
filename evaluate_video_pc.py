import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model, list_models
from utils.device import configure_runtime, resolve_device
from utils.metrics import calculate_psnr, calculate_ssim
from utils.video_dataset import VideoSISRDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PC-only video SR checkpoint.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--arch", type=str, default=None, choices=list_models())
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--save_samples", type=str, default=None)
    return parser.parse_args()


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_metadata(weights_path: Path, checkpoint: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        for key in ("arch", "scale", "num_frames", "hidden_channels", "grayscale", "config"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    config_path = weights_path.with_name("config.json")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata.setdefault("config", config)
        for key in ("arch", "scale", "num_frames", "hidden_channels", "grayscale", "patch_size"):
            metadata.setdefault(key, config.get(key))
    return metadata


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img.detach().cpu(), 0.0, 1.0)
    if img.shape[0] == 1:
        return (img.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def bicubic_center(lr_batch: torch.Tensor, num_frames: int, channels: int, size: tuple[int, int]) -> torch.Tensor:
    center = num_frames // 2
    start = center * channels
    center_lr = lr_batch[:, start:start + channels]
    return torch.clamp(F.interpolate(center_lr, size=size, mode="bicubic", align_corners=False), 0.0, 1.0)


def save_sample_panels(out_dir: Path, rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (bicubic, sr, hr) in enumerate(rows, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
        for ax, img, title in zip(axes, [bicubic, sr, hr], ["Bicubic", "Video SR", "HR Target"]):
            arr = tensor_to_uint8(img)
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:03d}.png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    checkpoint = torch.load(weights_path, map_location="cpu")
    metadata = load_metadata(weights_path, checkpoint)
    config = metadata.get("config") or {}

    arch = args.arch or metadata.get("arch") or config.get("arch")
    if arch is None:
        raise ValueError(f"Unable to determine architecture. Pass --arch. Available: {', '.join(list_models())}")
    scale = int(args.scale or metadata.get("scale") or config.get("scale") or 2)
    num_frames = int(args.num_frames or metadata.get("num_frames") or config.get("num_frames") or 3)
    hidden_channels = int(args.hidden_channels or metadata.get("hidden_channels") or config.get("hidden_channels") or 32)
    grayscale = bool(args.grayscale or metadata.get("grayscale") or config.get("grayscale", False))
    patch_size = int(args.patch_size or metadata.get("patch_size") or config.get("patch_size") or 96)

    device = resolve_device(args.device)
    configure_runtime(device)
    use_amp = args.amp and device.type == "cuda"
    channels = 1 if grayscale else 3
    model = get_model(arch, scale=scale, device=device, num_channels=channels, num_frames=num_frames, hidden_channels=hidden_channels)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = VideoSISRDataset(args.video_dir, scale=scale, patch_size=patch_size, num_frames=num_frames, grayscale=grayscale, split="eval", random_crop=False, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    total_time = 0.0
    total_samples = 0
    total_psnr = total_ssim = total_bicubic_psnr = total_bicubic_ssim = 0.0
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        warm_lr, _ = next(iter(loader))
        warm_lr = warm_lr.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with autocast_context(use_amp):
            _ = model(warm_lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for lr_imgs, hr_imgs in tqdm(loader, desc="evaluate"):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with autocast_context(use_amp):
                sr_imgs = torch.clamp(model(lr_imgs), 0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.perf_counter() - start
            bicubic_imgs = bicubic_center(lr_imgs, num_frames, channels, (hr_imgs.shape[2], hr_imgs.shape[3]))
            for bicubic_img, sr_img, hr_img in zip(bicubic_imgs, sr_imgs, hr_imgs):
                total_psnr += calculate_psnr(sr_img, hr_img)
                total_ssim += calculate_ssim(sr_img, hr_img)
                total_bicubic_psnr += calculate_psnr(bicubic_img, hr_img)
                total_bicubic_ssim += calculate_ssim(bicubic_img, hr_img)
                total_samples += 1
                if len(samples) < 4:
                    samples.append((bicubic_img.cpu(), sr_img.cpu(), hr_img.cpu()))

    avg_time_ms = (total_time / max(total_samples, 1)) * 1000.0
    fps = float("inf") if total_time == 0.0 else total_samples / total_time
    metrics = {
        "weights": str(weights_path),
        "video_dir": args.video_dir,
        "arch": arch,
        "scale": scale,
        "num_frames": num_frames,
        "hidden_channels": hidden_channels,
        "grayscale": grayscale,
        "patch_size": patch_size,
        "samples": total_samples,
        "avg_inference_time_ms": avg_time_ms,
        "fps": fps,
        "psnr": total_psnr / max(total_samples, 1),
        "ssim": total_ssim / max(total_samples, 1),
        "bicubic_psnr": total_bicubic_psnr / max(total_samples, 1),
        "bicubic_ssim": total_bicubic_ssim / max(total_samples, 1),
        "psnr_gain": (total_psnr - total_bicubic_psnr) / max(total_samples, 1),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    print("\nVideo SR Evaluation")
    print(f"  Arch: {arch} x{scale} frames={num_frames} grayscale={grayscale}")
    print(f"  Parameters: {metrics['parameters']:,}")
    print(f"  Time: {avg_time_ms:.4f} ms/sample  FPS: {fps:.4f}")
    print(f"  PSNR/SSIM: {metrics['psnr']:.4f} dB / {metrics['ssim']:.4f}")
    print(f"  Bicubic: {metrics['bicubic_psnr']:.4f} dB / {metrics['bicubic_ssim']:.4f}")
    print(f"  PSNR gain: {metrics['psnr_gain']:.4f} dB")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.save_samples:
        save_sample_panels(Path(args.save_samples), samples)


if __name__ == "__main__":
    main()
