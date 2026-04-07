import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from models import get_model, list_models
from utils.device import configure_runtime, resolve_device
from utils.dataset import SISRDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SISR inference speed.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--arch", type=str, default=None, choices=list_models())
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision during inference.")
    return parser.parse_args()


def load_checkpoint_metadata(weights_path: Path, checkpoint: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        metadata.update({k: checkpoint[k] for k in ("arch", "scale", "config") if k in checkpoint})

    config_path = weights_path.with_name("config.json")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata.setdefault("config", config)
        metadata.setdefault("arch", config.get("arch"))
        metadata.setdefault("scale", config.get("scale"))

    return metadata


def resolve_model_spec(args: argparse.Namespace, metadata: Dict[str, Any]) -> Tuple[str, int]:
    config = metadata.get("config") or {}
    arch = args.arch or metadata.get("arch") or config.get("arch")
    scale = args.scale or metadata.get("scale") or config.get("scale")

    if arch is None:
        available = ", ".join(list_models())
        raise ValueError(
            "Unable to determine checkpoint architecture. "
            f"Pass --arch explicitly. Available: {available}"
        )
    if scale is None:
        raise ValueError("Unable to determine checkpoint scale. Pass --scale explicitly.")
    return arch, int(scale)


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    device = resolve_device(args.device)
    configure_runtime(device)
    use_amp = args.amp and device.type == "cuda"

    checkpoint = torch.load(weights_path, map_location="cpu")
    metadata = load_checkpoint_metadata(weights_path, checkpoint)
    arch, scale = resolve_model_spec(args, metadata)

    model = get_model(arch, scale=scale, device=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[eval] arch={arch} scale={scale} device={device} amp={use_amp}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    patch_size = scale * 24
    val_dataset = SISRDataset(hr_dir=args.val_dir, scale=scale, patch_size=patch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        warmup_lr, _ = next(iter(val_loader))
        warmup_lr = warmup_lr.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with autocast_context(use_amp):
            _ = model(warmup_lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for lr_imgs, _ in val_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with autocast_context(use_amp):
                _ = model(lr_imgs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            total_time += elapsed
            total_images += lr_imgs.size(0)

    avg_time_ms = (total_time / max(total_images, 1)) * 1000.0
    avg_fps = float("inf") if total_time == 0.0 else total_images / total_time

    print(f"Average Inference Time (ms): {avg_time_ms:.4f}")
    print(f"FPS: {avg_fps:.4f}")


if __name__ == "__main__":
    main()
