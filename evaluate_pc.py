import argparse
import time

import torch
from torch.utils.data import DataLoader

from models.espcn import ESPCN
from utils.dataset import SISRDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ESPCN inference speed.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESPCN(scale_factor=args.scale, num_channels=3).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    patch_size = args.scale * 24
    val_dataset = SISRDataset(hr_dir=args.val_dir, scale=args.scale, patch_size=patch_size)
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
        _ = model(warmup_lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for lr_imgs, _ in val_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
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
