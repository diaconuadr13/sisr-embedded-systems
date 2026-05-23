import argparse
import csv
import gc
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from utils.device import configure_runtime, resolve_device
from utils.metrics import calculate_psnr, calculate_ssim
from utils.video_dataset import VideoSISRDataset


DEFAULT_VIDEO_CONFIG: Dict[str, Any] = {
    "hr_video_dir": "data/video/train",
    "val_video_dir": "data/video/val",
    "model_name": "VideoESPCN_x2_3f",
    "arch": "VideoESPCN",
    "dataset_name": "VideoDataset",
    "scale": 2,
    "num_frames": 3,
    "hidden_channels": 32,
    "batch_size": 16,
    "epochs": 100,
    "lr": 1e-3,
    "num_workers": 4,
    "device": "auto",
    "amp": True,
    "grayscale": False,
    "patch_size": 96,
    "samples_per_epoch": 16000,
    "seed": 42,
    "save_every": 10,
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight PC-only video SR models.")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    for key, value in DEFAULT_VIDEO_CONFIG.items():
        flag = f"--{key}"
        aliases = [f"--{key.replace('_', '-')}"] if "_" in key else []
        value_type = parse_bool if isinstance(value, bool) else type(value)
        parser.add_argument(flag, *aliases, dest=key, type=value_type, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_VIDEO_CONFIG)
    if args.config:
        path = Path(args.config)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            file_cfg = yaml.safe_load(raw) or {}
        else:
            file_cfg = json.loads(raw)
        cfg.update(file_cfg)
    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    return cfg


def create_experiment_dir(model_name: str, dataset_name: str) -> Path:
    runs_dir = Path("runs") / model_name / dataset_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        exp_dir = runs_dir / datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True)
            return exp_dir
        time.sleep(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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


def save_visual_samples(model: nn.Module, dataset: VideoSISRDataset, device: torch.device, epoch: int, out_dir: Path, cfg: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = 1 if cfg.get("grayscale", False) else 3
    indices = list(range(min(3, len(dataset))))
    model.eval()
    with torch.no_grad():
        for sample_id, idx in enumerate(indices, start=1):
            lr, hr = dataset[idx]
            lr_batch = lr.unsqueeze(0).to(device)
            hr_size = (hr.shape[1], hr.shape[2])
            with autocast_context(bool(cfg.get("amp", True)) and device.type == "cuda"):
                sr = torch.clamp(model(lr_batch), 0.0, 1.0).squeeze(0)
            bicubic = bicubic_center(lr_batch, int(cfg["num_frames"]), channels, hr_size).squeeze(0)
            panels = [(bicubic, "Bicubic"), (sr, "Video SR"), (hr, "HR Target")]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
            for ax, (img, title) in zip(axes, panels):
                arr = tensor_to_uint8(img)
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Epoch {epoch} - Sample {sample_id}")
            fig.tight_layout()
            fig.savefig(out_dir / f"epoch_{epoch:04d}_sample_{sample_id}.png", bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)


def cleanup_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Any, best_psnr: float, cfg: Dict[str, Any], arch: str, use_amp: bool) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_psnr": best_psnr,
        "arch": arch,
        "scale": int(cfg["scale"]),
        "num_frames": int(cfg["num_frames"]),
        "hidden_channels": int(cfg["hidden_channels"]),
        "grayscale": bool(cfg.get("grayscale", False)),
        "model_name": cfg["model_name"],
        "device": str(next(model.parameters()).device),
        "amp": use_amp,
        "config": cfg,
    }


def train(cfg: Dict[str, Any]) -> Path:
    seed_everything(int(cfg.get("seed", 42)))
    exp_dir = create_experiment_dir(str(cfg["model_name"]), str(cfg["dataset_name"]))
    visuals_dir = exp_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    device = resolve_device(str(cfg.get("device", "auto")))
    configure_runtime(device)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[train_video] device={device} amp={use_amp} exp_dir={exp_dir}")

    train_dataset = VideoSISRDataset(
        root_dir=str(cfg["hr_video_dir"]), scale=int(cfg["scale"]), patch_size=int(cfg["patch_size"]),
        num_frames=int(cfg["num_frames"]), grayscale=bool(cfg.get("grayscale", False)), split="train",
        random_crop=True, augment=True, samples_per_epoch=int(cfg["samples_per_epoch"]) if cfg.get("samples_per_epoch") else None,
    )
    val_dataset = VideoSISRDataset(
        root_dir=str(cfg["val_video_dir"]), scale=int(cfg["scale"]), patch_size=int(cfg["patch_size"]),
        num_frames=int(cfg["num_frames"]), grayscale=bool(cfg.get("grayscale", False)), split="val",
        random_crop=False, augment=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(cfg["num_workers"]) > 0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(cfg["num_workers"]) > 0)

    channels = 1 if cfg.get("grayscale", False) else 3
    arch = str(cfg.get("arch", "VideoESPCN"))
    model = get_model(arch, scale=int(cfg["scale"]), device=device, num_channels=channels, num_frames=int(cfg["num_frames"]), hidden_channels=int(cfg["hidden_channels"]))
    print(f"[train_video] arch={arch} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    best_psnr = -float("inf")

    log_path = exp_dir / "training_log.csv"
    try:
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["Epoch", "Model", "Dataset", "Train_Loss", "Val_Loss", "Val_PSNR", "Val_SSIM", "Bicubic_PSNR", "Bicubic_SSIM", "Val_PSNR_Gain", "LR"])
            for epoch in range(1, int(cfg["epochs"]) + 1):
                model.train()
                total_train_loss = 0.0
                train_count = 0
                for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]", leave=False):
                    lr_imgs = lr_imgs.to(device, non_blocking=True)
                    hr_imgs = hr_imgs.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(use_amp):
                        loss = criterion(model(lr_imgs), hr_imgs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    bs = lr_imgs.size(0)
                    total_train_loss += float(loss.item()) * bs
                    train_count += bs

                model.eval()
                total_val_loss = total_psnr = total_ssim = 0.0
                total_bicubic_psnr = total_bicubic_ssim = 0.0
                val_count = 0
                with torch.no_grad():
                    for lr_imgs, hr_imgs in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [val]", leave=False):
                        lr_imgs = lr_imgs.to(device, non_blocking=True)
                        hr_imgs = hr_imgs.to(device, non_blocking=True)
                        with autocast_context(use_amp):
                            sr_imgs = torch.clamp(model(lr_imgs), 0.0, 1.0)
                            val_loss = criterion(sr_imgs, hr_imgs)
                        bicubic_imgs = bicubic_center(lr_imgs, int(cfg["num_frames"]), channels, (hr_imgs.shape[2], hr_imgs.shape[3]))
                        bs = lr_imgs.size(0)
                        total_val_loss += float(val_loss.item()) * bs
                        for sr_img, hr_img, bicubic_img in zip(sr_imgs, hr_imgs, bicubic_imgs):
                            total_psnr += calculate_psnr(sr_img, hr_img)
                            total_ssim += calculate_ssim(sr_img, hr_img)
                            total_bicubic_psnr += calculate_psnr(bicubic_img, hr_img)
                            total_bicubic_ssim += calculate_ssim(bicubic_img, hr_img)
                            val_count += 1

                avg_train_loss = total_train_loss / max(train_count, 1)
                avg_val_loss = total_val_loss / max(val_count, 1)
                avg_psnr = total_psnr / max(val_count, 1)
                avg_ssim = total_ssim / max(val_count, 1)
                avg_bicubic_psnr = total_bicubic_psnr / max(val_count, 1)
                avg_bicubic_ssim = total_bicubic_ssim / max(val_count, 1)
                psnr_gain = avg_psnr - avg_bicubic_psnr
                current_lr = float(optimizer.param_groups[0]["lr"])
                writer.writerow([epoch, cfg["model_name"], cfg["dataset_name"], avg_train_loss, avg_val_loss, avg_psnr, avg_ssim, avg_bicubic_psnr, avg_bicubic_ssim, psnr_gain, current_lr])
                log_file.flush()
                print(f"Epoch {epoch}/{cfg['epochs']} | train_loss={avg_train_loss:.6f} | val_psnr={avg_psnr:.4f} | bicubic={avg_bicubic_psnr:.4f} | gain={psnr_gain:.4f}")

                checkpoint = make_checkpoint(epoch, model, optimizer, scaler, max(best_psnr, avg_psnr), cfg, arch, use_amp)
                torch.save(checkpoint, exp_dir / "last_model.pth")
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    checkpoint["best_psnr"] = best_psnr
                    torch.save(checkpoint, exp_dir / "best_model.pth")
                if epoch % int(cfg.get("save_every", 10)) == 0 or epoch == int(cfg["epochs"]):
                    save_visual_samples(model, val_dataset, device, epoch, visuals_dir, cfg)
    finally:
        del model, optimizer, criterion, scaler, train_loader, val_loader
        cleanup_vram()
    return exp_dir


def main() -> None:
    train(load_config(parse_args()))


if __name__ == "__main__":
    main()
