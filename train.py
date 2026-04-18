import argparse
import csv
import gc
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
from utils.dataset import SISRDataset
from utils.metrics import calculate_psnr, calculate_ssim


# --- Default config (single source of truth) ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "hr_dir": "data/train/DIV2K_train_HR",
    "val_dir": "data/val/DIV2K_valid_HR",
    "model_name": "ESPCN_base",
    "arch": "ESPCN",  # model class: ESPCN | ESPCN_Light | FSRCNN
    "dataset_name": "DIV2K",
    "scale": 2,
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "num_workers": 4,
    "device": "auto",
    "amp": True,
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SISR models (ESPCN, ESPCN_Light, FSRCNN).")
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config file (overrides CLI defaults)")
    for k, v in DEFAULT_CONFIG.items():
        flag = f"--{k}"
        aliases = [f"--{k.replace('_', '-')}"] if "_" in k else []
        value_type = parse_bool if isinstance(v, bool) else type(v)
        p.add_argument(flag, *aliases, dest=k, type=value_type, default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge: defaults < YAML file < CLI overrides."""
    cfg = dict(DEFAULT_CONFIG)

    if args.config:
        path = Path(args.config)
        raw = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml
            file_cfg = yaml.safe_load(raw) or {}
        else:
            file_cfg = json.loads(raw)
        cfg.update(file_cfg)

    # CLI overrides (only non-None values)
    cli = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg.update(cli)
    return cfg


def create_experiment_dir(model_name: str, dataset_name: str) -> Path:
    runs_dir = Path("runs") / model_name / dataset_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        exp_dir = runs_dir / exp_name
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=False)
            return exp_dir
        time.sleep(1)


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img, 0.0, 1.0).detach().cpu()
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_visual_samples(
    model: nn.Module,
    dataset: SISRDataset,
    device: torch.device,
    epoch: int,
    visuals_dir: Path,
) -> None:
    sample_count = min(3, len(dataset))
    if sample_count == 0:
        return

    indices = random.sample(range(len(dataset)), k=sample_count)
    model.eval()
    with torch.no_grad():
        for sample_id, idx in enumerate(indices, start=1):
            lr_img, hr_img = dataset[idx]
            lr_batch = lr_img.unsqueeze(0).to(device)
            sr_img = torch.clamp(model(lr_batch).squeeze(0), 0.0, 1.0)
            lr_up = F.interpolate(
                lr_batch,
                size=(hr_img.shape[1], hr_img.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            lr_np = tensor_to_uint8(lr_up)
            sr_np = tensor_to_uint8(sr_img)
            hr_np = tensor_to_uint8(hr_img)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
            panels = [
                (lr_np, "Bicubic (LR)"),
                (sr_np, "Model (SR)"),
                (hr_np, "Target (HR)"),
            ]
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")

            fig.suptitle(f"Epoch {epoch} • Sample {sample_id}")
            fig.tight_layout()
            out_path = visuals_dir / f"epoch_{epoch:04d}_sample_{sample_id}.png"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)


def cleanup_vram() -> None:
    """Force-release all CUDA tensors and cached allocator blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train(cfg: Dict[str, Any]) -> Path:
    """Run one full training experiment. Returns the experiment directory path."""
    exp_dir = create_experiment_dir(cfg["model_name"], cfg["dataset_name"])
    visuals_dir = exp_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    with (exp_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = resolve_device(str(cfg.get("device", "auto")))
    configure_runtime(device)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[train] device={device}  amp={use_amp}  exp_dir={exp_dir}")

    patch_size = cfg["scale"] * 24
    train_dataset = SISRDataset(hr_dir=cfg["hr_dir"], scale=cfg["scale"], patch_size=patch_size)
    val_dataset = SISRDataset(hr_dir=cfg["val_dir"], scale=cfg["scale"], patch_size=patch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
    )

    arch = cfg.get("arch", "ESPCN")
    model = get_model(arch, scale=cfg["scale"], device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] arch={arch}  params={trainable:,}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_psnr = -float("inf")

    log_path = exp_dir / "training_log.csv"
    try:
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["Epoch", "Model", "Dataset", "Train_Loss", "Val_Loss", "Val_PSNR", "Val_SSIM", "LR"])

            for epoch in range(1, cfg["epochs"] + 1):
                # --- Train ---
                model.train()
                total_train_loss = 0.0
                train_count = 0
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]", leave=False)
                for lr_imgs, hr_imgs in train_pbar:
                    lr_imgs = lr_imgs.to(device, non_blocking=True)
                    hr_imgs = hr_imgs.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(use_amp):
                        sr_imgs = model(lr_imgs)
                        loss = criterion(sr_imgs, hr_imgs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    bs = lr_imgs.size(0)
                    total_train_loss += float(loss.item()) * bs
                    train_count += bs
                    train_pbar.set_postfix(loss=f"{loss.item():.6f}")

                avg_train_loss = total_train_loss / max(train_count, 1)

                # --- Validate ---
                model.eval()
                total_val_loss = 0.0
                total_psnr = 0.0
                total_ssim = 0.0
                val_count = 0
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [val]", leave=False)
                    for lr_imgs, hr_imgs in val_pbar:
                        lr_imgs = lr_imgs.to(device, non_blocking=True)
                        hr_imgs = hr_imgs.to(device, non_blocking=True)
                        with autocast_context(use_amp):
                            sr_imgs = torch.clamp(model(lr_imgs), 0.0, 1.0)
                            val_loss = criterion(sr_imgs, hr_imgs)
                        bs = lr_imgs.size(0)
                        total_val_loss += float(val_loss.item()) * bs
                        for sr_img, hr_img in zip(sr_imgs, hr_imgs):
                            total_psnr += calculate_psnr(sr_img, hr_img)
                            total_ssim += calculate_ssim(sr_img, hr_img)
                            val_count += 1

                avg_val_loss = total_val_loss / max(val_count, 1)
                avg_psnr = total_psnr / max(val_count, 1)
                avg_ssim = total_ssim / max(val_count, 1)
                current_lr = float(optimizer.param_groups[0]["lr"])

                writer.writerow([epoch, cfg["model_name"], cfg["dataset_name"],
                                 avg_train_loss, avg_val_loss, avg_psnr, avg_ssim, current_lr])
                log_file.flush()

                print(f"Epoch {epoch}/{cfg['epochs']} | train_loss={avg_train_loss:.6f} | "
                      f"val_loss={avg_val_loss:.6f} | val_psnr={avg_psnr:.4f} | "
                      f"val_ssim={avg_ssim:.4f} | lr={current_lr:.6e}")

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "arch": arch,
                            "scale": cfg["scale"],
                            "model_name": cfg["model_name"],
                            "device": str(device),
                            "amp": use_amp,
                            "config": cfg,
                        },
                        exp_dir / "best_model.pth",
                    )

                if epoch % 10 == 0:
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "arch": arch,
                            "scale": cfg["scale"],
                            "model_name": cfg["model_name"],
                            "device": str(device),
                            "amp": use_amp,
                            "config": cfg,
                        },
                        exp_dir / "last_model.pth",
                    )
                    save_visual_samples(model, val_dataset, device, epoch, visuals_dir)

    finally:
        # Deterministic VRAM release regardless of success/failure
        del model, optimizer, criterion, scaler
        del train_loader, val_loader
        cleanup_vram()

    return exp_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    experiments = cfg.pop("experiments", None)
    if experiments:
        cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        for exp in experiments:
            run_cfg = dict(DEFAULT_CONFIG)
            run_cfg.update(exp)
            run_cfg.update(cli_overrides)
            train(run_cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
