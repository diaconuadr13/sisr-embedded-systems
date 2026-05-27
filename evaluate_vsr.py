import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import get_model
from utils.dataset import VideoFolderSRDataset
from utils.device import resolve_device
from utils.metrics import calculate_psnr, calculate_ssim, calculate_temporal_consistency_error
from utils.model_stats import count_trainable_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VSR checkpoint on a normalized video-folder dataset.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth or a state_dict checkpoint")
    p.add_argument("--video-root", required=True, help="Normalized root: DatasetName/clip/frame_000.png")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--arch", default="VSRBasic")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick checks")
    p.add_argument("--save-samples", type=int, default=6)
    p.add_argument("--hidden-channels", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    return p.parse_args()


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img.detach().cpu(), 0.0, 1.0)
    if img.shape[0] == 1:
        return (img.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_triptych(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: Path) -> None:
    lr_up = F.interpolate(
        lr.unsqueeze(0),
        size=(hr.shape[1], hr.shape[2]),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    panels = [tensor_to_uint8(lr_up), tensor_to_uint8(sr), tensor_to_uint8(hr)]
    if panels[0].ndim == 2:
        canvas = np.concatenate(panels, axis=1)
    else:
        canvas = np.concatenate(panels, axis=1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict]:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"], raw
    if isinstance(raw, dict):
        return raw, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def estimate_macs(model: torch.nn.Module, example: torch.Tensor, device: torch.device) -> int | None:
    macs = 0
    hooks = []

    def conv_hook(module: torch.nn.Conv2d, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal macs
        out = output
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs += int(out.numel() * kernel_ops)

    def linear_hook(module: torch.nn.Linear, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal macs
        macs += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(example.to(device))
        return macs
    except Exception:
        return None
    finally:
        for hook in hooks:
            hook.remove()


def build_model(args: argparse.Namespace, checkpoint_meta: Dict, device: torch.device) -> torch.nn.Module:
    cfg = checkpoint_meta.get("config", {}) if isinstance(checkpoint_meta, dict) else {}
    arch = args.arch or checkpoint_meta.get("arch") or cfg.get("arch", "VSRBasic")
    hidden_channels = args.hidden_channels or cfg.get("hidden_channels")
    num_blocks = args.num_blocks or cfg.get("num_blocks")
    model = get_model(
        arch,
        scale=args.scale,
        device=device,
        num_channels=1 if args.grayscale else 3,
        num_frames=args.num_frames,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
    )
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict, checkpoint_meta = load_checkpoint(Path(args.checkpoint), device)
    model = build_model(args, checkpoint_meta, device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = VideoFolderSRDataset(
        root_dir=args.video_root,
        scale=args.scale,
        num_frames=args.num_frames,
        grayscale=args.grayscale,
        include_all_frames=True,
    )
    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)

    rows: List[Dict] = []
    sr_by_clip: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    hr_by_clip: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    inference_times: List[float] = []

    params = count_trainable_params(model)
    macs = None

    with torch.no_grad():
        for idx in range(limit):
            lr_clip, hr = dataset[idx]
            info = dataset.sample_info(idx)
            model_input = lr_clip.unsqueeze(0).to(device)
            if macs is None:
                macs = estimate_macs(model, model_input, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            sr = torch.clamp(model(model_input).squeeze(0), 0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            hr = hr.to(device)
            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)
            rows.append(
                {
                    "clip": info["clip_name"],
                    "frame_index": info["frame_index"],
                    "psnr": psnr,
                    "ssim": ssim,
                    "inference_ms": elapsed * 1000.0,
                }
            )
            sr_by_clip.setdefault(info["clip_name"], []).append((info["frame_index"], sr.detach().cpu()))
            hr_by_clip.setdefault(info["clip_name"], []).append((info["frame_index"], hr.detach().cpu()))

            if idx < args.save_samples:
                center_lr = lr_clip[args.num_frames // 2]
                save_triptych(center_lr, sr.detach().cpu(), hr.detach().cpu(), samples_dir / f"{idx:04d}_{info['clip_name']}_{info['frame_index']:03d}.png")

    temporal_errors: List[float] = []
    for clip_name, sr_items in sr_by_clip.items():
        hr_items = hr_by_clip[clip_name]
        sr_stack = torch.stack([item for _idx, item in sorted(sr_items, key=lambda item: item[0])], dim=0)
        hr_stack = torch.stack([item for _idx, item in sorted(hr_items, key=lambda item: item[0])], dim=0)
        temporal_errors.append(calculate_temporal_consistency_error(sr_stack, hr_stack))

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "frame_index", "psnr", "ssim", "inference_ms"])
        writer.writeheader()
        writer.writerows(rows)

    avg_ms = float(np.mean(inference_times) * 1000.0) if inference_times else 0.0
    summary = {
        "dataset": args.dataset_name,
        "video_root": args.video_root,
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "scale": args.scale,
        "num_frames": args.num_frames,
        "frames_evaluated": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else 0.0,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else 0.0,
        "temporal_consistency_error": float(np.mean(temporal_errors)) if temporal_errors else 0.0,
        "avg_inference_ms_per_frame": avg_ms,
        "estimated_fps": (1000.0 / avg_ms) if avg_ms > 0 else 0.0,
        "parameters": params,
        "macs": macs,
        "flops": None if macs is None else macs * 2,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
